"""
Central event bus for Milan Capital.

All system events flow through here:
  - Dashboard (SSE)  ← ALL events, displayed + historised
  - Telegram         ← ONLY severity='critical' (circuit breaker, system errors)

Usage:
    from src.events.bus import get_bus, Event
    get_bus().emit(Event(type='execution', severity='info', title='BUY AAPL', body='...'))
"""
from __future__ import annotations

import json
import logging
import queue
import sqlite3
import threading
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Generator, Optional

logger = logging.getLogger(__name__)

# ── Constants ──────────────────────────────────────────────────────────────────

DB_PATH = Path("logs/events.db")

VALID_TYPES = frozenset({
    "execution", "briefing", "circuit_breaker", "regime_change",
    "error", "tearsheet", "monte_carlo", "news",
})
VALID_SEVERITIES = frozenset({"info", "warning", "critical"})

# Events with severity='critical' are also routed to Telegram.
# Everything else stays on the dashboard only.
_CRITICAL_TYPES = frozenset({"circuit_breaker", "error"})


# ── Event dataclass ────────────────────────────────────────────────────────────

@dataclass
class Event:
    type: str
    severity: str          # 'info' | 'warning' | 'critical'
    title: str
    body: str
    meta: dict = field(default_factory=dict)
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    timestamp: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "timestamp": self.timestamp,
            "type": self.type,
            "severity": self.severity,
            "title": self.title,
            "body": self.body,
            "meta": self.meta,
        }


# ── EventBus ───────────────────────────────────────────────────────────────────

class EventBus:
    """
    Thread-safe event bus backed by SQLite.

    - emit()         → persist + push SSE + Telegram if critical
    - subscribe_sse()→ blocking generator, yields JSON strings
    - get_recent()   → latest N events from DB (optional type filter)
    """

    def __init__(self, db_path: Path = DB_PATH) -> None:
        self._db_path = Path(db_path)
        self._write_lock = threading.Lock()
        self._sub_lock   = threading.Lock()
        self._queues: list[queue.Queue] = []
        self._init_db()

    # ── DB setup ──────────────────────────────────────────────────────────────

    def _init_db(self) -> None:
        self._db_path.parent.mkdir(parents=True, exist_ok=True)
        with self._connect() as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS events (
                    id          TEXT PRIMARY KEY,
                    timestamp   TEXT NOT NULL,
                    type        TEXT NOT NULL,
                    severity    TEXT NOT NULL,
                    title       TEXT NOT NULL,
                    body        TEXT NOT NULL,
                    meta        TEXT NOT NULL DEFAULT '{}',
                    acknowledged INTEGER NOT NULL DEFAULT 0
                )
            """)
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_ts   ON events(timestamp DESC)"
            )
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_type ON events(type)"
            )
            conn.commit()

    def _connect(self) -> sqlite3.Connection:
        return sqlite3.connect(self._db_path, check_same_thread=False)

    # ── Public API ────────────────────────────────────────────────────────────

    def emit(self, event: Event) -> None:
        """Persist, push SSE, and route to Telegram if critical."""
        self._persist(event)
        self._push_sse(event)
        if event.severity == "critical":
            try:
                self._route_to_telegram(event)
            except Exception as exc:
                logger.error("EventBus: critical routing failed — %s", exc)

    def subscribe_sse(self) -> Generator[str, None, None]:
        """
        Blocking generator — yields JSON strings for each new event.
        Yields '__HEARTBEAT__' every 25 s if no event arrives.
        Unregisters automatically when the caller stops iterating.
        """
        q: queue.Queue = queue.Queue(maxsize=200)
        self._register(q)
        try:
            while True:
                try:
                    yield q.get(timeout=25)
                except queue.Empty:
                    yield "__HEARTBEAT__"
        finally:
            self._unregister(q)

    def get_recent(
        self,
        limit: int = 100,
        type_filter: Optional[str] = None,
    ) -> list[dict]:
        """Return the most recent events, newest first."""
        with self._connect() as conn:
            conn.row_factory = sqlite3.Row
            if type_filter:
                rows = conn.execute(
                    "SELECT * FROM events WHERE type=? ORDER BY timestamp DESC LIMIT ?",
                    (type_filter, limit),
                ).fetchall()
            else:
                rows = conn.execute(
                    "SELECT * FROM events ORDER BY timestamp DESC LIMIT ?",
                    (limit,),
                ).fetchall()
        return [
            {**dict(r), "meta": json.loads(r["meta"])}
            for r in rows
        ]

    def acknowledge(self, event_id: str) -> None:
        """Mark a critical event as acknowledged (removes glow in UI)."""
        with self._write_lock:
            with self._connect() as conn:
                conn.execute(
                    "UPDATE events SET acknowledged=1 WHERE id=?", (event_id,)
                )
                conn.commit()

    # ── Internal ──────────────────────────────────────────────────────────────

    def _persist(self, event: Event) -> None:
        with self._write_lock:
            with self._connect() as conn:
                conn.execute(
                    """INSERT OR IGNORE INTO events
                       (id, timestamp, type, severity, title, body, meta, acknowledged)
                       VALUES (?,?,?,?,?,?,?,0)""",
                    (
                        event.id,
                        event.timestamp,
                        event.type,
                        event.severity,
                        event.title,
                        event.body,
                        json.dumps(event.meta or {}),
                    ),
                )
                conn.commit()

    def _push_sse(self, event: Event) -> None:
        payload = json.dumps(event.to_dict())
        with self._sub_lock:
            dead: list[queue.Queue] = []
            for q in self._queues:
                try:
                    q.put_nowait(payload)
                except queue.Full:
                    dead.append(q)
            for q in dead:
                self._queues.remove(q)

    def _route_to_telegram(self, event: Event) -> None:
        try:
            from src.notify.telegram import send_message
            icon = "🚨" if event.type == "error" else "⚠️"
            send_message(f"{icon} *{event.title}*\n{event.body}")
        except Exception as exc:
            logger.error("EventBus: Telegram routing failed — %s", exc)

    def _register(self, q: queue.Queue) -> None:
        with self._sub_lock:
            self._queues.append(q)

    def _unregister(self, q: queue.Queue) -> None:
        with self._sub_lock:
            try:
                self._queues.remove(q)
            except ValueError:
                pass


# ── Singleton ─────────────────────────────────────────────────────────────────

_bus: Optional[EventBus] = None
_bus_lock = threading.Lock()


def get_bus() -> EventBus:
    """Return the process-level EventBus singleton."""
    global _bus
    if _bus is None:
        with _bus_lock:
            if _bus is None:
                _bus = EventBus()
    return _bus
