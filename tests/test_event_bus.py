"""
Tests for src/events/bus.py
Run: pytest tests/test_event_bus.py -v
"""
from __future__ import annotations

import json
import queue
import tempfile
import threading
import time
from pathlib import Path

import pytest

from src.events.bus import Event, EventBus, VALID_TYPES, VALID_SEVERITIES


# ── Fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture
def tmp_bus(tmp_path):
    """Fresh EventBus backed by a temporary SQLite DB."""
    return EventBus(db_path=tmp_path / "test_events.db")


def make_event(**kwargs) -> Event:
    defaults = dict(type="briefing", severity="info", title="T", body="B")
    defaults.update(kwargs)
    return Event(**defaults)


# ── 1. Event creation ─────────────────────────────────────────────────────────

def test_event_has_uuid_id():
    ev = make_event()
    assert len(ev.id) == 36  # UUID4 format
    assert ev.id != make_event().id  # unique per instance


def test_event_has_iso_timestamp():
    ev = make_event()
    assert "T" in ev.timestamp  # ISO 8601
    assert ev.timestamp.endswith("+00:00") or ev.timestamp.endswith("Z") or "UTC" in ev.timestamp or "+" in ev.timestamp


def test_event_to_dict():
    ev = make_event(type="execution", severity="critical", title="BUY", body="AAPL x10", meta={"price": 150})
    d = ev.to_dict()
    assert d["type"] == "execution"
    assert d["severity"] == "critical"
    assert d["meta"]["price"] == 150
    assert d["title"] == "BUY"


# ── 2. DB init ────────────────────────────────────────────────────────────────

def test_db_file_created(tmp_path):
    db = tmp_path / "events.db"
    EventBus(db_path=db)
    assert db.exists()


def test_table_created(tmp_bus):
    with tmp_bus._connect() as conn:
        tables = conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table'"
        ).fetchall()
        names = {r[0] for r in tables}
    assert "events" in names


# ── 3. Persist ────────────────────────────────────────────────────────────────

def test_emit_persists_event(tmp_bus):
    ev = make_event(title="persist-me")
    tmp_bus.emit(ev)
    rows = tmp_bus.get_recent(limit=10)
    assert any(r["title"] == "persist-me" for r in rows)


def test_emit_deduplicates_same_id(tmp_bus):
    ev = make_event(title="dup")
    tmp_bus.emit(ev)
    tmp_bus.emit(ev)  # same id → INSERT OR IGNORE
    rows = tmp_bus.get_recent(limit=100)
    assert sum(1 for r in rows if r["title"] == "dup") == 1


def test_meta_roundtrips(tmp_bus):
    ev = make_event(meta={"symbol": "AAPL", "qty": 5})
    tmp_bus.emit(ev)
    rows = tmp_bus.get_recent()
    stored = next(r for r in rows if r["id"] == ev.id)
    assert stored["meta"]["symbol"] == "AAPL"
    assert stored["meta"]["qty"] == 5


# ── 4. get_recent ─────────────────────────────────────────────────────────────

def test_get_recent_order_newest_first(tmp_bus):
    for i in range(5):
        tmp_bus.emit(make_event(title=f"ev{i}"))
    rows = tmp_bus.get_recent(limit=5)
    titles = [r["title"] for r in rows]
    assert titles == list(reversed([f"ev{i}" for i in range(5)]))


def test_get_recent_limit(tmp_bus):
    for _ in range(10):
        tmp_bus.emit(make_event())
    assert len(tmp_bus.get_recent(limit=3)) == 3


def test_get_recent_type_filter(tmp_bus):
    tmp_bus.emit(make_event(type="briefing", title="brief"))
    tmp_bus.emit(make_event(type="execution", title="exec"))
    rows = tmp_bus.get_recent(type_filter="briefing")
    assert all(r["type"] == "briefing" for r in rows)
    assert any(r["title"] == "brief" for r in rows)


# ── 5. Telegram routing ───────────────────────────────────────────────────────

def test_critical_routes_to_telegram(tmp_bus, monkeypatch):
    sent = []
    monkeypatch.setattr(
        "src.events.bus.EventBus._route_to_telegram",
        lambda self, ev: sent.append(ev),
    )
    tmp_bus.emit(make_event(severity="critical", title="CB TRIGGERED"))
    assert len(sent) == 1
    assert sent[0].title == "CB TRIGGERED"


def test_info_does_not_route_to_telegram(tmp_bus, monkeypatch):
    sent = []
    monkeypatch.setattr(
        "src.events.bus.EventBus._route_to_telegram",
        lambda self, ev: sent.append(ev),
    )
    tmp_bus.emit(make_event(severity="info"))
    tmp_bus.emit(make_event(severity="warning"))
    assert sent == []


def test_telegram_failure_does_not_raise(tmp_bus, monkeypatch):
    def boom(self, ev):
        raise RuntimeError("Telegram down")
    monkeypatch.setattr("src.events.bus.EventBus._route_to_telegram", boom)
    # Should not raise despite Telegram failure — event still persisted
    tmp_bus.emit(make_event(severity="critical"))
    assert len(tmp_bus.get_recent()) >= 1


# ── 6. SSE subscription ───────────────────────────────────────────────────────

def test_sse_receives_emitted_event(tmp_bus):
    received = []

    def subscribe():
        gen = tmp_bus.subscribe_sse()
        payload = next(gen)
        received.append(payload)
        gen.close()  # stop generator → unregisters queue

    t = threading.Thread(target=subscribe, daemon=True)
    t.start()
    time.sleep(0.05)  # let thread block on queue.get()

    ev = make_event(title="sse-test")
    tmp_bus.emit(ev)
    t.join(timeout=3)

    assert received
    data = json.loads(received[0])
    assert data["title"] == "sse-test"


def test_sse_registers_and_unregisters(tmp_bus):
    """Queue is added on subscribe and removed after the generator is cleaned up."""
    got = []
    done = threading.Event()

    def consume():
        gen = tmp_bus.subscribe_sse()
        # Close generator inside the thread so there's no cross-thread ownership issue
        with tmp_bus._sub_lock:
            q_count_before = len(tmp_bus._queues)
        payload = next(gen)  # blocks until event arrives
        got.append(payload)
        gen.close()          # triggers finally → _unregister
        done.set()

    t = threading.Thread(target=consume, daemon=True)
    t.start()
    time.sleep(0.05)  # let thread reach q.get() and register queue

    with tmp_bus._sub_lock:
        assert len(tmp_bus._queues) == 1

    tmp_bus.emit(make_event(title="unblock"))  # unblocks q.get in thread
    done.wait(timeout=3)

    assert got  # thread received the event
    with tmp_bus._sub_lock:
        assert len(tmp_bus._queues) == 0


# ── 7. acknowledge ────────────────────────────────────────────────────────────

def test_acknowledge_sets_flag(tmp_bus):
    ev = make_event(severity="critical")
    tmp_bus.emit(ev)
    tmp_bus.acknowledge(ev.id)
    with tmp_bus._connect() as conn:
        row = conn.execute(
            "SELECT acknowledged FROM events WHERE id=?", (ev.id,)
        ).fetchone()
    assert row[0] == 1


# ── 8. Thread safety ─────────────────────────────────────────────────────────

def test_concurrent_writes_no_corruption(tmp_bus):
    errors = []

    def worker(n):
        try:
            for i in range(20):
                tmp_bus.emit(make_event(title=f"w{n}-ev{i}"))
        except Exception as e:
            errors.append(e)

    threads = [threading.Thread(target=worker, args=(n,)) for n in range(5)]
    for t in threads: t.start()
    for t in threads: t.join(timeout=10)

    assert not errors
    rows = tmp_bus.get_recent(limit=200)
    assert len(rows) == 100  # 5 workers × 20 events


# ── 9. Persistence across restarts ───────────────────────────────────────────

def test_events_survive_restart(tmp_path):
    db = tmp_path / "persist.db"
    bus1 = EventBus(db_path=db)
    ev = make_event(title="survives-restart")
    bus1.emit(ev)

    bus2 = EventBus(db_path=db)  # new instance, same DB
    rows = bus2.get_recent()
    assert any(r["title"] == "survives-restart" for r in rows)


# ── 10. Notification routing policy ─────────────────────────────────────────


def test_morning_briefing_emits_event_not_telegram(tmp_path, monkeypatch):
    """
    morning_briefing.run() must emit a 'briefing' event with severity='info'
    to the event bus and NEVER call Telegram directly.
    """
    telegram_called = []

    # Patch the singleton bus with our tmp_bus
    test_bus = EventBus(db_path=tmp_path / "brief_test.db")
    monkeypatch.setattr("src.events.bus._bus", test_bus)

    # Intercept any Telegram call at the transport level
    monkeypatch.setattr(
        "src.notify.telegram.send_message",
        lambda *a, **kw: telegram_called.append(a),
    )

    # Stub out I/O-bound helpers so the test doesn't hit the network
    monkeypatch.setattr(
        "src.notify.morning_briefing._build_market_section",
        lambda: "SPY +1.0%",
    )
    monkeypatch.setattr(
        "src.notify.morning_briefing._build_feed_section",
        lambda label, emoji, url: "",
    )
    monkeypatch.setattr(
        "src.notify.morning_briefing._build_dividend_section",
        lambda: "",
    )
    monkeypatch.setattr(
        "src.notify.morning_briefing._build_mc_line",
        lambda: "",
    )

    from src.notify.morning_briefing import run
    run()

    # No direct Telegram call
    assert telegram_called == [], "morning briefing must not call Telegram directly"

    # A briefing event must have been emitted with severity='info'
    rows = test_bus.get_recent(type_filter="briefing")
    assert len(rows) >= 1, "morning briefing must emit a 'briefing' event to the bus"
    assert rows[0]["severity"] == "info", "morning briefing event must have severity='info'"


def test_weekly_tearsheet_emits_event_not_telegram(tmp_path, monkeypatch):
    """
    weekly_tearsheet._emit_tearsheet_event() must emit a 'tearsheet' info event.
    The legacy _send_portfolio_performance / _send_drift_alerts / _send_dividend_arb_pnl
    functions must route through the bus, not call Telegram directly.
    """
    telegram_called = []
    test_bus = EventBus(db_path=tmp_path / "ts_test.db")
    monkeypatch.setattr("src.events.bus._bus", test_bus)
    monkeypatch.setattr(
        "src.notify.telegram.send_message",
        lambda *a, **kw: telegram_called.append(a),
    )

    from src.notify.weekly_tearsheet import _emit_tearsheet_event
    from unittest.mock import MagicMock

    scorer = MagicMock()
    _emit_tearsheet_event(scorer)

    assert telegram_called == [], "tearsheet emitter must not call Telegram directly"
    rows = test_bus.get_recent(type_filter="tearsheet")
    assert len(rows) >= 1, "tearsheet emitter must emit a 'tearsheet' event to the bus"
    assert rows[0]["severity"] == "info"


def test_stop_loss_routes_critical_to_telegram(tmp_path, monkeypatch):
    """
    A stop-loss event emitted as severity='critical' must be routed to Telegram
    through the event bus (not bypassing it), confirming the routing policy.
    """
    test_bus = EventBus(db_path=tmp_path / "sl_test.db")
    routed = []
    monkeypatch.setattr(
        "src.events.bus.EventBus._route_to_telegram",
        lambda self, ev: routed.append(ev),
    )

    test_bus.emit(Event(
        type="alert",
        severity="critical",
        title="STOP-LOSS déclenché — AAPL",
        body="🛑 STOP-LOSS AAPL: -8.5%",
        meta={"symbol": "AAPL"},
    ))

    assert len(routed) == 1, "critical event (stop-loss) must be routed to Telegram via bus"
    assert "STOP-LOSS" in routed[0].title


# ── 11. Fix 3 — VALID_TYPES coverage ─────────────────────────────────────────

def test_alert_in_valid_types():
    assert "alert" in VALID_TYPES

def test_system_in_valid_types():
    assert "system" in VALID_TYPES

def test_performance_in_valid_types():
    assert "performance" in VALID_TYPES

def test_legacy_types_still_present():
    for t in ("execution", "briefing", "circuit_breaker", "regime_change",
              "error", "tearsheet", "monte_carlo", "news"):
        assert t in VALID_TYPES, f"{t!r} removed from VALID_TYPES"

def test_unknown_type_emits_warning_and_stores(tmp_path, caplog):
    """Unknown type must NOT be silently dropped — it's stored with a log warning."""
    import logging
    bus = EventBus(db_path=tmp_path / "warn_test.db")
    with caplog.at_level(logging.WARNING, logger="src.events.bus"):
        bus.emit(Event(type="__sentinel_unknown__", severity="info",
                       title="test", body=""))
    recent = bus.get_recent(limit=10)
    assert any(e["type"] == "__sentinel_unknown__" for e in recent), "unknown type must still be stored"
    assert any("unknown event type" in r.message.lower() for r in caplog.records)

def test_stop_loss_alert_visible_in_activity_feed(tmp_path, monkeypatch):
    """
    Confirms the stop-loss alert (type='alert') appears when querying the feed
    without a type_filter (simulating the dashboard Activity Feed).
    """
    bus = EventBus(db_path=tmp_path / "feed_test.db")
    monkeypatch.setattr("src.events.bus.EventBus._route_to_telegram", lambda *a: None)
    bus.emit(Event(type="alert", severity="critical",
                   title="STOP-LOSS XOM", body="🛑 XOM -9.2%",
                   meta={"symbol": "XOM", "pnl_pct": -0.092}))
    all_events = bus.get_recent(limit=50)
    stop_loss_events = [e for e in all_events if e["type"] == "alert"]
    assert len(stop_loss_events) >= 1
    assert "XOM" in stop_loss_events[0]["title"]
