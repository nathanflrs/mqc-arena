"""
Live track record — one SHA256-signed row per runner run.
File: logs/live_track_record.csv

Each row captures what the system decided that day and is signed with a
SHA256 hash computed over all fields (excluding the hash itself).
Anyone can recompute the hash to verify the row was never edited.
"""
from __future__ import annotations

import csv
import hashlib
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

LOG_PATH = Path("logs/live_track_record.csv")

COLUMNS = [
    "date", "plan_id", "regime",
    "n_signals", "n_buy", "n_sell", "n_hold",
    "top_agent", "netliq", "sha256",
]


def _make_hash(row: dict) -> str:
    """SHA256 of all fields except sha256 itself, pipe-delimited."""
    payload = "|".join(str(row[c]) for c in COLUMNS if c != "sha256")
    return hashlib.sha256(payload.encode()).hexdigest()


def append_daily_entry(
    *,
    plan_id: str,
    regime: str,
    decisions_summary: list[dict],
    netliq: Optional[float] = None,
) -> str:
    """
    Append one signed row to logs/live_track_record.csv.

    decisions_summary: list of {"symbol": str, "agent": str, "action": str}
        collected during the runner's symbol loop (one entry per watchlist symbol).

    Returns the sha256 digest of the written row.
    """
    Path("logs").mkdir(parents=True, exist_ok=True)
    date_str = datetime.now(timezone.utc).strftime("%Y-%m-%d")

    actions = Counter(d.get("action", "HOLD") for d in decisions_summary)
    agents = Counter(
        d["agent"]
        for d in decisions_summary
        if d.get("agent") and d["agent"] not in ("NONE", "")
    )
    top_agent = agents.most_common(1)[0][0] if agents else "NONE"

    row: dict = {
        "date":      date_str,
        "plan_id":   plan_id,
        "regime":    regime,
        "n_signals": len(decisions_summary),
        "n_buy":     actions.get("BUY", 0),
        "n_sell":    actions.get("SELL", 0),
        "n_hold":    actions.get("HOLD", 0),
        "top_agent": top_agent,
        "netliq":    round(netliq, 2) if netliq is not None else "",
        "sha256":    "",
    }
    row["sha256"] = _make_hash(row)

    write_header = not LOG_PATH.exists()
    with LOG_PATH.open("a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=COLUMNS)
        if write_header:
            writer.writeheader()
        writer.writerow(row)

    return row["sha256"]


def verify(path: Path = LOG_PATH) -> list[tuple[int, str]]:
    """
    Verify every row's SHA256 hash.
    Returns a list of (line_number, date) for any tampered rows.
    Empty list means the entire file is intact.
    """
    if not path.exists():
        return []

    bad: list[tuple[int, str]] = []
    with path.open(newline="") as f:
        reader = csv.DictReader(f)
        for i, row in enumerate(reader, start=2):  # line 1 = header
            stored = row.get("sha256", "")
            row["sha256"] = ""
            expected = _make_hash(dict(row))
            if stored != expected:
                bad.append((i, row.get("date", "?")))

    return bad
