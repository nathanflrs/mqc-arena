# tests/test_live_scorer.py
from __future__ import annotations

import tempfile
from pathlib import Path

import pandas as pd
import pytest

from src.risk.live_scorer import (
    LiveScorer,
    LiveScorerConfig,
    RoundTrip,
    _sharpe_from_roundtrips,
)


# ─── Helpers ─────────────────────────────────────────────────────────────────

def _write_csv(tmp_path: Path, name: str, rows: list[dict]) -> Path:
    f = tmp_path / name
    pd.DataFrame(rows).to_csv(f, index=False)
    return f


def _decisions(pid: str, sym: str, agent: str, action: str = "BUY") -> dict:
    return {
        "plan_id": pid,
        "timestamp": "2024-01-01T10:00:00+00:00",
        "symbol": sym,
        "regime": "bull",
        "agent": agent,
        "action": action,
        "confidence": 0.80,
        "target_weight": 0.10,
        "reason": "test",
        "is_winner": True,
    }


def _execution(pid: str, sym: str, side: str, price: float, ts: str) -> dict:
    return {
        "plan_id": pid,
        "timestamp": ts,
        "symbol": sym,
        "side": side,
        "qty": 10,
        "limit_price": price,
        "last_price": price,
        "est_notional": price * 10,
        "target_weight": 0.10,
        "reason": "test",
        "status": "Filled",
    }


# ─── _sharpe_from_roundtrips ─────────────────────────────────────────────────

def test_sharpe_fewer_than_3_trades_returns_zero():
    trips = [
        RoundTrip("A", "AAPL", 100.0, 110.0,
                  pd.Timestamp("2024-01-01", tz="UTC"),
                  pd.Timestamp("2024-01-10", tz="UTC")),
    ]
    assert _sharpe_from_roundtrips(trips) == 0.0


def test_sharpe_positive_for_winning_trades():
    trips = [
        RoundTrip("A", "AAPL", 100.0, 110.0,
                  pd.Timestamp("2024-01-01", tz="UTC"),
                  pd.Timestamp("2024-01-10", tz="UTC")),
        RoundTrip("A", "AAPL", 110.0, 121.0,
                  pd.Timestamp("2024-02-01", tz="UTC"),
                  pd.Timestamp("2024-02-10", tz="UTC")),
        RoundTrip("A", "AAPL", 121.0, 133.0,
                  pd.Timestamp("2024-03-01", tz="UTC"),
                  pd.Timestamp("2024-03-10", tz="UTC")),
    ]
    assert _sharpe_from_roundtrips(trips) > 0.0


def test_sharpe_negative_for_losing_trades():
    trips = [
        RoundTrip("A", "AAPL", 100.0, 90.0,
                  pd.Timestamp("2024-01-01", tz="UTC"),
                  pd.Timestamp("2024-01-10", tz="UTC")),
        RoundTrip("A", "AAPL", 90.0, 81.0,
                  pd.Timestamp("2024-02-01", tz="UTC"),
                  pd.Timestamp("2024-02-10", tz="UTC")),
        RoundTrip("A", "AAPL", 81.0, 73.0,
                  pd.Timestamp("2024-03-01", tz="UTC"),
                  pd.Timestamp("2024-03-10", tz="UTC")),
    ]
    assert _sharpe_from_roundtrips(trips) < 0.0


# ─── RoundTrip properties ─────────────────────────────────────────────────────

def test_roundtrip_return_pct():
    t = RoundTrip("A", "AAPL", 100.0, 110.0,
                  pd.Timestamp("2024-01-01", tz="UTC"),
                  pd.Timestamp("2024-01-11", tz="UTC"))
    assert t.return_pct == pytest.approx(0.10)


def test_roundtrip_holding_days():
    t = RoundTrip("A", "AAPL", 100.0, 110.0,
                  pd.Timestamp("2024-01-01", tz="UTC"),
                  pd.Timestamp("2024-01-11", tz="UTC"))
    assert t.holding_days == 10


# ─── LiveScorer — fichiers manquants ──────────────────────────────────────────

def test_no_files_returns_empty(tmp_path):
    cfg = LiveScorerConfig(
        decisions_path=str(tmp_path / "nofile.csv"),
        executions_path=str(tmp_path / "nofile2.csv"),
    )
    scorer = LiveScorer(cfg)
    assert scorer.compute_live_sharpes() == {}
    assert scorer.get_n_trades("BuffettAgent", "AAPL") == 0


def test_no_is_winner_column_returns_empty(tmp_path):
    dec = tmp_path / "dec.csv"
    pd.DataFrame([{"agent": "A", "symbol": "AAPL", "action": "BUY"}]).to_csv(dec, index=False)
    cfg = LiveScorerConfig(decisions_path=str(dec), executions_path=str(tmp_path / "x.csv"))
    scorer = LiveScorer(cfg)
    assert scorer.compute_live_sharpes() == {}


# ─── LiveScorer — round-trip complet ─────────────────────────────────────────

def test_single_roundtrip_below_min_trades(tmp_path):
    dec_rows = [_decisions("P1", "AAPL", "BuffettAgent")]
    exc_rows = [
        _execution("P1", "AAPL", "BUY",  100.0, "2024-01-01T10:00:00+00:00"),
        _execution("P2", "AAPL", "SELL", 110.0, "2024-01-15T10:00:00+00:00"),
    ]
    dec_f = _write_csv(tmp_path, "dec.csv", dec_rows)
    exc_f = _write_csv(tmp_path, "exc.csv", exc_rows)
    cfg = LiveScorerConfig(decisions_path=str(dec_f), executions_path=str(exc_f), min_trades=3)
    scorer = LiveScorer(cfg)
    assert scorer.compute_live_sharpes() == {}
    assert scorer.get_n_trades("BuffettAgent", "AAPL") == 1


def test_three_roundtrips_produces_sharpe(tmp_path):
    dec_rows = [
        _decisions("P1", "AAPL", "BuffettAgent"),
        _decisions("P3", "AAPL", "BuffettAgent"),
        _decisions("P5", "AAPL", "BuffettAgent"),
    ]
    exc_rows = [
        _execution("P1", "AAPL", "BUY",  100.0, "2024-01-01T10:00:00+00:00"),
        _execution("P2", "AAPL", "SELL", 112.0, "2024-01-15T10:00:00+00:00"),
        _execution("P3", "AAPL", "BUY",  112.0, "2024-02-01T10:00:00+00:00"),
        _execution("P4", "AAPL", "SELL", 125.0, "2024-02-15T10:00:00+00:00"),
        _execution("P5", "AAPL", "BUY",  125.0, "2024-03-01T10:00:00+00:00"),
        _execution("P6", "AAPL", "SELL", 138.0, "2024-03-15T10:00:00+00:00"),
    ]
    dec_f = _write_csv(tmp_path, "dec.csv", dec_rows)
    exc_f = _write_csv(tmp_path, "exc.csv", exc_rows)
    cfg = LiveScorerConfig(decisions_path=str(dec_f), executions_path=str(exc_f), min_trades=3)
    scorer = LiveScorer(cfg)
    sharpes = scorer.compute_live_sharpes()
    assert "BuffettAgent" in sharpes
    assert "AAPL" in sharpes["BuffettAgent"]
    assert sharpes["BuffettAgent"]["AAPL"] > 0.0
    assert scorer.get_n_trades("BuffettAgent", "AAPL") == 3


def test_two_agents_independently_scored(tmp_path):
    dec_rows = [
        _decisions("P1", "AAPL", "BuffettAgent"),
        _decisions("P3", "AAPL", "BuffettAgent"),
        _decisions("P5", "AAPL", "BuffettAgent"),
        _decisions("P7", "SPY",  "CitadelAgent"),
        _decisions("P9", "SPY",  "CitadelAgent"),
        _decisions("P11","SPY",  "CitadelAgent"),
    ]
    exc_rows = [
        # BuffettAgent / AAPL — trades gagnants
        _execution("P1",  "AAPL", "BUY",  100.0, "2024-01-01T10:00:00+00:00"),
        _execution("P2",  "AAPL", "SELL", 110.0, "2024-01-15T10:00:00+00:00"),
        _execution("P3",  "AAPL", "BUY",  110.0, "2024-02-01T10:00:00+00:00"),
        _execution("P4",  "AAPL", "SELL", 121.0, "2024-02-15T10:00:00+00:00"),
        _execution("P5",  "AAPL", "BUY",  121.0, "2024-03-01T10:00:00+00:00"),
        _execution("P6",  "AAPL", "SELL", 133.0, "2024-03-15T10:00:00+00:00"),
        # CitadelAgent / SPY — trades perdants
        _execution("P7",  "SPY",  "BUY",  400.0, "2024-01-01T10:00:00+00:00"),
        _execution("P8",  "SPY",  "SELL", 380.0, "2024-01-15T10:00:00+00:00"),
        _execution("P9",  "SPY",  "BUY",  380.0, "2024-02-01T10:00:00+00:00"),
        _execution("P10", "SPY",  "SELL", 360.0, "2024-02-15T10:00:00+00:00"),
        _execution("P11", "SPY",  "BUY",  360.0, "2024-03-01T10:00:00+00:00"),
        _execution("P12", "SPY",  "SELL", 342.0, "2024-03-15T10:00:00+00:00"),
    ]
    dec_f = _write_csv(tmp_path, "dec.csv", dec_rows)
    exc_f = _write_csv(tmp_path, "exc.csv", exc_rows)
    cfg = LiveScorerConfig(decisions_path=str(dec_f), executions_path=str(exc_f), min_trades=3)
    scorer = LiveScorer(cfg)
    sharpes = scorer.compute_live_sharpes()
    assert sharpes["BuffettAgent"]["AAPL"] > 0
    assert sharpes["CitadelAgent"]["SPY"] < 0


def test_get_roundtrips_filter(tmp_path):
    dec_rows = [_decisions("P1", "AAPL", "BuffettAgent")]
    exc_rows = [
        _execution("P1", "AAPL", "BUY",  100.0, "2024-01-01T10:00:00+00:00"),
        _execution("P2", "AAPL", "SELL", 110.0, "2024-01-10T10:00:00+00:00"),
    ]
    dec_f = _write_csv(tmp_path, "dec.csv", dec_rows)
    exc_f = _write_csv(tmp_path, "exc.csv", exc_rows)
    cfg = LiveScorerConfig(decisions_path=str(dec_f), executions_path=str(exc_f))
    scorer = LiveScorer(cfg)
    trips = scorer.get_roundtrips(agent="BuffettAgent", symbol="AAPL")
    assert len(trips) == 1
    assert trips[0].return_pct == pytest.approx(0.10)
