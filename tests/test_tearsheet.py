# tests/test_tearsheet.py
from __future__ import annotations

import pandas as pd
import pytest

from src.risk.live_scorer import (
    LiveScorer,
    LiveScorerConfig,
    RoundTrip,
    AgentMetrics,
    kelly_half_fraction,
    _max_drawdown_from_trips,
)


# ─── Helpers ──────────────────────────────────────────────────────────────────

def _trip(agent: str, ret: float, entry: str = "2024-01-01", hold: int = 10) -> RoundTrip:
    entry_ts = pd.Timestamp(entry, tz="UTC")
    exit_ts  = entry_ts + pd.Timedelta(days=hold)
    entry_price = 100.0
    exit_price  = entry_price * (1 + ret)
    return RoundTrip(
        agent=agent, symbol="TEST",
        entry_price=entry_price, exit_price=exit_price,
        entry_date=entry_ts, exit_date=exit_ts,
    )


def _write_csvs(tmp_path, decisions, executions):
    dec = tmp_path / "dec.csv"
    exc = tmp_path / "exc.csv"
    pd.DataFrame(decisions).to_csv(dec, index=False)
    pd.DataFrame(executions).to_csv(exc, index=False)
    return str(dec), str(exc)


# ─── kelly_half_fraction ──────────────────────────────────────────────────────

def test_kelly_returns_zero_below_min_trades():
    trips = [_trip("A", 0.1)] * 9  # only 9
    assert kelly_half_fraction(trips, min_trades=10) == 0.0


def test_kelly_positive_for_winning_strategy():
    trips = [_trip("A", 0.10)] * 8 + [_trip("A", -0.02)] * 2  # 80% win rate
    frac = kelly_half_fraction(trips, min_trades=10)
    assert frac > 0.0


def test_kelly_zero_for_losing_strategy():
    trips = [_trip("A", -0.10)] * 8 + [_trip("A", 0.01)] * 2  # 20% win rate
    frac = kelly_half_fraction(trips, min_trades=10)
    assert frac == 0.0  # clipped at 0


def test_kelly_capped_at_max_fraction():
    # Perfect win rate → large f*, but capped
    trips = [_trip("A", 0.50)] * 15  # extreme winners
    frac = kelly_half_fraction(trips, min_trades=10, max_fraction=0.25)
    assert frac <= 0.25


def test_kelly_all_wins_no_losses():
    # If no losses, b is undefined — should return 0.0 safely
    trips = [_trip("A", 0.10)] * 10
    frac = kelly_half_fraction(trips, min_trades=10)
    assert frac == 0.0


def test_kelly_all_losses_no_wins():
    trips = [_trip("A", -0.10)] * 10
    frac = kelly_half_fraction(trips, min_trades=10)
    assert frac == 0.0


# ─── _max_drawdown_from_trips ─────────────────────────────────────────────────

def test_max_drawdown_empty_trips():
    assert _max_drawdown_from_trips([]) == 0.0


def test_max_drawdown_all_winners():
    trips = [_trip("A", 0.10)] * 5
    dd = _max_drawdown_from_trips(trips)
    assert dd == pytest.approx(0.0, abs=1e-9)


def test_max_drawdown_alternating():
    # +10%, -20%, +10% → cumulative: 1.1, 0.88, 0.968 → DD from 1.1 to 0.88
    trips = [
        _trip("A",  0.10, "2024-01-01"),
        _trip("A", -0.20, "2024-02-01"),
        _trip("A",  0.10, "2024-03-01"),
    ]
    dd = _max_drawdown_from_trips(trips)
    assert dd < 0.0  # some drawdown occurred


# ─── AgentMetrics via LiveScorer ─────────────────────────────────────────────

def test_compute_agent_metrics_empty(tmp_path):
    cfg = LiveScorerConfig(
        decisions_path=str(tmp_path / "x.csv"),
        executions_path=str(tmp_path / "y.csv"),
    )
    metrics = LiveScorer(cfg).compute_agent_metrics()
    assert metrics == {}


def test_compute_agent_metrics_single_agent(tmp_path):
    dec_rows = [
        {"plan_id": f"P{i}", "timestamp": "2024-01-01", "symbol": "AAPL",
         "regime": "bull", "agent": "BuffettAgent", "action": "BUY",
         "confidence": 0.8, "target_weight": 0.10, "reason": "t", "is_winner": True}
        for i in range(1, 4)
    ]
    exc_rows = [
        {"plan_id": "P1", "timestamp": "2024-01-01T10:00:00+00:00", "symbol": "AAPL",
         "side": "BUY",  "qty": 10, "limit_price": 100.0, "last_price": 100.0,
         "est_notional": 1000, "target_weight": 0.1, "reason": "t", "status": "Filled"},
        {"plan_id": "P2", "timestamp": "2024-01-15T10:00:00+00:00", "symbol": "AAPL",
         "side": "SELL", "qty": 10, "limit_price": 112.0, "last_price": 112.0,
         "est_notional": 1120, "target_weight": 0.1, "reason": "t", "status": "Filled"},
        {"plan_id": "P2", "timestamp": "2024-02-01T10:00:00+00:00", "symbol": "AAPL",
         "side": "BUY",  "qty": 10, "limit_price": 112.0, "last_price": 112.0,
         "est_notional": 1120, "target_weight": 0.1, "reason": "t", "status": "Filled"},
        {"plan_id": "P3", "timestamp": "2024-02-15T10:00:00+00:00", "symbol": "AAPL",
         "side": "SELL", "qty": 10, "limit_price": 125.0, "last_price": 125.0,
         "est_notional": 1250, "target_weight": 0.1, "reason": "t", "status": "Filled"},
    ]
    dec_path, exc_path = _write_csvs(tmp_path, dec_rows, exc_rows)
    cfg = LiveScorerConfig(decisions_path=dec_path, executions_path=exc_path)
    metrics = LiveScorer(cfg).compute_agent_metrics()
    assert "BuffettAgent" in metrics
    m = metrics["BuffettAgent"]
    assert m.n_trades == 2
    assert m.win_rate == pytest.approx(1.0)
    assert m.avg_return_pct > 0
    assert m.total_pnl_pct > 0


# ─── compute_kelly_weights via LiveScorer ────────────────────────────────────

def test_kelly_weights_empty_returns_empty(tmp_path):
    cfg = LiveScorerConfig(
        decisions_path=str(tmp_path / "x.csv"),
        executions_path=str(tmp_path / "y.csv"),
    )
    assert LiveScorer(cfg).compute_kelly_weights(min_trades=10) == {}


# ─── generate_tearsheet ───────────────────────────────────────────────────────

def test_generate_tearsheet_creates_csv(tmp_path):
    dec_rows = [
        {"plan_id": "P1", "timestamp": "2024-01-01", "symbol": "AAPL",
         "regime": "bull", "agent": "BuffettAgent", "action": "BUY",
         "confidence": 0.8, "target_weight": 0.10, "reason": "t", "is_winner": True}
    ]
    exc_rows = [
        {"plan_id": "P1", "timestamp": "2024-01-01T10:00:00+00:00", "symbol": "AAPL",
         "side": "BUY",  "qty": 10, "limit_price": 100.0, "last_price": 100.0,
         "est_notional": 1000, "target_weight": 0.1, "reason": "t", "status": "Filled"},
        {"plan_id": "P2", "timestamp": "2024-01-15T10:00:00+00:00", "symbol": "AAPL",
         "side": "SELL", "qty": 10, "limit_price": 115.0, "last_price": 115.0,
         "est_notional": 1150, "target_weight": 0.1, "reason": "t", "status": "Filled"},
    ]
    dec_path, exc_path = _write_csvs(tmp_path, dec_rows, exc_rows)
    cfg = LiveScorerConfig(decisions_path=dec_path, executions_path=exc_path)
    out = str(tmp_path / "tearsheet_test.csv")
    path = LiveScorer(cfg).generate_tearsheet(path=out)
    df = pd.read_csv(path)
    assert "agent" in df.columns
    assert "sharpe" in df.columns
    assert "n_trades" in df.columns
    assert len(df) >= 1


def test_generate_tearsheet_no_data(tmp_path):
    cfg = LiveScorerConfig(
        decisions_path=str(tmp_path / "x.csv"),
        executions_path=str(tmp_path / "y.csv"),
    )
    # Should not raise — just returns path without writing
    path = LiveScorer(cfg).generate_tearsheet(path=str(tmp_path / "ts.csv"))
    assert isinstance(path, str)
