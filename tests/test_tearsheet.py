# tests/test_tearsheet.py
from __future__ import annotations

import pathlib
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.risk.live_scorer import (
    AgentMetrics,
    LiveScorer,
    LiveScorerConfig,
    RoundTrip,
    _max_drawdown_from_trips,
    _sharpe_from_roundtrips,
)


# ─── Helpers ─────────────────────────────────────────────────────────────────

def _ts(date_str: str) -> pd.Timestamp:
    return pd.Timestamp(date_str, tz="UTC")


def _trip(
    agent: str,
    entry: float,
    exit_: float,
    entry_date: str = "2024-01-01",
    exit_date: str = "2024-01-15",
    symbol: str = "AAPL",
) -> RoundTrip:
    return RoundTrip(
        agent=agent,
        symbol=symbol,
        entry_price=entry,
        exit_price=exit_,
        entry_date=_ts(entry_date),
        exit_date=_ts(exit_date),
    )


def _write_csv(tmp_path: Path, name: str, rows: list[dict]) -> Path:
    f = tmp_path / name
    pd.DataFrame(rows).to_csv(f, index=False)
    return f


def _decisions(pid: str, sym: str, agent: str) -> dict:
    return {
        "plan_id": pid, "timestamp": "2024-01-01T10:00:00+00:00",
        "symbol": sym, "regime": "bull", "agent": agent,
        "action": "BUY", "confidence": 0.80, "target_weight": 0.10,
        "reason": "test", "is_winner": True,
    }


def _execution(pid: str, sym: str, side: str, price: float, ts: str) -> dict:
    return {
        "plan_id": pid, "timestamp": ts, "symbol": sym, "side": side,
        "qty": 10, "limit_price": price, "last_price": price,
        "est_notional": price * 10, "target_weight": 0.10,
        "reason": "test", "status": "Filled",
    }


# ─── _max_drawdown_from_trips ─────────────────────────────────────────────────

def test_drawdown_no_trips():
    assert _max_drawdown_from_trips([]) == 0.0


def test_drawdown_single_win_no_drawdown():
    trips = [_trip("A", 100, 110)]
    dd = _max_drawdown_from_trips(trips)
    assert dd == pytest.approx(0.0, abs=1e-9)


def test_drawdown_loss_after_win():
    """Win then loss → drawdown = loss / peak."""
    trips = [
        _trip("A", 100, 120, "2024-01-01", "2024-01-10"),  # +20%
        _trip("A", 100, 80,  "2024-01-11", "2024-01-20"),  # -20%
    ]
    dd = _max_drawdown_from_trips(trips)
    # After trip 1: equity = 1.20
    # After trip 2: equity = 1.20 * 0.80 = 0.96; peak = 1.20
    # DD = (0.96 - 1.20) / 1.20 = -0.20
    assert dd == pytest.approx(-0.20, abs=1e-6)


def test_drawdown_all_wins_is_zero():
    trips = [
        _trip("A", 100, 110, "2024-01-01", "2024-01-10"),
        _trip("A", 100, 105, "2024-01-11", "2024-01-20"),
        _trip("A", 100, 103, "2024-01-21", "2024-01-30"),
    ]
    dd = _max_drawdown_from_trips(trips)
    assert dd == pytest.approx(0.0, abs=1e-9)


# ─── AgentMetrics ─────────────────────────────────────────────────────────────

def test_agent_metrics_win_rate():
    trips = [
        _trip("A", 100, 110),  # win
        _trip("A", 100, 90),   # loss
        _trip("A", 100, 105),  # win
    ]
    returns = np.array([t.return_pct for t in trips])
    n = len(trips)
    m = AgentMetrics(
        agent="A",
        n_trades=n,
        win_rate=float(np.mean(returns > 0)),
        avg_return_pct=float(np.mean(returns)),
        total_pnl_pct=float(np.sum(returns)),
        max_drawdown=_max_drawdown_from_trips(trips),
        sharpe=_sharpe_from_roundtrips(trips),
        avg_holding_days=14.0,
    )
    assert m.win_rate == pytest.approx(2 / 3)


def test_agent_metrics_to_dict_keys():
    m = AgentMetrics("A", 5, 0.6, 0.05, 0.25, -0.10, 1.5, 12.0)
    d = m.to_dict()
    expected_keys = {
        "agent", "n_trades", "win_rate", "avg_return_pct",
        "total_pnl_pct", "max_drawdown", "sharpe", "avg_holding_days",
    }
    assert expected_keys == set(d.keys())


# ─── compute_agent_metrics ────────────────────────────────────────────────────

def test_compute_agent_metrics_no_data(tmp_path):
    cfg = LiveScorerConfig(
        decisions_path=str(tmp_path / "x.csv"),
        executions_path=str(tmp_path / "y.csv"),
    )
    scorer = LiveScorer(cfg)
    metrics = scorer.compute_agent_metrics()
    assert metrics == {}


def test_compute_agent_metrics_single_agent(tmp_path):
    dec_rows = [
        _decisions("P1", "AAPL", "BuffettAgent"),
        _decisions("P3", "AAPL", "BuffettAgent"),
        _decisions("P5", "AAPL", "BuffettAgent"),
    ]
    exc_rows = [
        _execution("P1", "AAPL", "BUY",  100.0, "2024-01-01T10:00:00+00:00"),
        _execution("P2", "AAPL", "SELL", 110.0, "2024-01-15T10:00:00+00:00"),
        _execution("P3", "AAPL", "BUY",  110.0, "2024-02-01T10:00:00+00:00"),
        _execution("P4", "AAPL", "SELL", 121.0, "2024-02-15T10:00:00+00:00"),
        _execution("P5", "AAPL", "BUY",  121.0, "2024-03-01T10:00:00+00:00"),
        _execution("P6", "AAPL", "SELL", 133.0, "2024-03-15T10:00:00+00:00"),
    ]
    dec_f = _write_csv(tmp_path, "dec.csv", dec_rows)
    exc_f = _write_csv(tmp_path, "exc.csv", exc_rows)
    cfg = LiveScorerConfig(decisions_path=str(dec_f), executions_path=str(exc_f))
    scorer = LiveScorer(cfg)
    metrics = scorer.compute_agent_metrics()
    assert "BuffettAgent" in metrics
    m = metrics["BuffettAgent"]
    assert m.n_trades == 3
    assert m.win_rate == pytest.approx(1.0)
    assert m.total_pnl_pct > 0
    assert m.avg_return_pct > 0


def test_compute_agent_metrics_multiple_agents(tmp_path):
    dec_rows = [
        _decisions("P1", "AAPL", "BuffettAgent"),
        _decisions("P3", "AAPL", "BuffettAgent"),
        _decisions("P5", "AAPL", "BuffettAgent"),
        _decisions("P7", "SPY",  "CitadelAgent"),
        _decisions("P9", "SPY",  "CitadelAgent"),
        _decisions("P11","SPY",  "CitadelAgent"),
    ]
    exc_rows = [
        _execution("P1",  "AAPL", "BUY",  100.0, "2024-01-01T10:00:00+00:00"),
        _execution("P2",  "AAPL", "SELL", 110.0, "2024-01-15T10:00:00+00:00"),
        _execution("P3",  "AAPL", "BUY",  110.0, "2024-02-01T10:00:00+00:00"),
        _execution("P4",  "AAPL", "SELL", 121.0, "2024-02-15T10:00:00+00:00"),
        _execution("P5",  "AAPL", "BUY",  121.0, "2024-03-01T10:00:00+00:00"),
        _execution("P6",  "AAPL", "SELL", 133.0, "2024-03-15T10:00:00+00:00"),
        _execution("P7",  "SPY",  "BUY",  400.0, "2024-01-01T10:00:00+00:00"),
        _execution("P8",  "SPY",  "SELL", 380.0, "2024-01-15T10:00:00+00:00"),
        _execution("P9",  "SPY",  "BUY",  380.0, "2024-02-01T10:00:00+00:00"),
        _execution("P10", "SPY",  "SELL", 360.0, "2024-02-15T10:00:00+00:00"),
        _execution("P11", "SPY",  "BUY",  360.0, "2024-03-01T10:00:00+00:00"),
        _execution("P12", "SPY",  "SELL", 342.0, "2024-03-15T10:00:00+00:00"),
    ]
    dec_f = _write_csv(tmp_path, "dec.csv", dec_rows)
    exc_f = _write_csv(tmp_path, "exc.csv", exc_rows)
    cfg = LiveScorerConfig(decisions_path=str(dec_f), executions_path=str(exc_f))
    metrics = LiveScorer(cfg).compute_agent_metrics()
    assert metrics["BuffettAgent"].total_pnl_pct > 0
    assert metrics["CitadelAgent"].total_pnl_pct < 0
    assert metrics["CitadelAgent"].win_rate == pytest.approx(0.0)


# ─── generate_tearsheet ───────────────────────────────────────────────────────

def test_generate_tearsheet_creates_csv(tmp_path):
    dec_rows = [
        _decisions("P1", "AAPL", "BuffettAgent"),
        _decisions("P3", "AAPL", "BuffettAgent"),
        _decisions("P5", "AAPL", "BuffettAgent"),
    ]
    exc_rows = [
        _execution("P1", "AAPL", "BUY",  100.0, "2024-01-01T10:00:00+00:00"),
        _execution("P2", "AAPL", "SELL", 110.0, "2024-01-15T10:00:00+00:00"),
        _execution("P3", "AAPL", "BUY",  110.0, "2024-02-01T10:00:00+00:00"),
        _execution("P4", "AAPL", "SELL", 121.0, "2024-02-15T10:00:00+00:00"),
        _execution("P5", "AAPL", "BUY",  121.0, "2024-03-01T10:00:00+00:00"),
        _execution("P6", "AAPL", "SELL", 133.0, "2024-03-15T10:00:00+00:00"),
    ]
    path = str(tmp_path / "tearsheet_test.csv")
    cfg = LiveScorerConfig(
        decisions_path=str(_write_csv(tmp_path, "dec.csv", dec_rows)),
        executions_path=str(_write_csv(tmp_path, "exc.csv", exc_rows)),
    )
    scorer = LiveScorer(cfg)
    returned_path = scorer.generate_tearsheet(path)
    assert returned_path == path
    assert pathlib.Path(path).exists()


def test_generate_tearsheet_columns(tmp_path):
    dec_rows = [
        _decisions("P1", "AAPL", "BuffettAgent"),
        _decisions("P3", "AAPL", "BuffettAgent"),
        _decisions("P5", "AAPL", "BuffettAgent"),
    ]
    exc_rows = [
        _execution("P1", "AAPL", "BUY",  100.0, "2024-01-01T10:00:00+00:00"),
        _execution("P2", "AAPL", "SELL", 110.0, "2024-01-15T10:00:00+00:00"),
        _execution("P3", "AAPL", "BUY",  110.0, "2024-02-01T10:00:00+00:00"),
        _execution("P4", "AAPL", "SELL", 121.0, "2024-02-15T10:00:00+00:00"),
        _execution("P5", "AAPL", "BUY",  121.0, "2024-03-01T10:00:00+00:00"),
        _execution("P6", "AAPL", "SELL", 133.0, "2024-03-15T10:00:00+00:00"),
    ]
    path = str(tmp_path / "sheet.csv")
    cfg = LiveScorerConfig(
        decisions_path=str(_write_csv(tmp_path, "dec.csv", dec_rows)),
        executions_path=str(_write_csv(tmp_path, "exc.csv", exc_rows)),
    )
    scorer = LiveScorer(cfg)
    scorer.generate_tearsheet(path)
    df = pd.read_csv(path)
    expected_cols = {
        "agent", "n_trades", "win_rate", "avg_return_pct",
        "total_pnl_pct", "max_drawdown", "sharpe", "avg_holding_days",
    }
    assert expected_cols.issubset(set(df.columns))


def test_generate_tearsheet_noop_when_no_trades(tmp_path):
    path = str(tmp_path / "empty.csv")
    cfg = LiveScorerConfig(
        decisions_path=str(tmp_path / "x.csv"),
        executions_path=str(tmp_path / "y.csv"),
    )
    scorer = LiveScorer(cfg)
    scorer.generate_tearsheet(path)
    # No data → file not created
    assert not pathlib.Path(path).exists()


def test_tearsheet_sorted_by_sharpe_descending(tmp_path):
    """Highest Sharpe agent should appear in first row."""
    dec_rows = [
        _decisions("P1", "AAPL", "WinAgent"),
        _decisions("P3", "AAPL", "WinAgent"),
        _decisions("P5", "AAPL", "WinAgent"),
        _decisions("P7", "SPY",  "LoseAgent"),
        _decisions("P9", "SPY",  "LoseAgent"),
        _decisions("P11","SPY",  "LoseAgent"),
    ]
    exc_rows = [
        _execution("P1",  "AAPL", "BUY",  100.0, "2024-01-01T10:00:00+00:00"),
        _execution("P2",  "AAPL", "SELL", 120.0, "2024-01-15T10:00:00+00:00"),
        _execution("P3",  "AAPL", "BUY",  120.0, "2024-02-01T10:00:00+00:00"),
        _execution("P4",  "AAPL", "SELL", 140.0, "2024-02-15T10:00:00+00:00"),
        _execution("P5",  "AAPL", "BUY",  140.0, "2024-03-01T10:00:00+00:00"),
        _execution("P6",  "AAPL", "SELL", 160.0, "2024-03-15T10:00:00+00:00"),
        _execution("P7",  "SPY",  "BUY",  400.0, "2024-01-01T10:00:00+00:00"),
        _execution("P8",  "SPY",  "SELL", 360.0, "2024-01-15T10:00:00+00:00"),
        _execution("P9",  "SPY",  "BUY",  360.0, "2024-02-01T10:00:00+00:00"),
        _execution("P10", "SPY",  "SELL", 320.0, "2024-02-15T10:00:00+00:00"),
        _execution("P11", "SPY",  "BUY",  320.0, "2024-03-01T10:00:00+00:00"),
        _execution("P12", "SPY",  "SELL", 280.0, "2024-03-15T10:00:00+00:00"),
    ]
    path = str(tmp_path / "sorted.csv")
    cfg = LiveScorerConfig(
        decisions_path=str(_write_csv(tmp_path, "dec.csv", dec_rows)),
        executions_path=str(_write_csv(tmp_path, "exc.csv", exc_rows)),
    )
    scorer = LiveScorer(cfg)
    scorer.generate_tearsheet(path)
    df = pd.read_csv(path)
    assert df.iloc[0]["agent"] == "WinAgent"
    assert df.iloc[1]["agent"] == "LoseAgent"
