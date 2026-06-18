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


# ─── compute_portfolio_performance ───────────────────────────────────────────

def test_portfolio_performance_no_trades_returns_none(tmp_path):
    cfg = LiveScorerConfig(
        decisions_path=str(tmp_path / "x.csv"),
        executions_path=str(tmp_path / "y.csv"),
    )
    assert LiveScorer(cfg).compute_portfolio_performance() is None


def test_portfolio_performance_basic(tmp_path):
    from unittest.mock import patch

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
    with patch("src.data.market_data.download_ohlcv", side_effect=Exception("no network")):
        perf = scorer.compute_portfolio_performance()

    assert perf is not None
    assert perf.n_trades == 3
    assert perf.portfolio_return > 0.0
    assert perf.spy_return == 0.0          # fallback when SPY fetch fails
    assert perf.alpha == pytest.approx(perf.portfolio_return)
    assert len(perf.equity_curve) == 3
    assert perf.equity_curve[0]["portfolio"] > 100.0
    assert perf.first_trade_date == "2024-01-01"
    assert perf.last_trade_date == "2024-03-15"


def test_portfolio_performance_equity_indexed_to_100(tmp_path):
    from unittest.mock import patch

    dec_rows = [_decisions("P1", "AAPL", "BuffettAgent")]
    exc_rows = [
        _execution("P1", "AAPL", "BUY",  100.0, "2024-01-01T10:00:00+00:00"),
        _execution("P2", "AAPL", "SELL", 110.0, "2024-01-15T10:00:00+00:00"),
    ]
    dec_f = _write_csv(tmp_path, "dec.csv", dec_rows)
    exc_f = _write_csv(tmp_path, "exc.csv", exc_rows)
    cfg = LiveScorerConfig(decisions_path=str(dec_f), executions_path=str(exc_f))
    scorer = LiveScorer(cfg)
    with patch("src.data.market_data.download_ohlcv", side_effect=Exception("no network")):
        perf = scorer.compute_portfolio_performance()
    assert perf is not None
    # +10% on base 100 → 110
    assert perf.equity_curve[0]["portfolio"] == pytest.approx(110.0, rel=1e-3)


def test_portfolio_performance_losing_trades(tmp_path):
    from unittest.mock import patch

    dec_rows = [
        _decisions("P1", "AAPL", "BuffettAgent"),
        _decisions("P3", "AAPL", "BuffettAgent"),
    ]
    exc_rows = [
        _execution("P1", "AAPL", "BUY",  100.0, "2024-01-01T10:00:00+00:00"),
        _execution("P2", "AAPL", "SELL",  90.0, "2024-01-15T10:00:00+00:00"),
        _execution("P3", "AAPL", "BUY",   90.0, "2024-02-01T10:00:00+00:00"),
        _execution("P4", "AAPL", "SELL",  81.0, "2024-02-15T10:00:00+00:00"),
    ]
    dec_f = _write_csv(tmp_path, "dec.csv", dec_rows)
    exc_f = _write_csv(tmp_path, "exc.csv", exc_rows)
    cfg = LiveScorerConfig(decisions_path=str(dec_f), executions_path=str(exc_f))
    scorer = LiveScorer(cfg)
    with patch("src.data.market_data.download_ohlcv", side_effect=Exception("no network")):
        perf = scorer.compute_portfolio_performance()
    assert perf is not None
    assert perf.portfolio_return < 0.0
    assert perf.equity_curve[-1]["portfolio"] < 100.0


def test_portfolio_performance_to_dict_keys(tmp_path):
    from unittest.mock import patch

    dec_rows = [_decisions("P1", "AAPL", "BuffettAgent")]
    exc_rows = [
        _execution("P1", "AAPL", "BUY",  100.0, "2024-01-01T10:00:00+00:00"),
        _execution("P2", "AAPL", "SELL", 110.0, "2024-01-15T10:00:00+00:00"),
    ]
    dec_f = _write_csv(tmp_path, "dec.csv", dec_rows)
    exc_f = _write_csv(tmp_path, "exc.csv", exc_rows)
    scorer = LiveScorer(LiveScorerConfig(decisions_path=str(dec_f), executions_path=str(exc_f)))
    with patch("src.data.market_data.download_ohlcv", side_effect=Exception("no network")):
        perf = scorer.compute_portfolio_performance()
    d = perf.to_dict()
    assert {"portfolio_return", "spy_return", "alpha", "n_trades",
            "first_trade_date", "last_trade_date"}.issubset(d.keys())
    assert d["n_trades"] == 1


# ─── compute_drift_alerts ─────────────────────────────────────────────────────

def test_drift_alerts_no_wf_file_returns_empty(tmp_path):
    cfg = LiveScorerConfig(
        decisions_path=str(tmp_path / "x.csv"),
        executions_path=str(tmp_path / "y.csv"),
    )
    alerts = LiveScorer(cfg).compute_drift_alerts(wf_path=str(tmp_path / "nope.csv"))
    assert alerts == []


def test_drift_alerts_no_live_data_returns_empty(tmp_path):
    wf_path = tmp_path / "wf.csv"
    pd.DataFrame([
        {"agent": "BuffettAgent", "symbol": "AAPL", "avg_oos_sharpe": 2.0},
    ]).to_csv(wf_path, index=False)
    cfg = LiveScorerConfig(
        decisions_path=str(tmp_path / "x.csv"),
        executions_path=str(tmp_path / "y.csv"),
    )
    alerts = LiveScorer(cfg).compute_drift_alerts(wf_path=str(wf_path))
    assert alerts == []


def test_drift_alerts_below_threshold_returns_empty(tmp_path):
    # oos_sharpe 0.8, live Sharpe will be very high (3 winning trades) → no alert
    wf_path = tmp_path / "wf.csv"
    pd.DataFrame([
        {"agent": "BuffettAgent", "symbol": "AAPL", "avg_oos_sharpe": 0.8},
    ]).to_csv(wf_path, index=False)

    dec_rows = [_decisions(f"P{i*2-1}", "AAPL", "BuffettAgent") for i in range(1, 4)]
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
    cfg = LiveScorerConfig(decisions_path=str(dec_f), executions_path=str(exc_f), min_trades=3)
    scorer = LiveScorer(cfg)
    alerts = scorer.compute_drift_alerts(drift_threshold=0.5, wf_path=str(wf_path))
    assert alerts == []


def test_drift_alerts_above_threshold_detected(tmp_path):
    # oos_sharpe=5.0, live Sharpe very negative (3 losing trades) → large drift
    wf_path = tmp_path / "wf.csv"
    pd.DataFrame([
        {"agent": "BuffettAgent", "symbol": "AAPL", "avg_oos_sharpe": 5.0},
    ]).to_csv(wf_path, index=False)

    dec_rows = [_decisions(f"P{i*2-1}", "AAPL", "BuffettAgent") for i in range(1, 4)]
    exc_rows = [
        _execution("P1", "AAPL", "BUY",  100.0, "2024-01-01T10:00:00+00:00"),
        _execution("P2", "AAPL", "SELL",  90.0, "2024-01-15T10:00:00+00:00"),
        _execution("P3", "AAPL", "BUY",   90.0, "2024-02-01T10:00:00+00:00"),
        _execution("P4", "AAPL", "SELL",  81.0, "2024-02-15T10:00:00+00:00"),
        _execution("P5", "AAPL", "BUY",   81.0, "2024-03-01T10:00:00+00:00"),
        _execution("P6", "AAPL", "SELL",  73.0, "2024-03-15T10:00:00+00:00"),
    ]
    dec_f = _write_csv(tmp_path, "dec.csv", dec_rows)
    exc_f = _write_csv(tmp_path, "exc.csv", exc_rows)
    cfg = LiveScorerConfig(decisions_path=str(dec_f), executions_path=str(exc_f), min_trades=3)
    scorer = LiveScorer(cfg)
    alerts = scorer.compute_drift_alerts(drift_threshold=0.5, wf_path=str(wf_path))
    assert len(alerts) == 1
    a = alerts[0]
    assert a.agent == "BuffettAgent"
    assert a.oos_sharpe == pytest.approx(5.0)
    assert a.drift > 0.5


def test_drift_alerts_sorted_by_drift_descending(tmp_path):
    # Two agents, both with large drift but different magnitudes → sorted desc
    wf_path = tmp_path / "wf.csv"
    pd.DataFrame([
        {"agent": "BuffettAgent", "symbol": "AAPL", "avg_oos_sharpe": 10.0},
        {"agent": "CitadelAgent", "symbol": "SPY",  "avg_oos_sharpe": 5.0},
    ]).to_csv(wf_path, index=False)

    dec_rows = (
        [_decisions(f"P{i*2-1}",    "AAPL", "BuffettAgent") for i in range(1, 4)] +
        [_decisions(f"P{i*2-1+6}",  "SPY",  "CitadelAgent") for i in range(1, 4)]
    )
    exc_rows = [
        _execution("P1",  "AAPL", "BUY",  100.0, "2024-01-01T10:00:00+00:00"),
        _execution("P2",  "AAPL", "SELL",  90.0, "2024-01-15T10:00:00+00:00"),
        _execution("P3",  "AAPL", "BUY",   90.0, "2024-02-01T10:00:00+00:00"),
        _execution("P4",  "AAPL", "SELL",  81.0, "2024-02-15T10:00:00+00:00"),
        _execution("P5",  "AAPL", "BUY",   81.0, "2024-03-01T10:00:00+00:00"),
        _execution("P6",  "AAPL", "SELL",  73.0, "2024-03-15T10:00:00+00:00"),
        _execution("P7",  "SPY",  "BUY",  400.0, "2024-01-01T10:00:00+00:00"),
        _execution("P8",  "SPY",  "SELL", 360.0, "2024-01-15T10:00:00+00:00"),
        _execution("P9",  "SPY",  "BUY",  360.0, "2024-02-01T10:00:00+00:00"),
        _execution("P10", "SPY",  "SELL", 324.0, "2024-02-15T10:00:00+00:00"),
        _execution("P11", "SPY",  "BUY",  324.0, "2024-03-01T10:00:00+00:00"),
        _execution("P12", "SPY",  "SELL", 292.0, "2024-03-15T10:00:00+00:00"),
    ]
    dec_f = _write_csv(tmp_path, "dec.csv", dec_rows)
    exc_f = _write_csv(tmp_path, "exc.csv", exc_rows)
    cfg = LiveScorerConfig(decisions_path=str(dec_f), executions_path=str(exc_f), min_trades=3)
    scorer = LiveScorer(cfg)
    alerts = scorer.compute_drift_alerts(drift_threshold=0.5, wf_path=str(wf_path))
    assert len(alerts) >= 2
    assert alerts[0].drift >= alerts[1].drift  # sorted descending


def test_drift_alert_to_dict_keys(tmp_path):
    wf_path = tmp_path / "wf.csv"
    pd.DataFrame([
        {"agent": "BuffettAgent", "symbol": "AAPL", "avg_oos_sharpe": 5.0},
    ]).to_csv(wf_path, index=False)

    dec_rows = [_decisions(f"P{i*2-1}", "AAPL", "BuffettAgent") for i in range(1, 4)]
    exc_rows = [
        _execution("P1", "AAPL", "BUY",  100.0, "2024-01-01T10:00:00+00:00"),
        _execution("P2", "AAPL", "SELL",  90.0, "2024-01-15T10:00:00+00:00"),
        _execution("P3", "AAPL", "BUY",   90.0, "2024-02-01T10:00:00+00:00"),
        _execution("P4", "AAPL", "SELL",  81.0, "2024-02-15T10:00:00+00:00"),
        _execution("P5", "AAPL", "BUY",   81.0, "2024-03-01T10:00:00+00:00"),
        _execution("P6", "AAPL", "SELL",  73.0, "2024-03-15T10:00:00+00:00"),
    ]
    dec_f = _write_csv(tmp_path, "dec.csv", dec_rows)
    exc_f = _write_csv(tmp_path, "exc.csv", exc_rows)
    cfg = LiveScorerConfig(decisions_path=str(dec_f), executions_path=str(exc_f), min_trades=3)
    scorer = LiveScorer(cfg)
    alerts = scorer.compute_drift_alerts(drift_threshold=0.5, wf_path=str(wf_path))
    assert len(alerts) == 1
    d = alerts[0].to_dict()
    assert set(d.keys()) == {"agent", "oos_sharpe", "live_sharpe", "drift"}
    assert d["agent"] == "BuffettAgent"
    assert d["drift"] > 0.5
