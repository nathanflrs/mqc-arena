# tests/test_walkforward.py
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.agents.dummy import DummyHoldAgent
from src.backtest.engine import WalkForwardEngine, BDAYS_PER_MONTH


# ─── Helpers ──────────────────────────────────────────────────────────────────

def _make_df(n: int, trend: float = 0.0003) -> pd.DataFrame:
    rng = np.random.default_rng(42)
    idx = pd.date_range("2020-01-01", periods=n, freq="B")
    close = 100.0 * np.cumprod(1 + trend + rng.normal(0, 0.01, n))
    noise = np.abs(close * 0.005)
    return pd.DataFrame(
        {
            "Open":   close - noise,
            "High":   close + noise * 2,
            "Low":    close - noise * 2,
            "Close":  close,
            "Volume": np.full(n, 1_000_000.0),
        },
        index=idx,
    )


MIN_ROWS = WalkForwardEngine.TRAIN_BDAYS + WalkForwardEngine.TEST_BDAYS  # 504


@pytest.fixture
def agent():
    return DummyHoldAgent()


# ─── Window count ─────────────────────────────────────────────────────────────

def test_one_window_exact_minimum(agent):
    result = WalkForwardEngine(agent=agent).run("T", _make_df(MIN_ROWS))
    assert len(result.windows) == 1


def test_five_windows_756_days(agent):
    result = WalkForwardEngine(agent=agent).run("T", _make_df(756))
    assert len(result.windows) == 5


def test_no_windows_if_too_short(agent):
    result = WalkForwardEngine(agent=agent).run("T", _make_df(MIN_ROWS - 1))
    assert len(result.windows) == 0
    assert result.avg_oos_sharpe == 0.0
    assert not result.lookahead_warning


# ─── Date boundaries (no lookahead) ──────────────────────────────────────────

def test_test_start_after_train_end(agent):
    result = WalkForwardEngine(agent=agent).run("T", _make_df(756))
    for w in result.windows:
        assert w.test_start > w.train_end


def test_test_periods_non_overlapping(agent):
    result = WalkForwardEngine(agent=agent).run("T", _make_df(756))
    for i in range(len(result.windows) - 1):
        assert result.windows[i + 1].test_start > result.windows[i].test_start


# ─── Metrics ─────────────────────────────────────────────────────────────────

def test_alpha_oos_minus_benchmark(agent):
    result = WalkForwardEngine(agent=agent).run("T", _make_df(756))
    for w in result.windows:
        assert w.alpha == pytest.approx(w.oos_return - w.benchmark_return, abs=1e-6)


def test_dummy_agent_zero_trades(agent):
    result = WalkForwardEngine(agent=agent).run("T", _make_df(MIN_ROWS))
    assert result.windows[0].oos_n_trades == 0


def test_avg_oos_sharpe_is_mean_of_windows(agent):
    result = WalkForwardEngine(agent=agent).run("T", _make_df(756))
    expected = float(np.mean([w.oos_sharpe for w in result.windows]))
    assert result.avg_oos_sharpe == pytest.approx(expected, abs=1e-6)


def test_agent_and_symbol_in_result(agent):
    result = WalkForwardEngine(agent=agent).run("AAPL", _make_df(MIN_ROWS))
    assert result.agent_name == "DummyHoldAgent"
    assert result.symbol == "AAPL"


def test_no_lookahead_warning_for_hold_agent(agent):
    # IS Sharpe ≈ 0 → no false positive
    result = WalkForwardEngine(agent=agent).run("T", _make_df(756))
    assert not result.lookahead_warning


# ─── Custom benchmark ────────────────────────────────────────────────────────

def test_custom_benchmark_changes_benchmark_return(agent):
    df = _make_df(MIN_ROWS, trend=0.0003)
    bench = _make_df(MIN_ROWS, trend=0.001)
    r_no  = WalkForwardEngine(agent=agent).run("T", df)
    r_yes = WalkForwardEngine(agent=agent).run("T", df, benchmark_df=bench)
    assert r_no.windows[0].benchmark_return != r_yes.windows[0].benchmark_return


# ─── CSV output ───────────────────────────────────────────────────────────────

def test_to_csv_rows_columns(agent):
    result = WalkForwardEngine(agent=agent).run("T", _make_df(756))
    rows = result.to_csv_rows()
    assert len(rows) == len(result.windows)
    required = {
        "agent", "symbol", "window", "train_start", "train_end",
        "test_start", "test_end", "is_sharpe", "oos_sharpe",
        "benchmark_return", "alpha", "lookahead_warning",
    }
    assert required.issubset(rows[0].keys())


def test_save_csv(agent, tmp_path):
    result = WalkForwardEngine(agent=agent).run("T", _make_df(MIN_ROWS))
    path = str(tmp_path / "wf.csv")
    WalkForwardEngine.save_csv([result], path=path)
    df = pd.read_csv(path)
    assert len(df) == 1
    assert "oos_sharpe" in df.columns
