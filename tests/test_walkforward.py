# tests/test_walkforward.py
from __future__ import annotations

import pathlib

import numpy as np
import pandas as pd
import pytest

from src.backtest.engine import (
    BDAYS_PER_MONTH,
    WalkForwardEngine,
    WalkForwardResult,
    WindowResult,
)
from src.agents.dummy import DummyHoldAgent


# ─── Helpers ─────────────────────────────────────────────────────────────────

def _make_ohlcv(n: int, start_price: float = 100.0, trend: float = 0.0) -> pd.DataFrame:
    """Generate n days of synthetic OHLCV. trend > 0 → up, < 0 → down."""
    rng = np.random.default_rng(42)
    prices = start_price * np.cumprod(1 + trend / 252 + rng.normal(0, 0.01, n))
    idx = pd.date_range("2020-01-01", periods=n, freq="B")
    noise = prices * 0.005
    return pd.DataFrame(
        {
            "Open":   prices - noise,
            "High":   prices + noise * 2,
            "Low":    prices - noise * 2,
            "Close":  prices,
            "Volume": np.ones(n) * 1_000_000,
        },
        index=idx,
    )


AGENT = DummyHoldAgent()
# Minimum rows for one window
MIN_ROWS = WalkForwardEngine.TRAIN_BDAYS + WalkForwardEngine.TEST_BDAYS  # 504


# ─── Window count ─────────────────────────────────────────────────────────────

def test_no_windows_when_data_too_short():
    df = _make_ohlcv(MIN_ROWS - 1)
    engine = WalkForwardEngine(AGENT)
    result = engine.run("TEST", df)
    assert len(result.windows) == 0


def test_one_window_at_exact_minimum():
    df = _make_ohlcv(MIN_ROWS)
    engine = WalkForwardEngine(AGENT)
    result = engine.run("TEST", df)
    assert len(result.windows) == 1


def test_correct_window_count_for_large_dataset():
    # 3y ≈ 756 bdays → expected windows = floor((756 - 504) / 63) + 1 = 5
    n = 3 * 252
    df = _make_ohlcv(n)
    engine = WalkForwardEngine(AGENT)
    result = engine.run("TEST", df)
    expected = (n - MIN_ROWS) // WalkForwardEngine.STEP_BDAYS + 1
    assert len(result.windows) == expected


# ─── Date ranges (no lookahead) ──────────────────────────────────────────────

def test_test_start_does_not_precede_train_end():
    df = _make_ohlcv(MIN_ROWS + WalkForwardEngine.STEP_BDAYS * 2)
    engine = WalkForwardEngine(AGENT)
    result = engine.run("TEST", df)
    for w in result.windows:
        assert w.test_start > w.train_end, (
            f"Window {w.window_idx}: test_start={w.test_start} <= train_end={w.train_end}"
        )


def test_windows_are_chronologically_ordered():
    """
    Walk-forward windows overlap by design (step < train+test).
    The invariant is that each window's test_start is strictly after the previous one.
    """
    n = 3 * 252
    df = _make_ohlcv(n)
    result = WalkForwardEngine(AGENT).run("TEST", df)
    for i in range(len(result.windows) - 1):
        assert result.windows[i].test_start < result.windows[i + 1].test_start


def test_window_idx_sequential():
    df = _make_ohlcv(3 * 252)
    result = WalkForwardEngine(AGENT).run("TEST", df)
    for i, w in enumerate(result.windows):
        assert w.window_idx == i


# ─── Metrics shape ───────────────────────────────────────────────────────────

def test_window_result_fields_are_finite():
    df = _make_ohlcv(MIN_ROWS)
    result = WalkForwardEngine(AGENT).run("TEST", df)
    w = result.windows[0]
    assert isinstance(w.is_sharpe, float)
    assert isinstance(w.oos_sharpe, float)
    assert isinstance(w.oos_return, float)
    assert isinstance(w.oos_max_drawdown, float)
    assert w.oos_max_drawdown <= 0.0  # must be ≤ 0


def test_dummy_agent_zero_trades():
    df = _make_ohlcv(MIN_ROWS)
    result = WalkForwardEngine(AGENT).run("TEST", df)
    # DummyHoldAgent never trades
    assert result.windows[0].oos_n_trades == 0
    assert result.windows[0].oos_win_rate == 0.0


def test_benchmark_return_computed():
    """Benchmark return should be nonzero when asset has a strong directional trend."""
    # Use a very strong upward trend (2.0 = +200%/yr) to dominate noise
    df = _make_ohlcv(MIN_ROWS, trend=2.0)
    result = WalkForwardEngine(AGENT).run("TEST", df)
    assert result.windows[0].benchmark_return != 0.0  # something was computed
    assert isinstance(result.windows[0].benchmark_return, float)


def test_alpha_equals_oos_minus_benchmark():
    df = _make_ohlcv(MIN_ROWS)
    result = WalkForwardEngine(AGENT).run("TEST", df)
    w = result.windows[0]
    assert abs(w.alpha - (w.oos_return - w.benchmark_return)) < 1e-9


# ─── Aggregates ──────────────────────────────────────────────────────────────

def test_avg_oos_sharpe_is_mean_of_windows():
    df = _make_ohlcv(3 * 252)
    result = WalkForwardEngine(AGENT).run("TEST", df)
    expected = float(np.mean([w.oos_sharpe for w in result.windows]))
    assert abs(result.avg_oos_sharpe - expected) < 1e-9


def test_no_lookahead_warning_for_hold_agent():
    # DummyHoldAgent has IS Sharpe ≈ 0 → no warning
    df = _make_ohlcv(3 * 252)
    result = WalkForwardEngine(AGENT).run("TEST", df)
    # IS Sharpe of a flat equity curve is ~0 → not > 1.5 → no warning
    assert result.lookahead_warning is False


# ─── CSV output ──────────────────────────────────────────────────────────────

def test_save_csv_creates_file(tmp_path):
    df = _make_ohlcv(MIN_ROWS)
    result = WalkForwardEngine(AGENT).run("TEST", df)
    path = str(tmp_path / "wf_test.csv")
    WalkForwardEngine.save_csv([result], path)
    assert pathlib.Path(path).exists()
    loaded = pd.read_csv(path)
    assert len(loaded) == len(result.windows)


def test_save_csv_columns(tmp_path):
    df = _make_ohlcv(MIN_ROWS)
    result = WalkForwardEngine(AGENT).run("TEST", df)
    path = str(tmp_path / "wf_test.csv")
    WalkForwardEngine.save_csv([result], path)
    loaded = pd.read_csv(path)
    expected_cols = {
        "agent", "symbol", "window",
        "train_start", "train_end", "test_start", "test_end",
        "is_sharpe", "is_return",
        "oos_sharpe", "oos_return", "oos_max_drawdown",
        "oos_n_trades", "oos_win_rate",
        "benchmark_return", "alpha",
        "avg_oos_sharpe", "avg_is_sharpe", "lookahead_warning",
    }
    assert expected_cols.issubset(set(loaded.columns))


def test_save_csv_multiple_results(tmp_path):
    df = _make_ohlcv(MIN_ROWS)
    r1 = WalkForwardEngine(AGENT).run("AAPL", df)
    r2 = WalkForwardEngine(AGENT).run("SPY", df)
    path = str(tmp_path / "wf_multi.csv")
    WalkForwardEngine.save_csv([r1, r2], path)
    loaded = pd.read_csv(path)
    assert set(loaded["symbol"].unique()) == {"AAPL", "SPY"}


def test_save_csv_noop_when_no_windows(tmp_path):
    df = _make_ohlcv(MIN_ROWS - 1)
    result = WalkForwardEngine(AGENT).run("TEST", df)
    path = str(tmp_path / "wf_empty.csv")
    WalkForwardEngine.save_csv([result], path)  # no rows → file not created
    assert not pathlib.Path(path).exists()
