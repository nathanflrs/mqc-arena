# tests/test_correlation.py
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.risk.correlation import CorrelationGuard, CorrelationCheckResult
from src.execution.planner import OrderPlan
from src.broker.portfolio import PortfolioSnapshot


# ─── Helpers ─────────────────────────────────────────────────────────────────

def _make_price_df(returns: np.ndarray, start_price: float = 100.0) -> pd.DataFrame:
    prices = start_price * np.cumprod(1 + returns)
    idx = pd.date_range("2024-01-01", periods=len(prices), freq="B")
    noise = prices * 0.002
    return pd.DataFrame({
        "Open":   prices - noise,
        "High":   prices + noise,
        "Low":    prices - noise,
        "Close":  prices,
        "Volume": np.ones(len(prices)) * 1_000_000,
    }, index=idx)


def _make_plan(
    symbol: str,
    action: str = "BUY",
    notional: float = 10_000.0,
    last_price: float = 100.0,
    current_qty: float = 0.0,
) -> OrderPlan:
    target_qty = notional / last_price
    return OrderPlan(
        symbol=symbol,
        action=action,
        target_weight=0.10,
        last_price=last_price,
        current_qty=current_qty,
        target_qty=target_qty,
        delta_qty=target_qty - current_qty,
        est_notional=notional,
        reason="test",
    )


def _snap(positions: dict | None = None) -> PortfolioSnapshot:
    return PortfolioSnapshot(net_liquidation=100_000.0, cash=80_000.0, positions=positions or {})


# ─── check_buy ────────────────────────────────────────────────────────────────

def test_no_open_positions_always_allowed():
    guard = CorrelationGuard(threshold=0.7)
    result = guard.check_buy("AAPL", [], {})
    assert result.allowed is True
    assert result.max_correlation == 0.0
    assert result.correlated_with is None


def test_perfectly_correlated_blocked():
    rng = np.random.default_rng(0)
    base_returns = rng.normal(0, 0.01, 120)
    price_data = {
        "SPY":  _make_price_df(base_returns),
        "QQQ":  _make_price_df(base_returns),  # identical returns → r = 1.0
    }
    guard = CorrelationGuard(threshold=0.7)
    result = guard.check_buy("QQQ", ["SPY"], price_data)
    assert result.allowed is False
    assert result.max_correlation > 0.99
    assert result.correlated_with == "SPY"
    assert "bloqué" in result.reason


def test_uncorrelated_asset_allowed():
    rng = np.random.default_rng(1)
    returns_spy  = rng.normal(0, 0.01, 120)
    returns_gold = rng.normal(0, 0.008, 120)  # independent noise
    price_data = {
        "SPY": _make_price_df(returns_spy),
        "GLD": _make_price_df(returns_gold),
    }
    guard = CorrelationGuard(threshold=0.7)
    result = guard.check_buy("GLD", ["SPY"], price_data)
    assert result.allowed is True


def test_negatively_correlated_blocked():
    """Inverse correlation (r ≈ -1) should also be blocked (abs value)."""
    rng = np.random.default_rng(2)
    returns = rng.normal(0, 0.01, 120)
    price_data = {
        "BULL": _make_price_df(returns),
        "BEAR": _make_price_df(-returns),   # perfectly anticorrelated
    }
    guard = CorrelationGuard(threshold=0.7)
    result = guard.check_buy("BEAR", ["BULL"], price_data)
    assert result.allowed is False
    assert result.max_correlation > 0.99


def test_missing_candidate_data_allows():
    guard = CorrelationGuard(threshold=0.7)
    result = guard.check_buy("MISSING", ["SPY"], {"SPY": _make_price_df(np.zeros(120))})
    assert result.allowed is True


def test_insufficient_data_length_allows():
    rng = np.random.default_rng(3)
    returns = rng.normal(0, 0.01, 10)  # < min_overlap=20
    price_data = {
        "SPY":  _make_price_df(returns),
        "AAPL": _make_price_df(returns),
    }
    guard = CorrelationGuard(threshold=0.7, min_overlap=20)
    result = guard.check_buy("AAPL", ["SPY"], price_data)
    assert result.allowed is True


# ─── filter_plans ─────────────────────────────────────────────────────────────

def test_filter_passes_sell_always():
    rng = np.random.default_rng(4)
    base = rng.normal(0, 0.01, 120)
    price_data = {
        "SPY":  _make_price_df(base),
        "AAPL": _make_price_df(base),
    }
    guard = CorrelationGuard(threshold=0.7)
    snap = _snap(positions={"SPY": 10})
    plans = [
        _make_plan("AAPL", action="SELL"),  # SELL: always pass
    ]
    approved, blocked = guard.filter_plans(plans, snap, price_data)
    assert len(approved) == 1
    assert len(blocked) == 0


def test_filter_blocks_correlated_buy():
    rng = np.random.default_rng(5)
    base = rng.normal(0, 0.01, 120)
    price_data = {
        "SPY":  _make_price_df(base),
        "QQQ":  _make_price_df(base),  # r = 1.0 with SPY
    }
    guard = CorrelationGuard(threshold=0.7)
    snap = _snap(positions={"SPY": 10})
    plans = [_make_plan("QQQ", action="BUY")]
    approved, blocked = guard.filter_plans(plans, snap, price_data)
    assert len(approved) == 0
    assert len(blocked) == 1
    assert blocked[0]["symbol"] == "QQQ"


def test_filter_allows_uncorrelated_buy():
    rng = np.random.default_rng(6)
    price_data = {
        "SPY": _make_price_df(rng.normal(0, 0.01, 120)),
        "GLD": _make_price_df(rng.normal(0, 0.008, 120)),
    }
    guard = CorrelationGuard(threshold=0.7)
    snap = _snap(positions={"SPY": 10})
    plans = [_make_plan("GLD", action="BUY")]
    approved, blocked = guard.filter_plans(plans, snap, price_data)
    assert len(approved) == 1
    assert len(blocked) == 0


def test_filter_sequential_block_accumulates_open():
    """
    When two BUYs are approved in sequence, the second sees the first in open_symbols.
    """
    rng = np.random.default_rng(7)
    base = rng.normal(0, 0.01, 120)
    indep = rng.normal(0, 0.008, 120)
    price_data = {
        "A": _make_price_df(base),
        "B": _make_price_df(indep),     # uncorrelated with A → passes
        "C": _make_price_df(indep),     # same as B → correlated with B after B is approved
    }
    guard = CorrelationGuard(threshold=0.7)
    snap = _snap(positions={"A": 10})
    plans = [
        _make_plan("B", action="BUY"),  # passes (uncorrelated with A)
        _make_plan("C", action="BUY"),  # blocked (correlated with B, now in open_symbols)
    ]
    approved, blocked = guard.filter_plans(plans, snap, price_data)
    assert any(p.symbol == "B" for p in approved)
    assert any(b["symbol"] == "C" for b in blocked)


def test_filter_empty_portfolio():
    rng = np.random.default_rng(8)
    price_data = {"AAPL": _make_price_df(rng.normal(0, 0.01, 120))}
    guard = CorrelationGuard(threshold=0.7)
    snap = _snap(positions={})  # no open positions
    plans = [_make_plan("AAPL", action="BUY")]
    approved, blocked = guard.filter_plans(plans, snap, price_data)
    assert len(approved) == 1
    assert len(blocked) == 0


# ─── correlation_matrix ───────────────────────────────────────────────────────

def test_correlation_matrix_shape():
    rng = np.random.default_rng(9)
    price_data = {
        "A": _make_price_df(rng.normal(0, 0.01, 120)),
        "B": _make_price_df(rng.normal(0, 0.01, 120)),
        "C": _make_price_df(rng.normal(0, 0.01, 120)),
    }
    guard = CorrelationGuard()
    corr = guard.correlation_matrix(["A", "B", "C"], price_data)
    assert corr.shape == (3, 3)
    assert list(corr.columns) == ["A", "B", "C"]


def test_correlation_matrix_diagonal_ones():
    rng = np.random.default_rng(10)
    price_data = {
        "A": _make_price_df(rng.normal(0, 0.01, 120)),
        "B": _make_price_df(rng.normal(0, 0.01, 120)),
    }
    guard = CorrelationGuard()
    corr = guard.correlation_matrix(["A", "B"], price_data)
    assert corr.loc["A", "A"] == pytest.approx(1.0)
    assert corr.loc["B", "B"] == pytest.approx(1.0)


def test_correlation_matrix_empty_when_insufficient_symbols():
    guard = CorrelationGuard()
    corr = guard.correlation_matrix(["ONLY"], {"ONLY": _make_price_df(np.zeros(120))})
    assert corr.empty
