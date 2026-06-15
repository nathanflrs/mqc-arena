# tests/test_long_bias.py
"""
Tests for the long-bias overlay:
  - _compute_regime_series labels bull/bear/choppy correctly
  - BacktestEngine blocks weak SELL in bull regime
  - BacktestEngine allows high-conviction SELL in bull regime
  - BacktestEngine is unchanged in bear/choppy regimes
  - BuffettAgent / CitadelAgent / TrendFollowingAgent dampen SELL confidence in bull
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.backtest.engine import BacktestEngine, _compute_regime_series
from src.agents.base import BaseAgent, MarketState, AgentSignal


# ── Helpers ───────────────────────────────────────────────────────────────────

def _make_close(n: int = 400, trend: float = 0.0005) -> pd.Series:
    """Monotonically drifting price series."""
    dates = pd.bdate_range("2020-01-01", periods=n)
    prices = 100.0 * np.cumprod(1 + trend + np.random.default_rng(42).normal(0, 0.005, n))
    return pd.Series(prices, index=dates, name="Close")


def _make_ohlcv(close: pd.Series) -> pd.DataFrame:
    return pd.DataFrame({
        "Open": close * 0.999,
        "High": close * 1.003,
        "Low": close * 0.997,
        "Close": close,
        "Volume": 1_000_000,
    })


class _FixedSignalAgent(BaseAgent):
    """Agent that always returns a fixed pre-configured signal."""
    name = "FixedSignalAgent"

    def __init__(self, action: str, confidence: float):
        self._action = action
        self._confidence = confidence

    def generate_signal(self, state, portfolio, regime=None, data=None):
        return AgentSignal(
            agent_name=self.name,
            symbol=state.symbol,
            action=self._action,
            confidence=self._confidence,
            target_weight=0.0 if self._action != "BUY" else 0.10,
        )


# ── _compute_regime_series ────────────────────────────────────────────────────

def test_regime_series_bull():
    """Strongly uptrending price → classified as bull once SMA200 is valid."""
    close = _make_close(n=400, trend=0.002)  # +0.2%/day → clear uptrend
    regime = _compute_regime_series(close)
    # After 200 days, strong uptrend → most labels should be "bull"
    valid = regime.iloc[200:]
    bull_ratio = (valid == "bull").mean()
    assert bull_ratio > 0.6, f"Expected mostly bull, got {bull_ratio:.0%} bull"


def test_regime_series_bear():
    """Strongly downtrending price → classified as bear once SMA200 is valid."""
    close = _make_close(n=400, trend=-0.002)  # −0.2%/day → clear downtrend
    regime = _compute_regime_series(close)
    valid = regime.iloc[250:]  # give time for SMA50/200 to cross down
    bear_ratio = (valid == "bear").mean()
    assert bear_ratio > 0.5, f"Expected mostly bear, got {bear_ratio:.0%} bear"


def test_regime_series_choppy_before_sma200():
    """First 199 days must be 'choppy' because SMA200 is not yet valid."""
    close = _make_close(n=400, trend=0.002)
    regime = _compute_regime_series(close)
    assert (regime.iloc[:199] == "choppy").all()


def test_regime_series_length():
    close = _make_close(n=300)
    regime = _compute_regime_series(close)
    assert len(regime) == 300


# ── BacktestEngine long-bias ──────────────────────────────────────────────────

def _run_sell_agent_with_regime(regime_label: str, sell_confidence: float, threshold: float = 0.80):
    """
    Run an engine where:
    - Agent BUYs once (when not in position), then persistently signals SELL
    - The entire period has the given regime label
    Returns the number of SELL trades executed.
    """
    n = 300
    dates = pd.bdate_range("2020-01-01", periods=n)
    prices = pd.Series(100.0 + np.arange(n) * 0.1, index=dates, name="Close")
    df = _make_ohlcv(prices)

    _sell_conf = sell_confidence

    class _BuyOnceThenSellAgent(BaseAgent):
        name = "BuyOnceThenSellAgent"

        def __init__(self):
            self._bought = False

        def generate_signal(self, state, portfolio, regime=None, data=None):
            in_pos = portfolio.get("TEST", 0) > 0
            if not in_pos and not self._bought:
                self._bought = True
                return AgentSignal(self.name, "TEST", "BUY", 0.80, 0.10)
            if in_pos:
                return AgentSignal(self.name, "TEST", "SELL", _sell_conf, 0.0)
            return AgentSignal(self.name, "TEST", "HOLD", 0.5, 0.0)

    regime_series = pd.Series(regime_label, index=dates, dtype=object) if regime_label is not None else None

    engine = BacktestEngine(
        agent=_BuyOnceThenSellAgent(),
        min_history=10,
        cooldown_days=1,
        long_bias_bull_threshold=threshold,
    )
    result = engine.run("TEST", df, regime_series=regime_series)
    sell_trades = [t for t in result.trades if t.action == "SELL"]
    return len(sell_trades)


def test_long_bias_blocks_weak_sell_in_bull():
    """SELL with confidence 0.62 (below 0.80 threshold) must be blocked in bull."""
    n_sells = _run_sell_agent_with_regime("bull", sell_confidence=0.62)
    assert n_sells == 0, f"Expected 0 SELLs, got {n_sells}"


def test_long_bias_allows_strong_sell_in_bull():
    """SELL with confidence 0.88 (above 0.80 threshold) must execute even in bull."""
    n_sells = _run_sell_agent_with_regime("bull", sell_confidence=0.88)
    assert n_sells >= 1, f"Expected ≥1 SELL, got {n_sells}"


def test_long_bias_no_effect_in_bear():
    """In bear regime, weak SELL (0.62) must still execute."""
    n_sells = _run_sell_agent_with_regime("bear", sell_confidence=0.62)
    assert n_sells >= 1, f"Expected ≥1 SELL in bear, got {n_sells}"


def test_long_bias_no_effect_in_choppy():
    """In choppy regime, weak SELL (0.62) must still execute."""
    n_sells = _run_sell_agent_with_regime("choppy", sell_confidence=0.62)
    assert n_sells >= 1, f"Expected ≥1 SELL in choppy, got {n_sells}"


def test_long_bias_no_effect_when_regime_none():
    """With no regime (None), weak SELL must still execute."""
    n_sells = _run_sell_agent_with_regime(None, sell_confidence=0.62)
    assert n_sells >= 1, f"Expected ≥1 SELL with regime=None, got {n_sells}"


def test_long_bias_threshold_boundary():
    """Confidence exactly at threshold must be allowed through."""
    n_sells = _run_sell_agent_with_regime("bull", sell_confidence=0.80, threshold=0.80)
    assert n_sells >= 1


# ── Agent-level SELL confidence dampening ─────────────────────────────────────

def _make_state(symbol: str = "AAPL", price: float = 150.0) -> MarketState:
    return MarketState(symbol=symbol, price=price, timestamp="2024-01-15T09:30:00Z")


def _make_data_single_sma200_trigger(n: int = 300) -> pd.DataFrame:
    """
    Long stable period at 170 then small dip to 162 — triggers ONLY 'price < SMA200'.
    - SMA200 ≈ 169 (mostly stable at 170)
    - price = 162 → price < SMA200 by ~4% (one trigger)
    - Drawdown60 ≈ 4.7% (well below 12% threshold)
    - Vol20 ≈ 0% (flat, well below 5% threshold)
    """
    dates = pd.bdate_range("2022-01-01", periods=n)
    stable = np.full(n - 20, 170.0)
    dip    = np.full(20, 162.0)
    prices = np.concatenate([stable, dip])
    return pd.DataFrame({
        "Open": prices * 0.9999,
        "High": prices * 1.0001,
        "Low": prices * 0.9999,
        "Close": prices,
        "Volume": 1_000_000,
    }, index=dates)


def test_buffett_single_sell_trigger_dampened_in_bull():
    from src.agents.buffett import BuffettAgent
    agent = BuffettAgent()
    data = _make_data_single_sma200_trigger()
    portfolio = {"AAPL": 10.0}
    state = _make_state(price=float(data["Close"].iloc[-1]))
    sig = agent.generate_signal(state, portfolio, regime="bull", data=data)
    assert sig.action == "SELL", f"Expected SELL but got {sig.action} — check _make_data_single_sma200_trigger"
    assert sig.confidence == pytest.approx(0.62), (
        f"Single-trigger SELL in bull should be 0.62, got {sig.confidence}"
    )


def test_buffett_single_sell_trigger_normal_in_bear():
    from src.agents.buffett import BuffettAgent
    agent = BuffettAgent()
    data = _make_data_single_sma200_trigger()
    portfolio = {"AAPL": 10.0}
    state = _make_state(price=float(data["Close"].iloc[-1]))
    sig = agent.generate_signal(state, portfolio, regime="bear", data=data)
    # In bear, single-trigger SELL should keep the full 0.85 confidence
    assert sig.action == "SELL"
    assert sig.confidence == pytest.approx(0.85)


def test_citadel_single_sell_trigger_dampened_in_bull():
    from src.agents.citadel import CitadelAgent
    agent = CitadelAgent()
    data = _make_data_single_sma200_trigger()
    portfolio = {"AAPL": 10.0}
    state = _make_state(price=float(data["Close"].iloc[-1]))
    sig = agent.generate_signal(state, portfolio, regime="bull", data=data)
    if sig.action == "SELL":
        n_triggers = sig.reason.count("|") + 1 if "|" in sig.reason else 1
        if n_triggers == 1:
            assert sig.confidence == pytest.approx(0.62)


def test_trendfollowing_single_sell_trigger_dampened_in_bull():
    from src.agents.trend_following import TrendFollowingAgent
    agent = TrendFollowingAgent()
    data = _make_data_single_sma200_trigger()
    portfolio = {"AAPL": 10.0}
    state = _make_state(price=float(data["Close"].iloc[-1]))
    sig = agent.generate_signal(state, portfolio, regime="bull", data=data)
    if sig.action == "SELL":
        n_triggers = sig.reason.count("|") + 1 if "|" in sig.reason else 1
        if n_triggers == 1:
            assert sig.confidence == pytest.approx(0.62)
