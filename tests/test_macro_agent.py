"""
Tests for the refactored MacroAgent (B + A combined):
  - Ticker-specific momentum required to confirm macro signal
  - target_weight reduced to 0.05
"""
from __future__ import annotations

import pandas as pd
import numpy as np
from unittest.mock import patch, MagicMock

import pytest

from src.agents.base import MarketState
from src.agents.macro import MacroAgent, MacroConfig


# ── helpers ───────────────────────────────────────────────────────────────────

def _state(symbol: str = "AAPL", price: float = 200.0) -> MarketState:
    return MarketState(symbol=symbol, price=price, timestamp="2026-07-12T09:30:00Z")


def _ohlcv(n: int = 130, trend: float = 0.20) -> pd.DataFrame:
    """Return a simple OHLCV DataFrame with `trend` total return over n bars."""
    prices = np.linspace(100.0, 100.0 * (1 + trend), n)
    return pd.DataFrame({"Close": prices, "Open": prices, "High": prices, "Low": prices, "Volume": [1e6] * n})


def _flat_ohlcv(n: int = 130) -> pd.DataFrame:
    return _ohlcv(n, trend=0.00)


def _weak_ohlcv(n: int = 130) -> pd.DataFrame:
    return _ohlcv(n, trend=-0.10)


# Macro proxy momentum values for a clear RISK ON environment
# SPY +0.13 → +2, GLD -0.07 (< 0) → +1, TLT -0.05 (< -tlt_threshold 0.03) → +1 = 4
_RISK_ON_MOMS  = {"SPY": 0.13, "GLD": -0.07, "TLT": -0.05}  # risk_on_score=4
_RISK_OFF_MOMS = {"SPY": -0.08, "GLD": 0.08, "TLT": 0.06}   # risk_off_score=4


def _mock_macro_download(moms: dict):
    """Patch _momentum so macros return the given values."""
    def side_effect(symbol, period=63):
        return moms.get(symbol, 0.0)
    return patch.object(MacroAgent, "_momentum", side_effect=side_effect)


# ── Target weight ─────────────────────────────────────────────────────────────

def test_target_weight_is_005():
    assert MacroConfig().target_weight == 0.05


# ── BUY requires ticker confirmation ─────────────────────────────────────────

def test_buy_when_risk_on_and_ticker_strong():
    """RISK ON + ticker >2% → BUY."""
    data = _ohlcv(trend=0.15)   # ticker momentum ≈ +15% over 63d
    with _mock_macro_download(_RISK_ON_MOMS):
        sig = MacroAgent().generate_signal(_state(), {}, regime="bull", data=data)
    assert sig.action == "BUY"
    assert sig.target_weight == pytest.approx(0.05)
    assert "RISK ON" in sig.reason
    assert "ticker" in sig.reason


def test_no_buy_when_risk_on_but_ticker_flat():
    """RISK ON globally, but ticker flat (<2%) → HOLD, not BUY."""
    data = _flat_ohlcv()
    with _mock_macro_download(_RISK_ON_MOMS):
        sig = MacroAgent().generate_signal(_state(), {}, regime="bull", data=data)
    assert sig.action == "HOLD"


def test_no_buy_when_risk_on_but_ticker_slightly_positive():
    """Ticker at +1.5% is below the 2% threshold → HOLD."""
    data = _ohlcv(trend=0.015)
    with _mock_macro_download(_RISK_ON_MOMS):
        sig = MacroAgent().generate_signal(_state(), {}, regime="bull", data=data)
    assert sig.action == "HOLD"


def test_no_buy_in_bear_regime_even_with_strong_ticker():
    """Bear regime blocks BUY regardless of ticker momentum."""
    data = _ohlcv(trend=0.20)
    with _mock_macro_download(_RISK_ON_MOMS):
        sig = MacroAgent().generate_signal(_state(), {}, regime="bear", data=data)
    assert sig.action == "HOLD"


# ── SELL requires ticker confirmation ─────────────────────────────────────────

def test_sell_when_risk_off_and_ticker_weak():
    """RISK OFF + open position + ticker < -2% → SELL."""
    data = _weak_ohlcv()  # ≈ -10% momentum
    with _mock_macro_download(_RISK_OFF_MOMS):
        sig = MacroAgent().generate_signal(
            _state(), portfolio={"AAPL": 100.0}, regime="bear", data=data
        )
    assert sig.action == "SELL"
    assert sig.confidence == pytest.approx(0.80)
    assert "RISK OFF" in sig.reason


def test_no_sell_when_risk_off_but_ticker_holding_up():
    """RISK OFF globally, but ticker still positive → HOLD (don't sell early)."""
    data = _ohlcv(trend=0.10)   # ticker still up
    with _mock_macro_download(_RISK_OFF_MOMS):
        sig = MacroAgent().generate_signal(
            _state(), portfolio={"AAPL": 100.0}, regime="bear", data=data
        )
    assert sig.action == "HOLD"


def test_no_sell_without_open_position():
    """SELL requires an open position — no position → HOLD even with RISK OFF + weak ticker."""
    data = _weak_ohlcv()
    with _mock_macro_download(_RISK_OFF_MOMS):
        sig = MacroAgent().generate_signal(_state(), portfolio={}, regime="bear", data=data)
    assert sig.action == "HOLD"


# ── Mixed macro → HOLD ────────────────────────────────────────────────────────

def test_hold_when_macro_mixed():
    """Mixed macro signals → HOLD regardless of ticker."""
    data = _ohlcv(trend=0.20)
    mixed_moms = {"SPY": 0.03, "GLD": 0.02, "TLT": 0.01}  # risk_on=0, risk_off=0
    with _mock_macro_download(mixed_moms):
        sig = MacroAgent().generate_signal(_state(), {}, regime="bull", data=data)
    assert sig.action == "HOLD"


# ── Meta ──────────────────────────────────────────────────────────────────────

def test_meta_includes_ticker_momentum():
    """mom_ticker must be included in signal meta for logging/audit."""
    data = _ohlcv(trend=0.15)
    with _mock_macro_download(_RISK_ON_MOMS):
        sig = MacroAgent().generate_signal(_state(), {}, regime="bull", data=data)
    assert "mom_ticker" in sig.meta
    assert sig.meta["mom_ticker"] != 0.0


def test_score_below_competitor_without_priority():
    """MacroAgent BUY score (0.05 × 0.60 = 0.03) is intentionally low — spec requirement."""
    from src.arena.selector import score_signal
    from src.agents.base import AgentSignal

    macro_sig = AgentSignal(
        agent_name="MacroAgent", symbol="AAPL",
        action="BUY", confidence=0.60, target_weight=0.05,
        reason="test", meta={},
    )
    buffett_sig = AgentSignal(
        agent_name="BuffettAgent", symbol="AAPL",
        action="BUY", confidence=0.90, target_weight=0.10,
        reason="test", meta={},
    )
    assert score_signal(macro_sig) < score_signal(buffett_sig)


# ── Fallback: no data passed ──────────────────────────────────────────────────

def test_no_crash_when_data_is_none():
    """If data=None, agent falls back to downloading. Mock the download."""
    with _mock_macro_download(_RISK_ON_MOMS):
        with patch.object(MacroAgent, "_ticker_momentum", return_value=0.15):
            sig = MacroAgent().generate_signal(_state(), {}, regime="bull", data=None)
    assert sig.action in ("BUY", "HOLD", "SELL")


def test_ticker_momentum_threshold_configurable():
    """Custom threshold respected.

    _ticker_momentum computes close[-1]/close[-64]-1 over 63 bars.
    With trend=0.035 over 130 bars the 63-bar window covers ≈half the range,
    giving 63d momentum ≈ 0.035 × 63/130 ≈ 1.7% — above 0.01 but below 0.02.
    """
    data = _ohlcv(trend=0.035)   # 63d momentum ≈ +1.7%
    cfg  = MacroConfig(ticker_momentum_threshold=0.01)
    with _mock_macro_download(_RISK_ON_MOMS):
        sig = MacroAgent(config=cfg).generate_signal(_state(), {}, regime="bull", data=data)
    assert sig.action == "BUY"


# ── FRED: score_fred_indicators ───────────────────────────────────────────────

from src.agents.macro import MacroIndicators, score_fred_indicators, FREDClient


def test_fred_score_risk_on_steep_curve_benign_spread_calm_vix():
    """Steep curve + benign HY + calm VIX → high risk_on score."""
    ind = MacroIndicators(yield_curve=1.5, hy_spread=250.0, vix=13.0, fed_trend=-0.6)
    risk_on, risk_off = score_fred_indicators(ind)
    # yield_curve >1 → +2, hy_spread <300 → +2, vix <15 → +1, fed easing → +1  = 6
    assert risk_on == 6
    assert risk_off == 0


def test_fred_score_risk_off_inverted_curve_stress_spread_high_vix():
    """Deep inversion + stressed HY + VIX panic → high risk_off score."""
    ind = MacroIndicators(yield_curve=-0.50, hy_spread=650.0, vix=38.0, fed_trend=1.0)
    risk_on, risk_off = score_fred_indicators(ind)
    # yield_curve <-0.25 → +2 off, hy_spread >600 → +2 off, vix >30 → +2 off, fed >0.75 → +1 off = 7
    assert risk_off == 7
    assert risk_on == 0


def test_fred_score_mild_risk_off():
    """Mild inversion + moderate HY + elevated VIX → moderate risk_off."""
    ind = MacroIndicators(yield_curve=-0.10, hy_spread=500.0, vix=25.0, fed_trend=0.3)
    risk_on, risk_off = score_fred_indicators(ind)
    # yield_curve -0.10 → +1 off, hy_spread 500 → +1 off (between 450-600), vix 25 → +1 off = 3
    assert risk_off == 3
    assert risk_on == 0


def test_fred_score_none_indicators_return_zero():
    """All None indicators → both scores 0 (no info, no signal)."""
    ind = MacroIndicators(yield_curve=None, hy_spread=None, vix=None, fed_trend=None)
    risk_on, risk_off = score_fred_indicators(ind)
    assert risk_on == 0
    assert risk_off == 0


def test_fred_score_partial_data():
    """Only yield curve available — other indicators missing → partial score only."""
    ind = MacroIndicators(yield_curve=1.2, hy_spread=None, vix=None, fed_trend=None)
    risk_on, risk_off = score_fred_indicators(ind)
    assert risk_on == 2   # yield_curve >1 → +2
    assert risk_off == 0


# ── FRED path: MacroAgent with mock FREDClient ────────────────────────────────

def _mock_fred_client(
    yield_curve: float | None = 1.5,
    hy_spread:   float | None = 250.0,
    vix:         float | None = 13.0,
    fed_series:  list | None = None,
    available:   bool = True,
) -> FREDClient:
    """Return a FREDClient double that returns pre-set values."""
    client = MagicMock(spec=FREDClient)
    client.available = available

    fed_series = fed_series or [5.25, 5.25, 5.0, 4.75, 4.5, 4.25, 4.0]

    def latest_side_effect(series_id, periods=1):
        mapping = {
            "T10Y2Y":        yield_curve,
            "BAMLH0A0HYM2":  hy_spread,
            "VIXCLS":        vix,
            "FEDFUNDS":      fed_series if periods > 1 else (fed_series[0] if fed_series else None),
        }
        return mapping.get(series_id)

    client.latest.side_effect = latest_side_effect
    return client


def test_fred_path_buy_risk_on_environment():
    """MacroAgent uses FRED path and generates BUY in risk-on environment."""
    fred = _mock_fred_client(
        yield_curve=1.5, hy_spread=250.0, vix=13.0,
        fed_series=[4.0, 4.25, 4.5, 4.75, 5.0, 5.25, 5.5],  # easing trend
    )
    data = _ohlcv(trend=0.15)  # strong ticker
    agent = MacroAgent(fred_client=fred)
    sig = agent.generate_signal(_state(), {}, regime="bull", data=data)
    assert sig.action == "BUY"
    assert sig.meta.get("source") == "fred"
    assert sig.meta.get("yield_curve") == pytest.approx(1.5)


def test_fred_path_sell_risk_off_environment():
    """MacroAgent uses FRED path and generates SELL in risk-off environment."""
    fred = _mock_fred_client(
        yield_curve=-0.50, hy_spread=650.0, vix=38.0,
        fed_series=[5.5, 5.25, 4.75, 4.5, 4.25, 4.0, 3.75],  # tightening trend
    )
    data = _weak_ohlcv()  # weak ticker
    agent = MacroAgent(fred_client=fred)
    sig = agent.generate_signal(_state(), portfolio={"AAPL": 100.0}, regime="bear", data=data)
    assert sig.action == "SELL"
    assert sig.meta.get("source") == "fred"


def test_fred_path_hold_when_mixed_signals():
    """MacroAgent uses FRED path and generates HOLD when indicators are mixed."""
    fred = _mock_fred_client(yield_curve=0.3, hy_spread=380.0, vix=22.0, fed_series=None)
    data = _ohlcv(trend=0.05)
    agent = MacroAgent(fred_client=fred)
    sig = agent.generate_signal(_state(), {}, regime="bull", data=data)
    assert sig.action == "HOLD"
    assert sig.meta.get("source") == "fred"


def test_fred_path_meta_contains_all_indicators():
    """BUY signal meta must contain all 4 FRED indicators."""
    fred = _mock_fred_client(
        yield_curve=1.5, hy_spread=250.0, vix=13.0,
        fed_series=[4.0, 4.25, 4.5, 4.75, 5.0, 5.25, 5.5],
    )
    data = _ohlcv(trend=0.15)
    sig = MacroAgent(fred_client=fred).generate_signal(_state(), {}, regime="bull", data=data)
    assert "yield_curve" in sig.meta
    assert "hy_spread"   in sig.meta
    assert "vix"         in sig.meta
    assert "fed_trend"   in sig.meta


def test_fred_path_bear_regime_blocks_buy():
    """bear regime blocks BUY even with FRED risk-on signal."""
    fred = _mock_fred_client(yield_curve=1.5, hy_spread=250.0, vix=13.0)
    data = _ohlcv(trend=0.15)
    sig = MacroAgent(fred_client=fred).generate_signal(_state(), {}, regime="bear", data=data)
    assert sig.action == "HOLD"


# ── Fallback: FRED unavailable → ETF proxy ────────────────────────────────────

def test_fallback_to_etf_proxy_when_fred_unavailable():
    """When FRED is unavailable (no key), ETF proxy is used instead."""
    fred = _mock_fred_client(available=False)
    data = _ohlcv(trend=0.15)
    with _mock_macro_download(_RISK_ON_MOMS):
        sig = MacroAgent(fred_client=fred).generate_signal(_state(), {}, regime="bull", data=data)
    assert sig.action == "BUY"
    assert sig.meta.get("source") == "etf_proxy"


def test_fallback_to_etf_proxy_when_fred_api_fails():
    """When FRED API call raises an exception, agent falls back to ETF proxy."""
    fred = _mock_fred_client()  # available=True but...
    fred.latest.side_effect = Exception("connection timeout")
    data = _ohlcv(trend=0.15)
    with _mock_macro_download(_RISK_ON_MOMS):
        sig = MacroAgent(fred_client=fred).generate_signal(_state(), {}, regime="bull", data=data)
    # Should not crash; ETF proxy takes over
    assert sig.action in ("BUY", "HOLD", "SELL")
    assert sig.meta.get("source") == "etf_proxy"


# ── FREDClient cache ──────────────────────────────────────────────────────────

def test_fredclient_no_key_returns_none(tmp_path):
    """FREDClient with empty key returns None without hitting the network."""
    client = FREDClient(api_key="")
    # Override cache dir to tmp_path so no real disk writes during CI
    import src.agents.macro as macro_mod
    original = macro_mod._CACHE_DIR
    macro_mod._CACHE_DIR = tmp_path
    try:
        result = client.latest("T10Y2Y")
        assert result is None
    finally:
        macro_mod._CACHE_DIR = original


def test_fredclient_available_property():
    assert FREDClient(api_key="").available is False
    assert FREDClient(api_key="somekey123").available is True


def test_fredclient_cache_write_and_read(tmp_path, monkeypatch):
    """Cache written by _save_cache is returned by _load_cache within TTL."""
    import src.agents.macro as macro_mod
    monkeypatch.setattr(macro_mod, "_CACHE_DIR", tmp_path)

    client = FREDClient(api_key="dummy")
    client._save_cache("T10Y2Y", 1.23)
    result = client._load_cache("T10Y2Y")
    assert result == pytest.approx(1.23)


def test_fredclient_cache_expired_returns_none(tmp_path, monkeypatch):
    """Expired cache (>4h) is not returned."""
    import src.agents.macro as macro_mod
    from datetime import datetime, timezone, timedelta
    monkeypatch.setattr(macro_mod, "_CACHE_DIR", tmp_path)

    client = FREDClient(api_key="dummy")
    # Write cache with a timestamp 5h ago
    cache_path = tmp_path / f"T10Y2Y_{datetime.now(timezone.utc).date().isoformat()}.json"
    import json
    cache_path.write_text(json.dumps({
        "cached_at": (datetime.now(timezone.utc) - timedelta(hours=5)).isoformat(),
        "value": 99.9,
    }))
    result = client._load_cache("T10Y2Y")
    assert result is None
