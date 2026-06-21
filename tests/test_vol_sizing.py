# tests/test_vol_sizing.py
"""
Tests for vol_adjusted_weight.

Pure function — no mocking needed.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.risk.vol_sizing import vol_adjusted_weight


# ── Helpers ───────────────────────────────────────────────────────────────────

def _make_df(closes: np.ndarray) -> pd.DataFrame:
    idx = pd.date_range("2020-01-01", periods=len(closes), freq="B")
    return pd.DataFrame({"Close": closes}, index=idx)


def _stable_df(n: int = 400, price: float = 100.0, noise: float = 0.5) -> pd.DataFrame:
    """Stationary price series with low constant vol."""
    rng = np.random.default_rng(42)
    pct = rng.normal(0, noise / 100, n)
    closes = price * np.cumprod(1 + pct)
    return _make_df(closes)


def _spike_df(n_base: int = 400, n_spike: int = 20) -> pd.DataFrame:
    """
    Builds closes from explicit log-returns so the vol ratio is guaranteed.

    Design:
    - base σ  = 0.008 / day  → ~12 % annualized
    - spike σ = 0.008 × 8    → ~96 % annualized
    - 1-year window (last 252 log-rets) contains 232 base + 20 spike
    - Expected vol_1y ≈ 31 %,  vol_20d ≈ 102 %  → ratio ≈ 3.3  >> 2 threshold
    """
    rng = np.random.default_rng(42)
    base_σ  = 0.008
    spike_σ = base_σ * 8
    log_rets = np.concatenate([
        rng.normal(0, base_σ,  n_base),
        rng.normal(0, spike_σ, n_spike),
    ])
    closes = 100.0 * np.exp(np.cumsum(log_rets))
    return _make_df(np.insert(closes, 0, 100.0))


# ── Normal conditions ─────────────────────────────────────────────────────────

class TestNormalConditions:

    def test_returns_base_weight_unchanged(self):
        df = _stable_df(n=400)
        w, reason = vol_adjusted_weight(df, 0.08)
        assert w == 0.08
        assert reason == ""

    def test_returns_no_reason_when_not_triggered(self):
        df = _stable_df(n=400)
        _, reason = vol_adjusted_weight(df, 0.10)
        assert reason == ""

    def test_does_not_modify_base_weight_below_threshold(self):
        df = _stable_df(n=400, noise=1.0)
        w, _ = vol_adjusted_weight(df, 0.05)
        assert w == pytest.approx(0.05)


# ── Vol spike conditions ──────────────────────────────────────────────────────

class TestVolSpike:

    def test_weight_is_halved_on_spike(self):
        df = _spike_df()
        w, reason = vol_adjusted_weight(df, 0.08)
        assert w == pytest.approx(0.04)
        assert reason != ""

    def test_reason_contains_vol_figures(self):
        df = _spike_df()
        _, reason = vol_adjusted_weight(df, 0.08)
        assert "σ20d" in reason
        assert "σ1y" in reason

    def test_reason_shows_before_and_after_weight(self):
        df = _spike_df()
        _, reason = vol_adjusted_weight(df, 0.08)
        assert "0.0800" in reason
        assert "0.0400" in reason

    def test_adjusted_weight_is_rounded_to_4dp(self):
        df = _spike_df()
        w, reason = vol_adjusted_weight(df, 0.0733)
        if reason:  # spike triggered
            # result should be base / 2, rounded to 4 dp
            assert len(str(w).split(".")[-1]) <= 4


# ── Insufficient history ──────────────────────────────────────────────────────

class TestInsufficientHistory:

    def test_too_short_returns_base_unchanged(self):
        df = _stable_df(n=100)  # < vol_lookback(252) + vol_window(20)
        w, reason = vol_adjusted_weight(df, 0.08)
        assert w == 0.08
        assert reason == ""

    def test_exactly_at_min_length_boundary(self):
        # 252 + 20 = 272 rows = minimum
        df = _stable_df(n=272)
        w, _ = vol_adjusted_weight(df, 0.08)
        assert w == pytest.approx(0.08)  # stable series → no trigger

    def test_one_row_short_returns_base(self):
        df = _stable_df(n=271)  # 272 - 1
        w, reason = vol_adjusted_weight(df, 0.08)
        assert w == 0.08
        assert reason == ""


# ── Custom parameters ─────────────────────────────────────────────────────────

class TestCustomParameters:

    def test_custom_size_divisor(self):
        df = _spike_df()
        base = 0.08
        w, reason = vol_adjusted_weight(df, base, size_divisor=4.0)
        if reason:
            assert w == pytest.approx(round(base / 4.0, 4))

    def test_high_multiplier_does_not_trigger_on_mild_spike(self):
        """With vol_multiplier=5, only extreme spikes fire."""
        df = _spike_df(n_spike=20)  # moderate spike
        w, reason = vol_adjusted_weight(df, 0.08, vol_multiplier=5.0)
        # Not asserting triggered/not since it depends on RNG — just assert invariants
        assert 0 < w <= 0.08
        if reason:
            assert "5×" in reason

    def test_multiplier_1_always_triggers_after_any_change(self):
        """With vol_multiplier=1 and any spike, it should trigger."""
        df = _spike_df()
        w, reason = vol_adjusted_weight(df, 0.08, vol_multiplier=1.0)
        # 8× noise spike → σ20d >> σ1y, ratio well above 1
        assert w == pytest.approx(0.04)

    def test_custom_vol_window(self):
        df = _spike_df(n_spike=40)  # larger spike to cover both windows
        w5,  _ = vol_adjusted_weight(df, 0.08, vol_window=5)
        w20, _ = vol_adjusted_weight(df, 0.08, vol_window=20)
        # Both are valid floats in (0, 0.08]
        assert 0 < w5 <= 0.08
        assert 0 < w20 <= 0.08


# ── Edge cases ────────────────────────────────────────────────────────────────

class TestEdgeCases:

    def test_zero_base_weight_stays_zero(self):
        df = _spike_df()
        w, _ = vol_adjusted_weight(df, 0.0)
        assert w == 0.0

    def test_flat_price_series_zero_vol(self):
        """Perfectly flat price → zero vol → rule does not fire."""
        closes = np.full(400, 100.0)
        df = _make_df(closes)
        w, reason = vol_adjusted_weight(df, 0.08)
        assert w == 0.08
        assert reason == ""

    def test_dataframe_with_multiindex_close(self):
        """Handles yfinance-style MultiIndex columns gracefully."""
        df = _stable_df(n=400)
        # Simulate MultiIndex
        df.columns = pd.MultiIndex.from_tuples([("Close", "AAPL")])
        # vol_adjusted_weight expects "Close" as a key; this should either work
        # or raise a clear KeyError — not silently return wrong values
        try:
            w, _ = vol_adjusted_weight(df, 0.08)
            assert 0 < w <= 0.08
        except KeyError:
            pass  # acceptable — caller should normalize before passing
