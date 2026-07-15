"""
Tests for HistoricalVaREngine (historical simulation VaR).

All tests use synthetic price series — no network access.
"""
from __future__ import annotations

import json
from datetime import date, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.risk.var_engine import HistoricalVaREngine, VaRConfig, VaRResult


# ── Helpers ───────────────────────────────────────────────────────────────────

def _price_series(
    n: int = 300,
    start: float = 100.0,
    daily_vol: float = 0.01,
    seed: int = 42,
) -> pd.Series:
    """Random-walk close prices with a DatetimeIndex."""
    rng = np.random.default_rng(seed)
    returns = rng.normal(0, daily_vol, n)
    prices  = start * np.cumprod(1 + returns)
    idx = pd.date_range(end=date.today(), periods=n, freq="B")
    return pd.Series(prices, index=idx, name="Close")


def _df(series: pd.Series) -> pd.DataFrame:
    return pd.DataFrame({"Close": series, "Open": series, "High": series,
                         "Low": series, "Volume": 1_000_000})


def _engine(lookback: int = 252, budget: float = 0.05) -> HistoricalVaREngine:
    return HistoricalVaREngine(VaRConfig(lookback_days=lookback, risk_budget_pct=budget))


# ── Basic computation ─────────────────────────────────────────────────────────

class TestVaRBasic:

    def test_returns_var_result(self):
        prices = _price_series()
        eng    = _engine()
        result = eng.compute({"AAPL": 50.0}, {"AAPL": _df(prices)}, net_liquidation=100_000)
        assert isinstance(result, VaRResult)

    def test_var_99_gt_var_95(self):
        """99% VaR must be >= 95% VaR (larger confidence = larger loss estimate)."""
        prices = _price_series()
        eng    = _engine()
        result = eng.compute({"AAPL": 50.0}, {"AAPL": _df(prices)}, net_liquidation=100_000)
        assert result.var_99_usd >= result.var_95_usd

    def test_cvar_99_gt_var_99(self):
        """CVaR (Expected Shortfall) must be >= VaR at same confidence."""
        prices = _price_series()
        result = _engine().compute({"AAPL": 50}, {"AAPL": _df(prices)}, net_liquidation=100_000)
        assert result.cvar_99_usd >= result.var_99_usd

    def test_var_positive(self):
        """VaR is always a positive number (potential loss)."""
        prices = _price_series(daily_vol=0.02)
        result = _engine().compute({"AAPL": 100}, {"AAPL": _df(prices)}, net_liquidation=100_000)
        assert result.var_95_usd >= 0
        assert result.var_99_usd >= 0

    def test_pct_consistent_with_usd(self):
        """var_*_pct == var_*_usd / portfolio_value."""
        prices = _price_series()
        result = _engine().compute({"AAPL": 50}, {"AAPL": _df(prices)}, net_liquidation=100_000)
        pv = result.portfolio_value
        assert result.var_95_pct == pytest.approx(result.var_95_usd / pv, rel=1e-4)
        assert result.var_99_pct == pytest.approx(result.var_99_usd / pv, rel=1e-4)

    def test_n_days_populated(self):
        prices = _price_series(n=300)
        result = _engine(lookback=252).compute(
            {"AAPL": 50}, {"AAPL": _df(prices)}, net_liquidation=100_000
        )
        assert result.n_days == 252

    def test_n_positions_populated(self):
        p1 = _price_series(seed=1)
        p2 = _price_series(seed=2)
        result = _engine().compute(
            {"AAPL": 50, "MSFT": 30},
            {"AAPL": _df(p1), "MSFT": _df(p2)},
            net_liquidation=100_000,
        )
        assert result.n_positions == 2

    def test_computed_at_is_iso8601(self):
        prices = _price_series()
        result = _engine().compute({"AAPL": 50}, {"AAPL": _df(prices)}, net_liquidation=100_000)
        from datetime import datetime
        dt = datetime.fromisoformat(result.computed_at)
        assert dt.tzname() == "UTC"


# ── Portfolio value scaling ───────────────────────────────────────────────────

class TestVaRScaling:

    def test_double_position_doubles_var_usd(self):
        """Doubling quantity doubles $ VaR (portfolio_value doubles)."""
        prices = _price_series()
        r1 = _engine().compute({"AAPL": 50},  {"AAPL": _df(prices)}, net_liquidation=100_000)
        r2 = _engine().compute({"AAPL": 100}, {"AAPL": _df(prices)}, net_liquidation=100_000)
        assert r2.var_99_usd == pytest.approx(r1.var_99_usd * 2, rel=1e-4)

    def test_two_uncorrelated_assets_diversification(self):
        """Portfolio of 2 uncorrelated assets has lower VaR % than each individually."""
        p1 = _price_series(seed=10, daily_vol=0.02)
        p2 = _price_series(seed=99, daily_vol=0.02)
        combined = _engine().compute(
            {"A": 50, "B": 50},
            {"A": _df(p1), "B": _df(p2)},
            net_liquidation=100_000,
        )
        single = _engine().compute(
            {"A": 100},
            {"A": _df(p1)},
            net_liquidation=100_000,
        )
        assert combined.var_99_pct < single.var_99_pct

    def test_portfolio_value_matches_position_notional(self):
        """portfolio_value = qty × last_price (sum of all positions)."""
        prices   = _price_series(start=200.0)
        last_px  = float(prices.iloc[-1])
        qty      = 25
        result   = _engine().compute({"AAPL": qty}, {"AAPL": _df(prices)}, net_liquidation=100_000)
        expected = qty * last_px
        assert result.portfolio_value == pytest.approx(expected, rel=1e-3)


# ── Edge cases — no result ────────────────────────────────────────────────────

class TestVaREdgeCases:

    def test_no_positions_returns_none(self):
        result = _engine().compute({}, {}, net_liquidation=100_000)
        assert result is None

    def test_zero_qty_skipped(self):
        prices = _price_series()
        result = _engine().compute(
            {"AAPL": 0, "MSFT": 0},
            {"AAPL": _df(prices), "MSFT": _df(prices)},
            net_liquidation=100_000,
        )
        assert result is None

    def test_negative_qty_skipped(self):
        """Short positions (qty < 0) are excluded from the long-only VaR."""
        prices = _price_series()
        result = _engine().compute({"AAPL": -10}, {"AAPL": _df(prices)}, net_liquidation=100_000)
        assert result is None

    def test_missing_data_returns_none(self):
        """Position exists but no OHLCV data → None."""
        result = _engine().compute({"AAPL": 50}, {}, net_liquidation=100_000)
        assert result is None

    def test_insufficient_history_returns_none(self):
        """Fewer than min_history_days of common data → None."""
        short_series = _price_series(n=10)
        cfg    = VaRConfig(lookback_days=252, min_history_days=21)
        result = HistoricalVaREngine(cfg).compute(
            {"AAPL": 50}, {"AAPL": _df(short_series)}, net_liquidation=100_000
        )
        assert result is None

    def test_missing_close_column_skipped(self):
        """DataFrame without 'Close' is silently skipped."""
        prices = _price_series()
        df_no_close = pd.DataFrame({"Open": prices.values}, index=prices.index)
        result = _engine().compute({"AAPL": 50}, {"AAPL": df_no_close}, net_liquidation=100_000)
        assert result is None

    def test_symbol_not_in_all_data_skipped(self):
        """Symbol in positions but not in all_data → excluded gracefully."""
        prices = _price_series()
        result = _engine().compute(
            {"AAPL": 50, "MISSING": 30},
            {"AAPL": _df(prices)},
            net_liquidation=100_000,
        )
        assert result is not None
        assert result.n_positions == 1


# ── Budget alert ──────────────────────────────────────────────────────────────

class TestVaRBudget:

    def test_exceeds_budget_high_vol(self):
        """Very high-vol position should exceed a tight 0.1% budget."""
        prices = _price_series(daily_vol=0.05)   # 5% daily vol, extreme
        result = _engine(budget=0.001).compute(
            {"AAPL": 100}, {"AAPL": _df(prices)}, net_liquidation=100_000
        )
        assert result is not None
        eng = HistoricalVaREngine(VaRConfig(risk_budget_pct=0.001))
        assert eng.exceeds_budget(result, net_liquidation=100_000)

    def test_does_not_exceed_budget_low_vol(self):
        """Very low-vol position should not exceed a 50% budget."""
        prices = _price_series(daily_vol=0.001)  # 0.1% daily vol
        result = _engine(budget=0.50).compute(
            {"AAPL": 10}, {"AAPL": _df(prices)}, net_liquidation=100_000
        )
        assert result is not None
        eng = HistoricalVaREngine(VaRConfig(risk_budget_pct=0.50))
        assert not eng.exceeds_budget(result, net_liquidation=100_000)

    def test_exceeds_budget_uses_netliq_not_position_value(self):
        """Budget check divides var_99_usd by netliq, not by portfolio_value."""
        prices  = _price_series(daily_vol=0.02)
        result  = _engine().compute(
            {"AAPL": 100}, {"AAPL": _df(prices)}, net_liquidation=200_000
        )
        assert result is not None
        eng_tight  = HistoricalVaREngine(VaRConfig(risk_budget_pct=0.00001))
        eng_loose  = HistoricalVaREngine(VaRConfig(risk_budget_pct=0.99))
        assert     eng_tight.exceeds_budget(result, net_liquidation=200_000)
        assert not eng_loose.exceeds_budget(result,  net_liquidation=200_000)

    def test_zero_netliq_never_exceeds(self):
        prices = _price_series()
        result = _engine().compute({"AAPL": 50}, {"AAPL": _df(prices)}, net_liquidation=100_000)
        eng = _engine()
        assert not eng.exceeds_budget(result, net_liquidation=0)


# ── Persistence ───────────────────────────────────────────────────────────────

class TestVaRPersistence:

    def test_writes_json_log(self, tmp_path, monkeypatch):
        import src.risk.var_engine as mod
        log_path = tmp_path / "var_latest.json"
        monkeypatch.setattr(mod, "_VAR_LOG", log_path)

        prices = _price_series()
        _engine().compute({"AAPL": 50}, {"AAPL": _df(prices)}, net_liquidation=100_000)

        assert log_path.exists()
        data = json.loads(log_path.read_text())
        assert "var_99_pct"  in data
        assert "var_95_usd"  in data
        assert "computed_at" in data

    def test_log_is_valid_json_after_compute(self, tmp_path, monkeypatch):
        import src.risk.var_engine as mod
        log_path = tmp_path / "var_latest.json"
        monkeypatch.setattr(mod, "_VAR_LOG", log_path)

        prices = _price_series()
        result = _engine().compute({"AAPL": 50}, {"AAPL": _df(prices)}, net_liquidation=100_000)
        data   = json.loads(log_path.read_text())

        assert data["var_99_usd"] == pytest.approx(result.var_99_usd, rel=1e-4)
        assert data["n_positions"] == result.n_positions

    def test_no_positions_does_not_write_log(self, tmp_path, monkeypatch):
        import src.risk.var_engine as mod
        log_path = tmp_path / "var_latest.json"
        monkeypatch.setattr(mod, "_VAR_LOG", log_path)

        _engine().compute({}, {}, net_liquidation=100_000)
        assert not log_path.exists()


# ── Determinism ───────────────────────────────────────────────────────────────

class TestVaRDeterminism:

    def test_same_input_same_output(self):
        """Historical simulation is deterministic — same input → same result."""
        prices = _price_series(seed=7)
        r1 = _engine().compute({"AAPL": 50}, {"AAPL": _df(prices)}, net_liquidation=100_000)
        r2 = _engine().compute({"AAPL": 50}, {"AAPL": _df(prices)}, net_liquidation=100_000)
        assert r1.var_99_usd == pytest.approx(r2.var_99_usd)
        assert r1.var_95_usd == pytest.approx(r2.var_95_usd)
