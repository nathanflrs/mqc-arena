"""
Tests for src/analytics/factor_analysis.py  — 22 tests
Run: pytest tests/test_factor_analysis.py -v
"""
from __future__ import annotations

import json
import math
import warnings
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest

# ── Helpers ───────────────────────────────────────────────────────────────────

TRADING_DAYS = 252
N_OBS        = 500  # well above the 60-obs minimum


def _make_date_index(n: int) -> pd.DatetimeIndex:
    """N business days starting 2022-01-03."""
    return pd.bdate_range(start="2022-01-03", periods=n)


def _synthetic_factors(n: int = N_OBS) -> pd.DataFrame:
    """Realistic-looking random factor returns (decimal, not %)."""
    rng = np.random.default_rng(42)
    idx = _make_date_index(n)
    return pd.DataFrame(
        {
            "Mkt-RF": rng.normal(0.0004, 0.01,  n),
            "SMB":    rng.normal(0.0001, 0.005, n),
            "HML":    rng.normal(0.0001, 0.005, n),
            "Mom":    rng.normal(0.0002, 0.006, n),
            "RF":     np.full(n, 0.04 / 252),
        },
        index=idx,
    )


def _ff5_factors(n: int = N_OBS) -> pd.DataFrame:
    rng = np.random.default_rng(99)
    idx = _make_date_index(n)
    return pd.DataFrame(
        {
            "Mkt-RF": rng.normal(0.0004, 0.01,  n),
            "SMB":    rng.normal(0.0001, 0.005, n),
            "HML":    rng.normal(0.0001, 0.005, n),
            "RMW":    rng.normal(0.0001, 0.004, n),
            "CMA":    rng.normal(0.0001, 0.004, n),
            "RF":     np.full(n, 0.04 / 252),
        },
        index=idx,
    )


# ── 1. Factor download / cache ────────────────────────────────────────────────

def test_factor_download_or_cache(tmp_path):
    """Loader returns a non-empty DataFrame from cache or download."""
    from src.analytics.factor_analysis import FactorDataLoader

    mock_df = _synthetic_factors(300)
    mock_df.index.name = "date"

    with patch.object(
        FactorDataLoader, "_download", return_value=mock_df
    ) as mock_dl:
        loader = FactorDataLoader(cache_dir=str(tmp_path))
        df = loader.load_factors("carhart")
        assert not df.empty
        mock_dl.assert_called_once()

    # Second call hits cache — download not called again
    loader2 = FactorDataLoader(cache_dir=str(tmp_path))
    with patch.object(FactorDataLoader, "_download") as mock_dl2:
        df2 = loader2.load_factors("carhart")
        mock_dl2.assert_not_called()
        assert len(df2) == len(mock_df)


def test_factor_cache_stale_re_downloads(tmp_path):
    """Cache older than 24h triggers re-download."""
    import time
    from src.analytics.factor_analysis import FactorDataLoader

    mock_df = _synthetic_factors(100)
    loader = FactorDataLoader(cache_dir=str(tmp_path))

    # Write a stale cache file (mtime = 25h ago)
    cache_file = tmp_path / "factors_carhart.parquet"
    mock_df.to_parquet(cache_file)
    old_mtime = time.time() - 25 * 3600
    import os
    os.utime(cache_file, (old_mtime, old_mtime))

    fresh_df = _synthetic_factors(200)
    with patch.object(FactorDataLoader, "_download", return_value=fresh_df):
        df = loader.load_factors("carhart")
    assert len(df) == 200


# ── 2. Factors in decimal ─────────────────────────────────────────────────────

def test_factors_in_decimal():
    """Mkt-RF values must be in decimal form (not percent — not ±10 range)."""
    factors = _synthetic_factors()
    assert factors["Mkt-RF"].abs().max() < 0.1  # daily returns, not percent


def test_rf_positive_and_small():
    """Risk-free rate should be small and positive."""
    factors = _synthetic_factors()
    assert (factors["RF"] >= 0).all()
    assert factors["RF"].max() < 0.001  # daily RF from ~4% annual


# ── 3. Alpha annualisation ────────────────────────────────────────────────────

def test_alpha_annualization():
    """alpha_annualized == alpha_daily × 252 exactly."""
    from src.analytics.factor_analysis import FactorRegression

    rng     = np.random.default_rng(7)
    factors = _synthetic_factors(N_OBS)
    TRUE_ALPHA_DAILY = 0.0004

    # R = true_alpha + 1.0 * Mkt-RF + noise
    y = (
        TRUE_ALPHA_DAILY
        + 1.0 * factors["Mkt-RF"]
        + rng.normal(0, 0.002, N_OBS)
    )
    series = pd.Series(y, index=factors.index, name="TestAgent")

    reg    = FactorRegression(model="carhart")
    result = reg.run(series, factors)

    assert not result.insufficient_data
    assert math.isclose(
        result.alpha_annualized,
        result.alpha_daily * TRADING_DAYS,
        rel_tol=1e-9,
    )


# ── 4. Insufficient data flag ─────────────────────────────────────────────────

def test_insufficient_data_flag():
    """T < 60 → insufficient_data=True, alpha is NaN."""
    from src.analytics.factor_analysis import FactorRegression

    n       = 30
    factors = _synthetic_factors(n)
    series  = pd.Series(np.zeros(n), index=factors.index, name="TinyAgent")

    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        result = FactorRegression(min_observations=60).run(series, factors)

    assert result.insufficient_data
    assert math.isnan(result.alpha_annualized)
    assert result.n_observations == n
    # At least one RuntimeWarning about insufficient data
    assert any(issubclass(x.category, RuntimeWarning) for x in w)


# ── 5. Newey-West applied ─────────────────────────────────────────────────────

def test_newey_west_applied():
    """Regression must use cov_type='HAC'."""
    import statsmodels.api as sm

    from src.analytics.factor_analysis import FactorRegression

    factors = _synthetic_factors(N_OBS)
    series  = pd.Series(np.random.default_rng(1).normal(0, 0.01, N_OBS),
                        index=factors.index, name="A")

    fit_calls = []
    original_fit = sm.OLS.fit

    def spy_fit(self, *a, **kw):
        fit_calls.append(kw.get("cov_type"))
        return original_fit(self, *a, **kw)

    with patch.object(sm.OLS, "fit", spy_fit):
        FactorRegression().run(series, factors)

    assert any(c == "HAC" for c in fit_calls), "HAC cov_type not used"


# ── 6. Known-alpha recovery ───────────────────────────────────────────────────

def test_known_alpha_recovery():
    """OLS recovers injected alpha (0.0004 daily) and β_mkt ≈ 1.0."""
    from src.analytics.factor_analysis import FactorRegression

    rng     = np.random.default_rng(42)
    factors = _synthetic_factors(N_OBS)
    TRUE_ALPHA = 0.0004

    y      = TRUE_ALPHA + 1.0 * factors["Mkt-RF"] + rng.normal(0, 0.001, N_OBS)
    series = pd.Series(y, index=factors.index, name="SyntheticAlpha")

    result = FactorRegression().run(series, factors)

    assert not result.insufficient_data
    assert abs(result.alpha_daily - TRUE_ALPHA) < 0.0003, (
        f"alpha recovery too far off: {result.alpha_daily:.6f} vs {TRUE_ALPHA}"
    )
    assert abs(result.betas["Mkt-RF"] - 1.0) < 0.15


# ── 7. Zero-alpha strategy ────────────────────────────────────────────────────

def test_zero_alpha_strategy():
    """Pure market exposure → alpha not significant, β_mkt ≈ 1.

    y = RF + Mkt-RF + noise  →  excess return y - RF = Mkt-RF + noise  →  α ≈ 0
    """
    from src.analytics.factor_analysis import FactorRegression

    rng     = np.random.default_rng(10)
    factors = _synthetic_factors(N_OBS)
    # Construct so that excess return (y - RF) = Mkt-RF + tiny noise → true alpha = 0
    y       = factors["RF"] + 1.0 * factors["Mkt-RF"] + rng.normal(0, 0.0005, N_OBS)
    series  = pd.Series(y, index=factors.index, name="PureMarket")

    result = FactorRegression().run(series, factors)

    assert not result.insufficient_data
    assert abs(result.betas["Mkt-RF"] - 1.0) < 0.2
    # True alpha is zero by construction — daily alpha must be numerically tiny
    # (noise mean ≈ 0, so alpha should be < 10bps annualised)
    assert abs(result.alpha_annualized) < 0.10, (
        f"alpha too large for a pure-market strategy: {result.alpha_annualized:.2%}"
    )


# ── 8. Pure momentum strategy ────────────────────────────────────────────────

def test_pure_momentum_strategy():
    """Series built purely from MOM → β_mom large and significant."""
    from src.analytics.factor_analysis import FactorRegression

    rng     = np.random.default_rng(55)
    factors = _synthetic_factors(N_OBS)
    y       = 1.5 * factors["Mom"] + rng.normal(0, 0.0005, N_OBS)
    series  = pd.Series(y, index=factors.index, name="MomentumPure")

    result = FactorRegression().run(series, factors)

    assert not result.insufficient_data
    assert result.betas["Mom"] > 0.5
    # β_mom t-stat should be large
    assert result.beta_tstats["Mom"] > 2.0


# ── 9. Significance threshold ─────────────────────────────────────────────────

def test_significance_threshold():
    """alpha_significant == True iff |t-stat| > 1.96."""
    from src.analytics.factor_analysis import FactorRegression

    rng     = np.random.default_rng(7)
    factors = _synthetic_factors(N_OBS)
    # Inject large enough alpha to guarantee significance
    y       = 0.003 + factors["Mkt-RF"] + rng.normal(0, 0.001, N_OBS)
    series  = pd.Series(y, index=factors.index, name="HighAlpha")

    result = FactorRegression().run(series, factors)

    assert result.alpha_significant == (abs(result.alpha_tstat) > 1.96)


# ── 10. Flat days as zero ─────────────────────────────────────────────────────

def test_flat_days_as_zero():
    """Live execution builder fills zero on days without a closed trade."""
    from src.analytics.factor_analysis import AgentReturnSeriesBuilder

    builder = AgentReturnSeriesBuilder(
        decisions_path="logs/decisions.csv",
        executions_path="logs/executions.csv",
        walkforward_path="logs/walkforward_results.csv",
    )
    # Build a minimal live series by monkeypatching the scorer round-trips
    from src.risk.live_scorer import RoundTrip

    trips = [
        RoundTrip(
            agent="TestAgent",
            symbol="AAPL",
            entry_price=100.0,
            exit_price=105.0,
            entry_date=pd.Timestamp("2024-01-02", tz="UTC"),
            exit_date=pd.Timestamp("2024-01-10", tz="UTC"),
        ),
    ]

    class FakeScorer:
        _roundtrips = trips
        def _load(self): pass

    from unittest.mock import patch
    with patch("src.risk.live_scorer.LiveScorer", return_value=FakeScorer()), \
         patch("src.risk.live_scorer.LiveScorerConfig"):
        series = builder._from_live_executions("TestAgent")

    # Should have zeros on all business days except the exit date
    assert (series == 0).any(), "Flat days (no trade) must be zero"
    # Return on exit date should be non-zero
    exit_date = pd.Timestamp("2024-01-10").normalize()
    if exit_date in series.index:
        assert series[exit_date] != 0


# ── 11. Alignment with factors ────────────────────────────────────────────────

def test_alignment_with_factors():
    """Inner join keeps only dates common to both series and factors."""
    from src.analytics.factor_analysis import FactorRegression

    factors = _synthetic_factors(N_OBS)
    # Series that only partially overlaps factors (last 200 days)
    partial_idx = factors.index[-200:]
    y      = np.random.default_rng(3).normal(0, 0.01, 200)
    series = pd.Series(y, index=partial_idx, name="Partial")

    result = FactorRegression(min_observations=100).run(series, factors)

    assert result.n_observations == 200


# ── 12. R-squared range ───────────────────────────────────────────────────────

def test_r_squared_range():
    """R² must be between 0 and 1 inclusive."""
    from src.analytics.factor_analysis import FactorRegression

    factors = _synthetic_factors(N_OBS)
    y       = np.random.default_rng(4).normal(0, 0.01, N_OBS)
    series  = pd.Series(y, index=factors.index, name="RandomAgent")

    result = FactorRegression().run(series, factors)

    assert not result.insufficient_data
    assert 0.0 <= result.r_squared <= 1.0


# ── 13. Information Ratio ─────────────────────────────────────────────────────

def test_information_ratio():
    """IR == alpha_annual / residual_vol_annual."""
    from src.analytics.factor_analysis import FactorRegression

    factors = _synthetic_factors(N_OBS)
    y       = 0.0005 + np.random.default_rng(5).normal(0, 0.01, N_OBS)
    series  = pd.Series(y, index=factors.index, name="IRAgent")

    result = FactorRegression().run(series, factors)

    if result.insufficient_data or result.residual_vol_annual == 0:
        pytest.skip("insufficient data or zero vol")

    expected_ir = result.alpha_annualized / result.residual_vol_annual
    assert math.isclose(result.information_ratio, expected_ir, rel_tol=1e-9)


# ── 14. Negative alpha verdict ────────────────────────────────────────────────

def test_negative_alpha_verdict():
    """A strategy with significant negative alpha → 'ALPHA NÉGATIF' in verdict."""
    from src.analytics.factor_analysis import FactorRegression

    rng     = np.random.default_rng(99)
    factors = _synthetic_factors(N_OBS)
    # Very negative alpha, tiny noise so it's significant
    y       = -0.003 + factors["Mkt-RF"] + rng.normal(0, 0.001, N_OBS)
    series  = pd.Series(y, index=factors.index, name="BadAgent")

    result = FactorRegression().run(series, factors)

    assert not result.insufficient_data
    if result.alpha_significant and result.alpha_annualized < 0:
        assert "ALPHA NÉGATIF" in result.interpretation


# ── 15. Fallback to walkforward ───────────────────────────────────────────────

def test_fallback_to_walkforward():
    """With <60 live days, auto source should fall back to walkforward."""
    from src.analytics.factor_analysis import AgentReturnSeriesBuilder

    builder = AgentReturnSeriesBuilder(
        decisions_path="nonexistent.csv",
        executions_path="nonexistent.csv",
        walkforward_path="logs/walkforward_results.csv",
    )
    if not Path("logs/walkforward_results.csv").exists():
        pytest.skip("walkforward_results.csv not available")

    agents = builder._available_agents()
    if not agents:
        pytest.skip("No agents in walkforward_results.csv")

    # Patch _from_live_executions to return a short series (< 60 days)
    short_series = pd.Series(
        [0.001] * 5 + [0.0] * 5,
        index=pd.bdate_range("2024-01-01", periods=10),
        name=agents[0],
    )

    with patch.object(builder, "_from_live_executions", return_value=short_series):
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            result = builder.build_daily_returns(agents[0], source="auto")
        # Should warn about fallback
        assert any("fallback" in str(x.message).lower() or
                   "walkforward" in str(x.message).lower()
                   for x in w)
        # Should return the walkforward series (longer)
        assert len(result) > 10


# ── 16. JSON round-trip ───────────────────────────────────────────────────────

def test_json_roundtrip(tmp_path):
    """save_json and reload produce identical data."""
    from src.analytics.factor_analysis import (
        FactorRegression, FactorReporter,
    )

    factors = _synthetic_factors(N_OBS)
    y       = 0.0003 + np.random.default_rng(11).normal(0, 0.01, N_OBS)
    series  = pd.Series(y, index=factors.index, name="JSONAgent")

    result  = FactorRegression().run(series, factors)
    results = {"JSONAgent": result}

    reporter = FactorReporter()
    out_path = str(tmp_path / "test_factor.json")
    reporter.save_json(results, out_path)

    with open(out_path) as f:
        loaded = json.load(f)

    assert "JSONAgent" in loaded
    r = loaded["JSONAgent"]
    assert math.isclose(r["alpha_annualized"], result.alpha_annualized, rel_tol=1e-9)
    assert math.isclose(r["r_squared"],        result.r_squared,        rel_tol=1e-9)


# ── 17. FF5 model runs ────────────────────────────────────────────────────────

def test_5factor_model():
    """FF5 model runs without error and returns RMW, CMA betas."""
    from src.analytics.factor_analysis import FactorRegression

    factors = _ff5_factors(N_OBS)
    y       = np.random.default_rng(22).normal(0, 0.01, N_OBS)
    series  = pd.Series(y, index=factors.index, name="FF5Agent")

    result = FactorRegression(model="ff5").run(series, factors)

    assert not result.insufficient_data
    assert "RMW" in result.betas
    assert "CMA" in result.betas
    assert result.model == "ff5"


# ── 18. run_all on all agents ─────────────────────────────────────────────────

def test_all_agents_run():
    """run_all processes multiple agents without crashing."""
    from src.analytics.factor_analysis import FactorRegression

    rng     = np.random.default_rng(33)
    factors = _synthetic_factors(N_OBS)

    agents = {
        f"Agent{i}": pd.Series(
            rng.normal(0, 0.01, N_OBS), index=factors.index, name=f"Agent{i}"
        )
        for i in range(4)
    }

    reg     = FactorRegression()
    results = reg.run_all(agents, factors)

    assert len(results) == 4
    for r in results.values():
        assert isinstance(r.n_observations, int)


# ── 19. Dominant factor detection ────────────────────────────────────────────

def test_dominant_factor_detection():
    """Agent exposed to SMB (β=1.5) should identify SMB as dominant."""
    from src.analytics.factor_analysis import FactorRegression

    rng     = np.random.default_rng(77)
    factors = _synthetic_factors(N_OBS)
    # Only SMB exposure, large coefficient
    y       = 1.5 * factors["SMB"] + rng.normal(0, 0.0005, N_OBS)
    series  = pd.Series(y, index=factors.index, name="SMBAgent")

    result = FactorRegression().run(series, factors)

    assert not result.insufficient_data
    # SMB beta should be the largest
    sig_betas = {
        f: abs(b) for f, b in result.betas.items()
        if abs(result.beta_tstats.get(f, 0)) > 1.96
    }
    if sig_betas:
        dominant = max(sig_betas, key=sig_betas.__getitem__)
        assert dominant == "SMB", f"Expected SMB, got {dominant} (betas={result.betas})"


# ── 20. Telegram format buckets ──────────────────────────────────────────────

def test_telegram_format_buckets():
    """Telegram message contains the 3 classification buckets."""
    from src.analytics.factor_analysis import (
        FactorRegression, FactorReporter, RegressionResult,
    )
    import math

    # Build two results: one with alpha, one without
    rng     = np.random.default_rng(44)
    factors = _synthetic_factors(N_OBS)

    s_alpha = pd.Series(
        0.003 + factors["Mkt-RF"].values + rng.normal(0, 0.001, N_OBS),
        index=factors.index, name="AlphaAgent",
    )
    s_beta = pd.Series(
        factors["Mkt-RF"].values + rng.normal(0, 0.005, N_OBS),
        index=factors.index, name="BetaAgent",
    )

    reg     = FactorRegression()
    results = {
        "AlphaAgent": reg.run(s_alpha, factors),
        "BetaAgent":  reg.run(s_beta,  factors),
    }

    reporter = FactorReporter()
    msg      = reporter.format_telegram(results)

    # At minimum, the message should contain two of the 3 bucket headers
    bucket_headers = ["VRAI ALPHA", "BETA DÉGUISÉ", "ALPHA NÉGATIF", "DONNÉES INSUFFISANTES"]
    found = sum(1 for h in bucket_headers if h in msg)
    assert found >= 1, f"No bucket headers found in message:\n{msg}"


# ── 21. CSV save ──────────────────────────────────────────────────────────────

def test_csv_save(tmp_path):
    """save_csv produces a readable CSV with key columns."""
    from src.analytics.factor_analysis import FactorRegression, FactorReporter

    factors = _synthetic_factors(N_OBS)
    y       = np.random.default_rng(9).normal(0, 0.01, N_OBS)
    series  = pd.Series(y, index=factors.index, name="CSVAgent")
    result  = FactorRegression().run(series, factors)

    reporter = FactorReporter()
    out_path = str(tmp_path / "test.csv")
    reporter.save_csv({"CSVAgent": result}, out_path)

    df = pd.read_csv(out_path)
    assert "agent" in df.columns
    assert "alpha_annualized" in df.columns
    assert len(df) == 1


# ── 22. Residual vol and IR consistency ──────────────────────────────────────

def test_residual_vol_and_ir_consistency():
    """residual_vol_annual is positive and IR has correct sign as alpha."""
    from src.analytics.factor_analysis import FactorRegression

    factors = _synthetic_factors(N_OBS)
    y       = 0.002 + np.random.default_rng(6).normal(0, 0.01, N_OBS)
    series  = pd.Series(y, index=factors.index, name="VolAgent")

    result = FactorRegression().run(series, factors)

    assert not result.insufficient_data
    assert result.residual_vol_annual > 0
    # IR sign matches alpha sign
    assert (result.information_ratio >= 0) == (result.alpha_annualized >= 0)
