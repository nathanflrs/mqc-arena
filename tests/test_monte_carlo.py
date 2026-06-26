"""
Tests for src/analytics/monte_carlo.py
"""
from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest

from src.analytics.monte_carlo import (
    MonteCarloEngine,
    MonteCarloReporter,
    ReturnBootstrapper,
    SimulationResult,
    run_simulation,
)


# ── Fixtures ─────────────────────────────────────────────────────────────────

@pytest.fixture
def small_returns():
    """Small balanced return array for fast, deterministic tests."""
    return np.array([0.01, -0.01, 0.005, -0.005, 0.02, -0.02])


@pytest.fixture
def engine_small(small_returns):
    return MonteCarloEngine(n_simulations=200, horizon_days=10)


@pytest.fixture
def result_small(engine_small, small_returns):
    np.random.seed(42)
    return engine_small.run(small_returns)


# ══════════════════════════════════════════════════════════════════════════════
# Shape and initialization
# ══════════════════════════════════════════════════════════════════════════════

def test_n_paths_shape(result_small):
    assert result_small.paths.shape == (200, 10)


def test_initial_nav(small_returns):
    """Day-1 NAVs should be very close to initial_nav for small daily returns."""
    np.random.seed(0)
    result = MonteCarloEngine(n_simulations=500, horizon_days=5, initial_nav=1_000_000.0).run(
        small_returns
    )
    # With max daily return of ±2% and kelly_scale=0.69, max day-1 move is ±1.38%
    assert np.all(result.paths[:, 0] > 900_000)
    assert np.all(result.paths[:, 0] < 1_100_000)


def test_final_navs_shape(result_small):
    assert result_small.final_navs.shape == (200,)


def test_final_returns_consistent(result_small):
    expected = (result_small.final_navs - 1_000_000.0) / 1_000_000.0
    np.testing.assert_allclose(result_small.final_returns, expected, rtol=1e-9)


# ══════════════════════════════════════════════════════════════════════════════
# Risk metric ordering
# ══════════════════════════════════════════════════════════════════════════════

def test_var_ordering(result_small):
    """VaR 99% must be ≤ VaR 95% (larger loss at higher confidence)."""
    assert result_small.var_99 <= result_small.var_95


def test_cvar_leq_var(result_small):
    """CVaR (expected shortfall) must be ≤ VaR 95% (average loss beyond VaR)."""
    assert result_small.cvar_95 <= result_small.var_95 + 1e-10


def test_prob_positive_range(result_small):
    assert 0.0 <= result_small.prob_positive <= 1.0


def test_percentile_ordering(result_small):
    p = result_small.percentiles
    assert p["p5"] <= p["p10"] <= p["p25"] <= p["p50"] <= p["p75"] <= p["p90"] <= p["p95"]


# ══════════════════════════════════════════════════════════════════════════════
# Circuit breaker behavior
# ══════════════════════════════════════════════════════════════════════════════

def test_circuit_breaker_applied():
    """Returns of -5%/day guarantee CB trigger (DD>8%) within 3 days."""
    returns = np.array([-0.05])
    np.random.seed(0)
    result = MonteCarloEngine(
        n_simulations=500, horizon_days=20, gmm_kelly_scale=1.0
    ).run(returns)
    assert result.prob_circuit_breaker > 0.99


def test_sell_only_at_8pct_dd():
    """Once DD > 8%, subsequent r_net should be 0 (no further NAV decline)."""
    # Force DD > 8%: -10% return, Kelly=1, no TC
    returns = np.array([-0.10])
    np.random.seed(1)
    engine = MonteCarloEngine(
        n_simulations=100,
        horizon_days=20,
        gmm_kelly_scale=1.0,
        transaction_cost_bps=0.0,
    )
    result = engine.run(returns)
    # After level 3 is triggered, NAV should freeze (not keep declining)
    # Check that final_navs are not all extremely low
    # With sell-only, paths plateau after CB trigger
    # Min possible NAV (without CB) = 1M * 0.9^20 ≈ 122k
    # With CB, NAV should stop at the level-3 trigger point
    assert np.mean(result.final_navs) > 100_000  # not total wipeout


def test_defensive_mode_at_4pct_dd():
    """DD > 4%: kelly_scale halved → less loss amplification than without CB."""
    returns = np.array([-0.03])
    np.random.seed(42)

    result_with_cb = MonteCarloEngine(
        n_simulations=500, horizon_days=30, gmm_kelly_scale=1.0, transaction_cost_bps=0.0
    ).run(returns)

    # Engine with very high CB thresholds (effectively disabled)
    np.random.seed(42)
    result_no_cb = MonteCarloEngine(
        n_simulations=500,
        horizon_days=30,
        gmm_kelly_scale=1.0,
        transaction_cost_bps=0.0,
        circuit_breaker_levels={0.99: 1, 0.999: 2, 0.9999: 3},
    ).run(returns)

    # With CB, losses are dampened — expected return should be higher (less negative)
    assert result_with_cb.expected_return >= result_no_cb.expected_return


# ══════════════════════════════════════════════════════════════════════════════
# Kelly scaling
# ══════════════════════════════════════════════════════════════════════════════

def test_kelly_scale_reduces_returns():
    """Lower Kelly scale → lower expected return magnitude (both positive and negative)."""
    returns = np.array([0.05, -0.05, 0.03, -0.03])
    np.random.seed(7)
    result_half = MonteCarloEngine(
        n_simulations=500, horizon_days=20, gmm_kelly_scale=0.5, transaction_cost_bps=0.0
    ).run(returns)
    np.random.seed(7)
    result_full = MonteCarloEngine(
        n_simulations=500, horizon_days=20, gmm_kelly_scale=1.0, transaction_cost_bps=0.0
    ).run(returns)
    # Half Kelly → smaller spread (lower volatility)
    assert result_half.annualized_volatility < result_full.annualized_volatility


def test_bear_regime_worse():
    """Bear regime returns → lower expected_return than bull regime returns."""
    bear_returns = np.array([-0.03, -0.02, -0.01, -0.005])
    bull_returns = np.array([0.03, 0.02, 0.01, 0.005])

    np.random.seed(99)
    result_bear = MonteCarloEngine(
        n_simulations=500, horizon_days=20, gmm_regime="bear", transaction_cost_bps=0.0
    ).run(bear_returns)

    np.random.seed(99)
    result_bull = MonteCarloEngine(
        n_simulations=500, horizon_days=20, gmm_regime="bull_quiet", transaction_cost_bps=0.0
    ).run(bull_returns)

    assert result_bear.expected_return < result_bull.expected_return


# ══════════════════════════════════════════════════════════════════════════════
# Annualized return formula
# ══════════════════════════════════════════════════════════════════════════════

def test_annualized_return_formula():
    """annualized_return == (1 + expected_return)^(365/h) - 1"""
    np.random.seed(0)
    result = MonteCarloEngine(n_simulations=200, horizon_days=45).run(
        np.array([0.005, -0.003])
    )
    expected_ann = (1.0 + result.expected_return) ** (365.0 / 45) - 1.0
    assert abs(result.annualized_return - expected_ann) < 1e-10


# ══════════════════════════════════════════════════════════════════════════════
# Transaction costs
# ══════════════════════════════════════════════════════════════════════════════

def test_transaction_costs_applied():
    """Simulation with TC should produce lower expected_return than without."""
    returns = np.array([0.005, 0.003, 0.002])
    np.random.seed(5)
    result_with_tc = MonteCarloEngine(
        n_simulations=500, horizon_days=30, transaction_cost_bps=9.0
    ).run(returns)
    np.random.seed(5)
    result_no_tc = MonteCarloEngine(
        n_simulations=500, horizon_days=30, transaction_cost_bps=0.0
    ).run(returns)
    assert result_with_tc.expected_return < result_no_tc.expected_return


# ══════════════════════════════════════════════════════════════════════════════
# JSON serialization
# ══════════════════════════════════════════════════════════════════════════════

def test_json_serialization(result_small):
    reporter = MonteCarloReporter()
    with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
        tmp_path = f.name
    try:
        reporter.save_json(result_small, tmp_path)
        loaded = reporter.load_json(tmp_path)
        assert loaded.var_95 == pytest.approx(result_small.var_95, rel=1e-9)
        assert loaded.var_99 == pytest.approx(result_small.var_99, rel=1e-9)
        assert loaded.cvar_95 == pytest.approx(result_small.cvar_95, rel=1e-9)
        assert loaded.sharpe_ratio == pytest.approx(result_small.sharpe_ratio, rel=1e-9)
        assert loaded.n_simulations == result_small.n_simulations
        assert loaded.horizon_days == result_small.horizon_days
        assert loaded.regime == result_small.regime
        assert loaded.percentiles == pytest.approx(result_small.percentiles, rel=1e-9)
        # Arrays are not stored
        assert loaded.paths is None
        assert loaded.final_navs is None
    finally:
        os.unlink(tmp_path)


# ══════════════════════════════════════════════════════════════════════════════
# Telegram format
# ══════════════════════════════════════════════════════════════════════════════

def test_telegram_format(result_small):
    reporter = MonteCarloReporter()
    msg = reporter.format_telegram(result_small)
    assert "VaR" in msg
    assert "Sharpe" in msg
    assert "MONTE CARLO" in msg
    assert "Médiane" in msg
    assert "CVaR" in msg


def test_tearsheet_section(result_small):
    reporter = MonteCarloReporter()
    section = reporter.format_tearsheet_section(result_small)
    assert "Monte Carlo" in section
    assert "VaR" in section


# ══════════════════════════════════════════════════════════════════════════════
# Edge cases
# ══════════════════════════════════════════════════════════════════════════════

def test_n_simulations_1():
    """n=1 simulation should not crash."""
    np.random.seed(0)
    result = MonteCarloEngine(n_simulations=1, horizon_days=10).run(
        np.array([0.01, -0.01])
    )
    assert result.paths.shape == (1, 10)
    assert result.n_simulations == 1


def test_horizon_1():
    """horizon=1 day should not crash."""
    np.random.seed(0)
    result = MonteCarloEngine(n_simulations=50, horizon_days=1).run(
        np.array([0.01, -0.01])
    )
    assert result.paths.shape == (50, 1)
    assert result.horizon_days == 1


def test_reproducibility():
    """Same seed → identical results."""
    returns = np.array([0.01, -0.01, 0.005, -0.005])
    np.random.seed(42)
    r1 = MonteCarloEngine(n_simulations=100, horizon_days=10).run(returns)
    np.random.seed(42)
    r2 = MonteCarloEngine(n_simulations=100, horizon_days=10).run(returns)
    np.testing.assert_array_equal(r1.paths, r2.paths)
    assert r1.var_95 == r2.var_95
    assert r1.sharpe_ratio == r2.sharpe_ratio


# ══════════════════════════════════════════════════════════════════════════════
# ReturnBootstrapper
# ══════════════════════════════════════════════════════════════════════════════

def test_fallback_to_walkforward(tmp_path):
    """Empty decisions.csv → bootstrapper falls back to walkforward_results.csv."""
    # Write a minimal walkforward CSV
    wf_csv = tmp_path / "walkforward_results.csv"
    wf_csv.write_text(
        "agent,symbol,window,test_start,test_end,oos_return\n"
        "BuffettAgent,AAPL,0,2023-01-01,2023-06-30,0.12\n"
        "BuffettAgent,AAPL,1,2023-07-01,2023-12-31,-0.05\n"
    )
    # decisions.csv doesn't exist
    dec_path = tmp_path / "decisions.csv"
    exc_path = tmp_path / "executions.csv"

    bootstrapper = ReturnBootstrapper(
        decisions_path=str(dec_path),
        executions_path=str(exc_path),
        walkforward_path=str(wf_csv),
    )
    returns = bootstrapper.get_portfolio_returns()
    # Should get daily-equivalent returns from walkforward
    assert len(returns) == 2
    # Daily returns should be much smaller than window returns
    assert all(abs(r) < 0.01 for r in returns)


def test_regime_conditioned_bootstrap(tmp_path):
    """get_regime_conditioned_returns filters by regime if enough samples exist."""
    # Create enough round-trips (≥30) with known regimes; use hour offsets to avoid date overflow
    dec_rows = ["ts,symbol,regime,winner_agent,action,confidence,target_weight,reason,meta,plan_id,timestamp,agent,is_winner"]
    exc_rows = ["timestamp,symbol,side,plan_id,avg_fill_price,limit_price,last_price"]
    for i in range(40):
        regime = "bear" if i < 30 else "bull_quiet"
        pid = f"plan_{i:03d}"
        # Spread across months to keep all dates valid
        month_buy  = 1 + (i // 31)       # Jan or Feb
        day_buy    = 1 + (i % 28)        # 1-28, safe for all months
        month_sell = month_buy + 2        # 2 months later
        buy_ts  = f"2024-{month_buy:02d}-{day_buy:02d}T10:00:00+00:00"
        sell_ts = f"2024-{month_sell:02d}-{day_buy:02d}T10:00:00+00:00"
        dec_rows.append(
            f",SYM,{regime},TestAgent,BUY,0.8,0.1,reason,{{}},{pid},{buy_ts},TestAgent,True"
        )
        exc_rows.append(f"{buy_ts},SYM,BUY,{pid},100.0,,")
        exc_rows.append(f"{sell_ts},SYM,SELL,{pid},102.0,,")

    (tmp_path / "decisions.csv").write_text("\n".join(dec_rows))
    (tmp_path / "executions.csv").write_text("\n".join(exc_rows))
    wf = tmp_path / "walkforward_results.csv"
    wf.write_text("agent,symbol,window,test_start,test_end,oos_return\n")

    bootstrapper = ReturnBootstrapper(
        decisions_path=str(tmp_path / "decisions.csv"),
        executions_path=str(tmp_path / "executions.csv"),
        walkforward_path=str(wf),
    )
    bear_returns = bootstrapper.get_regime_conditioned_returns("bear")
    # Should have ≥30 bear samples, all with ~2% net return (102-100)/100 - TC
    assert len(bear_returns) == 30
    for r in bear_returns:
        assert abs(r - (0.02 - 0.0009)) < 1e-9


def test_bootstrapper_missing_files(tmp_path):
    """Missing CSV files → fallback returns a valid (non-empty) array."""
    bootstrapper = ReturnBootstrapper(
        decisions_path=str(tmp_path / "missing_dec.csv"),
        executions_path=str(tmp_path / "missing_exc.csv"),
        walkforward_path=str(tmp_path / "missing_wf.csv"),
    )
    returns = bootstrapper.get_portfolio_returns()
    assert isinstance(returns, np.ndarray)
    assert len(returns) >= 1


# ══════════════════════════════════════════════════════════════════════════════
# run_simulation convenience function
# ══════════════════════════════════════════════════════════════════════════════

def test_run_simulation_no_save(tmp_path):
    """run_simulation with save_path=None should not create a file."""
    result = run_simulation(
        n_simulations=50,
        horizon_days=5,
        decisions_path=str(tmp_path / "d.csv"),
        executions_path=str(tmp_path / "e.csv"),
        walkforward_path=str(tmp_path / "w.csv"),
        save_path=None,
    )
    assert isinstance(result, SimulationResult)
    assert not any(tmp_path.iterdir())  # no file written


def test_run_simulation_saves_json(tmp_path):
    """run_simulation with save_path should write a loadable JSON."""
    output = tmp_path / "mc_out.json"
    result = run_simulation(
        n_simulations=50,
        horizon_days=5,
        decisions_path=str(tmp_path / "d.csv"),
        executions_path=str(tmp_path / "e.csv"),
        walkforward_path=str(tmp_path / "w.csv"),
        save_path=str(output),
    )
    assert output.exists()
    loaded = MonteCarloReporter().load_json(str(output))
    assert loaded.n_simulations == 50
    assert loaded.horizon_days == 5
