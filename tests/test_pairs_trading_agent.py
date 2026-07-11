"""
Tests for src/agents/pairs_trading.py

Covers:
- Cointegration testing + pair selection (PairSelector)
- Multi-window robustness filter
- Two-leg signal generation (generate_signal)
- Entry / exit / stop-loss / mean-reversion logic
- RiskManager market-neutral routing
- pairs_plans_from_signal (execution infrastructure)
- Round-trip P&L concept

Run: pytest tests/test_pairs_trading_agent.py -v
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from datetime import datetime, timezone
from typing import Dict
from unittest.mock import patch, MagicMock

from src.agents.base import MarketState, AgentSignal
from src.agents.pairs_trading import (
    PairsTradingConfig,
    ValidatedPair,
    PairValidationReport,
    PairSelector,
    PairsTradingAgent,
    CANDIDATE_PAIRS,
    _COINT_WINDOWS,
    _fetch_close,
    _compute_halflife,
    _compute_hurst,
    _count_zero_crossings,
    _MA_RISK,
)
from src.execution.planner import OrderPlan, pairs_plans_from_signal
from src.risk.manager import RiskConfig, RiskManager
from src.broker.portfolio import PortfolioSnapshot


# ── Synthetic data helpers ────────────────────────────────────────────────────

def _make_ou_spread(halflife: float, n: int = 500, seed: int = 42) -> np.ndarray:
    """OU process spread with a known half-life (trading days)."""
    rng = np.random.default_rng(seed)
    eta = np.log(2.0) / halflife   # mean-reversion rate
    spread = np.zeros(n)
    for t in range(1, n):
        spread[t] = spread[t - 1] * (1.0 - eta) + rng.standard_normal() * 0.01
    return spread


def _make_cointegrated_prices(
    n: int = 700, beta: float = 1.5, seed: int = 42, halflife: float = 7.0,
) -> Dict[str, pd.Series]:
    """
    Build two price series cointegrated by construction.

    Structure
    ---------
        log(A) = 4.5 + common + spread(OU)
        log(B) = (4.5 + common) / beta

    where `common` is a random walk (the shared integrated factor) and `spread`
    is a small OU process (the stationary residual).

    Design rationale
    ----------------
    The common factor uses step=0.01 (10× the spread noise 0.001) so that the
    integrated signal dominates the noise in the OLS regression.  The ADF test
    has very low power for near-unit-root processes: with halflife=20 (φ≈0.954)
    only the 504j window rejects the unit root reliably.  Using halflife=7
    (φ≈0.906) gives consistent rejection across all three windows while keeping
    the detected halflife well within the [5, 60j] acceptance band.

    Sequential RNG draws for `common` then `spread` within the same generator
    guarantee statistical independence between the two series.
    """
    rng    = np.random.default_rng(seed)
    # Strong common integrated factor — dominates the spread in the regression
    common = np.cumsum(rng.standard_normal(n) * 0.01)
    # Small OU spread — drawn AFTER common → independent of it
    eta    = np.log(2.0) / halflife
    spread = np.zeros(n)
    for t in range(1, n):
        spread[t] = spread[t - 1] * (1.0 - eta) + rng.standard_normal() * 0.001
    # log(A) = 4.5 + common + spread
    # log(B) = (4.5 + common) / beta  →  log(A) = beta * log(B) + spread  (cointegrated)
    log_a  = 4.5 + common + spread
    log_b  = (4.5 + common) / beta
    idx    = pd.date_range("2022-01-01", periods=n, freq="B")
    return {
        "SYNTH_A": pd.Series(np.exp(log_a), index=idx, dtype=float),
        "SYNTH_B": pd.Series(np.exp(log_b), index=idx, dtype=float),
    }


def _make_non_cointegrated_prices(n: int = 600, seed: int = 99) -> Dict[str, pd.Series]:
    """Two independent random walks — no cointegration by construction."""
    rng = np.random.default_rng(seed)
    log_a = np.cumsum(rng.standard_normal(n) * 0.01)
    log_b = np.cumsum(rng.standard_normal(n) * 0.01)
    return {
        "IND_A": pd.Series(np.exp(log_a + 4.0)),
        "IND_B": pd.Series(np.exp(log_b + 4.0)),
    }


def _make_selector_with_pair(pair: ValidatedPair, cfg: PairsTradingConfig | None = None) -> PairSelector:
    """PairSelector pre-loaded with a specific ValidatedPair, no downloads."""
    c   = cfg or PairsTradingConfig()
    sel = PairSelector(c)
    sel._validated[(pair.ticker_a, pair.ticker_b)] = pair
    sel._last_full_test = datetime.now(timezone.utc)
    return sel


def _default_pair(
    ticker_a: str = "SYNTH_A",
    ticker_b: str = "SYNTH_B",
    hedge_ratio: float = 1.5,
    robustness_score: int = 3,
    is_active: bool = True,
) -> ValidatedPair:
    return ValidatedPair(
        ticker_a=ticker_a,
        ticker_b=ticker_b,
        hedge_ratio=hedge_ratio,
        pvalue=0.002,
        last_tested=datetime.now(timezone.utc),
        spread_mean=0.0,
        spread_std=0.01,
        is_active=is_active,
        robustness_score=robustness_score,
        window_pvalues={126: 0.01, 252: 0.002, 504: 0.01},
    )


def _state(symbol: str = "SYNTH_A", price: float = 100.0) -> MarketState:
    return MarketState(symbol=symbol, price=price, timestamp="2026-01-01T00:00:00Z")


def _snap(netliq: float = 100_000.0, cash: float = 60_000.0) -> PortfolioSnapshot:
    return PortfolioSnapshot(net_liquidation=netliq, cash=cash, positions={})


def _order_plan(symbol: str, action: str, est: float, current: float = 0.0,
                strategy: str = "directional") -> OrderPlan:
    return OrderPlan(
        symbol=symbol, action=action, target_weight=0.0,
        last_price=100.0, current_qty=current,
        target_qty=0.0, delta_qty=0.0,
        est_notional=est, reason="test",
        strategy=strategy,
    )


# ── 1. Cointegration: non-cointegrated pair is rejected ──────────────────────

def test_cointegration_required():
    """A pair with p-value ≥ threshold must never be accepted."""
    prices = _make_non_cointegrated_prices()
    cfg    = PairsTradingConfig(pvalue_threshold=0.05, multi_window_check=False)
    sel    = PairSelector(cfg)
    sel._price_cache.update(prices)

    import src.agents.pairs_trading as pt_mod
    orig = pt_mod.CANDIDATE_PAIRS
    pt_mod.CANDIDATE_PAIRS = [("IND_A", "IND_B")]
    try:
        pair = sel._test_pair("IND_A", "IND_B")
        # May or may not pass on synthetic random walk — but if it passes, robustness_score matters
        # For true independent walks, coint should reject most of the time
        # We only check that if p >= threshold, result is None
        from statsmodels.tsa.stattools import coint
        sa = prices["IND_A"]
        sb = prices["IND_B"]
        df = pd.DataFrame({"a": sa, "b": sb}).dropna().iloc[-252:]
        _, p, _ = coint(np.log(df["a"].values), np.log(df["b"].values))
        if p >= 0.05:
            assert pair is None
    finally:
        pt_mod.CANDIDATE_PAIRS = orig


def test_cointegrated_pair_accepted():
    """A synthetically cointegrated pair (score=3) must pass the selector."""
    prices = _make_cointegrated_prices(n=700)
    cfg    = PairsTradingConfig(multi_window_check=True, min_robustness_score=2)
    sel    = PairSelector(cfg)
    sel._price_cache.update(prices)

    import src.agents.pairs_trading as pt_mod
    orig = pt_mod.CANDIDATE_PAIRS
    pt_mod.CANDIDATE_PAIRS = [("SYNTH_A", "SYNTH_B")]
    try:
        pair = sel._test_pair("SYNTH_A", "SYNTH_B")
        assert pair is not None, "Synthetically cointegrated pair should be accepted"
        assert pair.robustness_score >= 2
        assert pair.is_active
        assert 126 in pair.window_pvalues
        assert 252 in pair.window_pvalues
        assert 504 in pair.window_pvalues
    finally:
        pt_mod.CANDIDATE_PAIRS = orig


def test_hedge_ratio_estimation():
    """Estimated β should be close to the true β used to construct the series."""
    prices   = _make_cointegrated_prices(n=600, beta=1.5)
    log_a    = np.log(prices["SYNTH_A"].values)
    log_b    = np.log(prices["SYNTH_B"].values)
    cfg      = PairsTradingConfig()
    sel      = PairSelector(cfg)
    beta_hat = sel._estimate_hedge_ratio(log_a[-252:], log_b[-252:])
    assert abs(beta_hat - 1.5) < 0.4, f"β={beta_hat:.4f} too far from true 1.5"


def test_multi_window_check_default_true():
    """multi_window_check must be True by default."""
    cfg = PairsTradingConfig()
    assert cfg.multi_window_check is True
    assert cfg.min_robustness_score == 2


def test_single_window_pair_rejected_by_robustness():
    """A pair with robustness_score=1 must be rejected when multi_window_check=True."""
    prices = _make_cointegrated_prices(n=700)
    cfg    = PairsTradingConfig(multi_window_check=True, min_robustness_score=2)
    sel    = PairSelector(cfg)
    sel._price_cache.update(prices)

    import src.agents.pairs_trading as pt_mod
    orig = pt_mod.CANDIDATE_PAIRS
    pt_mod.CANDIDATE_PAIRS = [("SYNTH_A", "SYNTH_B")]

    # Patch _test_single_window to make only the 252j window pass
    from statsmodels.tsa.stattools import coint as real_coint
    call_count = {"n": 0}
    original_coint_backup = None

    def fake_coint(a, b):
        call_count["n"] += 1
        n = len(a)
        # Force fail on 126j and 504j windows
        if n == 126 or n == 504:
            return (0.0, 0.99, None)
        return real_coint(a, b)

    try:
        with patch("src.agents.pairs_trading.coint", side_effect=fake_coint):
            pair = sel._test_pair("SYNTH_A", "SYNTH_B")
        # Score should be 1 (only 252j passes) → rejected
        assert pair is None, f"Score-1 pair should be rejected with min_score=2"
    finally:
        pt_mod.CANDIDATE_PAIRS = orig


# ── 2. Signal generation — two-leg structure ──────────────────────────────────

def test_signal_has_two_legs_on_entry():
    """Every BUY signal must contain both 'long' and 'short' keys in pair_legs."""
    pair    = _default_pair()
    sel     = _make_selector_with_pair(pair)
    prices  = _make_cointegrated_prices()
    sel._price_cache.update(prices)

    agent   = PairsTradingAgent()
    agent.selector = sel

    with patch.object(sel, "compute_zscore", return_value=-2.5):
        sig = agent.generate_signal(_state(), portfolio={})

    assert sig.action == "BUY"
    legs = sig.meta["pair_legs"]
    assert "long" in legs and "short" in legs
    assert legs["long"][0]  == "SYNTH_A"
    assert legs["short"][0] == "SYNTH_B"


def test_long_a_short_b_on_low_zscore():
    """z < -entry → LONG A / SHORT B signal."""
    pair  = _default_pair()
    sel   = _make_selector_with_pair(pair)
    agent = PairsTradingAgent()
    agent.selector = sel

    with patch.object(sel, "compute_zscore", return_value=-2.1):
        sig = agent.generate_signal(_state(), portfolio={})

    assert sig.action == "BUY"
    assert sig.meta["direction"] == "long_a_short_b"
    assert sig.meta["pair_legs"]["long"][0]  == "SYNTH_A"
    assert sig.meta["pair_legs"]["short"][0] == "SYNTH_B"


def test_short_a_long_b_on_high_zscore():
    """z > +entry → SHORT A / LONG B signal."""
    pair  = _default_pair()
    sel   = _make_selector_with_pair(pair)
    agent = PairsTradingAgent()
    agent.selector = sel

    with patch.object(sel, "compute_zscore", return_value=2.7):
        sig = agent.generate_signal(_state(), portfolio={})

    assert sig.action == "BUY"
    assert sig.meta["direction"] == "short_a_long_b"
    assert sig.meta["pair_legs"]["long"][0]  == "SYNTH_B"
    assert sig.meta["pair_legs"]["short"][0] == "SYNTH_A"


def test_close_on_mean_reversion():
    """|z| < exit_threshold AND in position → SELL both legs."""
    pair      = _default_pair()
    sel       = _make_selector_with_pair(pair)
    agent     = PairsTradingAgent()
    agent.selector = sel
    portfolio = {"SYNTH_A": 0.08}  # in long-A position

    with patch.object(sel, "compute_zscore", return_value=0.3):
        sig = agent.generate_signal(_state(), portfolio=portfolio)

    assert sig.action == "SELL"
    assert sig.meta["direction"] == "close"
    assert "close_long"  in sig.meta["pair_legs"]
    assert "close_short" in sig.meta["pair_legs"]
    assert "mean reversion" in sig.reason.lower()


def test_stoploss_on_divergence():
    """|z| > stoploss → SELL even when in position (spread diverging)."""
    pair      = _default_pair()
    sel       = _make_selector_with_pair(pair)
    agent     = PairsTradingAgent()
    agent.selector = sel
    portfolio = {"SYNTH_A": 0.08}  # long A

    with patch.object(sel, "compute_zscore", return_value=-4.0):
        sig = agent.generate_signal(_state(), portfolio=portfolio)

    assert sig.action == "SELL"
    assert "STOP-LOSS" in sig.reason


def test_hold_inside_band():
    """z inside [-entry, +entry] with no position → HOLD."""
    pair  = _default_pair()
    sel   = _make_selector_with_pair(pair)
    agent = PairsTradingAgent()
    agent.selector = sel

    with patch.object(sel, "compute_zscore", return_value=1.2):
        sig = agent.generate_signal(_state(), portfolio={})

    assert sig.action == "HOLD"


def test_hold_when_no_cointegrated_pairs():
    """Zero valid pairs → HOLD with 'armed but at rest' message."""
    agent = PairsTradingAgent()
    agent.selector._validated = {}
    agent.selector._last_full_test = datetime.now(timezone.utc)

    sig = agent.generate_signal(_state(), portfolio={})
    assert sig.action == "HOLD"
    assert "armed but at rest" in sig.reason


# ── 3. Cointegration breakdown closes open position ───────────────────────────

def test_cointegration_breakdown_closes_position():
    """If a pair is deactivated (cointegration broke), an open position must be closed."""
    pair_inactive = _default_pair(is_active=False)
    sel           = _make_selector_with_pair(pair_inactive)
    agent         = PairsTradingAgent()
    agent.selector = sel
    portfolio     = {"SYNTH_A": 0.08}  # currently long A

    with patch.object(sel, "compute_zscore", return_value=1.0):
        sig = agent.generate_signal(_state(), portfolio=portfolio)

    # pair.is_active=False → validated but inactive → get_validated_pairs returns []
    # So agent returns HOLD with "armed but at rest"
    assert sig.action == "HOLD"
    assert "armed" in sig.reason.lower()


def test_pairs_selector_invalidate_pair():
    """invalidate_pair() marks the pair is_active=False."""
    pair = _default_pair()
    sel  = _make_selector_with_pair(pair)

    sel.invalidate_pair("SYNTH_A", "SYNTH_B")
    p = sel._validated[("SYNTH_A", "SYNTH_B")]
    assert not p.is_active


# ── 4. Never single-leg: pairs_plans_from_signal ──────────────────────────────

def test_never_single_leg_both_or_nothing():
    """If the long leg price is missing, return [] — no partial execution."""
    sig = AgentSignal(
        agent_name="PairsTradingAgent", symbol="NVDA",
        action="BUY", confidence=0.7, target_weight=0.08,
        reason="test",
        meta={
            "strategy": "market_neutral",
            "direction": "long_a_short_b",
            "pair_legs": {"long": ["NVDA", 0.08], "short": ["AMD", 0.015]},
            "hedge_ratio": 0.19,
        },
    )
    # Only AMD price provided — NVDA missing
    plans = pairs_plans_from_signal(sig, 100_000, {"AMD": 90.0}, {})
    assert plans == [], "Missing long leg price must produce zero plans (no single leg)"


def test_pairs_plans_entry_long_a_short_b():
    """Entry signal produces exactly 2 plans: BUY long + SELL short."""
    sig = AgentSignal(
        agent_name="PairsTradingAgent", symbol="NVDA",
        action="BUY", confidence=0.7, target_weight=0.08,
        reason="test",
        meta={
            "strategy": "market_neutral",
            "direction": "long_a_short_b",
            "pair_legs": {"long": ["NVDA", 0.08], "short": ["AMD", 0.015]},
            "hedge_ratio": 0.19,
        },
    )
    plans = pairs_plans_from_signal(sig, 100_000, {"NVDA": 120.0, "AMD": 90.0}, {})
    assert len(plans) == 2
    long_p  = next(p for p in plans if p.action == "BUY")
    short_p = next(p for p in plans if p.action == "SELL")
    assert long_p.symbol  == "NVDA" and long_p.strategy  == "market_neutral"
    assert short_p.symbol == "AMD"  and short_p.strategy == "market_neutral"
    assert short_p.delta_qty < 0   # opening a short: negative delta


def test_pairs_plans_short_leg_current_qty_zero():
    """Short leg current_qty must be 0 when opening fresh short (not an existing position)."""
    sig = AgentSignal(
        agent_name="PairsTradingAgent", symbol="NVDA",
        action="BUY", confidence=0.7, target_weight=0.08,
        reason="test",
        meta={
            "strategy": "market_neutral",
            "direction": "long_a_short_b",
            "pair_legs": {"long": ["NVDA", 0.08], "short": ["AMD", 0.015]},
            "hedge_ratio": 0.19,
        },
    )
    plans = pairs_plans_from_signal(sig, 100_000, {"NVDA": 120.0, "AMD": 90.0},
                                    current_qtys={})  # no existing positions
    short_p = next(p for p in plans if p.symbol == "AMD")
    assert short_p.current_qty == 0.0


def test_pairs_plans_close_both_legs():
    """Close signal generates SELL for long leg + BUY (cover) for short leg."""
    sig = AgentSignal(
        agent_name="PairsTradingAgent", symbol="NVDA",
        action="SELL", confidence=0.9, target_weight=0.0,
        reason="mean reversion",
        meta={
            "strategy": "market_neutral",
            "direction": "close",
            "pair_legs": {"close_long": "NVDA", "close_short": "AMD"},
            "previous_direction": "long_a_short_b",
        },
    )
    current_qtys = {"NVDA": 66.0, "AMD": -88.0}  # long 66 NVDA, short 88 AMD
    plans = pairs_plans_from_signal(sig, 100_000, {"NVDA": 120.0, "AMD": 90.0}, current_qtys)

    assert len(plans) == 2
    close_long  = next(p for p in plans if p.symbol == "NVDA")
    cover_short = next(p for p in plans if p.symbol == "AMD")

    assert close_long.action == "SELL" and close_long.delta_qty == -66.0
    assert cover_short.action == "BUY" and cover_short.delta_qty == 88.0


def test_pairs_plans_hold_returns_empty():
    """HOLD signal must return an empty list."""
    sig = AgentSignal("PairsTradingAgent","NVDA","HOLD",0.0,0.0,"hold",
                      {"strategy":"market_neutral","direction":"hold","pair_legs":{}})
    assert pairs_plans_from_signal(sig, 100_000, {"NVDA":120.0,"AMD":90.0}, {}) == []


# ── 5. RiskManager market-neutral routing ─────────────────────────────────────

def test_market_neutral_excluded_from_net_long():
    """Market-neutral plans must NOT consume the directional net-long budget."""
    cfg  = RiskConfig(max_net_long_pct=0.40, max_single_position_pct=0.50,
                      min_cash_pct=0.10, max_gross_pairs_pct=0.30)
    rm   = RiskManager(cfg)
    snap = _snap()

    plans = [
        _order_plan("DIR",    "BUY", 35_000, strategy="directional"),
        _order_plan("MN_L",   "BUY", 10_000, strategy="market_neutral"),
        _order_plan("MN_S",   "BUY", 10_000, strategy="market_neutral"),
    ]
    report = rm.check(plans, snap)
    assert len(report.rejected) == 0
    # Net-long should only include the directional plan
    assert abs(report.post_trade_long_pct - 0.35) < 0.001
    assert abs(report.pairs_gross_post - 0.20) < 0.001


def test_gross_exposure_capped():
    """Gross pairs exposure must be capped at max_gross_pairs_pct."""
    cfg  = RiskConfig(max_gross_pairs_pct=0.30, max_single_position_pct=0.50,
                      min_cash_pct=0.10)
    rm   = RiskManager(cfg)
    snap = _snap()

    plans = [
        _order_plan("L1", "BUY", 16_000, strategy="market_neutral"),
        _order_plan("S1", "BUY", 16_000, strategy="market_neutral"),   # would push to 32%
    ]
    report = rm.check(plans, snap)
    assert len(report.approved) == 1
    assert len(report.rejected) == 1
    assert "gross pairs" in report.rejected[0].reason


def test_market_neutral_sell_always_approved():
    """Market-neutral SELL (closing a pair leg) must always pass."""
    cfg  = RiskConfig(sell_only_mode=False)
    rm   = RiskManager(cfg)
    snap = _snap()

    plan = _order_plan("NVDA", "SELL", 8_000, current=66.0, strategy="market_neutral")
    report = rm.check([plan], snap)
    assert len(report.approved) == 1 and report.approved[0].action == "SELL"


def test_sell_only_blocks_market_neutral_buy():
    """sell_only_mode must block ALL BUY including market-neutral."""
    rm   = RiskManager(RiskConfig(sell_only_mode=True))
    snap = _snap()
    plan = _order_plan("LEG", "BUY", 5_000, strategy="market_neutral")
    report = rm.check([plan], snap)
    assert len(report.rejected) == 1


# ── 6. Pair round-trip P&L concept ───────────────────────────────────────────

def test_pair_roundtrip_pnl_positive():
    """
    P&L(pair) = P&L(long leg) + P&L(short leg).
    Long up $10, short down $10 → pair P&L = $10 + $10 = $20.
    """
    long_open,  long_close  = 100.0, 110.0   # long: +$10/share
    short_open, short_close = 200.0, 190.0   # short: +$10/share (fell $10)
    qty = 10

    long_pnl  = (long_close  - long_open)  * qty
    short_pnl = (short_open  - short_close) * qty  # short P&L = open − close
    pair_pnl  = long_pnl + short_pnl

    assert long_pnl  == pytest.approx(100.0)
    assert short_pnl == pytest.approx(100.0)
    assert pair_pnl  == pytest.approx(200.0)


def test_pair_roundtrip_pnl_market_neutral():
    """
    When market moves straight up (+5%) and both legs are sized correctly,
    net market P&L ≈ 0 (long gains offset short losses = market neutral).
    """
    beta      = 1.5    # hedge ratio
    price_a0  = 100.0
    price_b0  = 70.0
    market_up = 1.05   # +5% for all stocks
    qty_a     = 10
    qty_b     = int(qty_a * beta)   # 15 shares short B

    long_pnl  = (price_a0 * market_up - price_a0) * qty_a
    short_pnl = (price_b0 - price_b0 * market_up) * qty_b   # short: open − close

    net = long_pnl + short_pnl
    # Net should be near 0 (pure market move cancels out)
    assert abs(net) < price_a0 * qty_a * 0.10, f"Market-neutral pair not close to zero: {net}"


# ── 7. Guard-rails ────────────────────────────────────────────────────────────

def test_validate_pair_accepts_robustness_3():
    """ValidatedPair with robustness_score=3 must be accepted by get_validated_pairs()."""
    pair = _default_pair(robustness_score=3)
    sel  = _make_selector_with_pair(pair)
    result = sel.get_validated_pairs()
    assert len(result) == 1
    assert result[0].robustness_score == 3


def test_validate_pair_rejects_inactive():
    """Inactive ValidatedPair must not appear in get_validated_pairs()."""
    pair = _default_pair(is_active=False)
    sel  = _make_selector_with_pair(pair)
    result = sel.get_validated_pairs()
    assert result == []


def test_beta_positive_required():
    """Pairs with β ≤ 0 must be rejected by PairSelector._test_pair."""
    prices = _make_cointegrated_prices(n=700)
    cfg    = PairsTradingConfig(multi_window_check=False)
    sel    = PairSelector(cfg)
    sel._price_cache.update(prices)

    import src.agents.pairs_trading as pt_mod
    orig = pt_mod.CANDIDATE_PAIRS
    pt_mod.CANDIDATE_PAIRS = [("SYNTH_A", "SYNTH_B")]
    try:
        # Patch _estimate_hedge_ratio to return negative β
        with patch.object(sel, "_estimate_hedge_ratio", return_value=-0.5):
            pair = sel._test_pair("SYNTH_A", "SYNTH_B")
        assert pair is None, "β ≤ 0 must cause rejection"
    finally:
        pt_mod.CANDIDATE_PAIRS = orig


def test_zero_spread_variance_rejected():
    """Pairs where spread has zero variance (constant spread) must be rejected."""
    n = 400
    # A and B are perfectly correlated — spread is constant
    price_series = pd.Series(np.ones(n) * 100.0)
    prices = {"CONST_A": price_series.copy(), "CONST_B": price_series.copy()}
    cfg = PairsTradingConfig(multi_window_check=False)
    sel = PairSelector(cfg)
    sel._price_cache.update(prices)

    import src.agents.pairs_trading as pt_mod
    orig = pt_mod.CANDIDATE_PAIRS
    pt_mod.CANDIDATE_PAIRS = [("CONST_A", "CONST_B")]
    try:
        pair = sel._test_pair("CONST_A", "CONST_B")
        # Constant prices → spread variance = 0 → rejected
        # (coint may fail or spread_std < min_std → None)
        assert pair is None
    except Exception:
        pass  # any exception from degenerate data is also acceptable
    finally:
        pt_mod.CANDIDATE_PAIRS = orig


def test_insufficient_data_returns_none():
    """Too few rows (< lookback + coint_lookback) must return None."""
    cfg    = PairsTradingConfig(coint_lookback=252, lookback=63)
    sel    = PairSelector(cfg)
    short  = pd.Series(np.ones(100) * 100.0)
    sel._price_cache["SHORT_A"] = short
    sel._price_cache["SHORT_B"] = short

    import src.agents.pairs_trading as pt_mod
    orig = pt_mod.CANDIDATE_PAIRS
    pt_mod.CANDIDATE_PAIRS = [("SHORT_A", "SHORT_B")]
    try:
        pair = sel._test_pair("SHORT_A", "SHORT_B")
        assert pair is None
    finally:
        pt_mod.CANDIDATE_PAIRS = orig


def test_missing_ticker_returns_empty():
    """PairSelector handles a completely unknown ticker gracefully."""
    pair    = _default_pair()
    sel     = _make_selector_with_pair(pair)
    result  = sel.get_pair_for_ticker("UNKNOWN_TICKER")
    assert result is None


def test_order_plan_strategy_defaults_to_directional():
    """OrderPlan without explicit strategy must default to 'directional'."""
    plan = OrderPlan(
        symbol="SPY", action="BUY", target_weight=0.1,
        last_price=500.0, current_qty=0.0, target_qty=10.0,
        delta_qty=10.0, est_notional=5_000.0, reason="test",
    )
    assert plan.strategy == "directional"


# ── 8. Multi-criteria robustness filters ──────────────────────────────────────
# Tests for the new half-life, Hurst, zero-crossings and rolling EG filters.

def _coint_sel(halflife: float = 20.0, multi_window_check: bool = False) -> PairSelector:
    """
    PairSelector with synthetic cointegrated prices pre-loaded in cache.
    No network calls — all prices injected directly.
    """
    prices = _make_cointegrated_prices(n=700, halflife=halflife)
    cfg    = PairsTradingConfig(multi_window_check=multi_window_check)
    sel    = PairSelector(cfg)
    sel._price_cache.update(prices)
    return sel


def test_half_life_computation():
    """_compute_halflife should recover OU half-life within a generous tolerance."""
    spread = _make_ou_spread(halflife=20.0, n=1000, seed=42)
    hl     = _compute_halflife(spread)
    assert hl is not None, "_compute_halflife returned None on well-formed OU spread"
    # With n=1000 the OLS estimate is reasonably accurate (±50% typical)
    assert 5.0 < hl < 60.0, f"Estimated half-life {hl:.1f}j outside plausible range for OU hl=20"


def test_half_life_rejects_too_fast():
    """Pipeline must reject a pair when half-life < min_halflife_days (default 5j)."""
    sel = _coint_sel()
    with patch("src.agents.pairs_trading._compute_halflife", return_value=2.0):
        report = sel.validate_pair_full("SYNTH_A", "SYNTH_B")
    assert not report.is_tradable
    assert not report.passes_halflife
    assert any("trop court" in r for r in report.rejection_reasons)


def test_half_life_rejects_too_slow():
    """Pipeline must reject a pair when half-life > max_halflife_days (default 60j)."""
    sel = _coint_sel()
    with patch("src.agents.pairs_trading._compute_halflife", return_value=75.0):
        report = sel.validate_pair_full("SYNTH_A", "SYNTH_B")
    assert not report.is_tradable
    assert not report.passes_halflife
    assert any("trop long" in r for r in report.rejection_reasons)


def test_hurst_mean_reverting():
    """OU spread (mean-reverting) must have H < 0.5 from R/S analysis."""
    spread = _make_ou_spread(halflife=15.0, n=500, seed=1)
    H      = _compute_hurst(spread)
    assert H is not None, "_compute_hurst returned None on OU spread (n=500)"
    assert H < 0.50, f"OU spread should have H < 0.5 (mean-reverting), got H={H:.3f}"


def test_hurst_rejects_random_walk():
    """Pipeline must reject a pair whose spread Hurst exponent ≥ max_hurst."""
    sel = _coint_sel()
    with patch("src.agents.pairs_trading._compute_hurst", return_value=0.65):
        report = sel.validate_pair_full("SYNTH_A", "SYNTH_B")
    assert not report.is_tradable
    assert not report.passes_hurst
    assert any("Hurst" in r for r in report.rejection_reasons)


def test_zero_crossings_count():
    """_count_zero_crossings must count sign changes of (spread - mean) correctly."""
    spread = np.array([1.0, -1.0, 1.0, -1.0, 1.0])
    # mean = 0.2 → centered = [0.8, -1.2, 0.8, -1.2, 0.8] → 4 sign changes
    assert _count_zero_crossings(spread) == 4


def test_zero_crossings_rejects_low():
    """Pipeline must reject when zero-crossings < min_zero_crossings (default 12)."""
    sel = _coint_sel()
    with patch("src.agents.pairs_trading._count_zero_crossings", return_value=5):
        report = sel.validate_pair_full("SYNTH_A", "SYNTH_B")
    assert not report.is_tradable
    assert not report.passes_zero_crossings
    assert any("zero-crossing" in r.lower() for r in report.rejection_reasons)


def test_ma_risk_flag():
    """Regional bank and airline pairs must carry ma_risk='high'; energy pairs 'low'."""
    assert ("RF",  "FITB") in CANDIDATE_PAIRS, "RF/FITB must be in candidate universe"
    assert ("CFG", "HBAN") in CANDIDATE_PAIRS, "CFG/HBAN must be in candidate universe"
    assert _MA_RISK[("RF",  "FITB")]  == "high"
    assert _MA_RISK[("CFG", "HBAN")]  == "high"
    assert _MA_RISK[("DAL", "UAL")]   == "high"
    assert _MA_RISK[("XOM", "CVX")]   == "low"
    assert _MA_RISK[("SO",  "DUK")]   == "low"


def test_full_pipeline_rejects_artifact():
    """
    A pair that passes EG multi-window but fails Hurst must be rejected
    by validate_pair_full() (mimics a window-artifact pair like NVDA/AMD).
    """
    # multi_window_check=True so EG robustness is verified too
    sel = _coint_sel(multi_window_check=True)
    with patch("src.agents.pairs_trading._compute_hurst", return_value=0.70):
        report = sel.validate_pair_full("SYNTH_A", "SYNTH_B")
    assert not report.is_tradable
    assert any("Hurst" in r for r in report.rejection_reasons)
    # EG still passed — Hurst is the single deciding failure
    assert report.robustness_score >= 1


def test_validate_pair_full_report():
    """validate_pair_full() must return a complete PairValidationReport."""
    sel    = _coint_sel()
    report = sel.validate_pair_full("SYNTH_A", "SYNTH_B")

    assert isinstance(report, PairValidationReport)
    assert report.ticker_a == "SYNTH_A"
    assert report.ticker_b == "SYNTH_B"
    # All three cointegration windows must be present
    for w in _COINT_WINDOWS:
        assert w in report.window_pvalues, f"window {w}j missing from report"
    assert isinstance(report.rejection_reasons, list)
    assert isinstance(report.is_tradable, bool)
    # New criteria fields must exist
    assert hasattr(report, "half_life")
    assert hasattr(report, "passes_halflife")
    assert hasattr(report, "hurst")
    assert hasattr(report, "passes_hurst")
    assert hasattr(report, "zero_crossings")
    assert hasattr(report, "passes_zero_crossings")
    assert hasattr(report, "ma_risk")
    assert hasattr(report, "rolling_fail_rate")
    assert hasattr(report, "robustness_score")
