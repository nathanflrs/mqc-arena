"""
Integration tests for PairsTradingAgent routing in runner.py.

Verifies that a pairs signal produces exactly TWO plans (both-or-nothing),
and that the cancel-if-partner-fails guard is baked into pairs_plans_from_signal.
"""
from __future__ import annotations

import pytest

from src.agents.base import AgentSignal
from src.execution.planner import pairs_plans_from_signal


def _pairs_signal(
    action: str = "BUY",
    direction: str = "long_a_short_b",
    long_ticker: str = "XOM",
    long_weight: float = 0.10,
    short_ticker: str = "CVX",
    short_weight: float = 0.10,
    hedge_ratio: float = 0.85,
) -> AgentSignal:
    if action == "BUY":
        pair_legs = {
            "long":  (long_ticker,  long_weight),
            "short": (short_ticker, short_weight),
        }
    else:
        pair_legs = {
            "close_long":  long_ticker,
            "close_short": short_ticker,
        }
    return AgentSignal(
        agent_name="PairsTradingAgent",
        symbol=long_ticker,
        action=action,
        confidence=0.72,
        target_weight=long_weight,
        reason="test pairs signal",
        meta={
            "strategy":    "market_neutral",
            "direction":   direction if action == "BUY" else "close",
            "pair_legs":   pair_legs,
            "hedge_ratio": hedge_ratio,
            "zscore":      2.1,
        },
    )


PRICES = {"XOM": 110.0, "CVX": 130.0}
QTYS   = {"XOM": 0.0,   "CVX": 0.0}
NL     = 100_000.0


# ── Entry tests ───────────────────────────────────────────────────────────────

class TestPairsPlansEntry:

    def test_entry_produces_exactly_two_plans(self):
        sig = _pairs_signal(action="BUY", direction="long_a_short_b")
        plans = pairs_plans_from_signal(sig, NL, PRICES, QTYS)
        assert len(plans) == 2, f"Expected 2 plans, got {len(plans)}"

    def test_entry_long_leg_is_buy(self):
        sig = _pairs_signal(action="BUY", direction="long_a_short_b")
        plans = pairs_plans_from_signal(sig, NL, PRICES, QTYS)
        long_plans = [p for p in plans if p.action == "BUY"]
        assert len(long_plans) == 1
        assert long_plans[0].symbol == "XOM"

    def test_entry_short_leg_is_sell(self):
        sig = _pairs_signal(action="BUY", direction="long_a_short_b")
        plans = pairs_plans_from_signal(sig, NL, PRICES, QTYS)
        short_plans = [p for p in plans if p.action == "SELL"]
        assert len(short_plans) == 1
        assert short_plans[0].symbol == "CVX"

    def test_entry_both_legs_tagged_market_neutral(self):
        sig = _pairs_signal(action="BUY", direction="long_a_short_b")
        plans = pairs_plans_from_signal(sig, NL, PRICES, QTYS)
        for p in plans:
            assert p.strategy == "market_neutral", f"{p.symbol} missing market_neutral tag"

    def test_entry_reversed_direction_swaps_legs(self):
        """short_a_long_b → XOM is short leg, CVX is long leg."""
        sig = _pairs_signal(action="BUY", direction="short_a_long_b",
                            long_ticker="CVX", long_weight=0.10,
                            short_ticker="XOM", short_weight=0.10)
        plans = pairs_plans_from_signal(sig, NL, PRICES, QTYS)
        assert len(plans) == 2
        buys  = {p.symbol for p in plans if p.action == "BUY"}
        sells = {p.symbol for p in plans if p.action == "SELL"}
        assert buys  == {"CVX"}
        assert sells == {"XOM"}

    def test_entry_short_leg_delta_is_negative(self):
        sig = _pairs_signal(action="BUY", direction="long_a_short_b")
        plans = pairs_plans_from_signal(sig, NL, PRICES, QTYS)
        short_plan = next(p for p in plans if p.action == "SELL")
        assert short_plan.delta_qty < 0


# ── Both-or-nothing guard ─────────────────────────────────────────────────────

class TestPairsBothOrNothing:

    def test_missing_long_price_returns_empty(self):
        sig    = _pairs_signal(action="BUY", direction="long_a_short_b")
        prices = {"CVX": 130.0}          # XOM (long leg) missing
        plans  = pairs_plans_from_signal(sig, NL, prices, QTYS)
        assert plans == [], "Should cancel both legs when long price is unknown"

    def test_missing_short_price_returns_empty(self):
        sig    = _pairs_signal(action="BUY", direction="long_a_short_b")
        prices = {"XOM": 110.0}          # CVX (short leg) missing
        plans  = pairs_plans_from_signal(sig, NL, prices, QTYS)
        assert plans == [], "Should cancel both legs when short price is unknown"

    def test_long_price_zero_returns_empty(self):
        sig    = _pairs_signal(action="BUY", direction="long_a_short_b")
        prices = {"XOM": 0.0, "CVX": 130.0}
        plans  = pairs_plans_from_signal(sig, NL, prices, QTYS)
        assert plans == [], "Zero price must cancel both legs"

    def test_insufficient_notional_returns_empty(self):
        """Very small account → qty < 1 for both legs → cancel."""
        sig   = _pairs_signal(action="BUY", direction="long_a_short_b",
                               long_weight=0.0001, short_weight=0.0001)
        plans = pairs_plans_from_signal(sig, 1_000.0, PRICES, QTYS)
        assert plans == []

    def test_hold_returns_empty(self):
        sig   = _pairs_signal(action="HOLD", direction="long_a_short_b")
        plans = pairs_plans_from_signal(sig, NL, PRICES, QTYS)
        assert plans == []


# ── Close tests ───────────────────────────────────────────────────────────────

class TestPairsPlansClose:

    def _close_signal(self) -> AgentSignal:
        return _pairs_signal(action="SELL", direction="close")

    def test_close_with_open_long_and_short_produces_two_plans(self):
        """Held long XOM + short CVX → SELL XOM + BUY CVX (cover)."""
        qtys  = {"XOM": 90.0, "CVX": -76.0}   # short CVX stored as negative
        plans = pairs_plans_from_signal(self._close_signal(), NL, PRICES, qtys)
        assert len(plans) == 2

    def test_close_long_leg_is_sell(self):
        qtys  = {"XOM": 90.0, "CVX": -76.0}
        plans = pairs_plans_from_signal(self._close_signal(), NL, PRICES, qtys)
        sell  = next(p for p in plans if p.symbol == "XOM")
        assert sell.action == "SELL"

    def test_close_short_leg_is_buy_cover(self):
        qtys  = {"XOM": 90.0, "CVX": -76.0}
        plans = pairs_plans_from_signal(self._close_signal(), NL, PRICES, qtys)
        cover = next(p for p in plans if p.symbol == "CVX")
        assert cover.action == "BUY"

    def test_close_missing_price_returns_empty(self):
        qtys  = {"XOM": 90.0, "CVX": -76.0}
        plans = pairs_plans_from_signal(self._close_signal(), NL, {"XOM": 110.0}, qtys)
        assert plans == []
