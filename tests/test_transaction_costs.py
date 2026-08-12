"""
Tests for transaction cost estimation in order plans.
"""
from __future__ import annotations

import pytest
from src.execution.planner import _compute_tx_cost, OrderPlan, plan_from_signal
from src.agents.base import AgentSignal


# ── _compute_tx_cost ──────────────────────────────────────────────────────────

class TestComputeTxCost:

    def test_min_order_applies_small_qty(self):
        # 10 shares × $0.005 = $0.05 < $1 min → commission is $1
        cost = _compute_tx_cost(10, 100.0)
        # bid-ask: 10 × 100 × 0.0005 = $0.50
        # commission: max($1, 10×$0.005) = max($1, $0.05) = $1
        assert cost == pytest.approx(1.0 + 0.50, abs=0.01)

    def test_large_order_exceeds_min(self):
        # 1000 shares × $0.005 = $5 > $1 min
        cost = _compute_tx_cost(1000, 50.0)
        # commission: $5, bid-ask: 1000 × 50 × 0.0005 = $25
        assert cost == pytest.approx(5.0 + 25.0, abs=0.01)

    def test_negative_qty_handled(self):
        # SELL: qty is negative — abs() should apply
        cost_sell = _compute_tx_cost(-100, 200.0)
        cost_buy  = _compute_tx_cost(100, 200.0)
        assert cost_sell == pytest.approx(cost_buy)

    def test_cost_is_non_negative(self):
        assert _compute_tx_cost(0, 100.0) >= 0.0
        assert _compute_tx_cost(1, 1.0) > 0.0

    def test_cost_scales_with_price(self):
        low  = _compute_tx_cost(100, 10.0)
        high = _compute_tx_cost(100, 100.0)
        assert high > low  # bid-ask cost grows with notional


# ── OrderPlan.est_cost_usd ────────────────────────────────────────────────────

def _buy_signal(symbol: str, conf: float = 0.80, tw: float = 0.10) -> AgentSignal:
    return AgentSignal(
        agent_name="TestAgent",
        symbol=symbol,
        action="BUY",
        confidence=conf,
        target_weight=tw,
        reason="test",
    )


def _sell_signal(symbol: str) -> AgentSignal:
    return AgentSignal(
        agent_name="TestAgent",
        symbol=symbol,
        action="SELL",
        confidence=0.90,
        target_weight=0.0,
        reason="test sell",
    )


class TestOrderPlanCost:

    def test_buy_plan_has_cost(self):
        sig  = _buy_signal("AAPL")
        plan = plan_from_signal(sig, net_liquidation=100_000, last_price=200.0, current_qty=0.0)
        assert plan.est_cost_usd > 0

    def test_sell_plan_has_cost(self):
        sig  = _sell_signal("AAPL")
        plan = plan_from_signal(sig, net_liquidation=100_000, last_price=200.0, current_qty=50.0)
        assert plan.action == "SELL"
        assert plan.est_cost_usd > 0

    def test_hold_plan_has_zero_cost(self):
        sig = AgentSignal(
            agent_name="TestAgent", symbol="AAPL", action="HOLD",
            confidence=0.5, target_weight=0.0, reason="hold",
        )
        plan = plan_from_signal(sig, net_liquidation=100_000, last_price=200.0, current_qty=0.0)
        assert plan.est_cost_usd == 0.0

    def test_already_at_target_zero_cost(self):
        sig  = _buy_signal("AAPL", tw=0.10)
        # netliq=100k, tw=0.10 → target=$10k/$200=50 shares. current=50 → no trade
        plan = plan_from_signal(sig, net_liquidation=100_000, last_price=200.0, current_qty=50.0)
        assert plan.action == "HOLD"
        assert plan.est_cost_usd == 0.0

    def test_cost_reasonable_magnitude(self):
        # $100k × 10% = $10k trade at $200/share = 50 shares
        sig  = _buy_signal("AAPL", tw=0.10)
        plan = plan_from_signal(sig, net_liquidation=100_000, last_price=200.0, current_qty=0.0)
        # 50 shares: commission = max($1, 50×$0.005=$0.25) = $1
        #            bid-ask    = 50 × 200 × 0.0005 = $5
        # total ≈ $6
        assert 1.0 < plan.est_cost_usd < 50.0  # sanity bounds

    def test_default_est_cost_is_zero_for_manual_plan(self):
        """OrderPlan created without est_cost_usd defaults to 0."""
        plan = OrderPlan(
            symbol="TEST", action="BUY", target_weight=0.10,
            last_price=100.0, current_qty=0.0, target_qty=10.0,
            delta_qty=10.0, est_notional=1000.0, reason="manual",
        )
        assert plan.est_cost_usd == 0.0
