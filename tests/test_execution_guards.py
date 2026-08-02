"""
Tests de la garde d'exécution (src/execution/guards.py).

Régression principale couverte : avant le 2026-08-02, un plan dont
`est_notional > max_notional` était supprimé par un `continue` silencieux.
Sur le run du 2026-07-23 (NetLiq 1 026 098 $, MAX_NOTIONAL_PCT=0.02 →
plafond 20 522 $), les 6 plans approuvés valaient 41 k$ à 154 k$ :
100 % des ordres étaient jetés sans trace.
"""
from __future__ import annotations

import pytest

from src.execution.guards import build_execution_plan, _is_risk_reducing
from src.execution.planner import OrderPlan


NETLIQ = 1_000_000.0


def _plan(
    symbol="AAPL", action="BUY", delta=100.0, price=100.0, current=0.0,
    confidence=0.5, strategy="directional", target_qty=None,
) -> OrderPlan:
    tq = current + delta if target_qty is None else target_qty
    return OrderPlan(
        symbol=symbol, action=action, target_weight=0.10, last_price=price,
        current_qty=current, target_qty=tq, delta_qty=delta,
        est_notional=abs(delta) * price, reason="test",
        strategy=strategy, confidence=confidence,
    )


def _build(plans, *, pct=0.02, max_orders=5, buffer_bps=5.0):
    return build_execution_plan(
        plans, net_liquidation=NETLIQ, max_notional_pct=pct,
        max_orders=max_orders, limit_buffer_bps=buffer_bps,
    )


# ── Régression : le drop silencieux ───────────────────────────────────────────

class TestNoSilentDrop:
    def test_oversized_buy_is_resized_not_dropped(self):
        """Le bug d'origine : un BUY de 100 k$ sous plafond 20 k$ partait à la poubelle."""
        # 1000 titres × 100 $ = 100 000 $, plafond = 20 000 $
        res = _build([_plan(delta=1000.0, price=100.0)])
        assert len(res.candidates) == 1, "l'ordre doit survivre, redimensionné"
        c = res.candidates[0]
        assert c.qty == 200, "20 000 $ / 100 $ = 200 titres"
        assert c.resized is True
        assert c.requested_qty == 1000

    def test_resize_never_exceeds_cap(self):
        res = _build([_plan(delta=1000.0, price=100.0)])
        assert res.candidates[0].notional <= res.max_notional + 1e-6

    def test_resize_is_recorded_with_figures(self):
        res = _build([_plan(delta=1000.0, price=100.0)])
        adj = [a for a in res.adjustments if a.kind == "resized"]
        assert len(adj) == 1
        assert adj[0].requested_qty == 1000 and adj[0].final_qty == 200
        assert adj[0].requested_notional == pytest.approx(100_000.0)
        assert adj[0].final_notional == pytest.approx(20_000.0)

    def test_july23_scenario_all_six_plans_survive(self):
        """Reproduction du run réel du 2026-07-23 : 6 plans, 0 ordre envoyé."""
        netliq = 1_026_098.17
        plans = [
            _plan("AAPL", delta=167.0,   price=320.67,  confidence=0.90),
            _plan("SPY",  delta=161.0,   price=737.28,  confidence=0.85),
            _plan("GOOGL", delta=130.0,  price=319.46,  confidence=0.70),
            _plan("TLT",  "SELL", delta=-1852.0, price=83.095, confidence=0.60, strategy="cta_trend"),
            _plan("UUP",  delta=5388.0,  price=28.565, confidence=0.55, strategy="cta_trend"),
            _plan("DBC",  delta=5038.0,  price=30.55,  confidence=0.65, strategy="cta_trend"),
        ]
        res = build_execution_plan(
            plans, net_liquidation=netliq, max_notional_pct=0.02,
            max_orders=5, limit_buffer_bps=5.0,
        )
        # Ancien comportement : 6 plans → 0 ordre. Nouveau : 6 plans → 5 ordres.
        assert len(res.candidates) == 5, "5 ordres envoyés (quota), tous redimensionnés"
        assert all(c.notional <= res.max_notional + 1e-6 for c in res.candidates)

        # Aucun rejet pour cause de taille : le seul écarté l'est par quota,
        # et c'est bien le moins convaincu (UUP, conf=0.55).
        dropped = [a for a in res.adjustments if a.kind == "dropped"]
        assert len(dropped) == 1
        assert dropped[0].symbol == "UUP"
        assert "quota" in dropped[0].reason
        assert not any("plafond unitaire" in a.reason for a in dropped)

        # AAPL, la plus forte conviction, part en tête des augmentations.
        assert res.candidates[0].plan.symbol == "AAPL"

    def test_share_pricier_than_cap_is_dropped_with_explicit_reason(self):
        """Seul cas de rejet légitime : 1 titre coûte déjà plus que le plafond."""
        res = _build([_plan(delta=1.0, price=50_000.0)], pct=0.001)  # plafond 1 000 $
        assert res.candidates == []
        assert res.n_dropped == 1
        assert "MAX_NOTIONAL_PCT" in res.adjustments[0].reason


# ── Asymétrie risque : on ne rationne pas la réduction ────────────────────────

class TestRiskReducingPriority:
    def test_exit_is_never_resized(self):
        """Une sortie complète part en entier, même très au-dessus du plafond."""
        res = _build([_plan("SPY", "SELL", delta=-1000.0, price=700.0, current=1000.0)])
        c = res.candidates[0]
        assert c.qty == 1000 and c.resized is False
        assert c.risk_reducing is True
        assert c.notional > res.max_notional  # 700 k$ >> 20 k$ : c'est voulu

    def test_short_cover_is_risk_reducing(self):
        res = _build([_plan("TLT", "BUY", delta=500.0, price=83.0, current=-500.0,
                            strategy="cta_trend")])
        assert res.candidates[0].risk_reducing is True
        assert res.candidates[0].resized is False

    def test_exits_ignore_the_order_quota(self):
        """8 sorties avec max_orders=1 : les 8 doivent partir."""
        plans = [
            _plan(f"S{i}", "SELL", delta=-100.0, price=100.0, current=100.0)
            for i in range(8)
        ]
        res = _build(plans, max_orders=1)
        assert len(res.candidates) == 8
        assert res.n_dropped == 0

    def test_exits_come_first_in_the_queue(self):
        plans = [
            _plan("BUY_HI", delta=100.0, price=100.0, confidence=0.99),
            _plan("EXIT", "SELL", delta=-100.0, price=100.0, current=100.0, confidence=0.01),
        ]
        res = _build(plans)
        assert res.candidates[0].plan.symbol == "EXIT"

    def test_stop_loss_survives_a_quota_of_zero(self):
        """Cas limite : quota nul ne doit pas bloquer un stop-loss."""
        plans = [
            _plan("AAPL", delta=100.0, price=100.0, confidence=0.9),
            _plan("NVDA", "SELL", delta=-50.0, price=200.0, current=50.0, confidence=1.0),
        ]
        res = _build(plans, max_orders=0)
        assert [c.plan.symbol for c in res.candidates] == ["NVDA"]


# ── Troncature ordonnée par conviction ────────────────────────────────────────

class TestConvictionOrdering:
    def test_quota_keeps_highest_conviction(self):
        plans = [
            _plan("LOW",  delta=10.0, price=100.0, confidence=0.20),
            _plan("HIGH", delta=10.0, price=100.0, confidence=0.95),
            _plan("MID",  delta=10.0, price=100.0, confidence=0.60),
        ]
        res = _build(plans, max_orders=2)
        assert [c.plan.symbol for c in res.candidates] == ["HIGH", "MID"]
        assert res.n_dropped == 1
        assert res.adjustments[0].symbol == "LOW"

    def test_watchlist_order_no_longer_decides(self):
        """Régression : avant, candidates[:N] gardait le premier de la WATCHLIST."""
        plans = [
            _plan("AAPL_FIRST", delta=10.0, price=100.0, confidence=0.10),
            _plan("LLY_LAST",   delta=10.0, price=100.0, confidence=0.90),
        ]
        res = _build(plans, max_orders=1)
        assert res.candidates[0].plan.symbol == "LLY_LAST"

    def test_notional_breaks_confidence_ties(self):
        plans = [
            _plan("SMALL", delta=10.0,  price=100.0, confidence=0.50),
            _plan("BIG",   delta=100.0, price=100.0, confidence=0.50),
        ]
        res = _build(plans, max_orders=1)
        assert res.candidates[0].plan.symbol == "BIG"

    def test_dropped_by_quota_states_the_conviction(self):
        plans = [_plan("A", delta=10.0, price=100.0, confidence=0.9),
                 _plan("B", delta=10.0, price=100.0, confidence=0.3)]
        res = _build(plans, max_orders=1)
        assert "conviction 0.30" in res.adjustments[0].reason


# ── Anti-short accidentel ─────────────────────────────────────────────────────

class TestShortGuard:
    def test_directional_sell_clipped_to_holdings(self):
        res = _build([_plan("AAPL", "SELL", delta=-300.0, price=100.0, current=100.0)])
        assert res.candidates[0].qty == 100
        assert any(a.kind == "clipped" for a in res.adjustments)

    def test_directional_sell_with_no_holdings_emits_nothing(self):
        res = _build([_plan("AAPL", "SELL", delta=-300.0, price=100.0, current=0.0)])
        assert res.candidates == []

    def test_market_neutral_short_leg_allowed(self):
        res = _build([_plan("XLE", "SELL", delta=-100.0, price=100.0, current=0.0,
                            strategy="market_neutral")])
        assert len(res.candidates) == 1 and res.candidates[0].qty == 100

    def test_cta_can_open_a_short(self):
        """Régression : `!= market_neutral` clipait aussi le CTA, qui shorte pourtant."""
        res = _build([_plan("TLT", "SELL", delta=-200.0, price=80.0, current=0.0,
                            strategy="cta_trend")])
        assert len(res.candidates) == 1
        assert res.candidates[0].qty == 200
        assert not any(a.kind == "clipped" for a in res.adjustments)

    def test_cta_reversal_is_not_clipped_to_holdings(self):
        """long 100 → short 200 = SELL 300. L'ancien code n'en vendait que 100."""
        res = _build([_plan("TLT", "SELL", delta=-300.0, price=80.0, current=100.0,
                            strategy="cta_trend", target_qty=-200.0)], pct=1.0)
        assert res.candidates[0].qty == 300


# ── Mécanique de base ─────────────────────────────────────────────────────────

class TestBasics:
    def test_hold_produces_no_order_and_no_adjustment(self):
        res = _build([_plan("AAPL", "HOLD", delta=0.0, price=100.0)])
        assert res.candidates == [] and res.adjustments == []

    def test_zero_price_is_dropped_not_crashed(self):
        res = _build([_plan("AAPL", delta=100.0, price=0.0)])
        assert res.candidates == [] and res.n_dropped == 1

    def test_limit_buffer_direction(self):
        buy  = _build([_plan("A", "BUY",  delta=1.0,  price=100.0)], buffer_bps=10.0)
        sell = _build([_plan("B", "SELL", delta=-1.0, price=100.0, current=1.0)], buffer_bps=10.0)
        assert buy.candidates[0].limit_price == pytest.approx(100.10)
        assert sell.candidates[0].limit_price == pytest.approx(99.90)

    def test_report_is_not_misleading_when_everything_was_cut(self):
        """L'ancien message disait « filtered / HOLD » alors que rien n'était HOLD."""
        res = _build([_plan(delta=1.0, price=50_000.0)], pct=0.001)
        text = res.render()
        assert "HOLD" not in text
        assert "écarté" in text and "50,000" in text or "écarté" in text


class TestRiskReducingClassifier:
    @pytest.mark.parametrize("current,delta,expected", [
        (0.0,    100.0,  False),  # ouverture long
        (0.0,   -100.0,  False),  # ouverture short
        (100.0, -100.0,  True),   # sortie complète
        (100.0,  -40.0,  True),   # allègement
        (100.0,   50.0,  False),  # renforcement
        (-50.0,   50.0,  True),   # couverture de short
        (-50.0,  -50.0,  False),  # aggravation du short
        (100.0, -300.0,  False),  # reversal : nouvelle exposition opposée
    ])
    def test_classification(self, current, delta, expected):
        assert _is_risk_reducing(current, delta) is expected
