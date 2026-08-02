"""
Tests de l'allocation du budget par le RiskManager (point A, 2026-08-02).

Trois défauts corrigés :
  A1 — l'exposition existante était sommée sur `plans`, pas sur le portefeuille :
       une position détenue sans plan ce run était invisible au net long
       (sous-estimation, donc plafonds trop permissifs).
  A2 — les plans étaient évalués dans l'ordre de la WATCHLIST : sac-à-dos glouton
       premier-arrivé-premier-servi, et les SELL évalués après les BUY ne
       libéraient leur capacité que trop tard.
  A3 — rejet binaire : un dépassement marginal éliminait le plan entier.
"""
from __future__ import annotations

import pytest

from src.broker.portfolio import PortfolioSnapshot
from src.execution.planner import OrderPlan
from src.risk.manager import MIN_TRIM_FRACTION, RiskConfig, RiskManager


def _snap(netliq=1_000_000.0, cash=1_000_000.0, positions=None):
    return PortfolioSnapshot(net_liquidation=netliq, cash=cash, positions=positions or {})


def _buy(symbol, notional, price=100.0, conf=0.5, current=0.0):
    qty = notional / price
    return OrderPlan(
        symbol=symbol, action="BUY", target_weight=notional / 1_000_000.0,
        last_price=price, current_qty=current, target_qty=current + qty,
        delta_qty=qty, est_notional=notional, reason="test", confidence=conf,
    )


def _sell(symbol, qty, price=100.0, conf=0.5):
    return OrderPlan(
        symbol=symbol, action="SELL", target_weight=0.0, last_price=price,
        current_qty=qty, target_qty=0.0, delta_qty=-qty,
        est_notional=qty * price, reason="test", confidence=conf,
    )


def _mgr(**kw):
    cfg = dict(max_net_long_pct=0.40, max_single_position_pct=0.20, min_cash_pct=0.0)
    cfg.update(kw)
    return RiskManager(RiskConfig(**cfg))


# ── A1 — exposition détenue mais non planifiée ────────────────────────────────

class TestUnplannedExposure:
    def test_held_position_without_plan_counts_toward_net_long(self):
        """
        On détient 3 000 GS × 100 $ = 300 k$ mais GS ne produit aucun plan
        (données manquantes, ou aucun gagnant d'arène). Le plafond net long
        est 40 % de 1 M$ = 400 k$. Il ne reste donc que 100 k$ de capacité.
        """
        report = _mgr().check(
            [_buy("AAPL", 200_000)],
            _snap(positions={"GS": 3_000.0}),
            price_map={"GS": 100.0},
        )
        assert report.pre_trade_long_pct == pytest.approx(0.30)
        assert report.post_trade_long_pct <= 0.40 + 1e-6
        assert report.approved[0].est_notional == pytest.approx(100_000.0)

    def test_regression_unplanned_exposure_was_invisible(self):
        """Sans price_map, l'ancien comportement : GS invisible, 200k approuvés."""
        report = _mgr().check([_buy("AAPL", 200_000)], _snap(positions={"GS": 3_000.0}))
        assert report.pre_trade_long_pct == pytest.approx(0.0)
        assert "GS" in report.unpriced_positions, \
            "une position non valorisable doit être signalée, pas ignorée"

    def test_planned_symbol_not_double_counted(self):
        """AAPL détenu ET planifié : compté une seule fois, via le plan."""
        report = _mgr().check(
            [_buy("AAPL", 50_000, current=1_000.0)],
            _snap(positions={"AAPL": 1_000.0}),
            price_map={"AAPL": 100.0},
        )
        assert report.pre_trade_long_pct == pytest.approx(0.10)  # 1000×100 = 100k

    def test_cta_position_excluded_from_net_long(self):
        cta = OrderPlan(
            symbol="TLT", action="HOLD", target_weight=0.0, last_price=80.0,
            current_qty=1_000.0, target_qty=1_000.0, delta_qty=0.0,
            est_notional=0.0, reason="t", strategy="cta_trend",
        )
        report = _mgr().check([cta], _snap(positions={"TLT": 1_000.0}),
                              price_map={"TLT": 80.0})
        assert report.pre_trade_long_pct == pytest.approx(0.0)


# ── A2 — ordre d'évaluation ───────────────────────────────────────────────────

class TestEvaluationOrder:
    def test_highest_conviction_served_first(self):
        """
        Régression NVDA du 23/07 : capacité 400 k$, deux BUY de 300 k$.
        L'ordre WATCHLIST servait le premier venu ; on sert désormais le plus
        convaincu, et le second est rogné avec le reliquat.
        """
        report = _mgr(max_single_position_pct=0.40).check(
            [_buy("LOW_CONV", 300_000, conf=0.30),
             _buy("NVDA",     300_000, conf=0.95)],
            _snap(),
        )
        by_sym = {p.symbol: p.est_notional for p in report.approved}
        assert by_sym["NVDA"] == pytest.approx(300_000.0), "le plus convaincu passe entier"
        assert by_sym["LOW_CONV"] == pytest.approx(100_000.0), "le reste prend le reliquat"

    def test_sell_frees_budget_before_buys_are_evaluated(self):
        """
        On détient 400 k$ (plafond atteint). Un SELL de 200 k$ libère la moitié.
        Évalué après les BUY, ce SELL arrivait trop tard : le BUY était rejeté.
        """
        report = _mgr().check(
            [_buy("AAPL", 150_000, conf=0.9), _sell("GS", 4_000.0, price=100.0)],
            _snap(positions={"GS": 4_000.0}),
            price_map={"GS": 100.0},
        )
        assert report.pre_trade_long_pct == pytest.approx(0.40)
        buys = [p for p in report.approved if p.action == "BUY"]
        assert len(buys) == 1
        assert buys[0].est_notional == pytest.approx(150_000.0), \
            "le SELL doit avoir libéré la capacité avant l'évaluation du BUY"

    def test_ordering_is_deterministic_on_ties(self):
        plans = [_buy("ZZZ", 100_000, conf=0.5), _buy("AAA", 100_000, conf=0.5)]
        first  = _mgr().check(list(plans),           _snap())
        second = _mgr().check(list(reversed(plans)), _snap())
        assert [p.symbol for p in first.approved] == [p.symbol for p in second.approved]


# ── A3 — rognage plutôt que rejet binaire ─────────────────────────────────────

class TestTrimming:
    def test_marginal_breach_is_trimmed_not_rejected(self):
        report = _mgr().check([_buy("AAPL", 210_000)], _snap())  # 21% vs cap 20%
        assert report.rejected == []
        assert report.approved[0].est_notional == pytest.approx(200_000.0)

    def test_trim_updates_all_derived_fields(self):
        report = _mgr().check([_buy("AAPL", 250_000, price=100.0)], _snap())
        p = report.approved[0]
        assert p.delta_qty == pytest.approx(2_000.0)
        assert p.target_qty == pytest.approx(2_000.0)
        assert p.est_notional == pytest.approx(p.delta_qty * p.last_price)
        assert p.est_cost_usd > 0, "le coût de transaction doit suivre la nouvelle taille"

    def test_residual_too_small_is_rejected_cleanly(self):
        """
        Sous MIN_TRIM_FRACTION de la taille demandée, le résidu ne porte plus
        la thèse de l'agent : on rejette proprement plutôt que d'envoyer un
        moignon qui paiera la commission pour rien.
        """
        mgr = _mgr(max_net_long_pct=0.41, max_single_position_pct=1.0)
        report = mgr.check(
            [_buy("AAPL", 400_000, conf=0.9), _buy("SPY", 300_000, conf=0.1)],
            _snap(),
        )
        rejected = {r.plan.symbol for r in report.rejected}
        assert "SPY" in rejected                       # reliquat 10k / 300k = 3%
        assert f"{MIN_TRIM_FRACTION:.0%}" in report.rejected[0].reason

    def test_zero_capacity_rejects_with_binding_reason(self):
        report = _mgr(max_net_long_pct=0.0).check([_buy("AAPL", 10_000)], _snap())
        assert len(report.rejected) == 1
        assert "saturé" in report.rejected[0].reason

    def test_trimmed_plans_never_breach_any_cap(self):
        """Invariant global : quelle que soit l'entrée, aucun plafond ne saute."""
        mgr = _mgr(max_net_long_pct=0.35, max_single_position_pct=0.15, min_cash_pct=0.40)
        plans = [_buy(f"S{i}", 300_000, conf=i / 10) for i in range(8)]
        report = mgr.check(plans, _snap(cash=1_000_000.0))

        total = sum(p.est_notional for p in report.approved)
        assert total <= 0.35 * 1_000_000 + 1e-6,            "net long"
        assert all(p.est_notional <= 0.15 * 1_000_000 + 1e-6
                   for p in report.approved),                "taille unitaire"
        assert 1_000_000 - total >= 0.40 * 1_000_000 - 1e-6, "floor de cash"

    def test_sell_only_mode_still_blocks_everything(self):
        """Le rognage ne doit pas créer de porte dérobée au kill switch."""
        mgr = RiskManager(RiskConfig(sell_only_mode=True))
        report = mgr.check([_buy("AAPL", 10_000)], _snap())
        assert report.approved == []
        assert len(report.rejected) == 1
