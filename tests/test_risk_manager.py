# tests/test_risk_manager.py
from __future__ import annotations

import pytest

from src.broker.portfolio import PortfolioSnapshot
from src.execution.planner import OrderPlan
from src.risk.manager import RiskConfig, RiskManager


def _snap(netliq: float = 100_000.0, cash: float = 80_000.0, positions: dict | None = None) -> PortfolioSnapshot:
    return PortfolioSnapshot(net_liquidation=netliq, cash=cash, positions=positions or {})


def _plan(
    symbol: str = "AAPL",
    action: str = "BUY",
    est_notional: float = 10_000.0,
    current_qty: float = 0.0,
    last_price: float = 150.0,
    target_weight: float = 0.10,
) -> OrderPlan:
    target_qty = est_notional / last_price if action == "BUY" else current_qty
    delta = target_qty - current_qty
    return OrderPlan(
        symbol=symbol,
        action=action,
        target_weight=target_weight,
        last_price=last_price,
        current_qty=current_qty,
        target_qty=target_qty,
        delta_qty=delta,
        est_notional=est_notional if action != "HOLD" else 0.0,
        reason="test",
    )


@pytest.fixture
def mgr():
    return RiskManager(RiskConfig(
        max_net_long_pct=0.40,
        max_single_position_pct=0.20,
        min_cash_pct=0.30,
        sell_only_mode=False,
    ))


# ─── Tests de base ────────────────────────────────────────────────────────────

def test_buy_within_limits_approved(mgr):
    plans = [_plan("AAPL", "BUY", est_notional=10_000)]
    report = mgr.check(plans, _snap())
    assert len(report.approved) == 1
    assert len(report.rejected) == 0


def test_sell_always_approved(mgr):
    plans = [_plan("AAPL", "SELL", est_notional=10_000, current_qty=70)]
    report = mgr.check(plans, _snap())
    assert len(report.approved) == 1
    assert len(report.rejected) == 0


def test_hold_always_approved(mgr):
    plans = [_plan("AAPL", "HOLD")]
    report = mgr.check(plans, _snap())
    assert len(report.approved) == 1


# ─── Règle 0 : kill switch ────────────────────────────────────────────────────

def test_sell_only_mode_blocks_buy():
    mgr = RiskManager(RiskConfig(sell_only_mode=True))
    plans = [_plan("AAPL", "BUY", est_notional=5_000)]
    report = mgr.check(plans, _snap())
    assert len(report.approved) == 0
    assert len(report.rejected) == 1
    assert "SELL_ONLY_MODE" in report.rejected[0].reason


def test_sell_only_mode_lets_sell_through():
    mgr = RiskManager(RiskConfig(sell_only_mode=True))
    plans = [
        _plan("AAPL", "SELL", est_notional=5_000, current_qty=33),
        _plan("SPY", "BUY", est_notional=5_000),
    ]
    report = mgr.check(plans, _snap())
    assert len(report.approved) == 1
    assert report.approved[0].symbol == "AAPL"
    assert report.rejected[0].plan.symbol == "SPY"


# ─── Règle 1 : taille unitaire ────────────────────────────────────────────────

def test_single_position_too_large_rejected(mgr):
    # 25 000 / 100 000 = 25% > 20%
    plans = [_plan("AAPL", "BUY", est_notional=25_000)]
    report = mgr.check(plans, _snap())
    assert len(report.rejected) == 1
    assert "unitaire" in report.rejected[0].reason


def test_single_position_at_exact_limit_approved(mgr):
    # 20 000 / 100 000 = 20% = limite exacte → approuvé
    plans = [_plan("AAPL", "BUY", est_notional=20_000)]
    report = mgr.check(plans, _snap())
    assert len(report.approved) == 1


# ─── Règle 2 : exposition nette longue ────────────────────────────────────────

def test_net_long_cap_blocks_second_buy():
    # max_single=30% pour que la règle unitaire ne déclenche pas à 25%
    # max_net_long=40% : premier BUY (25%) passe, second (25+25=50%) est rejeté
    mgr = RiskManager(RiskConfig(max_net_long_pct=0.40, max_single_position_pct=0.30, min_cash_pct=0.0))
    plans = [
        _plan("AAPL", "BUY", est_notional=25_000),
        _plan("SPY",  "BUY", est_notional=25_000),
    ]
    report = mgr.check(plans, _snap(cash=80_000))
    assert len(report.approved) == 1
    assert len(report.rejected) == 1
    assert "net long" in report.rejected[0].reason


def test_existing_position_counts_toward_net_long():
    # Déjà 30k en position (current_qty=200 * prix=150)
    # Un BUY de 15k pousserait à 45% > 40%
    mgr = RiskManager(RiskConfig(max_net_long_pct=0.40, min_cash_pct=0.0))
    plans = [
        _plan("AAPL", "BUY", est_notional=15_000, current_qty=200, last_price=150),
    ]
    report = mgr.check(plans, _snap(netliq=100_000, cash=70_000))
    assert len(report.rejected) == 1
    assert "net long" in report.rejected[0].reason


# ─── Règle 3 : floor de cash ──────────────────────────────────────────────────

def test_cash_floor_blocks_buy():
    # Cash = 35k, min_cash_pct=30% → floor = 30k
    # Un BUY de 10k laisserait 25k = 25% < 30% → rejeté
    mgr = RiskManager(RiskConfig(max_net_long_pct=1.0, max_single_position_pct=1.0, min_cash_pct=0.30))
    plans = [_plan("AAPL", "BUY", est_notional=10_000)]
    report = mgr.check(plans, _snap(netliq=100_000, cash=35_000))
    assert len(report.rejected) == 1
    assert "cash" in report.rejected[0].reason


def test_cash_floor_sell_replenishes_cash():
    # SELL récupère du cash → le BUY suivant peut passer
    mgr = RiskManager(RiskConfig(max_net_long_pct=1.0, max_single_position_pct=1.0, min_cash_pct=0.30))
    plans = [
        _plan("AAPL", "SELL", est_notional=20_000, current_qty=133),
        _plan("SPY",  "BUY",  est_notional=10_000),
    ]
    report = mgr.check(plans, _snap(netliq=100_000, cash=25_000))
    # Après SELL: cash = 45k (45%) > 30% → BUY passe
    assert len(report.approved) == 2
    assert len(report.rejected) == 0


# ─── Métriques du rapport ─────────────────────────────────────────────────────

def test_pre_post_trade_pct_calculated(mgr):
    plans = [_plan("AAPL", "BUY", est_notional=10_000, current_qty=0)]
    report = mgr.check(plans, _snap())
    assert report.pre_trade_long_pct == pytest.approx(0.0)
    assert report.post_trade_long_pct == pytest.approx(0.10)


def test_telegram_summary_contains_key_info(mgr):
    plans = [_plan("AAPL", "BUY", est_notional=10_000)]
    report = mgr.check(plans, _snap())
    summary = report.telegram_summary()
    assert "Risk Manager" in summary
    assert "Approuvés" in summary
    assert "Rejetés" in summary


def test_sell_only_flag_in_summary():
    mgr = RiskManager(RiskConfig(sell_only_mode=True))
    report = mgr.check([], _snap())
    assert "SELL-ONLY" in report.telegram_summary()
