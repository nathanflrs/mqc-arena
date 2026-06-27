# tests/test_risk_manager.py
from __future__ import annotations

import pytest

from src.broker.portfolio import PortfolioSnapshot
from src.execution.planner import OrderPlan
from src.risk.manager import DrawdownCircuitBreaker, RiskConfig, RiskManager


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


# ─── Régime-aware : scaling max_net_long ─────────────────────────────────────

def test_regime_bull_quiet_uses_full_limit():
    """bull_quiet → scale 1.0 → même limite que sans régime."""
    # max_single=0.40 pour que Rule 1 ne bloque pas le test de Rule 2
    mgr = RiskManager(RiskConfig(max_net_long_pct=0.60, max_single_position_pct=0.40, min_cash_pct=0.0))
    plans = [_plan("AAPL", "BUY", est_notional=36_000)]  # 36% < 60%
    report = mgr.check(plans, _snap(cash=100_000), gmm_regime="bull_quiet")
    assert len(report.approved) == 1
    assert report.regime_scale == pytest.approx(1.0)
    assert report.effective_max_net_long == pytest.approx(0.60)


def test_regime_bear_tightens_limit():
    """bear → scale 0.35 → effective max = 0.60 * 0.35 = 0.21."""
    mgr = RiskManager(RiskConfig(max_net_long_pct=0.60, max_single_position_pct=0.30, min_cash_pct=0.0))
    # 25% notional > 21% effective limit → rejected
    plans = [_plan("AAPL", "BUY", est_notional=25_000)]
    report = mgr.check(plans, _snap(cash=100_000), gmm_regime="bear")
    assert len(report.rejected) == 1
    assert "régime bear" in report.rejected[0].reason
    assert report.regime_scale == pytest.approx(0.35)
    assert report.effective_max_net_long == pytest.approx(0.21)


def test_regime_bear_small_position_passes():
    """bear → 15% notional < 21% effective limit → approved."""
    mgr = RiskManager(RiskConfig(max_net_long_pct=0.60, max_single_position_pct=0.30, min_cash_pct=0.0))
    plans = [_plan("AAPL", "BUY", est_notional=15_000)]
    report = mgr.check(plans, _snap(cash=100_000), gmm_regime="bear")
    assert len(report.approved) == 1


def test_regime_bull_volatile_scale():
    """bull_volatile → scale 0.75 → effective max = 0.60 * 0.75 = 0.45."""
    mgr = RiskManager(RiskConfig(max_net_long_pct=0.60, max_single_position_pct=0.30, min_cash_pct=0.0))
    plans = [_plan("AAPL", "BUY", est_notional=50_000)]  # 50% > 45% → rejected
    report = mgr.check(plans, _snap(cash=100_000), gmm_regime="bull_volatile")
    assert len(report.rejected) == 1
    assert report.effective_max_net_long == pytest.approx(0.45)


def test_regime_sideways_scale():
    """sideways → scale 0.60 → effective max = 0.60 * 0.60 = 0.36."""
    mgr = RiskManager(RiskConfig(max_net_long_pct=0.60, max_single_position_pct=0.30, min_cash_pct=0.0))
    plans = [_plan("AAPL", "BUY", est_notional=40_000)]  # 40% > 36% → rejected
    report = mgr.check(plans, _snap(cash=100_000), gmm_regime="sideways")
    assert len(report.rejected) == 1
    assert report.effective_max_net_long == pytest.approx(0.36)


def test_no_regime_uses_full_limit():
    """Sans gmm_regime → scale 1.0 → limite inchangée."""
    # max_single=0.40 pour éviter que Rule 1 bloque avant Rule 2
    mgr = RiskManager(RiskConfig(max_net_long_pct=0.40, max_single_position_pct=0.40, min_cash_pct=0.0))
    plans = [_plan("AAPL", "BUY", est_notional=35_000)]  # 35% < 40% → approved
    report = mgr.check(plans, _snap(cash=100_000))
    assert len(report.approved) == 1
    assert report.regime_scale == pytest.approx(1.0)


def test_regime_in_telegram_summary():
    mgr = RiskManager(RiskConfig(max_net_long_pct=0.60, min_cash_pct=0.0))
    report = mgr.check([], _snap(), gmm_regime="bear")
    summary = report.telegram_summary()
    assert "bear" in summary.lower()
    assert "0.35" in summary or "35%" in summary


# ─── Filtre liquidité ADV ─────────────────────────────────────────────────────

def test_adv_filter_blocks_illiquid_buy():
    """Position > 1 % de l'ADV → rejetée."""
    mgr = RiskManager(RiskConfig(max_net_long_pct=1.0, max_single_position_pct=1.0, min_cash_pct=0.0))
    # 1 000 shares @ $150 = $150 000 notional
    # ADV = 50 000 shares → 1000/50000 = 2% > 1% → rejeté
    plans   = [_plan("AAPL", "BUY", est_notional=150_000, last_price=150.0)]
    adv_map = {"AAPL": 50_000}
    report  = mgr.check(plans, _snap(netliq=1_000_000, cash=900_000), adv_map=adv_map)
    assert len(report.rejected) == 1
    assert "ADV" in report.rejected[0].reason


def test_adv_filter_approves_liquid_buy():
    """Position < 1 % de l'ADV → approuvée."""
    mgr = RiskManager(RiskConfig(max_net_long_pct=1.0, max_single_position_pct=1.0, min_cash_pct=0.0))
    # 100 shares @ $150 = $15 000 notional
    # ADV = 50 000 shares → 100/50000 = 0.2% < 1% → approuvé
    plans   = [_plan("AAPL", "BUY", est_notional=15_000, last_price=150.0)]
    adv_map = {"AAPL": 50_000}
    report  = mgr.check(plans, _snap(netliq=500_000, cash=400_000), adv_map=adv_map)
    assert len(report.approved) == 1


def test_adv_filter_missing_symbol_skips():
    """Symbole absent de adv_map → filtre ADV ignoré (pas de rejet)."""
    mgr = RiskManager(RiskConfig(max_net_long_pct=1.0, max_single_position_pct=1.0, min_cash_pct=0.0))
    plans   = [_plan("TSLA", "BUY", est_notional=50_000, last_price=200.0)]
    adv_map = {"AAPL": 50_000}  # TSLA absent
    report  = mgr.check(plans, _snap(netliq=500_000, cash=400_000), adv_map=adv_map)
    assert len(report.approved) == 1


def test_adv_filter_sell_not_checked():
    """SELL n'est jamais soumis au filtre ADV."""
    mgr = RiskManager(RiskConfig())
    plans   = [_plan("AAPL", "SELL", est_notional=500_000, current_qty=10_000, last_price=150.0)]
    adv_map = {"AAPL": 1}   # ADV ridiculement bas
    report  = mgr.check(plans, _snap(), adv_map=adv_map)
    assert len(report.approved) == 1


# ─── Circuit breaker gradué ───────────────────────────────────────────────────

def test_cb_level_zero_no_restriction(tmp_path, monkeypatch):
    monkeypatch.setattr(DrawdownCircuitBreaker, "_STATE_PATH", tmp_path / "cb.json")
    cb = DrawdownCircuitBreaker()
    cb.evaluate(100_000)
    assert cb.level == 0
    assert cb.level_name == "NORMAL"
    assert not cb.is_triggered


def test_cb_level_1_defensive(tmp_path, monkeypatch):
    monkeypatch.setattr(DrawdownCircuitBreaker, "_STATE_PATH", tmp_path / "cb.json")
    cb = DrawdownCircuitBreaker()
    cb.evaluate(100_000)   # set peak
    cb.evaluate(95_000)    # 5% DD → level 1
    assert cb.level == 1
    assert cb.level_name == "DÉFENSIF"
    assert not cb.is_triggered


def test_cb_level_2_alert(tmp_path, monkeypatch):
    monkeypatch.setattr(DrawdownCircuitBreaker, "_STATE_PATH", tmp_path / "cb.json")
    cb = DrawdownCircuitBreaker()
    cb.evaluate(100_000)
    cb.evaluate(93_000)    # 7% DD → level 2
    assert cb.level == 2
    assert cb.level_name == "ALERTE"
    assert not cb.is_triggered


def test_cb_level_3_emergency(tmp_path, monkeypatch):
    monkeypatch.setattr(DrawdownCircuitBreaker, "_STATE_PATH", tmp_path / "cb.json")
    cb = DrawdownCircuitBreaker()
    cb.evaluate(100_000)
    cb.evaluate(90_000)    # 10% DD → level 3
    assert cb.level == 3
    assert cb.level_name == "URGENCE"
    assert cb.is_triggered


def test_cb_level_3_sticky(tmp_path, monkeypatch):
    """Une fois à level 3, le niveau reste 3 même si le portefeuille se redresse."""
    monkeypatch.setattr(DrawdownCircuitBreaker, "_STATE_PATH", tmp_path / "cb.json")
    cb = DrawdownCircuitBreaker()
    cb.evaluate(100_000)
    cb.evaluate(90_000)    # 10% DD → level 3
    cb.evaluate(99_000)    # quasi-recovery mais sticky
    assert cb.level == 3
    assert cb.is_triggered


def test_cb_levels_1_and_2_auto_revert(tmp_path, monkeypatch):
    """Levels 1 et 2 reviennent à 0 si le drawdown se réduit."""
    monkeypatch.setattr(DrawdownCircuitBreaker, "_STATE_PATH", tmp_path / "cb.json")
    cb = DrawdownCircuitBreaker()
    cb.evaluate(100_000)
    cb.evaluate(94_000)    # 6% → level 1 ou 2 (6% > 4%, < 6% → level 1... wait)
    # 6% drawdown: > 4% → level 1, not quite > 6% so level 1
    assert cb.level == 1
    cb.evaluate(98_000)    # 2% → level 0
    assert cb.level == 0


def test_cb_reset_clears_level_3(tmp_path, monkeypatch):
    monkeypatch.setattr(DrawdownCircuitBreaker, "_STATE_PATH", tmp_path / "cb.json")
    cb = DrawdownCircuitBreaker()
    cb.evaluate(100_000)
    cb.evaluate(88_000)    # 12% → level 3
    assert cb.is_triggered
    cb.reset()
    assert cb.level == 0
    assert not cb.is_triggered


def test_cb_level_in_risk_report():
    """Le cb_level passé à check() apparaît dans le rapport."""
    mgr = RiskManager(RiskConfig())
    report = mgr.check([], _snap(), cb_level=2)
    assert report.cb_level == 2
    assert report.cb_level_name == "ALERTE"
    assert "ALERTE" in report.telegram_summary()


def test_cb_migration_from_old_json(tmp_path, monkeypatch):
    """Ancien JSON sans 'level' est migré correctement."""
    monkeypatch.setattr(DrawdownCircuitBreaker, "_STATE_PATH", tmp_path / "cb.json")
    # Simuler un ancien fichier JSON sans champ 'level'
    old_state = {
        "triggered": True,
        "peak_netliq": 100_000,
        "current_netliq": 88_000,
        "drawdown": 0.12,
        "triggered_at": "2026-01-01T00:00:00+00:00",
        "drawdown_at_trigger": 0.12,
    }
    (tmp_path / "cb.json").write_text(__import__("json").dumps(old_state))
    cb = DrawdownCircuitBreaker()
    assert cb.level == 3    # triggered=True → level 3
    assert cb.is_triggered
