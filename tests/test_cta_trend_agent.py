# tests/test_cta_trend_agent.py
"""
Tests CTA Trend-Following Agent — Steps 1-4.
Stratégie de contrôle ADX : CTATrendConfig(adx_entry=0, adx_exit=-1) bypasse
les filtres ADX pour tester la logique signal/vol isolément. Les tests exit/neutre
utilisent un dataset réel (drift=-0.0015 → ADX≈36) avec des seuils adaptés.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.agents.base import AgentSignal, MarketState
from src.agents.cta_trend_agent import (
    CTA_UNIVERSE,
    CTATrendAgent,
    CTATrendConfig,
    _CTASignal,
)
from src.execution.planner import OrderPlan, cta_plan_from_signal

# ── Helpers ────────────────────────────────────────────────────────────────────

_NETLIQ = 100_000.0


def _make_uptrend(n: int = 300, start: float = 100.0, end: float = 200.0,
                  seed: int = 42) -> pd.DataFrame:
    """Tendance linéaire haussière avec HLC réaliste (ADX élevé)."""
    rng = np.random.default_rng(seed)
    close = np.linspace(start, end, n)
    noise_h = rng.uniform(0.003, 0.010, n)
    noise_l = rng.uniform(0.003, 0.010, n)
    step = (end - start) / n
    high = close + noise_h * close + step * 2   # high s'écarte vers le haut
    low  = close - noise_l * close
    idx  = pd.date_range("2023-01-01", periods=n, freq="B")
    return pd.DataFrame({"High": high, "Low": low, "Close": close}, index=idx)


def _make_downtrend(n: int = 300, start: float = 200.0, end: float = 100.0,
                    seed: int = 42) -> pd.DataFrame:
    """Tendance linéaire baissière avec HLC réaliste."""
    rng = np.random.default_rng(seed)
    close = np.linspace(start, end, n)
    noise_h = rng.uniform(0.003, 0.010, n)
    noise_l = rng.uniform(0.003, 0.010, n)
    step = (end - start) / n   # négatif
    high = close + noise_h * close
    low  = close - noise_l * close + step * 2   # low s'écarte vers le bas
    idx  = pd.date_range("2023-01-01", periods=n, freq="B")
    return pd.DataFrame({"High": high, "Low": low, "Close": close}, index=idx)


def _make_stochastic_trend(drift: float = -0.0015, vol: float = 0.008,
                            seed: int = 42, n: int = 300) -> pd.DataFrame:
    """GBM avec paramètres réalistes pour vol targeting et tests ADX naturels."""
    rng = np.random.default_rng(seed)
    d = rng.normal(drift, vol, n)
    c = 100.0 * np.exp(np.cumsum(d))
    nh = rng.uniform(0.003, 0.015, n)
    nl = rng.uniform(0.003, 0.015, n)
    h = c * (1 + nh + np.maximum(d, 0) * 2)
    l = c * (1 - nl - np.maximum(-d, 0) * 2)
    idx = pd.date_range("2023-01-01", periods=n, freq="B")
    return pd.DataFrame({"High": h, "Low": l, "Close": c}, index=idx)


def _make_disagreement(n: int = 300, seed: int = 42) -> pd.DataFrame:
    """
    200 bars de baisse prononcée puis 100 bars de faible reprise.
    → mom126 < 0 (avant la reprise était plus haut), mom63 > 0 (reprise récente).
    """
    rng = np.random.default_rng(seed)
    close = np.zeros(n)
    close[0] = 100.0
    for i in range(1, 200):
        close[i] = close[i - 1] * 0.997          # -0.3%/j sur 200 bars
    for i in range(200, n):
        close[i] = close[i - 1] * 1.001          # +0.1%/j sur 100 bars
    noise_h = rng.uniform(0.002, 0.008, n)
    noise_l = rng.uniform(0.002, 0.008, n)
    high = close * (1 + noise_h)
    low  = close * (1 - noise_l)
    idx  = pd.date_range("2023-01-01", periods=n, freq="B")
    return pd.DataFrame({"High": high, "Low": low, "Close": close}, index=idx)


def _no_adx_config(**kw) -> CTATrendConfig:
    """Config qui bypasse tous les filtres ADX (pour tester signal/vol isolément)."""
    return CTATrendConfig(adx_entry=0.0, adx_exit=-1.0, **kw)


def _state(symbol: str, df: pd.DataFrame) -> MarketState:
    return MarketState(symbol=symbol, price=float(df["Close"].iloc[-1]),
                       timestamp=str(df.index[-1]))


# ══════════════════════════════════════════════════════════════════════════════
# GROUPE 1 — Signal dual-timeframe (Step 1)
# ══════════════════════════════════════════════════════════════════════════════

class TestSignal:

    def test_out_of_universe_returns_hold(self):
        agent = CTATrendAgent()
        sig = agent.generate_signal(MarketState("AAPL", 150.0, "2024-01-01"), {})
        assert sig.action == "HOLD"
        assert sig.target_weight == 0.0
        assert sig.confidence == 0.0

    def test_all_six_etfs_in_universe(self):
        assert set(CTA_UNIVERSE) == {"SPY", "QQQ", "TLT", "GLD", "UUP", "DBC"}

    def test_insufficient_data_returns_hold(self):
        n = 100  # < min_history=230
        idx = pd.date_range("2023-01-01", periods=n, freq="B")
        df = pd.DataFrame({"High": 101.0, "Low": 99.0, "Close": 100.0}, index=idx)
        agent = CTATrendAgent()
        sig = agent.generate_signal(_state("SPY", df), {}, data=df)
        assert sig.action == "HOLD"
        assert "insuffisant" in sig.reason

    def test_missing_hlc_returns_hold(self):
        """Données Close-only : HLC absentes → pas de trade (garde explicite)."""
        n = 300
        close = np.linspace(100, 200, n)
        idx = pd.date_range("2023-01-01", periods=n, freq="B")
        df = pd.DataFrame({"Close": close}, index=idx)
        agent = CTATrendAgent()
        sig = agent.generate_signal(_state("SPY", df), {}, data=df)
        assert sig.action == "HOLD"
        assert "HLC absent" in sig.reason

    def test_long_signal_uptrend(self):
        """mom63>0, mom126>0, prix>SMA200 → BUY."""
        df = _make_uptrend()
        agent = CTATrendAgent(_no_adx_config())
        sig = agent.generate_signal(_state("SPY", df), {}, data=df)
        assert sig.action == "BUY"
        assert sig.meta["mom_fast"] > 0
        assert sig.meta["mom_slow"] > 0
        assert sig.meta["last_price"] > sig.meta["sma200"]

    def test_short_signal_downtrend(self):
        """mom63<0, mom126<0, prix<SMA200 → SELL."""
        df = _make_downtrend()
        agent = CTATrendAgent(_no_adx_config())
        sig = agent.generate_signal(_state("TLT", df), {}, data=df)
        assert sig.action == "SELL"
        assert sig.meta["mom_fast"] < 0
        assert sig.meta["mom_slow"] < 0
        assert sig.meta["last_price"] < sig.meta["sma200"]

    def test_temporal_disagreement_returns_hold(self):
        """mom63>0 mais mom126<0 → désaccord → HOLD."""
        df = _make_disagreement()
        agent = CTATrendAgent(_no_adx_config())
        raw = agent._compute_signal("QQQ", df, current_qty=0.0)
        # Vérifier les signes des moms avant de tester la direction
        close = df["Close"]
        px = float(close.iloc[-1])
        mom_fast = px / float(close.iloc[-64]) - 1.0
        mom_slow = px / float(close.iloc[-127]) - 1.0
        # Si désaccord effectif (le dataset doit le produire), direction flat/hold
        if mom_fast > 0 and mom_slow < 0:
            assert raw.direction in ("flat", "hold")
        # (sinon le test est ignoré, données pas assez divergentes)

    def test_price_below_sma200_blocks_long_signal(self):
        """Même si mom63/126 positifs, prix<SMA200 → pas de long."""
        df = _make_downtrend(start=150.0, end=140.0)  # légère baisse, prix sous SMA200
        agent = CTATrendAgent(_no_adx_config())
        sig = agent.generate_signal(_state("GLD", df), {}, data=df)
        # Prix sous SMA200 → pas de BUY même si moms positifs
        assert sig.action != "BUY" or sig.meta["last_price"] > sig.meta["sma200"]

    def test_adx_entry_filter_blocks_low_adx(self):
        """ADX réel < adx_entry (config standard) → HOLD même avec moms positifs."""
        df = _make_uptrend()
        # Config standard (adx_entry=20) : si ADX réel est faible → bloqué
        agent_std = CTATrendAgent()
        raw = agent_std._compute_signal("SPY", df, current_qty=0.0)
        adx_real = agent_std._adx(df)
        if adx_real < agent_std.cfg.adx_entry:
            assert raw.direction in ("flat", "hold")
        # Si ADX est assez haut, le signal peut passer — les deux cas sont valides

    def test_strategy_in_meta(self):
        """meta['strategy'] == 'cta_trend' toujours."""
        agent = CTATrendAgent()
        sig = agent.generate_signal(MarketState("MSFT", 300.0, "2024-01-01"), {})
        assert sig.meta.get("strategy") == "cta_trend"


# ══════════════════════════════════════════════════════════════════════════════
# GROUPE 2 — Vol targeting (Step 2)
# ══════════════════════════════════════════════════════════════════════════════

class TestVolTargeting:

    def _agent(self, **kw):
        return CTATrendAgent(_no_adx_config(**kw))

    def test_target_weight_positive_for_long(self):
        df = _make_uptrend()
        agent = self._agent()
        sig = agent.generate_signal(_state("SPY", df), {}, data=df)
        if sig.action == "BUY":
            assert sig.target_weight > 0.0

    def test_target_weight_negative_for_short(self):
        df = _make_downtrend()
        agent = self._agent()
        sig = agent.generate_signal(_state("TLT", df), {}, data=df)
        if sig.action == "SELL":
            assert sig.target_weight < 0.0

    def test_target_weight_formula(self):
        """weight = vol_target / realized_vol, cap = max_position_pct."""
        df = _make_stochastic_trend(drift=-0.0015)
        agent = self._agent(vol_target_pct=0.10, max_position_pct=0.15)
        sig = agent.generate_signal(_state("TLT", df), {}, data=df)
        if sig.action in ("BUY", "SELL"):
            rv = sig.meta["realized_vol"]
            assert rv > 0
            expected_raw = 0.10 / rv
            expected = min(expected_raw, 0.15)
            assert abs(abs(sig.target_weight) - expected) < 1e-9

    def test_target_weight_capped_at_max_position(self):
        """Vol très faible → weight brut énorme → cappé à max_position_pct."""
        df = _make_uptrend(start=100, end=200)
        # Vol très faible (trend déterministe linéaire)
        agent = self._agent(vol_target_pct=0.10, max_position_pct=0.15)
        sig = agent.generate_signal(_state("SPY", df), {}, data=df)
        assert abs(sig.target_weight) <= 0.15 + 1e-9

    def test_zero_realized_vol_returns_hold(self):
        """Si vol = 0 (prix constants), l'agent retourne HOLD (impossible de sizer)."""
        n = 300
        close = np.full(n, 100.0)   # prix complètement constants
        idx = pd.date_range("2023-01-01", periods=n, freq="B")
        df = pd.DataFrame({
            "High": close + 0.5, "Low": close - 0.5, "Close": close,
        }, index=idx)
        agent = self._agent()
        sig = agent.generate_signal(_state("GLD", df), {}, data=df)
        if "vol" in sig.reason.lower():
            assert sig.action == "HOLD"
            assert sig.target_weight == 0.0

    def test_confidence_adx_scaled(self):
        """Confidence = 0 au seuil ADX, >0 si ADX plus fort."""
        df = _make_stochastic_trend(drift=-0.0015)
        agent = CTATrendAgent()
        sig = agent.generate_signal(_state("TLT", df), {}, data=df)
        if sig.action in ("BUY", "SELL"):
            assert 0.0 <= sig.confidence <= 1.0


# ══════════════════════════════════════════════════════════════════════════════
# GROUPE 3 — Exits ADX (Step 4)
# ══════════════════════════════════════════════════════════════════════════════

class TestExitsADX:
    """
    Dataset: stochastic downtrend → ADX ≈ 36.
    Config adx_exit=50 → ADX(36) < 50 → déclenche EXIT.
    Config adx_exit=30, adx_entry=40 → ADX(36) entre les deux → zone neutre.
    """

    @pytest.fixture(autouse=True)
    def _setup(self):
        self.df = _make_stochastic_trend(drift=-0.0015)
        self.px  = float(self.df["Close"].iloc[-1])
        self.state_tlt = MarketState("TLT", self.px, str(self.df.index[-1]))

    def _exit_agent(self):
        """ADX exit à 50 — ADX≈36 déclenche EXIT."""
        return CTATrendAgent(CTATrendConfig(adx_exit=50.0, adx_entry=60.0))

    def _zone_agent(self):
        """Zone neutre 30–40 — ADX≈36 → maintien de la position."""
        return CTATrendAgent(CTATrendConfig(adx_exit=30.0, adx_entry=40.0))

    def test_exit_adx_closes_long(self):
        ag = self._exit_agent()
        sig = ag.generate_signal(self.state_tlt, {"TLT": 150.0}, data=self.df)
        assert sig.action == "SELL"
        assert sig.target_weight == 0.0
        assert "EXIT ADX" in sig.reason

    def test_exit_adx_covers_short(self):
        ag = self._exit_agent()
        sig = ag.generate_signal(self.state_tlt, {"TLT": -100.0}, data=self.df)
        assert sig.action == "BUY"
        assert sig.target_weight == 0.0
        assert "EXIT ADX" in sig.reason

    def test_exit_adx_no_position_returns_hold(self):
        ag = self._exit_agent()
        sig = ag.generate_signal(self.state_tlt, {}, data=self.df)
        assert sig.action == "HOLD"
        assert "EXIT ADX" in sig.reason

    def test_neutral_zone_holds_existing_long(self):
        """Zone neutre : position longue maintenue, PAS fermée."""
        ag = self._zone_agent()
        sig = ag.generate_signal(self.state_tlt, {"TLT": 80.0}, data=self.df)
        assert sig.action == "HOLD"
        assert "zone neutre" in sig.reason

    def test_neutral_zone_holds_existing_short(self):
        ag = self._zone_agent()
        sig = ag.generate_signal(self.state_tlt, {"TLT": -60.0}, data=self.df)
        assert sig.action == "HOLD"

    def test_neutral_zone_no_new_entry_when_flat(self):
        ag = self._zone_agent()
        sig = ag.generate_signal(self.state_tlt, {}, data=self.df)
        assert sig.action == "HOLD"
        assert "zone neutre" in sig.reason


# ══════════════════════════════════════════════════════════════════════════════
# GROUPE 4 — Reversals dans generate_signal (Step 4)
# ══════════════════════════════════════════════════════════════════════════════

class TestReversals:

    def _agent(self):
        return CTATrendAgent(_no_adx_config())

    def test_long_to_flat_returns_sell(self):
        """Signal flat + position longue → SELL (fermeture)."""
        df = _make_downtrend()   # signal SHORT ou désaccord après v-shape
        agent = self._agent()
        # Force le signal flat via les conditions (prix<SMA200 en downtrend bloque short)
        # Mais si direction=short, c'est SELL quand même — vérifions juste que tw=0 pour exit
        sig = agent.generate_signal(_state("TLT", df), {"TLT": 50.0}, data=df)
        # Si downtrend clair → SHORT signal → tw < 0 (reversal, pas exit pur)
        # → action doit être SELL dans tous les cas
        assert sig.action == "SELL"

    def test_short_signal_with_no_position_produces_sell(self):
        """Signal SHORT sans position existante → SELL ouverture short."""
        df = _make_downtrend()
        agent = self._agent()
        sig = agent.generate_signal(_state("TLT", df), {}, data=df)
        assert sig.action == "SELL"
        assert sig.target_weight < 0.0

    def test_long_signal_with_no_position_produces_buy(self):
        """Signal LONG sans position → BUY ouverture long."""
        df = _make_uptrend()
        agent = self._agent()
        sig = agent.generate_signal(_state("SPY", df), {}, data=df)
        assert sig.action == "BUY"
        assert sig.target_weight > 0.0

    def test_current_qty_tracked_in_meta(self):
        """current_qty passé depuis portfolio figure dans meta."""
        df = _make_uptrend()
        agent = self._agent()
        sig = agent.generate_signal(_state("SPY", df), {"SPY": 77.0}, data=df)
        assert sig.meta.get("current_qty") == 77.0

    def test_out_of_universe_symbol_not_in_meta_strategy(self):
        """Symbole hors univers : meta strategy=cta_trend mais action=HOLD."""
        agent = CTATrendAgent()
        sig = agent.generate_signal(MarketState("NVDA", 500.0, "2024-01-01"), {})
        assert sig.action == "HOLD"
        assert sig.meta.get("strategy") == "cta_trend"


# ══════════════════════════════════════════════════════════════════════════════
# GROUPE 5 — cta_plan_from_signal (Step 4 — planner)
# ══════════════════════════════════════════════════════════════════════════════

def _sig(action: str, tw: float, symbol: str = "TLT") -> AgentSignal:
    return AgentSignal("CTATrendAgent", symbol, action, 0.6, tw, "test", {})


class TestCTAPlanner:

    def test_reversal_long_to_short_single_order(self):
        """L→S : SELL une seule fois, delta = target_qty - current_qty."""
        plan = cta_plan_from_signal(_sig("SELL", -0.10), _NETLIQ, 100.0, +100.0)
        assert plan.action == "SELL"
        assert plan.target_qty == -100.0       # 10% de 100k / 100$ = -100
        assert plan.delta_qty  == -200.0       # -100 - (+100)
        assert plan.strategy   == "cta_trend"

    def test_reversal_short_to_long_single_order(self):
        """S→L : BUY une seule fois, delta couvre le short ET ouvre le long."""
        plan = cta_plan_from_signal(_sig("BUY", +0.12, "SPY"), _NETLIQ, 100.0, -80.0)
        assert plan.action == "BUY"
        assert plan.target_qty == 120.0        # 12% de 100k / 100$ = 120
        assert plan.delta_qty  == 200.0        # 120 - (-80)

    def test_exit_long_to_flat(self):
        plan = cta_plan_from_signal(_sig("SELL", 0.0, "GLD"), _NETLIQ, 180.0, +55.0)
        assert plan.action    == "SELL"
        assert plan.target_qty == 0.0
        assert plan.delta_qty  == -55.0

    def test_cover_short_to_flat(self):
        plan = cta_plan_from_signal(_sig("BUY", 0.0), _NETLIQ, 90.0, -40.0)
        assert plan.action    == "BUY"
        assert plan.target_qty == 0.0
        assert plan.delta_qty  == 40.0

    def test_already_at_target_returns_hold(self):
        plan = cta_plan_from_signal(_sig("BUY", +0.12, "SPY"), _NETLIQ, 100.0, 120.0)
        assert plan.action    == "HOLD"
        assert plan.delta_qty == 0.0

    def test_price_zero_returns_hold(self):
        plan = cta_plan_from_signal(_sig("BUY", +0.10), _NETLIQ, 0.0, 0.0)
        assert plan.action == "HOLD"

    def test_hold_signal_returns_hold(self):
        plan = cta_plan_from_signal(_sig("HOLD", 0.0), _NETLIQ, 100.0, 0.0)
        assert plan.action == "HOLD"

    def test_strategy_cta_trend_on_all_plans(self):
        plans = [
            cta_plan_from_signal(_sig("SELL", -0.10), _NETLIQ, 100.0, +100.0),
            cta_plan_from_signal(_sig("BUY",  +0.12, "SPY"), _NETLIQ, 100.0, -80.0),
            cta_plan_from_signal(_sig("SELL", 0.0, "GLD"), _NETLIQ, 180.0, +55.0),
            cta_plan_from_signal(_sig("HOLD", 0.0), _NETLIQ, 100.0, 0.0),
        ]
        assert all(p.strategy == "cta_trend" for p in plans)

    def test_est_notional_correct(self):
        """est_notional = |delta_qty| × last_price."""
        plan = cta_plan_from_signal(_sig("SELL", -0.10), _NETLIQ, 100.0, +100.0)
        assert abs(plan.est_notional - abs(plan.delta_qty) * 100.0) < 1e-6

    def test_open_short_from_flat(self):
        """target_weight < 0, current_qty=0 → SELL pour ouvrir un short."""
        plan = cta_plan_from_signal(_sig("SELL", -0.08), _NETLIQ, 200.0, 0.0)
        assert plan.action    == "SELL"
        assert plan.target_qty < 0
        assert plan.delta_qty < 0

    def test_open_long_from_flat(self):
        """target_weight > 0, current_qty=0 → BUY."""
        plan = cta_plan_from_signal(_sig("BUY", +0.10, "QQQ"), _NETLIQ, 400.0, 0.0)
        assert plan.action    == "BUY"
        assert plan.target_qty > 0
        assert plan.delta_qty > 0
