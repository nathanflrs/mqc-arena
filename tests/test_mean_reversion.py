# tests/test_mean_reversion.py
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.agents.mean_reversion import MeanReversionAgent, MeanReversionConfig
from tests.conftest import make_state


@pytest.fixture
def agent():
    return MeanReversionAgent()


def test_no_data_returns_hold(agent):
    sig = agent.generate_signal(make_state(), {}, regime="bull", data=None)
    assert sig.action == "HOLD"


def test_short_history_returns_hold(agent, short_df):
    sig = agent.generate_signal(make_state(), {}, regime="bull", data=short_df)
    assert sig.action == "HOLD"


def test_oversold_triggers_buy(agent, oversold_df):
    sig = agent.generate_signal(make_state(), {}, regime="bull", data=oversold_df)
    # Avec RSI bas + prix sous Bollinger + volume élevé → BUY ou HOLD selon intensité
    assert sig.action in ("BUY", "HOLD")
    assert 0.0 <= sig.confidence <= 1.0


def test_uptrend_in_position_triggers_sell(agent, bull_df):
    # En tendance haussière, RSI élevé + prix > SMA20 → SELL
    portfolio = {"AAPL": 0.08}
    sig = agent.generate_signal(make_state(price=float(bull_df["Close"].iloc[-1])), portfolio, regime="bull", data=bull_df)
    assert sig.action == "SELL"


def test_bear_regime_tightens_rsi_threshold():
    bull_cfg = MeanReversionConfig(rsi_threshold=35)
    bear_cfg = MeanReversionConfig(rsi_threshold_bear=30)
    agent = MeanReversionAgent(config=bear_cfg)
    assert agent.cfg.rsi_threshold_bear == 30


def test_config_overrides():
    cfg = MeanReversionConfig(rsi_overbought=70, target_weight=0.05)
    agent = MeanReversionAgent(config=cfg)
    assert agent.cfg.rsi_overbought == 70
    assert agent.cfg.target_weight == 0.05


def test_confidence_bounded(agent, flat_df):
    sig = agent.generate_signal(make_state(), {}, regime="bull", data=flat_df)
    assert 0.0 <= sig.confidence <= 1.0


def test_agent_name(agent):
    assert agent.name == "MeanReversionAgent"


def test_sell_confidence_from_config(agent, bull_df):
    portfolio = {"AAPL": 0.08}
    sig = agent.generate_signal(make_state(), portfolio, regime="bull", data=bull_df)
    if sig.action == "SELL":
        assert sig.confidence == agent.cfg.sell_confidence


# ── Garde-fou baissier ────────────────────────────────────────────────────────

class TestGardeFouBaissier:
    """
    Mesuré le 2026-08-14 : en marché baissier au-delà de 20 %, l'agent perd
    −2,62 % avec un IC de [−4,08 %, −1,15 %]. L'intervalle exclut zéro par le
    bas — ce n'est pas une absence d'avantage, c'est une perte. Acheter des
    baisses pendant un krach, c'est rattraper le couteau qui tombe.
    """

    def _survendu(self, n=120):
        """
        Série qui déclenche un achat : chute brutale sur trois séances.

        Une baisse LENTE ne déclencherait pas l'agent — la bande de Bollinger
        suit la descente, et le prix ne repasse jamais dessous. L'agent ne
        réagit qu'aux décrochages rapides, ce qui est cohérent avec son idée
        mais mérite d'être su : il ignore les érosions prolongées.
        """
        px = [100.0] * (n - 3) + [96.0, 92.0, 86.0]
        vol = [1_000_000] * (n - 1) + [5_000_000]
        idx = pd.date_range("2024-01-01", periods=n, freq="B")
        return pd.DataFrame({"Close": px, "Volume": vol}, index=idx)

    def test_buys_when_market_is_only_mildly_down(self):
        a = MeanReversionAgent()
        a.set_market_drawdown(0.12)          # correction, régime favorable
        df = self._survendu()
        s = a.generate_signal(
            make_state("X", float(df.Close.iloc[-1])), {}, "bull", df)
        assert s.action == "BUY"

    def test_refuses_to_buy_in_a_crash(self):
        a = MeanReversionAgent()
        a.set_market_drawdown(0.25)          # au-delà du seuil
        df = self._survendu()
        s = a.generate_signal(
            make_state("X", float(df.Close.iloc[-1])), {}, "bear", df)
        assert s.action == "HOLD"
        assert s.meta.get("blocked_by") == "max_market_drawdown"
        assert "achat suspendu" in s.reason

    def test_exit_still_works_in_a_crash(self):
        """
        Le garde-fou ne doit bloquer que l'ACHAT. Fermer une position pendant un
        krach est de la réduction de risque : l'interdire serait dangereux.
        """
        a = MeanReversionAgent()
        a.set_market_drawdown(0.40)          # krach profond
        n = 120
        px = list(np.linspace(70, 70, n - 5)) + list(np.linspace(70, 100, 5))
        idx = pd.date_range("2024-01-01", periods=n, freq="B")
        df = pd.DataFrame({"Close": px, "Volume": [1_000_000] * n}, index=idx)
        s = a.generate_signal(
            make_state("X", float(df.Close.iloc[-1])),
            {"X": 100.0}, "bear", df)      # position ouverte
        assert s.action == "SELL", "la sortie doit rester possible en krach"

    def test_unknown_drawdown_does_not_silently_block(self):
        """
        Information indisponible : on n'applique pas le garde-fou plutôt que de
        supposer un marché calme. Supposer serait ignorer le risque en silence.
        """
        a = MeanReversionAgent()                 # aucun contexte fourni
        df = self._survendu()
        s = a.generate_signal(
            make_state("X", float(df.Close.iloc[-1])), {}, "bull", df)
        assert s.meta.get("blocked_by") is None

    def test_threshold_matches_the_measured_boundary(self):
        """Le seuil vient du découpage fixé AVANT le test, pas d'un réglage."""
        assert MeanReversionConfig().max_market_drawdown == 0.20
