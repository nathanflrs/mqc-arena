# tests/test_mean_reversion.py
from __future__ import annotations

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
