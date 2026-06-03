# tests/test_trend_following.py
from __future__ import annotations

import pytest

from src.agents.trend_following import TrendFollowingAgent, TrendFollowingConfig
from tests.conftest import make_state


@pytest.fixture
def agent():
    return TrendFollowingAgent()


def test_no_data_returns_hold(agent):
    sig = agent.generate_signal(make_state(), {}, regime="bull", data=None)
    assert sig.action == "HOLD"


def test_short_history_returns_hold(agent, short_df):
    sig = agent.generate_signal(make_state(), {}, regime="bull", data=short_df)
    assert sig.action == "HOLD"


def test_bear_regime_no_new_longs(agent, bull_df):
    sig = agent.generate_signal(make_state(), {}, regime="bear", data=bull_df)
    assert sig.action != "BUY"


def test_downtrend_in_position_triggers_sell(agent, bear_df):
    portfolio = {"AAPL": 0.10}
    sig = agent.generate_signal(make_state(price=float(bear_df["Close"].iloc[-1])), portfolio, regime="bull", data=bear_df)
    assert sig.action == "SELL"


def test_bull_trend_returns_buy_or_hold(agent, bull_df):
    sig = agent.generate_signal(make_state(), {}, regime="bull", data=bull_df)
    assert sig.action in ("BUY", "HOLD")


def test_meta_fields_present(agent, bull_df):
    sig = agent.generate_signal(make_state(), {}, regime="bull", data=bull_df)
    for key in ("regime", "price", "sma20", "sma50", "sma200", "adx"):
        assert key in sig.meta


def test_config_overrides():
    cfg = TrendFollowingConfig(adx_threshold=20.0, target_weight=0.15)
    agent = TrendFollowingAgent(config=cfg)
    assert agent.cfg.adx_threshold == 20.0
    assert agent.cfg.target_weight == 0.15


def test_confidence_bounded(agent, bull_df):
    sig = agent.generate_signal(make_state(), {}, regime="bull", data=bull_df)
    assert 0.0 <= sig.confidence <= 1.0


def test_sell_confidence_from_config(agent, bear_df):
    portfolio = {"AAPL": 0.10}
    sig = agent.generate_signal(make_state(), portfolio, regime="bull", data=bear_df)
    if sig.action == "SELL":
        assert sig.confidence == agent.cfg.sell_confidence
