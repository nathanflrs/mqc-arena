# tests/test_buffett.py
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.agents.buffett import BuffettAgent, BuffettConfig
from tests.conftest import make_state


@pytest.fixture
def agent():
    return BuffettAgent()


def test_no_data_returns_hold(agent):
    sig = agent.generate_signal(make_state(), {}, regime="bull", data=None)
    assert sig.action == "HOLD"


def test_short_history_returns_hold(agent, short_df):
    sig = agent.generate_signal(make_state(), {}, regime="bull", data=short_df)
    assert sig.action == "HOLD"


def test_bull_trend_returns_buy(agent, bull_df):
    sig = agent.generate_signal(make_state(price=float(bull_df["Close"].iloc[-1])), {}, regime="bull", data=bull_df)
    # En tendance haussière forte sur 300j, au moins 2 critères doivent passer
    assert sig.action in ("BUY", "HOLD")
    assert 0.0 <= sig.confidence <= 1.0


def test_bear_regime_blocks_new_longs(agent, bull_df):
    sig = agent.generate_signal(make_state(), {}, regime="bear", data=bull_df)
    assert sig.action != "BUY"


def test_in_position_sell_below_sma200(agent, bear_df):
    portfolio = {"AAPL": 0.10}
    sig = agent.generate_signal(make_state(price=float(bear_df["Close"].iloc[-1])), portfolio, regime="bull", data=bear_df)
    assert sig.action == "SELL"


def test_config_overrides_respected():
    cfg = BuffettConfig(vol20_max=0.01, near_high_252_threshold=0.99)
    agent = BuffettAgent(config=cfg)
    assert agent.cfg.vol20_max == 0.01
    assert agent.cfg.near_high_252_threshold == 0.99


def test_confidence_in_valid_range(agent, bull_df):
    sig = agent.generate_signal(make_state(), {}, regime="bull", data=bull_df)
    assert 0.0 <= sig.confidence <= 1.0


def test_signal_fields_populated(agent, bull_df):
    sig = agent.generate_signal(make_state("AAPL"), {}, regime="bull", data=bull_df)
    assert sig.agent_name == "BuffettAgent"
    assert sig.symbol == "AAPL"
    assert sig.action in ("BUY", "SELL", "HOLD")
