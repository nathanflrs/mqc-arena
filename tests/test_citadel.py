# tests/test_citadel.py
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.agents.citadel import CitadelAgent, CitadelConfig
from tests.conftest import make_state


@pytest.fixture
def agent():
    return CitadelAgent()


def test_no_data_returns_hold(agent):
    sig = agent.generate_signal(make_state(), {}, regime="bull", data=None)
    assert sig.action == "HOLD"
    assert sig.confidence == 0.0


def test_short_history_returns_hold(agent, short_df):
    sig = agent.generate_signal(make_state(), {}, regime="bull", data=short_df)
    assert sig.action == "HOLD"


def test_bear_regime_no_new_longs(agent, bull_df):
    sig = agent.generate_signal(make_state(), {}, regime="bear", data=bull_df)
    assert sig.action != "BUY"


def test_sell_triggered_in_downtrend(agent, bear_df):
    portfolio = {"AAPL": 0.12}
    sig = agent.generate_signal(make_state(price=float(bear_df["Close"].iloc[-1])), portfolio, regime="bull", data=bear_df)
    assert sig.action == "SELL"
    assert sig.confidence == CitadelAgent().cfg.sell_confidence


def test_config_overrides():
    cfg = CitadelConfig(target_weight=0.20, mom63_entry=0.10)
    agent = CitadelAgent(config=cfg)
    assert agent.cfg.target_weight == 0.20
    assert agent.cfg.mom63_entry == 0.10


def test_signal_meta_populated(agent, bull_df):
    sig = agent.generate_signal(make_state(), {}, regime="bull", data=bull_df)
    assert "regime" in sig.meta


def test_confidence_bounded(agent, bull_df):
    sig = agent.generate_signal(make_state(), {}, regime="bull", data=bull_df)
    assert 0.0 <= sig.confidence <= 1.0


def test_agent_name(agent):
    assert agent.name == "CitadelAgent"
