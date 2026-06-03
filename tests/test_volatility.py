# tests/test_volatility.py
from __future__ import annotations

from unittest.mock import patch

import pytest

from src.agents.volatility import VolatilityAgent, VolatilityConfig
from tests.conftest import make_state


@pytest.fixture
def agent():
    return VolatilityAgent()


def _mock_vix(vix: float, zscore: float, mom5: float, vix_ratio: float):
    """Retourne un patcher qui court-circuite _analyze_vix."""
    return patch.object(
        VolatilityAgent,
        "_analyze_vix",
        return_value={
            "vix": vix,
            "vix_ma20": vix * 0.9,
            "zscore": zscore,
            "mom5": mom5,
            "vix_ratio": vix_ratio,
        },
    )


def test_vix_spike_triggers_sell(agent):
    with _mock_vix(vix=30.0, zscore=1.0, mom5=0.25, vix_ratio=1.5):
        portfolio = {"AAPL": 0.10}
        sig = agent.generate_signal(make_state(), portfolio, regime="bull")
    assert sig.action == "SELL"
    assert sig.confidence == agent.cfg.sell_confidence_spike


def test_extreme_vix_triggers_buy(agent):
    with _mock_vix(vix=40.0, zscore=2.0, mom5=0.01, vix_ratio=1.0):
        sig = agent.generate_signal(make_state(), {}, regime="bear")
    assert sig.action == "BUY"
    assert sig.confidence == agent.cfg.buy_confidence_extreme
    assert sig.target_weight == agent.cfg.target_weight_extreme


def test_fear_vix_triggers_buy(agent):
    with _mock_vix(vix=28.0, zscore=1.2, mom5=0.02, vix_ratio=1.1):
        sig = agent.generate_signal(make_state(), {}, regime="bull")
    assert sig.action == "BUY"
    assert sig.target_weight == agent.cfg.target_weight_fear


def test_complacency_triggers_sell(agent):
    with _mock_vix(vix=12.0, zscore=-2.0, mom5=0.01, vix_ratio=0.9):
        portfolio = {"AAPL": 0.10}
        sig = agent.generate_signal(make_state(), portfolio, regime="bull")
    assert sig.action == "SELL"
    assert sig.confidence == agent.cfg.sell_confidence_complacency


def test_neutral_vix_returns_hold(agent):
    with _mock_vix(vix=20.0, zscore=0.2, mom5=0.01, vix_ratio=1.0):
        sig = agent.generate_signal(make_state(), {}, regime="bull")
    assert sig.action == "HOLD"


def test_config_overrides():
    cfg = VolatilityConfig(vix_fear_threshold=20.0, target_weight_fear=0.12)
    agent = VolatilityAgent(config=cfg)
    assert agent.cfg.vix_fear_threshold == 20.0
    assert agent.cfg.target_weight_fear == 0.12


def test_confidence_bounded(agent):
    with _mock_vix(vix=28.0, zscore=1.2, mom5=0.02, vix_ratio=1.0):
        sig = agent.generate_signal(make_state(), {}, regime="bull")
    assert 0.0 <= sig.confidence <= 1.0
