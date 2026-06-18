# tests/test_selector.py
from __future__ import annotations

import pytest

from src.agents.base import AgentSignal
from src.arena.selector import score_signal, select_best


def _sig(name: str, action: str, confidence: float, weight: float) -> AgentSignal:
    return AgentSignal(
        agent_name=name,
        symbol="AAPL",
        action=action,
        confidence=confidence,
        target_weight=weight,
    )


def test_hold_scores_zero():
    sig = _sig("A", "HOLD", 0.9, 0.10)
    assert score_signal(sig) == 0.0


def test_buy_score_formula():
    sig = _sig("A", "BUY", 0.8, 0.10)
    assert score_signal(sig) == pytest.approx(0.8 * 0.10)


def test_sell_score_formula():
    sig = _sig("A", "SELL", 0.85, 0.0)
    # weight=0 → max(0, 0.05) = 0.05
    assert score_signal(sig) == pytest.approx(0.85 * 0.05)


def test_select_best_returns_highest_score():
    signals = [
        _sig("A", "BUY", 0.60, 0.10),
        _sig("B", "BUY", 0.90, 0.12),
        _sig("C", "HOLD", 0.99, 0.10),
    ]
    best = select_best(signals)
    assert best is not None
    assert best.agent_name == "B"


def test_select_best_returns_none_below_threshold():
    signals = [_sig("A", "BUY", 0.01, 0.01)]
    assert select_best(signals, min_score=0.10) is None


def test_priority_bonus_promotes_agent():
    signals = [
        _sig("CitadelAgent", "BUY", 0.60, 0.10),
        _sig("BuffettAgent", "BUY", 0.70, 0.10),
    ]
    # Sans bonus, BuffettAgent gagne (score plus élevé)
    best_no_priority = select_best(signals)
    assert best_no_priority.agent_name == "BuffettAgent"

    # Avec bonus sur CitadelAgent, il doit gagner
    best_with_priority = select_best(signals, priority_agent="CitadelAgent", priority_bonus=0.20)
    assert best_with_priority.agent_name == "CitadelAgent"


def test_hold_never_selected():
    signals = [_sig("A", "HOLD", 1.0, 1.0)]
    assert select_best(signals, min_score=0.0) is None


def test_buy_preferred_over_sell_on_tie():
    buy = _sig("A", "BUY", 0.80, 0.10)
    sell = _sig("B", "SELL", 0.80, 0.10)
    # Les deux ont le même score numérique, BUY doit être préféré
    best = select_best([buy, sell])
    assert best.action == "BUY"


def test_empty_signals_returns_none():
    assert select_best([]) is None
