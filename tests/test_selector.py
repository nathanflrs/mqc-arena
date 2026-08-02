# tests/test_selector.py
from __future__ import annotations

import pytest

from src.agents.base import AgentSignal
from src.arena.selector import FLOOR_WEIGHT, score_signal, select_best


def _sig(name: str, action: str, confidence: float, weight: float) -> AgentSignal:
    return AgentSignal(
        agent_name=name,
        symbol="AAPL",
        action=action,
        confidence=confidence,
        target_weight=weight,
    )


def test_hold_scored_on_floor_weight_not_its_own():
    """
    P0(a) : un HOLD n'est plus un score nul, mais il est scoré sur le poids
    plancher — jamais sur son propre target_weight. Un avis de ne rien faire
    ne se voit pas créditer d'une taille de position.
    """
    sig = _sig("A", "HOLD", 0.9, 0.10)
    assert score_signal(sig) == pytest.approx(0.9 * FLOOR_WEIGHT)
    assert score_signal(sig) != pytest.approx(0.9 * 0.10)


def test_hold_score_ignores_absurd_weight():
    """Garde-fou : même un HOLD à target_weight=1.0 reste plafonné au plancher."""
    assert score_signal(_sig("A", "HOLD", 1.0, 1.0)) == pytest.approx(FLOOR_WEIGHT)


def test_hold_loses_to_equivalent_buy():
    """À conviction égale, agir l'emporte sur ne pas agir (0.10 > 0.05)."""
    buy  = _sig("A", "BUY",  0.70, 0.10)
    hold = _sig("B", "HOLD", 0.70, 0.10)
    assert score_signal(buy) > score_signal(hold)
    assert select_best([buy, hold]).agent_name == "A"


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


def test_prioritized_hold_beats_weak_buy():
    """
    Régression du cas MSFT du 2026-07-23 (docs/audit_2026-07-23.md, Q1).

    Avant P0(a) : MeanReversionAgent était l'agent prioritaire sur MSFT et
    émettait HOLD → score 0. EarningsSentimentAgent (LLM, conf 0.68) émettait
    BUY et gagnait l'arène par défaut, sans avoir à battre qui que ce soit.

    Après P0(a) : le HOLD de l'agent prioritaire porte un score
    (0.70×0.05 + 0.70×0.15 = 0.14) supérieur au BUY faible (0.68×0.10 = 0.068).
    L'avis « ne rien faire » de l'agent qui connaît le mieux ce symbole
    l'emporte, ce qui est le comportement voulu.
    """
    hold_prio = _sig("MeanReversionAgent",     "HOLD", 0.70, 0.0)
    weak_buy  = _sig("EarningsSentimentAgent", "BUY",  0.68, 0.10)

    best = select_best([hold_prio, weak_buy], priority_agent="MeanReversionAgent")
    assert best is not None
    assert best.agent_name == "MeanReversionAgent"
    assert best.action == "HOLD"

    # Sans priorité, le BUY reprend la main : le bonus est bien ce qui bascule.
    assert select_best([hold_prio, weak_buy]).agent_name == "EarningsSentimentAgent"


def test_prioritized_hold_does_not_beat_a_strong_buy():
    """Le bonus de priorité ne doit pas rendre un HOLD imbattable."""
    hold_prio  = _sig("MeanReversionAgent", "HOLD", 0.40, 0.0)   # 0.40×0.05 + 0.40×0.15 = 0.08
    strong_buy = _sig("BuffettAgent",       "BUY",  0.90, 0.15)  # 0.135
    best = select_best([hold_prio, strong_buy], priority_agent="MeanReversionAgent")
    assert best.agent_name == "BuffettAgent"


def test_buy_preferred_over_sell_on_tie():
    buy = _sig("A", "BUY", 0.80, 0.10)
    sell = _sig("B", "SELL", 0.80, 0.10)
    # Les deux ont le même score numérique, BUY doit être préféré
    best = select_best([buy, sell])
    assert best.action == "BUY"


def test_empty_signals_returns_none():
    assert select_best([]) is None
