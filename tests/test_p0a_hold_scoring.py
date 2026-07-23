# tests/test_p0a_hold_scoring.py
"""
P0(a) — HOLD scorable et capable de gagner.

Régression MSFT 23/07/2026 : MeanReversionAgent (agent prioritaire par Sharpe)
émettait HOLD conf=0.30, EarningsSentimentAgent émettait BUY conf=0.68.
HOLD scorait 0, donc ESA BUY gagnait inconditionnellement.

Après correction : le HOLD de l'agent prioritaire doit l'emporter sur un BUY
faible d'un agent non prioritaire.
"""
from __future__ import annotations

import pytest

from src.agents.base import AgentSignal
from src.arena.selector import score_signal, select_best


# ── Helpers ──────────────────────────────────────────────────────────────────

def _sig(agent, action, confidence, target_weight=0.0, symbol="MSFT"):
    return AgentSignal(
        agent_name=agent,
        symbol=symbol,
        action=action,
        confidence=confidence,
        target_weight=target_weight,
    )


# ── Tests ─────────────────────────────────────────────────────────────────────

def test_hold_produces_nonzero_score():
    """Un HOLD d'un agent à confiance > 0 doit scorer > 0 (pas une abstention)."""
    sig = _sig("MeanReversionAgent", "HOLD", 0.30)
    assert score_signal(sig) > 0.0, (
        "HOLD devrait scorer confidence × 0.05 = 0.015, pas 0"
    )


def test_dummy_hold_stays_zero():
    """DummyHoldAgent (conf=0.0) score toujours 0 et ne peut pas gagner."""
    sig = _sig("DummyHoldAgent", "HOLD", 0.0)
    assert score_signal(sig) == 0.0


def test_priority_hold_beats_weak_non_priority_buy():
    """
    Régression MSFT 23/07/2026.

    Avant correction : ESA BUY gagne (HOLD score 0, pas de bonus priority).
    Après correction : HOLD prioritaire gagne (score 0.015 + bonus 0.15 = 0.165 > 0.068).
    """
    signals = [
        _sig("MeanReversionAgent", "HOLD", 0.30),               # prioritaire
        _sig("EarningsSentimentAgent", "BUY", 0.68, 0.10),       # non-prioritaire
        _sig("BuffettAgent", "BUY", 0.90, 0.10),                  # non-prioritaire fort
        _sig("MacroAgent", "HOLD", 0.40),                          # non-prioritaire HOLD
    ]
    winner = select_best(signals, priority_agent="MeanReversionAgent")
    assert winner is not None, "Doit y avoir un gagnant"
    assert winner.action == "HOLD", (
        f"Attendu HOLD (MeanReversionAgent prioritaire), obtenu {winner.action} "
        f"({winner.agent_name})"
    )
    assert winner.agent_name == "MeanReversionAgent"


def test_high_conviction_priority_buy_still_wins():
    """
    Le correctif ne doit pas bloquer un BUY fort de l'agent prioritaire lui-même.
    Si l'agent prioritaire émet BUY à haute conviction, il doit gagner.
    """
    signals = [
        _sig("BuffettAgent", "BUY", 0.90, 0.10),   # prioritaire, BUY fort
        _sig("EarningsSentimentAgent", "BUY", 0.68, 0.10),
        _sig("MacroAgent", "HOLD", 0.40),
    ]
    winner = select_best(signals, priority_agent="BuffettAgent")
    assert winner is not None
    assert winner.action == "BUY"
    assert winner.agent_name == "BuffettAgent"


def test_non_priority_hold_cannot_beat_typical_buy():
    """
    Un HOLD d'un agent NON prioritaire ne doit pas l'emporter sur un BUY raisonnable,
    faute de quoi on bloque l'arène.
    """
    signals = [
        _sig("MacroAgent", "HOLD", 0.40),                  # non-prioritaire
        _sig("BuffettAgent", "BUY", 0.55, 0.10),           # prioritaire, BUY modéré
    ]
    # MacroAgent HOLD : conf=0.40 × 0.05 = 0.020 (pas de bonus)
    # BuffettAgent BUY : conf=0.55 × 0.10 = 0.055 + bonus 0.15 = 0.205
    winner = select_best(signals, priority_agent="BuffettAgent")
    assert winner is not None
    assert winner.action == "BUY"
    assert winner.agent_name == "BuffettAgent"
