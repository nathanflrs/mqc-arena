# tests/test_p0a_hold_scoring.py
"""
P0(a) — HOLD scorable avec bonus proportionnel à la conviction.

Deux invariants clés :
  1. HOLD n'est plus une abstention (score > 0 si conf > 0)
  2. Le bonus de priorité est proportionnel : conf × 0.15, pas +0.15 flat.
     Un HOLD à faible conviction ne peut pas écraser un BUY solide ;
     un HOLD à haute conviction le peut légitimement.

Seuil de bascule avec tw=0.10 pour le BUY concurrent :
  conf_hold × 0.20 > conf_buy × 0.10  →  conf_hold > conf_buy / 2
  Exemple : BUY conf=0.68 → HOLD doit avoir conf > 0.34 pour gagner.
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
    """Un HOLD à confiance > 0 doit scorer > 0 (pas une abstention)."""
    sig = _sig("MeanReversionAgent", "HOLD", 0.30)
    assert score_signal(sig) > 0.0, "HOLD devrait scorer conf × 0.05, pas 0"


def test_dummy_hold_stays_zero():
    """DummyHoldAgent (conf=0.0) score toujours 0 et ne peut pas gagner."""
    sig = _sig("DummyHoldAgent", "HOLD", 0.0)
    assert score_signal(sig) == 0.0


def test_high_conviction_priority_hold_beats_typical_buy():
    """
    Un HOLD prioritaire à conviction élevée (≥0.50) bat un BUY non-prioritaire typique.

    Scores :
      MeanReversionAgent HOLD conf=0.50 : 0.50×0.05 + 0.50×0.15 = 0.025 + 0.075 = 0.100
      EarningsSentimentAgent BUY conf=0.68, tw=0.10 :  0.68×0.10 = 0.068
      → HOLD gagne (0.100 > 0.068)
    """
    signals = [
        _sig("MeanReversionAgent", "HOLD", 0.50),
        _sig("EarningsSentimentAgent", "BUY", 0.68, 0.10),
        _sig("MacroAgent", "HOLD", 0.40),
    ]
    winner = select_best(signals, priority_agent="MeanReversionAgent")
    assert winner is not None
    assert winner.action == "HOLD"
    assert winner.agent_name == "MeanReversionAgent"


def test_low_conviction_priority_hold_loses_to_typical_buy():
    """
    Un HOLD prioritaire à faible conviction (0.30) perd face à un BUY typique.

    Scores :
      MeanReversionAgent HOLD conf=0.30 : 0.30×0.05 + 0.30×0.15 = 0.015 + 0.045 = 0.060
      EarningsSentimentAgent BUY conf=0.68, tw=0.10 :  0.68×0.10 = 0.068
      → BUY gagne (0.068 > 0.060)

    C'est le cas MSFT 23/07 : conf=0.30 est insuffisant pour bloquer un BUY raisonnable.
    """
    signals = [
        _sig("MeanReversionAgent", "HOLD", 0.30),
        _sig("EarningsSentimentAgent", "BUY", 0.68, 0.10),
    ]
    winner = select_best(signals, priority_agent="MeanReversionAgent")
    assert winner is not None
    assert winner.action == "BUY"
    assert winner.agent_name == "EarningsSentimentAgent"


def test_high_conviction_priority_buy_still_wins():
    """
    Le correctif ne bloque pas un BUY fort de l'agent prioritaire.

    Scores :
      BuffettAgent BUY conf=0.90, tw=0.10 : 0.90×0.10 + 0.90×0.15 = 0.090 + 0.135 = 0.225
      EarningsSentimentAgent BUY conf=0.68, tw=0.10 : 0.068
      → BuffettAgent BUY gagne
    """
    signals = [
        _sig("BuffettAgent", "BUY", 0.90, 0.10),
        _sig("EarningsSentimentAgent", "BUY", 0.68, 0.10),
        _sig("MacroAgent", "HOLD", 0.40),
    ]
    winner = select_best(signals, priority_agent="BuffettAgent")
    assert winner is not None
    assert winner.action == "BUY"
    assert winner.agent_name == "BuffettAgent"


def test_non_priority_hold_cannot_beat_typical_buy():
    """
    Un HOLD non-prioritaire ne peut pas l'emporter sur un BUY de l'agent prioritaire.

    Scores :
      MacroAgent HOLD conf=0.40 (non-prioritaire) : 0.40×0.05 = 0.020
      BuffettAgent BUY conf=0.55, tw=0.10 (prioritaire) : 0.55×0.10 + 0.55×0.15 = 0.055 + 0.0825 = 0.1375
      → BUY gagne
    """
    signals = [
        _sig("MacroAgent", "HOLD", 0.40),
        _sig("BuffettAgent", "BUY", 0.55, 0.10),
    ]
    winner = select_best(signals, priority_agent="BuffettAgent")
    assert winner is not None
    assert winner.action == "BUY"
    assert winner.agent_name == "BuffettAgent"
