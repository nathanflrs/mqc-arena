# src/arena/selector.py
from __future__ import annotations

from typing import List, Optional
from src.agents.base import AgentSignal


def score_signal(sig: AgentSignal) -> float:
    """
    Convertit un signal en score numérique cohérent.
    BUY  : confidence * target_weight
    SELL : confidence * target_weight (symétrique)
    HOLD : 0.0 (jamais sélectionné)
    """
    if sig.action in ("BUY", "SELL"):
        return sig.confidence * max(sig.target_weight, 0.05)
    return 0.0


def select_best(
    signals: List[AgentSignal],
    min_score: float = 0.02,
    priority_agent: Optional[str] = None,
    priority_bonus: float = 0.15,
) -> Optional[AgentSignal]:
    """
    Sélectionne le meilleur signal s'il dépasse un seuil minimum.

    priority_agent : si spécifié, cet agent reçoit un bonus de score
                     basé sur les résultats du backtest
    priority_bonus : bonus appliqué au score de l'agent prioritaire
    """
    if not signals:
        return None

    scored = []
    for s in signals:
        score = score_signal(s)

        # Bonus si c'est l'agent prioritaire pour ce symbole
        if priority_agent and s.agent_name == priority_agent and score > 0:
            score += priority_bonus

        scored.append((score, s))

    scored.sort(key=lambda x: (x[0], x[1].action == "BUY"), reverse=True)

    best_score, best_sig = scored[0]

    if best_score <= 0.0 or best_score < min_score:
        return None

    return best_sig