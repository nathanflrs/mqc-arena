# src/arena/selector.py
from __future__ import annotations

from typing import List, Optional, Set
from src.agents.base import AgentSignal


# Poids plancher : ce que « vaut » une conviction qui ne demande aucune taille.
# Sert de plancher aux BUY/SELL sans target_weight, et de poids fixe aux HOLD.
FLOOR_WEIGHT = 0.05


def score_signal(sig: AgentSignal) -> float:
    """
    Convertit un signal en score numérique cohérent.
    BUY / SELL : confidence × max(target_weight, FLOOR_WEIGHT)
    HOLD       : confidence × FLOOR_WEIGHT

    HOLD n'est pas une abstention — c'est un avis de ne pas agir.
    Son score est structurellement plus faible qu'un BUY/SELL typique
    (×0.05 vs ×0.10+), mais peut l'emporter si l'agent est prioritaire
    (bonus conf×priority_bonus dans select_best, proportionnel à la conviction).

    Le HOLD est scoré sur un poids **fixe**, jamais sur son target_weight :
    un avis de ne rien faire ne se voit pas créditer d'une taille de position.
    En pratique tous les agents émettent target_weight=0 sur HOLD, donc ce
    branchement ne change rien sur données réelles — il empêche simplement
    qu'un futur agent émettant un HOLD avec un poids résiduel remporte
    l'arène pour une mauvaise raison.
    """
    if sig.action == "HOLD":
        return sig.confidence * FLOOR_WEIGHT
    return sig.confidence * max(sig.target_weight, FLOOR_WEIGHT)


def _corroboration_ok(
    winner: AgentSignal,
    all_signals: List[AgentSignal],
    qualified_voters: Set[str],
    abstain_threshold: float,
    min_quorum: int,
) -> bool:
    """
    Vérifie que le gagnant n'est pas isolé contre une majorité qualifiée active.

    Un agent qualifié est 'actif' si sa confidence (déjà normalisée) > abstain_threshold.
    La règle se déclenche uniquement si au moins min_quorum agents qualifiés sont actifs.

    Retourne False (bloqué) si :
      - Le gagnant a au plus 1 vote qualifié actif en sa faveur
      - ET au moins min_quorum agents qualifiés actifs existent
      - ET la majorité stricte de ces agents actifs s'opposent au gagnant

    DividendArbitrageAgent (override absolu) n'atteint jamais ce point.
    """
    active = [s for s in all_signals
              if s.agent_name in qualified_voters
              and s.confidence > abstain_threshold]

    if len(active) < min_quorum:
        return True  # Quorum insuffisant → pas de blocage

    n_support = sum(1 for s in active if s.action == winner.action)
    n_oppose  = len(active) - n_support

    if n_support <= 1 and n_oppose > len(active) / 2:
        return False  # Gagnant isolé face à majorité qualifiée → bloqué

    return True


def select_best(
    signals: List[AgentSignal],
    min_score: float = 0.02,
    priority_agent: Optional[str] = None,
    priority_bonus: float = 0.15,
    qualified_voters: Optional[Set[str]] = None,
    abstain_threshold: float = 0.25,
    min_quorum: int = 2,
) -> Optional[AgentSignal]:
    """
    Sélectionne le meilleur signal s'il dépasse un seuil minimum.

    priority_agent    : cet agent reçoit un bonus de score proportionnel à sa conviction
    priority_bonus    : facteur du bonus (conf × priority_bonus)
    qualified_voters  : agents éligibles au quorum de corroboration (P0c)
    abstain_threshold : confidence normalisée minimale pour compter comme vote actif
    min_quorum        : nombre minimal d'agents qualifiés actifs pour déclencher la règle

    Override absolu : DividendArbitrageAgent retourne immédiatement (hors quorum).
    """
    if not signals:
        return None

    # Absolute override: DividendArbitrageAgent during its active dividend window
    for s in signals:
        if s.meta.get("div_arb_priority") and s.action in ("BUY", "SELL"):
            return s

    scored = []
    for s in signals:
        score = score_signal(s)

        # Bonus proportionnel si c'est l'agent prioritaire pour ce symbole.
        # Proportionnel à la conviction : un HOLD à 10% n'écrase pas un BUY solide,
        # un HOLD à 70%+ le fait légitimement.
        if priority_agent and s.agent_name == priority_agent and score > 0:
            score += s.confidence * priority_bonus

        scored.append((score, s))

    scored.sort(key=lambda x: (x[0], x[1].action == "BUY"), reverse=True)

    best_score, best_sig = scored[0]

    if best_score <= 0.0 or best_score < min_score:
        return None

    # P0(c) — Règle de corroboration
    if qualified_voters and not _corroboration_ok(
        best_sig, signals, qualified_voters, abstain_threshold, min_quorum
    ):
        return None  # Bloqué : signal isolé contre majorité qualifiée

    return best_sig
