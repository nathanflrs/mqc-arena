# tests/test_p0c_corroboration.py
"""
P0(c) — Règle de corroboration : un signal isolé ne l'emporte pas contre
une majorité qualifiée d'agents indépendants en désaccord.

La liste des votants qualifiés vit dans logs/qualified_voters.json (versionné).
P0(c) lit ce fichier — aucune hypothèse codée en dur dans select_best.

Paramètres (lus depuis le JSON, testés ici en injection directe) :
  abstain_threshold = 0.25  (conf normalisée min pour compter comme vote actif)
  min_quorum        = 2     (au moins 2 agents qualifiés actifs requis)

Calibration historique (164 décisions P0(a+b)) :
  3 cas bloqués (1.8%) — deux MacroAgent BUY sur GLD contre 2 qualifiés HOLD,
  et un BuffettAgent BUY contre MeanReversionAgent SELL + ESA HOLD.
"""
from __future__ import annotations

import pytest

from src.agents.base import AgentSignal
from src.arena.selector import select_best

QUALIFIED = {
    "BuffettAgent", "CitadelAgent", "MeanReversionAgent",
    "TrendFollowingAgent", "PairsTradingAgent",
    "EarningsSentimentAgent", "CrossSectionalMomentumAgent",
}
ABSTAIN = 0.25
QUORUM  = 2


def _sig(agent, action, confidence, target_weight=0.10, symbol="SPY"):
    return AgentSignal(
        agent_name=agent, symbol=symbol, action=action,
        confidence=confidence, target_weight=target_weight,
    )


# ── Tests ─────────────────────────────────────────────────────────────────────

def test_isolated_winner_blocked_by_qualified_majority():
    """
    Cas GLD historique : MacroAgent BUY gagne sur score, mais 2 agents qualifiés
    actifs disent HOLD → bloqué.

    Sans P0(c) : MacroAgent gagne (il n'est pas dans qualified, mais a le score max).
    Avec P0(c) : bloqué (0 supporter qualifié, 2 opposants qualifiés actifs).
    """
    signals = [
        _sig("MacroAgent",      "BUY",  1.00, 0.05),   # max normalisé, non-qualifié
        _sig("BuffettAgent",    "HOLD", 0.36),          # qualifié, actif (0.36 > 0.25)
        _sig("PairsTradingAgent","HOLD",0.39),          # qualifié, actif (0.39 > 0.25)
        _sig("MeanReversionAgent","HOLD",0.00),         # qualifié mais abstention (0.00 ≤ 0.25)
    ]

    # Sans corroboration : MacroAgent gagne (score le plus haut)
    winner_raw = select_best(signals)
    assert winner_raw is not None
    assert winner_raw.agent_name == "MacroAgent"

    # Avec P0(c) : MacroAgent isolé (0 qualifié BUY) contre majorité (2 qualifiés HOLD) → bloqué
    winner_poc = select_best(signals, qualified_voters=QUALIFIED,
                             abstain_threshold=ABSTAIN, min_quorum=QUORUM)
    assert winner_poc is None, "MacroAgent isolé doit être bloqué"


def test_corroborated_winner_not_blocked():
    """
    Si au moins 1 autre agent qualifié actif soutient le gagnant,
    la règle ne se déclenche pas (n_support=2 > 1).

    MacroAgent BUY tw=0.15 → score 1.00×0.15=0.150 (gagnant sur score).
    BuffettAgent BUY tw=0.10 → score 0.80×0.10=0.080 (qualifié, soutient BUY → n_support=2).
    CitadelAgent HOLD → oppose.
    Résultat : n_support=2 > 1 → NOT bloqué, MacroAgent gagne.
    """
    signals = [
        _sig("MacroAgent",   "BUY", 1.00, 0.15),   # score 0.150, gagne
        _sig("BuffettAgent", "BUY", 0.80, 0.10),   # qualifié actif, soutient BUY
        _sig("CitadelAgent", "HOLD", 0.50),          # qualifié, oppose
    ]
    winner = select_best(signals, qualified_voters=QUALIFIED,
                         abstain_threshold=ABSTAIN, min_quorum=QUORUM)
    assert winner is not None
    assert winner.agent_name == "MacroAgent"


def test_quorum_not_reached_no_block():
    """
    Avec un seul agent qualifié actif (quorum=2 non atteint), pas de blocage.
    """
    signals = [
        _sig("MacroAgent",   "BUY",  1.00, 0.05),
        _sig("BuffettAgent", "HOLD", 0.36),    # 1 seul qualifié actif → quorum insuffisant
        _sig("MeanReversionAgent", "HOLD", 0.00),  # abstention
    ]
    winner = select_best(signals, qualified_voters=QUALIFIED,
                         abstain_threshold=ABSTAIN, min_quorum=QUORUM)
    assert winner is not None  # pas bloqué : quorum non atteint


def test_abstaining_agents_dont_count_as_opponents():
    """
    Les agents qualifiés sous le seuil d'abstention ne comptent pas comme opposants.
    MeanReversionAgent à conf normalisée = 0.0 ne doit pas former un quorum.
    """
    signals = [
        _sig("BuffettAgent",     "BUY",  1.00, 0.10),   # qualifié, gagnant
        _sig("MeanReversionAgent","HOLD", 0.00),         # qualifié mais abstention
        _sig("CitadelAgent",     "HOLD",  0.00),         # qualifié mais abstention
    ]
    winner = select_best(signals, qualified_voters=QUALIFIED,
                         abstain_threshold=ABSTAIN, min_quorum=QUORUM)
    assert winner is not None
    assert winner.agent_name == "BuffettAgent"


def test_div_arb_override_bypasses_corroboration():
    """
    DividendArbitrageAgent (override absolu) retourne immédiatement,
    avant que la corroboration soit vérifiée.
    """
    from src.agents.base import AgentSignal as AS
    div_sig = AS(
        agent_name="DividendArbitrageAgent", symbol="AAPL",
        action="BUY", confidence=0.20, target_weight=0.05,
        meta={"div_arb_priority": True},
    )
    signals = [
        div_sig,
        _sig("BuffettAgent",     "HOLD", 0.80, symbol="AAPL"),
        _sig("CitadelAgent",     "HOLD", 0.80, symbol="AAPL"),
        _sig("MeanReversionAgent","HOLD",0.80, symbol="AAPL"),
    ]
    winner = select_best(signals, qualified_voters=QUALIFIED,
                         abstain_threshold=ABSTAIN, min_quorum=QUORUM)
    assert winner is not None
    assert winner.agent_name == "DividendArbitrageAgent"
