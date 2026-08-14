# src/arena/consensus.py
"""
Agrégation des avis d'agents — remplaçant candidat de `selector.select_best`.

Ce que faisait l'arène, et pourquoi c'était faux
------------------------------------------------
`select_best` couronne un vainqueur unique par symbole, classé sur
`score = confidence × target_weight`. Ces deux nombres sont des constantes
écrites à la main dans le fichier de chaque agent. L'arène n'arbitrait donc pas
entre dix opinions : elle désignait celui dont l'auteur avait tapé le plus gros
chiffre.

Mesuré sur 13 séances réelles (2026-03-15 → 2026-08-14) : BuffettAgent remporte
128 décisions sur 200 parce qu'il porte `confidence=0.9` et propose d'agir 76 %
du temps. MacroAgent en prend 29 de plus. À eux deux, 157 sur 200 — les huit
autres agents sont décoratifs.

Et la confiance ne porte aucune information. Courbes de calibration mesurées
sur 955 séances (docs/agent_edge.md) :

    Buffett          conf 0.70 → 53.0 %    conf 0.90 → 53.5 %   (plat)
    TrendFollowing   conf 0.69 → 61.3 %    conf 0.90 → 53.6 %   (INVERSÉ)
    MeanReversion    conf 0.70 → 53.2 %    conf 0.90 → 65.3 %   (informatif)

Le seul agent dont la confiance prédit quelque chose est celui qui ne parle
presque jamais — 6 propositions d'agir sur 215 — et qui ne remporte l'arène que
2 fois sur 200. Le mécanisme écartait systématiquement le seul avis calibré.

Trois défauts, et ce que ce module change
-----------------------------------------
1. **Le gagnant emportait tout.** Neuf HOLD contre un BUY produisait un achat
   plein. Ici on agrège : le signal est la moyenne des directions, pas le cri
   du plus fort.

2. **La confiance auto-déclarée servait d'arbitre.** Ici le poids d'un agent
   vient de son avantage MESURÉ. Aucun n'en ayant démontré à ce jour, le poids
   honnête est égal pour tous — ce qui suffit à changer le portefeuille, parce
   que Buffett cesse d'écraser le vote avec un `0.9` codé en dur.

3. **Un avis isolé valait une unanimité.** Ici la taille est amortie par le
   taux de participation : un seul agent qui parle sur onze ne déclenche
   qu'une position réduite.

Ce que ce module NE corrige pas
-------------------------------
La corrélation entre symboles. Chaque titre reste décidé isolément : cinq
mégacaps achetées le même jour restent un pari fait cinq fois. Cela relève de
la construction de portefeuille, pas de l'agrégation d'avis, et c'est un
chantier distinct.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence

from src.agents.base import AgentSignal

# Agents exclus du vote : ce sont des témoins, pas des opinions.
NON_VOTANTS = frozenset({"DummyHoldAgent"})

# En deçà, l'avis agrégé est trop faible ou trop divisé pour agir.
MIN_CONVICTION = 0.15


@dataclass(frozen=True)
class ConsensusSignal:
    """
    Avis agrégé sur un symbole.

    `score` ∈ [−1, +1] : moyenne pondérée des directions parmi les agents qui
    se prononcent. +1 = unanimité acheteuse, −1 = unanimité vendeuse, 0 =
    parfaitement divisé.

    `conviction` ∈ [0, 1] : `|score|` amorti par le taux de participation.
    C'est lui qui dimensionne la position.
    """
    symbol: str
    action: str
    score: float
    conviction: float
    target_weight: float
    n_speaking: int
    n_eligible: int
    participation: float
    votes: Dict[str, str] = field(default_factory=dict)
    reason: str = ""

    def render(self) -> str:
        pour = [a for a, v in self.votes.items() if v == self.action]
        return (f"{self.symbol} {self.action} — score {self.score:+.2f}, "
                f"conviction {self.conviction:.2f}, "
                f"{self.n_speaking}/{self.n_eligible} agents se prononcent "
                f"({', '.join(sorted(pour))})")


def equal_weights(agents: Sequence[str]) -> Dict[str, float]:
    """
    Le poids honnête aujourd'hui : le même pour tout le monde.

    Ce n'est pas un choix par défaut faute de mieux — c'est le résultat de la
    mesure. Aucun agent n'a montré d'avantage dont l'intervalle de confiance
    exclue zéro (docs/agent_edge.md, docs/verdicts_agents.md). Leur accorder
    des poids différents reviendrait à affirmer une hiérarchie qu'on a
    justement échoué à démontrer.

    Le jour où un agent démontre un avantage, cette fonction est remplacée par
    une pondération issue de la mesure — pas d'une intuition.
    """
    votants = [a for a in agents if a not in NON_VOTANTS]
    if not votants:
        return {}
    return {a: 1.0 / len(votants) for a in votants}


def _direction(action: str) -> Optional[int]:
    """+1 acheteur, −1 vendeur, None pour une abstention."""
    if action == "BUY":
        return 1
    if action == "SELL":
        return -1
    return None                      # HOLD = abstention, décidé le 2026-08-14


def aggregate(
    signals: List[AgentSignal],
    weights: Optional[Dict[str, float]] = None,
    base_weight: float = 0.10,
    min_conviction: float = MIN_CONVICTION,
) -> Optional[ConsensusSignal]:
    """
    Agrège les avis d'un symbole en une décision unique.

    HOLD vaut abstention, pas vote contre
    -------------------------------------
    Trois agents sont structurellement muets sur cet univers : InsiderBuy (les
    dirigeants de mégacaps n'achètent pas sur le marché), DividendArbitrage
    (Microsoft verse 0.18 % par trimestre contre un seuil à 0.30 %) et
    Volatility (le VIX ne fournit pas d'extrême en marché calme). Compter leur
    HOLD comme un vote contre reviendrait à laisser trois agents qui n'ont rien
    à dire diluer mécaniquement tout signal — et à noyer InsiderBuy le jour où
    il détecte réellement quelque chose.

    Le prix de ce choix : un agent seul à parler obtient un score de ±1. C'est
    précisément ce que corrige l'amortissement par participation ci-dessous.

    Pourquoi la racine carrée
    -------------------------
    Si chaque agent est une estimation bruitée et indépendante de la même
    direction, l'erreur de la moyenne décroît en 1/√n : la confiance qu'on peut
    accorder au consensus croît donc en √n. Un agent sur onze qui parle donne
    √(1/11) ≈ 0.30, soit 30 % de la taille de base ; cinq agents d'accord
    donnent 0.67 ; l'unanimité donne 1.0.
    """
    utiles = [s for s in signals if s.agent_name not in NON_VOTANTS]
    if not utiles:
        return None

    w = weights or equal_weights([s.agent_name for s in utiles])

    total_signe = 0.0
    total_poids = 0.0
    votes: Dict[str, str] = {}
    parlants = 0

    for s in utiles:
        votes[s.agent_name] = s.action
        d = _direction(s.action)
        if d is None:
            continue
        pw = w.get(s.agent_name, 0.0)
        if pw <= 0:
            continue
        total_signe += d * pw
        total_poids += pw
        parlants += 1

    n_eligibles = len(utiles)
    if parlants == 0 or total_poids <= 0:
        return None

    score = total_signe / total_poids                 # ∈ [−1, +1]
    participation = parlants / n_eligibles
    conviction = abs(score) * math.sqrt(participation)

    if conviction < min_conviction:
        return None

    action = "BUY" if score > 0 else "SELL"
    pour = sorted(a for a, v in votes.items() if v == action)

    return ConsensusSignal(
        symbol=utiles[0].symbol,
        action=action,
        score=score,
        conviction=conviction,
        target_weight=base_weight * conviction,
        n_speaking=parlants,
        n_eligible=n_eligibles,
        participation=participation,
        votes=votes,
        reason=(f"consensus {score:+.2f} sur {parlants}/{n_eligibles} avis "
                f"({', '.join(pour)})"),
    )
