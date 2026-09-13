# src/agents/universe_base.py
"""
Contrat des agents d'univers (2026-09-13).

Les agents historiques sont interrogés titre par titre, sur 11 titres. Un agent
d'univers reçoit tout le marché d'un coup et renvoie ses propositions du jour :
le plus souvent aucune, parfois quelques-unes. C'est la forme naturelle des
agents d'événements — un achat groupé de dirigeants ne concerne qu'une poignée
de sociétés sur 500 — et des paires, qui comparent des titres entre eux.

Comme les autres, un agent d'univers ne trade pas : il propose.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, List

from src.agents.base import AgentSignal

if TYPE_CHECKING:
    from src.market.view import MarketView


class UniverseAgent(ABC):
    name: str = "UniverseAgent"

    @abstractmethod
    def propose(self, view: "MarketView") -> List[AgentSignal]:
        """
        Les propositions du jour sur l'univers entier — BUY ou SELL uniquement.

        Une liste vide est la réponse normale d'un jour sans événement, pas un
        échec. Une panne doit lever une exception : l'appelant la signalera au
        lieu de la confondre avec un jour calme.
        """
        raise NotImplementedError
