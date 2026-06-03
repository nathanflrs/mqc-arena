# src/agents/base.py
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Optional, Literal, Any
from abc import ABC, abstractmethod


Action = Literal["BUY", "SELL", "HOLD"]


@dataclass(frozen=True)
class MarketState:
    """
    Snapshot minimal du marché pour un ticker donné.
    (On ajoutera des champs plus tard sans casser l'interface.)
    """
    symbol: str
    price: float
    timestamp: str  # ISO string


@dataclass(frozen=True)
class AgentSignal:
    """
    Ce que renvoie un agent à l'arène.
    - action: BUY/SELL/HOLD
    - confidence: 0.0 → 1.0
    - target_weight: exposition souhaitée (0.0 → 1.0) si BUY/SELL
    - reason: explication courte pour logs/Telegram
    - meta: infos additionnelles (indicateurs, features, etc.)
    """
    agent_name: str
    symbol: str
    action: Action
    confidence: float
    target_weight: float
    reason: str = ""
    meta: Dict[str, Any] = field(default_factory=dict)



class BaseAgent(ABC):
    """
    Contrat que TOUS les agents doivent respecter.
    L'agent NE TRADE PAS. Il PROPOSE un signal.
    """

    name: str = "BaseAgent"

    @abstractmethod
    def generate_signal(
        self,
        state: MarketState,
        portfolio: Dict[str, float],
        regime: Optional[str] = None,
    ) -> AgentSignal:
        """
        Retourne un AgentSignal.
        - state: état marché du ticker
        - portfolio: expositions actuelles {symbol: weight} ou {symbol: position_value} (v1 on choisit weight)
        - regime: bull/choppy/bear (optionnel)
        """
        raise NotImplementedError
