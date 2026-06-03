# src/arena/arena.py
from __future__ import annotations

from typing import List, Dict
import pandas as pd

from src.agents.base import BaseAgent, MarketState, AgentSignal


class Arena:
    """
    L'arène compare les signaux des agents sur un même état de marché.
    Elle ne trade PAS. Elle observe, score, et log.
    """

    def __init__(self, agents: List[BaseAgent]):
        self.agents = agents

    def run(
        self,
        symbol: str,
        data: pd.DataFrame,
        portfolio: Dict[str, float] | None = None,
        regime: str | None = None,
    ) -> List[AgentSignal]:
        portfolio = portfolio or {}

        last_price = float(data["Close"].iloc[-1])
        state = MarketState(
            symbol=symbol,
            price=last_price,
            timestamp=str(data.index[-1]),
        )

        signals: List[AgentSignal] = []

        for agent in self.agents:
            sig = agent.generate_signal(
                state=state,
                portfolio=portfolio,
                regime=regime,
                data=data,
            )
            signals.append(sig)

        return signals
