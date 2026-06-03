# src/agents/dummy.py
from __future__ import annotations

from typing import Dict, Optional
import pandas as pd

from src.agents.base import BaseAgent, MarketState, AgentSignal


class DummyHoldAgent(BaseAgent):
    name = "DummyHoldAgent"

    def generate_signal(
        self,
        state: MarketState,
        portfolio: Dict[str, float],
        regime: Optional[str] = None,
        data: Optional[pd.DataFrame] = None,
    ) -> AgentSignal:
        return AgentSignal(
            agent_name=self.name,
            symbol=state.symbol,
            action="HOLD",
            confidence=0.0,
            target_weight=0.0,
            reason="Baseline hold",
            meta={"regime": regime},
        )
