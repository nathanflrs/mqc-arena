# src/agents/macro.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional

import pandas as pd

from src.agents.base import BaseAgent, MarketState, AgentSignal
from src.data.market_data import download_ohlcv


@dataclass
class MacroConfig:
    spy_threshold: float = 0.05
    gld_threshold: float = 0.05
    tlt_threshold: float = 0.03
    risk_on_min: int = 3
    risk_off_min: int = 3
    target_weight: float = 0.08
    sell_confidence: float = 0.80
    buy_base_confidence: float = 0.60
    momentum_period: int = 63


class MacroAgent(BaseAgent):
    """
    MacroAgent (inspiré Ray Dalio / Bridgewater):
    - Analyse le contexte macro via des proxies ETF
    - GLD  (or)   : valeur refuge, bon en bear/incertitude
    - TLT  (bonds): obligations long terme, bon quand taux baissent
    - SPY  (actions): risque on/off
    - UUP  (dollar): force du dollar

    Logique :
    - RISK ON  (bull)  : SPY fort, GLD faible, TLT faible → BUY actions
    - RISK OFF (bear)  : GLD fort, TLT fort, SPY faible   → SELL / cash
    - MIXED    (choppy): signaux contradictoires           → HOLD
    """
    name = "MacroAgent"

    def __init__(self, config: Optional[MacroConfig] = None):
        self.cfg = config or MacroConfig()

    def _momentum(self, symbol: str, period: int = 63) -> float:
        """Momentum simple sur `period` jours."""
        try:
            df = download_ohlcv(symbol, period="1y")
            close = pd.to_numeric(df["Close"], errors="coerce").dropna()
            if len(close) < period + 1:
                return 0.0
            return float(close.iloc[-1] / close.iloc[-period] - 1.0)
        except Exception:
            return 0.0

    def generate_signal(
        self,
        state: MarketState,
        portfolio: Dict[str, float],
        regime: Optional[str] = None,
        data: Optional[pd.DataFrame] = None,
    ) -> AgentSignal:

        # 1) Calcul des momentums macro
        mom_spy = self._momentum("SPY", period=self.cfg.momentum_period)
        mom_gld = self._momentum("GLD", period=self.cfg.momentum_period)
        mom_tlt = self._momentum("TLT", period=self.cfg.momentum_period)

        # 2) Score macro
        risk_on_score = 0
        risk_off_score = 0

        # SPY fort = risk on
        if mom_spy > self.cfg.spy_threshold:
            risk_on_score += 2
        elif mom_spy < -self.cfg.spy_threshold:
            risk_off_score += 2

        # GLD fort = risk off (valeur refuge)
        if mom_gld > self.cfg.gld_threshold:
            risk_off_score += 1
        elif mom_gld < 0.0:
            risk_on_score += 1

        # TLT fort = risk off (fuite vers obligations)
        if mom_tlt > self.cfg.tlt_threshold:
            risk_off_score += 1
        elif mom_tlt < -self.cfg.tlt_threshold:
            risk_on_score += 1

        meta = {
            "regime": regime,
            "mom_spy": round(mom_spy, 4),
            "mom_gld": round(mom_gld, 4),
            "mom_tlt": round(mom_tlt, 4),
            "risk_on_score": risk_on_score,
            "risk_off_score": risk_off_score,
        }

        in_position = portfolio.get(state.symbol, 0.0) > 0

        # 3) SELL logic (prioritaire)
        if in_position and risk_off_score >= self.cfg.risk_off_min:
            return AgentSignal(
                agent_name=self.name,
                symbol=state.symbol,
                action="SELL",
                confidence=self.cfg.sell_confidence,
                target_weight=0.0,
                reason=f"Macro RISK OFF: GLD={mom_gld:.2%} TLT={mom_tlt:.2%} SPY={mom_spy:.2%}",
                meta=meta,
            )

        # 4) BUY logic
        if risk_on_score >= self.cfg.risk_on_min and regime != "bear":
            conf = self.cfg.buy_base_confidence + (0.10 if risk_on_score >= self.cfg.risk_on_min + 1 else 0.0)
            conf = min(0.85, conf)
            return AgentSignal(
                agent_name=self.name,
                symbol=state.symbol,
                action="BUY",
                confidence=conf,
                target_weight=self.cfg.target_weight,
                reason=f"Macro RISK ON: SPY={mom_spy:.2%} GLD={mom_gld:.2%} TLT={mom_tlt:.2%}",
                meta=meta,
            )

        # 5) HOLD
        return AgentSignal(
            agent_name=self.name,
            symbol=state.symbol,
            action="HOLD",
            confidence=0.4,
            target_weight=0.0,
            reason=f"Macro MIXED: risk_on={risk_on_score} risk_off={risk_off_score}",
            meta=meta,
        )
    