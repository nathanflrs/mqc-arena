# src/agents/buffett.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Dict

import pandas as pd

from src.agents.base import BaseAgent, MarketState, AgentSignal


@dataclass
class BuffettConfig:
    target_weight: float = 0.10
    vol20_max: float = 0.03
    require_sma200: bool = True
    near_high_252_threshold: float = 0.85  # assoupli de 0.90 -> 0.85
    min_history: int = 260
    min_confidence: float = 0.60
    # SELL triggers
    sell_below_sma200: bool = True         # sort si prix passe sous SMA200
    sell_vol20_max: float = 0.05           # sort si volatilité explose > 5%
    sell_drawdown_max: float = 0.12        # sort si drawdown > 12% depuis le plus haut 60j


class BuffettAgent(BaseAgent):
    name = "BuffettAgent"

    def __init__(self, config: Optional[BuffettConfig] = None):
        self.cfg = config or BuffettConfig()

    def generate_signal(
        self,
        state: MarketState,
        portfolio: Dict[str, float],
        regime: Optional[str] = None,
        data: Optional[pd.DataFrame] = None,
    ) -> AgentSignal:

        # 1) Pas de data => HOLD
        if data is None or data.empty or "Close" not in data.columns:
            return AgentSignal(
                agent_name=self.name,
                symbol=state.symbol,
                action="HOLD",
                confidence=0.2,
                target_weight=0.0,
                reason="No data provided",
                meta={"regime": regime},
            )

        close = data["Close"]
        if isinstance(close, pd.DataFrame):
            close = close.iloc[:, 0]
        close = pd.to_numeric(close, errors="coerce").dropna()

        if len(close) < self.cfg.min_history:
            return AgentSignal(
                agent_name=self.name,
                symbol=state.symbol,
                action="HOLD",
                confidence=0.3,
                target_weight=0.0,
                reason=f"Insufficient history ({len(close)} < {self.cfg.min_history})",
                meta={"regime": regime},
            )

        # 2) Indicateurs
        ret = close.pct_change().dropna()
        vol20 = float(ret.tail(20).std()) if len(ret) >= 20 else float("inf")
        sma200 = float(close.rolling(200).mean().iloc[-1])
        price = float(close.iloc[-1])
        high252 = float(close.tail(252).max())
        high60 = float(close.tail(60).max())
        near_high = price / high252 if high252 > 0 else 0.0
        drawdown60 = (high60 - price) / high60 if high60 > 0 else 0.0

        in_position = portfolio.get(state.symbol, 0.0) > 0

        # 3) SELL logic (priorité sur BUY)
        if in_position:
            sell_reasons = []

            if self.cfg.sell_below_sma200 and price < sma200:
                sell_reasons.append("price < SMA200")

            if vol20 > self.cfg.sell_vol20_max:
                sell_reasons.append(f"vol20={vol20:.3f} > {self.cfg.sell_vol20_max}")

            if drawdown60 > self.cfg.sell_drawdown_max:
                sell_reasons.append(f"drawdown60={drawdown60:.2%} > {self.cfg.sell_drawdown_max:.0%}")

            if sell_reasons:
                # Long-bias: single-trigger SELL in bull is likely a temporary dip
                sell_conf = 0.85 if (regime != "bull" or len(sell_reasons) >= 2) else 0.62
                return AgentSignal(
                    agent_name=self.name,
                    symbol=state.symbol,
                    action="SELL",
                    confidence=sell_conf,
                    target_weight=0.0,
                    reason="Buffett EXIT: " + " | ".join(sell_reasons),
                    meta={
                        "regime": regime,
                        "price": price,
                        "sma200": sma200,
                        "vol20": vol20,
                        "drawdown60": drawdown60,
                    },
                )

        # 4) BUY logic
        # En bear: pas de nouveaux longs MAIS on garde le SELL actif
        if regime == "bear":
            return AgentSignal(
                agent_name=self.name,
                symbol=state.symbol,
                action="HOLD",
                confidence=0.7,
                target_weight=0.0,
                reason="Regime bear: Buffett holds (no new longs)",
                meta={"regime": regime},
            )

        sma_ok = price > sma200
        vol_ok = vol20 <= self.cfg.vol20_max
        high_ok = near_high >= self.cfg.near_high_252_threshold

        score = sum([sma_ok, vol_ok, high_ok])
        confidence = {0: 0.35, 1: 0.55, 2: 0.70, 3: 0.90}[score]

        if score >= 2 and confidence >= self.cfg.min_confidence:
            return AgentSignal(
                agent_name=self.name,
                symbol=state.symbol,
                action="BUY",
                confidence=confidence,
                target_weight=self.cfg.target_weight,
                reason="Buffett screen passed (stable + trend + near-high)",
                meta={
                    "regime": regime,
                    "vol20": vol20,
                    "sma200": sma200,
                    "price": price,
                    "near_high_252": near_high,
                    "drawdown60": drawdown60,
                    "flags": {"sma_ok": sma_ok, "vol_ok": vol_ok, "high_ok": high_ok},
                },
            )

        return AgentSignal(
            agent_name=self.name,
            symbol=state.symbol,
            action="HOLD",
            confidence=confidence,
            target_weight=0.0,
            reason="Buffett screen not strong enough",
            meta={
                "regime": regime,
                "vol20": vol20,
                "sma200": sma200,
                "price": price,
                "near_high_252": near_high,
                "drawdown60": drawdown60,
                "flags": {"sma_ok": sma_ok, "vol_ok": vol_ok, "high_ok": high_ok},
            },
        )