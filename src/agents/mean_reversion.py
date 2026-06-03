# src/agents/mean_reversion.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional

import pandas as pd
import numpy as np

from src.agents.base import BaseAgent, MarketState, AgentSignal


@dataclass
class MeanReversionConfig:
    min_history: int = 50
    rsi_period: int = 14
    rsi_threshold: int = 35
    rsi_threshold_bear: int = 30
    rsi_overbought: int = 65
    rsi_extreme_low: int = 25
    bb_period: int = 20
    bb_std: float = 2.0
    volume_ratio: float = 1.2
    target_weight: float = 0.08
    sell_confidence: float = 0.82
    buy_base_confidence: float = 0.65


def _rsi(close: pd.Series, period: int = 14) -> float:
    delta = close.diff().dropna()
    gain = delta.clip(lower=0).rolling(period).mean()
    loss = (-delta.clip(upper=0)).rolling(period).mean()
    rs = gain / loss.replace(0, float("inf"))
    rsi = 100 - (100 / (1 + rs))
    return float(rsi.iloc[-1])


def _bollinger(close: pd.Series, period: int = 20, std: float = 2.0):
    sma = close.rolling(period).mean()
    sigma = close.rolling(period).std()
    upper = sma + std * sigma
    lower = sma - std * sigma
    return float(sma.iloc[-1]), float(upper.iloc[-1]), float(lower.iloc[-1])


class MeanReversionAgent(BaseAgent):
    """
    MeanReversionAgent (inspiré Renaissance Technologies):
    - BUY  : RSI < 35 + prix sous Bollinger bas + volume élevé
    - SELL : RSI > 65 OU prix revenu à SMA20 (prise de profit)
    - HOLD : tout le reste
    Logique contrariante — l'opposé de CitadelAgent.
    """
    name = "MeanReversionAgent"

    def __init__(self, config: Optional[MeanReversionConfig] = None):
        self.cfg = config or MeanReversionConfig()

    def generate_signal(
        self,
        state: MarketState,
        portfolio: Dict[str, float],
        regime: Optional[str] = None,
        data: Optional[pd.DataFrame] = None,
    ) -> AgentSignal:

        # 1) Sécurité data
        if data is None or data.empty or "Close" not in data.columns:
            return AgentSignal(self.name, state.symbol, "HOLD", 0.0, 0.0, "No data", {"regime": regime})

        close = pd.to_numeric(data["Close"], errors="coerce").dropna()
        if len(close) < self.cfg.min_history:
            return AgentSignal(self.name, state.symbol, "HOLD", 0.0, 0.0, "Insufficient history", {"regime": regime})

        # 2) Indicateurs
        px = float(close.iloc[-1])
        rsi14 = _rsi(close, period=self.cfg.rsi_period)
        sma20, bb_upper, bb_lower = _bollinger(close, period=self.cfg.bb_period, std=self.cfg.bb_std)

        # Volume
        vol_ok = True
        if "Volume" in data.columns:
            vol = pd.to_numeric(data["Volume"], errors="coerce").dropna()
            if len(vol) >= 21:
                avg_vol = float(vol.iloc[-21:-1].mean())
                last_vol = float(vol.iloc[-1])
                vol_ok = last_vol > avg_vol * self.cfg.volume_ratio

        in_position = portfolio.get(state.symbol, 0.0) > 0

        # 3) SELL logic (prioritaire)
        if in_position:
            sell_reasons = []

            if rsi14 > self.cfg.rsi_overbought:
                sell_reasons.append(f"RSI={rsi14:.1f} > 65 (overbought)")

            if px >= sma20:
                sell_reasons.append(f"prix revenu à SMA20={sma20:.2f} (target atteint)")

            if sell_reasons:
                return AgentSignal(
                    agent_name=self.name,
                    symbol=state.symbol,
                    action="SELL",
                    confidence=self.cfg.sell_confidence,
                    target_weight=0.0,
                    reason="MeanRev EXIT: " + " | ".join(sell_reasons),
                    meta={
                        "regime": regime,
                        "price": px,
                        "rsi14": rsi14,
                        "sma20": sma20,
                        "bb_lower": bb_lower,
                    },
                )

        # 4) En bear : mean reversion fonctionne encore mais on est plus prudent
        rsi_threshold = self.cfg.rsi_threshold_bear if regime == "bear" else self.cfg.rsi_threshold

        # 5) BUY logic
        oversold = rsi14 < rsi_threshold
        below_bb = px < bb_lower
        buy = oversold and below_bb and vol_ok

        if buy:
            conf = self.cfg.buy_base_confidence
            conf += 0.10 if rsi14 < self.cfg.rsi_extreme_low else 0.0
            conf += 0.10 if px < bb_lower * 0.99 else 0.0
            conf += 0.05 if vol_ok else 0.0
            conf = float(min(0.92, conf))

            return AgentSignal(
                agent_name=self.name,
                symbol=state.symbol,
                action="BUY",
                confidence=conf,
                target_weight=self.cfg.target_weight,
                reason=f"MeanRev BUY: RSI={rsi14:.1f} + below Bollinger",
                meta={
                    "regime": regime,
                    "price": px,
                    "rsi14": rsi14,
                    "sma20": sma20,
                    "bb_lower": bb_lower,
                    "bb_upper": bb_upper,
                    "vol_ok": vol_ok,
                },
            )

        return AgentSignal(
            agent_name=self.name,
            symbol=state.symbol,
            action="HOLD",
            confidence=0.3,
            target_weight=0.0,
            reason=f"No mean reversion setup (RSI={rsi14:.1f})",
            meta={
                "regime": regime,
                "price": px,
                "rsi14": rsi14,
                "sma20": sma20,
                "bb_lower": bb_lower,
                "vol_ok": vol_ok,
            },
        )
    
    