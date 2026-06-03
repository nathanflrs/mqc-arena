# src/agents/citadel.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional

import pandas as pd
import numpy as np

from src.agents.base import BaseAgent, MarketState, AgentSignal


@dataclass
class CitadelConfig:
    min_history: int = 210
    target_weight: float = 0.12
    mom63_entry: float = 0.05
    mom63_exit: float = -0.05
    mom21_entry: float = 0.01
    volume_ratio: float = 0.80
    atr_low_threshold: float = 0.015
    sell_confidence: float = 0.88
    buy_base_confidence: float = 0.65


def _compute_atr(data: pd.DataFrame, period: int = 14) -> float:
    high = pd.to_numeric(data["High"], errors="coerce")
    low = pd.to_numeric(data["Low"], errors="coerce")
    close_prev = pd.to_numeric(data["Close"], errors="coerce").shift(1)

    tr = pd.concat([
        high - low,
        (high - close_prev).abs(),
        (low - close_prev).abs(),
    ], axis=1).max(axis=1)

    return float(tr.rolling(period).mean().iloc[-1])


def _compute_volume_ok(data: pd.DataFrame, lookback: int = 20, ratio: float = 0.8) -> bool:
    """Vérifie que le volume du jour est au-dessus de la moyenne 20j."""
    if "Volume" not in data.columns:
        return True  # pas de volume dispo, on ne bloque pas
    vol = pd.to_numeric(data["Volume"], errors="coerce").dropna()
    if len(vol) < lookback + 1:
        return True
    avg_vol = float(vol.iloc[-lookback-1:-1].mean())
    last_vol = float(vol.iloc[-1])
    return last_vol > avg_vol * ratio


class CitadelAgent(BaseAgent):
    """
    CitadelAgent v2:
    - Trend filter : Close > SMA200
    - Momentum     : breakout 20j + momentum 63j > 5%
    - Volume       : volume récent > 80% moyenne 20j
    - ATR          : utilisé pour qualifier la force du mouvement
    - SELL         : prix < SMA200 OU momentum négatif OU cassure bas 20j
    - Regime bear  : pas de nouveaux longs MAIS SELL toujours actif
    """
    name = "CitadelAgent"

    def __init__(self, config: Optional[CitadelConfig] = None):
        self.cfg = config or CitadelConfig()

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
        sma200 = float(close.rolling(200).mean().iloc[-1])
        sma50 = float(close.rolling(50).mean().iloc[-1])

        prev20_high = float(close.iloc[-21:-1].max())
        prev20_low = float(close.iloc[-21:-1].min())
        breakout_up = px > prev20_high
        breakdown = px < prev20_low

        mom63 = float(px / close.iloc[-64] - 1.0) if len(close) >= 64 else 0.0
        mom21 = float(px / close.iloc[-22] - 1.0) if len(close) >= 22 else 0.0

        trend_ok = px > sma200 and sma50 > sma200  # golden cross approximé
        vol_ok = _compute_volume_ok(data, ratio=self.cfg.volume_ratio)

        atr14 = _compute_atr(data) if "High" in data.columns and "Low" in data.columns else 0.0
        atr_pct = atr14 / px if px > 0 else 0.0

        in_position = portfolio.get(state.symbol, 0.0) > 0

        # 3) SELL logic (prioritaire)
        if in_position:
            sell_reasons = []

            if px < sma200:
                sell_reasons.append("price < SMA200")

            if mom63 < self.cfg.mom63_exit:
                sell_reasons.append(f"mom63={mom63:.2%} < -5%")

            if breakdown:
                sell_reasons.append("breakdown 20j low")

            if sell_reasons:
                return AgentSignal(
                    agent_name=self.name,
                    symbol=state.symbol,
                    action="SELL",
                    confidence=self.cfg.sell_confidence,
                    target_weight=0.0,
                    reason="Citadel EXIT: " + " | ".join(sell_reasons),
                    meta={
                        "regime": regime,
                        "price": px,
                        "sma200": sma200,
                        "mom63": mom63,
                        "breakdown": breakdown,
                    },
                )

        # 4) Bear = pas de nouveaux longs
        if regime == "bear":
            return AgentSignal(self.name, state.symbol, "HOLD", 0.2, 0.0, "Bear regime: no new longs", {"regime": regime})

        # 5) BUY logic
        mom_ok = mom63 > self.cfg.mom63_entry and mom21 > self.cfg.mom21_entry
        buy = trend_ok and vol_ok and (breakout_up or mom_ok)

        if buy:
            conf = self.cfg.buy_base_confidence
            conf += 0.10 if breakout_up else 0.0
            conf += 0.10 if mom63 > self.cfg.mom63_entry * 1.6 else 0.0
            conf += 0.05 if vol_ok else 0.0
            conf += 0.05 if atr_pct < self.cfg.atr_low_threshold else 0.0
            conf = float(min(0.95, conf))

            return AgentSignal(
                agent_name=self.name,
                symbol=state.symbol,
                action="BUY",
                confidence=conf,
                target_weight=self.cfg.target_weight,
                reason="Citadel v2: trend + momentum + volume",
                meta={
                    "regime": regime,
                    "price": px,
                    "sma200": sma200,
                    "sma50": sma50,
                    "breakout20": breakout_up,
                    "mom63": mom63,
                    "mom21": mom21,
                    "vol_ok": vol_ok,
                    "atr_pct": round(atr_pct, 4),
                },
            )

        return AgentSignal(
            agent_name=self.name,
            symbol=state.symbol,
            action="HOLD",
            confidence=0.3 if trend_ok else 0.1,
            target_weight=0.0,
            reason="No Citadel momentum setup",
            meta={
                "regime": regime,
                "price": px,
                "sma200": sma200,
                "trend_ok": trend_ok,
                "breakout20": breakout_up,
                "mom63": mom63,
                "vol_ok": vol_ok,
            },
        )