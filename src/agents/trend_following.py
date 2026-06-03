# src/agents/trend_following.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional

import pandas as pd

from src.agents.base import BaseAgent, MarketState, AgentSignal


@dataclass
class TrendFollowingConfig:
    min_history: int = 210
    adx_period: int = 14
    adx_threshold: float = 25.0
    adx_strong: float = 30.0
    mom20_entry: float = 0.01
    mom20_strong: float = 0.03
    mom50_entry: float = 0.05
    sell_mom20: float = -0.03
    sell_mom50: float = -0.05
    target_weight: float = 0.10
    sell_confidence: float = 0.83
    buy_base_confidence: float = 0.65


class TrendFollowingAgent(BaseAgent):
    """
    TrendFollowingAgent (inspiré CTA / Winton / Man AHL):
    - Suit la tendance sur plusieurs timeframes
    - Court terme  : SMA20
    - Moyen terme  : SMA50
    - Long terme   : SMA200
    - BUY  : les 3 MAs alignées à la hausse + ADX fort
    - SELL : les 3 MAs s'inversent OU ADX s'effondre
    - HOLD : signaux mixtes
    """
    name = "TrendFollowingAgent"

    def __init__(self, config: Optional[TrendFollowingConfig] = None):
        self.cfg = config or TrendFollowingConfig()

    def _adx(self, data: pd.DataFrame, period: int = 14) -> float:
        """Average Directional Index — mesure la force de la tendance."""
        try:
            high = pd.to_numeric(data["High"], errors="coerce")
            low = pd.to_numeric(data["Low"], errors="coerce")
            close = pd.to_numeric(data["Close"], errors="coerce")

            plus_dm = high.diff()
            minus_dm = -low.diff()
            plus_dm[plus_dm < 0] = 0
            minus_dm[minus_dm < 0] = 0

            tr = pd.concat([
                high - low,
                (high - close.shift()).abs(),
                (low - close.shift()).abs(),
            ], axis=1).max(axis=1)

            atr = tr.rolling(period).mean()
            plus_di = 100 * (plus_dm.rolling(period).mean() / atr)
            minus_di = 100 * (minus_dm.rolling(period).mean() / atr)

            dx = (100 * (plus_di - minus_di).abs() / (plus_di + minus_di))
            adx = dx.rolling(period).mean()

            return float(adx.iloc[-1])
        except Exception:
            return 0.0

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

        # 2) Indicateurs multi-timeframe
        px = float(close.iloc[-1])
        sma20 = float(close.rolling(20).mean().iloc[-1])
        sma50 = float(close.rolling(50).mean().iloc[-1])
        sma200 = float(close.rolling(200).mean().iloc[-1])

        # Alignement des MAs
        bull_aligned = px > sma20 > sma50 > sma200
        bear_aligned = px < sma20 < sma50 < sma200

        # ADX — force de la tendance
        adx = self._adx(data, period=self.cfg.adx_period) if "High" in data.columns and "Low" in data.columns else self.cfg.adx_threshold
        trend_strong = adx > self.cfg.adx_threshold

        # Momentum
        mom20 = float(px / close.iloc[-21] - 1.0) if len(close) >= 21 else 0.0
        mom50 = float(px / close.iloc[-51] - 1.0) if len(close) >= 51 else 0.0

        in_position = portfolio.get(state.symbol, 0.0) > 0

        meta = {
            "regime": regime,
            "price": px,
            "sma20": round(sma20, 2),
            "sma50": round(sma50, 2),
            "sma200": round(sma200, 2),
            "adx": round(adx, 2),
            "mom20": round(mom20, 4),
            "mom50": round(mom50, 4),
            "bull_aligned": bull_aligned,
            "bear_aligned": bear_aligned,
        }

        # 3) SELL logic (prioritaire)
        if in_position:
            sell_reasons = []

            if bear_aligned:
                sell_reasons.append("MAs bear aligned")

            if not trend_strong and mom20 < self.cfg.sell_mom20:
                sell_reasons.append(f"ADX={adx:.1f} faible + mom20={mom20:.2%}")

            if px < sma50 and mom50 < self.cfg.sell_mom50:
                sell_reasons.append(f"prix < SMA50 + mom50={mom50:.2%}")

            if sell_reasons:
                return AgentSignal(
                    agent_name=self.name,
                    symbol=state.symbol,
                    action="SELL",
                    confidence=self.cfg.sell_confidence,
                    target_weight=0.0,
                    reason="TrendFollow EXIT: " + " | ".join(sell_reasons),
                    meta=meta,
                )

        # 4) Bear = pas de nouveaux longs
        if regime == "bear":
            return AgentSignal(self.name, state.symbol, "HOLD", 0.2, 0.0, "Bear regime: no new longs", meta)

        # 5) BUY logic
        buy = bull_aligned and trend_strong and mom20 > self.cfg.mom20_entry

        if buy:
            conf = self.cfg.buy_base_confidence
            conf += 0.10 if adx > self.cfg.adx_strong else 0.0
            conf += 0.10 if mom50 > self.cfg.mom50_entry else 0.0
            conf += 0.05 if mom20 > self.cfg.mom20_strong else 0.0
            conf = float(min(0.92, conf))

            return AgentSignal(
                agent_name=self.name,
                symbol=state.symbol,
                action="BUY",
                confidence=conf,
                target_weight=self.cfg.target_weight,
                reason=f"TrendFollow BUY: MAs alignées + ADX={adx:.1f} + mom20={mom20:.2%}",
                meta=meta,
            )

        return AgentSignal(
            agent_name=self.name,
            symbol=state.symbol,
            action="HOLD",
            confidence=0.3,
            target_weight=0.0,
            reason=f"No trend setup (ADX={adx:.1f}, bull={bull_aligned})",
            meta=meta,
        )