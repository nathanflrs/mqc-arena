# src/agents/pairs_trading.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional

import numpy as np
import pandas as pd
import yfinance as yf

from src.agents.base import BaseAgent, MarketState, AgentSignal


@dataclass
class PairsTradingConfig:
    zscore_entry: float = 2.0
    zscore_exit: float = 0.5
    lookback: int = 60
    target_weight: float = 0.08


# ====== PAIRES DÉFINIES ======
PAIRS = {
    "SPY":  "QQQ",    # Large cap vs Tech
    "AAPL": "MSFT",   # Apple vs Microsoft
    "GLD":  "TLT",    # Or vs Obligations
    "JPM":  "GS",     # JPMorgan vs Goldman
}


def _download_close(symbol: str, period: str = "1y") -> pd.Series:
    """Télécharge la série de clôtures."""
    try:
        df = yf.download(symbol, period=period, auto_adjust=True, progress=False)
        close = df["Close"]
        if isinstance(close, pd.DataFrame):
            close = close.iloc[:, 0]
        return pd.to_numeric(close, errors="coerce").dropna()
    except Exception:
        return pd.Series(dtype=float)


class PairsTradingAgent(BaseAgent):
    """
    PairsTradingAgent — inspiré Stat Arb / Renaissance Technologies.

    Stratégie :
    - Pour chaque paire (A, B), calcule le spread = log(A) - log(B)
    - Normalise le spread en z-score sur 60 jours
    - BUY A (et implicitement short B) quand z-score < -2 (A sous-évalué vs B)
    - SELL quand z-score > +2 (A surévalué vs B) ou retour à 0

    C'est exactement la logique des desks Stat Arb des grandes banques.
    """
    name = "PairsTradingAgent"

    def __init__(self, config: Optional[PairsTradingConfig] = None):
        self.cfg = config or PairsTradingConfig()

    def _compute_zscore(self, symbol: str) -> dict:
        """Calcule le z-score du spread pour la paire de ce symbole."""
        pair = PAIRS.get(symbol)
        if not pair:
            return {}

        try:
            s1 = _download_close(symbol)
            s2 = _download_close(pair)

            # Aligner les deux séries
            df = pd.DataFrame({"s1": s1, "s2": s2}).dropna()
            if len(df) < self.cfg.lookback + 10:
                return {}

            # Spread log
            spread = np.log(df["s1"]) - np.log(df["s2"])

            # Z-score sur lookback jours
            spread_mean = spread.rolling(self.cfg.lookback).mean()
            spread_std = spread.rolling(self.cfg.lookback).std()
            zscore = (spread - spread_mean) / spread_std

            current_z = float(zscore.iloc[-1])
            current_spread = float(spread.iloc[-1])
            mean_spread = float(spread_mean.iloc[-1])

            return {
                "pair": pair,
                "zscore": round(current_z, 3),
                "spread": round(current_spread, 4),
                "mean_spread": round(mean_spread, 4),
                "s1_price": float(df["s1"].iloc[-1]),
                "s2_price": float(df["s2"].iloc[-1]),
            }
        except Exception:
            return {}

    def generate_signal(
        self,
        state: MarketState,
        portfolio: Dict[str, float],
        regime: Optional[str] = None,
        data: Optional[pd.DataFrame] = None,
    ) -> AgentSignal:

        # Vérifie si ce symbole a une paire
        if state.symbol not in PAIRS:
            return AgentSignal(
                agent_name=self.name,
                symbol=state.symbol,
                action="HOLD",
                confidence=0.1,
                target_weight=0.0,
                reason="No pair defined for this symbol",
                meta={"regime": regime},
            )

        z_data = self._compute_zscore(state.symbol)

        if not z_data:
            return AgentSignal(
                agent_name=self.name,
                symbol=state.symbol,
                action="HOLD",
                confidence=0.1,
                target_weight=0.0,
                reason="Insufficient data for pairs trading",
                meta={"regime": regime},
            )

        zscore = z_data["zscore"]
        pair = z_data["pair"]
        in_position = portfolio.get(state.symbol, 0.0) > 0

        meta = {
            "regime": regime,
            "pair": pair,
            "zscore": zscore,
            "spread": z_data["spread"],
            "mean_spread": z_data["mean_spread"],
            "s1_price": z_data["s1_price"],
            "s2_price": z_data["s2_price"],
            "entry_threshold": self.cfg.zscore_entry,
        }

        # SELL — retour à la moyenne ou surévaluation
        if in_position and abs(zscore) < self.cfg.zscore_exit:
            return AgentSignal(
                agent_name=self.name,
                symbol=state.symbol,
                action="SELL",
                confidence=0.85,
                target_weight=0.0,
                reason=f"PairsTrading EXIT: z-score={zscore:.2f} revenu à la moyenne",
                meta=meta,
            )

        # SELL — spread inversé (surévalué)
        if in_position and zscore > self.cfg.zscore_entry:
            return AgentSignal(
                agent_name=self.name,
                symbol=state.symbol,
                action="SELL",
                confidence=0.88,
                target_weight=0.0,
                reason=f"PairsTrading EXIT: z-score={zscore:.2f} surévalué vs {pair}",
                meta=meta,
            )

        # BUY — sous-évalué vs paire
        if not in_position and zscore < -self.cfg.zscore_entry:
            conf = min(0.90, 0.65 + abs(zscore) * 0.05)
            return AgentSignal(
                agent_name=self.name,
                symbol=state.symbol,
                action="BUY",
                confidence=conf,
                target_weight=self.cfg.target_weight,
                reason=f"PairsTrading BUY: {state.symbol} sous-évalué vs {pair} | z={zscore:.2f}",
                meta=meta,
            )

        # HOLD
        return AgentSignal(
            agent_name=self.name,
            symbol=state.symbol,
            action="HOLD",
            confidence=0.3,
            target_weight=0.0,
            reason=f"PairsTrading HOLD: z-score={zscore:.2f} (seuil={self.cfg.zscore_entry})",
            meta=meta,
        )
    
    