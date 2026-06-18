# src/agents/volatility.py
from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Dict, Optional

import numpy as np
import pandas as pd
import yfinance as yf

from src.agents.base import BaseAgent, MarketState, AgentSignal


@dataclass
class VolatilityConfig:
    vix_fear_threshold: float = 25.0
    vix_extreme_threshold: float = 35.0
    vix_complacency: float = 15.0
    zscore_lookback: int = 252
    spike_mom5: float = 0.20
    spike_ratio: float = 1.40
    complacency_zscore: float = -1.50
    extreme_zscore: float = 1.50
    fear_zscore: float = 0.80
    fear_max_mom5: float = 0.05
    target_weight_extreme: float = 0.10
    target_weight_fear: float = 0.08
    sell_confidence_spike: float = 0.85
    sell_confidence_complacency: float = 0.70
    buy_confidence_extreme: float = 0.88


def _get_vix() -> float:
    """Récupère le niveau actuel du VIX."""
    try:
        vix = yf.Ticker("^VIX")
        hist = vix.history(period="5d")
        if hist.empty:
            return 20.0
        return float(hist["Close"].iloc[-1])
    except Exception:
        return 20.0


def _get_vix_series(period: str = "1y") -> pd.Series:
    """Récupère la série historique du VIX."""
    try:
        vix = yf.download("^VIX", period=period, auto_adjust=True, progress=False)
        close = vix["Close"]
        if isinstance(close, pd.DataFrame):
            close = close.iloc[:, 0]
        return pd.to_numeric(close, errors="coerce").dropna()
    except Exception:
        return pd.Series(dtype=float)


class VolatilityAgent(BaseAgent):
    """
    VolatilityAgent — inspiré des desks Vol Trading / Equity Derivatives.

    Logique :
    - VIX élevé (> 25) = peur excessive = opportunité BUY (mean reversion)
    - VIX très élevé (> 35) = spike de peur = BUY fort
    - VIX faible (< 15) = complaisance = prudence, réduire exposure
    - VIX qui monte rapidement = SELL signal

    Indicateurs utilisés :
    - Niveau absolu du VIX
    - Z-score du VIX sur 252 jours
    - Vitesse de changement du VIX (momentum 5j)
    - Ratio VIX / VIX MA20 (VIX relatif)
    """
    name = "VolatilityAgent"
    _VIX_TTL = 300  # secondes entre re-downloads (5 min)

    def __init__(self, config: Optional[VolatilityConfig] = None):
        self.cfg = config or VolatilityConfig()
        self._vix_cache: Optional[dict] = None
        self._vix_ts: float = 0.0

    def _analyze_vix(self) -> dict:
        """Analyse complète du VIX — résultat mis en cache pour éviter N downloads par run."""
        if self._vix_cache is not None and (time.time() - self._vix_ts) < self._VIX_TTL:
            return self._vix_cache
        try:
            vix_series = _get_vix_series("2y")
            if len(vix_series) < 30:
                return {"vix": 20.0, "zscore": 0.0, "mom5": 0.0, "vix_ratio": 1.0}

            vix_now = float(vix_series.iloc[-1])
            vix_ma20 = float(vix_series.rolling(20).mean().iloc[-1])
            vix_ma252 = float(vix_series.rolling(min(252, len(vix_series))).mean().iloc[-1])
            vix_std252 = float(vix_series.rolling(min(252, len(vix_series))).std().iloc[-1])

            # Z-score
            zscore = (vix_now - vix_ma252) / vix_std252 if vix_std252 > 0 else 0.0

            # Momentum 5 jours
            mom5 = float(vix_series.iloc[-1] / vix_series.iloc[-6] - 1.0) if len(vix_series) >= 6 else 0.0

            # VIX ratio vs MA20
            vix_ratio = vix_now / vix_ma20 if vix_ma20 > 0 else 1.0

            result = {
                "vix": round(vix_now, 2),
                "vix_ma20": round(vix_ma20, 2),
                "zscore": round(zscore, 3),
                "mom5": round(mom5, 4),
                "vix_ratio": round(vix_ratio, 3),
            }
            self._vix_cache = result
            self._vix_ts = time.time()
            return result
        except Exception:
            return {"vix": 20.0, "zscore": 0.0, "mom5": 0.0, "vix_ratio": 1.0}

    def generate_signal(
        self,
        state: MarketState,
        portfolio: Dict[str, float],
        regime: Optional[str] = None,
        data: Optional[pd.DataFrame] = None,
    ) -> AgentSignal:

        vix_data = self._analyze_vix()
        vix = vix_data["vix"]
        zscore = vix_data["zscore"]
        mom5 = vix_data["mom5"]
        vix_ratio = vix_data["vix_ratio"]
        in_position = portfolio.get(state.symbol, 0.0) > 0

        meta = {
            "regime": regime,
            "vix": vix,
            "vix_ma20": vix_data.get("vix_ma20", 0),
            "vix_zscore": zscore,
            "vix_mom5": mom5,
            "vix_ratio": vix_ratio,
        }

        # SELL — VIX monte rapidement = danger
        if in_position and (mom5 > self.cfg.spike_mom5 or vix_ratio > self.cfg.spike_ratio):
            return AgentSignal(
                agent_name=self.name,
                symbol=state.symbol,
                action="SELL",
                confidence=self.cfg.sell_confidence_spike,
                target_weight=0.0,
                reason=f"Vol EXIT: VIX spike +{mom5:.1%} en 5j | ratio={vix_ratio:.2f}",
                meta=meta,
            )

        # SELL — complaisance extrême
        if in_position and vix < self.cfg.vix_complacency and zscore < self.cfg.complacency_zscore:
            return AgentSignal(
                agent_name=self.name,
                symbol=state.symbol,
                action="SELL",
                confidence=self.cfg.sell_confidence_complacency,
                target_weight=0.0,
                reason=f"Vol EXIT: VIX={vix} complaisance extrême — risque de correction",
                meta=meta,
            )

        # BUY — panique extrême = opportunité
        if not in_position and vix > self.cfg.vix_extreme_threshold and zscore > self.cfg.extreme_zscore:
            return AgentSignal(
                agent_name=self.name,
                symbol=state.symbol,
                action="BUY",
                confidence=self.cfg.buy_confidence_extreme,
                target_weight=self.cfg.target_weight_extreme,
                reason=f"Vol BUY: VIX={vix} panique extrême | z={zscore:.2f} → mean reversion",
                meta=meta,
            )

        # BUY — peur élevée
        if not in_position and vix > self.cfg.vix_fear_threshold and zscore > self.cfg.fear_zscore and mom5 < self.cfg.fear_max_mom5:
            conf = min(0.82, 0.60 + (vix - self.cfg.vix_fear_threshold) * 0.01)
            return AgentSignal(
                agent_name=self.name,
                symbol=state.symbol,
                action="BUY",
                confidence=conf,
                target_weight=self.cfg.target_weight_fear,
                reason=f"Vol BUY: VIX={vix} peur élevée | z={zscore:.2f}",
                meta=meta,
            )

        # HOLD
        return AgentSignal(
            agent_name=self.name,
            symbol=state.symbol,
            action="HOLD",
            confidence=0.3,
            target_weight=0.0,
            reason=f"Vol HOLD: VIX={vix} | z={zscore:.2f} | mom5={mom5:.1%}",
            meta=meta,
        )

        