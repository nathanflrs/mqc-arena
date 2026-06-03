# src/agents/dividend_arbitrage.py
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Dict, Optional

import pandas as pd
import yfinance as yf

from src.agents.base import BaseAgent, MarketState, AgentSignal


@dataclass
class DividendArbitrageConfig:
    min_yield_pct: float = 0.003
    days_before: int = 3
    days_after: int = 1
    target_weight: float = 0.08
    sell_confidence: float = 0.90


class DividendArbitrageAgent(BaseAgent):
    """
    DividendArbitrageAgent — inspiré des desks Equity Derivatives.

    Stratégie :
    - Identifie les titres avec une date ex-dividende proche (1-5 jours)
    - BUY avant la date ex-div pour capturer le dividende
    - SELL après la date ex-div (le lendemain)
    - Filtre sur le rendement du dividende (min 0.3% par dividende)

    Logique réelle des desks :
    - Les market makers achètent avant l'ex-date
    - Ils capturent le dividende
    - Ils revendent après la chute de prix liée au détachement
    - Le profit = dividende - coût de portage - impact marché
    """
    name = "DividendArbitrageAgent"

    def __init__(self, config: Optional[DividendArbitrageConfig] = None):
        self.cfg = config or DividendArbitrageConfig()

    def _get_dividend_info(self, symbol: str) -> dict:
        """Récupère les infos dividendes via yfinance."""
        try:
            ticker = yf.Ticker(symbol)
            info = ticker.info

            # Date ex-dividende
            ex_date_ts = info.get("exDividendDate")
            if not ex_date_ts:
                return {}

            ex_date = datetime.fromtimestamp(ex_date_ts).date()
            dividend_rate = info.get("dividendRate", 0.0) or 0.0
            price = info.get("currentPrice") or info.get("regularMarketPrice", 0.0)

            if price <= 0:
                return {}

            # Dividende par paiement (annuel / fréquence)
            dividend_freq = info.get("dividendYield", 0.0) or 0.0
            div_per_payment = dividend_rate / 4 if dividend_rate > 0 else 0.0  # quarterly approx

            yield_pct = div_per_payment / price if price > 0 else 0.0

            return {
                "ex_date": ex_date,
                "div_per_payment": round(div_per_payment, 4),
                "yield_pct": round(yield_pct, 6),
                "price": price,
                "annual_dividend": dividend_rate,
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

        # Pas de signal en bear market
        if regime == "bear":
            return AgentSignal(
                agent_name=self.name,
                symbol=state.symbol,
                action="HOLD",
                confidence=0.3,
                target_weight=0.0,
                reason="Bear regime: dividend arb paused",
                meta={"regime": regime},
            )

        div_info = self._get_dividend_info(state.symbol)

        if not div_info:
            return AgentSignal(
                agent_name=self.name,
                symbol=state.symbol,
                action="HOLD",
                confidence=0.2,
                target_weight=0.0,
                reason="No dividend data available",
                meta={"regime": regime},
            )

        ex_date = div_info["ex_date"]
        today = datetime.now().date()
        days_to_ex = (ex_date - today).days
        yield_pct = div_info["yield_pct"]
        in_position = portfolio.get(state.symbol, 0.0) > 0

        meta = {
            "regime": regime,
            "ex_date": str(ex_date),
            "days_to_ex": days_to_ex,
            "div_per_payment": div_info["div_per_payment"],
            "yield_pct": f"{yield_pct:.3%}",
            "price": div_info["price"],
        }

        # SELL — on est après l'ex-date et on a une position
        if in_position and days_to_ex < 0:
            return AgentSignal(
                agent_name=self.name,
                symbol=state.symbol,
                action="SELL",
                confidence=self.cfg.sell_confidence,
                target_weight=0.0,
                reason=f"DivArb EXIT: ex-date passée ({ex_date}), dividende capturé",
                meta=meta,
            )

        # BUY — ex-date dans la fenêtre et rendement suffisant
        if (
            not in_position
            and 0 < days_to_ex <= self.cfg.days_before
            and yield_pct >= self.cfg.min_yield_pct
        ):
            conf = min(0.85, 0.60 + yield_pct * 50)
            return AgentSignal(
                agent_name=self.name,
                symbol=state.symbol,
                action="BUY",
                confidence=conf,
                target_weight=self.cfg.target_weight,
                reason=f"DivArb BUY: ex-date dans {days_to_ex}j | div={div_info['div_per_payment']}$ | yield={yield_pct:.3%}",
                meta=meta,
            )

        # HOLD — pas de setup
        reason = (
            f"DivArb: ex-date dans {days_to_ex}j (hors fenêtre)"
            if days_to_ex > self.cfg.days_before
            else f"DivArb: yield={yield_pct:.3%} trop faible"
            if yield_pct < self.cfg.min_yield_pct
            else "DivArb: pas de setup"
        )

        return AgentSignal(
            agent_name=self.name,
            symbol=state.symbol,
            action="HOLD",
            confidence=0.2,
            target_weight=0.0,
            reason=reason,
            meta=meta,
        )