# src/agents/insider_buy.py
"""
InsiderBuyAgent — SEC Form 4 cluster detection.

Signal: BUY when 2+ distinct officers/directors each purchase ≥$100K
of the stock on the open market within a 30-day rolling window.

Rationale (Lakonishok & Lee, 2001):
  Open-market purchases by corporate insiders are the strongest predictive
  insider signal. Grants, exercises, and awards are excluded — only code-P
  transactions (open-market buys) count.

Integration notes:
  - ETFs (SPY, QQQ, GLD …) have no CIK → always returns HOLD.
  - target_weight is intentionally small (0.06): insider signal is
    confirming, not primary — it piggybacks on the Arena selector.
  - Confidence scales with the number of distinct buyers (capped at 0.85).
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Dict, List, Optional

import pandas as pd

from src.agents.base import BaseAgent, MarketState, AgentSignal
from src.data.sec_insider import SECInsiderClient, InsiderTransaction

logger = logging.getLogger(__name__)


@dataclass
class InsiderBuyConfig:
    # Minimum number of DISTINCT insiders each purchasing ≥ min_purchase_amount
    min_distinct_insiders: int   = 2
    # Minimum USD value per individual purchase to qualify
    min_purchase_amount:   float = 100_000.0
    # Rolling look-back window in calendar days
    lookback_days:         int   = 30
    # Target portfolio weight when signal fires
    target_weight:         float = 0.06
    # Base confidence (scales up with more insiders)
    base_confidence:       float = 0.60
    # Confidence bonus per extra insider beyond the minimum
    confidence_per_extra:  float = 0.08
    # Hard cap on confidence
    max_confidence:        float = 0.85


class InsiderBuyAgent(BaseAgent):
    """
    Generates BUY signals when a cluster of insiders purchase open-market
    stock (Form 4, transaction code P) above the configured size threshold.
    Returns HOLD for ETFs and when cluster criteria are not met.
    """

    name = "InsiderBuyAgent"

    def __init__(
        self,
        config: Optional[InsiderBuyConfig] = None,
        client: Optional[SECInsiderClient] = None,
    ) -> None:
        self.cfg    = config or InsiderBuyConfig()
        self._client = client or SECInsiderClient()

    def generate_signal(
        self,
        state:     MarketState,
        portfolio: Dict[str, float],
        regime:    Optional[str] = None,
        data:      Optional[pd.DataFrame] = None,
    ) -> AgentSignal:
        cfg = self.cfg

        try:
            transactions = self._client.get_recent_purchases(
                state.symbol,
                days_back=cfg.lookback_days,
            )
        except Exception as exc:
            logger.warning("InsiderBuyAgent: fetch failed for %s — %s", state.symbol, exc)
            transactions = []

        # Filter: officer or director, open-market purchase ≥ min amount
        qualifying: List[InsiderTransaction] = [
            t for t in transactions
            if (t.is_officer or t.is_director)
            and t.total_value >= cfg.min_purchase_amount
            and t.transaction_code == "P"
        ]

        distinct_insiders = len({t.reporter_name for t in qualifying})
        total_amount      = sum(t.total_value for t in qualifying)
        latest_date       = max((t.transaction_date for t in qualifying), default="")

        meta = {
            "regime":           regime,
            "n_qualifying":     len(qualifying),
            "distinct_insiders": distinct_insiders,
            "total_amount":     round(total_amount, 0),
            "lookback_days":    cfg.lookback_days,
            "latest_purchase":  latest_date,
        }

        if distinct_insiders >= cfg.min_distinct_insiders:
            extras = distinct_insiders - cfg.min_distinct_insiders
            conf   = min(cfg.max_confidence,
                         cfg.base_confidence + extras * cfg.confidence_per_extra)
            return AgentSignal(
                agent_name=self.name,
                symbol=state.symbol,
                action="BUY",
                confidence=round(conf, 3),
                target_weight=cfg.target_weight,
                reason=(
                    f"InsiderBuy: {distinct_insiders} insiders, "
                    f"${total_amount:,.0f} in {cfg.lookback_days}d "
                    f"(latest {latest_date})"
                ),
                meta=meta,
            )

        return AgentSignal(
            agent_name=self.name,
            symbol=state.symbol,
            action="HOLD",
            confidence=0.0,
            target_weight=0.0,
            reason=(
                f"InsiderBuy: {distinct_insiders}/{cfg.min_distinct_insiders} insiders "
                f"(${total_amount:,.0f}) — threshold not met"
            ),
            meta=meta,
        )
