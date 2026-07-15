# src/agents/momentum.py
"""
CrossSectionalMomentumAgent — Jegadeesh-Titman (1993) momentum strategy.

Method:
  1. For each ticker in the universe, compute the average return over
     3-month (63 trading days), 6-month (126 td) and 12-month (252 td)
     formation windows, SKIPPING the most recent 21 trading days (~1 month)
     to avoid the short-term reversal effect documented by JT.
  2. Rank all tickers by composite momentum score (descending).
  3. Top `top_pct`% → BUY signal with confidence scaled by rank.
  4. All others → HOLD.

Usage in runner.py:
    mom_agent = CrossSectionalMomentumAgent(universe=all_data)
    arena = Arena([..., mom_agent])

Rankings are computed once per day and cached; subsequent calls for
individual tickers just look up the pre-computed rank.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import date
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from src.agents.base import BaseAgent, MarketState, AgentSignal

logger = logging.getLogger(__name__)


@dataclass
class MomentumConfig:
    # Formation windows in trading days (~3m / 6m / 12m)
    periods:       List[int] = field(default_factory=lambda: [63, 126, 252])
    # Skip this many recent trading days (avoids short-term reversal)
    skip_days:     int   = 21
    # Fraction of universe that gets a BUY signal (top quartile = 0.25)
    top_pct:       float = 0.25
    # Minimum tickers needed to make cross-sectional ranking meaningful
    min_universe:  int   = 4
    # Minimum required price history (trading days)
    min_history:   int   = 280   # enough for 252 + 21 + buffer
    target_weight: float = 0.07
    base_confidence: float = 0.65
    # Extra confidence added to rank-1 ticker vs last top-quartile ticker
    top_conf_range: float = 0.15


def _momentum_score(
    close:     pd.Series,
    periods:   List[int],
    skip_days: int,
) -> Optional[float]:
    """
    Average return over all formation periods, using the price at
    `skip_days` ago as the end-point (to skip the reversal window).
    Returns None when there is insufficient history for all periods.
    """
    scores = []
    for p in periods:
        required = p + skip_days + 1
        if len(close) < required:
            continue
        end_px   = float(close.iloc[-(skip_days + 1)])
        start_px = float(close.iloc[-(skip_days + 1 + p)])
        if start_px <= 0:
            continue
        scores.append((end_px / start_px) - 1.0)
    return float(np.mean(scores)) if scores else None


class CrossSectionalMomentumAgent(BaseAgent):
    """
    Generates BUY signals for top-quartile momentum tickers, computed
    cross-sectionally across the full WATCHLIST universe.

    Call `set_universe(all_data)` once per run before generating signals.
    Inject `universe` in the constructor for testing.
    """

    name = "CrossSectionalMomentumAgent"

    def __init__(
        self,
        config:   Optional[MomentumConfig]           = None,
        universe: Optional[Dict[str, pd.DataFrame]]  = None,
    ) -> None:
        self.cfg = config or MomentumConfig()
        self._rankings:      Dict[str, dict] = {}
        self._rankings_date: Optional[date]  = None

        if universe is not None:
            self.set_universe(universe)

    # ── Universe / ranking ─────────────────────────────────────────────────────

    def set_universe(self, all_data: Dict[str, pd.DataFrame]) -> None:
        """
        Compute cross-sectional momentum rankings from a dict of OHLCV DataFrames.
        Call once per run (idempotent if called multiple times same day).
        """
        scores: Dict[str, float] = {}
        for sym, df in all_data.items():
            if "Close" not in df.columns:
                continue
            close = df["Close"].dropna()
            if len(close) < self.cfg.min_history:
                logger.debug("Momentum: %s has only %d rows — skipped", sym, len(close))
                continue
            sc = _momentum_score(close, self.cfg.periods, self.cfg.skip_days)
            if sc is not None:
                scores[sym] = sc

        if len(scores) < self.cfg.min_universe:
            logger.warning(
                "Momentum: only %d tickers with sufficient history (need %d) — rankings cleared",
                len(scores), self.cfg.min_universe,
            )
            self._rankings = {}
            return

        sorted_syms = sorted(scores, key=lambda s: scores[s], reverse=True)
        n      = len(sorted_syms)
        top_n  = max(1, int(np.ceil(n * self.cfg.top_pct)))
        top_set = set(sorted_syms[:top_n])

        self._rankings = {}
        for rank_idx, sym in enumerate(sorted_syms):
            rank = rank_idx + 1          # 1 = strongest momentum
            pct  = rank / n              # 0 → 1 (0 = top)
            self._rankings[sym] = {
                "rank":     rank,
                "n":        n,
                "score":    round(scores[sym], 5),
                "is_top":   sym in top_set,
                "top_n":    top_n,
                "pct_rank": round(pct, 4),
            }

        self._rankings_date = date.today()
        top_names = sorted_syms[:top_n]
        logger.info(
            "Momentum: ranked %d tickers | top %d: %s",
            n, top_n, top_names,
        )

    # ── Signal generation ──────────────────────────────────────────────────────

    def generate_signal(
        self,
        state:     MarketState,
        portfolio: Dict,
        regime:    Optional[str]       = None,
        data:      Optional[pd.DataFrame] = None,
    ) -> AgentSignal:
        cfg  = self.cfg
        info = self._rankings.get(state.symbol)

        meta = {
            "regime":        regime,
            "rankings_date": self._rankings_date.isoformat() if self._rankings_date else None,
            **({
                "rank":     info["rank"],
                "n":        info["n"],
                "score":    info["score"],
                "is_top":   info["is_top"],
                "top_n":    info["top_n"],
                "pct_rank": info["pct_rank"],
            } if info else {"rank": None}),
        }

        if info is None:
            return AgentSignal(
                agent_name   = self.name,
                symbol       = state.symbol,
                action       = "HOLD",
                confidence   = 0.0,
                target_weight= 0.0,
                reason       = "Momentum: ticker absent du ranking (données insuffisantes)",
                meta         = meta,
            )

        if not info["is_top"]:
            return AgentSignal(
                agent_name   = self.name,
                symbol       = state.symbol,
                action       = "HOLD",
                confidence   = 0.0,
                target_weight= 0.0,
                reason       = (
                    f"Momentum: rang {info['rank']}/{info['n']} "
                    f"(score={info['score']:+.2%}) — hors top {info['top_n']}"
                ),
                meta = meta,
            )

        # Confidence scales linearly within the top bucket:
        # rank 1 → base + top_conf_range; rank top_n → base
        top_n = max(info["top_n"], 1)
        rank  = info["rank"]
        conf_bonus = cfg.top_conf_range * (top_n - rank) / max(top_n - 1, 1)
        conf = round(min(0.90, cfg.base_confidence + conf_bonus), 3)

        return AgentSignal(
            agent_name   = self.name,
            symbol       = state.symbol,
            action       = "BUY",
            confidence   = conf,
            target_weight= cfg.target_weight,
            reason       = (
                f"Momentum JT: rang {rank}/{info['n']} "
                f"(score={info['score']:+.2%}, top {info['top_n']})"
            ),
            meta = meta,
        )
