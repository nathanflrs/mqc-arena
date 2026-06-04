# src/risk/correlation.py
"""
Portfolio correlation guard.

Blocks a BUY signal if the candidate asset is too correlated (|r| ≥ threshold)
with any currently open position, preventing over-concentration into a single
market factor (e.g. buying QQQ when SPY and AAPL are already held).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import pandas as pd
import numpy as np


@dataclass
class CorrelationCheckResult:
    allowed: bool
    max_correlation: float            # highest |r| found (0.0 if no open positions)
    correlated_with: Optional[str]    # symbol that triggered the block
    reason: str


class CorrelationGuard:
    """
    Computes rolling return correlations between open positions and a
    candidate BUY asset; blocks if |r| exceeds *threshold*.

    Parameters
    ----------
    threshold : float
        Maximum allowed absolute correlation (default 0.7).
    lookback_days : int
        Number of calendar days of daily returns to use (default 60).
    min_overlap : int
        Minimum number of overlapping data points required to compute a
        valid correlation; otherwise the check is skipped (default 20).
    """

    def __init__(
        self,
        threshold: float = 0.7,
        lookback_days: int = 60,
        min_overlap: int = 20,
    ) -> None:
        self.threshold = threshold
        self.lookback_days = lookback_days
        self.min_overlap = min_overlap

    def check_buy(
        self,
        candidate: str,
        open_symbols: List[str],
        price_data: Dict[str, pd.DataFrame],
    ) -> CorrelationCheckResult:
        """
        Check whether *candidate* is too correlated with any open position.

        Parameters
        ----------
        candidate : str
            Ticker of the asset being considered for purchase.
        open_symbols : list[str]
            Tickers of currently open positions.
        price_data : dict[str, DataFrame]
            OHLCV DataFrames keyed by ticker (as returned by download_ohlcv).

        Returns
        -------
        CorrelationCheckResult
        """
        # Nothing to compare against → always pass
        peers = [s for s in open_symbols if s != candidate]
        if not peers:
            return CorrelationCheckResult(
                allowed=True, max_correlation=0.0, correlated_with=None, reason=""
            )

        # Build return series for all relevant tickers
        series: Dict[str, pd.Series] = {}
        for sym in [candidate] + peers:
            df = price_data.get(sym)
            if df is None or df.empty:
                continue
            close = pd.to_numeric(df["Close"], errors="coerce").dropna()
            ret = close.pct_change().dropna().iloc[-self.lookback_days:]
            if len(ret) >= self.min_overlap:
                series[sym] = ret

        if candidate not in series or len(series) < 2:
            # Insufficient data — allow the trade but note it
            return CorrelationCheckResult(
                allowed=True,
                max_correlation=0.0,
                correlated_with=None,
                reason="données insuffisantes pour le contrôle de corrélation",
            )

        max_corr = 0.0
        worst_sym: Optional[str] = None

        for peer in peers:
            if peer not in series:
                continue
            # Align on common dates
            aligned = pd.concat(
                [series[candidate].rename("cand"), series[peer].rename("peer")],
                axis=1,
            ).dropna()
            if len(aligned) < self.min_overlap:
                continue
            corr = float(aligned["cand"].corr(aligned["peer"]))
            if np.isnan(corr):
                continue
            abs_corr = abs(corr)
            if abs_corr > max_corr:
                max_corr = abs_corr
                worst_sym = peer

        if worst_sym is None:
            return CorrelationCheckResult(
                allowed=True, max_correlation=0.0, correlated_with=None, reason=""
            )

        allowed = max_corr < self.threshold
        reason = (
            ""
            if allowed
            else (
                f"corrélation {max_corr:.2f} avec {worst_sym} "
                f"≥ seuil {self.threshold} — BUY bloqué"
            )
        )
        return CorrelationCheckResult(
            allowed=allowed,
            max_correlation=max_corr,
            correlated_with=worst_sym,
            reason=reason,
        )

    def filter_plans(
        self,
        plans: list,
        snap,
        all_data: Dict[str, pd.DataFrame],
    ) -> Tuple[list, List[dict]]:
        """
        Filter a list of OrderPlan objects; blocks BUY plans that are too
        correlated with already-held positions.

        Returns (approved_plans, blocked_info_list).
        *blocked_info_list* entries are dicts with keys:
            symbol, max_corr, correlated_with, reason.
        """
        # Start from actual IBKR positions; grow as we approve more BUYs
        open_symbols: List[str] = [
            sym for sym, qty in snap.positions.items() if qty > 0
        ]
        approved: list = []
        blocked: List[dict] = []

        for plan in plans:
            if plan.action != "BUY":
                approved.append(plan)
                continue

            check = self.check_buy(plan.symbol, open_symbols, all_data)
            if check.allowed:
                approved.append(plan)
                # Add to open set so subsequent candidates in this batch see it
                if plan.symbol not in open_symbols:
                    open_symbols.append(plan.symbol)
            else:
                blocked.append({
                    "symbol":          plan.symbol,
                    "max_corr":        round(check.max_correlation, 4),
                    "correlated_with": check.correlated_with,
                    "reason":          check.reason,
                })

        return approved, blocked

    def correlation_matrix(
        self,
        symbols: List[str],
        price_data: Dict[str, pd.DataFrame],
    ) -> pd.DataFrame:
        """
        Return a pairwise correlation matrix (DataFrame) for *symbols*,
        used by the dashboard for display.
        """
        series: Dict[str, pd.Series] = {}
        for sym in symbols:
            df = price_data.get(sym)
            if df is None or df.empty:
                continue
            close = pd.to_numeric(df["Close"], errors="coerce").dropna()
            ret = close.pct_change().dropna().iloc[-self.lookback_days:]
            if len(ret) >= self.min_overlap:
                series[sym] = ret

        if len(series) < 2:
            return pd.DataFrame()

        return pd.DataFrame(series).dropna().corr()
