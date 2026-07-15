# src/risk/var_engine.py
"""
Historical VaR engine — 1-day 95%/99% Value-at-Risk (historical simulation).

Method: full-revaluation historical simulation.
  1. For each position, align the last `lookback_days` of daily log-returns.
  2. Compute portfolio daily P&L series: sum(qty_i × price_i × return_i).
  3. VaR_α = −percentile(P&L, 1−α)   (positive number = potential loss).
  4. CVaR_α = −mean(P&L ≤ −VaR_α)    (Expected Shortfall).

Results are logged to logs/var_latest.json and returned as a VaRResult.
"""
from __future__ import annotations

import json
import logging
from dataclasses import dataclass, asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

_VAR_LOG = Path("logs/var_latest.json")


@dataclass
class VaRResult:
    var_95_pct: float       # 1-day VaR 95% as fraction of portfolio  (e.g. 0.018 = 1.8%)
    var_99_pct: float       # 1-day VaR 99% as fraction of portfolio
    var_95_usd: float       # absolute $ loss at 95%
    var_99_usd: float       # absolute $ loss at 99%
    cvar_99_pct: float      # Expected Shortfall 99% (fraction)
    cvar_99_usd: float      # Expected Shortfall 99% ($)
    portfolio_value: float  # mark-to-market value of positions used
    n_days: int             # trading days of history actually used
    n_positions: int        # number of non-zero positions included
    computed_at: str        # ISO-8601 UTC timestamp


@dataclass
class VaRConfig:
    lookback_days: int   = 252     # 1 calendar year of daily returns
    risk_budget_pct: float = 0.03  # alert if 1-day 99% VaR > 3% of netliq
    min_history_days: int = 21     # minimum days of common history required


class HistoricalVaREngine:
    """
    Compute portfolio historical VaR from positions and OHLCV DataFrames.

    All DataFrames must have a 'Close' column with a DatetimeIndex.
    """

    def __init__(self, config: Optional[VaRConfig] = None) -> None:
        self.cfg = config or VaRConfig()

    # ── Public API ─────────────────────────────────────────────────────────────

    def compute(
        self,
        positions: Dict[str, float],
        all_data:  Dict[str, pd.DataFrame],
        net_liquidation: float,
    ) -> Optional[VaRResult]:
        """
        Return VaRResult for the current positions, or None when there are
        insufficient positions / history to produce a meaningful estimate.

        Parameters
        ----------
        positions       : symbol → number of shares held (0 or negative skipped)
        all_data        : symbol → OHLCV DataFrame with 'Close' column
        net_liquidation : total portfolio value (used for % normalisation)
        """
        active = {sym: qty for sym, qty in positions.items()
                  if qty > 0 and sym in all_data}
        if not active:
            logger.debug("VaR: no long positions — skipping")
            return None

        # Build aligned returns matrix
        close_dict: Dict[str, pd.Series] = {}
        for sym, qty in active.items():
            df = all_data[sym]
            if "Close" not in df.columns or len(df) < self.cfg.min_history_days:
                continue
            close = df["Close"].sort_index().tail(self.cfg.lookback_days + 1)
            close_dict[sym] = close

        if not close_dict:
            logger.warning("VaR: no usable price history in positions")
            return None

        # Align to common date range
        closes = pd.DataFrame(close_dict).dropna()
        if len(closes) < self.cfg.min_history_days:
            logger.warning("VaR: only %d common days — need %d", len(closes), self.cfg.min_history_days)
            return None

        closes = closes.tail(self.cfg.lookback_days + 1)
        returns = closes.pct_change().dropna()

        # Notional weights: qty × last close price
        last_prices = closes.iloc[-1]
        notionals: Dict[str, float] = {}
        for sym in returns.columns:
            qty = active.get(sym, 0.0)
            notionals[sym] = qty * float(last_prices[sym])

        portfolio_value = sum(notionals.values())
        if portfolio_value <= 0:
            return None

        # Portfolio daily P&L (in $)
        pnl_series = sum(
            returns[sym] * notionals[sym]
            for sym in returns.columns
        )

        result = self._compute_var(pnl_series, portfolio_value, len(active))
        self._persist(result)
        return result

    def exceeds_budget(self, result: VaRResult, net_liquidation: float) -> bool:
        """True when 1-day 99% VaR exceeds the configured risk budget % of netliq."""
        if net_liquidation <= 0:
            return False
        return (result.var_99_usd / net_liquidation) > self.cfg.risk_budget_pct

    # ── Internals ──────────────────────────────────────────────────────────────

    def _compute_var(
        self,
        pnl: pd.Series,
        portfolio_value: float,
        n_positions: int,
    ) -> VaRResult:
        arr = pnl.values

        var_95_usd = float(-np.percentile(arr, 5))
        var_99_usd = float(-np.percentile(arr, 1))

        tail_99 = arr[arr <= -var_99_usd]
        cvar_99_usd = float(-tail_99.mean()) if len(tail_99) > 0 else var_99_usd

        def _pct(v: float) -> float:
            return round(v / portfolio_value, 6) if portfolio_value > 0 else 0.0

        return VaRResult(
            var_95_pct      = _pct(var_95_usd),
            var_99_pct      = _pct(var_99_usd),
            var_95_usd      = round(var_95_usd, 2),
            var_99_usd      = round(var_99_usd, 2),
            cvar_99_pct     = _pct(cvar_99_usd),
            cvar_99_usd     = round(cvar_99_usd, 2),
            portfolio_value = round(portfolio_value, 2),
            n_days          = len(pnl),
            n_positions     = n_positions,
            computed_at     = datetime.now(timezone.utc).replace(microsecond=0).isoformat(),
        )

    def _persist(self, result: VaRResult) -> None:
        try:
            _VAR_LOG.parent.mkdir(parents=True, exist_ok=True)
            _VAR_LOG.write_text(json.dumps(asdict(result), indent=2))
        except Exception as exc:
            logger.warning("VaR: could not write %s — %s", _VAR_LOG, exc)
