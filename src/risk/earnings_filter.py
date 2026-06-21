# src/risk/earnings_filter.py
"""
Earnings proximity filter.

Blocks BUY orders placed within N business days of a company's earnings
release to avoid unintended binary-event exposure. SELL orders are never
blocked — this is a defensive buy-side filter only.

Data source : yfinance Ticker.get_earnings_dates()
Caching     : in-memory, one fetch per run (call prefetch() once)
Fallback    : if yfinance fails for a symbol, that symbol is not blocked
              (fail-open — never blocks a trade due to a data error)
"""
from __future__ import annotations

import logging
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
import yfinance as yf

log = logging.getLogger(__name__)


class EarningsFilter:
    """
    Pre-fetches earnings dates for a universe of symbols and exposes a single
    `should_block_buy()` method used by the runner before placing orders.

    Parameters
    ----------
    buffer_days : int
        Number of *business* days around an earnings date during which a BUY
        is blocked.  0 = earnings day only, 3 = earnings day + 3 bdays before.
    fetch_limit : int
        How many upcoming (and recent historical) earnings dates to fetch per
        symbol via yfinance.  20 covers ~5 years of quarterly history.
    """

    def __init__(self, buffer_days: int = 3, fetch_limit: int = 20):
        self.buffer_days = buffer_days
        self.fetch_limit = fetch_limit
        self._cache: Dict[str, pd.DatetimeIndex] = {}

    # ── Public API ────────────────────────────────────────────────────────────

    def prefetch(self, symbols: List[str]) -> None:
        """
        Batch-download earnings dates for every symbol.  Call once per run
        before the main per-symbol loop so all subsequent calls are O(1).
        """
        for sym in symbols:
            self._cache[sym] = self._fetch_one(sym)
        loaded = sum(1 for v in self._cache.values() if len(v) > 0)
        print(f"[EarningsFilter] dates loaded for {loaded}/{len(symbols)} symbols")

    def should_block_buy(
        self,
        symbol: str,
        as_of: Optional[pd.Timestamp] = None,
    ) -> tuple[bool, str]:
        """
        Returns ``(blocked: bool, reason: str)``.

        blocked = True  iff any earnings date is within [today, today + buffer_days]
                         business days (inclusive).
        Fetches on-demand if the symbol was not prefetched.
        """
        today = (as_of or pd.Timestamp.today()).normalize().date()

        if symbol not in self._cache:
            self._cache[symbol] = self._fetch_one(symbol)

        for d in self._cache[symbol]:
            delta = int(np.busday_count(today, pd.Timestamp(d).date()))
            if 0 <= delta <= self.buffer_days:
                label = "today" if delta == 0 else f"in {delta} bday{'s' if delta > 1 else ''}"
                return True, f"earnings {label} ({pd.Timestamp(d).date()})"

        return False, ""

    # ── Internal ──────────────────────────────────────────────────────────────

    def _fetch_one(self, symbol: str) -> pd.DatetimeIndex:
        """Fetch earnings dates for one symbol; return empty index on failure."""
        try:
            t = yf.Ticker(symbol)
            df = t.get_earnings_dates(limit=self.fetch_limit)
            if df is None or df.empty:
                return pd.DatetimeIndex([])
            return pd.DatetimeIndex(df.index.normalize())
        except Exception as exc:
            log.debug("EarningsFilter: %s — fetch failed: %s", symbol, exc)
            return pd.DatetimeIndex([])
