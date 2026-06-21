# src/risk/vol_sizing.py
"""
Volatility-adjusted position sizing.

Rule: when a symbol's 20-day realized volatility exceeds 2× its 1-year
average realized volatility, the target weight is halved.

This is a variance-reduction mechanism — not an alpha signal.  It fires
only during genuine volatility spikes (earnings surprises, macro shocks,
idiosyncratic blowups) and leaves the allocation untouched in normal markets.

Usage (runner)
--------------
    from src.risk.vol_sizing import vol_adjusted_weight
    adj_w, reason = vol_adjusted_weight(df, winner.target_weight)
    if reason:
        winner = dataclasses.replace(winner, target_weight=adj_w)

Usage (backtest engine — inline, see engine.py)
------------------------------------------------
    The same logic is replicated inline inside BacktestEngine.run() to
    avoid re-creating a DataFrame slice on every iteration.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def vol_adjusted_weight(
    df: pd.DataFrame,
    base_weight: float,
    *,
    vol_window: int = 20,
    vol_lookback: int = 252,
    vol_multiplier: float = 2.0,
    size_divisor: float = 2.0,
) -> tuple[float, str]:
    """
    Parameters
    ----------
    df            : OHLCV DataFrame (needs at least vol_lookback + vol_window rows)
    base_weight   : current target weight (e.g. 0.08)
    vol_window    : short-term vol window in trading days (default 20)
    vol_lookback  : long-run vol lookback in trading days (default 252 = 1 year)
    vol_multiplier: ratio threshold above which the rule fires (default 2×)
    size_divisor  : divisor applied to base_weight when rule fires (default 2)

    Returns
    -------
    (adjusted_weight, reason_str)
    reason_str is non-empty only when the adjustment fires.
    """
    close = df["Close"]
    if isinstance(close, pd.DataFrame):
        close = close.iloc[:, 0]
    close = pd.to_numeric(close, errors="coerce").dropna()

    # Need enough history for both windows
    if len(close) < vol_lookback + vol_window:
        return base_weight, ""

    log_ret = np.log(close / close.shift(1)).dropna()

    vol_20d = float(log_ret.iloc[-vol_window:].std() * np.sqrt(252))
    vol_1y  = float(log_ret.iloc[-vol_lookback:].std() * np.sqrt(252))

    if vol_1y <= 0:
        return base_weight, ""

    if vol_20d / vol_1y > vol_multiplier:
        adjusted = round(base_weight / size_divisor, 4)
        reason = (
            f"vol_adj: σ20d={vol_20d:.0%} > {vol_multiplier:.0f}×σ1y={vol_1y:.0%}"
            f" [{base_weight:.4f}→{adjusted:.4f}]"
        )
        return adjusted, reason

    return base_weight, ""
