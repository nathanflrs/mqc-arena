# src/data/regime.py
from __future__ import annotations

import pandas as pd
from src.data.market_data import download_ohlcv


def compute_atr(df: pd.DataFrame, period: int = 14) -> float:
    high = df["High"]
    low = df["Low"]
    close = df["Close"].shift(1)

    tr = pd.concat([
        high - low,
        (high - close).abs(),
        (low - close).abs(),
    ], axis=1).max(axis=1)

    return float(tr.rolling(period).mean().iloc[-1])


def detect_regime(symbol: str = "SPY", df: pd.DataFrame | None = None) -> dict:
    """
    Détecte le régime de marché sur la base de SPY.

    Returns:
        {
            "regime": "bull" | "bear" | "choppy",
            "price": float,
            "sma50": float,
            "sma200": float,
            "atr14": float,
            "vol_regime": "low" | "normal" | "high",
        }
    """
    if df is None:
        df = download_ohlcv(symbol, period="2y")

    close = df["Close"]
    price = float(close.iloc[-1])
    sma50 = float(close.rolling(50).mean().iloc[-1])
    sma200 = float(close.rolling(200).mean().iloc[-1])
    atr14 = compute_atr(df)

    # --- régime de tendance ---
    if price > sma50 and sma50 > sma200:
        regime = "bull"
    elif price < sma50 and sma50 < sma200:
        regime = "bear"
    else:
        regime = "choppy"

    # --- régime de volatilité ---
    atr_pct = atr14 / price
    if atr_pct < 0.008:
        vol_regime = "low"
    elif atr_pct > 0.018:
        vol_regime = "high"
    else:
        vol_regime = "normal"

    return {
        "regime": regime,
        "price": round(price, 2),
        "sma50": round(sma50, 2),
        "sma200": round(sma200, 2),
        "atr14": round(atr14, 2),
        "vol_regime": vol_regime,
    }
