# src/data/market_data.py
from __future__ import annotations

import pandas as pd
import yfinance as yf


def _to_1d(x) -> pd.Series:
    """
    Force une Series 1D à partir de Series ou DataFrame (yfinance-safe).
    """
    if isinstance(x, pd.DataFrame):
        x = x.iloc[:, 0]
    return pd.to_numeric(x, errors="coerce")


def normalize_ohlcv(df: pd.DataFrame) -> pd.DataFrame:
    """
    Normalise un DataFrame OHLCV :
    - garantit Open/High/Low/Close/Volume en Series 1D float
    - supprime les NaN
    """
    required = ["Open", "High", "Low", "Close"]
    for col in required:
        if col not in df.columns:
            raise ValueError(f"Missing column {col}")

    out = pd.DataFrame(index=df.index)
    out["Open"] = _to_1d(df["Open"])
    out["High"] = _to_1d(df["High"])
    out["Low"] = _to_1d(df["Low"])
    out["Close"] = _to_1d(df["Close"])

    if "Volume" in df.columns:
        out["Volume"] = _to_1d(df["Volume"])

    return out.dropna()


def download_ohlcv(
    symbol: str,
    period: str = "2y",
    interval: str = "1d",
) -> pd.DataFrame:
    """
    Télécharge et normalise les données de marché.
    """
    raw = yf.download(
        symbol,
        period=period,
        interval=interval,
        auto_adjust=True,
        progress=False,
    )
    if raw is None or raw.empty:
        raise ValueError(f"No data for {symbol}")

    return normalize_ohlcv(raw)


def get_last_close_1d(df: pd.DataFrame) -> float:
    close = df["Close"]
    if isinstance(close, pd.DataFrame):
        close = close.iloc[:, 0]
    return float(close.dropna().iloc[-1])
