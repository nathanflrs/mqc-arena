# tests/test_market_data.py
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.data.market_data import normalize_ohlcv, get_last_close_1d


def _make_df(n: int = 10) -> pd.DataFrame:
    closes = np.linspace(100, 110, n)
    return pd.DataFrame(
        {
            "Open": closes - 0.5,
            "High": closes + 1.0,
            "Low": closes - 1.0,
            "Close": closes,
            "Volume": np.full(n, 1_000_000.0),
        }
    )


def test_normalize_ohlcv_returns_expected_columns():
    df = normalize_ohlcv(_make_df())
    assert set(["Open", "High", "Low", "Close", "Volume"]).issubset(df.columns)


def test_normalize_ohlcv_drops_nan():
    df = _make_df(10)
    df.loc[df.index[5], "Close"] = float("nan")
    result = normalize_ohlcv(df)
    assert result["Close"].isna().sum() == 0


def test_normalize_ohlcv_raises_on_missing_column():
    df = _make_df().drop(columns=["Close"])
    with pytest.raises(ValueError, match="Missing column"):
        normalize_ohlcv(df)


def test_get_last_close_1d_returns_float():
    df = normalize_ohlcv(_make_df())
    val = get_last_close_1d(df)
    assert isinstance(val, float)
    assert val == pytest.approx(110.0, rel=1e-3)


def test_get_last_close_1d_works_with_dataframe_close():
    df = _make_df()
    # Simule le cas où Close est un DataFrame multi-colonne (yfinance quirk)
    df["Close"] = pd.DataFrame({"A": df["Close"]})
    val = get_last_close_1d(df)
    assert isinstance(val, float)
