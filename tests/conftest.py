# tests/conftest.py
from __future__ import annotations

from datetime import date as _date

import numpy as np
import pandas as pd
import pytest

from src.agents.base import MarketState


def bdate_index(n: int, end=None) -> pd.DatetimeIndex:
    """
    Exactement `n` jours ouvrés se terminant à `end` (défaut : aujourd'hui).

    Pourquoi ce helper existe
    -------------------------
    `pd.date_range(end=X, periods=n, freq="B")` renvoie **n-1** dates quand X
    n'est pas un jour ouvré (pandas 3.0). Les fixtures qui utilisaient
    `end=date.today()` produisaient donc un index de n-1 dates pour n valeurs,
    et levaient `ValueError: Length of values (300) does not match length of
    index (299)` — mais **uniquement le samedi et le dimanche**. 47 tests
    passaient du lundi au vendredi et échouaient le week-end.

    On ramène `end` au jour ouvré précédent avant de générer l'intervalle :
    le compte est exact tous les jours, et la sémantique « série se terminant
    aujourd'hui » est préservée.
    """
    end_ts = pd.offsets.BDay().rollback(pd.Timestamp(end or _date.today()).normalize())
    return pd.date_range(end=end_ts, periods=n, freq="B")


def _make_ohlcv(closes: np.ndarray) -> pd.DataFrame:
    n = len(closes)
    idx = pd.date_range("2022-01-01", periods=n, freq="B")
    noise = np.abs(closes * 0.005)
    return pd.DataFrame(
        {
            "Open": closes - noise,
            "High": closes + noise * 2,
            "Low": closes - noise * 2,
            "Close": closes,
            "Volume": np.full(n, 1_000_000, dtype=float),
        },
        index=idx,
    )


@pytest.fixture
def bull_df():
    """300 jours de tendance haussière forte — déclenche les signaux BUY momentum."""
    closes = np.linspace(100, 160, 300)
    closes += np.random.default_rng(42).normal(0, 0.3, 300)
    return _make_ohlcv(closes)


@pytest.fixture
def bear_df():
    """300 jours de tendance baissière — déclenche les signaux SELL."""
    closes = np.linspace(160, 80, 300)
    closes += np.random.default_rng(42).normal(0, 0.3, 300)
    return _make_ohlcv(closes)


@pytest.fixture
def flat_df():
    """300 jours de marché plat — devrait produire HOLD."""
    closes = np.full(300, 100.0)
    closes += np.random.default_rng(42).normal(0, 0.5, 300)
    closes = np.abs(closes)
    return _make_ohlcv(closes)


@pytest.fixture
def oversold_df():
    """300 jours : hausse puis chute rapide → RSI bas + sous Bollinger → MeanReversion BUY."""
    up = np.linspace(100, 140, 250)
    down = np.linspace(140, 100, 50)
    closes = np.concatenate([up, down])
    closes += np.random.default_rng(0).normal(0, 0.2, 300)
    df = _make_ohlcv(closes)
    # Augmenter le volume sur les derniers jours (signal de volume pour MeanReversion)
    df.iloc[-5:, df.columns.get_loc("Volume")] = 2_500_000
    return df


@pytest.fixture
def short_df():
    """Historique trop court — tous les agents doivent retourner HOLD."""
    closes = np.linspace(100, 110, 20)
    return _make_ohlcv(closes)


def make_state(symbol: str = "AAPL", price: float = 150.0) -> MarketState:
    return MarketState(symbol=symbol, price=price, timestamp="2024-01-01T00:00:00")
