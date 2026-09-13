# src/data/prices.py
"""
Source des prix quotidiens — le seul point d'entrée du fonds vers un
fournisseur de prix.

Aujourd'hui yfinance, choisi le 2026-09-13 « pour le moment ». Le jour où
l'équipe paiera un fournisseur, on écrira une seconde classe qui respecte
`PriceSource` : rien d'autre ne changera.

Ce que ce module ne fait PAS : juger la qualité des séries. C'est le rôle de
`src/data/quality.py`, appliqué par l'appelant — un fournisseur ne note pas sa
propre copie.
"""
from __future__ import annotations

import logging
import time
from typing import Callable, Dict, Optional, Protocol, Sequence

import pandas as pd

from src.data.market_data import normalize_ohlcv

logger = logging.getLogger(__name__)


class PriceSource(Protocol):
    name: str

    def daily(self, tickers: Sequence[str], period: str = "3y") -> Dict[str, pd.DataFrame]:
        """
        OHLCV quotidien ajusté, par ticker.

        Un ticker sans données est ABSENT du résultat, jamais rempli de vide :
        c'est à l'appelant de chiffrer ce qui manque.
        """
        ...


class YFinanceSource:
    """yfinance par lots, avec le réglage qui tenait déjà pour l'univers de recherche."""

    name = "yfinance"

    def __init__(
        self,
        chunk: int = 40,          # au-delà, yfinance devient instable
        pause: float = 1.5,       # entre lots, pour ne pas se faire limiter
        min_rows: int = 60,       # trop court pour tout indicateur long
        downloader: Optional[Callable[..., pd.DataFrame]] = None,
    ) -> None:
        self.chunk = chunk
        self.pause = pause
        self.min_rows = min_rows
        self._download = downloader

    def _downloader(self) -> Callable[..., pd.DataFrame]:
        if self._download is None:
            import yfinance as yf
            self._download = yf.download
        return self._download

    def daily(self, tickers: Sequence[str], period: str = "3y") -> Dict[str, pd.DataFrame]:
        download = self._downloader()
        tickers = list(dict.fromkeys(tickers))
        out: Dict[str, pd.DataFrame] = {}

        for i in range(0, len(tickers), self.chunk):
            lot = tickers[i:i + self.chunk]
            try:
                brut = download(lot, period=period, interval="1d", auto_adjust=True,
                                progress=False, group_by="ticker", threads=True)
            except Exception as exc:
                logger.warning("yfinance : lot %d en échec (%s)", i // self.chunk + 1, exc)
                brut = None

            if brut is not None and not brut.empty:
                for t in lot:
                    df = _extract(brut, t, single=len(lot) == 1)
                    if df is not None and len(df) >= self.min_rows:
                        out[t] = df

            if i + self.chunk < len(tickers) and self.pause:
                time.sleep(self.pause)
        return out


def _extract(brut: pd.DataFrame, ticker: str, single: bool) -> Optional[pd.DataFrame]:
    """La série d'un ticker dans une réponse groupée — ou None si elle n'y est pas."""
    try:
        if isinstance(brut.columns, pd.MultiIndex):
            if ticker not in brut.columns.get_level_values(0):
                return None
            df = brut[ticker].copy()
        elif single:
            df = brut.copy()
        else:
            return None
        return normalize_ohlcv(df).dropna(subset=["Close"])
    except Exception:
        return None
