# src/market/view.py
"""
Ce que les agents voient du marché : l'univers entier, à une date.

Une seule règle, et elle protège tout le reste : un agent ne voit jamais une
donnée postérieure à `as_of`. `until()` coupe chaque série à une date passée,
ce qui permet de rejouer exactement ce qu'un agent aurait vu ce jour-là.

Limite à connaître : `until()` coupe les PRIX, pas la composition de l'indice.
Les membres sont ceux du jour de la collecte. Pour un rejeu historique sans
biais du survivant, reconstituer la composition avec `src/data/universe.py`.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import date
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd

from src.data.snapshot import load_snapshot

MARKET_DIR = Path("logs/market")
PRICES_DIR = "prices"
MEMBERS_FILE = "members.json"
REPORT_FILE = "report.json"


@dataclass
class MarketView:
    as_of: date
    members: pd.DataFrame                  # index = ticker ; sector, sub_industry, cik
    ohlcv: Dict[str, pd.DataFrame]
    _close: Optional[pd.DataFrame] = field(default=None, repr=False)

    @property
    def tickers(self) -> List[str]:
        return sorted(self.ohlcv)

    def close(self) -> pd.DataFrame:
        """Panneau des clôtures : dates × tickers."""
        if self._close is None:
            self._close = pd.DataFrame(
                {t: df["Close"] for t, df in self.ohlcv.items()}
            ).sort_index()
        return self._close

    def until(self, d: date | str | pd.Timestamp) -> "MarketView":
        """La vue telle qu'elle était à la clôture de `d` — rien après."""
        ts = pd.Timestamp(d)
        cut = {t: df.loc[:ts] for t, df in self.ohlcv.items()}
        return MarketView(
            as_of=ts.date(),
            members=self.members,
            ohlcv={t: df for t, df in cut.items() if len(df)},
        )

    def peers(self, ticker: str) -> List[str]:
        """Les autres membres de la même sous-industrie GICS, ayant des prix."""
        if ticker not in self.members.index:
            return []
        sub = self.members.at[ticker, "sub_industry"]
        if not sub:
            return []
        same = self.members.index[self.members["sub_industry"] == sub]
        return sorted(t for t in same if t != ticker and t in self.ohlcv)


def load_market_view(root: Path | str = MARKET_DIR) -> MarketView:
    """Charge la dernière collecte. Un snapshot altéré est refusé, pas lu en silence."""
    root = Path(root)
    data, _manifest, tampered = load_snapshot(root / PRICES_DIR)
    if tampered:
        raise ValueError(f"collecte altérée depuis son écriture : {', '.join(tampered)}")
    if not data:
        raise ValueError(f"collecte vide : {root / PRICES_DIR}")
    members = pd.DataFrame(
        json.loads((root / MEMBERS_FILE).read_text())["members"]
    ).set_index("ticker")
    as_of = max(df.index[-1] for df in data.values()).date()
    return MarketView(as_of=as_of, members=members, ohlcv=data)
