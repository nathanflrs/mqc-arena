from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, Tuple, Optional

from ib_insync import IB


@dataclass
class PortfolioSnapshot:
    net_liquidation: float
    cash: float
    positions: Dict[str, float]  # symbol -> quantity


def _to_float(x: Optional[str], default: float = 0.0) -> float:
    try:
        return float(x)
    except Exception:
        return default


def fetch_account_snapshot(ib: IB) -> PortfolioSnapshot:
    """
    Récupère NetLiquidation, TotalCashValue et positions (stocks) depuis IBKR.
    """
    summary = ib.accountSummary()

    nl = 0.0
    cash = 0.0
    for row in summary:
        if row.tag == "NetLiquidation":
            nl = _to_float(row.value)
        elif row.tag == "TotalCashValue":
            cash = _to_float(row.value)

    pos = {}
    for p in ib.positions():
        # p.contract.symbol (ex: AAPL), p.position (qty)
        sym = getattr(p.contract, "symbol", None)
        if not sym:
            continue
        pos[str(sym)] = float(p.position)

    return PortfolioSnapshot(net_liquidation=nl, cash=cash, positions=pos)
