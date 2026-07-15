from __future__ import annotations
from dataclasses import dataclass, field
from typing import Dict, Optional

from ib_insync import IB


@dataclass
class PortfolioSnapshot:
    net_liquidation: float
    cash: float
    positions: Dict[str, float]        # symbol → quantity
    avg_costs: Dict[str, float] = field(default_factory=dict)  # symbol → IBKR avgCost


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

    pos: Dict[str, float] = {}
    avg_costs: Dict[str, float] = {}
    for p in ib.positions():
        sym = getattr(p.contract, "symbol", None)
        if not sym:
            continue
        pos[str(sym)] = float(p.position)
        cost = getattr(p, "avgCost", 0.0)
        if cost and float(cost) > 0:
            avg_costs[str(sym)] = float(cost)

    return PortfolioSnapshot(
        net_liquidation=nl, cash=cash, positions=pos, avg_costs=avg_costs
    )
