from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, Optional

from src.agents.base import AgentSignal


@dataclass
class OrderPlan:
    symbol: str
    action: str              # BUY / SELL / HOLD
    target_weight: float     # ex 0.10
    last_price: float
    current_qty: float
    target_qty: float
    delta_qty: float         # qty à acheter/vendre (+/-)
    est_notional: float      # $ approximatif de delta
    reason: str


def plan_from_signal(
    signal: AgentSignal,
    net_liquidation: float,
    last_price: float,
    current_qty: float,
    max_leverage: float = 1.0,   # paper simple, 1.0 = no leverage
) -> Optional[OrderPlan]:
    """
    Convertit un signal (target_weight) en quantité cible.
    Ici on fait SIMPLE: target_$ = netliq * target_weight.
    target_qty = target_$ / price.
    """
    if signal.action == "HOLD" or signal.target_weight <= 0:
        return OrderPlan(
            symbol=signal.symbol,
            action="HOLD",
            target_weight=float(signal.target_weight),
            last_price=float(last_price),
            current_qty=float(current_qty),
            target_qty=float(current_qty),
            delta_qty=0.0,
            est_notional=0.0,
            reason=signal.reason,
        )

    target_dollars = (net_liquidation * max_leverage) * float(signal.target_weight)
    target_qty = target_dollars / float(last_price)

    # On arrondit à 1 action (stocks US)
    target_qty_rounded = float(int(target_qty))

    delta = target_qty_rounded - float(current_qty)
    if abs(delta) < 1e-9:
        # déjà au target -> HOLD
        return OrderPlan(
            symbol=signal.symbol,
            action="HOLD",
            target_weight=float(signal.target_weight),
            last_price=float(last_price),
            current_qty=float(current_qty),
            target_qty=target_qty_rounded,
            delta_qty=0.0,
            est_notional=0.0,
            reason="Already at target",
        )

    action = "BUY" if delta > 0 else "SELL"
    est_notional = abs(delta) * float(last_price)

    return OrderPlan(
        symbol=signal.symbol,
        action=action,
        target_weight=float(signal.target_weight),
        last_price=float(last_price),
        current_qty=float(current_qty),
        target_qty=target_qty_rounded,
        delta_qty=float(delta),
        est_notional=float(est_notional),
        reason=signal.reason,
    )
