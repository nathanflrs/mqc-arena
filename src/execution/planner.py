from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Dict, List, Optional

from src.agents.base import AgentSignal

logger = logging.getLogger(__name__)

_STRATEGY_CTA = "cta_trend"


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
    # "directional" (default) or "market_neutral" — used by RiskManager
    # to route market-neutral pairs trades through the gross-exposure cap
    # instead of the net-long cap.
    strategy: str = "directional"


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
    # SELL: full exit regardless of target_weight (agents emit target_weight=0 on SELL)
    if signal.action == "SELL":
        delta = -float(current_qty)
        if abs(delta) < 1e-9:
            return OrderPlan(
                symbol=signal.symbol,
                action="HOLD",
                target_weight=0.0,
                last_price=float(last_price),
                current_qty=float(current_qty),
                target_qty=float(current_qty),
                delta_qty=0.0,
                est_notional=0.0,
                reason="Already flat",
            )
        return OrderPlan(
            symbol=signal.symbol,
            action="SELL",
            target_weight=0.0,
            last_price=float(last_price),
            current_qty=float(current_qty),
            target_qty=0.0,
            delta_qty=delta,
            est_notional=abs(delta) * float(last_price),
            reason=signal.reason,
        )

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


def cta_plan_from_signal(
    signal: AgentSignal,
    net_liquidation: float,
    last_price: float,
    current_qty: float,
) -> OrderPlan:
    """
    Planner CTA-aware : target_weight est signé.
      target_weight > 0  → long N% NAV
      target_weight < 0  → short N% NAV
      target_weight = 0  → flat (fermeture de toute position existante)

    Gère les reversals en un seul ordre :
      current_qty=+100 (long), target_weight=-0.10 → SELL 200 shares
      (ferme le long ET ouvre le short en une seule instruction IBKR).

    Stop-loss géré au niveau runner (STOP_LOSS_PCT), pas dans ce planner.
    """
    px = float(last_price)
    cq = float(current_qty)

    # ── No-op ──────────────────────────────────────────────────────────────────
    if signal.action == "HOLD" or (signal.target_weight == 0.0 and abs(cq) < 0.5):
        return OrderPlan(
            symbol=signal.symbol, action="HOLD",
            target_weight=float(signal.target_weight),
            last_price=px, current_qty=cq, target_qty=cq,
            delta_qty=0.0, est_notional=0.0,
            reason=signal.reason, strategy=_STRATEGY_CTA,
        )

    if px <= 0:
        logger.warning("cta_plan_from_signal: last_price=0 pour %s — HOLD", signal.symbol)
        return OrderPlan(
            symbol=signal.symbol, action="HOLD",
            target_weight=0.0, last_price=0.0,
            current_qty=cq, target_qty=cq,
            delta_qty=0.0, est_notional=0.0,
            reason="price=0 — impossible de sizer", strategy=_STRATEGY_CTA,
        )

    # ── Qty cible signée (positive=long, négative=short) ──────────────────────
    target_notional = float(signal.target_weight) * float(net_liquidation)
    target_qty = float(int(target_notional / px))    # troncature entière (US stocks)

    delta = target_qty - cq
    if abs(delta) < 0.5:
        return OrderPlan(
            symbol=signal.symbol, action="HOLD",
            target_weight=float(signal.target_weight),
            last_price=px, current_qty=cq, target_qty=target_qty,
            delta_qty=0.0, est_notional=0.0,
            reason="Already at target", strategy=_STRATEGY_CTA,
        )

    action = "BUY" if delta > 0 else "SELL"

    return OrderPlan(
        symbol=signal.symbol,
        action=action,
        target_weight=float(signal.target_weight),
        last_price=px,
        current_qty=cq,
        target_qty=target_qty,
        delta_qty=float(delta),
        est_notional=abs(float(delta)) * px,
        reason=signal.reason,
        strategy=_STRATEGY_CTA,
    )


def pairs_plans_from_signal(
    signal: AgentSignal,
    net_liquidation: float,
    prices: Dict[str, float],
    current_qtys: Dict[str, float],
) -> List[OrderPlan]:
    """
    Decompose a market-neutral pairs signal into two OrderPlan objects —
    one for the long leg and one for the short leg. Both are tagged
    strategy='market_neutral' so RiskManager routes them through the gross
    pairs cap instead of the net-long cap.

    Entry (action=BUY, direction='long_a_short_b')
    ----------------------------------------------
    Plan 1 : BUY ticker_a   — standard long leg
    Plan 2 : SELL ticker_b, current_qty=0  — opening short position
             IBKR paper trading handles SELL on non-held position as a short
             sale automatically. The short-clipping guard in execute_plans_paper_ibkr
             is bypassed for strategy='market_neutral' plans.

    Entry (action=BUY, direction='short_a_long_b')
    -----------------------------------------------
    Plan 1 : BUY ticker_b   — long leg
    Plan 2 : SELL ticker_a, current_qty=0  — opening short

    Close (action=SELL, direction='close')
    ---------------------------------------
    meta['pair_legs']['close_long']  → SELL to close the long leg
    meta['pair_legs']['close_short'] → BUY (cover) to close the short leg

    Returns [] for HOLD or if sizing is impossible (unknown price, qty < 1).

    Pairs round-trip P&L
    --------------------
    P&L(pair) = P&L(long leg) + P&L(short leg)
              = (close_long_px − open_long_px) × long_qty
              + (open_short_px − close_short_px) × short_qty

    The existing executions.csv tracks individual legs. To compute pair P&L,
    sum the signed P&L of all executions sharing the same pair_id tag in meta.
    """
    if signal.action == "HOLD":
        return []

    meta      = signal.meta or {}
    direction = meta.get("direction")
    pair_legs = meta.get("pair_legs")

    if not pair_legs or not direction:
        logger.warning("pairs_plans_from_signal: missing direction or pair_legs in signal.meta")
        return []

    # ── Entry — open both legs ─────────────────────────────────────────────────
    if signal.action == "BUY" and direction in ("long_a_short_b", "short_a_long_b"):
        long_ticker,  long_weight  = pair_legs["long"]
        short_ticker, short_weight = pair_legs["short"]

        long_px  = prices.get(long_ticker)
        short_px = prices.get(short_ticker)

        if not long_px or long_px <= 0:
            logger.warning("pairs_plans_from_signal: no valid price for long leg %s", long_ticker)
            return []
        if not short_px or short_px <= 0:
            logger.warning("pairs_plans_from_signal: no valid price for short leg %s", short_ticker)
            return []

        long_notional  = net_liquidation * float(long_weight)
        short_notional = net_liquidation * float(short_weight)
        long_qty  = int(long_notional  / long_px)
        short_qty = int(short_notional / short_px)

        if long_qty <= 0 or short_qty <= 0:
            logger.warning(
                "pairs_plans_from_signal: qty too small — %s=%d, %s=%d (notional %.0f/%.0f)",
                long_ticker, long_qty, short_ticker, short_qty, long_notional, short_notional,
            )
            return []

        long_current  = float(current_qtys.get(long_ticker,  0.0))
        short_current = float(current_qtys.get(short_ticker, 0.0))

        hedge_ratio = meta.get("hedge_ratio", "?")
        short_reason = f"{signal.reason} [short leg β={hedge_ratio}]"

        long_plan = OrderPlan(
            symbol=long_ticker,
            action="BUY",
            target_weight=float(long_weight),
            last_price=float(long_px),
            current_qty=long_current,
            target_qty=long_current + long_qty,
            delta_qty=float(long_qty),
            est_notional=float(long_qty * long_px),
            reason=signal.reason,
            strategy="market_neutral",
        )
        # Short leg: SELL with delta_qty < 0, current_qty may be 0 (opening fresh short).
        # The IBKR executor must NOT clip qty to current_qty for market_neutral SELL —
        # see execute_plans_paper_ibkr fix in runner.py.
        short_plan = OrderPlan(
            symbol=short_ticker,
            action="SELL",
            target_weight=-float(short_weight),
            last_price=float(short_px),
            current_qty=short_current,
            target_qty=short_current - short_qty,
            delta_qty=-float(short_qty),
            est_notional=float(short_qty * short_px),
            reason=short_reason,
            strategy="market_neutral",
        )
        return [long_plan, short_plan]

    # ── Close — exit both legs ─────────────────────────────────────────────────
    if signal.action == "SELL" and direction == "close":
        close_long  = pair_legs.get("close_long")
        close_short = pair_legs.get("close_short")

        if not close_long or not close_short:
            logger.warning("pairs_plans_from_signal: SELL signal missing close_long/close_short keys")
            return []

        long_px  = prices.get(close_long)
        short_px = prices.get(close_short)

        if not long_px or long_px <= 0 or not short_px or short_px <= 0:
            logger.warning("pairs_plans_from_signal: missing prices for close legs %s/%s",
                           close_long, close_short)
            return []

        long_current  = float(current_qtys.get(close_long,  0.0))
        short_current = float(current_qtys.get(close_short, 0.0))  # negative for short

        plans: List[OrderPlan] = []

        # Close long leg: SELL all held shares
        if long_current > 0:
            plans.append(OrderPlan(
                symbol=close_long,
                action="SELL",
                target_weight=0.0,
                last_price=float(long_px),
                current_qty=long_current,
                target_qty=0.0,
                delta_qty=-long_current,
                est_notional=long_current * float(long_px),
                reason=signal.reason,
                strategy="market_neutral",
            ))

        # Cover short leg: BUY to cover (short_current < 0 when short)
        if short_current < 0:
            cover_qty = abs(short_current)
            plans.append(OrderPlan(
                symbol=close_short,
                action="BUY",
                target_weight=0.0,
                last_price=float(short_px),
                current_qty=short_current,
                target_qty=0.0,
                delta_qty=cover_qty,
                est_notional=cover_qty * float(short_px),
                reason=f"{signal.reason} [cover short]",
                strategy="market_neutral",
            ))

        return plans

    logger.warning(
        "pairs_plans_from_signal: unhandled action=%s direction=%s", signal.action, direction
    )
    return []
