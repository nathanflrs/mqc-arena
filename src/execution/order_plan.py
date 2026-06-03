# src/execution/order_plan.py
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Iterable, Mapping, Optional

import pandas as pd


@dataclass(frozen=True)
class OrderPlanRow:
    timestamp: str
    agent: str
    symbol: str
    side: str            # BUY / SELL
    confidence: float    # 0..1
    price_ref: str       # MKT / LAST / LIMIT
    comment: str         # free text


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def _norm_side(x: Any) -> Optional[str]:
    if x is None:
        return None
    s = str(x).strip().upper()
    if s in {"BUY", "B"}:
        return "BUY"
    if s in {"SELL", "S"}:
        return "SELL"
    return None


def _clip01(v: Any) -> float:
    try:
        f = float(v)
    except Exception:
        return 0.0
    if f < 0.0:
        return 0.0
    if f > 1.0:
        return 1.0
    return f


def build_order_plan_df(
    *,
    winner_name: str,
    decisions: Iterable[Mapping[str, Any]],
    price_ref: str = "MKT",
    comment: str = "arena_winner",
    timestamp: Optional[str] = None,
) -> pd.DataFrame:
    """
    Build an order plan DataFrame from winner decisions.

    Expected decision keys (flexible):
      - symbol/ticker
      - side/action (BUY/SELL)
      - confidence/score (0..1 preferred)

    This function NEVER places orders. It only formats intentions.
    """
    ts = timestamp or _utc_now_iso()
    rows: list[dict[str, Any]] = []

    for d in decisions:
        symbol = (d.get("symbol") or d.get("ticker") or "").strip().upper()
        side = _norm_side(d.get("side") or d.get("action"))
        conf = _clip01(d.get("confidence") or d.get("score") or 0.0)

        # skip invalid lines
        if not symbol or side is None:
            continue

        rows.append(
            {
                "timestamp": ts,
                "agent": winner_name,
                "symbol": symbol,
                "side": side,
                "confidence": conf,
                "price_ref": price_ref,
                "comment": comment,
            }
        )

    df = pd.DataFrame(
        rows,
        columns=["timestamp", "agent", "symbol", "side", "confidence", "price_ref", "comment"],
    )
    return df


def write_order_plan_csv(
    *,
    path: str,
    df: pd.DataFrame,
    mode: str = "overwrite",  # "overwrite" | "append"
) -> None:
    if mode not in {"overwrite", "append"}:
        raise ValueError("mode must be 'overwrite' or 'append'")

    if mode == "overwrite":
        df.to_csv(path, index=False)
        return

    # append mode
    try:
        existing = pd.read_csv(path)
        out = pd.concat([existing, df], ignore_index=True)
    except FileNotFoundError:
        out = df
    out.to_csv(path, index=False)
