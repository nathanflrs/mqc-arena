# src/execution/logger.py
from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

import pandas as pd


def _utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def log_order_plan(plans: list, plan_id: str) -> None:
    """
    Écrit les plans d'ordre dans logs/order_plan.csv
    """
    Path("logs").mkdir(parents=True, exist_ok=True)
    ts = _utc_now()

    rows = []
    for p in plans:
        rows.append({
            "plan_id": plan_id,
            "timestamp": ts,
            "symbol": p.symbol,
            "side": "BUY" if p.delta_qty > 0 else ("SELL" if p.delta_qty < 0 else "HOLD"),
            "delta_qty": float(p.delta_qty),
            "target_qty": float(p.target_qty),
            "last_price": float(p.last_price),
            "est_notional": float(p.est_notional),
            "target_weight": float(p.target_weight),
            "reason": str(p.reason),
        })

    df = pd.DataFrame(rows, columns=[
        "plan_id", "timestamp", "symbol", "side",
        "delta_qty", "target_qty", "last_price",
        "est_notional", "target_weight", "reason",
    ])
    df.to_csv("logs/order_plan.csv", index=False)
    print("✅ Wrote logs/order_plan.csv")


def log_execution(exec_rows: list) -> None:
    """
    Ajoute les exécutions dans logs/executions.csv
    """
    Path("logs").mkdir(parents=True, exist_ok=True)
    df_new = pd.DataFrame(exec_rows)

    out_path = "logs/executions.csv"
    try:
        df_old = pd.read_csv(out_path)
        df_all = pd.concat([df_old, df_new], ignore_index=True)
    except FileNotFoundError:
        df_all = df_new

    df_all.to_csv(out_path, index=False)
    print("✅ Logged to logs/executions.csv")


def log_decisions(
    signals: list,
    symbol: str,
    regime: str,
    plan_id: str,
    winner_agent: str | None = None,
) -> None:
    """
    Ajoute les signaux de chaque agent dans logs/decisions.csv.
    `winner_agent` : nom de l'agent sélectionné par le sélecteur pour ce run.
    """
    Path("logs").mkdir(parents=True, exist_ok=True)
    ts = _utc_now()

    rows = []
    for s in signals:
        rows.append({
            "plan_id": plan_id,
            "timestamp": ts,
            "symbol": symbol,
            "regime": regime,
            "agent": s.agent_name,
            "action": s.action,
            "confidence": s.confidence,
            "target_weight": s.target_weight,
            "reason": s.reason,
            "is_winner": s.agent_name == winner_agent,
        })

    cols = ["plan_id", "timestamp", "symbol", "regime", "agent",
            "action", "confidence", "target_weight", "reason", "is_winner"]
    df_new = pd.DataFrame(rows, columns=cols)
    out_path = "logs/decisions.csv"
    try:
        df_old = pd.read_csv(out_path)
        df_all = pd.concat([df_old, df_new], ignore_index=True)
    except FileNotFoundError:
        df_all = df_new

    df_all.to_csv(out_path, index=False)