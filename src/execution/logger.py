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


def compute_pair_roundtrip_pnl(
    executions_path: str = "logs/executions.csv",
    long_ticker: str = "",
    short_ticker: str = "",
) -> Optional[dict]:
    """
    Compute the round-trip P&L for a market-neutral pairs trade.

    Pairs P&L = P&L(long leg) + P&L(short leg)

        Long leg  P&L = (exit_price − entry_price) × long_qty
        Short leg P&L = (entry_short_price − exit_short_price) × short_qty

    A round-trip is identified by matching BUY/SELL pairs for each leg in
    executions.csv. This is a best-effort FIFO match — for live scoring,
    use the pair_id field in the execution log.

    Returns None if insufficient executions are found.

    Note on single-leg risk
    -----------------------
    If only one leg of a pair trade appears in executions (e.g., long opened
    but short failed), the position is NOT market-neutral. The caller must
    detect this by checking that BOTH legs have entries before interpreting
    the P&L as market-neutral.
    """
    import os
    if not os.path.exists(executions_path):
        return None

    df = pd.read_csv(executions_path)
    if df.empty or "symbol" not in df.columns:
        return None

    result = {}

    for ticker, is_long in [(long_ticker, True), (short_ticker, False)]:
        if not ticker:
            continue
        leg = df[df["symbol"] == ticker].copy()
        buys  = leg[leg["side"] == "BUY"].copy()
        sells = leg[leg["side"] == "SELL"].copy()

        if buys.empty or sells.empty:
            result[ticker] = {"status": "incomplete", "pnl": None}
            continue

        if is_long:
            # Long: bought first, sold to close
            open_px  = buys["fill_price"].mean()  if "fill_price"  in buys.columns  else buys["last_price"].mean()
            close_px = sells["fill_price"].mean() if "fill_price"  in sells.columns else sells["last_price"].mean()
            qty      = buys["qty"].sum()           if "qty"         in buys.columns  else 0.0
            pnl      = (close_px - open_px) * qty
        else:
            # Short: sold first (open short), then bought back (cover)
            open_px  = sells["fill_price"].mean() if "fill_price" in sells.columns else sells["last_price"].mean()
            close_px = buys["fill_price"].mean()  if "fill_price" in buys.columns  else buys["last_price"].mean()
            qty      = sells["qty"].sum()          if "qty"        in sells.columns else 0.0
            pnl      = (open_px - close_px) * qty

        result[ticker] = {"pnl": round(pnl, 2), "open_px": round(open_px, 4),
                          "close_px": round(close_px, 4), "qty": qty}

    if long_ticker in result and short_ticker in result:
        long_pnl  = result[long_ticker].get("pnl") or 0.0
        short_pnl = result[short_ticker].get("pnl") or 0.0
        result["pair_total_pnl"] = round(long_pnl + short_pnl, 2)
        result["pair"] = f"{long_ticker}/{short_ticker}"

    return result or None


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
        # Migrate old schema (pre-sprint-6): ts→timestamp, winner_agent→agent
        rename = {}
        if "ts" in df_old.columns and "timestamp" not in df_old.columns:
            rename["ts"] = "timestamp"
        if "winner_agent" in df_old.columns and "agent" not in df_old.columns:
            rename["winner_agent"] = "agent"
        if rename:
            df_old = df_old.rename(columns=rename)
        if "is_winner" not in df_old.columns:
            df_old["is_winner"] = True
        # Keep only current schema, drop legacy columns (meta, etc.)
        df_old = df_old.reindex(columns=cols)

        # Garde-fou anti-décalage de colonnes.
        # Trois lignes d'un schéma pré-migration à 7 colonnes ont été relues dans
        # un cadre à 10 : les valeurs se sont décalées de deux crans (symbol="bull",
        # target_weight="Buffett screen passed…"), rendant `decisions.csv` non
        # typable et polluant silencieusement edge_audit, live_scorer, monte_carlo
        # et factor_analysis. Elles ont été mises en quarantaine le 2026-08-02.
        # On refuse désormais de réécrire des lignes dont les colonnes numériques
        # ne le sont pas : mieux vaut perdre l'historique douteux que le propager.
        _numeric = ["confidence", "target_weight"]
        _corrupt = pd.Series(False, index=df_old.index)
        for _c in _numeric:
            _corrupt |= pd.to_numeric(df_old[_c], errors="coerce").isna() & df_old[_c].notna()
        if _corrupt.any():
            Path("logs").mkdir(parents=True, exist_ok=True)
            df_old[_corrupt].to_csv(
                "logs/decisions_quarantine.csv",
                mode="a",
                header=not Path("logs/decisions_quarantine.csv").exists(),
                index=False,
            )
            print(
                f"⚠️  log_decisions: {int(_corrupt.sum())} ligne(s) historique(s) "
                f"malformée(s) écartée(s) → logs/decisions_quarantine.csv"
            )
            df_old = df_old[~_corrupt]

        df_all = pd.concat([df_old, df_new], ignore_index=True)
    except FileNotFoundError:
        df_all = df_new

    df_all.to_csv(out_path, index=False)