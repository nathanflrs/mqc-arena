# src/backtest/run_walkforward.py
"""
Walk-forward validation on all agents × all symbols.

Usage:
    python -m src.backtest.run_walkforward

Outputs:
    logs/walkforward_results.csv   — per-window OOS/IS metrics
    logs/walkforward_summary.csv   — best agent per symbol (OOS Sharpe)
    src/config.py                  — AGENT_PRIORITY auto-patched with OOS results
"""
from __future__ import annotations

import pathlib
import re
import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

import pandas as pd

from src.agents.buffett import BuffettAgent
from src.agents.citadel import CitadelAgent
from src.agents.mean_reversion import MeanReversionAgent
from src.agents.trend_following import TrendFollowingAgent
from src.backtest.engine import WalkForwardEngine, WalkForwardResult
from src.data.market_data import download_ohlcv

# ── Config ────────────────────────────────────────────────────────────────────

SYMBOLS = [
    "AAPL", "SPY", "QQQ", "NVDA", "MSFT",
    "GOOGL", "META", "JPM", "GS", "GLD",
    "TSLA", "AMD", "AMZN", "LLY",
]

AGENTS = [
    BuffettAgent(),
    CitadelAgent(),
    MeanReversionAgent(),
    TrendFollowingAgent(),
]

DATA_PERIOD = "5y"          # 5 ans de données → ~9 fenêtres walk-forward
INITIAL_CAPITAL = 100_000
LOG_DIR = pathlib.Path("logs")

# ── Helpers ───────────────────────────────────────────────────────────────────

def _stars(sharpe: float) -> str:
    """Visual indicator: ★ per 0.5 Sharpe unit."""
    n = max(0, int(sharpe / 0.5))
    return "★" * min(n, 5) or "—"


def _patch_agent_priority(new_priority: dict[str, str]) -> None:
    """
    Rewrite AGENT_PRIORITY in src/config.py with OOS-validated values.
    Keeps all comments and surrounding code intact.
    """
    config_path = pathlib.Path("src/config.py")
    source = config_path.read_text()

    # Build new dict literal
    lines = ["AGENT_PRIORITY = {"]
    for sym, agent in new_priority.items():
        lines.append(f'    "{sym}": "{agent}",')
    lines.append("}")
    new_block = "\n".join(lines)

    # Replace the existing block (from AGENT_PRIORITY = { ... })
    pattern = r"AGENT_PRIORITY\s*=\s*\{[^}]*\}"
    if re.search(pattern, source, re.DOTALL):
        patched = re.sub(pattern, new_block, source, flags=re.DOTALL)
        config_path.write_text(patched)
        print(f"\n✅ src/config.py — AGENT_PRIORITY mis à jour avec les Sharpe OOS")
    else:
        print("\n⚠️  Impossible de patcher AGENT_PRIORITY automatiquement.")
        print("   Collez manuellement dans src/config.py :")
        print(new_block)


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    print("🚀 Milan Capital — Walk-Forward Validation")
    print(f"   Période  : {DATA_PERIOD}")
    print(f"   Fenêtres : Train 18m / Test 6m / Step 3m")
    print(f"   Symboles : {len(SYMBOLS)} | Agents : {len(AGENTS)}\n")

    LOG_DIR.mkdir(exist_ok=True)

    # ── Download all data once ────────────────────────────────────────────────
    print("📥 Téléchargement des données...")
    all_data: dict[str, pd.DataFrame] = {}
    for sym in SYMBOLS:
        try:
            all_data[sym] = download_ohlcv(sym, period=DATA_PERIOD)
            print(f"   {sym}: {len(all_data[sym])} jours")
        except Exception as e:
            print(f"   ⚠️  {sym}: {e}")

    # ── Run walk-forward ──────────────────────────────────────────────────────
    all_results: list[WalkForwardResult] = []

    for agent in AGENTS:
        print(f"\n{'='*60}")
        print(f"  Agent : {agent.name}")
        print(f"{'='*60}")

        for sym in SYMBOLS:
            df = all_data.get(sym)
            if df is None or len(df) < WalkForwardEngine.TRAIN_BDAYS + WalkForwardEngine.TEST_BDAYS:
                print(f"  {sym}: données insuffisantes, ignoré")
                continue

            engine = WalkForwardEngine(agent=agent, initial_capital=INITIAL_CAPITAL)
            result = engine.run(symbol=sym, df=df)

            warn = " ⚠️  LOOKAHEAD?" if result.lookahead_warning else ""
            oos_s = result.avg_oos_sharpe
            is_s = result.avg_is_sharpe
            alpha = result.avg_alpha
            fmt_s = lambda v: f"{v:+.2f}" if not (v != v) else "  nan"
            fmt_p = lambda v: f"{v:+.2%}" if not (v != v) else "  nan"
            print(
                f"  {sym:6s} | {len(result.windows)} fenêtres "
                f"| Sharpe OOS={fmt_s(oos_s)} "
                f"| Sharpe IS={fmt_s(is_s)} "
                f"| Alpha={fmt_p(alpha)} "
                f"| {_stars(oos_s if not (oos_s != oos_s) else 0.0)}{warn}"
            )
            all_results.append(result)

    # ── Save detailed CSV ─────────────────────────────────────────────────────
    all_rows = []
    for r in all_results:
        all_rows.extend(r.to_csv_rows())

    df_all = pd.DataFrame(all_rows)

    # Sanitize overflow Sharpe values (flat equity / no trades → near-zero std)
    sharpe_cols = [c for c in ["oos_sharpe", "is_sharpe", "avg_oos_sharpe", "avg_is_sharpe"]
                   if c in df_all.columns]
    for col in sharpe_cols:
        df_all[col] = pd.to_numeric(df_all[col], errors="coerce")
        df_all.loc[df_all[col].abs() > 100, col] = float("nan")

    wf_path = LOG_DIR / "walkforward_results.csv"
    df_all.to_csv(wf_path, index=False)
    print(f"\n✅ Résultats détaillés → {wf_path} ({len(df_all)} lignes)")

    # ── Summary: best agent per symbol (avg OOS Sharpe) ──────────────────────
    summary_cols = ["agent", "symbol", "avg_oos_sharpe", "avg_is_sharpe",
                    "avg_alpha", "lookahead_warning"]
    # One row per (agent, symbol) → take first occurrence (all rows share the same agg values)
    df_summary = (
        df_all[summary_cols]
        .drop_duplicates(subset=["agent", "symbol"])
        .sort_values(["symbol", "avg_oos_sharpe"], ascending=[True, False])
    )
    summary_path = LOG_DIR / "walkforward_summary.csv"
    df_summary.to_csv(summary_path, index=False)

    # ── Print ranking per symbol ──────────────────────────────────────────────
    print("\n\n🏆 MEILLEUR AGENT PAR SYMBOLE — Sharpe OOS")
    print("=" * 65)

    from src.config import AGENT_PRIORITY as OLD_PRIORITY

    new_priority: dict[str, str] = {}
    changed: list[str] = []

    for sym in SYMBOLS:
        sub = df_summary[df_summary["symbol"] == sym]
        if sub.empty:
            new_priority[sym] = OLD_PRIORITY.get(sym, "BuffettAgent")
            continue
        best = sub.iloc[0]
        agent_name = best["agent"]
        oos = best["avg_oos_sharpe"]
        is_ = best["avg_is_sharpe"]
        warn = " ⚠️" if best["lookahead_warning"] else ""

        old_agent = OLD_PRIORITY.get(sym, "?")
        changed_flag = " ← CHANGÉ" if agent_name != old_agent else ""

        fmt = lambda v: f"{v:+.2f}" if pd.notna(v) else "  nan"
        print(
            f"  {sym:6s} → {agent_name:25s} "
            f"| OOS={fmt(oos)} IS={fmt(is_)} "
            f"{warn}{changed_flag}"
        )
        new_priority[sym] = agent_name
        if agent_name != old_agent:
            changed.append(f"{sym}: {old_agent} → {agent_name}")

    print("=" * 65)

    if changed:
        print(f"\n🔄 {len(changed)} changement(s) de priorité :")
        for c in changed:
            print(f"   {c}")
    else:
        print("\n✅ Aucun changement de priorité — OOS confirme les IS.")

    # ── Auto-patch config.py ──────────────────────────────────────────────────
    _patch_agent_priority(new_priority)

    print(f"\n✅ Summary → {summary_path}")
    print("🎯 Walk-forward terminé.\n")


if __name__ == "__main__":
    main()
