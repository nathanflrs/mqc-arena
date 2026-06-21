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

import numpy as np
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


# ── P&L helper ────────────────────────────────────────────────────────────────

def _print_pnl_section(
    df_all: "pd.DataFrame",
    priority: dict[str, str],
    symbols: list[str],
    total_capital: float,
) -> None:
    """
    Print a concrete P&L summary for the AGENT_PRIORITY portfolio.

    Methodology
    -----------
    Each OOS window covers TEST_BDAYS ≈ 126 bdays (6 months).
    avg_oos_return_6m = arithmetic mean of per-window OOS returns.
    annualized        = (1 + avg_oos_return_6m)^2 − 1
    oos_years         = calendar span from first test_start to last test_end
    total_return      = (1 + avg_oos_return_6m)^(oos_years × 2) − 1
    pnl_$             = (total_capital / n_symbols) × total_return

    Windows overlap by ~50 % (step = 3m, test = 6m), so the compound is an
    approximation of holding the strategy continuously over the OOS span.
    """
    n_sym = len(symbols)
    cap_per_sym = total_capital / n_sym

    # ── Aggregate per (agent, symbol) ────────────────────────────────────────
    agg = (
        df_all
        .groupby(["agent", "symbol"])
        .agg(
            avg_6m_return=("oos_return", "mean"),
            avg_6m_bench=("benchmark_return", "mean"),
            n_windows=("oos_return", "count"),
            oos_start=("test_start", "min"),
            oos_end=("test_end", "max"),
        )
        .reset_index()
    )

    def _enrich(row):
        oos_years = (
            pd.Timestamp(row["oos_end"]) - pd.Timestamp(row["oos_start"])
        ).days / 365.25
        n_semi = max(oos_years * 2, 1.0)
        total_ret  = (1 + row["avg_6m_return"]) ** n_semi - 1
        total_bench = (1 + row["avg_6m_bench"]) ** n_semi - 1
        ann = (1 + row["avg_6m_return"]) ** 2 - 1
        return pd.Series({
            "oos_years":   round(oos_years, 1),
            "annualized":  ann,
            "total_return": total_ret,
            "total_bench": total_bench,
            "pnl":         cap_per_sym * total_ret,
            "bench_pnl":   cap_per_sym * total_bench,
        })

    enriched = pd.concat([agg, agg.apply(_enrich, axis=1)], axis=1)

    # ── Per-symbol table (AGENT_PRIORITY portfolio) ──────────────────────────
    W = 79
    print("\n\n" + "=" * W)
    print(
        f"  P&L OOS CONCRET — base {total_capital:,.0f} $"
        f"  ({n_sym} symboles = {cap_per_sym:,.0f} $/sym)"
    )
    print("=" * W)
    print(
        f"  {'Sym':<6}  {'Agent':<25}  {'Ann%':>7}  {'Total%':>7}  "
        f"{'Période':>6}  {'P&L $':>9}  {'vs SPY':>8}"
    )
    print("  " + "─" * (W - 2))

    portfolio_pnl    = 0.0
    portfolio_bench  = 0.0
    valid_symbols    = 0
    ann_returns      = []

    for sym in symbols:
        agent_name = priority.get(sym)
        if not agent_name:
            continue

        mask = (enriched["agent"] == agent_name) & (enriched["symbol"] == sym)
        if not mask.any():
            continue

        row = enriched[mask].iloc[0]
        ann    = row["annualized"]
        total  = row["total_return"]
        pnl    = row["pnl"]
        bench  = row["total_bench"]
        alpha  = total - bench
        yrs    = row["oos_years"]

        sign_ann   = "+" if ann   >= 0 else ""
        sign_total = "+" if total >= 0 else ""
        sign_alpha = "+" if alpha >= 0 else ""
        pnl_str    = f"{'+' if pnl >= 0 else ''}{pnl:,.0f}"

        print(
            f"  {sym:<6}  {agent_name:<25}  "
            f"{sign_ann}{ann*100:>5.1f}%  "
            f"{sign_total}{total*100:>5.1f}%  "
            f"{yrs:>5.1f}y  "
            f"{pnl_str:>9}$  "
            f"{sign_alpha}{alpha*100:>6.1f}%"
        )

        portfolio_pnl   += pnl
        portfolio_bench += row["bench_pnl"]
        ann_returns.append(ann)
        valid_symbols   += 1

    # ── Portfolio aggregate ───────────────────────────────────────────────────
    if valid_symbols:
        avg_ann = float(np.mean(ann_returns))
        avg_ann_bench = portfolio_bench / (cap_per_sym * valid_symbols)
        total_portfolio_ret = portfolio_pnl / total_capital
        total_bench_ret     = portfolio_bench / total_capital
        alpha_portfolio     = total_portfolio_ret - total_bench_ret

        print("  " + "─" * (W - 2))
        sign_p = "+" if portfolio_pnl  >= 0 else ""
        sign_r = "+" if total_portfolio_ret >= 0 else ""
        sign_a = "+" if alpha_portfolio >= 0 else ""
        print(
            f"  {'PORTFOLIO':32}  "
            f"{'+' if avg_ann >= 0 else ''}{avg_ann*100:>5.1f}%  "
            f"{sign_r}{total_portfolio_ret*100:>5.1f}%"
            f"{'':>8}  "
            f"{sign_p}{portfolio_pnl:>8,.0f}$  "
            f"{sign_a}{alpha_portfolio*100:>6.1f}%"
        )
        print(
            f"\n  Retour annualisé moyen (AGENT_PRIORITY) : "
            f"{'+' if avg_ann >= 0 else ''}{avg_ann*100:.1f}%"
        )
        print(
            f"  P&L total sur {total_capital:,.0f} $ : "
            f"{'+' if portfolio_pnl >= 0 else ''}{portfolio_pnl:,.0f} $"
        )
        print(
            f"  Alpha portefeuille vs SPY B&H : "
            f"{'+' if alpha_portfolio >= 0 else ''}{alpha_portfolio*100:.1f}%"
        )
    print("=" * W)
    print(
        "  Note: retours OOS par fenêtres 6m qui se chevauchent (step 3m)."
    )
    print(
        "  Compound = (1 + avg_6m)^(n_semi_annual)  — approximation valide"
    )
    print(
        "  pour évaluer l'ordre de grandeur du P&L, pas un backtest exact."
    )
    print("=" * W)


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

    # ── P&L concret : retours OOS en dollars ─────────────────────────────────
    _print_pnl_section(df_all, new_priority, SYMBOLS, INITIAL_CAPITAL)

    # ── Auto-patch config.py ──────────────────────────────────────────────────
    _patch_agent_priority(new_priority)

    print(f"\n✅ Summary → {summary_path}")
    print("🎯 Walk-forward terminé.\n")


if __name__ == "__main__":
    main()
