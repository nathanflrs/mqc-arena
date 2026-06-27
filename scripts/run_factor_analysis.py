#!/usr/bin/env python3
"""
Milan Capital — Factor Analysis Runner

Usage:
    python scripts/run_factor_analysis.py
    python scripts/run_factor_analysis.py --model ff5 --source walkforward --notify
"""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

# Make src/ importable
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

logging.basicConfig(level=logging.INFO, format="%(levelname)s  %(message)s")
logger = logging.getLogger(__name__)


def main() -> None:
    parser = argparse.ArgumentParser(description="Carhart / FF5 factor analysis per agent")
    parser.add_argument("--model",  default="carhart", choices=["carhart", "ff5"],
                        help="Factor model (default: carhart)")
    parser.add_argument("--source", default="auto",
                        choices=["auto", "walkforward", "live"],
                        help="Return series source (default: auto)")
    parser.add_argument("--notify", action="store_true",
                        help="Send Telegram summary after analysis")
    parser.add_argument("--cache-dir",   default="logs/factor_cache")
    parser.add_argument("--output-json", default="logs/factor_analysis_latest.json")
    parser.add_argument("--output-csv",  default="logs/factor_analysis_latest.csv")
    args = parser.parse_args()

    from src.analytics.factor_analysis import (
        FactorDataLoader, AgentReturnSeriesBuilder,
        FactorRegression, FactorReporter,
    )

    # ── 1. Load factors ────────────────────────────────────────────────────────
    print(f"\n⬇  Téléchargement facteurs {args.model.upper()}…")
    loader  = FactorDataLoader(cache_dir=args.cache_dir)
    factors = loader.load_factors(model=args.model)
    print(f"   → {len(factors)} jours de facteurs ({factors.index[0].date()} → {factors.index[-1].date()})")

    # ── 2. Build return series ─────────────────────────────────────────────────
    print(f"\n📐 Construction des séries de rendements (source={args.source})…")
    builder = AgentReturnSeriesBuilder()
    series  = builder.build_all_agents(source=args.source)

    for agent, s in series.items():
        n_total  = len(s)
        n_active = (s != 0).sum()
        src_hint = "walkforward" if (s != 0).mean() > 0.4 else "live/sparse"
        print(f"   {agent:<28} {n_total:>5} jours  ({n_active} actifs)  [{src_hint}]")

    if not series:
        print("⚠  Aucune série de rendements disponible. Vérifier decisions.csv / walkforward_results.csv.")
        sys.exit(1)

    # ── 3. Run regressions ─────────────────────────────────────────────────────
    print(f"\n🔬 Régression OLS Newey-West ({args.model.upper()})…")
    reg     = FactorRegression(model=args.model)
    results = reg.run_all(series, factors)

    # ── 4. Display table ───────────────────────────────────────────────────────
    reporter = FactorReporter()
    print("\n" + reporter.format_console_table(results))

    # ── 5. Save outputs ────────────────────────────────────────────────────────
    reporter.save_json(results, args.output_json)
    reporter.save_csv(results,  args.output_csv)
    print(f"\n💾 Résultats sauvegardés → {args.output_json}")

    # ── 6. Telegram notification ───────────────────────────────────────────────
    if args.notify:
        try:
            from src.notify.telegram import send_message
            msg = reporter.format_telegram(results)
            send_message(msg[:4096])
            print("📱 Telegram envoyé.")
        except Exception as exc:
            logger.warning("Telegram failed: %s", exc)

    # ── 7. Classification summary ──────────────────────────────────────────────
    true_alpha = [r for r in results.values() if r.alpha_significant and r.alpha_annualized > 0 and not r.insufficient_data]
    neg_alpha  = [r for r in results.values() if r.alpha_significant and r.alpha_annualized < 0 and not r.insufficient_data]
    beta_only  = [r for r in results.values() if not r.alpha_significant and not r.insufficient_data]
    insuf      = [r for r in results.values() if r.insufficient_data]

    print(f"\n{'═'*60}")
    print(f"  ✅ Vrai alpha      : {len(true_alpha)} agents")
    print(f"  ⚠️  Beta déguisé   : {len(beta_only)} agents")
    print(f"  ❌ Alpha négatif   : {len(neg_alpha)} agents")
    print(f"  ⏳ Données insuf.  : {len(insuf)} agents")
    print(f"{'═'*60}\n")


if __name__ == "__main__":
    main()
