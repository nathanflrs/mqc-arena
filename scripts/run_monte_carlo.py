#!/usr/bin/env python3
"""
Milan Capital — Monte Carlo Simulation (standalone script)

Usage:
    python scripts/run_monte_carlo.py
    python scripts/run_monte_carlo.py --n 10000 --horizon 90 --regime bull_volatile
    python scripts/run_monte_carlo.py --n 5000 --horizon 30 --notify
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

# Ensure project root is on the path
sys.path.insert(0, str(Path(__file__).parent.parent))

from dotenv import load_dotenv

load_dotenv()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Milan Capital — Monte Carlo Simulation",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--n", type=int, default=10_000, help="Number of simulations")
    parser.add_argument("--horizon", type=int, default=90, help="Horizon in trading days")
    parser.add_argument(
        "--regime",
        default="bull_volatile",
        choices=["bull_quiet", "bull_volatile", "sideways", "bear"],
        help="GMM regime for return conditioning",
    )
    parser.add_argument("--nav", type=float, default=1_000_000.0, help="Initial NAV ($)")
    parser.add_argument("--notify", action="store_true", help="Send result to Telegram")
    parser.add_argument(
        "--decisions", default="logs/decisions.csv", help="Path to decisions.csv"
    )
    parser.add_argument(
        "--executions", default="logs/executions.csv", help="Path to executions.csv"
    )
    parser.add_argument(
        "--walkforward", default="logs/walkforward_results.csv", help="Path to walkforward_results.csv"
    )
    parser.add_argument(
        "--output", default="logs/monte_carlo_latest.json", help="Output JSON path"
    )
    args = parser.parse_args()

    print(f"🎲 Milan Capital — Monte Carlo Simulation")
    print(f"   Régime : {args.regime} | N={args.n:,} | Horizon : {args.horizon}j")
    print(f"   NAV initiale : ${args.nav:,.0f}")
    print()

    from src.analytics.monte_carlo import (
        ReturnBootstrapper,
        MonteCarloEngine,
        MonteCarloReporter,
    )

    # Load returns
    print("📂 Chargement des rendements empiriques...")
    bootstrapper = ReturnBootstrapper(args.decisions, args.executions, args.walkforward)
    returns = bootstrapper.get_regime_conditioned_returns(args.regime)
    n_samples = len(returns)
    print(f"   {n_samples} échantillons de rendements chargés")
    if n_samples < 30:
        print("   ⚠️  Moins de 30 round-trips — utilisation des rendements walk-forward comme proxy")

    # Run simulation
    print(f"\n⚡ Simulation en cours ({args.n:,} × {args.horizon}j)...")
    t0 = time.perf_counter()
    engine = MonteCarloEngine(
        n_simulations=args.n,
        horizon_days=args.horizon,
        initial_nav=args.nav,
        gmm_regime=args.regime,
    )
    result = engine.run(returns)
    elapsed = time.perf_counter() - t0
    print(f"   ✅ Terminé en {elapsed:.2f}s")

    # Save JSON
    reporter = MonteCarloReporter()
    reporter.save_json(result, args.output)
    print(f"\n💾 Résultats sauvegardés dans {args.output}")

    # Print full report
    print()
    print(reporter.format_telegram(result))

    # Telegram notification
    if args.notify:
        try:
            from src.notify.telegram import send_message
            send_message(reporter.format_telegram(result))
            print("\n📲 Rapport envoyé sur Telegram.")
        except Exception as exc:
            print(f"\n⚠️  Envoi Telegram échoué : {exc}")


if __name__ == "__main__":
    main()
