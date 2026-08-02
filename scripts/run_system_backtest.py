#!/usr/bin/env python
"""
Backtest du système réel — script de reproduction.

    python -m scripts.run_system_backtest            # utilise le cache local
    python -m scripts.run_system_backtest --refresh  # retélécharge les données

Produit :
    logs/system_backtest_equity.csv     courbe d'equity + benchmark
    logs/system_backtest_summary.json   métriques et réserves
    logs/backtest_cache/*.parquet       données figées (reproductibilité)

Contrairement à `src/backtest/portfolio_backtest.py`, ce script rejoue le
pipeline de décision complet — arène, sélecteur, risk manager, garde
d'exécution — et non un agent isolé sur un symbole isolé.
"""
from __future__ import annotations

import argparse
import json
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
warnings.filterwarnings("ignore")

from src.backtest.system_backtest import SystemBacktestConfig, run_system_backtest  # noqa: E402
from src.config import WATCHLIST  # noqa: E402
from src.risk.manager import RiskConfig  # noqa: E402

CTA_EXTRA = ["TLT", "UUP", "DBC"]
CACHE = Path("logs/backtest_cache")


def load_data(refresh: bool) -> dict:
    syms = list(WATCHLIST) + CTA_EXTRA
    if not refresh and CACHE.exists() and len(list(CACHE.glob("*.parquet"))) >= len(syms):
        return {p.stem: pd.read_parquet(p) for p in CACHE.glob("*.parquet")}

    import yfinance as yf
    from src.data.market_data import normalize_ohlcv

    print("⬇️  Téléchargement 5 ans…")
    raw = yf.download(syms, period="5y", interval="1d", auto_adjust=True,
                      progress=False, group_by="ticker")
    data = {s: normalize_ohlcv(raw[s].copy()) for s in syms}
    CACHE.mkdir(parents=True, exist_ok=True)
    for s, df in data.items():
        df.to_parquet(CACHE / f"{s}.parquet")
    return data


def beta_matched_excess(result) -> tuple[float, float, float]:
    """
    Excès de rendement contre un portefeuille passif de MÊME bêta
    (β × SPY + (1−β) en cash à 4 %).

    C'est la seule comparaison qui isole l'apport du système : un portefeuille
    à moitié investi bat mécaniquement SPY en drawdown sans contenir la moindre
    information. Comparer à SPY 100 % flatte le drawdown et punit le rendement ;
    cette référence-ci ne fait ni l'un ni l'autre.
    """
    rp = result.equity.pct_change().dropna()
    rb = result.benchmark.pct_change().dropna()
    beta = float(np.polyfit(rb, rp, 1)[0])
    naive_r = beta * rb + (1 - beta) * (0.04 / 252)
    excess = float(result.equity.iloc[-1] / result.equity.iloc[0] - (1 + naive_r).prod())
    d = rp - naive_r
    ir = float(d.mean() / d.std() * np.sqrt(252)) if d.std() > 0 else 0.0
    return beta, excess, ir


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--refresh", action="store_true", help="retélécharge les données")
    ap.add_argument("--net-long", type=float, default=0.60)
    ap.add_argument("--cash-floor", type=float, default=0.30)
    ap.add_argument("--tranche", type=float, default=0.05, help="MAX_NOTIONAL_PCT")
    args = ap.parse_args()

    data = load_data(args.refresh)

    qv_path = Path("logs/qualified_voters.json")
    qv = set(json.loads(qv_path.read_text())["qualified"]) if qv_path.exists() else None

    cfg = SystemBacktestConfig(
        qualified_voters=qv,
        max_notional_pct=args.tranche,
        risk=RiskConfig(
            max_net_long_pct=args.net_long,
            max_single_position_pct=0.20,
            min_cash_pct=args.cash_floor,
        ),
    )
    res = run_system_backtest(data, symbols=WATCHLIST, cta_symbols=CTA_EXTRA, cfg=cfg)

    print("\n" + res.render())

    beta, excess, ir = beta_matched_excess(res)
    print(f"\nBêta vs SPY                        : {beta:.2f}")
    print(f"Excès vs portefeuille de même bêta : {excess:+.1%}  (IR {ir:.2f})")

    Path("logs").mkdir(exist_ok=True)
    pd.DataFrame({"equity": res.equity, "benchmark": res.benchmark}).to_csv(
        "logs/system_backtest_equity.csv"
    )
    Path("logs/system_backtest_summary.json").write_text(json.dumps({
        "start": str(res.equity.index[0].date()),
        "end": str(res.equity.index[-1].date()),
        "config": {
            "max_net_long_pct": args.net_long,
            "min_cash_pct": args.cash_floor,
            "max_notional_pct": args.tranche,
        },
        "total_return": res.total_return,
        "cagr": res.cagr,
        "sharpe": res.sharpe,
        "max_drawdown": res.max_drawdown,
        "benchmark_return": res.benchmark_return,
        "benchmark_cagr": res.benchmark_cagr,
        "benchmark_sharpe": res.benchmark_sharpe,
        "benchmark_max_drawdown": res.benchmark_max_drawdown,
        "alpha_vs_spy": res.alpha,
        "beta": beta,
        "excess_vs_beta_matched": excess,
        "information_ratio": ir,
        "orders_sent": res.n_orders_sent,
        "orders_filled": res.n_orders_filled,
        "fill_rate": res.fill_rate,
        "transaction_costs_usd": res.total_costs,
        "caveats": res.caveats,
    }, indent=2, ensure_ascii=False))
    print("\n✅ logs/system_backtest_equity.csv + logs/system_backtest_summary.json")


if __name__ == "__main__":
    main()
