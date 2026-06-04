# src/backtest/run_backtest.py
from __future__ import annotations

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

import pandas as pd
from src.backtest.engine import BacktestEngine
from src.agents.buffett import BuffettAgent
from src.agents.citadel import CitadelAgent
from src.agents.mean_reversion import MeanReversionAgent
from src.agents.trend_following import TrendFollowingAgent
from src.data.market_data import download_ohlcv
from src.data.regime import detect_regime

# ====== CONFIG ======
SYMBOLS = [
    "AAPL", "SPY", "QQQ", "NVDA", "MSFT",
    "GOOGL", "META", "JPM", "GS", "GLD",
    "TLT", "BRK-B", "JNJ", "TSLA", "AMD",
]
INITIAL_CAPITAL = 100_000

AGENTS = [
    BuffettAgent(),
    CitadelAgent(),
    MeanReversionAgent(),
    TrendFollowingAgent(),
]


def print_result(agent_name, symbol, r):
    print(f"\n{'='*55}")
    print(f"  {agent_name} | {symbol}")
    print(f"{'='*55}")
    print(f"  Total Return     : {r.total_return:+.2%}")
    print(f"  Annualized       : {r.annualized_return:+.2%}")
    print(f"  Sharpe Ratio     : {r.sharpe_ratio:.2f}")
    print(f"  Max Drawdown     : {r.max_drawdown:.2%}")
    print(f"  Win Rate         : {r.win_rate:.2%}")
    print(f"  Nb Trades        : {r.n_trades}")
    print(f"{'='*55}")


def main():
    print("\n🚀 Milan Capital — Backtest Engine")
    print(f"Capital initial : ${INITIAL_CAPITAL:,}")
    print(f"Symboles        : {SYMBOLS}")
    print(f"Agents          : {[a.name for a in AGENTS]}")
    print(f"Période         : 3 ans\n")

    all_results = []

    for sym in SYMBOLS:
        print(f"\n📊 Téléchargement données {sym}...")
        df = download_ohlcv(sym, period="3y")
        print(f"   {len(df)} jours de données")

        for agent in AGENTS:
            engine = BacktestEngine(
                agent=agent,
                initial_capital=INITIAL_CAPITAL,
            )
            r = engine.run(symbol=sym, df=df)
            print_result(agent.name, sym, r)

            all_results.append({
                "agent": agent.name,
                "symbol": sym,
                "total_return": r.total_return,
                "annualized_return": r.annualized_return,
                "sharpe_ratio": r.sharpe_ratio,
                "max_drawdown": r.max_drawdown,
                "win_rate": r.win_rate,
                "n_trades": r.n_trades,
            })

    # ====== SUMMARY TABLE ======
    print("\n\n📋 RÉSUMÉ GLOBAL")
    print("="*75)
    df_results = pd.DataFrame(all_results)
    df_results = df_results.sort_values("sharpe_ratio", ascending=False)

    print(df_results.to_string(index=False))

    # Sauvegarde
    import pathlib
    pathlib.Path("logs").mkdir(exist_ok=True)
    df_results.to_csv("logs/backtest_results.csv", index=False)
    print("\n✅ Résultats sauvegardés dans logs/backtest_results.csv")

    # ====== BEST AGENT PAR SYMBOLE ======
    print("\n\n🏆 MEILLEUR AGENT PAR SYMBOLE (Sharpe)")
    print("="*55)
    for sym in SYMBOLS:
        sub = df_results[df_results["symbol"] == sym]
        if not sub.empty:
            best = sub.iloc[0]
            print(f"  {sym} → {best['agent']} | Sharpe={best['sharpe_ratio']:.2f} | Return={best['total_return']:+.2%}")


if __name__ == "__main__":
    main()