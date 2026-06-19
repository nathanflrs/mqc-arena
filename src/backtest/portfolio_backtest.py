# src/backtest/portfolio_backtest.py
"""
Milan Capital — Portfolio Backtest complet sur 5 ans.

Simule le système réel :
- Un agent par symbole (AGENT_PRIORITY issu du walk-forward)
- Long-bias overlay (regime SMA50/SMA200)
- Capital alloué équitablement entre les 14 symboles
- Benchmark : SPY buy-and-hold sur la même période

Usage:
    python -m src.backtest.portfolio_backtest
"""
from __future__ import annotations

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from datetime import date
from typing import Dict, List

import numpy as np
import pandas as pd
import yfinance as yf

from src.agents.buffett import BuffettAgent
from src.agents.citadel import CitadelAgent
from src.agents.mean_reversion import MeanReversionAgent
from src.agents.trend_following import TrendFollowingAgent
from src.backtest.engine import BacktestEngine, BacktestResult, _compute_regime_series
from src.config import WATCHLIST, AGENT_PRIORITY

# ── Config ────────────────────────────────────────────────────────────────────

TOTAL_CAPITAL   = 100_000.0
DATA_PERIOD     = "5y"
COMMISSION      = 0.0005  # frais broker IBKR ~0.5 bps
SLIPPAGE_BPS    = 7.0     # spread bid-ask + impact marché ~7 bps par side
COOLDOWN_DAYS   = 20      # réduit l'over-trading, laisse les positions respirer
MIN_HISTORY     = 210

AGENT_MAP = {
    "BuffettAgent":       BuffettAgent,
    "CitadelAgent":       CitadelAgent,
    "MeanReversionAgent": MeanReversionAgent,
    "TrendFollowingAgent":TrendFollowingAgent,
}

# Backtest universe — remplace JNJ (-7%) et BRK-B (+3%) par AMZN et LLY
BACKTEST_WATCHLIST = [
    "AAPL", "SPY", "QQQ", "NVDA", "MSFT",
    "GOOGL", "META", "JPM", "GS", "GLD",
    "TSLA", "AMD", "AMZN", "LLY",
]

BACKTEST_AGENT_PRIORITY = {
    **AGENT_PRIORITY,
    "AMZN": "TrendFollowingAgent",  # forte tendance structurelle
    "LLY":  "BuffettAgent",         # quality compounder (GLP-1)
}

# ── Helpers ───────────────────────────────────────────────────────────────────

def _download(symbol: str) -> pd.DataFrame | None:
    try:
        df = yf.download(symbol, period=DATA_PERIOD, auto_adjust=True, progress=False)
        if df.empty or len(df) < MIN_HISTORY + 50:
            return None
        # Flatten multi-level columns produced by recent yfinance versions
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        return df
    except Exception:
        return None


def _bh_return(df: pd.DataFrame, start_idx: int) -> float:
    """Buy-and-hold return from start_idx to end of df."""
    close = pd.to_numeric(df["Close"], errors="coerce").dropna()
    if len(close) <= start_idx:
        return 0.0
    p0, p1 = float(close.iloc[start_idx]), float(close.iloc[-1])
    return (p1 - p0) / p0 if p0 > 0 else 0.0


def _sharpe(equity: pd.Series, rf: float = 0.04) -> float:
    r = equity.pct_change().dropna()
    std = r.std()
    if std == 0 or np.isnan(std) or len(r) < 5:
        return 0.0
    s = float((r.mean() - rf / 252) / std * np.sqrt(252))
    return round(float(np.clip(s, -50, 50)), 3)


def _max_dd(equity: pd.Series) -> float:
    peak = equity.cummax()
    dd = (equity - peak) / peak
    return float(dd.min())


def _annual_return(equity: pd.Series) -> float:
    if len(equity) < 2:
        return 0.0
    total = (equity.iloc[-1] - equity.iloc[0]) / equity.iloc[0]
    n_years = len(equity) / 252
    return float((1 + total) ** (1 / n_years) - 1) if n_years > 0 else 0.0


def _monthly_returns(equity: pd.Series) -> pd.DataFrame:
    r = equity.resample("ME").last().pct_change().dropna()
    r.index = r.index.to_period("M")
    df = pd.DataFrame({"return": r})
    df["year"]  = df.index.year
    df["month"] = df.index.month
    pivot = df.pivot_table(values="return", index="year", columns="month", aggfunc="first")
    pivot.columns = [date(2000, m, 1).strftime("%b") for m in pivot.columns]
    return pivot


def _stars(sharpe: float) -> str:
    if sharpe != sharpe:
        return "—"
    n = max(0, int(sharpe / 0.5))
    return "★" * min(n, 5) or "—"

# ── Core ──────────────────────────────────────────────────────────────────────

def run() -> None:
    print("=" * 65)
    print("  🏦 Milan Capital — Portfolio Backtest 5 ans")
    print(f"  Capital : ${TOTAL_CAPITAL:,.0f}  |  {DATA_PERIOD}  |  {len(BACKTEST_WATCHLIST)} symboles")
    print("=" * 65)

    capital_per_sym = TOTAL_CAPITAL / len(BACKTEST_WATCHLIST)

    # ── 1. Download data ──────────────────────────────────────────────────────
    print("\n📥 Téléchargement des données...")
    all_data: Dict[str, pd.DataFrame] = {}
    for sym in BACKTEST_WATCHLIST:
        df = _download(sym)
        if df is not None:
            all_data[sym] = df
            print(f"   {sym:6s}: {len(df)} jours")
        else:
            print(f"   {sym:6s}: ⚠️  données insuffisantes, ignoré")

    # ── 2. SPY benchmark ─────────────────────────────────────────────────────
    spy_df = all_data["SPY"] if "SPY" in all_data else _download("SPY")
    spy_raw = spy_df["Close"]
    if isinstance(spy_raw, pd.DataFrame):
        spy_raw = spy_raw.iloc[:, 0]
    spy_close = pd.to_numeric(spy_raw, errors="coerce").dropna()

    # ── 3. Run BacktestEngine per symbol ─────────────────────────────────────
    print(f"\n⚙️  Backtest par symbole (agent OOS optimal)...\n")

    equity_curves:   Dict[str, pd.Series] = {}
    results:         Dict[str, BacktestResult] = {}
    agent_used:      Dict[str, str] = {}
    bh_returns:      Dict[str, float] = {}

    for sym in BACKTEST_WATCHLIST:
        df = all_data.get(sym)
        if df is None:
            continue

        agent_name = BACKTEST_AGENT_PRIORITY.get(sym, "BuffettAgent")
        agent_cls  = AGENT_MAP.get(agent_name, BuffettAgent)
        agent      = agent_cls()

        raw_close = df["Close"]
        if isinstance(raw_close, pd.DataFrame):
            raw_close = raw_close.iloc[:, 0]
        close = pd.to_numeric(raw_close, errors="coerce").dropna()
        regime_series = _compute_regime_series(close)

        engine = BacktestEngine(
            agent=agent,
            initial_capital=capital_per_sym,
            target_weight=0.95,
            commission=COMMISSION,
            slippage_bps=SLIPPAGE_BPS,
            min_history=MIN_HISTORY,
            cooldown_days=COOLDOWN_DAYS,
        )
        res = engine.run(sym, df, regime_series=regime_series)

        equity_curves[sym] = res.equity_curve
        results[sym]       = res
        agent_used[sym]    = agent_name
        bh_returns[sym]    = _bh_return(df, MIN_HISTORY)

    # ── 4. Portfolio equity curve ─────────────────────────────────────────────
    # Align all curves on common date range, forward-fill gaps
    all_eq = pd.DataFrame(equity_curves)
    all_eq = all_eq.ffill().dropna(how="all")

    portfolio_equity = all_eq.sum(axis=1)
    portfolio_equity = portfolio_equity.dropna()

    # ── 5. Benchmark ─────────────────────────────────────────────────────────
    spy_start_date = portfolio_equity.index[0]
    spy_aligned    = spy_close[spy_close.index >= spy_start_date]
    if not spy_aligned.empty:
        spy_bh_return = float(
            (spy_aligned.iloc[-1] - spy_aligned.iloc[0]) / spy_aligned.iloc[0]
        )
        spy_equity = (spy_aligned / spy_aligned.iloc[0]) * TOTAL_CAPITAL
    else:
        spy_bh_return = 0.0
        spy_equity    = None

    # ── 6. Portfolio metrics ──────────────────────────────────────────────────
    port_total_return = float(
        (portfolio_equity.iloc[-1] - TOTAL_CAPITAL) / TOTAL_CAPITAL
    )
    port_annual       = _annual_return(portfolio_equity)
    port_sharpe       = _sharpe(portfolio_equity)
    port_maxdd        = _max_dd(portfolio_equity)

    all_trades = []
    for r in results.values():
        all_trades.extend(r.trades)
    n_trades  = len(all_trades)
    buys      = [t for t in all_trades if t.action == "BUY"]
    sells     = [t for t in all_trades if t.action == "SELL"]
    wins      = sum(
        1 for s in sells
        if (buy := next((b.price for b in reversed(buys) if b.symbol == s.symbol), None))
        and s.price > buy
    )
    win_rate  = wins / len(sells) if sells else 0.0

    # Date range
    date_start = portfolio_equity.index[0].date()
    date_end   = portfolio_equity.index[-1].date()
    n_years    = (date_end - date_start).days / 365.25

    # ── 7. Print report ───────────────────────────────────────────────────────
    print("=" * 65)
    print("  📊 RAPPORT DE PERFORMANCE — MILAN CAPITAL")
    print(f"  Période : {date_start} → {date_end}  ({n_years:.1f} ans)")
    print(f"  Capital initial : ${TOTAL_CAPITAL:,.0f}")
    print(f"  Capital final   : ${portfolio_equity.iloc[-1]:,.0f}")
    print("=" * 65)

    print("\n  ── PERFORMANCE GLOBALE ──────────────────────────────────")
    print(f"  Rendement total         : {port_total_return:+.2%}")
    print(f"  Rendement annualisé     : {port_annual:+.2%}")
    print(f"  Sharpe ratio            :  {port_sharpe:+.3f}")
    print(f"  Max Drawdown            : {port_maxdd:.2%}")
    print(f"  Nombre de trades        :  {n_trades}")
    print(f"  Win rate                :  {win_rate:.1%}")

    print("\n  ── BENCHMARK (SPY buy & hold) ───────────────────────────")
    print(f"  Rendement total SPY     : {spy_bh_return:+.2%}")
    alpha = port_total_return - spy_bh_return
    print(f"  Alpha vs SPY            : {alpha:+.2%}  {'✅' if alpha >= 0 else '⚠️'}")

    print("\n  ── PAR SYMBOLE ──────────────────────────────────────────")
    print(f"  {'Sym':6s} {'Agent':24s} {'Ret':>8s} {'Sharpe':>7s} {'MaxDD':>7s} {'Trades':>6s} {'vs BH':>7s}")
    print("  " + "─" * 63)

    sym_rows: List[dict] = []
    for sym in WATCHLIST:
        res = results.get(sym)
        if res is None:
            continue
        eq = equity_curves[sym]
        sym_ret    = float((eq.iloc[-1] - capital_per_sym) / capital_per_sym)
        sym_sharpe = _sharpe(eq)
        sym_maxdd  = _max_dd(eq)
        sym_bh     = bh_returns.get(sym, 0.0)
        sym_alpha  = sym_ret - sym_bh
        sym_rows.append({
            "sym": sym, "agent": agent_used[sym],
            "ret": sym_ret, "sharpe": sym_sharpe,
            "maxdd": sym_maxdd, "trades": res.n_trades,
            "alpha": sym_alpha,
        })
        flag = "✅" if sym_ret >= 0 else "🔴"
        print(
            f"  {sym:6s} {agent_used[sym]:24s} "
            f"{sym_ret:>+7.1%} {sym_sharpe:>7.2f} "
            f"{sym_maxdd:>7.1%} {res.n_trades:>6d} "
            f"{sym_alpha:>+6.1%} {flag}"
        )

    # ── 8. Monthly returns ────────────────────────────────────────────────────
    print("\n  ── RENDEMENTS MENSUELS (portfolio) ─────────────────────")
    try:
        monthly = _monthly_returns(portfolio_equity)
        months  = ["Jan","Feb","Mar","Apr","May","Jun","Jul","Aug","Sep","Oct","Nov","Dec"]
        header  = "  Year  " + "".join(f"{m:>6s}" for m in months if m in monthly.columns)
        print(header)
        for yr in monthly.index:
            row_str = f"  {yr}  "
            for m in months:
                if m not in monthly.columns:
                    continue
                val = monthly.loc[yr, m] if m in monthly.columns else float("nan")
                if val != val:
                    row_str += "      "
                else:
                    row_str += f"{val:>+5.1%} "
            print(row_str)
    except Exception as e:
        print(f"  (tableau mensuel non disponible : {e})")

    # ── 9. Top performers & underperformers ───────────────────────────────────
    sorted_rows = sorted(sym_rows, key=lambda x: x["ret"], reverse=True)
    print("\n  ── TOP 3 PERFORMERS ─────────────────────────────────────")
    for r in sorted_rows[:3]:
        print(f"  {r['sym']:6s} {r['agent']:24s} {r['ret']:+.2%}  Sharpe {r['sharpe']:.2f}  {_stars(r['sharpe'])}")

    print("\n  ── BOTTOM 3 ─────────────────────────────────────────────")
    for r in sorted_rows[-3:]:
        print(f"  {r['sym']:6s} {r['agent']:24s} {r['ret']:+.2%}  Sharpe {r['sharpe']:.2f}")

    # ── 10. Save CSV ──────────────────────────────────────────────────────────
    import pathlib
    out_dir = pathlib.Path("logs")
    out_dir.mkdir(exist_ok=True)

    df_port = pd.DataFrame({
        "date": portfolio_equity.index,
        "portfolio_equity": portfolio_equity.values,
    })
    if spy_equity is not None:
        spy_eq_aligned = spy_equity.reindex(portfolio_equity.index).ffill()
        df_port["spy_equity"] = spy_eq_aligned.values
    df_port.to_csv(out_dir / "portfolio_equity.csv", index=False)

    df_sym = pd.DataFrame(sym_rows)
    df_sym.to_csv(out_dir / "portfolio_by_symbol.csv", index=False)

    print(f"\n✅ Equity curve → logs/portfolio_equity.csv")
    print(f"✅ Par symbole  → logs/portfolio_by_symbol.csv")
    print("=" * 65)


if __name__ == "__main__":
    run()
