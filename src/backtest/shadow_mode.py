# src/backtest/shadow_mode.py
"""
EarningsSentimentAgent shadow mode — runs daily, logs signals without executing.

On each run:
  1. Load logs/shadow_signals.csv
  2. Fill in actual price outcomes (1d / 3d / 5d) for past signals
  3. Run EarningsSentimentAgent for every WATCHLIST symbol
  4. Append today's signals and save

After ~2 weeks the CSV contains enough data to compute signal accuracy
and decide whether the agent generates alpha vs buy-and-hold.

Usage:
    python -m src.backtest.shadow_mode
    python -m src.backtest.shadow_mode --report   # accuracy only, no new signals
"""
from __future__ import annotations

import sys
from datetime import date, timedelta
from pathlib import Path

import pandas as pd
import yfinance as yf

from src.agents.base import MarketState
from src.agents.earnings_sentiment import EarningsSentimentAgent
from src.config import WATCHLIST
from src.data.market_data import download_ohlcv

LOG_PATH = Path("logs/shadow_signals.csv")

HORIZONS = {"1d": 1, "3d": 3, "5d": 5}

COLUMNS = [
    "date", "symbol", "action", "confidence",
    "reason", "key_catalyst", "sentiment_score",
    "price_at_signal",
    "price_1d", "price_3d", "price_5d",
    "return_1d", "return_3d", "return_5d",
]


# ── Helpers ───────────────────────────────────────────────────────────────────

def _bdays_after(from_date: date, n: int) -> date:
    d = from_date
    counted = 0
    while counted < n:
        d += timedelta(days=1)
        if d.weekday() < 5:
            counted += 1
    return d


def _load() -> pd.DataFrame:
    if LOG_PATH.exists():
        df = pd.read_csv(LOG_PATH, parse_dates=["date"])
        df["date"] = pd.to_datetime(df["date"]).dt.date
        return df
    return pd.DataFrame(columns=COLUMNS)


def _save(df: pd.DataFrame) -> None:
    LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(LOG_PATH, index=False)


def _fill_outcomes(df: pd.DataFrame) -> pd.DataFrame:
    """Fill price_Xd / return_Xd for signals old enough to have outcomes."""
    if df.empty:
        return df

    today = date.today()

    for label, n_days in HORIZONS.items():
        price_col  = f"price_{label}"
        return_col = f"return_{label}"

        def _signal_date(val):
            return val if isinstance(val, date) else val.date()

        age_ok = df["date"].apply(lambda d: (today - _signal_date(d)).days >= n_days)
        missing = df[price_col].isna()
        mask = age_ok & missing

        if mask.sum() == 0:
            continue

        for sym in df.loc[mask, "symbol"].unique():
            sym_mask = mask & (df["symbol"] == sym)
            try:
                hist = yf.download(sym, period="15d", auto_adjust=True, progress=False)
                close = hist["Close"]
                if isinstance(close, pd.DataFrame):
                    close = close.iloc[:, 0]
                close.index = pd.to_datetime(close.index).normalize()

                for idx, row in df[sym_mask].iterrows():
                    target = _bdays_after(_signal_date(row["date"]), n_days)
                    future = close[close.index.date >= target]
                    if future.empty:
                        continue
                    exit_price = float(future.iloc[0])
                    entry = float(row["price_at_signal"])
                    if entry > 0:
                        df.loc[idx, price_col]  = round(exit_price, 4)
                        df.loc[idx, return_col] = round((exit_price - entry) / entry, 6)

            except Exception as e:
                print(f"    ⚠️  Outcome {sym}/{label}: {e}")

    return df


# ── Accuracy report ───────────────────────────────────────────────────────────

def _report(df: pd.DataFrame) -> str:
    lines = ["📊 EarningsSentiment — Shadow Mode Accuracy"]
    lines.append(f"   Total signals logged: {len(df)}")
    lines.append(f"   Period: {df['date'].min()} → {df['date'].max()}" if not df.empty else "")
    lines.append("")

    for label in ["1d", "3d", "5d"]:
        ret_col = f"return_{label}"
        done = df[df[ret_col].notna()]
        if done.empty:
            lines.append(f"  [{label}] no completed signals yet")
            continue

        buys  = done[done["action"] == "BUY"]
        sells = done[done["action"] == "SELL"]

        buy_acc  = (buys[ret_col] > 0).mean()  if len(buys)  else float("nan")
        sell_acc = (sells[ret_col] < 0).mean() if len(sells) else float("nan")
        avg_ret  = done[ret_col].mean()

        def fmt_acc(v):
            return f"{v:.0%}" if v == v else "n/a"

        lines.append(
            f"  [{label}] n={len(done):3d} | "
            f"BUY {fmt_acc(buy_acc)} ({len(buys)}) | "
            f"SELL {fmt_acc(sell_acc)} ({len(sells)}) | "
            f"avg_ret {avg_ret:+.2%}"
        )

    # Action distribution
    if not df.empty:
        dist = df["action"].value_counts().to_dict()
        lines.append(f"\n  Distribution: {dist}")

    return "\n".join(lines)


# ── Main ──────────────────────────────────────────────────────────────────────

def run(report_only: bool = False) -> None:
    print("🔍 Milan Capital — EarningsSentiment Shadow Mode")

    df = _load()
    print(f"   Loaded {len(df)} existing signal(s)")

    # Step 1: fill outcomes
    print("   Filling outcomes for past signals...")
    df = _fill_outcomes(df)

    if report_only:
        print("\n" + _report(df))
        return

    # Step 2: today's signals
    today = date.today()
    already_today = set(df[df["date"] == today]["symbol"].tolist()) if not df.empty else set()

    agent = EarningsSentimentAgent()
    new_rows = []

    for sym in WATCHLIST:
        if sym in already_today:
            print(f"   {sym}: skipped (already logged today)")
            continue
        try:
            hist = download_ohlcv(sym, period="5d")
            last_price = float(
                pd.to_numeric(hist["Close"], errors="coerce").dropna().iloc[-1]
            )
            state = MarketState(symbol=sym, price=last_price, timestamp=str(today))
            sig = agent.generate_signal(state, portfolio={})

            emoji = {"BUY": "🟢", "SELL": "🔴"}.get(sig.action, "⚪")
            print(
                f"   {sym:6s} {emoji} {sig.action:4s} | "
                f"conf={sig.confidence:.2f} | {sig.reason[:55]}"
            )

            new_rows.append({
                "date":              today,
                "symbol":            sym,
                "action":            sig.action,
                "confidence":        round(sig.confidence, 4),
                "reason":            sig.reason,
                "key_catalyst":      sig.meta.get("key_catalyst", ""),
                "sentiment_score":   sig.meta.get("sentiment_score", 0.0),
                "price_at_signal":   round(last_price, 4),
                "price_1d":          None,
                "price_3d":          None,
                "price_5d":          None,
                "return_1d":         None,
                "return_3d":         None,
                "return_5d":         None,
            })
        except Exception as e:
            print(f"   ⚠️  {sym}: {e}")

    if new_rows:
        df = pd.concat([df, pd.DataFrame(new_rows)], ignore_index=True)

    _save(df)
    print(f"\n✅ Shadow log → {LOG_PATH} ({len(df)} lignes)")
    print("\n" + _report(df))


if __name__ == "__main__":
    report_only = "--report" in sys.argv
    run(report_only=report_only)
