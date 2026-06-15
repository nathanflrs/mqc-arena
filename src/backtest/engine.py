# src/backtest/engine.py
from __future__ import annotations

import pandas as pd
import numpy as np
from dataclasses import dataclass
from typing import List, Dict, Optional
from src.agents.base import BaseAgent, MarketState, AgentSignal


@dataclass
class Trade:
    date: str
    symbol: str
    action: str
    qty: float
    price: float
    notional: float
    agent: str
    reason: str


@dataclass
class BacktestResult:
    symbol: str
    trades: List[Trade]
    equity_curve: pd.Series
    total_return: float
    annualized_return: float
    sharpe_ratio: float
    max_drawdown: float
    win_rate: float
    n_trades: int


def _compute_regime_series(close: pd.Series) -> pd.Series:
    """
    Per-day regime label from SMA50/SMA200 — used by WalkForwardEngine
    so agents receive regime context during historical backtest.
    - bull  : close > SMA200 AND SMA50 > SMA200
    - bear  : close < SMA200 AND SMA50 < SMA200
    - choppy: else (incl. insufficient history for SMA200)
    """
    sma50 = close.rolling(50).mean()
    sma200 = close.rolling(200).mean()
    regime = pd.Series("choppy", index=close.index, dtype=object)
    valid = sma200.notna()
    regime[valid & (close > sma200) & (sma50 > sma200)] = "bull"
    regime[valid & (close < sma200) & (sma50 < sma200)] = "bear"
    return regime


def _sharpe(returns: pd.Series, risk_free: float = 0.04) -> float:
    excess = returns - risk_free / 252
    std = excess.std()
    if std == 0 or np.isnan(std):
        return 0.0
    result = float(excess.mean() / std * np.sqrt(252))
    return 0.0 if (np.isnan(result) or abs(result) > 100) else result


def _max_drawdown(equity: pd.Series) -> float:
    peak = equity.cummax()
    dd = (equity - peak) / peak
    return float(dd.min())


def _win_rate(trades: List[Trade]) -> float:
    buys = [t for t in trades if t.action == "BUY"]
    sells = [t for t in trades if t.action == "SELL"]
    if not sells:
        return 0.0
    wins = 0
    for s in sells:
        buy_price = next(
            (b.price for b in reversed(buys) if b.symbol == s.symbol), None
        )
        if buy_price is not None and s.price > buy_price:
            wins += 1
    return wins / len(sells)


class BacktestEngine:
    """
    Backtest walk-forward jour par jour.
    Fixes v2:
    - portfolio mis à jour correctement chaque jour
    - target_weight élevé (0.95) pour investir vraiment
    - cooldown entre trades pour éviter over-trading
    """

    def __init__(
        self,
        agent: BaseAgent,
        initial_capital: float = 100_000.0,
        target_weight: float = 0.95,   # investi 95% du capital
        commission: float = 0.001,
        min_history: int = 210,
        cooldown_days: int = 20,       # minimum 20 jours entre trades (réduit l'over-trading)
        long_bias_bull_threshold: float = 0.90,  # min confidence to exit in bull regime
    ):
        self.agent = agent
        self.initial_capital = initial_capital
        self.target_weight = target_weight
        self.commission = commission
        self.min_history = min_history
        self.cooldown_days = cooldown_days
        self.long_bias_bull_threshold = long_bias_bull_threshold

    def run(
        self,
        symbol: str,
        df: pd.DataFrame,
        regime_series: Optional[pd.Series] = None,
    ) -> BacktestResult:

        close = pd.to_numeric(df["Close"], errors="coerce").dropna()
        dates = close.index

        capital = self.initial_capital
        position = 0.0
        trades: List[Trade] = []
        equity_values = []
        last_trade_idx = -999  # cooldown tracker

        for i in range(self.min_history, len(dates)):
            date = dates[i]
            px = float(close.iloc[i])
            window = df.iloc[:i + 1]

            # Portfolio correctement mis à jour
            portfolio = {symbol: position}

            # Régime du jour
            regime = None
            if regime_series is not None and date in regime_series.index:
                regime = regime_series.loc[date]

            state = MarketState(
                symbol=symbol,
                price=px,
                timestamp=str(date),
            )

            sig = self.agent.generate_signal(
                state=state,
                portfolio=portfolio,
                regime=regime,
                data=window,
            )

            portfolio_value = capital + position * px
            days_since_trade = i - last_trade_idx

            # BUY
            if sig.action == "BUY" and position == 0 and days_since_trade >= self.cooldown_days:
                target_notional = portfolio_value * self.target_weight
                qty = int(target_notional / px)
                if qty > 0:
                    cost = qty * px * (1 + self.commission)
                    if cost <= capital:
                        capital -= cost
                        position += qty
                        last_trade_idx = i
                        trades.append(Trade(
                            date=str(date),
                            symbol=symbol,
                            action="BUY",
                            qty=qty,
                            price=px,
                            notional=qty * px,
                            agent=sig.agent_name,
                            reason=sig.reason,
                        ))

            # SELL — long-bias: in bull regime hold through weak-conviction exits
            elif sig.action == "SELL" and position > 0 and days_since_trade >= self.cooldown_days:
                bull_dampen = regime == "bull" and sig.confidence < self.long_bias_bull_threshold
                if not bull_dampen:
                    proceeds = position * px * (1 - self.commission)
                    capital += proceeds
                    last_trade_idx = i
                    trades.append(Trade(
                        date=str(date),
                        symbol=symbol,
                        action="SELL",
                        qty=position,
                        price=px,
                        notional=position * px,
                        agent=sig.agent_name,
                        reason=sig.reason,
                    ))
                    position = 0

            equity_values.append(capital + position * px)

        equity = pd.Series(equity_values, index=dates[self.min_history:])
        returns = equity.pct_change().dropna()

        total_return = float((equity.iloc[-1] - self.initial_capital) / self.initial_capital)
        n_days = len(equity)
        annualized = float((1 + total_return) ** (252 / n_days) - 1) if n_days > 0 else 0.0

        return BacktestResult(
            symbol=symbol,
            trades=trades,
            equity_curve=equity,
            total_return=total_return,
            annualized_return=annualized,
            sharpe_ratio=_sharpe(returns),
            max_drawdown=_max_drawdown(equity),
            win_rate=_win_rate(trades),
            n_trades=len(trades),
        )


# ─── Walk-Forward Engine ──────────────────────────────────────────────────────

BDAYS_PER_MONTH = 21


@dataclass
class WindowResult:
    window_idx: int
    train_start: str
    train_end: str
    test_start: str
    test_end: str
    # In-sample (train) metrics
    is_sharpe: float
    is_return: float
    # Out-of-sample (test) metrics
    oos_sharpe: float
    oos_return: float
    oos_max_drawdown: float
    oos_n_trades: int
    oos_win_rate: float
    # Benchmark = buy & hold of the tested asset on the test period
    benchmark_return: float
    # Alpha = oos_return − benchmark_return
    alpha: float


@dataclass
class WalkForwardResult:
    agent_name: str
    symbol: str
    windows: List[WindowResult]
    avg_oos_sharpe: float
    avg_is_sharpe: float
    avg_alpha: float
    # True if IS Sharpe >> OOS Sharpe across all windows (overfitting signal)
    lookahead_warning: bool

    def to_csv_rows(self) -> List[dict]:
        rows = []
        for w in self.windows:
            rows.append({
                "agent":             self.agent_name,
                "symbol":            self.symbol,
                "window":            w.window_idx,
                "train_start":       w.train_start,
                "train_end":         w.train_end,
                "test_start":        w.test_start,
                "test_end":          w.test_end,
                "is_sharpe":         round(w.is_sharpe, 4),
                "is_return":         round(w.is_return, 4),
                "oos_sharpe":        round(w.oos_sharpe, 4),
                "oos_return":        round(w.oos_return, 4),
                "oos_max_drawdown":  round(w.oos_max_drawdown, 4),
                "oos_n_trades":      w.oos_n_trades,
                "oos_win_rate":      round(w.oos_win_rate, 4),
                "benchmark_return":  round(w.benchmark_return, 4),
                "alpha":             round(w.alpha, 4),
                "avg_oos_sharpe":    round(self.avg_oos_sharpe, 4),
                "avg_is_sharpe":     round(self.avg_is_sharpe, 4),
                "avg_alpha":         round(self.avg_alpha, 4),
                "lookahead_warning": self.lookahead_warning,
            })
        return rows


class WalkForwardEngine:
    """
    Walk-forward validation wrapper around BacktestEngine.

    Divides data into rolling windows:
        Train : 18 months (≈378 bdays)
        Test  :  6 months (≈126 bdays)
        Step  :  3 months (≈ 63 bdays)

    For each window, the BacktestEngine is run on the full train+test slice.
    IS metrics come from the equity curve over the train period;
    OOS metrics come from the equity curve over the test period.

    Lookahead-bias warning: structural lookahead bias (using future prices inside
    generate_signal) cannot be detected statically, but is proxied by checking
    whether IS Sharpe is systematically much higher than OOS Sharpe — a strong
    sign that the strategy overfits the training period.
    """

    TRAIN_BDAYS: int = 18 * BDAYS_PER_MONTH   # 378
    TEST_BDAYS: int  =  6 * BDAYS_PER_MONTH   # 126
    STEP_BDAYS: int  =  3 * BDAYS_PER_MONTH   # 63

    def __init__(
        self,
        agent: "BaseAgent",
        initial_capital: float = 100_000.0,
        target_weight: float = 0.95,
        commission: float = 0.001,
        min_history: int = 210,
        cooldown_days: int = 20,
    ):
        self.agent = agent
        self.initial_capital = initial_capital
        self.target_weight = target_weight
        self.commission = commission
        self.min_history = min_history
        self.cooldown_days = cooldown_days

    def run(
        self,
        symbol: str,
        df: pd.DataFrame,
        benchmark_df: Optional[pd.DataFrame] = None,
    ) -> WalkForwardResult:
        """
        benchmark_df: optional external price series (e.g. SPY) to use as
        buy-and-hold benchmark on each test window. If None, uses the asset
        itself as benchmark (passive hold).
        """
        n = len(df)
        min_rows = self.TRAIN_BDAYS + self.TEST_BDAYS
        windows: List[WindowResult] = []

        # IS equity split point (relative to equity curve which starts at min_history)
        is_split = self.TRAIN_BDAYS - self.min_history  # 378 - 210 = 168

        start = 0
        w_idx = 0
        while start + min_rows <= n:
            end = start + min_rows
            full_slice = df.iloc[start:end].copy()

            inner = BacktestEngine(
                agent=self.agent,
                initial_capital=self.initial_capital,
                target_weight=self.target_weight,
                commission=self.commission,
                min_history=self.min_history,
                cooldown_days=self.cooldown_days,
            )
            close_slice = pd.to_numeric(full_slice["Close"], errors="coerce").dropna()
            regime_series = _compute_regime_series(close_slice)
            result = inner.run(symbol, full_slice, regime_series=regime_series)
            equity = result.equity_curve

            # ── IS metrics (equity from min_history to TRAIN_BDAYS) ──────────
            eq_is = equity.iloc[:is_split]
            if len(eq_is) > 1:
                r_is = eq_is.pct_change().dropna()
                is_sharpe = _sharpe(r_is)
                is_return = float((eq_is.iloc[-1] - eq_is.iloc[0]) / eq_is.iloc[0])
            else:
                is_sharpe, is_return = 0.0, 0.0

            # ── OOS metrics (equity from TRAIN_BDAYS to end) ─────────────────
            eq_oos = equity.iloc[is_split:]
            if len(eq_oos) > 1:
                r_oos = eq_oos.pct_change().dropna()
                oos_sharpe    = _sharpe(r_oos)
                oos_return    = float((eq_oos.iloc[-1] - eq_oos.iloc[0]) / eq_oos.iloc[0])
                oos_max_dd    = _max_drawdown(eq_oos)
            else:
                oos_sharpe, oos_return, oos_max_dd = 0.0, 0.0, 0.0

            # OOS trade count (only trades executed in the test period)
            test_start_date = full_slice.index[self.TRAIN_BDAYS]
            oos_trades = [
                t for t in result.trades
                if pd.Timestamp(t.date) >= test_start_date
            ]

            # ── Benchmark: buy & hold on test period ─────────────────────────
            bench_src = benchmark_df if benchmark_df is not None else df
            try:
                bench_close = pd.to_numeric(
                    bench_src.iloc[start + self.TRAIN_BDAYS: end]["Close"],
                    errors="coerce",
                ).dropna()
                benchmark_return = float(
                    (bench_close.iloc[-1] - bench_close.iloc[0]) / bench_close.iloc[0]
                )
            except (IndexError, ZeroDivisionError, KeyError):
                benchmark_return = 0.0

            windows.append(WindowResult(
                window_idx=w_idx,
                train_start=str(full_slice.index[0].date()),
                train_end=str(full_slice.index[self.TRAIN_BDAYS - 1].date()),
                test_start=str(full_slice.index[self.TRAIN_BDAYS].date()),
                test_end=str(full_slice.index[-1].date()),
                is_sharpe=is_sharpe,
                is_return=is_return,
                oos_sharpe=oos_sharpe,
                oos_return=oos_return,
                oos_max_drawdown=oos_max_dd,
                oos_n_trades=len(oos_trades),
                oos_win_rate=_win_rate(oos_trades) if oos_trades else 0.0,
                benchmark_return=benchmark_return,
                alpha=oos_return - benchmark_return,
            ))

            start += self.STEP_BDAYS
            w_idx += 1

        avg_oos_sharpe = float(np.mean([w.oos_sharpe for w in windows])) if windows else 0.0
        avg_is_sharpe  = float(np.mean([w.is_sharpe  for w in windows])) if windows else 0.0
        avg_alpha      = float(np.mean([w.alpha       for w in windows])) if windows else 0.0

        # Lookahead proxy: IS Sharpe systematically positive and OOS negative
        lookahead_warning = (avg_is_sharpe > 1.5) and (avg_oos_sharpe < 0.0)

        return WalkForwardResult(
            agent_name=self.agent.name,
            symbol=symbol,
            windows=windows,
            avg_oos_sharpe=avg_oos_sharpe,
            avg_is_sharpe=avg_is_sharpe,
            avg_alpha=avg_alpha,
            lookahead_warning=lookahead_warning,
        )

    @staticmethod
    def save_csv(
        results: List[WalkForwardResult],
        path: str = "logs/walkforward_results.csv",
    ) -> None:
        import pathlib
        rows = []
        for r in results:
            rows.extend(r.to_csv_rows())
        if rows:
            pathlib.Path(path).parent.mkdir(parents=True, exist_ok=True)
            pd.DataFrame(rows).to_csv(path, index=False)
            print(f"✅ Walk-forward results → {path} ({len(rows)} rows)")