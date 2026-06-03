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


def _sharpe(returns: pd.Series, risk_free: float = 0.04) -> float:
    excess = returns - risk_free / 252
    if excess.std() == 0:
        return 0.0
    return float(excess.mean() / excess.std() * np.sqrt(252))


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
        cooldown_days: int = 5,        # minimum 5 jours entre trades
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

            # SELL
            elif sig.action == "SELL" and position > 0 and days_since_trade >= self.cooldown_days:
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