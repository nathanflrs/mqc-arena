# src/risk/live_scorer.py
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


@dataclass
class PortfolioPerformance:
    portfolio_return: float   # cumulative return from all round-trips (indexed to 1.0)
    spy_return: float         # SPY buy-and-hold for the same period
    alpha: float              # portfolio_return − spy_return
    first_trade_date: str | None
    last_trade_date: str | None
    n_trades: int
    equity_curve: list        # [{date, portfolio, spy?}] indexed to 100

    def to_dict(self) -> dict:
        return {
            "portfolio_return": self.portfolio_return,
            "spy_return":       self.spy_return,
            "alpha":            self.alpha,
            "first_trade_date": self.first_trade_date,
            "last_trade_date":  self.last_trade_date,
            "n_trades":         self.n_trades,
        }


@dataclass
class DriftAlert:
    agent: str
    oos_sharpe: float    # from walk-forward avg_oos_sharpe
    live_sharpe: float   # current live Sharpe
    drift: float         # oos_sharpe − live_sharpe (positive = deterioration)

    def to_dict(self) -> dict:
        return {
            "agent":       self.agent,
            "oos_sharpe":  self.oos_sharpe,
            "live_sharpe": self.live_sharpe,
            "drift":       self.drift,
        }


@dataclass
class RegimeStats:
    regime: str
    n_observations: int        # décisions avec données forward disponibles
    mean_fwd_return_5d: float  # rendement forward moyen à 5j
    hit_rate: float            # fraction de rendements positifs

    def to_dict(self) -> dict:
        return {
            "regime":             self.regime,
            "n_observations":     self.n_observations,
            "mean_fwd_return_5d": round(self.mean_fwd_return_5d, 4),
            "hit_rate":           round(self.hit_rate, 4),
        }


@dataclass
class FillStats:
    n_orders: int
    fill_rate: float           # fraction of orders with avg_fill_price > 0
    mean_slippage_bps: float   # average slippage vs signal price (all filled orders)
    mean_slippage_buy_bps: float
    mean_slippage_sell_bps: float
    worst_slippage_bps: float  # max unfavourable slippage seen

    def to_dict(self) -> dict:
        return {
            "n_orders":              self.n_orders,
            "fill_rate":             round(self.fill_rate, 4),
            "mean_slippage_bps":     round(self.mean_slippage_bps, 2),
            "mean_slippage_buy_bps": round(self.mean_slippage_buy_bps, 2),
            "mean_slippage_sell_bps":round(self.mean_slippage_sell_bps, 2),
            "worst_slippage_bps":    round(self.worst_slippage_bps, 2),
        }


@dataclass
class LiveScorerConfig:
    decisions_path: str = "logs/decisions.csv"
    executions_path: str = "logs/executions.csv"
    # Nombre minimum de round-trips pour considérer le Sharpe live fiable
    min_trades: int = 3


@dataclass
class RoundTrip:
    agent: str
    symbol: str
    entry_price: float
    exit_price: float
    entry_date: pd.Timestamp
    exit_date: pd.Timestamp

    @property
    def return_pct(self) -> float:
        return (self.exit_price - self.entry_price) / self.entry_price

    @property
    def holding_days(self) -> int:
        return max(1, (self.exit_date - self.entry_date).days)


def _sharpe_from_roundtrips(trips: List[RoundTrip], risk_free: float = 0.04) -> float:
    """
    Sharpe annualisé à partir des returns de round-trips.
    Annualisation : sqrt(252 / avg_holding_days).
    """
    if len(trips) < 3:
        return 0.0
    returns = np.array([t.return_pct for t in trips])
    avg_hold = float(np.mean([t.holding_days for t in trips]))
    trades_per_year = 252.0 / max(avg_hold, 1.0)

    # Ajustement risk-free proportionnel à la durée moyenne
    rf_per_trade = risk_free * avg_hold / 252
    excess = returns - rf_per_trade

    std = excess.std()
    if std == 0:
        return 0.0
    return float(excess.mean() / std * np.sqrt(trades_per_year))


def kelly_half_fraction(
    trips: List[RoundTrip],
    min_trades: int = 10,
    max_fraction: float = 0.25,
) -> float:
    """
    Half-Kelly criterion: optimal position size = f*/2 to reduce variance.

    f* = (p * b - q) / b
    where:
        p = win rate (fraction of profitable trades)
        q = 1 - p
        b = avg_win / avg_loss  (odds ratio)

    Returns 0.0 if fewer than *min_trades* round-trips (estimates too noisy).
    Capped at *max_fraction* (default 25%) regardless of formula output.
    """
    if len(trips) < min_trades:
        return 0.0

    returns = np.array([t.return_pct for t in trips])
    wins  = returns[returns > 0]
    losses = np.abs(returns[returns < 0])

    p = len(wins) / len(returns)
    q = 1.0 - p

    if len(wins) == 0 or len(losses) == 0:
        return 0.0

    b = float(wins.mean()) / float(losses.mean())  # avg win / avg loss
    if b <= 0:
        return 0.0

    f_star = (p * b - q) / b
    half_kelly = f_star / 2.0

    # Clamp: never short (negative) and never exceed max_fraction
    return float(np.clip(half_kelly, 0.0, max_fraction))


def _max_drawdown_from_trips(trips: List[RoundTrip]) -> float:
    """
    Max drawdown on a synthetic cumulative-equity curve built from
    sequential round-trip returns sorted by exit date.
    """
    if not trips:
        return 0.0
    sorted_trips = sorted(trips, key=lambda t: t.exit_date)
    equity = np.cumprod([1.0 + t.return_pct for t in sorted_trips])
    peak = np.maximum.accumulate(equity)
    dd = (equity - peak) / peak
    return float(dd.min())


# ─── AgentMetrics ─────────────────────────────────────────────────────────────

@dataclass
class AgentMetrics:
    agent: str
    n_trades: int             # completed round-trips
    win_rate: float           # fraction of profitable round-trips
    avg_return_pct: float     # mean per-trade return
    total_pnl_pct: float      # simple sum of round-trip returns
    max_drawdown: float       # max drawdown on cumulative equity from round-trips
    sharpe: float             # annualised live Sharpe
    avg_holding_days: float

    def to_dict(self) -> dict:
        return {
            "agent":            self.agent,
            "n_trades":         self.n_trades,
            "win_rate":         round(self.win_rate, 4),
            "avg_return_pct":   round(self.avg_return_pct, 4),
            "total_pnl_pct":    round(self.total_pnl_pct, 4),
            "max_drawdown":     round(self.max_drawdown, 4),
            "sharpe":           round(self.sharpe, 4),
            "avg_holding_days": round(self.avg_holding_days, 1),
        }


# ─── LiveScorer ───────────────────────────────────────────────────────────────

class LiveScorer:
    """
    Reconstruit les round-trips par agent depuis les logs de décisions
    et d'exécutions, puis calcule un Sharpe live par (agent, symbole).

    Attribution : le BUY définit le propriétaire de la position.
    Le SELL suivant ferme la position et est attribué à l'agent qui a ouvert.
    """

    def __init__(self, config: LiveScorerConfig | None = None):
        self.cfg = config or LiveScorerConfig()
        self._roundtrips: List[RoundTrip] = []
        self._loaded = False

    def _load(self) -> None:
        if self._loaded:
            return
        self._roundtrips = self._build_roundtrips()
        self._loaded = True

    def _build_roundtrips(self) -> List[RoundTrip]:
        dec = self._load_decisions()
        exc = self._load_executions()
        if dec is None or exc is None:
            return []

        # Attribution : plan_id + symbol → agent gagnant
        winner_map: Dict[Tuple[str, str], str] = {}
        for _, row in dec.iterrows():
            pid = str(row.get("plan_id", ""))
            sym = str(row.get("symbol", ""))
            is_winner = bool(row.get("is_winner", False))
            agent = str(row.get("agent", ""))
            if pid and sym and is_winner and agent:
                winner_map[(pid, sym)] = agent

        open_positions: Dict[str, Tuple[str, float, pd.Timestamp]] = {}
        roundtrips: List[RoundTrip] = []

        for _, row in exc.sort_values("timestamp").iterrows():
            sym = str(row.get("symbol", ""))
            side = str(row.get("side", "")).upper()
            pid = str(row.get("plan_id", ""))
            price = self._safe_float(row.get("limit_price") or row.get("last_price"))
            ts = pd.to_datetime(row.get("timestamp"), utc=True, errors="coerce")

            if not sym or not side or price is None or pd.isna(ts):
                continue

            if side == "BUY":
                agent = winner_map.get((pid, sym), "")
                if agent:
                    open_positions[sym] = (agent, price, ts)

            elif side == "SELL" and sym in open_positions:
                agent, entry_price, entry_date = open_positions.pop(sym)
                roundtrips.append(RoundTrip(
                    agent=agent,
                    symbol=sym,
                    entry_price=entry_price,
                    exit_price=price,
                    entry_date=entry_date,
                    exit_date=ts,
                ))

        return roundtrips

    def _load_decisions(self) -> pd.DataFrame | None:
        path = Path(self.cfg.decisions_path)
        if not path.exists():
            return None
        try:
            df = pd.read_csv(path)
            if "is_winner" not in df.columns:
                logger.info("decisions.csv sans colonne is_winner — live scoring désactivé pour l'instant")
                return None
            return df
        except Exception as e:
            logger.warning("Impossible de lire decisions.csv : %s", e)
            return None

    def _load_executions(self) -> pd.DataFrame | None:
        path = Path(self.cfg.executions_path)
        if not path.exists():
            return None
        try:
            return pd.read_csv(path)
        except Exception as e:
            logger.warning("Impossible de lire executions.csv : %s", e)
            return None

    @staticmethod
    def _safe_float(val) -> float | None:
        try:
            f = float(val)
            return f if not np.isnan(f) else None
        except (TypeError, ValueError):
            return None

    # ── Existing public API ───────────────────────────────────────────────────

    def compute_live_sharpes(self) -> Dict[str, Dict[str, float]]:
        """Retourne {agent_name: {symbol: sharpe_live}}."""
        self._load()
        result: Dict[str, Dict[str, float]] = {}
        groups: Dict[Tuple[str, str], List[RoundTrip]] = {}
        for t in self._roundtrips:
            groups.setdefault((t.agent, t.symbol), []).append(t)
        for (agent, sym), trips in groups.items():
            if len(trips) < self.cfg.min_trades:
                continue
            result.setdefault(agent, {})[sym] = round(_sharpe_from_roundtrips(trips), 4)
        return result

    def get_n_trades(self, agent: str, symbol: str) -> int:
        self._load()
        return sum(1 for t in self._roundtrips if t.agent == agent and t.symbol == symbol)

    def get_roundtrips(self, agent: str | None = None, symbol: str | None = None) -> List[RoundTrip]:
        self._load()
        return [
            t for t in self._roundtrips
            if (agent is None or t.agent == agent)
            and (symbol is None or t.symbol == symbol)
        ]

    # ── New: per-agent aggregate metrics ─────────────────────────────────────

    def compute_agent_metrics(self) -> Dict[str, AgentMetrics]:
        """
        Returns {agent_name: AgentMetrics} for every agent with ≥1 round-trip,
        aggregated across all symbols.
        """
        self._load()
        groups: Dict[str, List[RoundTrip]] = {}
        for t in self._roundtrips:
            groups.setdefault(t.agent, []).append(t)

        result: Dict[str, AgentMetrics] = {}
        for agent, trips in groups.items():
            n = len(trips)
            returns = np.array([t.return_pct for t in trips])
            result[agent] = AgentMetrics(
                agent=agent,
                n_trades=n,
                win_rate=float(np.mean(returns > 0)) if n > 0 else 0.0,
                avg_return_pct=float(np.mean(returns)) if n > 0 else 0.0,
                total_pnl_pct=float(np.sum(returns)),
                max_drawdown=_max_drawdown_from_trips(trips),
                sharpe=_sharpe_from_roundtrips(trips),
                avg_holding_days=float(np.mean([t.holding_days for t in trips])) if n > 0 else 0.0,
            )
        return result

    def compute_kelly_weights(
        self,
        min_trades: int = 10,
        max_fraction: float = 0.25,
    ) -> Dict[str, float]:
        """
        Returns {agent_name: half_kelly_fraction} for agents with sufficient data.
        Agents with < min_trades round-trips are omitted (use default target_weight).
        """
        self._load()
        groups: Dict[str, List[RoundTrip]] = {}
        for t in self._roundtrips:
            groups.setdefault(t.agent, []).append(t)
        return {
            agent: kelly_half_fraction(trips, min_trades=min_trades, max_fraction=max_fraction)
            for agent, trips in groups.items()
            if len(trips) >= min_trades
        }

    def compute_portfolio_performance(self) -> PortfolioPerformance | None:
        """
        Builds a cumulative equity curve from all round-trips (indexed to 100)
        and compares with SPY buy-and-hold for the same period.
        Returns None if there are no round-trips yet.
        """
        self._load()
        if not self._roundtrips:
            return None

        sorted_trips = sorted(self._roundtrips, key=lambda t: t.exit_date)
        first_date = sorted_trips[0].entry_date
        last_date  = sorted_trips[-1].exit_date

        equity = 100.0
        equity_curve: list = []
        for t in sorted_trips:
            equity *= (1.0 + t.return_pct)
            equity_curve.append({
                "date":      t.exit_date.strftime("%Y-%m-%d"),
                "portfolio": round(equity, 2),
            })

        portfolio_return = (equity - 100.0) / 100.0

        spy_return = 0.0
        try:
            from src.data.market_data import download_ohlcv
            spy_df    = download_ohlcv("SPY")
            spy_close = pd.to_numeric(spy_df["Close"], errors="coerce").dropna()
            spy_close.index = pd.to_datetime(spy_close.index, utc=True)

            spy_range = spy_close[
                (spy_close.index >= first_date) & (spy_close.index <= last_date)
            ]
            if len(spy_range) >= 2:
                spy_start  = float(spy_range.iloc[0])
                spy_return = float(spy_range.iloc[-1] / spy_start - 1.0)
                spy_map    = {
                    str(ts.date()): round(100.0 * float(px) / spy_start, 2)
                    for ts, px in spy_range.items()
                }
                for pt in equity_curve:
                    if pt["date"] in spy_map:
                        pt["spy"] = spy_map[pt["date"]]
        except Exception as e:
            logger.warning("SPY benchmark fetch failed: %s", e)

        return PortfolioPerformance(
            portfolio_return=round(portfolio_return, 4),
            spy_return=round(spy_return, 4),
            alpha=round(portfolio_return - spy_return, 4),
            first_trade_date=first_date.strftime("%Y-%m-%d"),
            last_trade_date=last_date.strftime("%Y-%m-%d"),
            n_trades=len(sorted_trips),
            equity_curve=equity_curve,
        )

    def compute_drift_alerts(
        self,
        drift_threshold: float = 0.5,
        wf_path: str = "logs/walkforward_results.csv",
    ) -> list:
        """
        Compares live Sharpe per agent with the walk-forward avg_oos_sharpe.
        Returns DriftAlert list for agents where oos_sharpe − live_sharpe > threshold.
        """
        wf_file = Path(wf_path)
        if not wf_file.exists():
            return []
        try:
            wf_df = pd.read_csv(wf_file)
        except Exception as e:
            logger.warning("Cannot read walkforward_results.csv: %s", e)
            return []

        # Mean avg_oos_sharpe per agent across all symbols
        wf_by_agent: dict[str, float] = {}
        for agent in wf_df["agent"].dropna().unique():
            rows = wf_df[wf_df["agent"] == agent]
            wf_by_agent[str(agent)] = float(rows["avg_oos_sharpe"].mean())

        live_metrics = self.compute_agent_metrics()

        alerts: list = []
        for agent, oos_sharpe in wf_by_agent.items():
            if agent not in live_metrics:
                continue
            live_sharpe = live_metrics[agent].sharpe
            drift = oos_sharpe - live_sharpe
            if drift > drift_threshold:
                alerts.append(DriftAlert(
                    agent=agent,
                    oos_sharpe=round(oos_sharpe, 4),
                    live_sharpe=round(live_sharpe, 4),
                    drift=round(drift, 4),
                ))

        return sorted(alerts, key=lambda a: a.drift, reverse=True)

    def compute_regime_accuracy(self, fwd_days: int = 5) -> list:
        """
        For each signal in decisions.csv, compute the forward N-day return of
        the underlying symbol, then group by regime (bull/bear/choppy).
        Returns a list of RegimeStats — empty if insufficient data.
        """
        from src.data.market_data import download_ohlcv

        path = Path(self.cfg.decisions_path)
        if not path.exists():
            return []
        try:
            df = pd.read_csv(path)
        except Exception:
            return []

        needed = {"regime", "symbol", "ts"}
        if not needed.issubset(df.columns):
            return []

        df = df[["ts", "symbol", "regime"]].dropna(subset=list(needed))
        df["ts"] = pd.to_datetime(df["ts"], errors="coerce", utc=True)
        df = df.dropna(subset=["ts"])
        if df.empty:
            return []

        records: list = []
        for sym in df["symbol"].unique():
            try:
                ohlcv  = download_ohlcv(sym, period="1y")
                close  = pd.to_numeric(ohlcv["Close"], errors="coerce").dropna()
                close.index = pd.to_datetime(close.index, utc=True)
            except Exception as e:
                logger.warning("compute_regime_accuracy: %s — %s", sym, e)
                continue

            for _, row in df[df["symbol"] == sym].iterrows():
                ts      = row["ts"]
                regime  = row["regime"]
                future  = close[close.index > ts]
                if len(future) < fwd_days:
                    continue
                past = close[close.index <= ts]
                if past.empty:
                    continue
                entry_px = float(past.iloc[-1])
                exit_px  = float(future.iloc[fwd_days - 1])
                if entry_px <= 0:
                    continue
                records.append({"regime": regime, "fwd": (exit_px - entry_px) / entry_px})

        if not records:
            return []

        fwd_df = pd.DataFrame(records)
        result: list = []
        for regime in ["bull", "bear", "choppy"]:
            sub = fwd_df[fwd_df["regime"] == regime]["fwd"]
            if sub.empty:
                continue
            result.append(RegimeStats(
                regime=regime,
                n_observations=len(sub),
                mean_fwd_return_5d=float(sub.mean()),
                hit_rate=float((sub > 0).mean()),
            ))
        return result

    def compute_fill_stats(self, executions_path: str | None = None) -> FillStats | None:
        """
        Reads executions.csv and computes fill rate + slippage statistics.
        Requires avg_fill_price and slippage_bps columns (new schema from runner.py).
        Returns None if the file doesn't exist or has no data.
        """
        path = Path(executions_path or self.cfg.executions_path)
        if not path.exists():
            return None
        try:
            df = pd.read_csv(path)
        except Exception as e:
            logger.warning("compute_fill_stats: cannot read %s: %s", path, e)
            return None

        if df.empty:
            return None

        n_orders = len(df)

        # Fill rate: rows where avg_fill_price column exists and > 0
        if "avg_fill_price" in df.columns:
            filled = df[df["avg_fill_price"] > 0]
            fill_rate = len(filled) / n_orders if n_orders else 0.0
        else:
            # Legacy schema — all rows treated as filled (no avg_fill_price column)
            return FillStats(
                n_orders=n_orders,
                fill_rate=1.0,
                mean_slippage_bps=0.0,
                mean_slippage_buy_bps=0.0,
                mean_slippage_sell_bps=0.0,
                worst_slippage_bps=0.0,
            )

        if "slippage_bps" not in df.columns or filled.empty:
            return FillStats(
                n_orders=n_orders,
                fill_rate=fill_rate,
                mean_slippage_bps=0.0,
                mean_slippage_buy_bps=0.0,
                mean_slippage_sell_bps=0.0,
                worst_slippage_bps=0.0,
            )

        slip = filled["slippage_bps"].dropna()
        buys  = filled[filled["side"].str.upper() == "BUY"]["slippage_bps"].dropna()
        sells = filled[filled["side"].str.upper() == "SELL"]["slippage_bps"].dropna()

        return FillStats(
            n_orders=n_orders,
            fill_rate=fill_rate,
            mean_slippage_bps=float(slip.mean()) if len(slip) else 0.0,
            mean_slippage_buy_bps=float(buys.mean()) if len(buys) else 0.0,
            mean_slippage_sell_bps=float(sells.mean()) if len(sells) else 0.0,
            worst_slippage_bps=float(slip.max()) if len(slip) else 0.0,
        )

    def generate_tearsheet(self, path: str | None = None) -> str:
        """
        Generate a weekly tearsheet CSV.
        Default: logs/tearsheet_YYYY-WW.csv. Returns the path written.
        """
        from datetime import datetime

        if path is None:
            week = datetime.now().strftime("%Y-W%V")
            path = f"logs/tearsheet_{week}.csv"

        metrics = self.compute_agent_metrics()
        if not metrics:
            logger.info("No round-trips yet — tearsheet not generated.")
            return path

        rows = [m.to_dict() for m in metrics.values()]
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(rows).sort_values("sharpe", ascending=False).to_csv(path, index=False)
        logger.info("Tearsheet written → %s", path)
        return path

    def send_weekly_tearsheet(self) -> None:
        """
        Generate tearsheet and send formatted Telegram summary.
        Called every Monday by src/notify/weekly_tearsheet.py.
        """
        from datetime import datetime
        from src.notify.telegram import send_message

        path = self.generate_tearsheet()
        metrics = self.compute_agent_metrics()

        if not metrics:
            send_message("📊 Tearsheet hebdo : aucun round-trip enregistré pour l'instant.")
            return

        now = datetime.now()
        lines = [f"📊 TEARSHEET — Semaine {now.strftime('%V')} ({now.year})\n"]

        for m in sorted(metrics.values(), key=lambda x: x.sharpe, reverse=True):
            sign = "+" if m.total_pnl_pct >= 0 else ""
            lines.append(
                f"\n🤖 {m.agent}\n"
                f"   Trades: {m.n_trades} | Win: {m.win_rate:.0%} | Sharpe: {m.sharpe:.2f}\n"
                f"   PnL total: {sign}{m.total_pnl_pct:.1%} | Max DD: {m.max_drawdown:.1%}\n"
                f"   Avg/trade: {m.avg_return_pct:+.2%} | Hold moy: {m.avg_holding_days:.0f}j"
            )

        lines.append(f"\n📁 {path}")
        send_message("\n".join(lines)[:4096])
