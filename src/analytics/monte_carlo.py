"""
Milan Capital — Institutional Monte Carlo Simulation Engine

Bootstraps empirical round-trip returns from decisions.csv + executions.csv
(no Gaussian assumption). Applies the real Milan Capital pipeline per path:
GMM Kelly scaling, graduated circuit breaker, transaction costs.
"""
from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# ── Constants ────────────────────────────────────────────────────────────────
_TC_ROUND_TRIP: float = 0.0009   # 2×0.0002 spread + 0.0005 commission = 9 bps
_MIN_ROUNDTRIPS: int = 30        # fallback to walkforward if below this


# ══════════════════════════════════════════════════════════════════════════════
# Part 1 — ReturnBootstrapper
# ══════════════════════════════════════════════════════════════════════════════

@dataclass
class _RoundTrip:
    symbol: str
    entry_price: float
    exit_price: float
    net_return: float   # (exit - entry) / entry - TC
    regime: str         # GMM regime at entry time


class ReturnBootstrapper:
    """
    Loads completed round-trips from decisions.csv + executions.csv and
    exposes arrays of net returns for Monte Carlo bootstrapping.

    Fallback: if fewer than _MIN_ROUNDTRIPS round-trips are available,
    uses OOS returns from walkforward_results.csv converted to daily
    equivalents so the MC engine receives calibrated daily return samples.
    """

    def __init__(
        self,
        decisions_path: str = "logs/decisions.csv",
        executions_path: str = "logs/executions.csv",
        walkforward_path: str = "logs/walkforward_results.csv",
    ) -> None:
        self._decisions_path = Path(decisions_path)
        self._executions_path = Path(executions_path)
        self._walkforward_path = Path(walkforward_path)
        self._roundtrips: List[_RoundTrip] | None = None

    # ── Public API ───────────────────────────────────────────────────────────

    def load_roundtrips(self) -> pd.DataFrame:
        """Returns a DataFrame with columns: symbol, net_return, regime."""
        trips = self._get_roundtrips()
        if not trips:
            return pd.DataFrame(columns=["symbol", "net_return", "regime"])
        return pd.DataFrame([
            {"symbol": t.symbol, "net_return": t.net_return, "regime": t.regime}
            for t in trips
        ])

    def get_portfolio_returns(self) -> np.ndarray:
        """Net return array for all completed round-trips (or WF fallback)."""
        trips = self._get_roundtrips()
        if len(trips) >= _MIN_ROUNDTRIPS:
            return np.array([t.net_return for t in trips], dtype=float)
        return self._walkforward_daily_returns()

    def get_agent_returns(self, agent_name: str) -> np.ndarray:
        """Net return array for a specific agent."""
        trips = self._get_roundtrips()
        arr = np.array([t.net_return for t in trips], dtype=float)
        if len(arr) >= _MIN_ROUNDTRIPS:
            return arr
        return self._walkforward_daily_returns()

    def get_regime_conditioned_returns(self, regime: str) -> np.ndarray:
        """
        Net returns filtered to round-trips entered during *regime*.
        Falls back to portfolio returns (then walkforward) if < _MIN_ROUNDTRIPS.
        """
        trips = self._get_roundtrips()
        filtered = [t.net_return for t in trips if t.regime == regime]
        if len(filtered) >= _MIN_ROUNDTRIPS:
            return np.array(filtered, dtype=float)
        # Fallback: all round-trips
        if len(trips) >= _MIN_ROUNDTRIPS:
            return np.array([t.net_return for t in trips], dtype=float)
        return self._walkforward_daily_returns()

    # ── Internal helpers ─────────────────────────────────────────────────────

    def _get_roundtrips(self) -> List[_RoundTrip]:
        if self._roundtrips is None:
            self._roundtrips = self._build_roundtrips()
        return self._roundtrips

    def _build_roundtrips(self) -> List[_RoundTrip]:
        dec = self._load_decisions()
        exc = self._load_executions()
        if dec is None or exc is None:
            logger.info("decisions.csv ou executions.csv manquant — bootstrap depuis walkforward")
            return []

        # plan_id → (regime, is_winner, agent)
        plan_regime: Dict[str, str] = {}
        for _, row in dec.iterrows():
            pid = str(row.get("plan_id", ""))
            is_winner = bool(row.get("is_winner", False))
            if pid and is_winner:
                plan_regime[pid] = str(row.get("regime", "unknown"))

        open_positions: Dict[str, List[tuple]] = {}
        roundtrips: List[_RoundTrip] = []

        for _, row in exc.sort_values("timestamp").iterrows():
            sym = str(row.get("symbol", ""))
            side = str(row.get("side", "")).upper()
            pid = str(row.get("plan_id", ""))

            fill_px = _safe_float(row.get("avg_fill_price"))
            if fill_px is not None and fill_px > 0:
                price = fill_px
            else:
                price = _safe_float(row.get("limit_price") or row.get("last_price"))

            ts = pd.to_datetime(row.get("timestamp"), utc=True, errors="coerce")
            if not sym or not side or price is None or pd.isna(ts):
                continue

            if side == "BUY":
                raw_fill = _safe_float(row.get("avg_fill_price"))
                if raw_fill is not None and raw_fill == 0.0:
                    continue
                regime = plan_regime.get(pid, "unknown")
                open_positions.setdefault(sym, []).append((price, ts, regime))

            elif side == "SELL" and open_positions.get(sym):
                entry_price, _, entry_regime = open_positions[sym].pop(0)
                if not open_positions[sym]:
                    del open_positions[sym]
                if entry_price > 0:
                    gross = (price - entry_price) / entry_price
                    net = gross - _TC_ROUND_TRIP
                    roundtrips.append(_RoundTrip(
                        symbol=sym,
                        entry_price=entry_price,
                        exit_price=price,
                        net_return=net,
                        regime=entry_regime,
                    ))

        logger.info("ReturnBootstrapper: %d round-trips chargés", len(roundtrips))
        return roundtrips

    def _load_decisions(self) -> pd.DataFrame | None:
        if not self._decisions_path.exists():
            return None
        try:
            df = pd.read_csv(self._decisions_path)
            if "is_winner" not in df.columns:
                return None
            return df
        except Exception as exc:
            logger.warning("Impossible de lire decisions.csv: %s", exc)
            return None

    def _load_executions(self) -> pd.DataFrame | None:
        if not self._executions_path.exists():
            return None
        try:
            return pd.read_csv(self._executions_path)
        except Exception as exc:
            logger.warning("Impossible de lire executions.csv: %s", exc)
            return None

    def _walkforward_daily_returns(self) -> np.ndarray:
        """
        Converts OOS window returns from walkforward_results.csv to
        approximate daily returns using geometric mean over the window.
        """
        if not self._walkforward_path.exists():
            logger.warning("walkforward_results.csv introuvable — retour array vide")
            return np.array([0.0])
        try:
            df = pd.read_csv(self._walkforward_path)
            if "oos_return" not in df.columns:
                return np.array([0.0])

            daily_returns: List[float] = []
            for _, row in df.iterrows():
                oos_r = _safe_float(row.get("oos_return"))
                if oos_r is None:
                    continue
                # Estimate number of trading days from window dates
                try:
                    t_start = pd.to_datetime(row.get("test_start"))
                    t_end = pd.to_datetime(row.get("test_end"))
                    n_cal_days = max(1, (t_end - t_start).days)
                    n_trading_days = max(1, int(n_cal_days * 252 / 365))
                except Exception:
                    n_trading_days = 126  # ~6-month default

                daily_r = (1.0 + oos_r) ** (1.0 / n_trading_days) - 1.0
                daily_returns.append(daily_r)

            if not daily_returns:
                return np.array([0.0])
            logger.info(
                "ReturnBootstrapper: fallback walkforward — %d échantillons journaliers",
                len(daily_returns),
            )
            return np.array(daily_returns, dtype=float)
        except Exception as exc:
            logger.warning("Erreur lecture walkforward_results.csv: %s", exc)
            return np.array([0.0])


def _safe_float(val) -> float | None:
    try:
        f = float(val)
        return f if not np.isnan(f) else None
    except (TypeError, ValueError):
        return None


# ══════════════════════════════════════════════════════════════════════════════
# Part 2 — SimulationResult + MonteCarloEngine
# ══════════════════════════════════════════════════════════════════════════════

@dataclass
class SimulationResult:
    # Arrays — excluded from JSON (too large); set to None after load_json
    paths: np.ndarray | None            # shape (n_simulations, horizon_days)
    final_navs: np.ndarray | None       # shape (n_simulations,)
    final_returns: np.ndarray | None    # total returns per simulation
    # Risk metrics
    var_95: float
    var_99: float
    cvar_95: float
    prob_positive: float
    prob_sharpe_above_1: float
    prob_circuit_breaker: float
    # Return distribution
    percentiles: Dict[str, float]       # p5, p10, p25, p50, p75, p90, p95
    best_case: float
    worst_case: float
    median_return: float
    expected_return: float
    annualized_return: float
    annualized_volatility: float
    sharpe_ratio: float
    # Metadata
    regime: str
    n_simulations: int
    horizon_days: int
    timestamp: str


class MonteCarloEngine:
    """
    Vectorized Monte Carlo engine.
    Outer loop: horizon_days (90 iterations for default).
    Inner axis: n_simulations — all paths computed in parallel via numpy.
    Target: < 3s for N=10,000, horizon=90.
    """

    # CB level → kelly multiplier (level 3 → zero exposure handled separately)
    _KELLY_MULTS = {0: 1.0, 1: 0.5, 2: 0.25, 3: 0.0}

    def __init__(
        self,
        n_simulations: int = 10_000,
        horizon_days: int = 90,
        initial_nav: float = 1_000_000.0,
        gmm_regime: str = "bull_volatile",
        gmm_kelly_scale: float = 0.69,
        circuit_breaker_levels: dict | None = None,
        transaction_cost_bps: float = 9.0,
    ) -> None:
        self.n_simulations = n_simulations
        self.horizon_days = horizon_days
        self.initial_nav = initial_nav
        self.gmm_regime = gmm_regime
        self.gmm_kelly_scale = gmm_kelly_scale
        self.transaction_cost_bps = transaction_cost_bps
        # CB thresholds: {drawdown_threshold: (level, kelly_multiplier)}
        # Default mirrors DrawdownCircuitBreaker._THRESHOLDS
        self._cb_levels = circuit_breaker_levels or {
            0.04: 1,  # DD > 4% → level 1 (kelly × 0.5)
            0.06: 2,  # DD > 6% → level 2 (kelly × 0.25)
            0.08: 3,  # DD > 8% → level 3 (sell-only, no exposure)
        }

    def run(self, returns: np.ndarray) -> SimulationResult:
        """
        Bootstrap *returns* (empirical daily return array) across N×H paths.
        Returns a SimulationResult with full statistics.
        """
        if len(returns) == 0:
            returns = np.array([0.0])

        n = self.n_simulations
        h = self.horizon_days
        initial = self.initial_nav
        tc = self.transaction_cost_bps / 10_000.0

        # Pre-sample: shape (N, H) — uses global numpy random state (respects np.random.seed)
        idx = np.random.choice(len(returns), size=(n, h), replace=True)
        r_raw = returns[idx]   # shape (N, H)

        paths = np.empty((n, h), dtype=float)
        nav = np.full(n, initial, dtype=float)
        peak_nav = np.full(n, initial, dtype=float)
        cb_level = np.zeros(n, dtype=int)
        cb_ever_triggered = np.zeros(n, dtype=bool)

        # Sorted ascending: last write wins (highest level overrides lower ones)
        _cb_sorted = sorted(self._cb_levels.items(), key=lambda x: x[0])

        for t in range(h):
            r = r_raw[:, t]

            # Apply Kelly factor from current CB level (vectorized)
            kelly = self._kelly_factor_vec(cb_level)
            r_net = r * kelly - tc
            # Level 3: zero market exposure (all cash, no TC)
            r_net = np.where(cb_level >= 3, 0.0, r_net)

            nav = nav * (1.0 + r_net)
            paths[:, t] = nav

            # Update peak and drawdown
            peak_nav = np.maximum(peak_nav, nav)
            dd = np.where(peak_nav > 0, (peak_nav - nav) / peak_nav, 0.0)

            # Assign new CB level (level 3 is sticky)
            new_level = np.zeros(n, dtype=int)
            for threshold, level in _cb_sorted:
                new_level = np.where(dd > threshold, level, new_level)
            cb_level = np.where(cb_level >= 3, 3, new_level)
            cb_ever_triggered |= (cb_level >= 3)

        final_navs = paths[:, -1]
        final_returns = (final_navs - initial) / initial

        return self._build_result(paths, final_navs, final_returns, cb_ever_triggered)

    def _kelly_factor_vec(self, cb_level: np.ndarray) -> np.ndarray:
        """Vectorized kelly factor: base × level_multiplier."""
        mult = np.where(
            cb_level >= 3, 0.0,
            np.where(
                cb_level == 2, 0.25,
                np.where(cb_level == 1, 0.5, 1.0),
            ),
        )
        return self.gmm_kelly_scale * mult

    def _simulate_path(self, returns: np.ndarray) -> np.ndarray:
        """Single-path simulation — used for unit testing; run() is vectorized."""
        tc = self.transaction_cost_bps / 10_000.0
        nav = self.initial_nav
        peak_nav = self.initial_nav
        cb_level = 0
        path = np.empty(self.horizon_days)
        _cb_sorted = sorted(self._cb_levels.items(), key=lambda x: x[0], reverse=True)

        for t in range(self.horizon_days):
            r = np.random.choice(returns)
            kelly = self._apply_kelly_scale(0.0 if cb_level >= 3 else r)
            if cb_level >= 3:
                r_net = 0.0
            else:
                mult = {1: 0.5, 2: 0.25}.get(cb_level, 1.0)
                r_net = r * self.gmm_kelly_scale * mult - tc

            nav = nav * (1.0 + r_net)
            path[t] = nav

            peak_nav = max(peak_nav, nav)
            dd = (peak_nav - nav) / peak_nav if peak_nav > 0 else 0.0

            new_level = 0
            for threshold, level in _cb_sorted:
                if dd > threshold:
                    new_level = level
                    break
            cb_level = 3 if cb_level >= 3 else new_level

        return path

    def _apply_circuit_breaker(
        self, nav: float, peak_nav: float, level: int
    ) -> tuple[float, int]:
        """Returns (updated_level, new_level) given current drawdown."""
        dd = (peak_nav - nav) / peak_nav if peak_nav > 0 else 0.0
        _cb_sorted = sorted(self._cb_levels.items(), key=lambda x: x[0], reverse=True)
        new_level = 0
        for threshold, lv in _cb_sorted:
            if dd > threshold:
                new_level = lv
                break
        final_level = 3 if level >= 3 else new_level
        return nav, final_level

    def _apply_kelly_scale(self, raw_return: float) -> float:
        """Single-return Kelly scaling (used by _simulate_path)."""
        return raw_return * self.gmm_kelly_scale

    def _build_result(
        self,
        paths: np.ndarray,
        final_navs: np.ndarray,
        final_returns: np.ndarray,
        cb_ever_triggered: np.ndarray,
    ) -> SimulationResult:
        n = self.n_simulations
        h = self.horizon_days
        initial = self.initial_nav

        var_95 = float(np.percentile(final_returns, 5))
        var_99 = float(np.percentile(final_returns, 1))
        tail_95 = final_returns[final_returns <= var_95]
        cvar_95 = float(np.mean(tail_95)) if len(tail_95) > 0 else var_95

        prob_positive = float(np.mean(final_returns > 0))

        # Per-path annualized Sharpe from daily path returns
        prev = np.concatenate([np.full((n, 1), initial), paths[:, :-1]], axis=1)
        with np.errstate(invalid="ignore", divide="ignore"):
            daily_r = paths / prev - 1.0
        path_mean = np.mean(daily_r, axis=1)
        path_std = np.std(daily_r, axis=1)
        with np.errstate(invalid="ignore", divide="ignore"):
            path_sharpe = np.where(path_std > 0, path_mean / path_std * np.sqrt(252), 0.0)
        prob_sharpe_above_1 = float(np.mean(path_sharpe > 1.0))
        prob_circuit_breaker = float(np.mean(cb_ever_triggered))

        pct_vals = np.percentile(final_returns, [5, 10, 25, 50, 75, 90, 95])
        percentiles = {
            "p5":  float(pct_vals[0]), "p10": float(pct_vals[1]),
            "p25": float(pct_vals[2]), "p50": float(pct_vals[3]),
            "p75": float(pct_vals[4]), "p90": float(pct_vals[5]),
            "p95": float(pct_vals[6]),
        }

        expected_return = float(np.mean(final_returns))
        median_return = float(np.percentile(final_returns, 50))
        ann_factor = 365.0 / h
        annualized_return = float((1.0 + expected_return) ** ann_factor - 1.0)
        annualized_volatility = float(np.std(final_returns) * np.sqrt(ann_factor))
        with np.errstate(invalid="ignore", divide="ignore"):
            sharpe_ratio = (
                annualized_return / annualized_volatility
                if annualized_volatility > 1e-12
                else 0.0
            )

        return SimulationResult(
            paths=paths,
            final_navs=final_navs,
            final_returns=final_returns,
            var_95=var_95,
            var_99=var_99,
            cvar_95=cvar_95,
            prob_positive=prob_positive,
            prob_sharpe_above_1=prob_sharpe_above_1,
            prob_circuit_breaker=prob_circuit_breaker,
            percentiles=percentiles,
            best_case=float(pct_vals[6]),
            worst_case=float(pct_vals[0]),
            median_return=median_return,
            expected_return=expected_return,
            annualized_return=annualized_return,
            annualized_volatility=annualized_volatility,
            sharpe_ratio=float(sharpe_ratio),
            regime=self.gmm_regime,
            n_simulations=self.n_simulations,
            horizon_days=self.horizon_days,
            timestamp=datetime.now(timezone.utc).isoformat(),
        )


# ══════════════════════════════════════════════════════════════════════════════
# Part 3 — MonteCarloReporter
# ══════════════════════════════════════════════════════════════════════════════

class MonteCarloReporter:
    """Formats SimulationResult for Telegram, tearsheet, and JSON persistence."""

    def format_telegram(self, result: SimulationResult) -> str:
        p = result.percentiles
        r = result

        def _pct(v: float) -> str:
            sign = "+" if v >= 0 else ""
            return f"{sign}{v:.1%}"

        lines = [
            "🎲 SIMULATION MONTE CARLO — Milan Capital",
            f"Régime : {r.regime} (Kelly ×{0.69:.2f}) | Horizon : {r.horizon_days}j | N={r.n_simulations:,}",
            "",
            f"📊 Distribution des rendements ({r.horizon_days}j) :",
            f"  Pire cas   (p5)  : {_pct(p['p5'])}",
            f"  Pessimiste (p25) : {_pct(p['p25'])}",
            f"  Médiane    (p50) : {_pct(p['p50'])}",
            f"  Optimiste  (p75) : {_pct(p['p75'])}",
            f"  Meilleur   (p95) : {_pct(p['p95'])}",
            "",
            "⚠️ Risk Metrics :",
            f"  VaR 95%  : {_pct(r.var_95)} (perte max avec 95% de confiance)",
            f"  VaR 99%  : {_pct(r.var_99)}",
            f"  CVaR 95% : {_pct(r.cvar_95)} (perte moyenne au-delà de la VaR)",
            "",
            "🎯 Probabilités :",
            f"  P(rendement > 0)   : {r.prob_positive:.1%}",
            f"  P(Sharpe > 1.0)    : {r.prob_sharpe_above_1:.1%}",
            f"  P(circuit breaker) : {r.prob_circuit_breaker:.1%}",
            "",
            f"📈 Rendement attendu annualisé : {_pct(r.annualized_return)}",
            f"   Volatilité annualisée        : {_pct(r.annualized_volatility)}",
            f"   Sharpe estimé                : {r.sharpe_ratio:.2f}",
        ]
        return "\n".join(lines)

    def format_tearsheet_section(self, result: SimulationResult) -> str:
        r = result
        p = result.percentiles

        def _pct(v: float) -> str:
            sign = "+" if v >= 0 else ""
            return f"{sign}{v:.1%}"

        lines = [
            f"🎲 Monte Carlo ({r.horizon_days}j) — N={r.n_simulations:,} | Régime {r.regime}",
            f"  Médiane : {_pct(p['p50'])} | VaR 95% : {_pct(r.var_95)} | CVaR 95% : {_pct(r.cvar_95)}",
            f"  P(+) : {r.prob_positive:.1%} | P(Sharpe>1) : {r.prob_sharpe_above_1:.1%} | P(CB) : {r.prob_circuit_breaker:.1%}",
            f"  Sharpe annualisé estimé : {r.sharpe_ratio:.2f} | Vol : {_pct(r.annualized_volatility)}",
        ]
        return "\n".join(lines)

    def save_json(self, result: SimulationResult, path: str) -> None:
        """Persists scalar metrics to JSON. Large numpy arrays are excluded."""
        data = {
            "var_95": result.var_95,
            "var_99": result.var_99,
            "cvar_95": result.cvar_95,
            "prob_positive": result.prob_positive,
            "prob_sharpe_above_1": result.prob_sharpe_above_1,
            "prob_circuit_breaker": result.prob_circuit_breaker,
            "percentiles": result.percentiles,
            "best_case": result.best_case,
            "worst_case": result.worst_case,
            "median_return": result.median_return,
            "expected_return": result.expected_return,
            "annualized_return": result.annualized_return,
            "annualized_volatility": result.annualized_volatility,
            "sharpe_ratio": result.sharpe_ratio,
            "regime": result.regime,
            "n_simulations": result.n_simulations,
            "horizon_days": result.horizon_days,
            "timestamp": result.timestamp,
        }
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(data, indent=2))
        logger.info("Monte Carlo résultats sauvegardés dans %s", path)

    def load_json(self, path: str) -> SimulationResult:
        """Loads scalar metrics from JSON. Array fields are set to None."""
        data = json.loads(Path(path).read_text())
        return SimulationResult(
            paths=None,
            final_navs=None,
            final_returns=None,
            var_95=float(data["var_95"]),
            var_99=float(data["var_99"]),
            cvar_95=float(data["cvar_95"]),
            prob_positive=float(data["prob_positive"]),
            prob_sharpe_above_1=float(data["prob_sharpe_above_1"]),
            prob_circuit_breaker=float(data["prob_circuit_breaker"]),
            percentiles=dict(data["percentiles"]),
            best_case=float(data["best_case"]),
            worst_case=float(data["worst_case"]),
            median_return=float(data["median_return"]),
            expected_return=float(data["expected_return"]),
            annualized_return=float(data["annualized_return"]),
            annualized_volatility=float(data["annualized_volatility"]),
            sharpe_ratio=float(data["sharpe_ratio"]),
            regime=str(data["regime"]),
            n_simulations=int(data["n_simulations"]),
            horizon_days=int(data["horizon_days"]),
            timestamp=str(data["timestamp"]),
        )


# ── Convenience function ─────────────────────────────────────────────────────

def run_simulation(
    n_simulations: int = 10_000,
    horizon_days: int = 90,
    regime: str = "bull_volatile",
    decisions_path: str = "logs/decisions.csv",
    executions_path: str = "logs/executions.csv",
    walkforward_path: str = "logs/walkforward_results.csv",
    save_path: str | None = "logs/monte_carlo_latest.json",
    initial_nav: float = 1_000_000.0,
) -> SimulationResult:
    """
    End-to-end simulation: bootstrap real returns → MC engine → save JSON.
    Returns the SimulationResult.
    """
    bootstrapper = ReturnBootstrapper(decisions_path, executions_path, walkforward_path)
    returns = bootstrapper.get_regime_conditioned_returns(regime)

    engine = MonteCarloEngine(
        n_simulations=n_simulations,
        horizon_days=horizon_days,
        initial_nav=initial_nav,
        gmm_regime=regime,
    )
    result = engine.run(returns)

    if save_path:
        MonteCarloReporter().save_json(result, save_path)

    return result
