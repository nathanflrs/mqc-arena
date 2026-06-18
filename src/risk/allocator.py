# src/risk/allocator.py
from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd

from src.agents.base import BaseAgent
from src.backtest.engine import BacktestEngine
from src.risk.live_scorer import LiveScorer, LiveScorerConfig

logger = logging.getLogger(__name__)


@dataclass
class AllocatorConfig:
    lookback_days: int = 126      # fenêtre rolling Sharpe (6 mois)
    base_weight: float = 0.10     # poids pour un agent au Sharpe moyen
    max_weight: float = 0.25      # plafond par agent
    min_weight: float = 0.02      # plancher — agent négatif reste dans le jeu
    cache_ttl_hours: float = 24.0 # recompute si cache plus vieux que X heures
    cache_path: str = "logs/allocator_cache.json"
    # Blending backtest ↔ live : alpha = min(1, n_trades / blend_threshold)
    blend_threshold: int = 20     # à 20 round-trips → 100% live scoring


@dataclass
class AllocationResult:
    weights: Dict[str, Dict[str, float]]   # [agent_name][symbol] -> target_weight
    best_agent: Dict[str, str]             # [symbol] -> agent_name le plus performant
    sharpes: Dict[str, Dict[str, float]]   # [agent_name][symbol] -> sharpe rolling
    computed_at: str                       # ISO UTC

    def telegram_summary(self) -> str:
        lines = ["📐 Allocation dynamique (Sharpe rolling)"]
        for sym, agent in self.best_agent.items():
            sharpe = self.sharpes.get(agent, {}).get(sym, 0.0)
            weight = self.weights.get(agent, {}).get(sym, 0.0)
            lines.append(f"  {sym} → {agent} | Sharpe={sharpe:.2f} | w={weight:.0%}")
        lines.append(f"  (calculé le {self.computed_at[:16]})")
        return "\n".join(lines)

    def to_dict(self) -> dict:
        return {
            "weights": self.weights,
            "best_agent": self.best_agent,
            "sharpes": self.sharpes,
            "computed_at": self.computed_at,
        }

    @classmethod
    def from_dict(cls, d: dict) -> AllocationResult:
        return cls(
            weights=d["weights"],
            best_agent=d["best_agent"],
            sharpes=d["sharpes"],
            computed_at=d["computed_at"],
        )


def _rolling_sharpe(equity: pd.Series, lookback: int, risk_free: float = 0.04) -> float:
    """Sharpe annualisé sur les `lookback` derniers jours de la courbe d'equity."""
    window = equity.iloc[-lookback:] if len(equity) >= lookback else equity
    returns = window.pct_change().dropna()
    std = returns.std()
    if len(returns) < 5 or std == 0 or np.isnan(std):
        return 0.0
    excess = returns - risk_free / 252
    result = float(excess.mean() / std * np.sqrt(252))
    if np.isnan(result):
        return 0.0
    # Clamp overflow (near-zero std from floating-point noise → 10^16) while
    # preserving sign and relative ordering for legitimate high-Sharpe curves.
    return float(np.clip(result, -500.0, 500.0))


def _weights_from_sharpes(
    sharpes: Dict[str, float],
    base_weight: float,
    min_weight: float,
    max_weight: float,
) -> Dict[str, float]:
    """
    Traduit un dict {agent_name: sharpe} en poids cibles.
    Logique : weight_i = base_weight * (sharpe_i / mean_positive_sharpe)
    Agents négatifs → min_weight. Clamp à [min_weight, max_weight].
    """
    positive = {a: s for a, s in sharpes.items() if s > 0}

    if not positive:
        # Tous négatifs → poids égaux au plancher
        return {a: min_weight for a in sharpes}

    mean_pos = float(np.mean(list(positive.values())))

    weights: Dict[str, float] = {}
    for agent, sharpe in sharpes.items():
        if sharpe <= 0:
            weights[agent] = min_weight
        else:
            w = base_weight * (sharpe / mean_pos)
            weights[agent] = float(np.clip(w, min_weight, max_weight))

    return weights


class DynamicAllocator:
    """
    Calcule des poids dynamiques par agent et par symbole
    en backtestant chaque agent sur une fenêtre rolling.

    Seuls les agents "OHLCV-only" sont backtestabes sans appels API externes.
    Les autres agents (Macro, Vol, Pairs, DivArb) ne sont pas inclus —
    ils continueront d'utiliser leur target_weight de config.
    """

    def __init__(self, config: AllocatorConfig | None = None):
        self.cfg = config or AllocatorConfig()

    def _load_cache(self) -> AllocationResult | None:
        path = Path(self.cfg.cache_path)
        if not path.exists():
            return None
        try:
            data = json.loads(path.read_text())
            computed_at = datetime.fromisoformat(data["computed_at"])
            now = datetime.now(timezone.utc)
            age_hours = (now - computed_at).total_seconds() / 3600
            if age_hours > self.cfg.cache_ttl_hours:
                return None
            return AllocationResult.from_dict(data)
        except Exception as e:
            logger.warning(f"Allocator cache invalide: {e}")
            return None

    def _load_walkforward_oos_sharpes(self) -> dict:
        """
        Loads avg_oos_sharpe per (agent, symbol) from walkforward_results.csv.
        Returns {} if the file doesn't exist or is unreadable.
        """
        path = Path("logs/walkforward_results.csv")
        if not path.exists():
            return {}
        try:
            df = pd.read_csv(path)
            result: dict = {}
            seen: set = set()
            for _, row in df.iterrows():
                agent = str(row.get("agent", ""))
                sym = str(row.get("symbol", ""))
                key = (agent, sym)
                if key in seen or not agent or not sym:
                    continue
                sharpe = row.get("avg_oos_sharpe")
                if pd.notna(sharpe):
                    result.setdefault(agent, {})[sym] = float(sharpe)
                    seen.add(key)
            return result
        except Exception as e:
            logger.warning("Impossible de lire walkforward_results.csv: %s", e)
            return {}

    def _save_cache(self, result: AllocationResult) -> None:
        path = Path(self.cfg.cache_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(result.to_dict(), indent=2))

    def compute(
        self,
        data: Dict[str, pd.DataFrame],
        agents: List[BaseAgent],
    ) -> AllocationResult:
        """
        Retourne l'AllocationResult depuis le cache si frais, sinon le recalcule.
        `data` : {symbol: DataFrame OHLCV}
        `agents` : liste des agents à backtester (filtrés sur OHLCV-only)
        """
        cached = self._load_cache()
        if cached is not None:
            logger.info("Allocator: cache frais utilisé (%s)", cached.computed_at[:16])
            return cached

        result = self._run(data, agents)
        self._save_cache(result)
        return result

    def _run(
        self,
        data: Dict[str, pd.DataFrame],
        agents: List[BaseAgent],
    ) -> AllocationResult:
        # ── 1. Backtest Sharpes (ou OOS Sharpes walk-forward si disponibles) ─
        wf_sharpes = self._load_walkforward_oos_sharpes()
        bt_sharpes: Dict[str, Dict[str, float]] = {}

        if wf_sharpes:
            logger.info("Allocator: walk-forward OOS Sharpes chargés — backtests ignorés")
            for agent in agents:
                bt_sharpes[agent.name] = {}
                for sym in data:
                    bt_sharpes[agent.name][sym] = round(
                        wf_sharpes.get(agent.name, {}).get(sym, 0.0), 4
                    )
        else:
            for agent in agents:
                bt_sharpes[agent.name] = {}
                for sym, df in data.items():
                    try:
                        engine = BacktestEngine(agent=agent)
                        result = engine.run(symbol=sym, df=df)
                        sharpe = _rolling_sharpe(result.equity_curve, self.cfg.lookback_days)
                        bt_sharpes[agent.name][sym] = round(sharpe, 4)
                        logger.info("  [BT] %s / %s → Sharpe=%.2f", agent.name, sym, sharpe)
                    except Exception as e:
                        logger.warning("  [BT] %s / %s → erreur: %s", agent.name, sym, e)
                        bt_sharpes[agent.name][sym] = 0.0

        # ── 2. Live Sharpes (depuis logs) ─────────────────────────────────
        live_scorer = LiveScorer(LiveScorerConfig())
        live_sharpes = live_scorer.compute_live_sharpes()

        # ── 3. Blending backtest ↔ live ────────────────────────────────────
        blended: Dict[str, Dict[str, float]] = {}

        for agent in agents:
            blended[agent.name] = {}
            for sym in data:
                bt = bt_sharpes[agent.name].get(sym, 0.0)
                live = live_sharpes.get(agent.name, {}).get(sym)

                if live is not None:
                    n = live_scorer.get_n_trades(agent.name, sym)
                    alpha = min(1.0, n / max(self.cfg.blend_threshold, 1))
                    blended_sharpe = (1.0 - alpha) * bt + alpha * live
                    logger.info(
                        "  [BLEND] %s / %s → bt=%.2f live=%.2f α=%.2f → %.2f",
                        agent.name, sym, bt, live, alpha, blended_sharpe,
                    )
                else:
                    blended_sharpe = bt

                blended[agent.name][sym] = round(blended_sharpe, 4)

        # ── 4. Poids et meilleur agent ─────────────────────────────────────
        weights: Dict[str, Dict[str, float]] = {}
        best_agent: Dict[str, str] = {}

        for sym in data:
            sharpes_for_sym = {a: blended[a].get(sym, 0.0) for a in blended}

            w_for_sym = _weights_from_sharpes(
                sharpes_for_sym,
                base_weight=self.cfg.base_weight,
                min_weight=self.cfg.min_weight,
                max_weight=self.cfg.max_weight,
            )

            for agent_name, w in w_for_sym.items():
                weights.setdefault(agent_name, {})[sym] = w

            best = max(sharpes_for_sym, key=lambda a: sharpes_for_sym[a])
            best_agent[sym] = best

        return AllocationResult(
            weights=weights,
            best_agent=best_agent,
            sharpes=blended,
            computed_at=datetime.now(timezone.utc).isoformat(),
        )
