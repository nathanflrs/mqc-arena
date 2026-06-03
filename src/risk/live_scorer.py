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

        # Reconstruction des round-trips par symbole
        # Pour chaque symbole, on suit les BUY ouverts {symbol: (agent, entry_price, entry_date)}
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
                # Fichier pré-migration : pas de colonne is_winner → on ne peut pas scorer
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

    def compute_live_sharpes(self) -> Dict[str, Dict[str, float]]:
        """
        Retourne {agent_name: {symbol: sharpe_live}}.
        Agents avec moins de min_trades round-trips sont absents du dict.
        """
        self._load()
        result: Dict[str, Dict[str, float]] = {}

        # Regroupe par (agent, symbol)
        groups: Dict[Tuple[str, str], List[RoundTrip]] = {}
        for t in self._roundtrips:
            groups.setdefault((t.agent, t.symbol), []).append(t)

        for (agent, sym), trips in groups.items():
            if len(trips) < self.cfg.min_trades:
                continue
            sharpe = _sharpe_from_roundtrips(trips)
            result.setdefault(agent, {})[sym] = round(sharpe, 4)

        return result

    def get_n_trades(self, agent: str, symbol: str) -> int:
        """Nombre de round-trips complétés pour un agent/symbole donné."""
        self._load()
        return sum(1 for t in self._roundtrips if t.agent == agent and t.symbol == symbol)

    def get_roundtrips(self, agent: str | None = None, symbol: str | None = None) -> List[RoundTrip]:
        """Accès direct aux round-trips (pour debug/dashboard)."""
        self._load()
        return [
            t for t in self._roundtrips
            if (agent is None or t.agent == agent)
            and (symbol is None or t.symbol == symbol)
        ]
