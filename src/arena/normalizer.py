# src/arena/normalizer.py
from __future__ import annotations

import dataclasses
import os
from typing import Dict, Optional, Tuple

import pandas as pd

from src.agents.base import AgentSignal


class ConfidenceNormalizer:
    """
    Normalise la confidence de chaque agent dans son propre intervalle historique
    via min-max scaling → [0, 1].

    Objectif : corriger les biais d'échelle inter-agents.
    Un agent qui émet naturellement 0.80-0.90 n'est pas structurellement
    "plus convaincu" qu'un agent qui émet 0.30-0.50 — il a juste une échelle plus haute.

    Ce que ça ne fait PAS : calibrer (conf=0.8 ne signifie pas 80% de taux de succès).
    Ce que ça fait : ramener tous les agents à la même échelle relative.

    Agents à signal constant (std=0) sont laissés inchangés — impossible de normaliser
    un intervalle dégénéré [x, x].
    """

    def __init__(self, stats: Optional[Dict[str, Tuple[float, float]]] = None):
        # stats: agent_name → (min_conf, max_conf) observés historiquement
        self._stats: Dict[str, Tuple[float, float]] = stats or {}

    @classmethod
    def from_csv(cls, path: str) -> "ConfidenceNormalizer":
        if not os.path.exists(path):
            return cls()
        try:
            df = pd.read_csv(path)
        except Exception:
            return cls()

        if "confidence" not in df.columns or "agent" not in df.columns:
            return cls()

        df["confidence"] = pd.to_numeric(df["confidence"], errors="coerce")
        stats: Dict[str, Tuple[float, float]] = {}
        for agent, grp in df.groupby("agent"):
            lo = float(grp["confidence"].min())
            hi = float(grp["confidence"].max())
            if hi > lo:
                stats[str(agent)] = (lo, hi)
        return cls(stats)

    def normalize(self, sig: AgentSignal) -> AgentSignal:
        """Retourne un signal avec confidence normalisée. Non-destructif (copie)."""
        if sig.agent_name not in self._stats:
            return sig
        lo, hi = self._stats[sig.agent_name]
        norm_conf = max(0.0, min(1.0, (sig.confidence - lo) / (hi - lo)))
        return dataclasses.replace(sig, confidence=norm_conf)

    def normalize_all(self, signals: list[AgentSignal]) -> list[AgentSignal]:
        return [self.normalize(s) for s in signals]
