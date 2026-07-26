# src/arena/normalizer.py
from __future__ import annotations

import dataclasses
import json
import os
from typing import Dict, Optional, Tuple

from src.agents.base import AgentSignal

# Chemin par défaut du fichier de stats gelées.
# Ce fichier est calculé UNE FOIS sur une fenêtre de référence et committé.
# Il ne change pas au fil des runs — un recalibrage est un acte explicite.
DEFAULT_STATS_PATH = "logs/normalizer_stats.json"


class ConfidenceNormalizer:
    """
    Normalise la confidence de chaque agent dans son propre intervalle historique
    via min-max scaling → [0, 1].

    STATIONNARITÉ : les bornes (lo, hi) sont lues depuis un fichier JSON gelé,
    pas recalculées depuis decisions.csv à chaque run. Un signal conf=0.60 de
    BuffettAgent produit le même score normalisé quel que soit le moment du replay.

    Ce que ça ne fait PAS : calibrer (conf=0.8 ne signifie pas 80% de taux de succès).
    Ce que ça fait : ramener tous les agents à la même échelle relative.

    Agents à signal constant (range=0) sont laissés inchangés.
    """

    def __init__(self, stats: Optional[Dict[str, Tuple[float, float]]] = None):
        # stats: agent_name → (lo, hi)
        self._stats: Dict[str, Tuple[float, float]] = stats or {}

    @classmethod
    def from_frozen_json(cls, path: str = DEFAULT_STATS_PATH) -> "ConfidenceNormalizer":
        """Charge les stats depuis le fichier gelé. Ne lit jamais decisions.csv."""
        if not os.path.exists(path):
            return cls()
        try:
            with open(path) as f:
                data = json.load(f)
        except Exception:
            return cls()
        agents = data.get("agents", {})
        stats: Dict[str, Tuple[float, float]] = {
            agent: (float(v["lo"]), float(v["hi"]))
            for agent, v in agents.items()
            if float(v["hi"]) > float(v["lo"])
        }
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
