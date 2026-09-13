# src/agents/pairs_universe.py
"""
Pairs trading sur tout le S&P 500 (2026-09-13).

Candidats : toutes les paires de titres d'une même sous-industrie GICS, soit
environ 1 450. Validation : les filtres de PairsTradingAgent, seuils
inchangés — Engle-Granger sur trois fenêtres, stabilité glissante, demi-vie,
exposant de Hurst, passages à zéro. Entrée : |z| > 2, comme l'agent historique.

Pourquoi ce passage
-------------------
L'agent historique n'a jamais passé un ordre : ses paires d'ETF sont interdites
au compte (PRIIPs), et ses paires d'actions n'ont jamais été branchées. Sur
l'univers entier, il peut enfin chercher.

Mise en garde statistique — à relire au moment de conclure
----------------------------------------------------------
Tester 1 450 paires au seuil de 5 %, c'est en déclarer environ 70
cointégrées par pur hasard, à chaque fenêtre. Les filtres successifs en
éliminent la plupart, pas toutes. Une paire validée ici est un CANDIDAT,
pas une preuve : seule la mesure hors échantillon décidera.
"""
from __future__ import annotations

from collections import defaultdict
from itertools import combinations
from typing import Dict, List, Optional, Tuple

from src.agents.base import AgentSignal
from src.agents.pairs_trading import _COINT_WINDOWS, PairSelector, PairsTradingConfig
from src.agents.universe_base import UniverseAgent


class PairsUniverseAgent(UniverseAgent):
    name = "PairsUniverseAgent"

    def __init__(self, config: Optional[PairsTradingConfig] = None) -> None:
        self.cfg = config or PairsTradingConfig()
        self.last_stats: Dict[str, object] = {}

    def candidates(self, view) -> List[Tuple[str, str]]:
        """Toutes les paires d'une même sous-industrie, parmi les titres ayant des prix."""
        groups: Dict[str, List[str]] = defaultdict(list)
        members = view.members
        for t in view.tickers:
            if t in members.index and members.at[t, "sub_industry"]:
                groups[members.at[t, "sub_industry"]].append(t)
        return [p for g in groups.values() for p in combinations(sorted(g), 2)]

    def propose(self, view) -> List[AgentSignal]:
        cfg = self.cfg
        pairs = self.candidates(view)
        selector = PairSelector(cfg)
        selector.load_prices({t: view.ohlcv[t]["Close"] for p in pairs for t in p})

        validated, out = [], []
        for a, b in pairs:
            pair = selector.validate(a, b)
            if pair is None:
                continue
            validated.append(pair)
            z = selector.compute_zscore(pair)
            if z is None or abs(z) <= cfg.zscore_entry:
                continue

            # Même dosage de conviction que PairsTradingAgent.
            conf = min(0.90, 0.60 + (abs(z) - cfg.zscore_entry) * 0.10)
            conf = min(0.93, conf + 0.03 * max(0, pair.robustness_score - cfg.min_robustness_score))
            if z < 0:      # spread anormalement bas : A bon marché face à B
                direction, long_, short_ = "long_a_short_b", a, b
            else:          # spread anormalement haut : A cher face à B
                direction, long_, short_ = "short_a_long_b", b, a
            out.append(AgentSignal(
                agent_name=self.name,
                symbol=a,
                action="BUY",
                confidence=round(conf, 3),
                target_weight=cfg.target_weight,
                reason=(f"Paire {a}/{b} : LONG {long_} / SHORT {short_}  z={z:+.2f}  "
                        f"β={pair.hedge_ratio:.3f}  demi-vie={pair.half_life or 0:.0f} j  "
                        f"score={pair.robustness_score}/{len(_COINT_WINDOWS)}"),
                meta={
                    "strategy": "market_neutral",
                    "pair": [a, b],
                    "direction": direction,
                    "zscore": z,
                    "hedge_ratio": pair.hedge_ratio,
                    "half_life": pair.half_life,
                    "hurst": pair.hurst,
                    "pvalue": pair.pvalue,
                    "sub_industry": view.members.at[a, "sub_industry"],
                },
            ))

        self.last_stats = {
            "paires_candidates": len(pairs),
            "paires_validees": len(validated),
            "validees": [f"{p.ticker_a}/{p.ticker_b}" for p in validated],
        }
        return out
