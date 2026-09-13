# src/agents/insider_cluster.py
"""
InsiderBuy sur tout le S&P 500 (2026-09-13).

Même règle, mêmes seuils (InsiderBuyConfig) : au moins deux dirigeants ou
administrateurs distincts, chacun ayant acheté pour 100 000 $ ou plus sur le
marché en trente jours. Seul l'univers change.

Pourquoi ce passage
-------------------
Sur les 11 mégacaps, l'agent n'a jamais rien vu : 100 % de HOLD. Les
dirigeants de très grandes sociétés n'achètent presque pas leurs propres
actions sur le marché. Ce n'était pas un défaut de la règle, c'était un
pêcheur devant une baignoire.

Ce que l'observation mesurera d'abord : l'agent trouve-t-il des événements, et
combien ? Qu'ils rapportent se mesurera ensuite, hors échantillon, avant toute
exécution — la littérature (Lakonishok et Lee, 2001) dit que l'effet existe,
pas qu'il existe encore, ni ici.
"""
from __future__ import annotations

from collections import defaultdict
from datetime import timedelta
from typing import Dict, List, Optional

from src.agents.base import AgentSignal
from src.agents.insider_buy import InsiderBuyConfig, evaluate_cluster
from src.agents.universe_base import UniverseAgent
from src.data.form4_feed import Form4Feed

# Au-delà de 10 % de dépôts illisibles, l'agent se déclare en panne.
MAX_UNREADABLE = 0.10


class InsiderClusterAgent(UniverseAgent):
    name = "InsiderClusterAgent"

    def __init__(
        self,
        config: Optional[InsiderBuyConfig] = None,
        feed: Optional[Form4Feed] = None,
    ) -> None:
        self.cfg = config or InsiderBuyConfig()
        self._feed = feed or Form4Feed()
        self.last_stats: Dict[str, object] = {}

    def propose(self, view) -> List[AgentSignal]:
        members = view.members
        ciks = members["cik"].dropna() if "cik" in members.columns else []
        cik_to_ticker = {int(c): t for t, c in ciks.items() if t in view.ohlcv}

        end = view.as_of
        start = end - timedelta(days=self.cfg.lookback_days)
        # Fenêtre sur la date de TRANSACTION, comme l'agent historique : un
        # dépôt tardif d'un achat ancien ne fait pas un signal frais.
        txns = [t for t in self._feed.purchases(cik_to_ticker, start, end)
                if t.transaction_date >= start.isoformat()]

        # Des dépôts illisibles en nombre rendent le résultat incomplet : une
        # liste vide ne doit pas pouvoir passer pour un jour sans achat.
        fetched = getattr(self._feed, "fetches", 0)
        failed = getattr(self._feed, "failures", 0)
        if failed > MAX_UNREADABLE * max(fetched, 1):
            raise RuntimeError(f"{failed} dépôt(s) SEC illisible(s) sur {fetched} téléchargé(s) "
                               "— résultat incomplet, pas un jour calme")

        by_ticker = defaultdict(list)
        for t in txns:
            by_ticker[t.ticker].append(t)

        out = [s for s in (evaluate_cluster(self.name, tk, by_ticker[tk], self.cfg)
                           for tk in sorted(by_ticker))
               if s.action == "BUY"]

        self.last_stats = {
            "societes_suivies": len(cik_to_ticker),
            "achats_de_dirigeants": len(txns),
            "societes_avec_achats": len(by_ticker),
            "depots_illisibles": getattr(self._feed, "failures", 0),
        }
        return out
