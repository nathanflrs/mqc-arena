# src/data/universe.py
"""
Composition du S&P 500 **telle qu'elle était** à une date passée.

Le problème
-----------
Un univers construit à partir de la liste d'aujourd'hui exclut toutes les
sociétés qui en sont sorties. On mesurerait alors une stratégie sur un
échantillon dont on a retiré, par construction, une partie des trajectoires.
C'est le biais du survivant, et il invalide silencieusement un backtest.

L'ampleur, mesurée le 2026-08-13 : **114 des 505 membres de janvier 2020 ne
figurent plus dans la liste actuelle, soit 23 % de l'univers.**

La source
---------
L'historique des révisions de Wikipédia. La table des constituants y est
maintenue depuis des années ; demander la révision en vigueur à une date donnée
restitue la liste telle qu'elle était affichée ce jour-là.

C'est gratuit, vérifiable — chaque révision porte un identifiant permanent — et
suffisant pour reconstituer l'appartenance à l'indice. Ce n'est pas une source
officielle : Wikipédia peut avoir du retard sur une entrée ou une sortie, de
quelques jours en général. Cette imprécision est sans commune mesure avec les
23 % d'univers que le biais du survivant supprimerait.

Ce que ce module ne résout PAS
------------------------------
L'appartenance à l'indice n'est pas le prix. Environ **47 % des sociétés
sorties n'ont plus de données de prix** chez yfinance — Activision, Hess,
Noble, Ansys… presque toutes rachetées.

Nuance importante avant de crier au biais : une société rachetée quitte
l'indice **par le haut**, souvent avec une prime. Les exclure retire donc
surtout des trajectoires favorables, et le biais ne va pas nécessairement dans
le sens flatteur qu'on redoute d'ordinaire. Le sens réel dépend de la
stratégie testée — d'où `coverage_report()`, qui chiffre le trou plutôt que de
le supposer négligeable.
"""
from __future__ import annotations

import io
import json
import logging
import time
from dataclasses import dataclass
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Set

import pandas as pd
import requests

logger = logging.getLogger(__name__)

CACHE_DIR = Path("logs/universe_cache")
WIKI_API = "https://en.wikipedia.org/w/api.php"
WIKI_PAGE = "List_of_S&P 500 companies".replace(" ", "_")
USER_AGENT = "MQC_ARENA research/1.0 (research@milancapital.io)"

# Wikipédia refuse l'agent par défaut des bibliothèques Python (HTTP 403).
_HEADERS = {"User-Agent": USER_AGENT}

# Une révision passée ne change jamais : son contenu est figé par son
# identifiant. Le cache n'a donc pas de durée de validité.
_RATE_SLEEP = 0.3


@dataclass(frozen=True)
class UniverseSnapshot:
    """La composition de l'indice à une date, et d'où elle vient."""
    as_of: date
    tickers: List[str]
    revision_id: int
    revision_date: str      # date réelle de la révision, souvent antérieure
    source: str = "wikipedia"

    def __len__(self) -> int:
        return len(self.tickers)

    @property
    def lag_days(self) -> int:
        """
        Écart entre la date demandée et celle de la révision utilisée.

        Exposé plutôt que masqué : une révision vieille de trois semaines
        décrit un indice qui a pu changer entre-temps.
        """
        return (self.as_of - date.fromisoformat(self.revision_date[:10])).days


def _cache_path(as_of: date) -> Path:
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    return CACHE_DIR / f"sp500_{as_of.isoformat()}.json"


def _revision_at(as_of: date) -> tuple[int, str]:
    """Identifiant de la dernière révision publiée au plus tard à `as_of`."""
    time.sleep(_RATE_SLEEP)
    r = requests.get(WIKI_API, headers=_HEADERS, timeout=30, params={
        "action": "query", "prop": "revisions", "titles": WIKI_PAGE,
        "rvlimit": 1, "rvdir": "older",
        "rvstart": f"{as_of.isoformat()}T23:59:59Z",
        "rvprop": "ids|timestamp", "format": "json",
    })
    r.raise_for_status()
    page = next(iter(r.json()["query"]["pages"].values()))
    rev = page["revisions"][0]
    return int(rev["revid"]), str(rev["timestamp"])


def _tickers_from_revision(revid: int) -> List[str]:
    time.sleep(_RATE_SLEEP)
    r = requests.get(f"https://en.wikipedia.org/w/index.php?oldid={revid}",
                     headers=_HEADERS, timeout=30)
    r.raise_for_status()
    tables = pd.read_html(io.StringIO(r.text))
    if not tables:
        raise ValueError(f"aucune table dans la révision {revid}")
    t = tables[0]
    col = "Symbol" if "Symbol" in t.columns else ("Ticker symbol"
          if "Ticker symbol" in t.columns else t.columns[0])
    # Wikipédia écrit BRK.B ; les fournisseurs de prix attendent BRK-B.
    out = sorted({str(x).strip().replace(".", "-") for x in t[col]
                  if isinstance(x, str) or not pd.isna(x)})
    return [s for s in out if s and s.isascii() and len(s) <= 6]


def sp500_at(as_of: date, use_cache: bool = True) -> UniverseSnapshot:
    """
    Composition de l'indice à `as_of`, reconstituée depuis Wikipédia.

    Le résultat est mis en cache sur disque et n'expire pas : une révision
    passée est immuable.
    """
    p = _cache_path(as_of)
    if use_cache and p.exists():
        try:
            d = json.loads(p.read_text())
            return UniverseSnapshot(
                as_of=date.fromisoformat(d["as_of"]), tickers=d["tickers"],
                revision_id=d["revision_id"], revision_date=d["revision_date"])
        except Exception:
            logger.warning("cache univers illisible : %s", p)

    revid, ts = _revision_at(as_of)
    snap = UniverseSnapshot(as_of=as_of, tickers=_tickers_from_revision(revid),
                            revision_id=revid, revision_date=ts)
    try:
        p.write_text(json.dumps({
            "as_of": snap.as_of.isoformat(), "tickers": snap.tickers,
            "revision_id": snap.revision_id, "revision_date": snap.revision_date,
        }, indent=2))
    except Exception as exc:
        logger.warning("cache univers non écrit : %s", exc)
    return snap


def ever_members(start: date, end: date, step_days: int = 90) -> Set[str]:
    """
    Toutes les sociétés ayant appartenu à l'indice entre deux dates.

    C'est l'univers à télécharger : une société sortie en 2022 doit avoir ses
    prix, sans quoi les décisions la concernant avant sa sortie deviennent
    invisibles. L'échantillonnage trimestriel suffit — l'indice change d'une
    vingtaine de noms par an, et manquer un membre resté moins de trois mois
    est sans conséquence face aux 23 % que le biais du survivant retirerait.
    """
    out: Set[str] = set()
    cur = start
    while cur <= end:
        out |= set(sp500_at(cur).tickers)
        cur = date.fromordinal(cur.toordinal() + step_days)
    out |= set(sp500_at(end).tickers)
    return out


def coverage_report(tickers: Set[str], available: Set[str]) -> Dict[str, object]:
    """
    Chiffre le trou de données, au lieu de le supposer négligeable.

    À publier avec tout résultat obtenu sur cet univers. Un backtest dont on
    ignore la couverture réelle n'est pas interprétable — et un backtest dont
    on la connaît reste utilisable, à condition de la dire.
    """
    missing = sorted(tickers - available)
    n = len(tickers)
    return {
        "n_univers": n,
        "n_disponibles": len(tickers & available),
        "n_manquants": len(missing),
        "couverture": round(len(tickers & available) / n, 4) if n else 0.0,
        "manquants": missing,
    }
