# src/data/form4_feed.py
"""
Flux des Form 4 déposés à la SEC, lu depuis l'index quotidien d'EDGAR.

Pourquoi un second client
-------------------------
`SECInsiderClient` interroge la SEC société par société : pour 11 titres,
c'était raisonnable. Pour 500, il faudrait chaque nuit 500 requêtes de
soumissions puis une par dépôt. L'index quotidien d'EDGAR liste en un seul
fichier TOUS les dépôts d'une journée : on y filtre les Form 4 des sociétés de
l'univers, et on ne télécharge que ceux-là.

Ce que ça coûte
---------------
- Premier passage : un mois de dépôts, soit quelques milliers de requêtes à
  environ six par seconde (limite de la SEC : dix). Une dizaine de minutes,
  une seule fois.
- Ensuite, seuls les jours nouveaux sont lus. Un dépôt ne change jamais : son
  analyse est mise en cache pour toujours, sous son numéro d'enregistrement.

Le piège évité
--------------
Un Form 4 apparaît dans l'index sous le CIK de la société ET sous celui de
chacun de ses déclarants. Quand le déclarant est lui-même une société de
l'univers — Berkshire déclarant ses achats d'Occidental —, une ligne porte le
CIK de Berkshire, et attribuer l'achat à Berkshire serait faux. Le CIK de
l'émetteur est donc relu dans le document lui-même : une ligne qui ne lui
correspond pas est écartée.
"""
from __future__ import annotations

import json
import logging
import re
import urllib.error
from dataclasses import asdict, dataclass, replace
from datetime import date, timedelta
from pathlib import Path
from typing import Callable, Dict, Iterator, List, Optional, Tuple

from src.data.sec_insider import InsiderTransaction, _parse_form4_xml, _sec_get

logger = logging.getLogger(__name__)

INDEX_URL = "https://www.sec.gov/Archives/edgar/daily-index/{y}/QTR{q}/form.{ymd}.idx"
ARCHIVE_URL = "https://www.sec.gov/Archives/{path}"
CACHE_DIR = Path("logs/insider_cache/form4")

# Un index absent peut être un jour férié… ou un index pas encore publié. On ne
# fige donc un index vide qu'une fois ce délai passé.
_EMPTY_INDEX_GRACE = timedelta(days=3)

_ISSUER_CIK = re.compile(rb"<issuerCik>\s*0*(\d+)\s*</issuerCik>")
_OPEN, _CLOSE = b"<ownershipDocument", b"</ownershipDocument>"


@dataclass(frozen=True)
class IndexEntry:
    form: str
    company: str
    cik: int
    date_filed: str
    path: str

    @property
    def accession(self) -> str:
        return self.path.rsplit("/", 1)[-1].removesuffix(".txt")


def parse_daily_index(text: str) -> List[IndexEntry]:
    """
    Lignes d'un `form.AAAAMMJJ.idx`. Les trois dernières colonnes (CIK, date,
    chemin) sont sans espace : on les prend par la droite, ce qui laisse au nom
    de société ses espaces.
    """
    entries: List[IndexEntry] = []
    started = False
    for line in text.splitlines():
        if not started:
            started = line.startswith("-----")
            continue
        parts = line.rsplit(None, 3)
        if len(parts) != 4:
            continue
        head, cik, filed, path = parts
        bits = head.split(None, 1)
        if len(bits) != 2:
            continue
        try:
            cik_i = int(cik)
        except ValueError:
            continue
        entries.append(IndexEntry(bits[0], bits[1].strip(), cik_i, filed, path))
    return entries


def extract_ownership_xml(raw: bytes) -> Optional[bytes]:
    """Le document XML d'un dépôt complet (.txt), ou None s'il n'y en a pas."""
    start, end = raw.find(_OPEN), raw.find(_CLOSE)
    if start < 0 or end < 0:
        return None
    return raw[start:end + len(_CLOSE)]


def _business_days(start: date, end: date) -> Iterator[date]:
    d = start
    while d <= end:
        if d.weekday() < 5:
            yield d
        d += timedelta(days=1)


class Form4Feed:
    """Achats de dirigeants déclarés à la SEC, pour un ensemble de sociétés."""

    def __init__(
        self,
        get: Callable[[str], bytes] = _sec_get,
        cache_dir: Path | str = CACHE_DIR,
        today: Optional[date] = None,
    ) -> None:
        self._get = get
        self._dir = Path(cache_dir)
        self._today = today
        self.failures = 0          # dépôts illisibles ce passage — à publier

    def _index(self, day: date) -> List[IndexEntry]:
        today = self._today or date.today()
        p = self._dir / "index" / f"{day:%Y%m%d}.json"
        if p.exists():
            return [IndexEntry(**e) for e in json.loads(p.read_text())]

        url = INDEX_URL.format(y=day.year, q=(day.month - 1) // 3 + 1, ymd=f"{day:%Y%m%d}")
        try:
            entries = [e for e in parse_daily_index(self._get(url).decode("latin-1"))
                       if e.form == "4"]
        except urllib.error.HTTPError as exc:
            if exc.code != 404:
                raise          # 403 = la SEC nous bloque : ça doit se voir
            entries = []
            if day > today - _EMPTY_INDEX_GRACE:
                return entries  # peut-être pas encore publié : ne rien figer

        # L'index du jour est encore incomplet : on ne le fige qu'une fois passé.
        if day < today:
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_text(json.dumps([asdict(e) for e in entries]))
        return entries

    def _filing(self, entry: IndexEntry) -> Tuple[Optional[int], List[InsiderTransaction]]:
        """(CIK de l'émetteur lu dans le document, achats qualifiés). Caché à vie."""
        p = self._dir / "filings" / f"{entry.accession}.json"
        if p.exists():
            d = json.loads(p.read_text())
            return d["issuer_cik"], [InsiderTransaction(**t) for t in d["transactions"]]

        xml = extract_ownership_xml(self._get(ARCHIVE_URL.format(path=entry.path)))
        issuer: Optional[int] = None
        txns: List[InsiderTransaction] = []
        if xml is not None:
            m = _ISSUER_CIK.search(xml)
            issuer = int(m.group(1)) if m else None
            txns = _parse_form4_xml(xml, ticker="")   # étiqueté à la lecture

        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps({"issuer_cik": issuer,
                                 "transactions": [asdict(t) for t in txns]}))
        return issuer, txns

    def purchases(
        self,
        cik_to_ticker: Dict[int, str],
        start: date,
        end: date,
    ) -> List[InsiderTransaction]:
        """
        Achats sur le marché par des dirigeants ou administrateurs (code P),
        déposés entre `start` et `end` inclus, pour les sociétés demandées.
        """
        seen: set[str] = set()
        out: List[InsiderTransaction] = []
        for day in _business_days(start, end):
            for e in self._index(day):
                ticker = cik_to_ticker.get(e.cik)
                if ticker is None or e.accession in seen:
                    continue
                try:
                    issuer, txns = self._filing(e)
                except Exception as exc:
                    self.failures += 1
                    logger.warning("Form 4 illisible %s — %s", e.accession, exc)
                    continue
                if issuer != e.cik:
                    continue       # ligne d'un déclarant, pas de l'émetteur
                seen.add(e.accession)
                out.extend(replace(t, ticker=ticker) for t in txns)
        return out
