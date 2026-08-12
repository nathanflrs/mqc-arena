# src/data/sec_fundamentals.py
"""
SEC EDGAR XBRL — données comptables *telles qu'elles étaient connues*.

Pourquoi ce module existe
-------------------------
Aucun agent de Milan Capital ne regardait jusqu'ici les comptes d'une
entreprise : treize agents, tous branchés sur des courbes de prix. C'est le
chaînon manquant de la thèse « avantage structurel » — sur la courbe de prix
d'une mégacap, un acteur retail n'a aucun avantage ; sur les comptes d'une
société que personne n'analyse, il peut en avoir un.

Le problème du point-in-time, et pourquoi il est décisif
--------------------------------------------------------
Les comptes publiés sont **révisés après coup**. Exemple mesuré sur Apple : le
total de bilan au 27/09/2008 valait 39,57 Md$ tel que publié le 22/07/2009, puis
36,17 Md$ après correction déposée le 25/01/2010 — un écart de 8,6 %.

Un backtest qui lit les données d'aujourd'hui pour une décision de 2009
utiliserait 36,17 Md$, un chiffre que personne ne pouvait connaître à l'époque.
C'est du look-ahead : la stratégie paraît meilleure qu'elle ne l'était, et
l'illusion ne se dissipe qu'avec de l'argent réel.

C'est précisément la raison pour laquelle six agents sont exclus du replay
(voir EXCLUDED_AGENTS dans src/backtest/system_backtest.py). EDGAR échappe à
cette limite parce que chaque fait porte sa **date de dépôt** (`filed`) : on
peut donc reconstituer exactement ce qui était connaissable à une date donnée.
`as_of()` ne renvoie jamais rien d'autre.

Coût et couverture
------------------
Gratuit, sans clé d'API. 10 387 sociétés, historique XBRL depuis ~2009 pour les
grandes et ~2011 pour les petites. Limite SEC de 10 requêtes/seconde, respectée
ici par une pause de 0,15 s et un cache disque.
"""
from __future__ import annotations

import json
import logging
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence

import pandas as pd

logger = logging.getLogger(__name__)

_CACHE_DIR = Path("logs/fundamentals_cache")
_CIK_TTL = timedelta(days=7)
# Les faits déjà déposés ne changent pas ; seules de nouvelles publications
# s'ajoutent. Un jour de cache suffit et évite de marteler EDGAR pendant la
# mise au point d'une hypothèse.
_FACTS_TTL = timedelta(hours=24)
_RATE_SLEEP = 0.15
_USER_AGENT = "MQC_ARENA research@milancapital.io"

_TICKERS_URL = "https://www.sec.gov/files/company_tickers.json"
_FACTS_URL = "https://data.sec.gov/api/xbrl/companyfacts/CIK{cik:010d}.json"

# Les entreprises n'emploient pas toutes la même étiquette XBRL pour la même
# notion — le chiffre d'affaires en compte à lui seul quatre variantes
# courantes. On essaie les étiquettes dans l'ordre et on retient la première
# qui répond, plutôt que de perdre l'entreprise faute de correspondance exacte.
CONCEPTS: Dict[str, List[str]] = {
    "revenue": [
        "RevenueFromContractWithCustomerExcludingAssessedTax",
        "RevenueFromContractWithCustomerIncludingAssessedTax",
        "Revenues",
        "SalesRevenueNet",
    ],
    "net_income": ["NetIncomeLoss", "ProfitLoss"],
    "operating_income": ["OperatingIncomeLoss"],
    "assets": ["Assets"],
    "liabilities": ["Liabilities"],
    "equity": [
        "StockholdersEquity",
        "StockholdersEquityIncludingPortionAttributableToNoncontrollingInterest",
    ],
    "cash": [
        "CashAndCashEquivalentsAtCarryingValue",
        "CashCashEquivalentsRestrictedCashAndRestrictedCashEquivalents",
    ],
    "long_term_debt": ["LongTermDebtNoncurrent", "LongTermDebt"],
    "operating_cash_flow": [
        "NetCashProvidedByUsedInOperatingActivities",
        "NetCashProvidedByUsedInOperatingActivitiesContinuingOperations",
    ],
    "shares": [
        "WeightedAverageNumberOfDilutedSharesOutstanding",
        "WeightedAverageNumberOfSharesOutstandingBasic",
        "CommonStockSharesOutstanding",
    ],
}


# Grandeurs de flux : elles se rapportent à une DURÉE (un trimestre, un an).
# Les autres sont des grandeurs de stock, mesurées à un instant (le bilan).
# Confondre les deux est l'erreur qui rend une analyse fondamentale absurde —
# voir la docstring de `as_of`.
FLOW_METRICS = frozenset({
    "revenue", "net_income", "operating_income", "operating_cash_flow", "shares",
})

# Tolérance autour des durées théoriques (91 j et 365 j) : les exercices
# comptables ne tombent pas sur des dates rondes.
_QUARTER_DAYS = (80, 100)
_ANNUAL_DAYS = (350, 380)


@dataclass(frozen=True)
class Fact:
    """Un chiffre comptable, avec la date à laquelle il est devenu public."""
    metric: str
    tag: str            # étiquette XBRL d'origine
    period_end: date
    value: float
    filed: date         # ce qui rend le point-in-time possible
    form: str           # 10-K, 10-Q, 10-K/A…
    fiscal_year: Optional[int] = None
    fiscal_period: Optional[str] = None
    # Comment la valeur a été obtenue : "instant" (bilan), "ttm" (somme de
    # quatre trimestres), "annual" (exercice complet publié tel quel).
    # Exposé pour que deux sociétés ne soient jamais comparées sur des bases
    # différentes sans qu'on puisse le voir.
    basis: str = "instant"
    n_quarters: int = 0  # nombre de trimestres sommés quand basis == "ttm"


def _period_kind(start: Optional[date], end: date) -> str:
    """
    Classe une observation : instantanée, trimestrielle, annuelle, ou cumulée.

    Les observations cumulées depuis le début d'exercice (« year-to-date », par
    exemple 272 jours) sont le piège principal : elles portent la même date de
    fin qu'un trimestre, mais couvrent trois trimestres. Les additionner à des
    trimestres compterait deux fois la même activité. On les écarte.
    """
    if start is None:
        return "instant"
    d = (end - start).days
    if _QUARTER_DAYS[0] <= d <= _QUARTER_DAYS[1]:
        return "quarter"
    if _ANNUAL_DAYS[0] <= d <= _ANNUAL_DAYS[1]:
        return "annual"
    return "ytd"


# ── Accès réseau et cache ─────────────────────────────────────────────────────

def _cache_path(name: str) -> Path:
    _CACHE_DIR.mkdir(parents=True, exist_ok=True)
    return _CACHE_DIR / name


def _cache_valid(path: Path, ttl: timedelta) -> bool:
    if not path.exists():
        return False
    age = datetime.now(timezone.utc) - datetime.fromtimestamp(
        path.stat().st_mtime, tz=timezone.utc)
    return age < ttl


def _sec_get(url: str) -> Optional[dict]:
    """GET EDGAR avec User-Agent conforme et pause anti-throttling."""
    time.sleep(_RATE_SLEEP)
    req = urllib.request.Request(url, headers={"User-Agent": _USER_AGENT})
    try:
        with urllib.request.urlopen(req, timeout=30) as resp:
            return json.loads(resp.read())
    except urllib.error.HTTPError as exc:
        # 404 = société sans données XBRL. Ce n'est pas une panne : beaucoup de
        # petites structures n'ont jamais déposé au format structuré.
        if exc.code == 404:
            logger.info("EDGAR 404 (pas de données XBRL) : %s", url)
        else:
            logger.warning("EDGAR HTTP %s : %s", exc.code, url)
        return None
    except Exception as exc:
        logger.warning("EDGAR indisponible (%s) : %s", exc, url)
        return None


def _trailing_twelve_months(metric: str, dedup: pd.DataFrame) -> Optional[Fact]:
    """
    Ramène un flux à douze mois glissants, la seule base comparable entre
    sociétés.

    Deux stratégies, dans cet ordre :

    1. **Somme des quatre derniers trimestres.** Préférée parce qu'elle intègre
       l'information la plus fraîche : un exercice annuel peut dater de onze
       mois. On vérifie que les quatre trimestres couvrent bien ~365 jours
       bout à bout — sans quoi il manque un trimestre et la somme serait
       silencieusement amputée.

    2. **Exercice annuel publié tel quel**, quand les trimestres manquent.

    Sans ce traitement, le module comparait un chiffre d'affaires trimestriel à
    un résultat net annuel — constaté sur UFPT, 101,5 M$ contre 44,9 M$, soit
    une marge nette apparente de 44 % pour un sous-traitant industriel. Les
    deux observations portaient la même date de fin ; seule la durée les
    distinguait.
    """
    quarters = dedup[dedup["period_kind"] == "quarter"].sort_values("period_end")
    if len(quarters) >= 4:
        last4 = quarters.iloc[-4:]
        start = last4.iloc[0]["period_start"]
        end = last4.iloc[-1]["period_end"]
        if start is not None:
            span = (end - start).days
            if _ANNUAL_DAYS[0] <= span <= _ANNUAL_DAYS[1]:
                return Fact(
                    metric=metric,
                    tag=str(last4.iloc[-1]["tag"]),
                    period_end=end,
                    value=float(last4["value"].sum()),
                    # La donnée n'est publique qu'une fois le DERNIER trimestre
                    # déposé : c'est cette date qui fait foi pour le
                    # point-in-time, pas la plus ancienne.
                    filed=max(last4["filed"]),
                    form=str(last4.iloc[-1]["form"]),
                    fiscal_year=last4.iloc[-1].get("fiscal_year"),
                    fiscal_period=last4.iloc[-1].get("fiscal_period"),
                    basis="ttm",
                    n_quarters=4,
                )

    annual = dedup[dedup["period_kind"] == "annual"].sort_values("period_end")
    if not annual.empty:
        row = annual.iloc[-1]
        return Fact(
            metric=metric, tag=str(row["tag"]),
            period_end=row["period_end"], value=float(row["value"]),
            filed=row["filed"], form=str(row["form"]),
            fiscal_year=row.get("fiscal_year"),
            fiscal_period=row.get("fiscal_period"),
            basis="annual",
        )
    return None


# ── Client ────────────────────────────────────────────────────────────────────

class FundamentalsClient:
    """
    Client EDGAR XBRL avec garantie point-in-time.

    Usage :
        c = FundamentalsClient()
        c.as_of("AAPL", date(2024, 3, 1))   # ce qu'on savait ce jour-là
    """

    def __init__(self, cache_dir: Optional[Path] = None) -> None:
        self._cache_dir = Path(cache_dir) if cache_dir else _CACHE_DIR
        self._cik_map: Optional[Dict[str, int]] = None

    # ── Résolution ticker → identifiant SEC ───────────────────────────────────

    def _load_cik_map(self) -> Dict[str, int]:
        if self._cik_map is not None:
            return self._cik_map
        path = _cache_path("cik_map.json")
        if _cache_valid(path, _CIK_TTL):
            try:
                self._cik_map = {k: int(v) for k, v in
                                 json.loads(path.read_text()).items()}
                return self._cik_map
            except Exception:
                pass

        raw = _sec_get(_TICKERS_URL)
        mapping: Dict[str, int] = {}
        if raw:
            for entry in raw.values():
                t = str(entry.get("ticker", "")).upper()
                if t:
                    mapping[t] = int(entry["cik_str"])
        self._cik_map = mapping
        try:
            path.write_text(json.dumps(mapping))
        except Exception as exc:
            logger.warning("cache cik_map non écrit : %s", exc)
        return mapping

    def cik(self, ticker: str) -> Optional[int]:
        return self._load_cik_map().get(ticker.upper())

    # ── Faits bruts ───────────────────────────────────────────────────────────

    def _raw_facts(self, ticker: str) -> Optional[dict]:
        cik = self.cik(ticker)
        if cik is None:
            logger.info("ticker inconnu d'EDGAR : %s", ticker)
            return None
        path = _cache_path(f"{cik:010d}_facts.json")
        if _cache_valid(path, _FACTS_TTL):
            try:
                return json.loads(path.read_text())
            except Exception:
                pass
        data = _sec_get(_FACTS_URL.format(cik=cik))
        if data is not None:
            try:
                path.write_text(json.dumps(data))
            except Exception as exc:
                logger.warning("cache facts non écrit : %s", exc)
        return data

    def facts(
        self,
        ticker: str,
        metrics: Optional[Sequence[str]] = None,
    ) -> pd.DataFrame:
        """
        Tous les faits publiés, un par ligne, avec leur date de dépôt.

        Ne filtre RIEN sur la date : c'est `as_of()` qui applique la contrainte
        point-in-time. Séparer les deux permet d'inspecter l'historique des
        révisions, ce qui est utile pour comprendre une anomalie de backtest.
        """
        raw = self._raw_facts(ticker)
        if not raw:
            return pd.DataFrame(columns=[
                "metric", "tag", "period_end", "value", "filed", "form",
                "fiscal_year", "fiscal_period"])

        gaap = raw.get("facts", {}).get("us-gaap", {})
        wanted = list(metrics) if metrics else list(CONCEPTS)
        rows: List[dict] = []

        for metric in wanted:
            # On collecte TOUTES les étiquettes de la notion, sans s'arrêter à
            # la première trouvée.
            #
            # S'arrêter était un bug, constaté sur UFPT : la société déclare
            # 25 observations sous l'étiquette prioritaire et 161 sous la
            # suivante. Retenir la première donnait un chiffre d'affaires
            # absent — un `nan` silencieux, qui dans un classement d'univers
            # aurait éliminé la société sans le moindre signal.
            #
            # C'est aussi le comportement correct sur le fond : le passage à la
            # norme ASC 606 en 2018 a fait changer d'étiquette à presque tout
            # le monde. Une société a donc légitimement `SalesRevenueNet` avant
            # 2018 et `RevenueFromContractWithCustomer…` après. Il faut les
            # deux pour couvrir l'historique — c'est `_tag_rank` qui départage
            # quand plusieurs étiquettes couvrent la même période.
            for rank, tag in enumerate(CONCEPTS.get(metric, [metric])):
                node = gaap.get(tag)
                if not node:
                    continue
                for unit_key, observations in node.get("units", {}).items():
                    if unit_key not in ("USD", "shares", "USD/shares"):
                        continue
                    for o in observations:
                        if "filed" not in o or "end" not in o:
                            continue
                        try:
                            end = date.fromisoformat(o["end"])
                            start = (date.fromisoformat(o["start"])
                                     if o.get("start") else None)
                            rows.append({
                                "metric": metric,
                                "tag": tag,
                                "_tag_rank": rank,
                                "period_start": start,
                                "period_end": end,
                                "period_kind": _period_kind(start, end),
                                "value": float(o["val"]),
                                "filed": date.fromisoformat(o["filed"]),
                                "form": str(o.get("form", "")),
                                "fiscal_year": o.get("fy"),
                                "fiscal_period": o.get("fp"),
                            })
                        except (ValueError, TypeError):
                            continue

        return pd.DataFrame(rows)

    # ── Le cœur : ce qui était connaissable à une date ─────────────────────────

    def as_of(
        self,
        ticker: str,
        on: date,
        metrics: Optional[Sequence[str]] = None,
    ) -> Dict[str, Fact]:
        """
        Renvoie, pour chaque métrique, le fait le plus récent **publié au plus
        tard le `on`**.

        Deux filtres, et les deux comptent :

        1. `filed <= on` — écarte tout ce qui n'était pas encore public. Sans
           ça, on lirait un résultat trimestriel jusqu'à 34 jours avant sa
           publication.

        2. À période comptable identique, on retient le dépôt le plus récent
           parmi ceux autorisés — jamais la version corrigée déposée plus tard.
           C'est ce qui reproduit la révision telle qu'elle a été vécue : avant
           correction on voyait l'ancien chiffre, et la stratégie doit être
           jugée sur celui-là.

        Puis, entre périodes comptables, on garde la plus récente : c'est le
        dernier état connu de l'entreprise à cette date.
        """
        df = self.facts(ticker, metrics)
        if df.empty:
            return {}

        visible = df[df["filed"] <= on]
        if visible.empty:
            return {}

        out: Dict[str, Fact] = {}
        for metric_raw, grp in visible.groupby("metric"):
            metric = str(metric_raw)
            is_flow = metric in FLOW_METRICS

            # Un flux ne se lit que sur une durée connue ; un stock ne se lit
            # qu'à un instant. On ne mélange jamais les deux registres.
            kinds = ("quarter", "annual") if is_flow else ("instant",)
            g = grp[grp["period_kind"].isin(kinds)]
            if g.empty:
                continue

            # Dépôt le plus récent pour chaque période, et à dépôt égal,
            # l'étiquette la plus prioritaire. Le tri décroissant sur
            # `_tag_rank` place le rang le plus faible (= meilleur) en dernier,
            # là où `.last()` va le chercher.
            dedup = (
                g.sort_values(["filed", "_tag_rank"], ascending=[True, False])
                 .groupby(["period_end", "period_kind"], as_index=False)
                 .last()
            )

            if not is_flow:
                row = dedup.sort_values("period_end").iloc[-1]
                out[metric] = Fact(
                    metric=metric, tag=str(row["tag"]),
                    period_end=row["period_end"], value=float(row["value"]),
                    filed=row["filed"], form=str(row["form"]),
                    fiscal_year=row.get("fiscal_year"),
                    fiscal_period=row.get("fiscal_period"),
                    basis="instant",
                )
                continue

            fact = _trailing_twelve_months(metric, dedup)
            if fact is not None:
                out[metric] = fact

        return out

    def as_of_frame(
        self,
        tickers: Iterable[str],
        on: date,
        metrics: Optional[Sequence[str]] = None,
    ) -> pd.DataFrame:
        """
        Une ligne par société, une colonne par métrique — la forme utile pour
        classer un univers à une date donnée.

        `_stale_days` mesure l'ancienneté de l'information : une société dont
        les derniers comptes datent de 400 jours n'est pas comparable à une
        autre qui vient de publier. Sans cette colonne, un classement mélange
        silencieusement du frais et du périmé.
        """
        rows = []
        for t in tickers:
            facts = self.as_of(t, on, metrics)
            if not facts:
                continue
            row: Dict[str, object] = {"ticker": t}
            for metric, f in facts.items():
                row[metric] = f.value
                row[f"{metric}_period_end"] = f.period_end
            newest = max(f.period_end for f in facts.values())
            row["_stale_days"] = (on - newest).days
            rows.append(row)
        return pd.DataFrame(rows)
