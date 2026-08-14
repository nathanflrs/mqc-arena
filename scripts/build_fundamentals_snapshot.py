#!/usr/bin/env python
"""
Télécharge les comptes point-in-time nécessaires à l'hypothèse 01.

    python -m scripts.build_fundamentals_snapshot

Ce qu'on récupère, et pourquoi seulement ça
-------------------------------------------
L'hypothèse teste les régularisations comptables :

    régularisations = (résultat net − trésorerie d'exploitation) / actif total

Trois grandeurs suffisent donc. On les prend une par une via `companyconcept`
plutôt que d'aspirer `companyfacts` : 4 Mo par société contre 0,05 Mo par
concept, soit 7,7 Go contre 115 Mo pour l'univers entier, à durée équivalente.

Toutes les étiquettes XBRL de chaque notion sont interrogées, pas seulement la
première qui répond. La leçon vient d'UFPT le 2026-08-13 : 25 observations sous
l'étiquette prioritaire, 161 sous la suivante, et s'arrêter à la première
produisait un `nan` silencieux qui aurait fait disparaître la société d'un
classement.

Ce qui est conservé
-------------------
Chaque fait garde sa **date de dépôt** (`filed`) et la durée de sa période.
Sans la première, aucun point-in-time possible — on lirait un résultat jusqu'à
34 jours avant sa publication. Sans la seconde, on comparerait un trimestre à
un exercice, l'erreur constatée sur UFPT (marge apparente de 44 %).

Sortie : logs/fundamentals_snapshot/facts.parquet + manifest.json
"""
from __future__ import annotations

import json
import sys
import time
import urllib.error
import urllib.request
import warnings
from datetime import date, datetime, timezone
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
warnings.filterwarnings("ignore")

from src.data.sec_fundamentals import (  # noqa: E402
    CONCEPTS, FundamentalsClient, _period_kind,
)

OUT = Path("logs/fundamentals_snapshot")
UA = {"User-Agent": "MQC_ARENA research@milancapital.io"}
RATE = 0.12                       # SEC autorise 10 req/s
CONCEPT_URL = ("https://data.sec.gov/api/xbrl/companyconcept/"
               "CIK{cik:010d}/us-gaap/{tag}.json")

# Seules notions nécessaires au calcul des régularisations.
METRIQUES = ["net_income", "operating_cash_flow", "assets"]


def univers() -> list[str]:
    """Union des deux instantanés de prix — mêmes sociétés, mêmes périodes."""
    tickers: set[str] = set()
    for rep in ("logs/universe_snapshot", "logs/universe_oos_2010_2019"):
        tickers |= {p.stem for p in Path(rep).glob("*.parquet")}
    return sorted(tickers)


def fetch_concept(cik: int, tag: str) -> list[dict]:
    time.sleep(RATE)
    try:
        with urllib.request.urlopen(
                urllib.request.Request(CONCEPT_URL.format(cik=cik, tag=tag),
                                       headers=UA), timeout=30) as r:
            data = json.loads(r.read())
    except urllib.error.HTTPError as e:
        if e.code != 404:
            print(f"      HTTP {e.code} sur {tag}")
        return []
    except Exception:
        return []

    rows = []
    for unit, obs in data.get("units", {}).items():
        if unit != "USD":
            continue
        for o in obs:
            if "filed" not in o or "end" not in o:
                continue
            try:
                end = date.fromisoformat(o["end"])
                start = date.fromisoformat(o["start"]) if o.get("start") else None
                rows.append({
                    "tag": tag,
                    "period_start": start,
                    "period_end": end,
                    "period_kind": _period_kind(start, end),
                    "value": float(o["val"]),
                    "filed": date.fromisoformat(o["filed"]),
                    "form": str(o.get("form", "")),
                })
            except (ValueError, TypeError):
                continue
    return rows


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    tickers = univers()
    print(f"📚 Univers : {len(tickers)} sociétés\n")

    client = FundamentalsClient()
    lignes, sans_cik, sans_donnees = [], [], []

    for i, t in enumerate(tickers, 1):
        cik = client.cik(t)
        if cik is None:
            sans_cik.append(t)
            continue
        trouve = False
        for metrique in METRIQUES:
            for rank, tag in enumerate(CONCEPTS[metrique]):
                for r in fetch_concept(cik, tag):
                    lignes.append({"ticker": t, "metric": metrique,
                                   "_tag_rank": rank, **r})
                    trouve = True
        if not trouve:
            sans_donnees.append(t)
        if i % 25 == 0:
            print(f"   {i:4d}/{len(tickers)}   {len(lignes):7,} faits   "
                  f"sans CIK {len(sans_cik):3d}   sans données {len(sans_donnees):3d}")

    df = pd.DataFrame(lignes)
    df.to_parquet(OUT / "facts.parquet")

    couverts = df["ticker"].nunique() if len(df) else 0
    manifest = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "n_univers": len(tickers),
        "n_couverts": couverts,
        "couverture": round(couverts / len(tickers), 4) if tickers else 0.0,
        "n_faits": len(df),
        "metriques": METRIQUES,
        "sans_cik": sans_cik,
        "sans_donnees": sans_donnees,
        "note": (
            "Chaque fait porte sa date de dépôt et la durée de sa période. "
            "Sans la première il n'y a pas de point-in-time possible ; sans la "
            "seconde on compare un trimestre à un exercice."
        ),
    }
    (OUT / "manifest.json").write_text(json.dumps(manifest, indent=2, default=str))

    print(f"\n{'='*66}")
    print(f"  sociétés couvertes : {couverts}/{len(tickers)}  "
          f"({manifest['couverture']:.1%})")
    print(f"  faits collectés    : {len(df):,}")
    print(f"  sans identifiant   : {len(sans_cik)}")
    print(f"  sans données XBRL  : {len(sans_donnees)}")
    print(f"{'='*66}\n✅ {OUT}/facts.parquet")


if __name__ == "__main__":
    main()
