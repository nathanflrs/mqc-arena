#!/usr/bin/env python
"""
Hypothèse 01 — la qualité des bénéfices prédit-elle la performance ?

    python -m scripts.test_hypothese_01_accruals

Protocole : docs/hypothese_01_accruals.md, écrit le 2026-08-12 AVANT toute
mesure. Rien n'est modifié ici.

L'hypothèse
-----------
Une entreprise peut afficher un bénéfice sans encaisser d'argent. L'écart entre
le résultat comptable et la trésorerie réellement produite s'appelle les
régularisations :

    régularisations = (résultat net − trésorerie d'exploitation) / actif total

Deux raisons de penser qu'elles prédisent une sous-performance. Mécanique : un
bénéfice non encaissé finit par se corriger. Comportementale : le bénéfice par
action est le chiffre que tout le monde regarde, l'état des flux de trésorerie
demande du travail.

Construction
------------
**Chiffres annuels**, comme Sloan (1996). Les trois grandeurs proviennent du
même exercice : mélanger un résultat annuel et un actif trimestriel produirait
la même erreur que celle constatée sur UFPT le 2026-08-13.

**Point-in-time strict.** Un ratio n'est utilisable qu'à partir de la date de
dépôt la plus tardive des trois grandeurs qui le composent — pas de la clôture
de l'exercice. L'écart est de 30 à 90 jours, et l'ignorer donnerait accès aux
comptes avant leur publication.

**Financières et immobilières exclues** : chez une banque, la dette est
l'activité, et les régularisations n'y ont pas le même sens. Le secteur vient
de la table Wikipédia actuelle — il change rarement, mais ce n'est pas du
point-in-time et c'est dit.

**Déciles.** Long les régularisations les plus faibles (bénéfice adossé au
cash), short les plus élevées. L'hypothèse prédit un écart positif.

Périodes, fixées le 2026-08-12
------------------------------
    2011-2019  conception   — on a le droit de regarder
    2020-2023  contrôle     — un seul passage
    2024-2026  VALIDATION   — interdite jusqu'à la fin

Ce script mesure les trois d'un coup parce que rien n'a été ajusté entre-temps :
aucun seuil n'a été choisi en regardant ces données. La séparation reste
affichée pour que la lecture soit honnête.
"""
from __future__ import annotations

import io
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import requests

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
warnings.filterwarnings("ignore")

from src.analysis.agent_edge import block_bootstrap_indices  # noqa: E402
from src.data.quality import filter_universe                 # noqa: E402

FACTS = Path("logs/fundamentals_snapshot/facts.parquet")
PRIX = [Path("logs/universe_oos_2010_2019"), Path("logs/universe_snapshot")]
H = 60                      # les comptes bougent lentement : 3 mois
PAS = 5                     # une date de classement sur cinq
BLOC = H // PAS             # 12 lignes couvrent une fenêtre de 60 jours
DECILE = 0.10
N_BOOT = 2000
SEED = 20260814

PERIODES = [("2011-2019 conception", "2011-01-01", "2019-12-31"),
            ("2020-2023 contrôle",   "2020-01-01", "2023-12-31"),
            ("2024-2026 VALIDATION", "2024-01-01", "2026-12-31")]

EXCLUS = {"Financials", "Real Estate"}


def secteurs() -> dict[str, str]:
    UA = {"User-Agent": "MQC_ARENA research@milancapital.io"}
    h = requests.get("https://en.wikipedia.org/wiki/List_of_S%26P_500_companies",
                     headers=UA, timeout=30).text
    t = pd.read_html(io.StringIO(h))[0]
    return {str(r["Symbol"]).replace(".", "-"): str(r["GICS Sector"])
            for _, r in t.iterrows()}


def accruals_pit() -> pd.DataFrame:
    """Un ratio par société et par exercice, avec sa date de disponibilité."""
    f = pd.read_parquet(FACTS)

    # Les FLUX se lisent sur un exercice, le BILAN à un instant. Filtrer
    # uniformément sur "annual" éliminait tout l'actif total — 76 190 faits,
    # tous instantanés. C'est la même confusion flux/stock que celle constatée
    # sur UFPT le 2026-08-13, sous une autre forme.
    flux = f[(f["metric"].isin(["net_income", "operating_cash_flow"]))
             & (f["period_kind"] == "annual")]
    bilan = f[(f["metric"] == "assets") & (f["period_kind"] == "instant")]
    f = pd.concat([flux, bilan], ignore_index=True)

    # PREMIER dépôt, pas le dernier — le moment où l'information est devenue
    # publique.
    #
    # Prendre le dernier datait chaque exercice de sa dernière republication :
    # les entreprises reprennent les années antérieures en comparatif dans
    # chaque rapport suivant. L'exercice d'Apple clos le 2022-09-24 ressortait
    # ainsi « disponible le 2024-11-01 », deux ans trop tard, et plus aucun
    # compte ne passait le filtre de fraîcheur.
    #
    # Conséquence assumée : on retient la valeur TELLE QUE PUBLIÉE À L'ÉPOQUE,
    # pas sa version corrigée. C'est précisément ce qu'un investisseur voyait,
    # et c'est la construction habituelle de la littérature sur les
    # régularisations.
    f = (f.sort_values(["filed", "_tag_rank"])
           .groupby(["ticker", "metric", "period_end"], as_index=False).first())

    piv = f.pivot_table(index=["ticker", "period_end"], columns="metric",
                        values="value", aggfunc="first")
    dep = f.pivot_table(index=["ticker", "period_end"], columns="metric",
                        values="filed", aggfunc="first")

    besoin = ["net_income", "operating_cash_flow", "assets"]
    piv = piv.dropna(subset=[c for c in besoin if c in piv.columns])
    if not set(besoin) <= set(piv.columns):
        raise SystemExit(f"métriques manquantes : {set(besoin) - set(piv.columns)}")

    piv = piv[piv["assets"] > 0]
    piv["accruals"] = (piv["net_income"] - piv["operating_cash_flow"]) / piv["assets"]

    # Disponible seulement quand la DERNIÈRE des trois grandeurs est déposée.
    # Conversion explicite : la table croisée mêle des dates et des NaN, et
    # comparer un `date` à un `float` lève.
    dep = dep[besoin].apply(pd.to_datetime, errors="coerce")
    piv["dispo"] = dep.max(axis=1)
    return piv.reset_index()[["ticker", "period_end", "accruals", "dispo"]].dropna()


def prix() -> pd.DataFrame:
    data = {}
    for rep in PRIX:
        for p in rep.glob("*.parquet"):
            df = pd.read_parquet(p)
            if p.stem in data:
                data[p.stem] = pd.concat([data[p.stem], df])
            else:
                data[p.stem] = df
    data = {t: d[~d.index.duplicated(keep="last")].sort_index()
            for t, d in data.items()}
    data, rejets = filter_universe(data)
    print(f"   {len(data)} séries de prix ({len(rejets)} écartées pour qualité)")
    c = pd.DataFrame({t: pd.to_numeric(d["Close"], errors="coerce")
                      for t, d in data.items()})
    c.index = pd.to_datetime(c.index).tz_localize(None)
    return c.sort_index()


def main() -> None:
    print(__doc__.split("Périodes, fixées")[0])
    print("📦 Chargement")
    acc = accruals_pit()
    print(f"   {len(acc):,} ratios sur {acc.ticker.nunique()} sociétés")

    sect = secteurs()
    avant = acc.ticker.nunique()
    acc = acc[~acc.ticker.map(lambda t: sect.get(t, "")).isin(EXCLUS)]
    print(f"   {avant - acc.ticker.nunique()} sociétés financières/immobilières exclues")

    close = prix()
    fwd = np.log(close.shift(-H) / close)
    acc["dispo"] = pd.to_datetime(acc["dispo"])

    lignes = []
    for d in close.index[::PAS]:                   # une date sur cinq : suffisant
        vus = acc[acc["dispo"] <= d]
        if vus.empty:
            continue
        dernier = vus.sort_values("period_end").groupby("ticker").last()
        # Un compte de plus de deux ans n'informe plus sur l'entreprise d'aujourd'hui.
        frais = dernier[(d - pd.to_datetime(dernier["period_end"])).dt.days <= 730]
        f = fwd.loc[d].dropna()
        s = frais["accruals"]
        s = s[s.index.isin(f.index)]
        if len(s) < 50:
            continue
        n = max(1, int(len(s) * DECILE))
        rang = s.sort_values()                     # croissant : faibles d'abord
        bas = f[rang.index[:n]].mean()             # régularisations FAIBLES → long
        haut = f[rang.index[-n:]].mean()           # ÉLEVÉES → short
        lignes.append((d, bas, haut, bas - haut, len(s)))

    g = pd.DataFrame(lignes, columns=["date", "faibles", "elevees", "ecart", "n"])
    print(f"   {len(g):,} dates de classement, univers médian {g['n'].median():.0f}\n")

    rng = np.random.default_rng(SEED)
    for label, d0, d1 in PERIODES:
        p = g[(g.date >= d0) & (g.date <= d1)]
        if len(p) < 30:
            print(f"  {label:24s} échantillon insuffisant ({len(p)})")
            continue
        idx = block_bootstrap_indices(len(p), BLOC, N_BOOT, rng)
        boot = p["ecart"].to_numpy()[idx].mean(axis=1)
        lo, hi = np.percentile(boot, [2.5, 97.5])
        marque = "✅" if lo > 0 else ("❌" if hi < 0 else "—")
        print(f"  {label:24s} {len(p):4d} dates   "
              f"faibles {p['faibles'].mean():+.2%}   élevées {p['elevees'].mean():+.2%}   "
              f"écart {p['ecart'].mean():+.2%}  [{lo:+.2%}, {hi:+.2%}] {marque}")

    print(f"\n  Rappel du critère de rejet (docs/hypothese_01_accruals.md) :")
    print(f"  « rejetée si l'IC contient zéro sur la période de validation »")


if __name__ == "__main__":
    main()
