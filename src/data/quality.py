# src/data/quality.py
"""
Détection des séries de prix corrompues.

Pourquoi ce module existe
-------------------------
Le 2026-08-14, le test hors échantillon de MeanReversion sur 2010-2019 affichait
un rendement moyen par signal de +0,376 % et un skew de **+43,6**. Un skew
pareil sur 35 000 observations ne décrit pas un marché : il décrit quelques
valeurs aberrantes.

Elles venaient d'un seul ticker. `TIE` (Titanium Metals) affichait des
rendements journaliers de **+758 %**, soit +197 000 % en prix, avec une série
allant de 1,40 à 33 700. À elles seules, **16 observations portaient la moitié
du rendement moyen** — +0,376 % avec, +0,185 % sans.

L'inspection de l'univers entier a montré 37 séries touchées sur 547, et le même
défaut sur 3 % de l'instantané 2020-2026. Ce sont des artefacts du fournisseur
sur des tickers disparus : symbole réattribué à un autre instrument, ou facteurs
d'ajustement cassés.

Le critère, et pourquoi celui-là
--------------------------------
Le premier réflexe — exclure les séries dont l'amplitude max/min est absurde —
s'est révélé faux : il retirait **NVDA sur 2010-2019**, dont la hausse d'un
facteur mille sur la décennie est parfaitement réelle. Un filtre qui supprime
des données correctes est pire que pas de filtre.

Le critère retenu porte sur la **fréquence des sauts impossibles**. Une action
véritable ne varie pas de ±50 % en une séance plus de quelques fois par
décennie ; une série corrompue le fait des dizaines à des centaines de fois
(CBE : 336, TIE : 197, CPWR : 98).

Un second cas doit être attrapé : `PARA` ne saute que 4 fois, mais couvre une
amplitude de 68 614×. Un saut isolé peut être réel ; conjugué à une amplitude
impossible, il signale l'ajustement cassé. D'où la règle en deux branches.

Ces seuils ont été choisis en regardant la **distribution des défauts**, jamais
leur effet sur un résultat de stratégie. Ils sont appliqués identiquement aux
deux périodes.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional

import numpy as np
import pandas as pd

# Un rendement journalier au-delà de ce seuil est possible mais rare : krach
# sur résultats, offre de rachat, valeur spéculative. Il devient suspect par sa
# répétition, pas par son existence.
JUMP_THRESHOLD = 0.50

# Au-delà, ce n'est plus une action volatile : c'est une série qui n'a pas de
# sens. Douze séries sur 547 en 2010-2019, aucune en 2020-2026.
MAX_JUMPS = 5

# Amplitude au-delà de laquelle un saut isolé cesse d'être crédible.
# NVDA sur dix ans atteint ~1 000× sans aucun saut : l'amplitude seule ne
# suffit donc pas à condamner, elle n'aggrave qu'un saut déjà constaté.
MAX_RANGE_WITH_JUMP = 1000.0


@dataclass(frozen=True)
class SeriesQuality:
    symbol: str
    n_obs: int
    n_jumps: int
    price_range: float
    ok: bool
    reason: str = ""


def assess(symbol: str, close: pd.Series) -> SeriesQuality:
    """Verdict de qualité sur une série de prix."""
    s = pd.to_numeric(close, errors="coerce").dropna()
    if len(s) < 10:
        return SeriesQuality(symbol, len(s), 0, 0.0, False, "historique trop court")

    rets = np.log(s.shift(-1) / s).dropna()
    n_jumps = int((rets.abs() > JUMP_THRESHOLD).sum())
    rng = float(s.max() / s.min()) if s.min() > 0 else float("inf")

    if n_jumps > MAX_JUMPS:
        return SeriesQuality(symbol, len(s), n_jumps, rng, False,
                             f"{n_jumps} sauts > ±{JUMP_THRESHOLD:.0%} — "
                             "ajustements cassés ou ticker réattribué")
    if n_jumps > 0 and rng > MAX_RANGE_WITH_JUMP:
        return SeriesQuality(symbol, len(s), n_jumps, rng, False,
                             f"amplitude {rng:,.0f}× avec {n_jumps} saut(s) — "
                             "série incohérente")
    return SeriesQuality(symbol, len(s), n_jumps, rng, True)


def filter_universe(
    data: Dict[str, pd.DataFrame],
    column: str = "Close",
) -> tuple[Dict[str, pd.DataFrame], list[SeriesQuality]]:
    """
    Retire les séries corrompues, et renvoie ce qui a été retiré.

    Les rejets sont retournés plutôt que journalisés : ils doivent apparaître
    dans le rapport d'un test, pas seulement dans un fichier de log. Une
    exclusion silencieuse de données est aussi dangereuse qu'une donnée fausse.
    """
    gardees: Dict[str, pd.DataFrame] = {}
    rejets: list[SeriesQuality] = []
    for sym, df in data.items():
        if column not in df.columns:
            rejets.append(SeriesQuality(sym, 0, 0, 0.0, False,
                                        f"colonne {column} absente"))
            continue
        q = assess(sym, df[column])
        if q.ok:
            gardees[sym] = df
        else:
            rejets.append(q)
    return gardees, rejets


def render_rejects(rejets: list[SeriesQuality], limit: int = 10) -> str:
    """Résumé lisible, à afficher avec tout résultat obtenu sur cet univers."""
    if not rejets:
        return "  aucune série écartée"
    lignes = [f"  {len(rejets)} série(s) écartée(s) pour qualité de données :"]
    for q in sorted(rejets, key=lambda x: -x.n_jumps)[:limit]:
        lignes.append(f"    {q.symbol:6s} {q.reason}")
    if len(rejets) > limit:
        lignes.append(f"    … et {len(rejets) - limit} autre(s)")
    return "\n".join(lignes)
