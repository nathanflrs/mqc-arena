#!/usr/bin/env python
"""
Rejoue le mécanisme d'agrégation sur les décisions déjà enregistrées.

    python -m scripts.replay_consensus [chemin/decisions.csv]

Ne modifie rien. Répond à une seule question : qu'aurait fait l'arène agrégée
là où l'arène gagnant-emporte-tout a réellement décidé ?

Les signaux de TOUS les agents sont enregistrés dans `logs/decisions.csv`, pas
seulement ceux du vainqueur — c'est ce qui rend ce replay possible sans
relancer la moindre stratégie. Les prix ne sont pas rejoués : on compare des
DÉCISIONS, pas des rendements. Sur 13 séances, aucun rendement ne serait
interprétable de toute façon.
"""
from __future__ import annotations

import sys
from collections import Counter
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.agents.base import AgentSignal   # noqa: E402
from src.arena.consensus import aggregate  # noqa: E402

DEFAUT = Path("logs/decisions.csv")


def charger(p: Path) -> pd.DataFrame:
    d = pd.read_csv(p)
    ts = pd.to_datetime(d["timestamp"], format="mixed", utc=True, errors="coerce")
    perdues = int(ts.isna().sum())
    if perdues:
        print(f"⚠️  {perdues} ligne(s) sans horodatage lisible, écartées")
    d = d[ts.notna()].copy()
    d["jour"] = ts[ts.notna()].dt.date
    return d


def main() -> None:
    chemin = Path(sys.argv[1]) if len(sys.argv) > 1 else DEFAUT
    if not chemin.exists():
        raise SystemExit(f"introuvable : {chemin}")

    d = charger(chemin)
    print(f"📦 {len(d):,} signaux — {d.jour.nunique()} séances "
          f"({d.jour.min()} → {d.jour.max()})\n")

    anciennes, nouvelles = [], []
    for (jour, sym), g in d.groupby(["jour", "symbol"], sort=True):
        gagnant = g[g["is_winner"] == True]           # noqa: E712
        ancienne = gagnant.iloc[0]["action"] if len(gagnant) else "—"

        sigs = [AgentSignal(agent_name=r["agent"], symbol=sym,
                            action=r["action"], confidence=float(r["confidence"]),
                            target_weight=float(r["target_weight"]))
                for _, r in g.iterrows()]
        c = aggregate(sigs)

        anciennes.append((jour, sym, ancienne))
        nouvelles.append((jour, sym, c.action if c else "—",
                          c.target_weight if c else 0.0,
                          c.n_speaking if c else 0,
                          c.n_eligible if c else len(sigs),
                          c.score if c else 0.0))

    a = pd.DataFrame(anciennes, columns=["jour", "symbol", "ancienne"])
    n = pd.DataFrame(nouvelles, columns=["jour", "symbol", "nouvelle", "poids",
                                         "parlants", "eligibles", "score"])
    m = a.merge(n, on=["jour", "symbol"])

    def agir(x):
        return x in ("BUY", "SELL")

    print("═" * 70)
    print("  ANCIEN (gagnant emporte tout)  vs  NOUVEAU (agrégation)")
    print("═" * 70)
    print(f"  décisions d'agir — ancien  : {m.ancienne.map(agir).sum():4d} "
          f"sur {len(m)} occasions")
    print(f"  décisions d'agir — nouveau : {m.nouvelle.map(agir).sum():4d} "
          f"sur {len(m)} occasions")
    print()

    croise = pd.crosstab(m.ancienne, m.nouvelle,
                         rownames=["ancien"], colnames=["nouveau"])
    print(croise.to_string())
    print()

    identiques = (m.ancienne == m.nouvelle).sum()
    print(f"  décisions identiques : {identiques}/{len(m)} "
          f"({identiques/len(m):.0%})")

    change = m[m.ancienne != m.nouvelle]
    if len(change):
        print(f"\n  {len(change)} divergence(s). Répartition :")
        for (av, ap), k in Counter(zip(change.ancienne, change.nouvelle)).most_common():
            print(f"    {av:5s} → {ap:5s}  ×{k}")

    act = m[m.nouvelle.map(agir)]
    if len(act):
        print(f"\n  Quand le nouveau mécanisme agit :")
        print(f"    poids médian      : {act.poids.median():.3f} "
              f"(base 0.100)")
        print(f"    agents qui parlent: {act.parlants.median():.0f} "
              f"sur {act.eligibles.median():.0f}")
        print(f"    score médian      : {act.score.median():+.2f}")

    print(f"\n  Exposition totale demandée :")
    print(f"    ancien  : {m.ancienne.map(agir).sum() * 0.10:6.2f} × NAV "
          f"(poids fixe 0.10 par décision)")
    print(f"    nouveau : {act.poids.sum():6.2f} × NAV")


if __name__ == "__main__":
    main()
