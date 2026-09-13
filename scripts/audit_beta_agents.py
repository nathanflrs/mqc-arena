#!/usr/bin/env python
"""
Les agents historiques font-ils autre chose que détenir leurs 11 titres ?

    python -m scripts.audit_beta_agents

Écrit le 2026-09-13, après plusieurs semaines de fonctionnement où Nathan a
constaté que les agents suivaient le marché. Sources : les signaux rejoués de
logs/agent_signals.parquet (scripts/measure_agent_edge.py), le snapshot figé de
logs/backtest_cache, les facteurs Carhart de logs/factor_cache.

Pour chaque agent, un portefeuille équipondéré des titres qu'il achète chaque
jour, tenu jusqu'au lendemain (aucun regard vers l'avenir), régressé sur :

- le marché seul                  → le bêta ;
- les quatre facteurs de Carhart  → l'alpha « académique » ;
- le panier passif des 11 titres  → ce que l'agent ajoute à la simple détention
  de sa liste. C'est la référence qui compte : la liste elle-même a été choisie
  en connaissant la fin de l'histoire.

Écarts-types de Newey-West (5 retards) : les rendements quotidiens d'un même
portefeuille ne sont pas indépendants.
"""
from __future__ import annotations

import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
warnings.filterwarnings("ignore")

from src.config import WATCHLIST  # noqa: E402
from src.data.snapshot import load_snapshot  # noqa: E402

AGENTS = ["BuffettAgent", "CitadelAgent", "TrendFollowingAgent",
          "MeanReversionAgent", "CrossSectionalMomentumAgent"]


def _fit(y: pd.Series, X: pd.DataFrame):
    return sm.OLS(y, sm.add_constant(X)).fit(cov_type="HAC", cov_kwds={"maxlags": 5})


def _ann(m, name="const") -> str:
    lo, hi = m.conf_int().loc[name] * 25200
    return f"{m.params[name] * 25200:+.1f} % [{lo:+.1f} ; {hi:+.1f}]"


def main() -> None:
    data, manifest, tampered = load_snapshot("logs/backtest_cache")
    if tampered:
        raise SystemExit(f"snapshot altéré : {tampered}")
    close = pd.DataFrame({s: pd.to_numeric(data[s]["Close"], errors="coerce")
                          for s in WATCHLIST})
    fwd = close.pct_change().shift(-1)            # rendement du lendemain

    sig = pd.read_parquet("logs/agent_signals.parquet")
    sig["date"] = pd.to_datetime(sig["date"])
    f = pd.read_parquet("logs/factor_cache/factors_carhart.parquet")
    if f["Mkt-RF"].abs().mean() > 0.01:
        f = f / 100
    f.index = pd.to_datetime(f.index)
    f = f.shift(-1)                               # aligné sur le rendement du lendemain
    ew = fwd[WATCHLIST].mean(axis=1).rename("ew")

    print(f"Snapshot du {manifest.created_at[:10]} — signaux "
          f"{sig.date.min().date()} → {sig.date.max().date()}\n")
    rows, series = [], {}
    buys = sig[sig.action == "BUY"]
    for a in AGENTS:
        held = (buys[buys.agent == a]
                .pivot_table(index="date", columns="symbol", values="confidence", aggfunc="size")
                .notna().reindex(columns=WATCHLIST, fill_value=False))
        r = fwd.loc[held.index, WATCHLIST].where(held).mean(axis=1).dropna()
        series[a] = r
        d = pd.concat([r.rename("r"), f.reindex(r.index), ew.reindex(r.index)], axis=1).dropna()
        y = d.r - d.RF
        m1 = _fit(y, d[["Mkt-RF"]])
        m4 = _fit(y, d[["Mkt-RF", "SMB", "HML", "Mom"]])
        me = _fit(y, (d.ew - d.RF).rename("panier"))
        rows.append({"agent": a, "jours": len(d),
                     "bêta marché": round(m1.params["Mkt-RF"], 2),
                     "alpha 4 facteurs (an)": _ann(m4),
                     "alpha vs panier (an)": _ann(me)})

    print(pd.DataFrame(rows).set_index("agent").to_string())

    d = pd.concat([ew.rename("r"), f], axis=1).dropna().loc[sig.date.min():]
    m = _fit(d.r - d.RF, d[["Mkt-RF", "SMB", "HML", "Mom"]])
    print(f"\nPanier passif des 11 titres, alpha 4 facteurs : {_ann(m)}")

    print("\nAchats aussi faits par Buffett le même jour :")
    ref = set(map(tuple, buys[buys.agent == "BuffettAgent"][["date", "symbol"]].values))
    for a in ["CitadelAgent", "TrendFollowingAgent"]:
        own = set(map(tuple, buys[buys.agent == a][["date", "symbol"]].values))
        print(f"  {a}: {len(own & ref) / len(own):.0%} de ses {len(own)} achats")


if __name__ == "__main__":
    main()
