#!/usr/bin/env python
"""
L'avantage de MeanReversion est-il un effet de régime ?

    python -m scripts.test_regime_hypothesis

L'hypothèse
-----------
Le test hors échantillon du 2026-08-14 a montré que l'avantage de rendement
mesuré sur 2020-2026 ne se reproduit pas sur 2010-2019. L'explication la plus
simple : 2020-2026 a offert des rebonds d'ampleur inhabituelle — creux de mars
2020, puis 2022 — que la décennie précédente n'avait pas. L'agent capterait
alors un **régime**, pas une inefficience persistante.

Le test, spécifié AVANT exécution
---------------------------------
Chaque signal est classé selon la baisse du marché (SPY) depuis son plus haut
glissant sur un an, au jour du signal :

    0-5 %   marché proche de ses sommets
    5-10 %  correction ordinaire
    10-20 % correction marquée
    > 20 %  marché baissier

On compare ensuite, dans chaque tranche et pour les deux périodes, le rendement
de l'agent à celui d'un achat sans condition sur le même univers.

**Confirme l'hypothèse** : l'avantage se concentre dans les tranches de forte
baisse, et ces tranches sont plus fréquentes en 2020-2026.

**Infirme l'hypothèse** : l'avantage est réparti uniformément entre régimes.
L'explication serait alors ailleurs, et resterait à trouver.

Ce découpage est fixé une fois pour toutes. Le déplacer après avoir vu les
résultats reviendrait à chercher jusqu'à trouver ce qui arrange.
"""
from __future__ import annotations

import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
warnings.filterwarnings("ignore")

from src.analysis.agent_edge import block_bootstrap_indices  # noqa: E402
from src.data.quality import filter_universe                 # noqa: E402

HORIZON = 20            # celui où l'écart entre périodes est le plus net
TRANCHES = [(0.00, 0.05, "0-5 %   sommets"),
            (0.05, 0.10, "5-10 %  correction"),
            (0.10, 0.20, "10-20 % marquée"),
            (0.20, 1.00, "> 20 %  baissier")]

PERIODES = [
    ("2020-2026", Path("logs/universe_snapshot"),
     Path("logs/mr_sp500_signals.parquet")),
    ("2010-2019", Path("logs/universe_oos_2010_2019"),
     Path("logs/mr_oos_2010_2019_signals.parquet")),
]


def drawdown_marche(depuis: str, jusqu_a: str) -> pd.Series:
    """Baisse de SPY depuis son plus haut glissant sur un an."""
    import yfinance as yf
    spy = yf.download("SPY", start=depuis, end=jusqu_a, interval="1d",
                      auto_adjust=True, progress=False)
    c = spy["Close"]
    if isinstance(c, pd.DataFrame):
        c = c.iloc[:, 0]
    c = pd.to_numeric(c, errors="coerce").dropna()
    c.index = pd.to_datetime(c.index).tz_localize(None)
    return (c.rolling(252, min_periods=60).max() - c) / c.rolling(
        252, min_periods=60).max()


def rendements_forward(snap: Path, horizon: int) -> dict[str, pd.Series]:
    data = {p.stem: pd.read_parquet(p) for p in snap.glob("*.parquet")}
    data, _ = filter_universe(data)
    out = {}
    for t, df in data.items():
        c = pd.to_numeric(df["Close"], errors="coerce")
        c.index = pd.to_datetime(c.index).tz_localize(None)
        out[t] = np.log(c.shift(-horizon) / c)
    return out


def analyser(nom: str, snap: Path, signaux: Path) -> pd.DataFrame:
    sig = pd.read_parquet(signaux)
    sig["date"] = pd.to_datetime(sig["date"]).dt.tz_localize(None)
    fwd = rendements_forward(snap, HORIZON)

    dd = drawdown_marche(str(sig["date"].min().date()),
                         str((sig["date"].max() + pd.Timedelta(days=40)).date()))

    lignes = []
    for _, r in sig.iterrows():
        s = fwd.get(r["symbol"])
        if s is None:
            continue
        v = s.get(r["date"], np.nan)
        d = dd.get(r["date"], np.nan)
        if np.isfinite(v) and np.isfinite(d):
            lignes.append((r["date"], r["symbol"], float(v), float(d)))
    ag = pd.DataFrame(lignes, columns=["date", "symbol", "fwd", "dd"])

    # Référence : le rendement inconditionnel de l'univers, aux mêmes dates et
    # dans la même tranche de régime. Comparer à une moyenne globale mélangerait
    # l'effet de régime et l'effet d'agent.
    base_rows = []
    for t, s in fwd.items():
        x = s.dropna()
        x = x[x.index.isin(dd.index)]
        for d_, v_ in x.items():
            base_rows.append((d_, float(v_), float(dd.get(d_, np.nan))))
    base = pd.DataFrame(base_rows, columns=["date", "fwd", "dd"]).dropna()

    res = []
    for lo, hi, label in TRANCHES:
        a = ag[(ag.dd >= lo) & (ag.dd < hi)]
        b = base[(base.dd >= lo) & (base.dd < hi)]
        if len(a) < 50:
            res.append((label, len(a), np.nan, np.nan, np.nan))
            continue
        # Bootstrap par blocs de HORIZON dates, comme partout ailleurs : les
        # rendements à 20 jours mesurés quotidiennement se chevauchent.
        per = a.groupby("date")["fwd"].agg(["sum", "size"])
        rng = np.random.default_rng(20260814)
        idx = block_bootstrap_indices(len(per), HORIZON, 2000, rng)
        tot, cnt = per["sum"].to_numpy(), per["size"].to_numpy()
        boot = tot[idx].sum(1) / cnt[idx].sum(1)
        lo_ic, hi_ic = np.percentile(boot - b.fwd.mean(), [2.5, 97.5])
        res.append((label, len(a), a.fwd.mean() - b.fwd.mean(), lo_ic, hi_ic))

    df = pd.DataFrame(res, columns=["tranche", "n", "exces", "ic_lo", "ic_hi"])
    df["part"] = df["n"] / df["n"].sum()
    print(f"\n{'='*78}\n  {nom}   ({len(ag):,} signaux)\n{'='*78}")
    print(f"  {'régime de marché':22s}{'signaux':>9s}{'part':>8s}"
          f"{'excès vs base':>15s}{'IC 95 %':>22s}")
    for _, r in df.iterrows():
        if np.isnan(r.exces):
            print(f"  {r.tranche:22s}{r.n:>9,}{r.part:>8.1%}"
                  f"{'—':>15s}{'échantillon insuffisant':>22s}")
            continue
        sig_mark = " ✅" if r.ic_lo > 0 else (" ❌" if r.ic_hi < 0 else "")
        print(f"  {r.tranche:22s}{r.n:>9,}{r.part:>8.1%}{r.exces:>+15.2%}"
              f"{f'[{r.ic_lo:+.2%}, {r.ic_hi:+.2%}]':>22s}{sig_mark}")
    return df


def main() -> None:
    print(__doc__.split("Ce découpage")[0])
    tables = {}
    for nom, snap, sigs in PERIODES:
        if not sigs.exists():
            print(f"⚠️  {sigs} absent — lancer d'abord le replay")
            continue
        tables[nom] = analyser(nom, snap, sigs)

    if len(tables) == 2:
        a, b = tables["2020-2026"], tables["2010-2019"]
        print(f"\n{'='*78}\n  VERDICT SUR L'HYPOTHÈSE DE RÉGIME\n{'='*78}")
        print(f"  {'régime':22s}{'part 2020-26':>14s}{'part 2010-19':>14s}"
              f"{'excès 20-26':>13s}{'excès 10-19':>13s}")
        for i, r in a.iterrows():
            rb = b.iloc[i]
            ea = "—" if np.isnan(r.exces) else f"{r.exces:+.2%}"
            eb = "—" if np.isnan(rb.exces) else f"{rb.exces:+.2%}"
            print(f"  {r.tranche:22s}{r.part:>14.1%}{rb.part:>14.1%}"
                  f"{ea:>13s}{eb:>13s}")
        print()
        stress_a = a[a.tranche.str.startswith((">", "10-20"))]["part"].sum()
        stress_b = b[b.tranche.str.startswith((">", "10-20"))]["part"].sum()
        print(f"  Part des signaux en marché tendu (baisse > 10 %) :")
        print(f"     2020-2026 : {stress_a:.1%}")
        print(f"     2010-2019 : {stress_b:.1%}")


if __name__ == "__main__":
    main()
