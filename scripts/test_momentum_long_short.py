#!/usr/bin/env python
"""
Le momentum transversal fonctionne-t-il avec ses DEUX jambes ?

    python -m scripts.test_momentum_long_short

Pourquoi ce test
----------------
CrossSectionalMomentumAgent a été retiré le 2026-08-13 : significativement
perdant à 20 jours, −3,0 % avec un IC de [−5,0 %, −1,1 %]. Le diagnostic
n'était pas « mauvaise stratégie » mais « mauvaise implémentation », pour deux
raisons identifiées :

  1. Jegadeesh-Titman classe plusieurs centaines de titres ; l'agent en avait 14,
     donc son quartile supérieur valait 3 valeurs. Un classement sur 14 titres
     corrélés départage du bruit.
  2. Le momentum transversal tire son rendement de l'ÉCART entre le haut et le
     bas du classement. L'agent était long seulement : il gardait la
     concentration, qui coûte, et supprimait la jambe qui rapporte.

Les deux obstacles sont levés — 500 titres disponibles, et la vente à découvert
autorisée sur le compte (vérifiée le 2026-08-14).

Conception, fixée avant exécution
---------------------------------
**Score inchangé.** Rendement moyen sur 63, 126 et 252 séances, en sautant les
21 dernières pour éviter le retournement court terme. Ce sont les fenêtres de
l'agent existant, reprises telles quelles — aucun réglage.

**Déciles plutôt que quartiles.** Jegadeesh-Titman construisent des déciles ; le
quartile de l'agent était une concession à un univers de 14 titres, qui n'a plus
lieu d'être. C'est le seul paramètre qui change, et il change pour revenir à la
référence, pas pour améliorer un résultat.

**Mesure : le rendement de l'écart** — moyenne du décile haut moins moyenne du
décile bas, à 20 jours. C'est le rendement réel de la stratégie, pas un excès
contre une référence : un portefeuille long/short n'en a pas.

**Comparaison à zéro**, bootstrap par dates, 2000 tirages.

**Les deux périodes**, avec les mêmes règles. 2010-2019 n'a jamais servi à
concevoir quoi que ce soit.

Ce qui confirmerait
-------------------
L'écart est significativement positif sur les DEUX périodes, et survit aux
coûts d'un portefeuille long/short — rotation double, plus frais d'emprunt.

Ce qui infirmerait
------------------
L'écart traverse zéro, ou ne tient que sur la période de conception. Dans ce
cas l'agent reste mort, et pour une raison plus profonde que son univers.
"""
from __future__ import annotations

import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
warnings.filterwarnings("ignore")

from src.agents.momentum import MomentumConfig  # noqa: E402
from src.data.quality import filter_universe    # noqa: E402
from src.data.universe import sp500_at          # noqa: E402

H = 20
N_BOOT = 2000
SEED = 20260814
DECILE = 0.10

PERIODES = [
    ("2020-2026", Path("logs/universe_snapshot")),
    ("2010-2019", Path("logs/universe_oos_2010_2019")),
]


def scores_momentum(close: pd.DataFrame, cfg: MomentumConfig) -> pd.DataFrame:
    """
    Score de momentum, identique à celui de l'agent : moyenne des rendements
    sur chaque fenêtre, calculée jusqu'à `skip_days` avant la date.
    """
    fin = close.shift(cfg.skip_days)
    parts = [fin / close.shift(cfg.skip_days + p) - 1.0 for p in cfg.periods]
    return sum(parts) / len(parts)


def appartenance(dates: pd.DatetimeIndex, colonnes) -> pd.DataFrame:
    """Masque point-in-time : le titre appartenait-il à l'indice ce jour-là ?"""
    jalons = pd.date_range(dates.min(), dates.max(), freq="90D")
    masque = pd.DataFrame(False, index=dates, columns=colonnes)
    for i, j in enumerate(jalons):
        membres = set(sp500_at(j.date()).tickers)
        fin = jalons[i + 1] if i + 1 < len(jalons) else dates.max() + pd.Timedelta(days=1)
        tranche = (dates >= j) & (dates < fin)
        for c in colonnes:
            if c in membres:
                masque.loc[tranche, c] = True
    return masque


def analyser(label: str, snap: Path) -> dict:
    data = {p.stem: pd.read_parquet(p) for p in snap.glob("*.parquet")}
    data, rejets = filter_universe(data)
    print(f"\n{'='*74}\n  {label}   {len(data)} sociétés "
          f"({len(rejets)} écartées pour qualité)\n{'='*74}")

    close = pd.DataFrame({
        t: pd.to_numeric(df["Close"], errors="coerce") for t, df in data.items()})
    close.index = pd.to_datetime(close.index).tz_localize(None)
    close = close.sort_index()

    cfg = MomentumConfig()
    sc = scores_momentum(close, cfg)
    fwd = np.log(close.shift(-H) / close)

    dates = close.index[cfg.periods[-1] + cfg.skip_days + 5:]
    memb = appartenance(dates, close.columns)

    lignes = []
    for d in dates:
        s = sc.loc[d].where(memb.loc[d]).dropna()
        f = fwd.loc[d].dropna()
        s = s[s.index.isin(f.index)]
        if len(s) < 50:
            continue
        n = max(1, int(len(s) * DECILE))
        rang = s.sort_values(ascending=False)
        haut = f[rang.index[:n]].mean()
        bas = f[rang.index[-n:]].mean()
        lignes.append((d, haut, bas, haut - bas, len(s)))

    g = pd.DataFrame(lignes, columns=["date", "haut", "bas", "ecart", "n"])
    rng = np.random.default_rng(SEED)
    idx = rng.integers(0, len(g), size=(N_BOOT, len(g)))
    boot = g["ecart"].to_numpy()[idx].mean(axis=1)
    lo, hi = np.percentile(boot, [2.5, 97.5])

    print(f"  séances retenues            : {len(g):,}   "
          f"(univers médian {g['n'].median():.0f} titres, décile = {g['n'].median()*DECILE:.0f})")
    print(f"  décile HAUT  (long)         : {g['haut'].mean():+.2%} à {H} jours")
    print(f"  décile BAS   (short)        : {g['bas'].mean():+.2%} à {H} jours")
    print(f"  ÉCART (rendement long/short): {g['ecart'].mean():+.2%}   "
          f"IC [{lo:+.2%}, {hi:+.2%}]   "
          f"{'✅ POSITIF' if lo > 0 else ('❌ NÉGATIF' if hi < 0 else '— traverse zéro')}")
    return {"ecart": g["ecart"].mean(), "lo": lo, "hi": hi, "n": len(g)}


def main() -> None:
    print(__doc__.split("Ce qui confirmerait")[0])
    res = {}
    for label, snap in PERIODES:
        if not snap.exists():
            print(f"⚠️  {snap} absent")
            continue
        res[label] = analyser(label, snap)

    if len(res) == 2:
        print(f"\n{'='*74}\n  VERDICT\n{'='*74}")
        a, b = res["2020-2026"], res["2010-2019"]
        deux_positifs = a["lo"] > 0 and b["lo"] > 0
        print(f"  conception 2020-2026 : {a['ecart']:+.2%}  IC [{a['lo']:+.2%}, {a['hi']:+.2%}]")
        print(f"  validation 2010-2019 : {b['ecart']:+.2%}  IC [{b['lo']:+.2%}, {b['hi']:+.2%}]")
        print()
        if deux_positifs:
            # Rotation double d'un long/short, plus frais d'emprunt.
            for c in (20, 40, 60):
                print(f"    net de {c} bps d'aller-retour : "
                      f"conception {a['ecart']-c/1e4:+.2%}   "
                      f"validation {b['ecart']-c/1e4:+.2%}")
        else:
            print("  → l'écart ne tient pas sur les deux périodes.")


if __name__ == "__main__":
    main()
