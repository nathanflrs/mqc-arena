#!/usr/bin/env python
"""
Rejoue MeanReversionAgent sur l'univers S&P 500 point-in-time.

    python -m scripts.replay_meanreversion_sp500

Le test décisif du verdict 2 (docs/verdicts_agents.md). La prédiction a été
écrite AVANT ce script :

    « Sur 100 titres ou plus, l'intervalle doit exclure zéro à 20 jours.
      S'il s'évanouit en s'élargissant, c'était du bruit. »

Trois exigences, sans lesquelles le résultat ne vaudrait rien
------------------------------------------------------------

**Seuils inchangés.** RSI < 35, prix sous Bollinger 2σ, volume > 1,2× la
moyenne. Aucun réglage. Toucher un seuil en regardant ce résultat le
détruirait — c'est le mécanisme qui a produit `# assoupli de 0.90 -> 0.85`
dans buffett.py.

**Appartenance point-in-time.** À chaque date, l'univers est celui de l'indice
CE JOUR-LÀ, pas celui d'aujourd'hui. Sans quoi on retire 21 % des trajectoires.

**Équivalence vérifiée.** Les indicateurs sont vectorisés pour tenir en
mémoire — 1 500 séances × 500 titres. Le script PROUVE d'abord que sa version
vectorisée donne les mêmes valeurs que l'agent, sur un échantillon tiré au
hasard. Sans cette preuve, on testerait autre chose que MeanReversion.
"""
from __future__ import annotations

import json
import random
import sys
import warnings
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
warnings.filterwarnings("ignore")

from src.agents.base import MarketState                              # noqa: E402
from src.agents.mean_reversion import (                              # noqa: E402
    MeanReversionAgent, MeanReversionConfig, _bollinger, _rsi,
)
from src.analysis.agent_edge import (                                # noqa: E402
    HORIZONS, compute_agent_edge, render_signed_table, render_table,
    signed_return_edge,
)
from src.data.universe import sp500_at                               # noqa: E402

SNAP = Path("logs/universe_snapshot")
OUT = Path("logs/mr_sp500_signals.parquet")
WARMUP = 60          # assez pour RSI 14 + Bollinger 20 + volume 20
MEMBERSHIP_STEP = 90


# ── Indicateurs vectorisés ────────────────────────────────────────────────────

def indicateurs(df: pd.DataFrame, cfg: MeanReversionConfig) -> pd.DataFrame:
    """
    RSI, Bollinger et volume relatif sur toute la série, en une passe.

    Chaque valeur à la date t n'utilise que des données jusqu'à t : les
    fenêtres glissantes de pandas sont causales par construction. C'est
    l'équivalence exacte avec l'agent qui est vérifiée plus bas, pas supposée.
    """
    close = pd.to_numeric(df["Close"], errors="coerce")
    vol = pd.to_numeric(df["Volume"], errors="coerce")

    delta = close.diff()
    gain = delta.clip(lower=0).rolling(cfg.rsi_period).mean()
    perte = (-delta.clip(upper=0)).rolling(cfg.rsi_period).mean()
    rs = gain / perte.replace(0, np.inf)
    rsi = 100 - (100 / (1 + rs))

    sma = close.rolling(cfg.bb_period).mean()
    sigma = close.rolling(cfg.bb_period).std()
    bb_bas = sma - cfg.bb_std * sigma

    # L'agent compare le volume du jour à la moyenne des 20 séances
    # PRÉCÉDENTES, celle du jour exclue : vol.iloc[-21:-1].
    vol_moy = vol.shift(1).rolling(20).mean()

    return pd.DataFrame({
        "close": close, "rsi": rsi, "bb_bas": bb_bas,
        "vol_ratio": vol / vol_moy,
    })


def verifier_equivalence(data: dict, cfg: MeanReversionConfig, n: int = 40) -> None:
    """Compare la version vectorisée à l'agent lui-même, sur des points tirés."""
    agent = MeanReversionAgent(cfg)
    rng = random.Random(20260813)
    symbols = [s for s in data if len(data[s]) > 400]
    ecarts = 0
    testes = 0

    for _ in range(n):
        s = rng.choice(symbols)
        df = data[s]
        i = rng.randrange(300, len(df))
        fenetre = df.iloc[: i + 1]

        sig = agent.generate_signal(
            MarketState(symbol=s, price=float(fenetre["Close"].iloc[-1]),
                        timestamp=str(fenetre.index[-1])),
            portfolio={}, regime="bull", data=fenetre)
        attendu = sig.action == "BUY"

        ind = indicateurs(fenetre, cfg).iloc[-1]
        obtenu = bool(ind["rsi"] < cfg.rsi_threshold
                      and ind["close"] < ind["bb_bas"]
                      and ind["vol_ratio"] > cfg.volume_ratio)

        testes += 1
        if attendu != obtenu:
            ecarts += 1
            print(f"   écart sur {s} à {fenetre.index[-1].date()} : "
                  f"agent={attendu} vectorisé={obtenu}")

    print(f"   {testes - ecarts}/{testes} points identiques")
    if ecarts:
        raise SystemExit(
            "❌ La version vectorisée diffère de l'agent — le replay testerait "
            "autre chose que MeanReversion. Arrêt.")
    print("   ✅ équivalence prouvée\n")


# ── Appartenance point-in-time ────────────────────────────────────────────────

def table_appartenance(dates: pd.DatetimeIndex) -> dict:
    """date de séance → ensemble des membres de l'indice ce jour-là."""
    jalons = {}
    d = dates[0].date()
    fin = dates[-1].date()
    while d <= fin:
        jalons[d] = set(sp500_at(d).tickers)
        d = date.fromordinal(d.toordinal() + MEMBERSHIP_STEP)
    jalons[fin] = set(sp500_at(fin).tickers)

    cles = sorted(jalons)
    out, j = {}, 0
    for ts in dates:
        dd = ts.date()
        while j + 1 < len(cles) and cles[j + 1] <= dd:
            j += 1
        out[ts] = jalons[cles[j]]
    return out


def main() -> None:
    cfg = MeanReversionConfig()
    manifest = json.loads((SNAP / "manifest.json").read_text())

    print("📦 Chargement de l'univers")
    data = {}
    for p in sorted(SNAP.glob("*.parquet")):
        df = pd.read_parquet(p)
        if len(df) > WARMUP + 60:
            data[p.stem] = df
    print(f"   {len(data)} sociétés   couverture "
          f"{manifest['coverage']['couverture']:.1%}\n")

    print("🔬 Vérification de l'équivalence vectorisée")
    verifier_equivalence(data, cfg)

    ref = max(data.values(), key=len).index
    dates = ref[ref >= ref[WARMUP]]
    print(f"📅 Appartenance point-in-time sur {len(dates)} séances")
    appartenance = table_appartenance(dates)

    print("🔁 Calcul des signaux")
    ind = {s: indicateurs(df, cfg) for s, df in data.items()}

    lignes = []
    for k, ts in enumerate(dates):
        membres = appartenance[ts]
        for s in membres:
            t = ind.get(s)
            if t is None or ts not in t.index:
                continue
            r = t.loc[ts]
            if (r["rsi"] < cfg.rsi_threshold and r["close"] < r["bb_bas"]
                    and r["vol_ratio"] > cfg.volume_ratio):
                lignes.append((ts, s, "MeanReversionAgent", "BUY", 0.7, 0.08))
        if k % 200 == 0:
            print(f"   {ts.date()}   {len(lignes):,} signaux")

    sig = pd.DataFrame(lignes, columns=["date", "symbol", "agent", "action",
                                        "confidence", "target_weight"])
    sig.to_parquet(OUT)
    print(f"\n✅ {len(sig):,} signaux sur {sig['date'].nunique()} dates → {OUT}\n")

    symbols = sorted(data)
    edges = compute_agent_edge(sig, data, symbols)
    signed = signed_return_edge(sig, data, symbols)
    for tag in HORIZONS:
        print(render_table(edges, tag))
        print(render_signed_table(signed, tag))
        print()

    print("=" * 70)
    print("PRÉDICTION ÉCRITE AVANT LE TEST (docs/verdicts_agents.md) :")
    print("  « sur 100 titres ou plus, l'IC doit exclure zéro à H20 »")
    h20 = next(e for e in edges if e.horizon == "H20")
    verdict = ("TENUE — l'intervalle exclut zéro"
               if h20.is_significant and h20.excess > 0
               else "NON TENUE — l'effet ne survit pas à l'élargissement")
    print(f"\n  IC H20 : [{h20.ci_lo:+.1%}, {h20.ci_hi:+.1%}]   sur {h20.n_dates} dates")
    print(f"  → {verdict}")
    print("=" * 70)


if __name__ == "__main__":
    main()
