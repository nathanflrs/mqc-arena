#!/usr/bin/env python
"""
Télécharge les prix de TOUTES les sociétés ayant appartenu au S&P 500.

    python -m scripts.build_universe_snapshot [--start 2020-01-01]

Pourquoi l'union dans le temps et pas la liste du jour
-----------------------------------------------------
Sur 2020 → aujourd'hui, 640 sociétés ont appartenu à l'indice ; 503 y figurent
encore. Télécharger la liste actuelle supprimerait 137 trajectoires — 21 % de
l'univers, choisies précisément parce qu'elles se sont terminées d'une certaine
façon. Voir src/data/universe.py.

Ce que le script produit
------------------------
- logs/universe_snapshot/<TICKER>.parquet   un fichier par société
- logs/universe_snapshot/manifest.json      couverture réelle, chiffrée

Le manifeste est le livrable qui compte autant que les données : il dit quelles
sociétés sont ABSENTES et pourquoi le résultat doit en tenir compte. Un backtest
dont on ignore la couverture n'est pas interprétable ; un backtest dont on la
connaît reste utilisable, à condition de la publier.

Reprise
-------
Idempotent : une société déjà téléchargée est ignorée. Le script peut être
interrompu et relancé.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
import warnings
from datetime import date, datetime, timezone
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
warnings.filterwarnings("ignore")

from src.data.market_data import normalize_ohlcv          # noqa: E402
from src.data.universe import coverage_report, ever_members, sp500_at  # noqa: E402

DEFAULT_OUT = Path("logs/universe_snapshot")
CHUNK = 40          # yfinance devient instable au-delà
PAUSE = 1.5         # entre lots, pour ne pas se faire limiter


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--start", default="2020-01-01")
    ap.add_argument("--end", default=None,
                    help="défaut : aujourd'hui")
    # Destination séparée : la période de VALIDATION (2010-2019) ne doit pas
    # se mélanger à celle de conception. Deux dossiers, deux manifestes, aucune
    # confusion possible sur ce qui a servi à quoi.
    ap.add_argument("--out", default=str(DEFAULT_OUT))
    ap.add_argument("--force", action="store_true",
                    help="retélécharge même ce qui existe déjà")
    args = ap.parse_args()

    out = Path(args.out)
    start = date.fromisoformat(args.start)
    today = date.fromisoformat(args.end) if args.end else date.today()
    out.mkdir(parents=True, exist_ok=True)

    print(f"📚 Reconstitution de l'univers {start} → {today}")
    membres = sorted(ever_members(start, today, step_days=90))
    actuels = set(sp500_at(today).tickers)
    print(f"   {len(membres)} sociétés ont appartenu à l'indice")
    print(f"   {len(actuels)} y figurent encore")
    print(f"   → {len(set(membres) - actuels)} seraient perdues avec la liste du jour\n")

    import yfinance as yf

    reste = [t for t in membres
             if args.force or not (out / f"{t}.parquet").exists()]
    print(f"⬇️  {len(reste)} à télécharger "
          f"({len(membres) - len(reste)} déjà en cache)\n")

    obtenus, vides = [], []
    for i in range(0, len(reste), CHUNK):
        lot = reste[i:i + CHUNK]
        try:
            brut = yf.download(lot, start=str(start), interval="1d",
                               auto_adjust=True, progress=False,
                               group_by="ticker", threads=True)
        except Exception as exc:
            print(f"   lot {i//CHUNK + 1} en échec ({exc}) — ignoré")
            vides.extend(lot)
            continue

        for t in lot:
            try:
                df = (brut[t].copy()
                      if isinstance(brut.columns, pd.MultiIndex) else brut.copy())
                df = normalize_ohlcv(df).dropna(subset=["Close"])
                if len(df) < 60:      # trop court pour tout indicateur long
                    vides.append(t)
                    continue
                df.to_parquet(out / f"{t}.parquet")
                obtenus.append(t)
            except Exception:
                vides.append(t)

        fait = min(i + CHUNK, len(reste))
        print(f"   {fait:4d}/{len(reste)}   obtenus {len(obtenus):4d}   "
              f"sans données {len(vides):3d}")
        time.sleep(PAUSE)

    disponibles = {p.stem for p in out.glob("*.parquet")}
    rapport = coverage_report(set(membres), disponibles)

    manifest = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "start": str(start), "end": str(today),
        "n_ever_members": len(membres),
        "n_current_members": len(actuels),
        "coverage": rapport,
        "note": (
            "Les sociétés manquantes sont majoritairement des rachats — elles "
            "ont quitté l'indice par le haut, souvent avec une prime. Les "
            "exclure retire donc surtout des trajectoires favorables : le biais "
            "ne va pas nécessairement dans le sens flatteur habituel. Son sens "
            "réel dépend de la stratégie testée, et doit être discuté avec tout "
            "résultat obtenu sur cet univers."
        ),
    }
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2))

    print(f"\n{'='*66}")
    print(f"  univers réel            : {rapport['n_univers']}")
    print(f"  prix disponibles        : {rapport['n_disponibles']}")
    print(f"  manquants               : {rapport['n_manquants']}")
    print(f"  COUVERTURE              : {rapport['couverture']:.1%}")
    print(f"{'='*66}")
    print(f"\n✅ {out}/manifest.json")


if __name__ == "__main__":
    main()
