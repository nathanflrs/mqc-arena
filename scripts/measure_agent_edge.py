#!/usr/bin/env python
"""
Mesure d'edge par agent sur l'historique complet.

    python -m scripts.measure_agent_edge

Rejoue l'arène jour par jour sur le snapshot figé, collecte le signal de CHAQUE
agent (pas seulement du gagnant), et mesure le taux de succès directionnel
contre le taux de base inconditionnel, avec un intervalle bootstrap par date.

Répond à la question laissée ouverte par docs/edge_audit.md, qui manquait de
puissance statistique : 11 dates de marché contre 60 requises.

Produit : docs/agent_edge.md, logs/agent_signals.parquet
"""
from __future__ import annotations

import sys
import warnings
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
warnings.filterwarnings("ignore")

from src.agents.momentum import CrossSectionalMomentumAgent  # noqa: E402
from src.analysis.agent_edge import (  # noqa: E402
    HORIZONS, MATERIALITY, MIN_DATES, calibration_curve, compute_agent_edge,
    render_table,
)
from src.arena.arena import Arena  # noqa: E402
from src.backtest.system_backtest import EXCLUDED_AGENTS, _replayable_agents  # noqa: E402
from src.config import WATCHLIST  # noqa: E402
from src.data.regime import detect_regime  # noqa: E402
from src.data.snapshot import load_snapshot  # noqa: E402

CACHE = Path("logs/backtest_cache")
WARMUP = 300
SIGNALS_PATH = Path("logs/agent_signals.parquet")


def replay_signals(data: dict, symbols: list[str]) -> pd.DataFrame:
    """
    Rejoue l'arène et collecte tous les signaux.

    Comme dans le backtest, chaque agent ne voit que `data[:t]` et les
    classements momentum sont recalculés sur la fenêtre tronquée à chaque date
    — sans quoi l'agent verrait tout l'historique futur.
    """
    agents = _replayable_agents(data)
    momentum = next(a for a in agents if isinstance(a, CrossSectionalMomentumAgent))
    arena = Arena(agents)
    dates = data["SPY"].index

    rows = []
    for i in range(WARMUP, len(dates)):
        today = dates[i]
        window = {s: data[s].iloc[: i + 1] for s in symbols if s in data}
        if "SPY" not in window:
            continue
        regime = detect_regime(df=window["SPY"])["regime"]
        momentum.set_universe(window)

        for sym in symbols:
            df = window.get(sym)
            if df is None or len(df) < 200:
                continue
            for s in arena.run(sym, df, portfolio={}, regime=regime):
                rows.append((today, sym, s.agent_name, s.action,
                             float(s.confidence), float(s.target_weight)))

        if i % 200 == 0:
            print(f"  {today.date()}  {len(rows):,} signaux")

    if arena.failures:
        print(f"\n⚠️  {len(arena.failures)} panne(s) d'agent pendant le replay :")
        for line in arena.failure_summary():
            print(f"   • {line}")

    return pd.DataFrame(rows, columns=[
        "date", "symbol", "agent", "action", "confidence", "target_weight"])


def main() -> None:
    data, manifest, tampered = load_snapshot(CACHE)
    if tampered:
        print(f"⚠️  Snapshot altéré : {', '.join(tampered)}")
    print(f"📦 Snapshot du {manifest.created_at[:16]}")

    print("\n🔁 Replay de l'arène…")
    signals = replay_signals(data, list(WATCHLIST))
    signals.to_parquet(SIGNALS_PATH)
    print(f"✅ {len(signals):,} signaux sur {signals['date'].nunique()} séances "
          f"→ {SIGNALS_PATH}")

    edges = compute_agent_edge(signals, data, list(WATCHLIST))
    for tag in HORIZONS:
        print("\n" + render_table(edges, tag))

    # ── Rapport ───────────────────────────────────────────────────────────────
    n_dates = signals["date"].nunique()
    md = [
        "# Edge par agent — Milan Capital",
        "",
        f"*Généré le {pd.Timestamp.now().date()} — snapshot du "
        f"{manifest.created_at[:10]}.*",
        "",
        "## Méthode",
        "",
        f"L'arène est rejouée jour par jour sur **{n_dates} séances**, et le signal de "
        "chaque agent est collecté — pas seulement celui du gagnant. Aucun agent ne "
        "voit de données postérieures à sa date de décision.",
        "",
        f"Un signal est **correct** si le rendement forward dépasse ±{MATERIALITY:.2%} "
        "dans le sens annoncé (seuil de matérialité : environ la moitié d'un "
        "aller-retour IBKR large-cap).",
        "",
        "### Deux corrections par rapport à `docs/edge_audit.md`",
        "",
        "**Hypothèse nulle.** L'audit précédent testait contre une pièce équilibrée. "
        "Sur un marché haussier, un agent qui dit toujours BUY obtient bien plus de "
        "50 % sans contenir la moindre information. La référence retenue ici est le "
        "**taux de base inconditionnel** de la même action sur le même univers et la "
        "même période. La colonne `excès` est ce que l'agent apporte au-delà.",
        "",
        "**Intervalles de confiance.** L'audit précédent calculait un intervalle de "
        "Wilson sur le nombre de signaux, alors que son propre texte reconnaissait "
        "qu'un run où un agent dit BUY sur 12 actifs vaut une observation et non "
        "douze. On utilise ici un bootstrap sur les **dates** (2 000 tirages), qui "
        "conserve la corrélation entre actifs d'une même journée.",
        "",
        f"Seuil de puissance : {MIN_DATES} dates indépendantes minimum. "
        f"Atteint ({n_dates}).",
        "",
    ]
    for tag in HORIZONS:
        md += [f"## Horizon {tag}", "", "```", render_table(edges, tag), "```", ""]

    md += ["## Calibration de la confiance", "",
           "Un agent calibré a un taux de succès croissant avec la confiance qu'il "
           "émet. Une courbe plate signifie que sa confiance ne porte aucune "
           "information : l'agent peut avoir un edge global tout en étant incapable "
           "de dire *quand* il est fiable — ce qui rend toute pondération par la "
           "confiance illusoire (le blending Kelly, notamment).", ""]
    for agent in sorted(signals["agent"].unique()):
        curve = calibration_curve(signals, data, list(WATCHLIST), agent, horizon=5)
        if curve.empty or len(curve) < 2:
            continue
        md += [f"### {agent}", "", "| tranche | N | dates | conf. moyenne | taux de succès |",
               "|---|---|---|---|---|"]
        for _, r in curve.iterrows():
            md.append(f"| {r['bin']} | {r['n']} | {r['n_dates']} | "
                      f"{r['mean_confidence']:.2f} | {r['hit_rate']:.1%} |")
        md.append("")

    md += ["---", "",
           "## Limites",
           "",
           "- Les seuils des agents (RSI, ADX, fenêtres) ont été choisis en regardant "
           "ces mêmes marchés. Un edge mesuré ici est une **borne haute**, pas une "
           "promesse hors échantillon.",
           f"- Six agents sont exclus du replay ({', '.join(sorted(EXCLUDED_AGENTS))}) : "
           "leurs données ne sont pas reconstituables point-in-time.",
           "- Prix ajustés : l'historique est réécrit rétroactivement à chaque "
           "dividende. Le snapshot fige les données, mais un `--refresh` change la "
           "base de mesure.",
           ]
    Path("docs/agent_edge.md").write_text("\n".join(md))
    print("\n✅ docs/agent_edge.md")


if __name__ == "__main__":
    main()
