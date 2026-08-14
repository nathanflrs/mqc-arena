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

from src.agents.base import MarketState  # noqa: E402
from src.agents.cta_trend_agent import CTA_UNIVERSE, CTATrendAgent  # noqa: E402
from src.agents.momentum import CrossSectionalMomentumAgent  # noqa: E402
from src.analysis.agent_edge import (  # noqa: E402
    HORIZONS, MATERIALITY, MIN_DATES, calibration_curve, compute_agent_edge,
    render_signed_table, render_table, signed_return_edge,
)
from src.arena.arena import Arena  # noqa: E402
from src.backtest.system_backtest import EXCLUDED_AGENTS, _replayable_agents  # noqa: E402
from src.config import DATA_ONLY, WATCHLIST  # noqa: E402
from src.data.regime import detect_regime  # noqa: E402
from src.data.snapshot import load_snapshot  # noqa: E402

CACHE = Path("logs/backtest_cache")
WARMUP = 300
SIGNALS_PATH = Path("logs/agent_signals.parquet")
CTA_SIGNALS_PATH = Path("logs/cta_signals.parquet")

# Historique minimal exigé par CTATrendAgent (200j de SMA + marge).
CTA_MIN_HISTORY = 230


def replay_signals(data: dict, symbols: list[str]) -> pd.DataFrame:
    """
    Rejoue l'arène et collecte tous les signaux.

    Comme dans le backtest, chaque agent ne voit que `data[:t]` et les
    classements momentum sont recalculés sur la fenêtre tronquée à chaque date
    — sans quoi l'agent verrait tout l'historique futur.

    Deux listes distinctes, et c'est délibéré
    -----------------------------------------
    `symbols` sont les titres SUR LESQUELS on émet un signal. La fenêtre passée
    au momentum et au détecteur de régime y ajoute `DATA_ONLY` — SPY, GLD — qui
    servent de contexte de marché sans jamais être tradés.

    Confondre les deux a coûté un document entier le 2026-08-14 : SPY est passé
    de WATCHLIST à DATA_ONLY quand l'univers a été réduit aux actions, le garde
    `if "SPY" not in window` a sauté TOUTES les dates, et le script a écrit
    « 0 signaux sur 0 séances » avant de régénérer docs/agent_edge.md avec des
    tableaux vides. Sans erreur, sans code de retour non nul.
    """
    agents = _replayable_agents(data)
    momentum = next(a for a in agents if isinstance(a, CrossSectionalMomentumAgent))
    arena = Arena(agents)
    dates = data["SPY"].index
    contexte = list(dict.fromkeys(list(symbols) + list(DATA_ONLY)))

    rows = []
    for i in range(WARMUP, len(dates)):
        today = dates[i]
        window = {s: data[s].iloc[: i + 1] for s in contexte if s in data}
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


def replay_cta_signals(data: dict) -> pd.DataFrame:
    """
    Rejoue CTATrendAgent sur son propre univers de 6 ETF.

    Pourquoi une fonction séparée
    -----------------------------
    CTA ne passe pas par l'arène — le runner l'appelle directement sur un chemin
    parallèle (runner.py, boucle « CTA TREND »). Il échappait donc à la mesure
    d'edge, alors qu'il porte la plus grosse allocation de risque du fonds :
    `max_gross_cta_pct = 0.60` du NAV, dans une catégorie explicitement exclue
    du budget net-long (risk/manager.py). Douze agents mesurés se partageaient
    ~21 % du portefeuille pendant qu'un treizième, jamais mesuré, disposait
    seul d'une autorisation à 60 % de brut.

    Rien ne justifiait cette exclusion : contrairement aux six agents écartés
    du replay pour de vraies raisons point-in-time (révisions FRED, absence
    d'archive de news, dates de dépôt SEC), CTA ne lit que des prix d'ETF.

    portfolio={} à chaque date
    --------------------------
    On mesure le **signal directionnel émis**, pas la gestion de position.
    L'agent traduit `direction='flat'` en SELL lorsqu'une position longue est
    ouverte (fermeture) : compter ce SELL comme un pari baissier fausserait la
    mesure. Avec un portefeuille vide, BUY signifie « je veux être long » et
    SELL « je veux être short », ce qui est exactement ce que juge
    `label_success`. C'est aussi la convention du replay de l'arène ci-dessus.
    """
    agent = CTATrendAgent()
    dates = data["SPY"].index
    rows = []

    for i in range(WARMUP, len(dates)):
        today = dates[i]
        for sym in CTA_UNIVERSE:
            df = data.get(sym)
            if df is None:
                continue
            window = df.iloc[: i + 1]
            if len(window) < CTA_MIN_HISTORY:
                continue
            state = MarketState(
                symbol=sym,
                price=float(window["Close"].iloc[-1]),
                timestamp=str(window.index[-1]),
            )
            s = agent.generate_signal(state, portfolio={}, regime=None, data=window)
            rows.append((today, sym, s.agent_name, s.action,
                         float(s.confidence), float(s.target_weight)))

        if i % 200 == 0:
            print(f"  {today.date()}  {len(rows):,} signaux CTA")

    return pd.DataFrame(rows, columns=[
        "date", "symbol", "agent", "action", "confidence", "target_weight"])


def main() -> None:
    data, manifest, tampered = load_snapshot(CACHE)
    if tampered:
        print(f"⚠️  Snapshot altéré : {', '.join(tampered)}")
    print(f"📦 Snapshot du {manifest.created_at[:16]}")

    print("\n🔁 Replay de l'arène…")
    signals = replay_signals(data, list(WATCHLIST))

    # Un replay vide n'est jamais un résultat : c'est une panne de plomberie.
    # Sans ce garde-fou, le script écrivait des tableaux vides par-dessus les
    # mesures publiées et sortait avec un code 0 (incident du 2026-08-14).
    if signals.empty or signals["date"].nunique() < MIN_DATES:
        raise SystemExit(
            f"❌ replay de l'arène vide ou trop court "
            f"({len(signals)} signaux, {signals['date'].nunique() if len(signals) else 0} "
            f"séances, minimum {MIN_DATES}). docs/agent_edge.md n'est PAS réécrit.\n"
            f"   Vérifier que WATCHLIST et DATA_ONLY sont couverts par le snapshot.")

    signals.to_parquet(SIGNALS_PATH)
    print(f"✅ {len(signals):,} signaux sur {signals['date'].nunique()} séances "
          f"→ {SIGNALS_PATH}")

    edges = compute_agent_edge(signals, data, list(WATCHLIST))
    for tag in HORIZONS:
        print("\n" + render_table(edges, tag))

    # ── CTA : univers propre, donc taux de base propre ────────────────────────
    print("\n🔁 Replay CTA (univers 6 ETF, hors arène)…")
    cta_signals = replay_cta_signals(data)
    cta_signals.to_parquet(CTA_SIGNALS_PATH)
    print(f"✅ {len(cta_signals):,} signaux sur "
          f"{cta_signals['date'].nunique()} séances → {CTA_SIGNALS_PATH}")

    cta_edges = compute_agent_edge(cta_signals, data, list(CTA_UNIVERSE))
    cta_signed = signed_return_edge(cta_signals, data, list(CTA_UNIVERSE))
    for tag in HORIZONS:
        print("\n" + render_table(cta_edges, tag))
        print(render_signed_table(cta_signed, tag))

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
        "### Trois corrections successives",
        "",
        "**Hypothèse nulle.** L'audit précédent (`docs/edge_audit.md`) testait contre "
        "une pièce équilibrée. Sur un marché haussier, un agent qui dit toujours BUY "
        "obtient bien plus de 50 % sans contenir la moindre information. La référence "
        "retenue ici est le **taux de base inconditionnel** de la même action sur le "
        "même univers et la même période. La colonne `excès` est ce que l'agent "
        "apporte au-delà.",
        "",
        "**Corrélation entre actifs.** L'audit précédent calculait un intervalle de "
        "Wilson sur le nombre de signaux, alors que son propre texte reconnaissait "
        "qu'un run où un agent dit BUY sur 12 actifs vaut une observation et non "
        "douze. On regroupe donc par **date** avant de bootstrapper.",
        "",
        "**Chevauchement des fenêtres — correction du 2026-08-14.** Regrouper par date "
        "ne suffisait pas. Un rendement à 20 jours mesuré chaque séance partage 19 "
        "jours avec le précédent : tirer les dates indépendamment revenait à compter "
        "la même information vingt fois, et divisait l'intervalle par la racine d'un "
        "effectif fictif. Les intervalles ci-dessous viennent d'un **bootstrap par "
        "blocs mobiles** de longueur H (2 000 tirages), qui préserve cette dépendance.",
        "",
        "Deux résultats publiés auparavant n'y ont pas survécu : le momentum "
        "transversal « significativement perdant » à H20, et le rendement par signal "
        "du CTA à H20, seul intervalle strictement positif du document. Les deux "
        "traversent désormais zéro. **Aucun résultat n'a été renforcé par la "
        "correction** — c'est le signe attendu quand on cesse de surestimer sa "
        "propre information.",
        "",
        f"Seuil de puissance : {MIN_DATES} dates minimum. Atteint ({n_dates}) — mais "
        "le bootstrap par blocs rappelle que ces séances ne valent pas autant "
        "d'observations indépendantes.",
        "",
    ]
    for tag in HORIZONS:
        md += [f"## Horizon {tag}", "", "```", render_table(edges, tag), "```", ""]

    # ── Section CTA ───────────────────────────────────────────────────────────
    cta_dates = cta_signals["date"].nunique()
    md += [
        "## CTATrendAgent — univers séparé",
        "",
        "CTA ne passe pas par l'arène : le runner l'appelle sur un chemin "
        "parallèle. Il était pour cette seule raison absent de toute mesure "
        "d'edge, alors qu'il porte la plus grosse allocation de risque du fonds "
        "— `max_gross_cta_pct = 60 %` du NAV, dans une catégorie explicitement "
        "exclue du budget net-long.",
        "",
        f"**Les tableaux ci-dessous ne sont pas comparables à ceux de l'arène.** "
        f"Le taux de base est calculé sur l'univers CTA ({', '.join(CTA_UNIVERSE)}), "
        "dont la dérive inconditionnelle n'a rien à voir avec celle des mégacaps. "
        f"Replay sur {cta_dates} séances, portefeuille vide à chaque date : on "
        "mesure le signal directionnel émis, pas la gestion de position.",
        "",
    ]
    for tag in HORIZONS:
        md += [f"### Horizon {tag}", "", "```",
               render_table(cta_edges, tag), "",
               render_signed_table(cta_signed, tag), "```", ""]

    md += [
        "### Pourquoi deux tableaux",
        "",
        "Le taux de réussite suppose que tous les succès se valent. C'est faux "
        "pour un suiveur de tendance, qui a classiquement raison 35-40 % du temps "
        "et gagne quand même parce que ses gains dépassent largement ses pertes. "
        "Juger un CTA au seul taux de réussite reviendrait à le condamner sur le "
        "mauvais critère — d'où la mesure du rendement par signal, avec le skew "
        "et le ratio gain/perte qui rendent la forme du gain visible.",
        "",
        "La colonne `passif` est la référence honnête : le rendement d'un dollar "
        "simplement investi long sur le même univers et la même période. Une "
        "espérance positive mais inférieure à cette référence ne crée pas de "
        "valeur par unité d'exposition.",
        "",
        "### Le vol targeting ne s'est jamais déclenché",
        "",
        f"Sur les {len(cta_signals[cta_signals['action'] != 'HOLD']):,} signaux "
        "directionnels du replay, **100 % sortent exactement au plafond de "
        "15 %** (écart-type des poids : `2.8e-17`, la constante machine).",
        "",
        "La cause est arithmétique. Le poids vaut `min(vol_target / vol, "
        "max_position)` = `min(0.10 / vol, 0.15)` : le plafond ne cède que si la "
        "volatilité annualisée dépasse **66.7 %**. Le maximum jamais observé sur "
        "les six ETF en cinq ans est 61.1 % (QQQ), et UUP plafonne à 15.1 %.",
        "",
        "Conséquence : UUP (vol médiane 6.5 %) reçoit le même poids que QQQ "
        "(18.6 %), soit environ trois fois plus de risque sur le second — "
        "l'inverse exact de ce que le vol targeting est censé produire. La "
        "fonctionnalité annoncée dans le docstring de l'agent (« vol targeting, "
        "style Winton / Man AHL ») est inerte.",
        "",
    ]

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
           f"- {len(EXCLUDED_AGENTS)} agents sont exclus du replay de l'arène "
           f"({', '.join(sorted(EXCLUDED_AGENTS))}) : leurs données ne sont pas "
           "reconstituables point-in-time. CTATrendAgent, lui, ne lit que des "
           "prix d'ETF — son absence des mesures antérieures était un oubli, pas "
           "une impossibilité ; il est désormais mesuré ci-dessus.",
           "- Prix ajustés : l'historique est réécrit rétroactivement à chaque "
           "dividende. Le snapshot fige les données, mais un `--refresh` change la "
           "base de mesure.",
           f"- L'univers de l'arène compte {len(WATCHLIST)} titres. Les ETF en ont "
           "été retirés (un fonds de fonds ne teste pas la sélection de titres) et "
           f"servent désormais de contexte seul : {', '.join(DATA_ONLY)}. Les "
           "effectifs `N` ne sont donc pas comparables à ceux des versions "
           "antérieures de ce document.",
           "- Onze titres ne suffisent à établir aucun edge. Ce document sert à "
           "**réfuter**, pas à valider : un agent qui n'y ressort pas est écarté, "
           "un agent qui y ressort demande une confirmation sur l'univers élargi "
           "(S&P 500, `logs/universe_snapshot`).",
           ]
    Path("docs/agent_edge.md").write_text("\n".join(md))
    print("\n✅ docs/agent_edge.md")


if __name__ == "__main__":
    main()
