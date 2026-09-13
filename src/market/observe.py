# src/market/observe.py
"""
Le passage nocturne du fonds en observation (2026-09-13).

    python -m src.market.observe [--skip-collect]

1. Collecte le S&P 500 du jour (src/market/collect.py).
2. Présente l'univers entier à chaque agent d'univers.
3. Écrit leurs propositions dans logs/universe_proposals.csv.

Aucun ordre ne part d'ici : ce module n'importe même pas le courtier. Les
propositions s'accumulent pour être mesurées ; un agent ne passera en
exécution qu'après une validation hors échantillon, décidée explicitement.
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Sequence

from src.agents.universe_base import UniverseAgent
from src.market.collect import collect
from src.market.view import MARKET_DIR, load_market_view

PROPOSALS = Path("logs/universe_proposals.csv")
COLUMNS = ["as_of", "run_at", "agent", "symbol", "action", "confidence", "reason", "meta"]


def default_agents() -> List[UniverseAgent]:
    from src.agents.insider_cluster import InsiderClusterAgent
    from src.agents.pairs_universe import PairsUniverseAgent
    return [InsiderClusterAgent(), PairsUniverseAgent()]


def _append(rows: List[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    new = not path.exists()
    with path.open("a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=COLUMNS)
        if new:
            w.writeheader()
        w.writerows(rows)


def _notify(title: str, body: str, severity: str) -> None:
    try:
        from src.events.bus import Event, get_bus
        get_bus().emit(Event(type="system", severity=severity, title=title, body=body[:2000]))
    except Exception:
        pass


def run(
    agents: Optional[Sequence[UniverseAgent]] = None,
    collect_first: bool = True,
    root: Path | str = MARKET_DIR,
    proposals_path: Path | str = PROPOSALS,
    notify: bool = True,
) -> Dict[str, object]:
    summary: Dict[str, object] = {"agents": {}}

    if collect_first:
        report = collect(root=root)
        print(report.render())
        summary["collecte"] = {"ok": report.ok, "couverture": report.coverage,
                               "derniere_seance": report.last_bar}
        if not report.ok and notify:
            _notify("Collecte du marché en échec", report.render(), "warning")

    view = load_market_view(root)
    summary["as_of"] = view.as_of.isoformat()
    run_at = datetime.now(timezone.utc).replace(microsecond=0).isoformat()
    print(f"🔭 Vue du marché au {view.as_of} — {len(view.tickers)} titres")

    for agent in agents if agents is not None else default_agents():
        t0 = time.monotonic()
        try:
            props = agent.propose(view)
        except Exception as exc:
            msg = f"{type(exc).__name__}: {exc}"
            print(f"🚨 {agent.name} en panne — {msg}")
            summary["agents"][agent.name] = {"erreur": msg}
            continue

        _append([{
            "as_of": view.as_of.isoformat(), "run_at": run_at, "agent": s.agent_name,
            "symbol": s.symbol, "action": s.action, "confidence": s.confidence,
            "reason": s.reason, "meta": json.dumps(s.meta, default=str, ensure_ascii=False),
        } for s in props], Path(proposals_path))

        stats = {k: v for k, v in getattr(agent, "last_stats", {}).items()
                 if not isinstance(v, list)}
        summary["agents"][agent.name] = {"propositions": len(props),
                                         "duree_s": round(time.monotonic() - t0, 1), **stats}
        print(f"   {agent.name}: {len(props)} proposition(s) "
              f"en {summary['agents'][agent.name]['duree_s']} s — {stats}")
        for s in props:
            print(f"      {s.action} {s.symbol} ({s.confidence:.2f}) {s.reason}")

    failed = [n for n, a in summary["agents"].items() if "erreur" in a]
    if notify:
        n = sum(a.get("propositions", 0) for a in summary["agents"].values())
        _notify(
            f"Observation — {n} proposition(s)" + (f", {len(failed)} agent(s) en panne" if failed else ""),
            json.dumps(summary, ensure_ascii=False, indent=1),
            "warning" if failed else "info",
        )
    summary["pannes"] = failed
    return summary


def main() -> None:
    ap = argparse.ArgumentParser(description="Passage nocturne en observation")
    ap.add_argument("--skip-collect", action="store_true",
                    help="réutilise la dernière collecte au lieu d'en refaire une")
    args = ap.parse_args()
    summary = run(collect_first=not args.skip_collect)
    # Un agent en panne fait échouer le service : systemd le montre, le chien
    # de garde et le tableau de bord aussi. Un jour calme, lui, sort à 0.
    sys.exit(1 if summary["pannes"] else 0)


if __name__ == "__main__":
    main()
