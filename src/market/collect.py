# src/market/collect.py
"""
Collecte nocturne : le S&P 500 du jour, ses prix, leur contrôle.

    python -m src.market.collect

Trois garanties
---------------
1. **Rien n'est remplacé par pire.** Si la couverture tombe sous
   MIN_COVERAGE (panne du fournisseur, blocage), la collecte précédente reste
   en place et l'échec est écrit à part. Une nuit ratée coûte une nuit de
   fraîcheur, pas les données.
2. **Aucune collecte à moitié écrite n'est visible.** La nouvelle est écrite à
   côté, puis substituée par renommage.
3. **Ce qui manque est chiffré.** Les membres sans prix et les séries rejetées
   pour qualité sont listés dans report.json, jamais passés sous silence.

Pourquoi tout retélécharger chaque nuit
---------------------------------------
yfinance ajusté réécrit l'historique à chaque dividende (voir snapshot.py) :
ajouter le jour nouveau à des prix d'hier mélangerait deux échelles.
Retélécharger trois ans pour 500 titres prend quelques minutes, et donne des
séries cohérentes entre elles.
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
from dataclasses import asdict, dataclass, field
from datetime import date, datetime, timezone
from pathlib import Path
from typing import List, Optional

import pandas as pd

from src.data.prices import PriceSource, YFinanceSource
from src.data.quality import filter_universe
from src.data.snapshot import write_snapshot
from src.data.universe import constituents_at
from src.market.view import MARKET_DIR, MEMBERS_FILE, PRICES_DIR, REPORT_FILE

# Sous ce seuil, la collecte est jugée en panne et n'écrase rien.
MIN_COVERAGE = 0.90

# Deux ans de fenêtres de cointégration (504 séances) plus une marge.
PERIOD = "3y"

FAILED_FILE = "failed_attempt.json"


@dataclass
class CollectReport:
    ok: bool
    collected_at: str
    last_bar: str                  # séance la plus récente, la plus fréquente
    revision_id: int
    source: str
    n_members: int
    n_prices: int
    coverage: float
    missing: List[str] = field(default_factory=list)
    rejected: List[dict] = field(default_factory=list)   # {symbol, reason}
    message: str = ""

    def render(self) -> str:
        head = "✅" if self.ok else "🚨"
        lines = [
            f"{head} Collecte {self.collected_at[:16]} — {self.n_prices}/{self.n_members} "
            f"membres ({self.coverage:.1%}), dernière séance {self.last_bar}, source {self.source}",
        ]
        if self.missing:
            lines.append(f"   sans prix ({len(self.missing)}) : {', '.join(self.missing[:15])}"
                         + (" …" if len(self.missing) > 15 else ""))
        for r in self.rejected[:10]:
            lines.append(f"   rejeté {r['symbol']} : {r['reason']}")
        if self.message:
            lines.append(f"   {self.message}")
        return "\n".join(lines)


def _write_json(path: Path, payload: dict) -> None:
    """Écriture par renommage : un lecteur voit l'ancien fichier ou le nouveau, jamais un morceau."""
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2, ensure_ascii=False, default=str))
    os.replace(tmp, path)


def collect(
    source: Optional[PriceSource] = None,
    as_of: Optional[date] = None,
    root: Path | str = MARKET_DIR,
    period: str = PERIOD,
    min_coverage: float = MIN_COVERAGE,
) -> CollectReport:
    as_of = as_of or date.today()
    source = source or YFinanceSource()
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)

    snap, members = constituents_at(as_of)
    tickers = [c.ticker for c in members]
    raw = source.daily(tickers, period=period)
    kept, rejets = filter_universe(raw)

    last_bars = pd.Series([str(df.index[-1].date()) for df in kept.values()], dtype=str)
    report = CollectReport(
        ok=False,
        collected_at=datetime.now(timezone.utc).replace(microsecond=0).isoformat(),
        last_bar=str(last_bars.mode().iat[0]) if len(last_bars) else "",
        revision_id=snap.revision_id,
        source=source.name,
        n_members=len(tickers),
        n_prices=len(kept),
        coverage=round(len(kept) / len(tickers), 4) if tickers else 0.0,
        missing=sorted(set(tickers) - set(raw)),
        rejected=[{"symbol": q.symbol, "reason": q.reason} for q in rejets],
    )
    report.ok = report.coverage >= min_coverage

    if not report.ok:
        report.message = (f"couverture {report.coverage:.1%} < {min_coverage:.0%} — "
                          "collecte précédente conservée")
        _write_json(root / FAILED_FILE, asdict(report))
        return report

    dst, tmp, old = root / PRICES_DIR, root / f"{PRICES_DIR}.tmp", root / f"{PRICES_DIR}.old"
    shutil.rmtree(tmp, ignore_errors=True)
    shutil.rmtree(old, ignore_errors=True)
    write_snapshot(kept, tmp, period=period, source=source.name)
    if dst.exists():
        dst.rename(old)
    tmp.rename(dst)
    shutil.rmtree(old, ignore_errors=True)

    _write_json(root / MEMBERS_FILE, {
        "as_of": as_of.isoformat(),
        "revision_id": snap.revision_id,
        "revision_date": snap.revision_date,
        "members": [{**asdict(c), "has_prices": c.ticker in kept} for c in members],
    })
    _write_json(root / REPORT_FILE, asdict(report))
    (root / FAILED_FILE).unlink(missing_ok=True)
    return report


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    ap.add_argument("--period", default=PERIOD)
    args = ap.parse_args()
    report = collect(period=args.period)
    print(report.render())
    sys.exit(0 if report.ok else 1)


if __name__ == "__main__":
    main()
