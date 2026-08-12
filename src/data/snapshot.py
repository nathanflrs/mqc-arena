# src/data/snapshot.py
"""
Milan Capital — Snapshots de données figés et détection de réécriture.

Le problème
-----------
`yfinance` avec `auto_adjust=True` réécrit **rétroactivement tout l'historique**
à chaque détachement de dividende ou split. Mesuré le 2026-08-02 sur 3 ans :

    GS   : la plus ancienne barre bouge de −6.79 %
    JPM  : −6.39 %
    SPY  : −3.66 %

soit environ 2 % par an pour un titre à dividende. Deux backtests lancés à
quelques semaines d'écart ne portent donc pas sur les mêmes prix, et leurs
résultats ne sont pas comparables. C'est ce qui explique que l'alpha de
`portfolio_backtest.py` soit passé de +10.7 pts à −12.5 pts entre deux
exécutions distantes de dix jours : ce n'était pas un changement de code, mais
un changement de données.

La réponse
----------
Un snapshot daté, versionné, avec une empreinte par symbole. Le backtest lit le
snapshot, jamais le réseau. Rejouer un backtest six mois plus tard donne alors
exactement le même chiffre.

`diff_snapshots()` fait mieux que contourner le problème : il le **mesure**.
Comparer un nouveau téléchargement au snapshot dit quels symboles ont été
réécrits, de combien, et à partir de quelle date — ce qui transforme une source
d'irreproductibilité silencieuse en une quantité observable.
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pandas as pd

MANIFEST_NAME = "manifest.json"

# Colonnes prises en compte dans l'empreinte. `Volume` en est exclu : yfinance
# le révise fréquemment sans que cela change une décision de trading, et
# l'inclure ferait crier au loup à chaque téléchargement.
CHECKSUM_COLUMNS = ("Open", "High", "Low", "Close")

# En deçà de ce seuil relatif, un écart de prix relève du bruit d'arrondi
# flottant et non d'une réécriture.
RESTATEMENT_EPSILON = 1e-6


def _checksum(df: pd.DataFrame) -> str:
    """Empreinte stable d'une série OHLC — insensible à l'ordre des colonnes."""
    cols = [c for c in CHECKSUM_COLUMNS if c in df.columns]
    payload = df[cols].round(6).to_csv().encode()
    return hashlib.sha256(payload).hexdigest()[:16]


@dataclass
class SnapshotManifest:
    created_at: str
    period: str
    auto_adjust: bool
    source: str
    symbols: List[str] = field(default_factory=list)
    checksums: Dict[str, str] = field(default_factory=dict)
    n_rows: Dict[str, int] = field(default_factory=dict)
    first_date: Dict[str, str] = field(default_factory=dict)
    last_date: Dict[str, str] = field(default_factory=dict)

    def to_json(self) -> str:
        return json.dumps(asdict(self), indent=2, ensure_ascii=False)

    @classmethod
    def from_json(cls, raw: str) -> "SnapshotManifest":
        return cls(**json.loads(raw))


def write_snapshot(
    data: Dict[str, pd.DataFrame],
    path: str | Path,
    *,
    period: str = "5y",
    auto_adjust: bool = True,
    source: str = "yfinance",
) -> SnapshotManifest:
    """Écrit les séries en parquet et le manifeste qui les identifie."""
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)

    manifest = SnapshotManifest(
        created_at=datetime.now(timezone.utc).replace(microsecond=0).isoformat(),
        period=period, auto_adjust=auto_adjust, source=source,
    )
    for sym in sorted(data):
        df = data[sym]
        df.to_parquet(p / f"{sym}.parquet")
        manifest.symbols.append(sym)
        manifest.checksums[sym] = _checksum(df)
        manifest.n_rows[sym] = int(len(df))
        manifest.first_date[sym] = str(df.index[0].date()) if len(df) else ""
        manifest.last_date[sym] = str(df.index[-1].date()) if len(df) else ""

    (p / MANIFEST_NAME).write_text(manifest.to_json())
    return manifest


def load_snapshot(
    path: str | Path, *, verify: bool = True,
) -> Tuple[Dict[str, pd.DataFrame], SnapshotManifest, List[str]]:
    """
    Charge un snapshot. Retourne (données, manifeste, symboles altérés).

    `verify` recalcule l'empreinte de chaque série : un parquet modifié depuis
    l'écriture du manifeste est signalé, pas chargé en silence. Un snapshot
    dont on ne peut plus garantir le contenu ne vaut pas mieux que le réseau.
    """
    p = Path(path)
    manifest_path = p / MANIFEST_NAME
    if not manifest_path.exists():
        raise FileNotFoundError(f"Snapshot sans manifeste : {manifest_path}")

    manifest = SnapshotManifest.from_json(manifest_path.read_text())
    data: Dict[str, pd.DataFrame] = {}
    tampered: List[str] = []

    for sym in manifest.symbols:
        f = p / f"{sym}.parquet"
        if not f.exists():
            tampered.append(sym)
            continue
        df = pd.read_parquet(f)
        if verify and _checksum(df) != manifest.checksums.get(sym):
            tampered.append(sym)
        data[sym] = df

    return data, manifest, tampered


@dataclass
class SymbolRestatement:
    symbol: str
    n_bars_changed: int
    n_bars_total: int
    max_abs_change: float          # écart relatif maximal observé
    first_changed_date: str
    rows_added: int

    @property
    def pct_bars_changed(self) -> float:
        return self.n_bars_changed / self.n_bars_total if self.n_bars_total else 0.0

    def render(self) -> str:
        return (
            f"  📝 {self.symbol}: {self.n_bars_changed}/{self.n_bars_total} barres réécrites "
            f"({self.pct_bars_changed:.0%}), écart max {self.max_abs_change:.2%}, "
            f"à partir du {self.first_changed_date}"
            + (f", +{self.rows_added} barre(s) nouvelle(s)" if self.rows_added else "")
        )


@dataclass
class RestatementReport:
    restated: List[SymbolRestatement] = field(default_factory=list)
    unchanged: List[str] = field(default_factory=list)
    new_symbols: List[str] = field(default_factory=list)
    missing_symbols: List[str] = field(default_factory=list)

    @property
    def is_clean(self) -> bool:
        return not self.restated and not self.missing_symbols

    def render(self) -> str:
        if self.is_clean:
            return (f"🔍 Aucune réécriture — {len(self.unchanged)} série(s) identiques"
                    + (f", {len(self.new_symbols)} nouvelle(s)" if self.new_symbols else ""))
        lines = [
            f"⚠️  Historique réécrit sur {len(self.restated)} série(s) "
            f"({len(self.unchanged)} inchangée(s))"
        ]
        lines += [r.render() for r in self.restated]
        if self.missing_symbols:
            lines.append(f"  ❌ Absentes du nouveau jeu : {', '.join(self.missing_symbols)}")
        lines.append(
            "   Un backtest lancé sur ces données ne reproduira pas le précédent. "
            "Régénérer le snapshot est un acte explicite, pas un effet de bord."
        )
        return "\n".join(lines)


def diff_snapshots(
    old: Dict[str, pd.DataFrame],
    new: Dict[str, pd.DataFrame],
    *,
    epsilon: float = RESTATEMENT_EPSILON,
) -> RestatementReport:
    """
    Quantifie ce qui a changé dans l'historique commun aux deux jeux.

    Seules les dates présentes des deux côtés sont comparées : les barres
    ajoutées depuis (le marché a continué) sont comptées à part et ne
    constituent pas une réécriture.
    """
    report = RestatementReport()

    for sym in sorted(set(old) | set(new)):
        if sym not in new:
            report.missing_symbols.append(sym)
            continue
        if sym not in old:
            report.new_symbols.append(sym)
            continue

        o, n = old[sym], new[sym]
        common = o.index.intersection(n.index)
        rows_added = int(len(n.index.difference(o.index)))

        if len(common) == 0:
            report.restated.append(SymbolRestatement(
                sym, 0, 0, 0.0, "aucune date commune", rows_added))
            continue

        oc = pd.to_numeric(o.loc[common, "Close"], errors="coerce")
        nc = pd.to_numeric(n.loc[common, "Close"], errors="coerce")
        rel = ((nc - oc) / oc.where(oc != 0)).abs().fillna(0.0)
        changed = rel > epsilon

        if not changed.any():
            report.unchanged.append(sym)
            continue

        report.restated.append(SymbolRestatement(
            symbol=sym,
            n_bars_changed=int(changed.sum()),
            n_bars_total=int(len(common)),
            max_abs_change=float(rel.max()),
            first_changed_date=str(changed.index[changed.argmax()].date()),
            rows_added=rows_added,
        ))

    # Trié par gravité à la construction, pas seulement à l'affichage : un
    # appelant qui lit `report.restated[0]` doit obtenir la réécriture la plus
    # importante, pas la première dans l'ordre alphabétique.
    report.restated.sort(key=lambda r: r.max_abs_change, reverse=True)
    return report
