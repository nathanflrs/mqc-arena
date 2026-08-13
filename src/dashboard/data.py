# src/dashboard/data.py
"""
Couche de données du dashboard — un chiffre, sa date, sa source, sa nature.

Pourquoi ce module existe
-------------------------
Chaque carte de l'écran lisait directement le fichier qui lui semblait le plus
proche de sa question, sans jamais vérifier si ce fichier décrivait aujourd'hui,
le compte réel, ou une simulation. Résultat, constaté le 2026-08-13 :

- « SIGNAUX ACTIFS : 16 » comptait 12 décisions du jour, 2 du 23 juillet, et
  2 du 4 juin portant sur BRK-B et JNJ — deux titres sortis de l'univers depuis.
  La cause : on dédupliquait par symbole sur tout l'historique, si bien qu'un
  symbole abandonné restait affiché indéfiniment.
- « EQUITY CURVE VS SPY » traçait un backtest 2022→juin 2026 sur une base de
  100 000 $, sans le dire.
- « NET ASSET VALUE » affichait le sommet historique, jamais la valeur du jour.
- Le track record mêlait une ligne à 100 000 $ produite par l'ancienne
  automatisation sans courtier à la vraie ligne du compte.

Le principe retenu
------------------
Une question a **une** source. La réponse transporte toujours :

  - `as_of`   : la date des données sous-jacentes, pour que « périmé » se voie
  - `source`  : le fichier d'où sort le chiffre, pour pouvoir le vérifier
  - `kind`    : `live`, `simulated` ou `unavailable`

`unavailable` est une réponse légitime et fréquente. Un fonds qui n'a pas encore
tradé n'a pas de P&L : l'afficher à zéro serait faux, et l'afficher depuis un
backtest serait pire.
"""
from __future__ import annotations

import io
import json
from dataclasses import asdict, dataclass, field
from datetime import date, datetime, timezone
from typing import Any, Callable, Dict, List, Optional

import pandas as pd

# Au-delà, les données ne décrivent plus le marché d'aujourd'hui. Deux séances
# de marge : un week-end ou un férié ne doit pas alerter, trois jours de silence
# si.
FRESH_MAX_HOURS = 72.0

LIVE = "live"
SIMULATED = "simulated"
UNAVAILABLE = "unavailable"


@dataclass
class Figure:
    """Un chiffre affichable, et tout ce qu'il faut pour lui faire confiance."""
    value: Any = None
    kind: str = UNAVAILABLE
    as_of: Optional[str] = None       # ISO — date des données, pas de la lecture
    source: str = ""                  # fichier d'origine
    note: str = ""                    # pourquoi indisponible, ou quoi savoir

    @property
    def age_hours(self) -> Optional[float]:
        if not self.as_of:
            return None
        try:
            ts = pd.to_datetime(self.as_of, utc=True)
        except Exception:
            return None
        return (pd.Timestamp.now(tz="UTC") - ts).total_seconds() / 3600.0

    @property
    def is_fresh(self) -> Optional[bool]:
        a = self.age_hours
        return None if a is None else bool(a <= FRESH_MAX_HOURS)

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d["age_hours"] = round(self.age_hours, 2) if self.age_hours is not None else None
        d["is_fresh"] = self.is_fresh
        return d

    @classmethod
    def missing(cls, note: str, source: str = "") -> "Figure":
        return cls(value=None, kind=UNAVAILABLE, note=note, source=source)


class DashboardData:
    """
    Répond aux questions du dashboard à partir des fichiers du fonds.

    `read` est injectable pour que les tests n'aient pas besoin du disque, et
    pour que le mode cloud (lecture via l'API GitHub) passe par le même chemin.
    """

    def __init__(self, read: Callable[[str], Optional[str]]):
        self._read = read

    # ── Outils ───────────────────────────────────────────────────────────────

    def _csv(self, path: str) -> Optional[pd.DataFrame]:
        raw = self._read(path)
        if not raw:
            return None
        try:
            df = pd.read_csv(io.StringIO(raw))
            return df if not df.empty else None
        except Exception:
            return None

    def _json(self, path: str) -> Optional[dict]:
        raw = self._read(path)
        if not raw:
            return None
        try:
            return json.loads(raw)
        except Exception:
            return None

    # ── Combien vaut le portefeuille ? ───────────────────────────────────────

    def nav(self) -> Figure:
        """
        Valeur du portefeuille, telle que relevée au dernier run.

        Source unique : `equity_curve.csv`, écrit à chaque run avec le
        NetLiquidation lu chez le courtier. On n'utilise PAS `peak_netliq` du
        circuit breaker : c'est le sommet historique, donc toujours supérieur ou
        égal à la valeur réelle — l'écran ne pouvait que flatter.
        """
        df = self._csv("logs/equity_curve.csv")
        if df is None or "netliq" not in df.columns:
            return Figure.missing(
                "aucun relevé de capital — le premier run n'a pas encore eu lieu",
                "logs/equity_curve.csv")
        last = df.iloc[-1]
        return Figure(
            value=float(last["netliq"]),
            kind=LIVE,
            as_of=str(last.get("date", "")),
            source="logs/equity_curve.csv",
        )

    def equity_curve(self) -> Figure:
        """
        Capital jour après jour, depuis le premier run **sur ce serveur**.

        Distinct de `logs/portfolio_equity.csv`, qui est un backtest 2022→2026
        sur base 100 000 $ : le tracer comme « courbe d'équité » revenait à
        montrer une simulation à la place du compte.

        Un seul point est une réponse valable au début — la courbe se construit
        une séance à la fois.
        """
        df = self._csv("logs/equity_curve.csv")
        if df is None:
            return Figure.missing("la courbe se remplira à partir du premier run",
                                  "logs/equity_curve.csv")
        df = df.drop_duplicates(subset=["date"], keep="last")
        points = [{"date": str(r["date"]), "netliq": float(r["netliq"])}
                  for _, r in df.iterrows()]
        note = ""
        if len(points) < 2:
            note = ("un seul relevé — il faut au moins deux séances pour "
                    "dessiner une évolution")
        return Figure(value=points, kind=LIVE, as_of=points[-1]["date"],
                      source="logs/equity_curve.csv", note=note)

    def total_return(self) -> Figure:
        """Rendement depuis le premier relevé. Exige au moins deux points."""
        curve = self.equity_curve()
        pts = curve.value or []
        if len(pts) < 2:
            return Figure.missing(
                f"{len(pts)} relevé(s) de capital — il en faut deux",
                "logs/equity_curve.csv")
        first, last = pts[0]["netliq"], pts[-1]["netliq"]
        if first <= 0:
            return Figure.missing("premier relevé nul ou négatif",
                                  "logs/equity_curve.csv")
        return Figure(value=(last / first) - 1.0, kind=LIVE,
                      as_of=pts[-1]["date"], source="logs/equity_curve.csv",
                      note=f"depuis le {pts[0]['date']}")

    # ── Qu'a décidé le système aujourd'hui ? ─────────────────────────────────

    def latest_decisions(self) -> Figure:
        """
        Décisions du **dernier run seulement**.

        L'ancien endpoint gardait le dernier gagnant de chaque symbole sur tout
        l'historique. BRK-B et JNJ, retirés de l'univers en juin, restaient donc
        listés comme signaux « live » en août. On filtre désormais sur le
        `plan_id` du dernier run : un symbole absent de ce run est absent de
        l'écran.
        """
        df = self._csv("logs/decisions.csv")
        if df is None or "plan_id" not in df.columns:
            return Figure.missing("aucune décision enregistrée",
                                  "logs/decisions.csv")

        last_id = df["plan_id"].iloc[-1]
        run = df[df["plan_id"] == last_id]

        if "is_winner" in run.columns:
            winners = run[run["is_winner"].astype(str).str.lower()
                          .isin(["true", "1"])]
        else:
            winners = run

        ts_col = "timestamp" if "timestamp" in run.columns else "ts"
        as_of = str(run[ts_col].max()) if ts_col in run.columns else None

        rows = []
        for _, r in winners.sort_values("symbol").iterrows():
            rows.append({
                "symbol": str(r.get("symbol", "")),
                "agent": str(r.get("agent", r.get("winner_agent", ""))),
                "action": str(r.get("action", "")),
                "confidence": float(r.get("confidence", 0) or 0),
                "reason": str(r.get("reason", "")),
                "regime": str(r.get("regime", "")),
            })
        return Figure(value=rows, kind=LIVE, as_of=as_of,
                      source="logs/decisions.csv",
                      note=f"run {last_id} — {len(run)} avis, {len(rows)} retenus")

    def strategies(self) -> Figure:
        """
        Stratégies ayant emporté une décision au dernier run, et leur poids.

        Répond à « qu'est-ce qui pilote mon portefeuille en ce moment ? »
        — question qu'aucune carte ne traitait.
        """
        dec = self.latest_decisions()
        if dec.kind == UNAVAILABLE:
            return Figure.missing(dec.note, dec.source)
        rows: List[dict] = dec.value or []
        counts: Dict[str, int] = {}
        for r in rows:
            if r["action"] in ("BUY", "SELL"):
                counts[r["agent"]] = counts.get(r["agent"], 0) + 1
        total = sum(counts.values())
        out = [{"agent": a, "n_decisions": n,
                "share": round(n / total, 4) if total else 0.0}
               for a, n in sorted(counts.items(), key=lambda kv: -kv[1])]
        return Figure(value=out, kind=LIVE, as_of=dec.as_of,
                      source=dec.source,
                      note="parts calculées sur les décisions actives du run")

    # ── Quel est mon P&L ? ───────────────────────────────────────────────────

    def pnl(self) -> Figure:
        """
        Résultat réalisé, à partir des exécutions **réellement remplies**.

        Tant qu'aucun ordre n'a été exécuté, la réponse est `unavailable` — pas
        zéro. Zéro affirmerait « j'ai tradé et je suis à l'équilibre », ce qui
        est faux. Et le déduire d'un backtest serait pire encore.
        """
        df = self._csv("logs/executions.csv")
        if df is None:
            return Figure.missing(
                "aucun ordre exécuté à ce jour — le P&L n'existe pas encore",
                "logs/executions.csv")
        if "avg_fill_price" in df.columns:
            filled = df[pd.to_numeric(df["avg_fill_price"],
                                      errors="coerce").fillna(0) > 0]
        else:
            filled = df.iloc[0:0]
        if filled.empty:
            return Figure.missing(
                f"{len(df)} ordre(s) journalisé(s), aucun rempli — pas de P&L",
                "logs/executions.csv")
        return Figure(value={"n_fills": int(len(filled))}, kind=LIVE,
                      as_of=str(filled["timestamp"].max())
                      if "timestamp" in filled.columns else None,
                      source="logs/executions.csv",
                      note="P&L par aller-retour à brancher sur LiveScorer")

    # ── Santé ────────────────────────────────────────────────────────────────

    def circuit_breaker(self) -> Figure:
        cb = self._json("logs/circuit_breaker.json")
        if not cb:
            return Figure.missing("aucun état de circuit breaker",
                                  "logs/circuit_breaker.json")
        nav = self.nav()
        return Figure(value=cb, kind=LIVE, as_of=nav.as_of,
                      source="logs/circuit_breaker.json",
                      note="le drawdown se mesure contre le sommet historique "
                           "du compte, pas contre le début du suivi")

    def last_run(self) -> Figure:
        """Quand le moteur a tourné pour la dernière fois."""
        df = self._csv("logs/decisions.csv")
        if df is None:
            return Figure.missing("aucun run enregistré", "logs/decisions.csv")
        col = "timestamp" if "timestamp" in df.columns else "ts"
        if col not in df.columns:
            return Figure.missing("horodatage absent", "logs/decisions.csv")
        try:
            ts = pd.to_datetime(df[col], format="mixed", utc=True).max()
        except Exception:
            return Figure.missing("horodatage illisible", "logs/decisions.csv")
        return Figure(value=ts.isoformat(), kind=LIVE, as_of=ts.isoformat(),
                      source="logs/decisions.csv")

    # ── Recherche : simulé, et jamais présenté autrement ──────────────────────

    def backtest_agents(self) -> Figure:
        """
        Classement d'agents issu d'un backtest.

        Marqué `simulated` à la source. C'est ce qui empêche l'écran d'afficher
        « Buffett +196 % » sous une pastille LIVE, comme il le faisait — un
        rendement simulé est une borne haute obtenue en connaissant la période,
        pas un résultat.
        """
        df = self._csv("logs/portfolio_by_symbol.csv")
        if df is None:
            return Figure.missing("aucun backtest disponible",
                                  "logs/portfolio_by_symbol.csv")
        rows = json.loads(df.sort_values("ret", ascending=False)
                          .to_json(orient="records"))
        return Figure(value=rows, kind=SIMULATED,
                      source="logs/portfolio_by_symbol.csv",
                      note="backtest — paramètres choisis en connaissant cette "
                           "période, à lire comme une borne haute")
