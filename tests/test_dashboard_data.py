"""
Couche de données du dashboard (src/dashboard/data.py).

Chaque test correspond à un mensonge constaté à l'écran le 2026-08-13 :

- `latest_decisions` ne doit plus mêler les runs. L'écran annonçait
  « SIGNAUX ACTIFS : 16 », dont 2 du 23 juillet et 2 du 4 juin portant sur
  BRK-B et JNJ, sortis de l'univers depuis.
- `nav` ne doit plus renvoyer le sommet historique à la place de la valeur.
- `pnl` et `total_return` doivent répondre « indisponible » plutôt qu'un zéro
  ou un chiffre emprunté à un backtest.
- `backtest_agents` doit être marqué `simulated` à la source.
"""
from __future__ import annotations

from typing import Dict, Optional

import pytest

from src.dashboard.data import (
    LIVE, SIMULATED, UNAVAILABLE, DashboardData, Figure,
)


def make(files: Dict[str, str]) -> DashboardData:
    def read(path: str) -> Optional[str]:
        return files.get(path)
    return DashboardData(read)


DECISIONS_HEADER = "plan_id,timestamp,symbol,regime,agent,action,confidence,target_weight,reason,is_winner\n"


# ── Le test central : ne pas mélanger les runs ───────────────────────────────

class TestLatestDecisions:

    def test_only_the_last_run_is_returned(self):
        """
        Le bug d'origine : on gardait le dernier gagnant de CHAQUE symbole sur
        tout l'historique. Un titre retiré de l'univers restait affiché comme
        signal « live » des mois plus tard.
        """
        csv = DECISIONS_HEADER + (
            "old1,2026-06-04T10:00:00+00:00,BRK-B,bull,BuffettAgent,BUY,0.9,0.1,r,True\n"
            "old1,2026-06-04T10:00:00+00:00,JNJ,bull,BuffettAgent,BUY,0.9,0.1,r,True\n"
            "new1,2026-08-13T09:00:00+00:00,AAPL,bull,BuffettAgent,BUY,0.9,0.1,r,True\n"
            "new1,2026-08-13T09:00:00+00:00,SPY,bull,CitadelAgent,BUY,0.8,0.1,r,False\n"
        )
        f = make({"logs/decisions.csv": csv}).latest_decisions()
        syms = [r["symbol"] for r in f.value]
        assert syms == ["AAPL"], "seul le gagnant du dernier run doit sortir"
        assert "BRK-B" not in syms and "JNJ" not in syms, \
            "un symbole sorti de l'univers ne doit plus apparaître"

    def test_as_of_is_the_run_timestamp_not_now(self):
        csv = DECISIONS_HEADER + (
            "p1,2026-08-13T09:06:38+00:00,AAPL,bull,BuffettAgent,BUY,0.9,0.1,r,True\n"
        )
        f = make({"logs/decisions.csv": csv}).latest_decisions()
        assert f.as_of.startswith("2026-08-13T09:06")

    def test_no_decisions_is_unavailable_not_empty_list(self):
        f = make({}).latest_decisions()
        assert f.kind == UNAVAILABLE
        assert f.value is None


# ── NAV : la valeur, jamais le sommet ────────────────────────────────────────

class TestNav:

    def test_uses_last_equity_point_not_the_peak(self):
        """
        L'écran affichait peak_netliq (1 026 192 $) comme « Net Asset Value »
        alors que le compte valait 1 020 810 $. Le sommet est par construction
        supérieur ou égal : l'erreur ne pouvait que flatter.
        """
        d = make({
            "logs/equity_curve.csv": "date,netliq\n2026-08-12,1026192.13\n2026-08-13,1020809.57\n",
            "logs/circuit_breaker.json": '{"peak_netliq": 1026192.13, "current_netliq": 1020809.57}',
        })
        assert d.nav().value == pytest.approx(1020809.57)
        assert d.nav().kind == LIVE

    def test_missing_curve_is_unavailable(self):
        assert make({}).nav().kind == UNAVAILABLE


# ── Ce qui n'existe pas encore doit le dire ──────────────────────────────────

class TestHonestGaps:

    def test_pnl_without_any_execution_is_unavailable_not_zero(self):
        """
        Zéro affirmerait « j'ai tradé et je suis à l'équilibre ». Le fonds
        n'avait exécuté aucun ordre le jour de l'audit.
        """
        f = make({"logs/executions.csv":
                  "timestamp,symbol,side,avg_fill_price\n"}).pnl()
        assert f.kind == UNAVAILABLE
        assert f.value is None

    def test_pnl_ignores_orders_that_never_filled(self):
        csv = ("timestamp,symbol,side,avg_fill_price\n"
               "2026-08-13T09:00:00+00:00,AAPL,BUY,0\n")
        f = make({"logs/executions.csv": csv}).pnl()
        assert f.kind == UNAVAILABLE, \
            "un ordre non rempli n'est pas une transaction"

    def test_total_return_needs_two_points(self):
        one = make({"logs/equity_curve.csv": "date,netliq\n2026-08-13,1020809.57\n"})
        assert one.total_return().kind == UNAVAILABLE

        two = make({"logs/equity_curve.csv":
                    "date,netliq\n2026-08-13,1000000\n2026-08-14,1010000\n"})
        r = two.total_return()
        assert r.kind == LIVE
        assert r.value == pytest.approx(0.01)

    def test_single_equity_point_is_returned_with_a_warning(self):
        f = make({"logs/equity_curve.csv":
                  "date,netliq\n2026-08-13,1020809.57\n"}).equity_curve()
        assert f.kind == LIVE and len(f.value) == 1
        assert "deux séances" in f.note

    def test_equity_curve_deduplicates_same_day_runs(self):
        """Deux runs le même jour ne doivent pas produire deux points."""
        f = make({"logs/equity_curve.csv":
                  "date,netliq\n2026-08-13,1000\n2026-08-13,1010\n"}).equity_curve()
        assert len(f.value) == 1
        assert f.value[0]["netliq"] == 1010, "le dernier relevé du jour fait foi"


# ── Simulé contre réel ───────────────────────────────────────────────────────

class TestProvenance:

    def test_backtest_is_flagged_simulated(self):
        csv = "sym,agent,ret,sharpe,trades\nNVDA,BuffettAgent,1.96,0.84,22\n"
        f = make({"logs/portfolio_by_symbol.csv": csv}).backtest_agents()
        assert f.kind == SIMULATED, \
            "un rendement de backtest ne doit jamais passer pour du réalisé"
        assert "borne haute" in f.note

    def test_every_figure_carries_its_source(self):
        d = make({"logs/equity_curve.csv": "date,netliq\n2026-08-13,1000\n"})
        for f in (d.nav(), d.equity_curve(), d.total_return()):
            assert f.source, "un chiffre sans source n'est pas vérifiable"

    def test_freshness_is_derived_from_as_of(self):
        old = Figure(value=1, kind=LIVE, as_of="2020-01-01T00:00:00+00:00")
        assert old.is_fresh is False and old.age_hours > 1000
        assert Figure(value=1, kind=LIVE).is_fresh is None

    def test_serialisation_exposes_age_and_freshness(self):
        d = Figure(value=1, kind=LIVE, as_of="2020-01-01T00:00:00+00:00").to_dict()
        assert {"value", "kind", "as_of", "source", "note",
                "age_hours", "is_fresh"} <= set(d)


# ── Stratégies : la question qu'aucune carte ne traitait ─────────────────────

class TestStrategies:

    def test_counts_only_active_decisions(self):
        csv = DECISIONS_HEADER + (
            "p1,2026-08-13T09:00:00+00:00,AAPL,bull,BuffettAgent,BUY,0.9,0.1,r,True\n"
            "p1,2026-08-13T09:00:00+00:00,SPY,bull,BuffettAgent,BUY,0.9,0.1,r,True\n"
            "p1,2026-08-13T09:00:00+00:00,GLD,bull,CitadelAgent,HOLD,0.3,0.0,r,True\n"
        )
        f = make({"logs/decisions.csv": csv}).strategies()
        assert f.value[0]["agent"] == "BuffettAgent"
        assert f.value[0]["n_decisions"] == 2
        assert f.value[0]["share"] == pytest.approx(1.0), \
            "un HOLD ne pilote aucun capital"

    def test_no_run_is_unavailable(self):
        assert make({}).strategies().kind == UNAVAILABLE
