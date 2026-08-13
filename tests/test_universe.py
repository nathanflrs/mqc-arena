"""
Univers point-in-time (src/data/universe.py).

Ce que ces tests protègent : la reconstitution ne doit jamais se rabattre
silencieusement sur la composition d'aujourd'hui. 114 des 505 membres de
janvier 2020 ne figurent plus dans la liste actuelle — 23 % de l'univers. Un
repli discret sur « la liste du jour » réintroduirait exactement le biais que
ce module existe pour éliminer.

Aucun accès réseau : les révisions Wikipédia sont simulées.
"""
from __future__ import annotations

import json
from datetime import date
from pathlib import Path

import pytest

from src.data import universe as U


@pytest.fixture(autouse=True)
def cache_isole(tmp_path, monkeypatch):
    monkeypatch.setattr(U, "CACHE_DIR", tmp_path / "univers")
    monkeypatch.setattr(U, "_RATE_SLEEP", 0)


def _faux_wiki(monkeypatch, par_revision: dict, revisions: dict):
    """revisions : date demandée -> (revid, horodatage de la révision)."""
    def revision_at(as_of):
        return revisions[as_of]
    def tickers(revid):
        return par_revision[revid]
    monkeypatch.setattr(U, "_revision_at", revision_at)
    monkeypatch.setattr(U, "_tickers_from_revision", tickers)


class TestPointInTime:

    def test_returns_the_composition_of_that_date_not_today(self, monkeypatch):
        """
        Le cœur du module. ABMD appartenait à l'indice en 2020 et en est sorti
        après son rachat : il doit apparaître dans la composition de 2020 et
        pas dans celle de 2026.
        """
        _faux_wiki(monkeypatch,
                   par_revision={11: ["AAPL", "ABMD", "MSFT"],
                                 22: ["AAPL", "MSFT", "NVDA"]},
                   revisions={date(2020, 1, 1): (11, "2019-12-31T10:00:00Z"),
                              date(2026, 1, 1): (22, "2025-12-30T10:00:00Z")})

        vieux = U.sp500_at(date(2020, 1, 1))
        recent = U.sp500_at(date(2026, 1, 1))

        assert "ABMD" in vieux.tickers, \
            "une société sortie depuis doit rester dans la composition passée"
        assert "ABMD" not in recent.tickers
        assert "NVDA" not in vieux.tickers, \
            "une société entrée depuis ne doit pas apparaître rétroactivement"

    def test_snapshot_carries_its_source(self, monkeypatch):
        """Un identifiant de révision rend le résultat vérifiable par un tiers."""
        _faux_wiki(monkeypatch, {11: ["AAPL"]},
                   {date(2020, 1, 1): (11, "2019-12-31T10:00:00Z")})
        s = U.sp500_at(date(2020, 1, 1))
        assert s.revision_id == 11
        assert s.source == "wikipedia"

    def test_lag_is_exposed_not_hidden(self, monkeypatch):
        """
        Wikipédia peut avoir du retard sur une entrée ou une sortie. L'écart
        doit être lisible : une révision vieille de trois semaines décrit un
        indice qui a pu changer entre-temps.
        """
        _faux_wiki(monkeypatch, {11: ["AAPL"]},
                   {date(2020, 1, 22): (11, "2020-01-01T10:00:00Z")})
        assert U.sp500_at(date(2020, 1, 22)).lag_days == 21

    def test_dot_tickers_are_normalised(self, monkeypatch):
        """Wikipédia écrit BRK.B ; les fournisseurs de prix attendent BRK-B."""
        monkeypatch.setattr(U, "_revision_at",
                            lambda d: (11, "2020-01-01T00:00:00Z"))
        import pandas as pd
        html = pd.DataFrame({"Symbol": ["BRK.B", "BF.B", "AAPL"]}).to_html()

        class R:
            text = html
            def raise_for_status(self): pass
        monkeypatch.setattr(U.requests, "get", lambda *a, **k: R())

        t = U._tickers_from_revision(11)
        assert "BRK-B" in t and "BF-B" in t
        assert "BRK.B" not in t


class TestCache:

    def test_second_call_does_not_hit_the_network(self, monkeypatch):
        appels = {"n": 0}
        def revision_at(as_of):
            appels["n"] += 1
            return (11, "2019-12-31T10:00:00Z")
        monkeypatch.setattr(U, "_revision_at", revision_at)
        monkeypatch.setattr(U, "_tickers_from_revision", lambda r: ["AAPL"])

        U.sp500_at(date(2020, 1, 1))
        U.sp500_at(date(2020, 1, 1))
        assert appels["n"] == 1, "une révision passée est immuable : un seul appel"

    def test_corrupt_cache_is_refetched(self, monkeypatch, tmp_path):
        _faux_wiki(monkeypatch, {11: ["AAPL"]},
                   {date(2020, 1, 1): (11, "2019-12-31T10:00:00Z")})
        U.sp500_at(date(2020, 1, 1))
        U._cache_path(date(2020, 1, 1)).write_text("{ cassé")
        assert U.sp500_at(date(2020, 1, 1)).tickers == ["AAPL"]


class TestEverMembers:

    def test_union_across_time_keeps_the_departed(self, monkeypatch):
        """
        L'univers à télécharger est l'UNION dans le temps. Une société sortie en
        cours de route doit avoir ses prix, sinon les décisions la concernant
        avant sa sortie deviennent invisibles — ce qui est le biais du survivant
        sous une autre forme.
        """
        revs = {}
        d = date(2020, 1, 1)
        while d <= date(2020, 12, 31):
            revs[d] = (11 if d.month <= 6 else 22, "2020-01-01T00:00:00Z")
            d = date.fromordinal(d.toordinal() + 90)
        revs[date(2020, 12, 31)] = (22, "2020-12-01T00:00:00Z")
        _faux_wiki(monkeypatch,
                   par_revision={11: ["AAPL", "SORTIE"], 22: ["AAPL", "ENTREE"]},
                   revisions=revs)

        membres = U.ever_members(date(2020, 1, 1), date(2020, 12, 31))
        assert {"AAPL", "SORTIE", "ENTREE"} <= membres, \
            "l'union doit contenir les entrantes ET les sortantes"


class TestCoverage:

    def test_report_quantifies_the_gap(self):
        r = U.coverage_report({"A", "B", "C", "D"}, {"A", "B"})
        assert r["n_univers"] == 4
        assert r["n_manquants"] == 2
        assert r["couverture"] == 0.5
        assert r["manquants"] == ["C", "D"]

    def test_full_coverage_is_reported_as_such(self):
        r = U.coverage_report({"A", "B"}, {"A", "B", "Z"})
        assert r["couverture"] == 1.0 and r["n_manquants"] == 0

    def test_empty_universe_does_not_divide_by_zero(self):
        assert U.coverage_report(set(), {"A"})["couverture"] == 0.0
