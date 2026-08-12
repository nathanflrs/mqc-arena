"""
Données comptables point-in-time (src/data/sec_fundamentals.py).

Deux erreurs sont visées, et chacune a été constatée sur des données réelles
avant d'être corrigée :

1. **Le look-ahead.** Renvoyer un chiffre révisé pour une date antérieure à sa
   correction produit des backtests flatteurs et faux. Constaté sur Apple :
   total de bilan 2008 publié à 39,57 Md$, corrigé à 36,17 Md$ six mois plus
   tard. Test : `test_restatement_returns_value_known_at_the_time`.

2. **Le mélange trimestre / exercice.** Constaté sur UFPT : chiffre d'affaires
   trimestriel (101,5 M$) comparé à un résultat net annuel (44,9 M$), soit une
   marge nette apparente de 44 % pour un sous-traitant industriel. Les deux
   observations portaient la même date de fin ; seule la durée les
   distinguait. Test : `TestFlowVersusStock`.

Aucun accès réseau : les tests injectent une charge utile EDGAR fabriquée, ce
qui les rend déterministes et permet de reproduire des cas (révision, étiquette
sparse, cumul depuis début d'exercice) qu'on ne peut pas provoquer à la demande.
"""
from __future__ import annotations

from datetime import date, timedelta

import pandas as pd
import pytest

from src.data.sec_fundamentals import CONCEPTS, FundamentalsClient, _period_kind


# ── Fabrication de charges utiles EDGAR ──────────────────────────────────────

def _obs(end: str, val: float, filed: str, form: str, start: str | None) -> dict:
    o = {"end": end, "val": val, "filed": filed, "form": form,
         "fy": 2024, "fp": "Q1"}
    if start:
        o["start"] = start
    return o


def _instant(end: str, val: float, filed: str, form: str = "10-Q") -> dict:
    """Grandeur de bilan : pas de durée."""
    return _obs(end, val, filed, form, None)


def _quarter(end: str, val: float, filed: str, form: str = "10-Q") -> dict:
    start = (date.fromisoformat(end) - timedelta(days=91)).isoformat()
    return _obs(end, val, filed, form, start)


def _annual(end: str, val: float, filed: str, form: str = "10-K") -> dict:
    start = (date.fromisoformat(end) - timedelta(days=364)).isoformat()
    return _obs(end, val, filed, form, start)


def _ytd(end: str, val: float, filed: str, days: int = 272) -> dict:
    """Cumul depuis le début d'exercice — ni trimestre, ni année."""
    start = (date.fromisoformat(end) - timedelta(days=days)).isoformat()
    return _obs(end, val, filed, "10-Q", start)


def _payload(tags: dict[str, list[dict]]) -> dict:
    return {"facts": {"us-gaap": {
        tag: {"units": {"USD": obs}} for tag, obs in tags.items()}}}


def _four_quarters(tag: str, values, filed_dates) -> dict:
    ends = ["2024-03-31", "2024-06-30", "2024-09-30", "2024-12-31"]
    return {tag: [_quarter(e, v, f)
                  for e, v, f in zip(ends, values, filed_dates)]}


class _FakeClient(FundamentalsClient):
    """Client EDGAR sans réseau."""

    def __init__(self, payload: dict | None):
        super().__init__()
        self._payload = payload

    def cik(self, ticker: str):
        return 1 if self._payload is not None else None

    def _raw_facts(self, ticker: str):
        return self._payload


# ── Classification des périodes ──────────────────────────────────────────────

class TestPeriodKind:

    def test_no_start_is_an_instant(self):
        assert _period_kind(None, date(2024, 3, 31)) == "instant"

    def test_quarter_and_year_are_recognised(self):
        assert _period_kind(date(2024, 1, 1), date(2024, 3, 31)) == "quarter"
        assert _period_kind(date(2023, 1, 1), date(2023, 12, 31)) == "annual"

    def test_year_to_date_is_neither(self):
        """
        Un cumul de trois trimestres porte la même date de fin qu'un trimestre.
        L'additionner à des trimestres compterait deux fois la même activité.
        """
        assert _period_kind(date(2023, 1, 1), date(2023, 9, 30)) == "ytd"


# ── Le test qui justifie le module ───────────────────────────────────────────

class TestPointInTime:

    def test_restatement_returns_value_known_at_the_time(self):
        """
        Cas réel Apple : total de bilan au 27/09/2008 publié à 39,57 Md$, puis
        corrigé à 36,17 Md$ six mois plus tard. Une décision de novembre 2009
        devait voir 39,57.
        """
        c = _FakeClient(_payload({"Assets": [
            _instant("2008-09-27", 39_572_000_000, "2009-07-22"),
            _instant("2008-09-27", 36_171_000_000, "2010-01-25", "10-K/A"),
        ]}))

        avant = c.as_of("AAPL", date(2009, 11, 1), ["assets"])["assets"]
        apres = c.as_of("AAPL", date(2010, 6, 1), ["assets"])["assets"]

        assert avant.value == 39_572_000_000, \
            "avant la correction, la stratégie devait voir le chiffre d'origine"
        assert avant.filed == date(2009, 7, 22)
        assert apres.value == 36_171_000_000, \
            "après la correction, le chiffre révisé devient la vérité connue"

    def test_future_filing_is_invisible(self):
        """Un résultat paraît ~34 jours après la clôture : avant, il n'existe pas."""
        c = _FakeClient(_payload({"Assets": [
            _instant("2024-03-31", 1_000, "2024-05-04"),
        ]}))
        assert c.as_of("X", date(2024, 5, 3), ["assets"]) == {}, \
            "la veille du dépôt, l'information n'est pas publique"
        assert c.as_of("X", date(2024, 5, 4), ["assets"])["assets"].value == 1_000

    def test_period_end_alone_would_leak(self):
        """
        Garde-fou explicite : filtrer sur la fin de période au lieu de la date
        de dépôt donnerait accès au chiffre 34 jours trop tôt. Ce test échoue
        si quelqu'un « simplifie » le filtre un jour.
        """
        c = _FakeClient(_payload({"Assets": [
            _instant("2024-03-31", 42, "2024-05-04"),
        ]}))
        assert c.as_of("X", date(2024, 4, 15), ["assets"]) == {}, \
            "la période est close mais le chiffre n'est pas encore déposé"

    def test_latest_closed_period_wins(self):
        c = _FakeClient(_payload({"Assets": [
            _instant("2024-03-31", 100, "2024-05-01"),
            _instant("2024-06-30", 200, "2024-08-01"),
        ]}))
        f = c.as_of("X", date(2024, 9, 1), ["assets"])["assets"]
        assert f.value == 200
        assert f.period_end == date(2024, 6, 30)

    def test_older_filing_does_not_override_newer_period(self):
        """
        Un 10-K déposé tard qui rappelle un trimestre ancien ne doit pas
        écraser un trimestre plus récent déjà publié.
        """
        c = _FakeClient(_payload({"Assets": [
            _instant("2024-06-30", 200, "2024-08-01"),
            _instant("2024-03-31", 100, "2024-11-01", "10-K"),   # rappel tardif
        ]}))
        f = c.as_of("X", date(2024, 12, 1), ["assets"])["assets"]
        assert f.value == 200, "la période la plus récente reste la bonne"


# ── Flux contre stock : le bug UFPT ──────────────────────────────────────────

class TestFlowVersusStock:

    def test_quarterly_flow_is_never_reported_raw(self):
        """
        Le cœur du bug UFPT : un trimestre isolé ne doit jamais sortir tel quel,
        sinon il finit comparé à un exercice annuel d'une autre métrique.
        """
        c = _FakeClient(_payload({"NetIncomeLoss": [
            _quarter("2024-03-31", 11_000_000, "2024-05-01"),
        ]}))
        # Un seul trimestre : ni TTM possible, ni exercice → rien plutôt que faux.
        assert c.as_of("X", date(2024, 6, 1), ["net_income"]) == {}

    def test_ttm_sums_four_quarters(self):
        c = _FakeClient(_payload(_four_quarters(
            "NetIncomeLoss", [10, 20, 30, 40],
            ["2024-05-01", "2024-08-01", "2024-11-01", "2025-02-01"])))
        f = c.as_of("X", date(2025, 3, 1), ["net_income"])["net_income"]
        assert f.value == 100, "douze mois glissants = somme des quatre trimestres"
        assert f.basis == "ttm"
        assert f.n_quarters == 4

    def test_ttm_is_public_only_once_the_last_quarter_is_filed(self):
        """
        La somme n'est connaissable qu'au dépôt du DERNIER trimestre. Dater le
        fait du plus ancien rendrait l'information disponible des mois trop tôt.
        """
        c = _FakeClient(_payload(_four_quarters(
            "NetIncomeLoss", [10, 20, 30, 40],
            ["2024-05-01", "2024-08-01", "2024-11-01", "2025-02-01"])))
        assert c.as_of("X", date(2025, 1, 31), ["net_income"]) == {}
        assert c.as_of("X", date(2025, 2, 1), ["net_income"])["net_income"].value == 100

    def test_year_to_date_never_enters_the_sum(self):
        """
        Un cumul 9 mois ajouté à quatre trimestres gonflerait le total. Il porte
        la même date de fin qu'un trimestre : seule la durée le trahit.
        """
        tags = _four_quarters("NetIncomeLoss", [10, 20, 30, 40],
                              ["2024-05-01", "2024-08-01", "2024-11-01", "2025-02-01"])
        tags["NetIncomeLoss"].append(_ytd("2024-09-30", 60, "2024-11-01"))
        f = _FakeClient(_payload(tags)).as_of(
            "X", date(2025, 3, 1), ["net_income"])["net_income"]
        assert f.value == 100, "le cumul depuis début d'exercice doit être ignoré"

    def test_falls_back_to_annual_when_quarters_are_incomplete(self):
        tags = {"NetIncomeLoss": [
            _quarter("2024-03-31", 10, "2024-05-01"),
            _annual("2023-12-31", 44_900_000, "2024-02-29"),
        ]}
        f = _FakeClient(_payload(tags)).as_of(
            "X", date(2024, 6, 1), ["net_income"])["net_income"]
        assert f.value == 44_900_000
        assert f.basis == "annual", "l'exercice publié sert de repli"

    def test_missing_quarter_is_not_silently_summed(self):
        """
        Trois trimestres et un quatrième bien plus ancien couvrent plus de
        douze mois. Les additionner donnerait un TTM amputé d'un trimestre sans
        que rien ne le signale.
        """
        tags = {"NetIncomeLoss": [
            _quarter("2023-03-31", 5, "2023-05-01"),      # trou après celui-ci
            _quarter("2024-06-30", 20, "2024-08-01"),
            _quarter("2024-09-30", 30, "2024-11-01"),
            _quarter("2024-12-31", 40, "2025-02-01"),
        ]}
        assert _FakeClient(_payload(tags)).as_of(
            "X", date(2025, 3, 1), ["net_income"]) == {}, \
            "quatre trimestres non contigus ne font pas douze mois"

    def test_balance_sheet_stays_instantaneous(self):
        c = _FakeClient(_payload({"Assets": [
            _instant("2024-12-31", 353_000, "2025-02-01"),
        ]}))
        f = c.as_of("X", date(2025, 3, 1), ["assets"])["assets"]
        assert f.basis == "instant", "un bilan ne se somme pas sur douze mois"

    def test_margin_is_computable_and_sane(self):
        """
        Le test de bout en bout du bug : marge nette = résultat / chiffre
        d'affaires. Avec le bug, UFPT affichait 44 % ; sur les mêmes bases,
        elle doit être plausible.
        """
        tags = {}
        tags.update(_four_quarters(
            "RevenueFromContractWithCustomerIncludingAssessedTax",
            [100, 100, 100, 100],
            ["2024-05-01", "2024-08-01", "2024-11-01", "2025-02-01"]))
        tags.update(_four_quarters(
            "NetIncomeLoss", [11, 11, 11, 11],
            ["2024-05-01", "2024-08-01", "2024-11-01", "2025-02-01"]))
        facts = _FakeClient(_payload(tags)).as_of(
            "X", date(2025, 3, 1), ["revenue", "net_income"])
        marge = facts["net_income"].value / facts["revenue"].value
        assert marge == pytest.approx(0.11), \
            "les deux grandeurs doivent couvrir la même durée"


# ── Robustesse ────────────────────────────────────────────────────────────────

class TestTagFallback:

    def test_falls_back_to_alternate_revenue_tag(self):
        assert len(CONCEPTS["revenue"]) > 1
        c = _FakeClient(_payload(_four_quarters(
            "Revenues", [1_000, 1_000, 1_000, 2_000],
            ["2024-05-01", "2024-08-01", "2024-11-01", "2025-02-01"])))
        f = c.as_of("X", date(2025, 3, 1), ["revenue"])["revenue"]
        assert f.value == 5_000
        assert f.tag == "Revenues"

    def test_unknown_ticker_returns_empty_not_crash(self):
        c = _FakeClient(None)
        assert c.as_of("INCONNU", date(2024, 6, 1)) == {}
        assert c.facts("INCONNU").empty

    def test_malformed_observation_is_skipped(self):
        """Une observation sans date de dépôt est inexploitable : on l'ignore."""
        payload = _payload({"Assets": [_instant("2024-03-31", 10, "2024-05-01")]})
        payload["facts"]["us-gaap"]["Assets"]["units"]["USD"].append(
            {"end": "2024-06-30", "val": 20})          # pas de `filed`
        f = _FakeClient(payload).as_of("X", date(2025, 1, 1), ["assets"])["assets"]
        assert f.value == 10, "le fait sans date de dépôt ne peut pas être daté"


class TestMultiTagCoverage:
    """
    Régression : s'arrêter à la première étiquette trouvée faisait disparaître
    des sociétés. Constaté sur UFPT — 25 observations sous l'étiquette
    prioritaire, 161 sous la suivante, résultat : chiffre d'affaires absent.
    """

    def test_sparse_priority_tag_does_not_hide_dense_fallback(self):
        tags = {"RevenueFromContractWithCustomerExcludingAssessedTax": [
            _quarter("2019-03-31", 111, "2019-05-01")]}
        tags.update(_four_quarters(
            "RevenueFromContractWithCustomerIncludingAssessedTax",
            [200, 200, 200, 400],
            ["2024-05-01", "2024-08-01", "2024-11-01", "2025-02-01"]))
        f = _FakeClient(_payload(tags)).as_of(
            "UFPT", date(2025, 3, 1), ["revenue"])["revenue"]
        assert f.value == 1_000, \
            "l'étiquette secondaire doit être lue quand la prioritaire est muette"

    def test_accounting_standard_change_is_bridged(self):
        """
        Le passage à ASC 606 en 2018 a fait changer d'étiquette presque tout le
        monde. Les deux doivent être lues pour couvrir l'historique complet.
        """
        tags = {
            "SalesRevenueNet": [_annual("2017-12-31", 100, "2018-02-01")],
            "RevenueFromContractWithCustomerExcludingAssessedTax": [
                _annual("2024-12-31", 300, "2025-02-01")],
        }
        c = _FakeClient(_payload(tags))
        avant = c.as_of("X", date(2018, 6, 1), ["revenue"])["revenue"]
        apres = c.as_of("X", date(2025, 3, 1), ["revenue"])["revenue"]
        assert (avant.value, avant.tag) == (100, "SalesRevenueNet")
        assert apres.value == 300

    def test_priority_tag_wins_when_both_cover_the_period(self):
        """À période et dépôt identiques, l'étiquette prioritaire tranche."""
        tags = {
            "RevenueFromContractWithCustomerExcludingAssessedTax": [
                _annual("2024-12-31", 1_000, "2025-02-01")],
            "Revenues": [_annual("2024-12-31", 1_050, "2025-02-01")],
        }
        f = _FakeClient(_payload(tags)).as_of(
            "X", date(2025, 3, 1), ["revenue"])["revenue"]
        assert f.value == 1_000, "l'ordre de CONCEPTS doit être respecté"


class TestUniverseFrame:

    def test_stale_days_exposes_outdated_accounts(self):
        """
        Comparer une société qui vient de publier avec une autre muette depuis
        un an, sans le signaler, revient à mélanger du frais et du périmé.
        """
        c = _FakeClient(_payload({"Assets": [
            _instant("2023-03-31", 1_000, "2023-05-01")]}))
        df = c.as_of_frame(["X"], date(2024, 5, 1), ["assets"])
        assert df.loc[0, "_stale_days"] == (date(2024, 5, 1) - date(2023, 3, 31)).days

    def test_company_without_data_is_absent_not_null(self):
        assert _FakeClient(None).as_of_frame(["X", "Y"], date(2024, 5, 1)).empty
