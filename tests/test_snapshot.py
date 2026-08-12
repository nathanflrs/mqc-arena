"""
Snapshots de données figés et détection de réécriture (src/data/snapshot.py).

`yfinance` avec auto_adjust=True réécrit rétroactivement tout l'historique à
chaque dividende ou split. Mesuré le 2026-08-02 sur 3 ans : la plus ancienne
barre de GS bouge de −6.79 %, JPM de −6.39 %. C'est ce qui a fait passer
l'alpha de portfolio_backtest.py de +10.7 pts à −12.5 pts entre deux runs
distants de dix jours — sans le moindre changement de code.

Aucun accès réseau : toutes les séries sont synthétiques.
"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

from src.data.snapshot import (
    MANIFEST_NAME, RESTATEMENT_EPSILON, SnapshotManifest, diff_snapshots,
    load_snapshot, write_snapshot,
)


def _df(n=100, start=100.0, seed=0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    close = start * np.cumprod(1 + rng.normal(0, 0.01, n))
    idx = pd.date_range("2024-01-01", periods=n, freq="B")
    return pd.DataFrame(
        {"Open": close * 0.999, "High": close * 1.01, "Low": close * 0.99,
         "Close": close, "Volume": np.full(n, 1e6)},
        index=idx,
    )


@pytest.fixture
def data():
    return {"AAPL": _df(seed=1), "MSFT": _df(seed=2), "SPY": _df(seed=3)}


# ── Écriture et relecture ─────────────────────────────────────────────────────

class TestRoundTrip:
    def test_write_then_load_returns_identical_frames(self, tmp_path, data):
        write_snapshot(data, tmp_path)
        loaded, _, tampered = load_snapshot(tmp_path)
        assert tampered == []
        for sym in data:
            # check_freq=False : parquet ne conserve pas l'attribut `freq` de
            # l'index. C'est une métadonnée inférée, pas de la donnée — et les
            # séries yfinance réelles arrivent de toute façon avec freq=None.
            pd.testing.assert_frame_equal(loaded[sym], data[sym], check_freq=False)

    def test_manifest_records_provenance(self, tmp_path, data):
        m = write_snapshot(data, tmp_path, period="5y", auto_adjust=True)
        assert m.period == "5y" and m.auto_adjust is True and m.source == "yfinance"
        assert sorted(m.symbols) == ["AAPL", "MSFT", "SPY"]
        assert m.created_at.startswith("20")

    def test_manifest_records_shape_and_span(self, tmp_path, data):
        m = write_snapshot(data, tmp_path)
        assert m.n_rows["AAPL"] == 100
        assert m.first_date["AAPL"] == "2024-01-01"
        assert m.last_date["AAPL"] == str(data["AAPL"].index[-1].date())

    def test_manifest_is_valid_json_on_disk(self, tmp_path, data):
        write_snapshot(data, tmp_path)
        parsed = json.loads((tmp_path / MANIFEST_NAME).read_text())
        assert set(parsed["symbols"]) == set(data)

    def test_missing_manifest_is_an_error_not_a_silent_empty(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            load_snapshot(tmp_path)


# ── Intégrité ─────────────────────────────────────────────────────────────────

class TestIntegrity:
    def test_tampered_file_is_reported(self, tmp_path, data):
        write_snapshot(data, tmp_path)
        altered = data["AAPL"].copy()
        altered.iloc[10, altered.columns.get_loc("Close")] *= 1.05
        altered.to_parquet(tmp_path / "AAPL.parquet")

        _, _, tampered = load_snapshot(tmp_path)
        assert tampered == ["AAPL"], "un snapshot modifié après coup ne vaut pas mieux que le réseau"

    def test_deleted_file_is_reported(self, tmp_path, data):
        write_snapshot(data, tmp_path)
        (tmp_path / "MSFT.parquet").unlink()
        _, _, tampered = load_snapshot(tmp_path)
        assert "MSFT" in tampered

    def test_verification_can_be_skipped(self, tmp_path, data):
        write_snapshot(data, tmp_path)
        altered = data["AAPL"].copy()
        altered.iloc[10, altered.columns.get_loc("Close")] *= 1.05
        altered.to_parquet(tmp_path / "AAPL.parquet")
        _, _, tampered = load_snapshot(tmp_path, verify=False)
        assert tampered == []

    def test_volume_revision_alone_does_not_trip_the_checksum(self, tmp_path, data):
        """
        yfinance révise fréquemment le volume sans que cela change une décision.
        L'inclure dans l'empreinte ferait crier au loup à chaque téléchargement.
        """
        write_snapshot(data, tmp_path)
        d = data["AAPL"].copy()
        d["Volume"] = d["Volume"] * 1.3
        d.to_parquet(tmp_path / "AAPL.parquet")
        _, _, tampered = load_snapshot(tmp_path)
        assert tampered == []


# ── Détection de réécriture ───────────────────────────────────────────────────

class TestRestatementDetection:
    def test_identical_data_is_clean(self, data):
        assert diff_snapshots(data, data).is_clean

    def test_dividend_adjustment_is_detected(self, data):
        """
        Un dividende détaché rescale tout l'historique antérieur. C'est le cas
        réel : sur 3 ans, GS voit sa plus ancienne barre bouger de −6.79 %.
        """
        new = {k: v.copy() for k, v in data.items()}
        new["AAPL"] = new["AAPL"].copy()
        new["AAPL"].iloc[:50, new["AAPL"].columns.get_loc("Close")] *= 0.995

        report = diff_snapshots(data, new)
        assert not report.is_clean
        r = report.restated[0]
        assert r.symbol == "AAPL"
        assert r.n_bars_changed == 50
        assert r.max_abs_change == pytest.approx(0.005, rel=1e-3)
        assert r.first_changed_date == "2024-01-01"

    def test_unchanged_symbols_are_listed_separately(self, data):
        new = {k: v.copy() for k, v in data.items()}
        new["AAPL"] = new["AAPL"] * 1.01
        report = diff_snapshots(data, new)
        assert set(report.unchanged) == {"MSFT", "SPY"}

    def test_new_bars_are_not_a_restatement(self, data):
        """Le marché a simplement continué : ce n'est pas une réécriture."""
        new = {k: v.copy() for k, v in data.items()}
        extra = _df(n=105, seed=1)
        new["AAPL"] = pd.concat([data["AAPL"], extra.iloc[100:]])
        report = diff_snapshots(data, new)
        assert report.is_clean
        assert "AAPL" in report.unchanged

    def test_added_bars_are_counted_alongside_a_restatement(self, data):
        new = {k: v.copy() for k, v in data.items()}
        a = data["AAPL"].copy()
        a["Close"] *= 0.99
        tail = _df(n=103, seed=1).iloc[100:]
        new["AAPL"] = pd.concat([a, tail])
        r = diff_snapshots(data, new).restated[0]
        assert r.rows_added == 3
        assert r.n_bars_changed == 100

    def test_missing_symbol_is_flagged(self, data):
        new = {k: v for k, v in data.items() if k != "SPY"}
        report = diff_snapshots(data, new)
        assert report.missing_symbols == ["SPY"] and not report.is_clean

    def test_added_symbol_is_not_a_problem(self, data):
        new = dict(data); new["NVDA"] = _df(seed=9)
        report = diff_snapshots(data, new)
        assert report.new_symbols == ["NVDA"] and report.is_clean

    def test_float_noise_below_epsilon_is_ignored(self, data):
        new = {k: v.copy() for k, v in data.items()}
        new["AAPL"] = new["AAPL"] * (1 + RESTATEMENT_EPSILON / 10)
        assert diff_snapshots(data, new).is_clean

    def test_report_ranks_by_severity(self, data):
        new = {k: v.copy() for k, v in data.items()}
        new["AAPL"] = new["AAPL"] * 1.001
        new["MSFT"] = new["MSFT"] * 1.05
        order = [r.symbol for r in diff_snapshots(data, new).restated]
        assert order == ["MSFT", "AAPL"]

    def test_report_warns_that_backtests_will_not_reproduce(self, data):
        new = {k: v.copy() for k, v in data.items()}
        new["AAPL"] = new["AAPL"] * 1.01
        assert "ne reproduira pas" in diff_snapshots(data, new).render()


# ── Reproductibilité, la raison d'être du module ──────────────────────────────

class TestReproducibility:
    def test_a_snapshot_makes_a_backtest_repeatable(self, tmp_path, data):
        """
        Deux chargements successifs du même snapshot doivent donner des séries
        strictement identiques — c'est la propriété que le réseau ne fournit
        pas et pour laquelle ce module existe.
        """
        write_snapshot(data, tmp_path)
        first, _, _ = load_snapshot(tmp_path)
        second, _, _ = load_snapshot(tmp_path)
        for sym in data:
            pd.testing.assert_frame_equal(first[sym], second[sym], check_freq=False)
        assert diff_snapshots(first, second).is_clean
