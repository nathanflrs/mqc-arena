"""
Mesure d'edge par agent (src/analysis/agent_edge.py).

Les deux corrections statistiques par rapport à docs/edge_audit.md sont
l'objet principal de ces tests :
  1. l'hypothèse nulle est le taux de base inconditionnel, pas 0,5 ;
  2. l'intervalle est bootstrappé sur les DATES, pas sur les signaux.

Un test qui ne vérifierait que « la fonction renvoie un nombre » laisserait
passer les deux erreurs qu'on cherche justement à corriger.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.analysis.agent_edge import (
    MATERIALITY, base_rates, calibration_curve, compute_agent_edge,
    forward_log_returns, label_success,
)

DATES = pd.date_range("2024-01-01", periods=400, freq="B")
SYMS = ["AAA", "BBB", "CCC"]


def _data(drift=0.0):
    out = {}
    for i, s in enumerate(SYMS):
        rng = np.random.default_rng(i)
        close = 100 * np.cumprod(1 + rng.normal(drift, 0.01, len(DATES)))
        out[s] = pd.DataFrame({"Close": close}, index=DATES)
    return out


def _signals(agent, action, n_dates=300, conf=0.8):
    rows = [(d, s, agent, action, conf, 0.1)
            for d in DATES[:n_dates] for s in SYMS]
    return pd.DataFrame(rows, columns=[
        "date", "symbol", "agent", "action", "confidence", "target_weight"])


# ── Briques de base ───────────────────────────────────────────────────────────

class TestLabelling:
    def test_buy_needs_to_clear_materiality(self):
        assert label_success("BUY", MATERIALITY * 2) is True
        assert label_success("BUY", MATERIALITY / 2) is False, \
            "un mouvement sous le seuil est absorbé par la friction"

    def test_sell_is_symmetric(self):
        assert label_success("SELL", -MATERIALITY * 2) is True
        assert label_success("SELL", MATERIALITY * 2) is False

    def test_hold_is_not_judged_here(self):
        assert label_success("HOLD", 0.0) is None

    def test_nan_is_not_judged(self):
        assert label_success("BUY", np.nan) is None

    def test_forward_return_looks_forward_only(self):
        s = pd.Series([100.0, 110.0, 121.0], index=DATES[:3])
        fwd = forward_log_returns(s, 1)
        assert fwd.iloc[0] == pytest.approx(np.log(1.1))
        assert np.isnan(fwd.iloc[-1]), "la dernière barre n'a pas de futur"


# ── Correction 1 : l'hypothèse nulle est le taux de base ──────────────────────

class TestBaseRateIsTheNullHypothesis:
    def test_base_rate_rises_with_market_drift(self):
        flat_buy, _ = base_rates(_data(drift=0.0), SYMS, DATES, 5)
        bull_buy, _ = base_rates(_data(drift=0.003), SYMS, DATES, 5)
        assert bull_buy > flat_buy + 0.1

    def test_always_buy_in_a_bull_market_shows_no_edge(self):
        """
        Le test central. Un agent qui dit toujours BUY sur un marché haussier
        affiche un taux de succès élevé. Comparé à 0,5 il paraîtrait brillant ;
        comparé au taux de base, son excès est nul. C'est exactement l'erreur
        que faisait edge_audit.
        """
        data = _data(drift=0.003)
        edges = compute_agent_edge(_signals("Perroquet", "BUY"), data, SYMS,
                                   horizons={"H5": 5})
        e = edges[0]
        assert e.hit_rate > 0.55, "le marché monte, le taux brut est flatteur"
        assert abs(e.excess) < 0.05, "mais l'excès sur le taux de base est nul"
        assert not e.is_significant
        assert "indistinguable" in e.verdict

    def test_expected_rate_follows_the_agent_action_mix(self):
        data = _data(drift=0.003)
        buy = compute_agent_edge(_signals("A", "BUY"), data, SYMS, {"H5": 5})[0]
        sell = compute_agent_edge(_signals("B", "SELL"), data, SYMS, {"H5": 5})[0]
        assert buy.expected_rate > sell.expected_rate, \
            "sur un marché haussier, réussir un SELL est plus dur qu'un BUY"


# ── Correction 2 : bootstrap sur les dates ────────────────────────────────────

class TestDateClusteredInterval:
    def test_interval_brackets_the_excess(self):
        e = compute_agent_edge(_signals("A", "BUY"), _data(), SYMS, {"H5": 5})[0]
        assert e.ci_lo <= e.excess <= e.ci_hi

    def test_more_symbols_same_dates_does_not_shrink_the_interval(self):
        """
        Le point qui distingue ce module d'edge_audit. Passer de 3 à 9 symboles
        sur les MÊMES dates triple le nombre de signaux sans ajouter la moindre
        observation de marché indépendante : l'intervalle ne doit pas se
        resserrer sensiblement. Un intervalle de Wilson sur le nombre de
        signaux l'aurait divisé par √3.
        """
        base = _data()
        wide = dict(base)
        for k in range(6):
            wide[f"X{k}"] = base[SYMS[k % 3]].copy()

        narrow_syms, wide_syms = SYMS, list(wide)
        s_narrow = _signals("A", "BUY")
        rows = [(d, s, "A", "BUY", 0.8, 0.1) for d in DATES[:300] for s in wide_syms]
        s_wide = pd.DataFrame(rows, columns=s_narrow.columns)

        e1 = compute_agent_edge(s_narrow, base, narrow_syms, {"H5": 5})[0]
        e2 = compute_agent_edge(s_wide, wide, wide_syms, {"H5": 5})[0]

        assert e2.n_signals == 3 * e1.n_signals
        assert e2.n_dates == e1.n_dates
        w1, w2 = e1.ci_hi - e1.ci_lo, e2.ci_hi - e2.ci_lo
        assert w2 > w1 * 0.6, \
            "tripler les signaux sans ajouter de dates ne doit pas resserrer l'IC de √3"

    def test_underpowered_sample_is_refused(self):
        e = compute_agent_edge(_signals("A", "BUY", n_dates=20), _data(), SYMS,
                               {"H5": 5})[0]
        assert "insuffisant" in e.verdict

    def test_bootstrap_is_deterministic(self):
        a = compute_agent_edge(_signals("A", "BUY"), _data(), SYMS, {"H5": 5})[0]
        b = compute_agent_edge(_signals("A", "BUY"), _data(), SYMS, {"H5": 5})[0]
        assert (a.ci_lo, a.ci_hi) == (b.ci_lo, b.ci_hi)


# ── Détection d'un vrai edge ──────────────────────────────────────────────────

class TestRealEdgeIsDetected:
    def test_a_clairvoyant_agent_is_flagged_significant(self):
        """Contrôle positif : sans lui, un module qui ne détecte jamais rien passerait."""
        data = _data()
        fwd = {s: forward_log_returns(data[s]["Close"], 5) for s in SYMS}
        rows = []
        for d in DATES[:300]:
            for s in SYMS:
                f = fwd[s].get(d, np.nan)
                if not np.isfinite(f):
                    continue
                rows.append((d, s, "Oracle", "BUY" if f > 0 else "SELL", 0.9, 0.1))
        edges = compute_agent_edge(
            pd.DataFrame(rows, columns=["date", "symbol", "agent", "action",
                                        "confidence", "target_weight"]),
            data, SYMS, {"H5": 5})
        assert edges[0].excess > 0.20
        assert edges[0].is_significant and "edge mesurable" in edges[0].verdict

    def test_an_inverted_agent_is_flagged_as_anti_edge(self):
        data = _data()
        fwd = {s: forward_log_returns(data[s]["Close"], 5) for s in SYMS}
        rows = []
        for d in DATES[:300]:
            for s in SYMS:
                f = fwd[s].get(d, np.nan)
                if not np.isfinite(f):
                    continue
                rows.append((d, s, "Inverse", "SELL" if f > 0 else "BUY", 0.9, 0.1))
        e = compute_agent_edge(
            pd.DataFrame(rows, columns=["date", "symbol", "agent", "action",
                                        "confidence", "target_weight"]),
            data, SYMS, {"H5": 5})[0]
        assert e.excess < -0.20 and "anti-edge" in e.verdict


# ── Calibration ───────────────────────────────────────────────────────────────

class TestCalibration:
    def test_curve_is_binned_by_confidence(self):
        s = pd.concat([_signals("A", "BUY", conf=0.6), _signals("A", "BUY", conf=0.9)])
        c = calibration_curve(s, _data(), SYMS, "A", horizon=5)
        assert len(c) >= 2
        assert c["mean_confidence"].is_monotonic_increasing

    def test_curve_reports_dates_not_only_signals(self):
        c = calibration_curve(_signals("A", "BUY"), _data(), SYMS, "A", horizon=5)
        assert "n_dates" in c.columns and (c["n_dates"] <= c["n"]).all()

    def test_unknown_agent_yields_empty_curve(self):
        assert calibration_curve(_signals("A", "BUY"), _data(), SYMS,
                                 "Absent", horizon=5).empty
