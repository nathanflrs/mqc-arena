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
    forward_log_returns, label_success, signed_return_edge,
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


# ── Rendement signé : la forme du gain, pas seulement sa fréquence ────────────

class TestSignedReturnEdge:
    """
    `signed_return_edge` existe parce que le taux de réussite suppose que tous
    les succès se valent. C'est faux dès qu'une stratégie est asymétrique.
    Ces tests vérifient qu'elle voit ce que le taux de réussite manque.
    """

    def test_short_signal_gains_when_price_falls(self):
        """Convention de signe : un SELL correct doit compter POSITIVEMENT."""
        data = _data(drift=-0.004)          # marché baissier franc
        sig = _signals("Bear", "SELL")
        e = signed_return_edge(sig, data, SYMS, horizons={"H5": 5})[0]
        assert e.mean_signed > 0, \
            "un short sur un marché qui baisse doit afficher un rendement positif"
        assert e.ci_lo > 0

    def test_rare_big_wins_are_detected_where_hit_rate_fails(self):
        """
        Le profil exact d'un suiveur de tendance : rarement raison, mais des
        gains bien plus grands que les pertes. Le taux de réussite le condamne,
        le rendement signé le sauve — c'est toute la raison d'être du module.
        """
        dates = pd.date_range("2024-01-01", periods=300, freq="B")
        # 20 % de hausses de +10 %, 80 % de baisses de −1 % → espérance positive
        rng = np.random.default_rng(7)
        steps = np.where(rng.random(len(dates)) < 0.20, 0.10, -0.01)
        close = 100 * np.cumprod(1 + steps)
        data = {"AAA": pd.DataFrame({"Close": close}, index=dates)}
        sig = pd.DataFrame(
            [(d, "AAA", "Trend", "BUY", 0.8, 0.1) for d in dates[:250]],
            columns=["date", "symbol", "agent", "action", "confidence",
                     "target_weight"])

        hit = compute_agent_edge(sig, data, ["AAA"], horizons={"H1": 1})[0]
        ret = signed_return_edge(sig, data, ["AAA"], horizons={"H1": 1})[0]

        assert hit.hit_rate < 0.35, "l'agent a rarement raison"
        assert ret.mean_signed > 0, "et gagne pourtant de l'argent"
        assert ret.win_loss_ratio > 5, "parce que ses gains écrasent ses pertes"
        assert ret.skew > 0, "profil convexe attendu"

    def test_flags_positive_expectancy_that_trails_passive(self):
        """
        Le verdict le plus utile du module : une espérance positive n'est pas
        une création de valeur si un dollar simplement investi long fait mieux.
        """
        data = _data(drift=0.004)          # marché haussier
        # Profil du CTA réel : majoritairement long, mais short une fois sur
        # quatre. Sur un marché qui monte, ces shorts rognent l'espérance par
        # signal sans la faire passer sous zéro — exactement le cas que le
        # verdict doit savoir nommer. (Alterner BUY/HOLD ne marcherait pas :
        # émettre moins de signaux ne change pas le rendement de chacun.)
        rows = [(d, s, "Mixte", "SELL" if i % 4 == 0 else "BUY", 0.8, 0.1)
                for i, d in enumerate(DATES[:300]) for s in SYMS]
        sig = pd.DataFrame(rows, columns=[
            "date", "symbol", "agent", "action", "confidence", "target_weight"])

        e = signed_return_edge(sig, data, SYMS, horizons={"H5": 5})[0]
        assert e.mean_signed > 0
        assert e.passive_mean > 0
        assert "inférieure au passif" in e.verdict

    def test_bootstrap_groups_by_date_not_by_signal(self):
        """
        Même correction statistique que pour le taux de réussite : trois actifs
        d'une même journée valent une observation, pas trois. Ajouter des actifs
        corrélés ne doit pas resserrer artificiellement l'intervalle.
        """
        dates = pd.date_range("2024-01-01", periods=250, freq="B")
        rng = np.random.default_rng(3)
        base = rng.normal(0.0005, 0.01, len(dates))

        def build(n_sym):
            data, rows = {}, []
            for k in range(n_sym):
                # Actifs parfaitement corrélés : aucune information nouvelle.
                data[f"S{k}"] = pd.DataFrame(
                    {"Close": 100 * np.cumprod(1 + base)}, index=dates)
                rows += [(d, f"S{k}", "A", "BUY", 0.8, 0.1) for d in dates[:200]]
            sig = pd.DataFrame(rows, columns=[
                "date", "symbol", "agent", "action", "confidence",
                "target_weight"])
            e = signed_return_edge(sig, data, list(data), horizons={"H5": 5})[0]
            return e.ci_hi - e.ci_lo

        width_1, width_5 = build(1), build(5)
        assert width_5 == pytest.approx(width_1, rel=0.10), \
            "cinq actifs identiques ne doivent pas diviser l'intervalle par √5"

    def test_insufficient_sample_refuses_to_conclude(self):
        data = _data(drift=0.003)
        sig = _signals("Court", "BUY", n_dates=20)
        e = signed_return_edge(sig, data, SYMS, horizons={"H5": 5})[0]
        assert "échantillon insuffisant" in e.verdict
        assert not e.is_significant


# ── Chevauchement des fenêtres ────────────────────────────────────────────────

class TestBlockBootstrap:
    """
    Régression du 2026-08-14, la plus coûteuse du projet.

    Un rendement à H jours mesuré chaque jour partage H−1 jours avec le
    précédent. Les tirer indépendamment revient à compter la même information H
    fois, et divise l'intervalle par la racine d'un effectif fictif.

    Deux résultats « significatifs » n'y ont pas survécu : le momentum
    long/short, dont l'IC passait de [+0.22 %, +0.82 %] à [−0.85 %, +1.89 %],
    et l'hypothèse sur les régularisations comptables.
    """

    def test_blocks_are_contiguous(self):
        """Contigus modulo n : la série est refermée en anneau."""
        from src.analysis.agent_edge import block_bootstrap_indices
        rng = np.random.default_rng(0)
        n = 100
        idx = block_bootstrap_indices(n_dates=n, block=20, n_boot=5, rng=rng)
        for ligne in idx:
            bloc = ligne[:20]
            attendu = [(bloc[0] + k) % n for k in range(20)]
            assert list(bloc) == attendu

    def test_every_observation_carries_the_same_weight(self):
        """
        Régression du 2026-08-14, seconde passe. La première version tirait les
        débuts dans [0, n−H] : la position 0 était 23 fois moins échantillonnée
        que le centre, et le symptôme était une moyenne tombant HORS de son
        propre intervalle de confiance.
        """
        from src.analysis.agent_edge import block_bootstrap_indices
        n, block = 70, 20
        idx = block_bootstrap_indices(n, block, 20_000, np.random.default_rng(0))
        poids = np.bincount(idx.ravel(), minlength=n) / idx.size * n
        assert poids.max() / poids.min() < 1.15, \
            f"échantillonnage non uniforme : {poids.min():.2f} à {poids.max():.2f}"

    def test_bootstrap_mean_brackets_the_sample_mean(self):
        """
        Le test qui aurait attrapé le défaut tout de suite : la moyenne de la
        distribution bootstrap doit coïncider avec la moyenne de l'échantillon.
        """
        from src.analysis.agent_edge import block_bootstrap_indices
        rng = np.random.default_rng(11)
        # Série à tendance marquée : les extrémités portent les valeurs
        # extrêmes, donc les sous-pondérer se voit immédiatement.
        serie = np.linspace(-1.0, 1.0, 80) + rng.normal(0, 0.05, 80)
        idx = block_bootstrap_indices(80, 20, 5000, np.random.default_rng(2))
        boot = serie[idx].mean(axis=1)
        assert abs(boot.mean() - serie.mean()) < 0.02
        lo, hi = np.percentile(boot, [2.5, 97.5])
        assert lo <= serie.mean() <= hi

    def test_indices_stay_in_range(self):
        from src.analysis.agent_edge import block_bootstrap_indices
        rng = np.random.default_rng(0)
        idx = block_bootstrap_indices(n_dates=37, block=20, n_boot=50, rng=rng)
        assert idx.shape == (50, 37)
        assert idx.min() >= 0 and idx.max() <= 36

    def test_block_of_one_is_the_old_behaviour(self):
        """À horizon 1 il n'y a pas de chevauchement : rien ne doit changer."""
        from src.analysis.agent_edge import block_bootstrap_indices
        a = block_bootstrap_indices(50, 1, 10, np.random.default_rng(1))
        b = np.random.default_rng(1).integers(0, 50, size=(10, 50))
        assert (a == b).all()

    def test_overlap_widens_the_interval(self):
        """
        Le point qui compte. Sur une série autocorrélée — ce que produit une
        fenêtre glissante — le bootstrap par blocs doit donner un intervalle
        PLUS LARGE que le tirage indépendant, parce qu'il ne fait pas semblant
        d'avoir plus d'information qu'il n'y en a.
        """
        from src.analysis.agent_edge import block_bootstrap_indices
        rng = np.random.default_rng(3)
        base = rng.normal(0.005, 0.02, 400)
        # Moyenne glissante sur 20 : chaque point partage 19 valeurs avec le
        # suivant, exactement comme un rendement forward mesuré quotidiennement.
        serie = pd.Series(base).rolling(20).mean().dropna().to_numpy()

        def largeur(block):
            idx = block_bootstrap_indices(len(serie), block, 2000, np.random.default_rng(7))
            b = serie[idx].mean(axis=1)
            lo, hi = np.percentile(b, [2.5, 97.5])
            return hi - lo

        assert largeur(20) > largeur(1) * 1.5, \
            "le bootstrap par blocs doit refléter la dépendance, pas l'ignorer"

    def test_horizon_is_actually_passed_through(self):
        """
        Sans ce test, une refonte pourrait perdre le paramètre en silence et
        rétablir des intervalles trop étroits sans que rien ne le signale.
        """
        data = _data(drift=0.002)
        sig = _signals("A", "BUY", n_dates=300)
        e1 = compute_agent_edge(sig, data, SYMS, horizons={"H": 1})[0]
        e20 = compute_agent_edge(sig, data, SYMS, horizons={"H": 20})[0]
        assert (e20.ci_hi - e20.ci_lo) != (e1.ci_hi - e1.ci_lo)
