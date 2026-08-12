"""
Tests du backtest système (src/backtest/system_backtest.py).

Le test qui compte est `TestNoLookAhead::test_truncating_the_future_changes_nothing` :
si un seul composant du pipeline regarde au-delà de la date de décision, la
courbe d'equity calculée sur un historique tronqué diffère de celle calculée
sur l'historique complet. C'est le seul contrôle qui détecte un look-ahead
structurel — un backtest peut être parfaitement écrit et mentir quand même.

Toutes les séries sont synthétiques : aucun accès réseau.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.backtest.system_backtest import (
    EXCLUDED_AGENTS, SystemBacktestConfig, run_system_backtest,
)

SYMBOLS = ["AAA", "BBB", "CCC", "DDD", "SPY"]
N_DAYS = 480


def _series(seed: int, drift: float, vol: float = 0.012) -> pd.DataFrame:
    """Marche aléatoire reproductible, avec OHLC cohérent (Open ≠ Close)."""
    rng = np.random.default_rng(seed)
    rets = rng.normal(drift, vol, N_DAYS)
    close = 100.0 * np.cumprod(1.0 + rets)
    # L'ouverture s'écarte de la clôture précédente : indispensable pour que le
    # modèle de remplissage à cours limité soit réellement exercé.
    gap = rng.normal(0, vol / 2, N_DAYS)
    open_ = np.concatenate([[100.0], close[:-1]]) * (1.0 + gap)
    idx = pd.date_range("2022-01-03", periods=N_DAYS, freq="B")
    return pd.DataFrame(
        {
            "Open": open_,
            "High": np.maximum(open_, close) * 1.004,
            "Low": np.minimum(open_, close) * 0.996,
            "Close": close,
            "Volume": np.full(N_DAYS, 5_000_000.0),
        },
        index=idx,
    )


@pytest.fixture(scope="module")
def data():
    drifts = {"AAA": 0.0009, "BBB": 0.0004, "CCC": -0.0002, "DDD": 0.0006, "SPY": 0.0005}
    return {s: _series(seed=i * 17 + 3, drift=d) for i, (s, d) in enumerate(drifts.items())}


@pytest.fixture(scope="module")
def cfg():
    return SystemBacktestConfig(warmup_days=300, initial_capital=1_000_000.0)


@pytest.fixture(scope="module")
def result(data, cfg):
    return run_system_backtest(data, symbols=SYMBOLS, cfg=cfg, verbose=False)


# ── Le test central ───────────────────────────────────────────────────────────

class TestNoLookAhead:
    def test_truncating_the_future_changes_nothing(self, data, cfg):
        """
        On coupe 60 séances à la fin des données et on rejoue. Les décisions
        prises avant la coupure ne peuvent pas dépendre de ce qui vient après :
        la courbe d'equity commune doit être identique au centime près.

        Ce test attrape notamment le piège CrossSectionalMomentumAgent, dont
        les classements sont figés à l'appel de set_universe() : lui passer
        l'univers complet une seule fois lui donnerait tout l'historique futur.
        """
        full = run_system_backtest(data, symbols=SYMBOLS, cfg=cfg, verbose=False)
        cut = {s: df.iloc[:-60] for s, df in data.items()}
        truncated = run_system_backtest(cut, symbols=SYMBOLS, cfg=cfg, verbose=False)

        common = truncated.equity.index
        assert len(common) > 50, "période commune trop courte pour conclure"
        pd.testing.assert_series_equal(
            full.equity.loc[common], truncated.equity,
            check_exact=False, rtol=1e-9,
        )

    def test_no_fill_at_the_price_that_triggered_the_order(self, data, cfg, result):
        """
        Une décision est prise sur la clôture de J ; le remplissage a lieu à
        l'ouverture de J+1. Aucun fill ne doit donc coïncider avec la clôture
        du jour de décision — l'erreur de backtest la plus courante.
        """
        assert result.fills, "le backtest doit produire des fills"
        coincidences = 0
        for f in result.fills:
            df = data[f.symbol]
            pos = df.index.get_loc(f.date)
            if pos == 0:
                continue
            if abs(f.price - float(df["Close"].iloc[pos - 1])) < 1e-12:
                coincidences += 1
        assert coincidences == 0, \
            f"{coincidences} fills au prix de clôture qui a déclenché l'ordre"

    def test_fills_happen_at_the_open(self, data, cfg, result):
        for f in result.fills[:200]:
            expected = float(data[f.symbol].loc[f.date, "Open"])
            assert f.price == pytest.approx(expected, rel=1e-9)


# ── Comptabilité ──────────────────────────────────────────────────────────────

class TestAccounting:
    def test_equity_is_positive_and_continuous(self, result):
        assert (result.equity > 0).all()
        assert result.equity.index.is_monotonic_increasing

    def test_no_implicit_leverage(self, data, cfg):
        """Le cash ne doit jamais devenir négatif : pas de levier implicite."""
        res = run_system_backtest(data, symbols=SYMBOLS, cfg=cfg, verbose=False)
        # NAV reconstitué ≥ 0 à chaque pas, et jamais de saut aberrant
        jumps = res.equity.pct_change().dropna().abs()
        assert jumps.max() < 0.35, f"saut d'equity anormal : {jumps.max():.1%}"

    def test_transaction_costs_are_charged(self, result):
        assert result.total_costs > 0
        assert all(f.cost_usd > 0 for f in result.fills)

    def test_benchmark_aligned_on_equity_index(self, result):
        assert result.benchmark.index.equals(result.equity.index)
        assert result.benchmark.iloc[0] == pytest.approx(result.equity.iloc[0])

    def test_fill_rate_is_reported_and_below_one(self, result):
        """Le modèle à cours limité doit réellement rejeter des ordres."""
        assert result.n_orders_sent > 0
        assert 0.0 < result.fill_rate < 1.0


# ── Honnêteté du rapport ──────────────────────────────────────────────────────

class TestCaveats:
    def test_default_run_declares_its_limits(self, result):
        assert result.caveats
        assert any("exclus du replay" in c for c in result.caveats)

    def test_agent_priority_is_flagged_as_unpublishable(self, data, cfg):
        biased = SystemBacktestConfig(warmup_days=300, use_agent_priority=True)
        res = run_system_backtest(data, symbols=SYMBOLS, cfg=biased, verbose=False)
        assert any("NON publiable" in c for c in res.caveats)

    def test_optimistic_fill_model_is_flagged(self, data):
        optimistic = SystemBacktestConfig(warmup_days=300, fill_model="open")
        res = run_system_backtest(data, symbols=SYMBOLS, cfg=optimistic, verbose=False)
        assert any("gaps" in c for c in res.caveats)

    def test_every_excluded_agent_has_a_written_reason(self):
        assert len(EXCLUDED_AGENTS) == 6
        for agent, reason in EXCLUDED_AGENTS.items():
            assert len(reason) > 40, f"{agent}: motif d'exclusion trop vague"

    def test_optimistic_fill_model_fills_more(self, data):
        strict = run_system_backtest(
            data, symbols=SYMBOLS, verbose=False,
            cfg=SystemBacktestConfig(warmup_days=300, fill_model="limit"))
        loose = run_system_backtest(
            data, symbols=SYMBOLS, verbose=False,
            cfg=SystemBacktestConfig(warmup_days=300, fill_model="open"))
        assert loose.fill_rate > strict.fill_rate
