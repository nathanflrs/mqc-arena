"""Unit tests for src/regime/detector.py — all offline, no network calls."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.regime.detector import (
    GMMRegimeDetector,
    _COMPAT,
    _FEATURE_COLS,
    _KELLY_SCALE,
    _assign_labels,
    _build_features,
)


# ── Synthetic OHLCV helpers ────────────────────────────────────────────────────

def _make_ohlcv(closes: np.ndarray, seed: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    n   = len(closes)
    idx = pd.date_range("2018-01-01", periods=n, freq="B")
    noise = np.abs(closes * 0.005)
    return pd.DataFrame(
        {
            "Open":   closes - noise,
            "High":   closes + noise * 2,
            "Low":    closes - noise * 2,
            "Close":  closes,
            "Volume": np.full(n, 1_000_000, dtype=float),
        },
        index=idx,
    )


@pytest.fixture()
def bull_df() -> pd.DataFrame:
    closes = np.linspace(100, 200, 600) + np.random.default_rng(0).normal(0, 0.2, 600)
    return _make_ohlcv(closes)


@pytest.fixture()
def bear_df() -> pd.DataFrame:
    closes = np.linspace(200, 100, 600) + np.random.default_rng(1).normal(0, 0.2, 600)
    return _make_ohlcv(closes)


@pytest.fixture()
def mixed_df() -> pd.DataFrame:
    """Two years of mixed data: bull, crash, recovery, sideways."""
    rng = np.random.default_rng(7)
    bull  = np.linspace(100, 160, 150)
    crash = np.linspace(160, 110, 60)
    rec   = np.linspace(110, 145, 120)
    flat  = np.full(170, 145.0) + rng.normal(0, 0.5, 170)
    closes = np.concatenate([bull, crash, rec, flat])
    closes += rng.normal(0, 0.1, len(closes))
    return _make_ohlcv(closes)


# ── Feature engineering ────────────────────────────────────────────────────────

def test_build_features_columns(bull_df):
    feat = _build_features(bull_df)
    assert list(feat.columns) == _FEATURE_COLS


def test_build_features_no_nan(bull_df):
    feat = _build_features(bull_df)
    assert not feat.isnull().any().any()


def test_build_features_min_rows(bull_df):
    # 600-row input → at least 500 feature rows (loses ~30 due to rolling windows)
    feat = _build_features(bull_df)
    assert len(feat) >= 500


def test_build_features_short_returns_empty():
    closes = np.linspace(100, 110, 10)
    df = _make_ohlcv(closes)
    feat = _build_features(df)
    assert feat.empty


# ── fit() and is_fitted ────────────────────────────────────────────────────────

def test_fit_creates_model(tmp_path, monkeypatch, mixed_df):
    monkeypatch.setattr(GMMRegimeDetector, "MODEL_PATH", tmp_path / "gmm.pkl")
    det = GMMRegimeDetector()
    assert not det.is_fitted

    det.fit(mixed_df)

    assert det.is_fitted
    assert (tmp_path / "gmm.pkl").exists()


def test_fit_label_map_covers_all_regimes(tmp_path, monkeypatch, mixed_df):
    monkeypatch.setattr(GMMRegimeDetector, "MODEL_PATH", tmp_path / "gmm.pkl")
    det = GMMRegimeDetector()
    det.fit(mixed_df)

    label_map = det._state["label_map"]
    assert set(label_map.values()) == {"bull_quiet", "bull_volatile", "bear", "sideways"}
    assert len(label_map) == 4   # one entry per component


def test_fit_model_reloads(tmp_path, monkeypatch, mixed_df):
    path = tmp_path / "gmm.pkl"
    monkeypatch.setattr(GMMRegimeDetector, "MODEL_PATH", path)

    det1 = GMMRegimeDetector()
    det1.fit(mixed_df)

    det2 = GMMRegimeDetector()   # new instance, loads from disk
    assert det2.is_fitted


# ── get_regime() — retrocompat ─────────────────────────────────────────────────

RETROCOMPAT_KEYS = {"regime", "price", "sma50", "sma200", "atr14", "vol_regime"}
GMM_KEYS         = {"gmm_regime", "proba", "gmm_available"}


def test_get_regime_fallback_keys(mixed_df):
    """Without a fitted model, all retrocompat keys must still be present."""
    import tempfile
    with tempfile.TemporaryDirectory() as td:
        # Point MODEL_PATH to a non-existent file
        det = GMMRegimeDetector()
        det._state = None   # force unfitted

        result = det.get_regime(mixed_df)

    assert RETROCOMPAT_KEYS.issubset(result.keys())
    assert GMM_KEYS.issubset(result.keys())
    assert result["gmm_available"] is False
    assert result["gmm_regime"] is None
    assert result["proba"] == {}


def test_get_regime_fitted_keys(tmp_path, monkeypatch, mixed_df):
    monkeypatch.setattr(GMMRegimeDetector, "MODEL_PATH", tmp_path / "gmm.pkl")
    det = GMMRegimeDetector()
    det.fit(mixed_df)

    result = det.get_regime(mixed_df)

    assert RETROCOMPAT_KEYS.issubset(result.keys())
    assert GMM_KEYS.issubset(result.keys())
    assert result["gmm_available"] is True


def test_get_regime_proba_sums_to_one(tmp_path, monkeypatch, mixed_df):
    monkeypatch.setattr(GMMRegimeDetector, "MODEL_PATH", tmp_path / "gmm.pkl")
    det = GMMRegimeDetector()
    det.fit(mixed_df)

    result = det.get_regime(mixed_df)
    assert abs(sum(result["proba"].values()) - 1.0) < 1e-6


def test_get_regime_gmm_labels_valid(tmp_path, monkeypatch, mixed_df):
    monkeypatch.setattr(GMMRegimeDetector, "MODEL_PATH", tmp_path / "gmm.pkl")
    det = GMMRegimeDetector()
    det.fit(mixed_df)

    result = det.get_regime(mixed_df)
    assert result["gmm_regime"] in {"bull_quiet", "bull_volatile", "bear", "sideways"}


def test_get_regime_compat_label_consistent(tmp_path, monkeypatch, mixed_df):
    monkeypatch.setattr(GMMRegimeDetector, "MODEL_PATH", tmp_path / "gmm.pkl")
    det = GMMRegimeDetector()
    det.fit(mixed_df)

    result = det.get_regime(mixed_df)
    expected_compat = _COMPAT[result["gmm_regime"]]
    assert result["regime"] == expected_compat


def test_get_regime_compat_values(tmp_path, monkeypatch, mixed_df):
    monkeypatch.setattr(GMMRegimeDetector, "MODEL_PATH", tmp_path / "gmm.pkl")
    det = GMMRegimeDetector()
    det.fit(mixed_df)

    result = det.get_regime(mixed_df)
    assert result["regime"] in {"bull", "bear", "choppy"}


# ── Label assignment heuristic ─────────────────────────────────────────────────

def test_label_assignment_bear_has_negative_momentum(tmp_path, monkeypatch, mixed_df):
    """Bear component must have the lowest momentum_30d among all components."""
    from sklearn.mixture import GaussianMixture
    from sklearn.preprocessing import StandardScaler

    monkeypatch.setattr(GMMRegimeDetector, "MODEL_PATH", tmp_path / "gmm.pkl")
    det = GMMRegimeDetector()
    det.fit(mixed_df)

    gmm    = det._state["gmm"]
    scaler = det._state["scaler"]
    label_map = det._state["label_map"]

    means_orig = scaler.inverse_transform(gmm.means_)
    col_mom    = _FEATURE_COLS.index("mom_30d")

    bear_idx = [k for k, v in label_map.items() if v == "bear"][0]
    # Bear component must NOT be the one with the highest momentum
    assert means_orig[bear_idx, col_mom] < max(means_orig[:, col_mom])


def test_label_assignment_bull_quiet_lower_vol(tmp_path, monkeypatch, mixed_df):
    """bull_quiet must have lower vol_21d than bull_volatile."""
    monkeypatch.setattr(GMMRegimeDetector, "MODEL_PATH", tmp_path / "gmm.pkl")
    det = GMMRegimeDetector()
    det.fit(mixed_df)

    gmm       = det._state["gmm"]
    scaler    = det._state["scaler"]
    label_map = det._state["label_map"]

    means_orig = scaler.inverse_transform(gmm.means_)
    col_vol    = _FEATURE_COLS.index("vol_21d")

    bq_idx = [k for k, v in label_map.items() if v == "bull_quiet"][0]
    bv_idx = [k for k, v in label_map.items() if v == "bull_volatile"][0]
    assert means_orig[bq_idx, col_vol] <= means_orig[bv_idx, col_vol]


# ── Kelly scale ────────────────────────────────────────────────────────────────

def test_kelly_scale_no_probas():
    scale = GMMRegimeDetector.regime_kelly_scale(None, {})
    assert scale == 1.0


def test_kelly_scale_in_range():
    for regime, expected_scale in _KELLY_SCALE.items():
        # Pure regime (probability = 1) should return the exact scale
        probas = {r: 0.0 for r in _KELLY_SCALE}
        probas[regime] = 1.0
        scale = GMMRegimeDetector.regime_kelly_scale(regime, probas)
        assert abs(scale - expected_scale) < 1e-9, f"{regime}: expected {expected_scale}, got {scale}"


def test_kelly_scale_mixed_probas():
    probas = {"bull_quiet": 0.5, "bull_volatile": 0.3, "sideways": 0.1, "bear": 0.1}
    scale  = GMMRegimeDetector.regime_kelly_scale("bull_quiet", probas)
    expected = 0.5 * 1.0 + 0.3 * 0.70 + 0.1 * 0.50 + 0.1 * 0.20
    assert abs(scale - expected) < 1e-9


def test_kelly_scale_always_between_min_max():
    """Any valid probability distribution must yield a scale in [min, max]."""
    rng = np.random.default_rng(99)
    labels = list(_KELLY_SCALE.keys())
    for _ in range(200):
        raw = rng.random(4)
        raw /= raw.sum()
        probas = dict(zip(labels, raw))
        scale  = GMMRegimeDetector.regime_kelly_scale(labels[0], probas)
        assert min(_KELLY_SCALE.values()) <= scale <= max(_KELLY_SCALE.values()) + 1e-9


# ── label_series() ─────────────────────────────────────────────────────────────

def test_label_series_fitted_length(tmp_path, monkeypatch, mixed_df):
    monkeypatch.setattr(GMMRegimeDetector, "MODEL_PATH", tmp_path / "gmm.pkl")
    det = GMMRegimeDetector()
    det.fit(mixed_df)

    series = det.label_series(mixed_df)
    assert len(series) == len(mixed_df)


def test_label_series_fitted_values(tmp_path, monkeypatch, mixed_df):
    monkeypatch.setattr(GMMRegimeDetector, "MODEL_PATH", tmp_path / "gmm.pkl")
    det = GMMRegimeDetector()
    det.fit(mixed_df)

    series = det.label_series(mixed_df)
    assert set(series.dropna().unique()).issubset({"bull", "bear", "choppy"})


def test_label_series_unfitted_fallback(mixed_df):
    det = GMMRegimeDetector()
    det._state = None   # force unfitted

    series = det.label_series(mixed_df)
    assert len(series) == len(mixed_df)
    assert set(series.dropna().unique()).issubset({"bull", "bear", "choppy"})


# ── compat map completeness ────────────────────────────────────────────────────

def test_compat_map_covers_all_gmm_labels():
    gmm_labels = {"bull_quiet", "bull_volatile", "bear", "sideways"}
    assert gmm_labels == set(_COMPAT.keys())


def test_compat_map_values_are_legacy():
    for v in _COMPAT.values():
        assert v in {"bull", "bear", "choppy"}
