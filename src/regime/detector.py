"""
GMM-based market regime detector.

4 regimes:
    bull_quiet   (0) — uptrend, low volatility
    bull_volatile (1) — uptrend, elevated volatility
    bear          (2) — downtrend
    sideways      (3) — no clear trend (maps to "choppy" for backwards compat)

Usage
-----
    # First-time fit (offline, run once and commit models/gmm_regime.pkl):
    python -m src.regime.detector

    # Inference:
    from src.regime.detector import GMMRegimeDetector
    det = GMMRegimeDetector()
    result = det.get_regime(spy_df)   # drop-in replacement for detect_regime()
    scale  = GMMRegimeDetector.regime_kelly_scale(result["gmm_regime"], result["proba"])
"""
from __future__ import annotations

import math
import pickle
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

# ── Retrocompat mapping: 4-way GMM label → 3-way legacy label ─────────────────
_COMPAT = {
    "bull_quiet":    "bull",
    "bull_volatile": "bull",
    "bear":          "bear",
    "sideways":      "choppy",
}

# Probability-weighted Kelly scale per regime
_KELLY_SCALE = {
    "bull_quiet":    1.00,
    "bull_volatile": 0.70,
    "sideways":      0.50,
    "bear":          0.20,
}


# ── Feature engineering ────────────────────────────────────────────────────────

_FEATURE_COLS = ["vol_5d", "vol_21d", "mom_10d", "mom_30d", "ret_21d", "atr_pct"]


def _atr_series(df: pd.DataFrame, period: int = 14) -> pd.Series:
    high  = df["High"].astype(float)
    low   = df["Low"].astype(float)
    prev  = df["Close"].astype(float).shift(1)
    tr    = pd.concat([high - low, (high - prev).abs(), (low - prev).abs()], axis=1).max(axis=1)
    return tr.rolling(period).mean() / df["Close"].astype(float)


def _build_features(df: pd.DataFrame) -> pd.DataFrame:
    """Compute the 6-feature matrix from an OHLCV DataFrame. Drops NaN rows."""
    close = df["Close"].astype(float)
    rets  = close.pct_change()
    ann   = math.sqrt(252)

    feat = pd.DataFrame(index=df.index)
    feat["vol_5d"]  = rets.rolling(5).std()  * ann
    feat["vol_21d"] = rets.rolling(21).std() * ann
    feat["mom_10d"] = close / close.shift(10) - 1
    feat["mom_30d"] = close / close.shift(30) - 1
    feat["ret_21d"] = rets.rolling(21).sum()
    feat["atr_pct"] = _atr_series(df)
    return feat.dropna()


# ── Label assignment ───────────────────────────────────────────────────────────

def _assign_labels(gmm, scaler) -> dict[int, str]:
    """
    Map each GMM component index to a semantic regime label.

    Strategy:
        - Sort components by mom_30d (col 3).
        - Bottom 2 (negative/low momentum): distinguish bear vs sideways by vol_21d.
          Higher vol → bear, lower vol → sideways.
        - Top 2 (positive momentum): distinguish bull_volatile vs bull_quiet by vol_21d.
          Higher vol → bull_volatile, lower vol → bull_quiet.
    """
    means   = scaler.inverse_transform(gmm.means_)   # (4, n_features)
    col_mom = _FEATURE_COLS.index("mom_30d")
    col_vol = _FEATURE_COLS.index("vol_21d")

    order = np.argsort(means[:, col_mom])   # ascending momentum

    bear_grp = order[:2]
    bull_grp = order[2:]

    vols_bear = means[bear_grp, col_vol]
    bear_idx     = int(bear_grp[np.argmax(vols_bear)])
    sideways_idx = int(bear_grp[np.argmin(vols_bear)])

    vols_bull = means[bull_grp, col_vol]
    bull_volatile_idx = int(bull_grp[np.argmax(vols_bull)])
    bull_quiet_idx    = int(bull_grp[np.argmin(vols_bull)])

    return {
        bear_idx:          "bear",
        sideways_idx:      "sideways",
        bull_volatile_idx: "bull_volatile",
        bull_quiet_idx:    "bull_quiet",
    }


# ── Main class ─────────────────────────────────────────────────────────────────

class GMMRegimeDetector:
    MODEL_PATH = Path("models/gmm_regime.pkl")

    def __init__(self) -> None:
        self._state: Optional[dict] = self._try_load()

    # ── Persistence ───────────────────────────────────────────────────────────

    def _try_load(self) -> Optional[dict]:
        try:
            if self.MODEL_PATH.exists():
                with self.MODEL_PATH.open("rb") as f:
                    return pickle.load(f)
        except Exception:
            pass
        return None

    def _save(self, state: dict) -> None:
        self.MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
        with self.MODEL_PATH.open("wb") as f:
            pickle.dump(state, f)

    @property
    def is_fitted(self) -> bool:
        return self._state is not None

    # ── Training ──────────────────────────────────────────────────────────────

    def fit(self, df: Optional[pd.DataFrame] = None) -> None:
        """
        Train the GMM on historical SPY data and persist the model.
        Recommended: run offline with ≥2 years of daily data.
        """
        from sklearn.mixture import GaussianMixture
        from sklearn.preprocessing import StandardScaler

        if df is None:
            from src.data.market_data import download_ohlcv
            df = download_ohlcv("SPY", period="5y")

        feat = _build_features(df)
        X    = feat[_FEATURE_COLS].values

        scaler = StandardScaler()
        X_sc   = scaler.fit_transform(X)

        gmm = GaussianMixture(
            n_components=4,
            covariance_type="full",
            n_init=20,
            random_state=42,
        )
        gmm.fit(X_sc)

        label_map = _assign_labels(gmm, scaler)

        state = {
            "gmm":        gmm,
            "scaler":     scaler,
            "label_map":  label_map,
            "fitted_at":  pd.Timestamp.now().isoformat(),
        }
        self._save(state)
        self._state = state
        print(f"✅ GMM fitted — label map: {label_map}")

    # ── Inference ─────────────────────────────────────────────────────────────

    def get_regime(self, df: pd.DataFrame) -> dict:
        """
        Return regime dict compatible with the legacy detect_regime() output,
        extended with GMM fields.

        Retrocompat keys (always present):
            regime     — "bull" | "bear" | "choppy"
            price      — latest close
            sma50, sma200, atr14, vol_regime — same semantics as before

        GMM-only keys (present only when model is fitted):
            gmm_regime  — "bull_quiet" | "bull_volatile" | "bear" | "sideways"
            proba       — {label: probability, ...}
            gmm_available — True

        Fallback when model not fitted:
            gmm_regime    → None
            proba         → {}
            gmm_available → False
        """
        from src.data.regime import detect_regime
        base = detect_regime(df=df)

        if not self.is_fitted:
            return {**base, "gmm_regime": None, "proba": {}, "gmm_available": False}

        feat = _build_features(df)
        if feat.empty:
            return {**base, "gmm_regime": None, "proba": {}, "gmm_available": False}

        gmm       = self._state["gmm"]
        scaler    = self._state["scaler"]
        label_map = self._state["label_map"]

        X_last = feat[_FEATURE_COLS].values[-1:, :]
        X_sc   = scaler.transform(X_last)

        comp_proba = gmm.predict_proba(X_sc)[0]   # (4,) — one value per component
        label_proba = {
            label_map[i]: float(comp_proba[i])
            for i in range(len(comp_proba))
        }
        gmm_regime = max(label_proba, key=label_proba.__getitem__)

        return {
            **base,
            "regime":        _COMPAT[gmm_regime],
            "gmm_regime":    gmm_regime,
            "proba":         label_proba,
            "gmm_available": True,
        }

    def label_series(self, df: pd.DataFrame) -> pd.Series:
        """
        Return a GMM regime label for every row in df (used by backtest engine).
        Falls back to the legacy 3-way labels when model is not fitted.
        """
        if not self.is_fitted:
            from src.data.regime import detect_regime
            from src.backtest.engine import _compute_regime_series
            close = df["Close"].astype(float)
            return _compute_regime_series(close)

        feat      = _build_features(df)
        gmm       = self._state["gmm"]
        scaler    = self._state["scaler"]
        label_map = self._state["label_map"]

        X_sc       = scaler.transform(feat[_FEATURE_COLS].values)
        components = gmm.predict(X_sc)
        labels_4   = pd.Series(
            [label_map[c] for c in components],
            index=feat.index,
        )
        # Map to compat 3-way labels for use by existing agents
        return labels_4.map(_COMPAT).reindex(df.index).ffill()

    # ── Kelly scale ───────────────────────────────────────────────────────────

    @staticmethod
    def regime_kelly_scale(
        gmm_regime: Optional[str],
        probas: dict[str, float],
    ) -> float:
        """
        Probability-weighted Kelly scale in [0.2, 1.0].

        When probas is available, uses the full distribution for a smooth
        continuous scale rather than a hard switch between regimes.
        If probas is empty (model not fitted), defaults to 1.0.
        """
        if not probas:
            return 1.0
        return float(
            sum(probas.get(r, 0.0) * s for r, s in _KELLY_SCALE.items())
        )


# ── CLI entry point — train and report current regime ─────────────────────────

if __name__ == "__main__":
    from src.data.market_data import download_ohlcv

    print("Downloading SPY (5y)…")
    spy = download_ohlcv("SPY", period="5y")

    det = GMMRegimeDetector()
    det.fit(spy)

    result = det.get_regime(spy)
    print(f"\nCurrent regime : {result['gmm_regime']} → compat: {result['regime']}")
    print("Probabilities  :")
    for label, p in sorted(result["proba"].items(), key=lambda x: -x[1]):
        bar = "█" * int(p * 20)
        print(f"  {label:<15} {p:5.1%}  {bar}")
    scale = GMMRegimeDetector.regime_kelly_scale(result["gmm_regime"], result["proba"])
    print(f"\nKelly scale    : {scale:.3f}")
