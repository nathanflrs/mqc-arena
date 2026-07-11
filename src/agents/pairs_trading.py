# src/agents/pairs_trading.py
#
# PairsTradingAgent — Statistical arbitrage, market-neutral.
#
# STATE: "ARMED BUT AT REST"
# As of 2026-06-29, no pair in the candidate universe passes the
# multi-criteria filter (Engle-Granger multi-window + half-life + Hurst +
# zero-crossings + rolling p-value stability).
# NVDA/AMD passes only on 252j (15% of rolling windows), window artifact.
#
# The agent will generate BUY/SELL signals the moment a robust pair appears.
# Until then it emits HOLD legitimately. Trading a single fragile pair would
# not constitute genuine market-neutral diversification.
#
from __future__ import annotations

import logging
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import statsmodels.api as sm
import yfinance as yf
from statsmodels.tsa.stattools import coint

from src.agents.base import BaseAgent, MarketState, AgentSignal

logger = logging.getLogger(__name__)

# ── Configuration ──────────────────────────────────────────────────────────────

@dataclass
class PairsTradingConfig:
    # Signal thresholds
    zscore_entry: float      = 2.0    # open position when |z| exceeds this
    zscore_exit: float       = 0.5    # close when |z| falls below this
    zscore_stoploss: float   = 3.5    # close when |z| exceeds this (diverging)
    # Rolling window for spread z-score (trading days)
    lookback: int            = 63
    # Primary cointegration test window (trading days)
    coint_lookback: int      = 252
    # Days between cointegration re-tests
    recheck_days: int        = 21
    # Engle-Granger p-value acceptance threshold
    pvalue_threshold: float  = 0.05
    # Gross notional weight per leg as fraction of NAV
    target_weight: float     = 0.08
    # Guard: near-zero spread variance
    min_std: float           = 1e-6
    # ── Multi-window robustness filter (default ON) ────────────────────────
    multi_window_check: bool   = True   # require ≥ min_robustness_score windows
    min_robustness_score: int  = 2      # of 3 windows [126, 252, 504j]
    # Gross pairs exposure cap enforced by RiskManager
    max_gross_pairs_pct: float = 0.30
    # ── Rolling p-value stability filter ─────────────────────────────────
    rolling_pvalue_threshold: float = 0.10   # "fail" if p > this per window
    max_rolling_fail_rate: float    = 0.20   # reject if fail_rate > 20%
    rolling_lookback: int           = 504    # 2 years of history
    rolling_window: int             = 252    # each rolling window size
    rolling_step: int               = 21     # step between windows
    # ── Ornstein-Uhlenbeck half-life filter ───────────────────────────────
    min_halflife_days: float = 5.0    # reject if reversion is faster (too costly)
    max_halflife_days: float = 60.0   # reject if reversion is slower (capital at risk)
    # ── Hurst exponent filter ─────────────────────────────────────────────
    max_hurst: float = 0.50   # reject if H ≥ 0.5 (not mean-reverting)
    # ── Zero-crossings filter ─────────────────────────────────────────────
    min_zero_crossings: int = 12   # minimum crossings of spread mean in 252j


# ── Candidate pairs — genuine economic substitutes by sector ───────────────────
# These pairs exist ONLY for PairsTradingAgent validation.
# They are NOT added to the directional-agent WATCHLIST.

CANDIDATE_PAIRS: List[Tuple[str, str]] = [
    # Energy majors
    ("XOM",  "CVX"),
    # E&P vs integrated
    ("COP",  "XOM"),
    # Downstream refiners
    ("PSX",  "VLO"),
    # Regulated utilities
    ("SO",   "DUK"),
    ("AEP",  "EXC"),
    # Apartment REITs
    ("EQR",  "AVB"),
    # Industrial REITs
    ("PLD",  "REXR"),
    # Network airlines (M&A risk high)
    ("DAL",  "UAL"),
    # Low-cost carriers (M&A risk high)
    ("LUV",  "JBLU"),
    # Life insurance
    ("MET",  "PRU"),
    # P&C insurance
    ("CB",   "TRV"),
    # Regional banks — M&A risk HIGH
    ("RF",   "FITB"),
    ("CFG",  "HBAN"),
]

# Three static cointegration test windows for robustness scoring
_COINT_WINDOWS = [126, 252, 504]

# M&A event risk by pair (affects sizing, not acceptance)
_MA_RISK: Dict[Tuple[str, str], str] = {
    # High: frequent consolidation in sector
    ("RF",   "FITB"):  "high",
    ("CFG",  "HBAN"):  "high",
    ("DAL",  "UAL"):   "high",
    ("LUV",  "JBLU"):  "high",
    # Medium: occasional M&A activity
    ("PSX",  "VLO"):   "medium",
    ("MET",  "PRU"):   "medium",
    ("CB",   "TRV"):   "medium",
    # Low: stable regulated or capital-light structures
    ("XOM",  "CVX"):   "low",
    ("COP",  "XOM"):   "low",
    ("SO",   "DUK"):   "low",
    ("AEP",  "EXC"):   "low",
    ("EQR",  "AVB"):   "low",
    ("PLD",  "REXR"):  "low",
}


# ── Validated pair state ───────────────────────────────────────────────────────

@dataclass
class ValidatedPair:
    """
    A pair that has passed all multi-criteria validation filters.

        spread = log(A) − hedge_ratio · log(B)

    hedge_ratio (β) from OLS: log(A) = α + β · log(B) + ε
    → short exactly β notional-units of B per unit of A.

    Additional OU / mean-reversion characteristics
    -----------------------------------------------
    half_life       : days for spread to revert halfway to mean (OU regression).
                      Valid range: [min_halflife_days, max_halflife_days].
    hurst           : Hurst exponent from R/S analysis. H < 0.5 = mean-reverting.
    zero_crossings  : number of mean crossings in the 252j formation window.
    ma_risk         : 'low' | 'medium' | 'high' — M&A event risk in sector.
    rolling_fail_rate: fraction of rolling EG windows with p > 0.10 (last 2y).
                       NVDA/AMD would have had 85% fail rate → auto-rejected.
    """
    ticker_a: str
    ticker_b: str
    hedge_ratio: float
    pvalue: float
    last_tested: datetime
    spread_mean: float
    spread_std: float
    is_active: bool               = True
    robustness_score: int         = 0
    window_pvalues: Dict          = field(default_factory=dict)
    # New OU / mean-reversion fields (all optional for backward compat)
    half_life: Optional[float]    = None
    hurst: Optional[float]        = None
    zero_crossings: int           = 0
    ma_risk: str                  = "low"
    rolling_fail_rate: Optional[float] = None


# ── Full validation report ─────────────────────────────────────────────────────

@dataclass
class PairValidationReport:
    """
    Complete multi-criteria diagnostic for a candidate pair.
    Returned by PairSelector.validate_pair_full().
    """
    ticker_a: str
    ticker_b: str
    ma_risk: str
    # Engle-Granger (3 windows)
    window_pvalues: Dict[int, Optional[float]]
    robustness_score: int
    passes_robustness: bool
    # Rolling EG stability (2-year history)
    rolling_fail_rate: Optional[float]   # fraction of windows with p > threshold
    passes_rolling: bool
    # Hedge ratio and spread stats
    hedge_ratio: Optional[float]
    spread_mean: Optional[float]
    spread_std: Optional[float]
    # Ornstein-Uhlenbeck half-life
    half_life: Optional[float]
    passes_halflife: bool
    # Hurst exponent
    hurst: Optional[float]
    passes_hurst: bool
    # Zero-crossings
    zero_crossings: Optional[int]
    passes_zero_crossings: bool
    # Final verdict
    is_tradable: bool
    rejection_reasons: List[str]


# ── Price download helper ──────────────────────────────────────────────────────

def _fetch_close(ticker: str, period: str = "4y") -> pd.Series:
    try:
        df = yf.download(
            ticker, period=period, auto_adjust=True,
            progress=False, threads=False,
        )
        close = df["Close"]
        if isinstance(close, pd.DataFrame):
            close = close.iloc[:, 0]
        return pd.to_numeric(close, errors="coerce").dropna()
    except Exception as exc:
        logger.warning("PairSelector: download failed for %s — %s", ticker, exc)
        return pd.Series(dtype=float)


# ── Spread analysis helpers ────────────────────────────────────────────────────

def _compute_halflife(spread: np.ndarray) -> Optional[float]:
    """
    Half-life of mean-reversion via Ornstein-Uhlenbeck OLS regression.

        Δε(t) = α + b·ε(t-1) + noise   →   b = −η
        half_life = ln(2) / η           (in trading days)

    Returns None if the spread is not mean-reverting (b ≥ 0) or if data
    is insufficient for regression.
    """
    if len(spread) < 20:
        return None
    delta = np.diff(spread)
    lag   = spread[:-1]
    X = sm.add_constant(lag)
    try:
        res = sm.OLS(delta, X).fit()
        b   = float(res.params[1])
    except Exception:
        return None
    if b >= 0:
        return None   # non-mean-reverting: spread is diverging
    return float(np.log(2.0) / (-b))


def _compute_hurst(spread: np.ndarray, min_points: int = 4) -> Optional[float]:
    """
    Hurst exponent via the variance-time (VT) method.

        V(lag) = Var(spread[t + lag] - spread[t])

        H = slope / 2   where slope = OLS slope of log(V) vs log(lag)

    Interpretation
    ---------------
    - H < 0.5  →  mean-reverting: V(lag) saturates as spread stays bounded
    - H = 0.5  →  random walk: V(lag) grows linearly with lag
    - H > 0.5  →  persistent: V(lag) grows faster than linear

    Why VT and not classical R/S
    ----------------------------
    The classical R/S (Hurst, 1951) gives H → 0.5 asymptotically for any
    short-memory stationary process (OU, AR(1)) because Var(partial-sum) ∝ n.
    The VT method correctly identifies OU as mean-reverting (H < 0.5) because
    the increment variance V(lag) saturates once lag >> half-life.
    """
    n = len(spread)
    if n < 50:
        return None

    max_lag  = n // 4
    raw_lags = np.unique(
        np.logspace(1.0, np.log10(max(max_lag, 11)), num=12).astype(int)
    )
    lags = [int(l) for l in raw_lags if 2 <= int(l) <= max_lag]

    vt_pairs: List[Tuple[float, float]] = []
    for lag in lags:
        increments = spread[lag:] - spread[:-lag]
        if len(increments) < 10:
            continue
        v = float(np.var(increments, ddof=1))
        if v < 1e-20:
            continue
        vt_pairs.append((np.log(lag), np.log(v)))

    if len(vt_pairs) < min_points:
        return None

    lags_arr = np.array([x[0] for x in vt_pairs])
    v_arr    = np.array([x[1] for x in vt_pairs])
    X = sm.add_constant(lags_arr)
    try:
        slope = float(sm.OLS(v_arr, X).fit().params[1])
    except Exception:
        return None
    # V ∝ lag^{2H} → slope = 2H → H = slope/2
    return slope / 2.0


def _count_zero_crossings(spread: np.ndarray) -> int:
    """
    Number of times the spread crosses its mean in the given window.
    Pairs with few crossings show weak mean-reversion in practice.
    """
    centered = spread - np.mean(spread)
    signs    = np.sign(centered)
    signs[signs == 0] = 1   # tie-break: treat exact-mean as positive
    return int(np.sum(np.diff(signs) != 0))


# ── Pair selector ──────────────────────────────────────────────────────────────

class PairSelector:
    """
    Multi-criteria cointegration validator for candidate pairs.

    Validation pipeline (all criteria must pass)
    -------------------------------------------
    1. Engle-Granger on 3 windows (126/252/504j)  → robustness_score
    2. Multi-window robustness (≥ min_score)       → rejects single-window flukes
    3. Rolling EG stability (last 2y, step 21j)    → rejects intermittent coint
    4. Half-life [5, 60j] via OU regression        → rejects untradeably fast/slow
    5. Hurst H < 0.50 via R/S analysis             → confirms anti-persistence
    6. Zero-crossings ≥ 12 in 252j window          → confirms sufficient activity

    State: "ARMED BUT AT REST"
    --------------------------
    As of 2026-06-29, zero pairs in the sector universe pass all criteria.
    The agent correctly emits HOLD until a robust pair emerges.
    """

    def __init__(self, cfg: PairsTradingConfig) -> None:
        self._cfg = cfg
        self._validated: Dict[Tuple[str, str], ValidatedPair] = {}
        self._price_cache: Dict[str, pd.Series] = {}
        self._last_full_test: Optional[datetime] = None

    # ── Price access ───────────────────────────────────────────────────────────

    def _get_price(self, ticker: str) -> pd.Series:
        if ticker not in self._price_cache:
            self._price_cache[ticker] = _fetch_close(ticker)
        return self._price_cache[ticker]

    def _prefetch_prices(self, tickers: List[str], period: str = "4y") -> None:
        """Bulk-download all tickers not yet in cache (one yf call)."""
        missing = [t for t in tickers if t not in self._price_cache]
        if not missing:
            return
        try:
            raw = yf.download(
                missing, period=period, auto_adjust=True,
                progress=False, threads=True,
            )
            if isinstance(raw.columns, pd.MultiIndex):
                closes = raw["Close"]
            elif "Close" in raw.columns:
                closes = raw[["Close"]].rename(columns={"Close": missing[0]})
            else:
                closes = pd.DataFrame()

            if isinstance(closes, pd.Series):
                closes = closes.to_frame(name=missing[0] if len(missing) == 1 else "UNKNOWN")

            for t in missing:
                if t in closes.columns:
                    s = pd.to_numeric(closes[t], errors="coerce").dropna()
                    if not s.empty:
                        self._price_cache[t] = s
                        continue
                # Fallback to individual download for missed tickers
                s = _fetch_close(t, period)
                if not s.empty:
                    self._price_cache[t] = s
        except Exception as exc:
            logger.warning(
                "PairSelector: bulk prefetch failed (%s) — falling back to individual", exc
            )
            for t in missing:
                if t not in self._price_cache:
                    self._price_cache[t] = _fetch_close(t, period)

    # ── Internals ──────────────────────────────────────────────────────────────

    @staticmethod
    def _estimate_hedge_ratio(log_a: np.ndarray, log_b: np.ndarray) -> float:
        """OLS: log(A) = α + β·log(B) + ε  →  β."""
        X = sm.add_constant(log_b)
        return float(sm.OLS(log_a, X).fit().params[1])

    def _rolling_pvalue_filter(
        self, log_a: np.ndarray, log_b: np.ndarray,
    ) -> Tuple[float, int]:
        """
        Rolling Engle-Granger test on the last `rolling_lookback` days.
        Returns (fail_rate, n_windows) where fail_rate is the fraction of
        windows where p > rolling_pvalue_threshold.

        NVDA/AMD would score fail_rate = 0.85 (85% of windows p > 0.10)
        → automatically rejected by the max_rolling_fail_rate = 0.20 guard.
        """
        cfg    = self._cfg
        n      = len(log_a)
        lookb  = min(n, cfg.rolling_lookback)
        la     = log_a[-lookb:]
        lb     = log_b[-lookb:]
        n      = len(la)

        fail_count = 0
        total      = 0
        start      = 0
        while start + cfg.rolling_window <= n:
            chunk_a = la[start : start + cfg.rolling_window]
            chunk_b = lb[start : start + cfg.rolling_window]
            try:
                _, p, _ = coint(chunk_a, chunk_b)
                if p > cfg.rolling_pvalue_threshold:
                    fail_count += 1
                total += 1
            except Exception:
                pass
            start += cfg.rolling_step

        fail_rate = fail_count / total if total > 0 else 1.0
        return fail_rate, total

    # ── Full multi-criteria validation ─────────────────────────────────────────

    def validate_pair_full(self, a: str, b: str) -> PairValidationReport:
        """
        Run the complete validation pipeline on a candidate pair and return
        a PairValidationReport with all criteria results.

        Criteria order
        --------------
        1. Data availability
        2. Engle-Granger on 3 windows  (→ robustness_score)
        3. Multi-window robustness gate
        4. Rolling EG p-value stability (2-year look-back)
        5. Half-life [min_halflife_days, max_halflife_days]
        6. Hurst H < max_hurst
        7. Zero-crossings ≥ min_zero_crossings

        A pair is tradable only if ALL criteria pass (rejection_reasons empty).
        """
        cfg = self._cfg
        ma_risk = _MA_RISK.get((a, b), _MA_RISK.get((b, a), "low"))
        rejection_reasons: List[str] = []

        # Default values (all criteria unmet until proven otherwise)
        window_pvalues: Dict[int, Optional[float]] = {}
        robustness_score    = 0
        passes_robustness   = True
        rolling_fail_rate: Optional[float] = None
        passes_rolling      = True
        hedge_ratio: Optional[float]       = None
        spread_mean: Optional[float]       = None
        spread_std: Optional[float]        = None
        half_life: Optional[float]         = None
        passes_halflife     = False
        hurst: Optional[float]             = None
        passes_hurst        = False
        zero_crossings: Optional[int]      = None
        passes_zero_crossings = False

        # ── Data check ────────────────────────────────────────────────────────
        sa = self._get_price(a)
        sb = self._get_price(b)
        df = pd.DataFrame({"a": sa, "b": sb}).dropna()
        min_rows = cfg.coint_lookback + cfg.lookback

        if len(df) < min_rows:
            rejection_reasons.append(
                f"données insuffisantes ({len(df)} lignes, besoin {min_rows})"
            )
            return PairValidationReport(
                ticker_a=a, ticker_b=b, ma_risk=ma_risk,
                window_pvalues={}, robustness_score=0, passes_robustness=False,
                rolling_fail_rate=None, passes_rolling=False,
                hedge_ratio=None, spread_mean=None, spread_std=None,
                half_life=None, passes_halflife=False,
                hurst=None, passes_hurst=False,
                zero_crossings=None, passes_zero_crossings=False,
                is_tradable=False, rejection_reasons=rejection_reasons,
            )

        # ── 1. Engle-Granger on 3 windows ─────────────────────────────────────
        # window_results: {w → (pvalue, beta)} for windows that PASS
        window_results: Dict[int, Tuple[float, float]] = {}

        for w in _COINT_WINDOWS:
            if len(df) < w:
                window_pvalues[w] = None
                continue
            chunk = df.iloc[-w:]
            la    = np.log(chunk["a"].values)
            lb    = np.log(chunk["b"].values)
            try:
                _, pval, _ = coint(la, lb)
                beta       = self._estimate_hedge_ratio(la, lb)
            except Exception as exc:
                logger.debug("PairSelector: %s/%s w=%d EG failed — %s", a, b, w, exc)
                window_pvalues[w] = None
                continue
            window_pvalues[w] = round(float(pval), 5)
            if pval < cfg.pvalue_threshold and np.isfinite(beta) and beta > 0:
                window_results[w] = (float(pval), float(beta))

        robustness_score = len(window_results)

        # ── 2. Multi-window robustness check ─────────────────────────────────
        if cfg.multi_window_check and robustness_score < cfg.min_robustness_score:
            passes_robustness = False
            rejection_reasons.append(
                f"robustesse insuffisante ({robustness_score}/{len(_COINT_WINDOWS)} "
                f"fenêtres, min={cfg.min_robustness_score})"
            )

        # ── Hedge ratio from best available window ─────────────────────────────
        if cfg.coint_lookback in window_results:
            primary_w = cfg.coint_lookback
        elif window_results:
            primary_w = max(window_results.keys())
        else:
            # No window passed — can't compute spread
            rejection_reasons.append(
                "aucune fenêtre cointégrée avec β > 0 et p < threshold"
            )
            return PairValidationReport(
                ticker_a=a, ticker_b=b, ma_risk=ma_risk,
                window_pvalues=window_pvalues, robustness_score=robustness_score,
                passes_robustness=passes_robustness,
                rolling_fail_rate=None, passes_rolling=False,
                hedge_ratio=None, spread_mean=None, spread_std=None,
                half_life=None, passes_halflife=False,
                hurst=None, passes_hurst=False,
                zero_crossings=None, passes_zero_crossings=False,
                is_tradable=False, rejection_reasons=rejection_reasons,
            )

        primary_pvalue, primary_beta = window_results[primary_w]
        hedge_ratio = round(primary_beta, 4)

        # ── 3. Rolling EG stability ────────────────────────────────────────────
        lookb_rows = min(len(df), cfg.rolling_lookback)
        df_2y = df.iloc[-lookb_rows:]
        la_2y = np.log(df_2y["a"].values)
        lb_2y = np.log(df_2y["b"].values)
        rolling_fail_rate, n_windows = self._rolling_pvalue_filter(la_2y, lb_2y)

        if n_windows < 2:
            # Not enough history for rolling test — skip filter
            passes_rolling    = True
            rolling_fail_rate = None
        else:
            passes_rolling = rolling_fail_rate <= cfg.max_rolling_fail_rate
            if not passes_rolling:
                rejection_reasons.append(
                    f"filtre glissant: {rolling_fail_rate:.0%} des fenêtres "
                    f"ont p > {cfg.rolling_pvalue_threshold} "
                    f"(max toléré {cfg.max_rolling_fail_rate:.0%})"
                )

        # ── Spread computation on primary window ───────────────────────────────
        df_primary = df.iloc[-primary_w:]
        la_p  = np.log(df_primary["a"].values)
        lb_p  = np.log(df_primary["b"].values)
        spread = la_p - primary_beta * lb_p

        # Spread stats on the rolling zscore window
        tail       = spread[-cfg.lookback:]
        spread_mean = float(np.mean(tail))
        spread_std  = float(np.std(tail, ddof=1))

        if spread_std < cfg.min_std or not np.isfinite(spread_std):
            rejection_reasons.append("variance du spread nulle — paire dégénérée")
            return PairValidationReport(
                ticker_a=a, ticker_b=b, ma_risk=ma_risk,
                window_pvalues=window_pvalues, robustness_score=robustness_score,
                passes_robustness=passes_robustness,
                rolling_fail_rate=rolling_fail_rate, passes_rolling=passes_rolling,
                hedge_ratio=hedge_ratio, spread_mean=spread_mean, spread_std=spread_std,
                half_life=None, passes_halflife=False,
                hurst=None, passes_hurst=False,
                zero_crossings=None, passes_zero_crossings=False,
                is_tradable=False, rejection_reasons=rejection_reasons,
            )

        # ── 4. Half-life (OU regression) ──────────────────────────────────────
        half_life = _compute_halflife(spread)
        if half_life is None:
            passes_halflife = False
            rejection_reasons.append(
                "half-life: non calculable (spread non stationnaire ou données trop courtes)"
            )
        elif half_life < cfg.min_halflife_days:
            passes_halflife = False
            rejection_reasons.append(
                f"half-life trop court ({half_life:.1f}j < {cfg.min_halflife_days}j) "
                "— coûts de transaction éroderaient le profit"
            )
        elif half_life > cfg.max_halflife_days:
            passes_halflife = False
            rejection_reasons.append(
                f"half-life trop long ({half_life:.1f}j > {cfg.max_halflife_days}j) "
                "— capital bloqué trop longtemps"
            )
        else:
            passes_halflife = True

        # ── 5. Hurst exponent ─────────────────────────────────────────────────
        hurst = _compute_hurst(spread)
        if hurst is None:
            passes_hurst = False
            rejection_reasons.append(
                "Hurst: non calculable (données insuffisantes ou variance nulle)"
            )
        elif hurst >= cfg.max_hurst:
            passes_hurst = False
            rejection_reasons.append(
                f"Hurst H={hurst:.3f} ≥ {cfg.max_hurst} "
                "— spread non mean-reverting (random walk ou persistant)"
            )
        else:
            passes_hurst = True

        # ── 6. Zero-crossings ─────────────────────────────────────────────────
        zero_crossings = _count_zero_crossings(spread)
        if zero_crossings < cfg.min_zero_crossings:
            passes_zero_crossings = False
            rejection_reasons.append(
                f"zero-crossings insuffisants ({zero_crossings} < {cfg.min_zero_crossings}) "
                "— relation potentiellement non stationnaire"
            )
        else:
            passes_zero_crossings = True

        is_tradable = len(rejection_reasons) == 0

        return PairValidationReport(
            ticker_a=a, ticker_b=b, ma_risk=ma_risk,
            window_pvalues=window_pvalues, robustness_score=robustness_score,
            passes_robustness=passes_robustness,
            rolling_fail_rate=rolling_fail_rate, passes_rolling=passes_rolling,
            hedge_ratio=hedge_ratio, spread_mean=spread_mean, spread_std=spread_std,
            half_life=half_life, passes_halflife=passes_halflife,
            hurst=hurst, passes_hurst=passes_hurst,
            zero_crossings=zero_crossings, passes_zero_crossings=passes_zero_crossings,
            is_tradable=is_tradable, rejection_reasons=rejection_reasons,
        )

    def _test_pair(self, a: str, b: str) -> Optional[ValidatedPair]:
        """
        Run full validation. Returns ValidatedPair if tradable, else None.
        Logs a warning when a high-M&A-risk pair becomes active.
        """
        report = self.validate_pair_full(a, b)
        if not report.is_tradable:
            logger.debug(
                "PairSelector: %s/%s REJECTED — %s",
                a, b,
                " | ".join(report.rejection_reasons) or "unknown",
            )
            return None

        if report.ma_risk == "high":
            logger.warning(
                "PairSelector: %s/%s ACCEPTED with ma_risk='high' — "
                "M&A event risk elevated; use conservative sizing",
                a, b,
            )

        primary_pvalue = report.window_pvalues.get(self._cfg.coint_lookback)
        if primary_pvalue is None:
            best_w = max(
                (w for w in _COINT_WINDOWS if report.window_pvalues.get(w) is not None),
                default=None,
            )
            primary_pvalue = report.window_pvalues.get(best_w, 1.0) if best_w else 1.0

        return ValidatedPair(
            ticker_a=a,
            ticker_b=b,
            hedge_ratio=report.hedge_ratio,
            pvalue=round(float(primary_pvalue), 5),
            last_tested=datetime.now(timezone.utc),
            spread_mean=round(report.spread_mean, 6) if report.spread_mean is not None else 0.0,
            spread_std=round(report.spread_std, 6)  if report.spread_std  is not None else 0.01,
            is_active=True,
            robustness_score=report.robustness_score,
            window_pvalues=report.window_pvalues,
            half_life=report.half_life,
            hurst=report.hurst,
            zero_crossings=report.zero_crossings or 0,
            ma_risk=report.ma_risk,
            rolling_fail_rate=report.rolling_fail_rate,
        )

    def _refresh(self, now: datetime) -> None:
        self._price_cache.clear()
        # Bulk-download all candidate tickers at once
        all_tickers = list({t for pair in CANDIDATE_PAIRS for t in pair})
        self._prefetch_prices(all_tickers)

        new: Dict[Tuple[str, str], ValidatedPair] = {}

        for a, b in CANDIDATE_PAIRS:
            pair = self._test_pair(a, b)
            if pair is not None:
                new[(a, b)] = pair
                logger.info(
                    "PairSelector: %s/%s ACCEPTED  p=%.5f  β=%.4f  "
                    "score=%d/%d  HL=%.1fj  H=%.3f  ZC=%d  M&A=%s  windows=%s",
                    a, b, pair.pvalue, pair.hedge_ratio,
                    pair.robustness_score, len(_COINT_WINDOWS),
                    pair.half_life or 0.0, pair.hurst or 0.0,
                    pair.zero_crossings, pair.ma_risk,
                    pair.window_pvalues,
                )
            else:
                logger.info("PairSelector: %s/%s REJECTED", a, b)

        # Preserve deactivated entries (cointegration broke mid-trade)
        for key, old in self._validated.items():
            if not old.is_active and key not in new:
                new[key] = old

        self._validated = new
        self._last_full_test = now
        n = sum(1 for p in new.values() if p.is_active)
        logger.info(
            "PairSelector: refresh done — %d/%d pairs accepted "
            "(multi_window_check=%s, min_score=%d)",
            n, len(CANDIDATE_PAIRS),
            self._cfg.multi_window_check, self._cfg.min_robustness_score,
        )

    # ── Public API ─────────────────────────────────────────────────────────────

    def get_validated_pairs(self) -> List[ValidatedPair]:
        """Return active validated pairs; triggers refresh when cache is stale."""
        now = datetime.now(timezone.utc)
        if (
            self._last_full_test is None
            or (now - self._last_full_test).days >= self._cfg.recheck_days
        ):
            self._refresh(now)
        return [p for p in self._validated.values() if p.is_active]

    def recheck_existing(self) -> None:
        """
        Re-test active pairs older than recheck_days without a full refresh.
        Deactivates any pair whose cointegration has broken.
        """
        now = datetime.now(timezone.utc)
        for key, pair in list(self._validated.items()):
            if not pair.is_active:
                continue
            if (now - pair.last_tested).days < self._cfg.recheck_days:
                continue
            rechecked = self._test_pair(*key)
            if rechecked is None:
                logger.warning(
                    "PairSelector: %s/%s cointegration BROKEN — deactivating", *key
                )
                d = asdict(pair)
                d["is_active"]    = False
                d["last_tested"]  = now
                self._validated[key] = ValidatedPair(**d)
            else:
                self._validated[key] = rechecked
                logger.info(
                    "PairSelector: %s/%s re-validated p=%.5f β=%.4f score=%d",
                    *key, rechecked.pvalue, rechecked.hedge_ratio, rechecked.robustness_score,
                )

    def get_pair_for_ticker(self, ticker: str) -> Optional[ValidatedPair]:
        """Return the active pair where ticker is the A leg, or None."""
        for (a, _b), pair in self._validated.items():
            if a == ticker and pair.is_active:
                return pair
        return None

    def invalidate_pair(self, ticker_a: str, ticker_b: str) -> None:
        """Force-deactivate a pair (e.g., single-leg execution abort)."""
        key = (ticker_a, ticker_b)
        if key in self._validated:
            d = asdict(self._validated[key])
            d["is_active"]   = False
            d["last_tested"] = datetime.now(timezone.utc)
            self._validated[key] = ValidatedPair(**d)
            logger.warning(
                "PairSelector: %s/%s force-invalidated (single-leg abort)",
                ticker_a, ticker_b,
            )

    def compute_zscore(self, pair: ValidatedPair) -> Optional[float]:
        """
        Compute the current z-score of the spread using cached prices.

            spread  = log(A) − β · log(B)
            z       = (spread_current − rolling_mean) / rolling_std

        Returns None if data is insufficient or std ≈ 0.
        The price cache is populated during get_validated_pairs(); call that first.
        """
        sa = self._price_cache.get(pair.ticker_a)
        sb = self._price_cache.get(pair.ticker_b)
        if sa is None or sb is None:
            return None

        df = pd.DataFrame({"a": sa, "b": sb}).dropna()
        n  = self._cfg.lookback + 5
        if len(df) < n:
            return None

        df     = df.iloc[-n:]
        spread = np.log(df["a"].values) - pair.hedge_ratio * np.log(df["b"].values)
        s      = pd.Series(spread)
        rm     = s.rolling(self._cfg.lookback).mean()
        rs     = s.rolling(self._cfg.lookback).std()

        std_val = float(rs.iloc[-1])
        if std_val < self._cfg.min_std or not np.isfinite(std_val):
            return None

        return round((float(s.iloc[-1]) - float(rm.iloc[-1])) / std_val, 4)

    # ── Diagnostic tool ────────────────────────────────────────────────────────

    def diagnostic_report(self) -> None:
        """
        Run the full multi-criteria validation on ALL candidate pairs and print
        a structured diagnostic table.

        Use this to see exactly why each pair passes or fails, and how many
        pairs in the sector universe are currently tradable.

            sel = PairSelector(PairsTradingConfig())
            sel.diagnostic_report()
        """
        all_tickers = sorted({t for pair in CANDIDATE_PAIRS for t in pair})
        self._prefetch_prices(all_tickers)

        reports: List[PairValidationReport] = []
        for a, b in CANDIDATE_PAIRS:
            report = self.validate_pair_full(a, b)
            reports.append(report)

        tradable = [r for r in reports if r.is_tradable]

        sep = "═" * 82
        print(f"\n{sep}")
        print("  PAIRS SELECTOR DIAGNOSTIC REPORT")
        print(f"  Universe: {len(CANDIDATE_PAIRS)} pairs  |  Date: "
              f"{datetime.now(timezone.utc).strftime('%Y-%m-%d')}")
        print(f"  Filter: EG multi-window + rolling + OU half-life + Hurst + zero-crossings")
        print(sep)

        for r in reports:
            verdict = "✓ TRADABLE" if r.is_tradable else "✗ REJECTED"

            pval_str = "  ".join(
                f"{w}j=" + ("—" if r.window_pvalues.get(w) is None
                            else f"{r.window_pvalues[w]:.3f}")
                for w in _COINT_WINDOWS
            )
            hl_str   = f"HL={r.half_life:.1f}j"   if r.half_life   is not None else "HL=—"
            h_str    = f"H={r.hurst:.3f}"          if r.hurst       is not None else "H=—"
            zc_str   = f"ZC={r.zero_crossings}"    if r.zero_crossings is not None else "ZC=—"
            roll_str = (f"roll_fail={r.rolling_fail_rate:.0%}"
                        if r.rolling_fail_rate is not None else "roll=—")

            print(f"\n  {r.ticker_a}/{r.ticker_b}  [{verdict}]"
                  f"  score={r.robustness_score}/3  M&A={r.ma_risk}")
            print(f"    EG:     {pval_str}")
            print(f"    OU:     {hl_str}  {h_str}  {zc_str}")
            print(f"    Filtre: {roll_str}")
            if r.hedge_ratio is not None:
                print(f"    β = {r.hedge_ratio:.4f}")
            if r.rejection_reasons:
                for reason in r.rejection_reasons:
                    print(f"    ✗  {reason}")

        print(f"\n{'─'*82}")
        print(f"  VERDICT: {len(tradable)}/{len(CANDIDATE_PAIRS)} paires tradables")
        if tradable:
            for r in tradable:
                hl_s = f"{r.half_life:.1f}j" if r.half_life is not None else "—"
                h_s  = f"{r.hurst:.3f}"      if r.hurst is not None else "—"
                print(f"    ✓  {r.ticker_a}/{r.ticker_b}  β={r.hedge_ratio:.4f}  "
                      f"score={r.robustness_score}/3  HL={hl_s}  H={h_s}  M&A={r.ma_risk}")
        else:
            print("    — Agent armé mais au repos. Aucune paire tradable actuellement.")
        print(f"{sep}\n")


# ── Agent ──────────────────────────────────────────────────────────────────────

class PairsTradingAgent(BaseAgent):
    """
    Statistical arbitrage agent — market-neutral pairs trading.

    Status: ARMED BUT AT REST
    -------------------------
    The agent is fully implemented and will trade the moment a robust
    cointegrated pair exists. Currently no pair in the sector universe
    passes the full multi-criteria filter, so the agent emits HOLD.

    Note on diversification
    -----------------------
    Trading a single pair does NOT constitute genuine market-neutral
    diversification. The correlated legs reduce market beta but idiosyncratic
    risk from a single relationship remains concentrated. The minimum
    meaningful operational state is 2+ uncorrelated active pairs.

    Signal structure (two-leg)
    --------------------------
    Every entry signal carries both legs in meta['pair_legs']:

        meta['pair_legs'] = {
            'long':  [ticker, gross_weight],   # the leg to buy
            'short': [ticker, gross_weight],   # the leg to sell short
        }
        meta['direction'] = 'long_a_short_b' | 'short_a_long_b'
        meta['strategy']  = 'market_neutral'

    Position state convention
    -------------------------
    portfolio[ticker_a] > 0  → in "long A / short B" trade
    portfolio[ticker_a] < 0  → in "short A / long B" trade
    portfolio[ticker_a] == 0 → flat
    """

    name = "PairsTradingAgent"

    def __init__(self, config: Optional[PairsTradingConfig] = None) -> None:
        self.cfg      = config or PairsTradingConfig()
        self.selector = PairSelector(self.cfg)

    def generate_signal(
        self,
        state: MarketState,
        portfolio: Dict[str, float],
        regime: Optional[str] = None,
        data: Optional[pd.DataFrame] = None,
    ) -> AgentSignal:
        cfg = self.cfg

        # ── 0. Intra-cycle cointegration re-check ─────────────────────────────
        self.selector.recheck_existing()

        # ── 1. Ensure at least one robust pair exists ──────────────────────────
        valid_pairs = self.selector.get_validated_pairs()
        if not valid_pairs:
            return self._hold(
                state, regime,
                reason=(
                    "PairsTrading: 0 robust cointegrated pairs "
                    f"(multi_window_check={cfg.multi_window_check}, "
                    f"min_score={cfg.min_robustness_score}) — armed but at rest"
                ),
            )

        # ── 2. This ticker must be the A leg of an active pair ─────────────────
        pair = self.selector.get_pair_for_ticker(state.symbol)
        if pair is None:
            return self._hold(
                state, regime,
                reason=f"PairsTrading: {state.symbol} is not the A-leg of any active pair",
            )

        # ── 3. Compute z-score ────────────────────────────────────────────────
        zscore = self.selector.compute_zscore(pair)
        if zscore is None:
            return self._hold(
                state, regime,
                reason=(
                    f"PairsTrading: {pair.ticker_a}/{pair.ticker_b} — "
                    "insufficient price data for z-score"
                ),
            )

        # ── 4. Current position state ──────────────────────────────────────────
        a_pos       = portfolio.get(state.symbol, 0.0)
        in_long_a   = a_pos >  1e-9   # long A / short B
        in_short_a  = a_pos < -1e-9   # short A / long B
        in_position = in_long_a or in_short_a

        base_meta = {
            "strategy":         "market_neutral",
            "regime":           regime,
            "pair":             f"{pair.ticker_a}/{pair.ticker_b}",
            "hedge_ratio":      pair.hedge_ratio,
            "pvalue":           pair.pvalue,
            "robustness_score": pair.robustness_score,
            "window_pvalues":   pair.window_pvalues,
            "zscore":           zscore,
            "half_life":        pair.half_life,
            "hurst":            pair.hurst,
            "ma_risk":          pair.ma_risk,
        }

        # ── 5. Exit logic ─────────────────────────────────────────────────────

        if in_position:
            direction = "long_a_short_b" if in_long_a else "short_a_long_b"

            # 5a. Cointegration breakdown
            if not pair.is_active:
                return self._close_signal(
                    state, pair, zscore, direction, base_meta,
                    reason=(
                        f"PairsTrading CLOSE {pair.ticker_a}/{pair.ticker_b} "
                        "— cointegration broke, forced exit"
                    ),
                    confidence=0.99,
                )

            # 5b. Stop-loss
            if abs(zscore) > cfg.zscore_stoploss:
                return self._close_signal(
                    state, pair, zscore, direction, base_meta,
                    reason=(
                        f"PairsTrading STOP-LOSS {pair.ticker_a}/{pair.ticker_b} "
                        f"z={zscore:+.3f} > {cfg.zscore_stoploss}"
                    ),
                    confidence=0.98,
                )

            # 5c. Mean reversion complete
            if abs(zscore) < cfg.zscore_exit:
                return self._close_signal(
                    state, pair, zscore, direction, base_meta,
                    reason=(
                        f"PairsTrading EXIT {pair.ticker_a}/{pair.ticker_b} "
                        f"z={zscore:+.3f} — mean reversion complete"
                    ),
                    confidence=0.90,
                )

            # 5d. Waiting for convergence
            return self._hold(
                state, regime,
                reason=(
                    f"PairsTrading HOLD {pair.ticker_a}/{pair.ticker_b} "
                    f"z={zscore:+.3f} in [{-cfg.zscore_exit:.1f}, "
                    f"+{cfg.zscore_exit:.1f}] — waiting for convergence"
                ),
                extra_meta=base_meta,
            )

        # ── 6. Entry logic ─────────────────────────────────────────────────────

        # 6a. Spread abnormally low → LONG A / SHORT β·B
        if zscore < -cfg.zscore_entry:
            conf = min(0.90, 0.60 + (abs(zscore) - cfg.zscore_entry) * 0.10)
            conf = min(0.93, conf + 0.03 * max(0, pair.robustness_score - cfg.min_robustness_score))
            return AgentSignal(
                agent_name=self.name,
                symbol=state.symbol,
                action="BUY",
                confidence=round(conf, 3),
                target_weight=cfg.target_weight,
                reason=(
                    f"PairsTrading LONG {pair.ticker_a} / SHORT {pair.ticker_b} "
                    f"z={zscore:+.3f}  β={pair.hedge_ratio:.4f}  "
                    f"score={pair.robustness_score}/{len(_COINT_WINDOWS)}"
                ),
                meta={
                    **base_meta,
                    "direction":  "long_a_short_b",
                    "pair_legs": {
                        "long":  [pair.ticker_a, cfg.target_weight],
                        "short": [pair.ticker_b, cfg.target_weight * pair.hedge_ratio],
                    },
                },
            )

        # 6b. Spread abnormally high → SHORT A / LONG β·B
        if zscore > cfg.zscore_entry:
            conf = min(0.90, 0.60 + (abs(zscore) - cfg.zscore_entry) * 0.10)
            conf = min(0.93, conf + 0.03 * max(0, pair.robustness_score - cfg.min_robustness_score))
            return AgentSignal(
                agent_name=self.name,
                symbol=state.symbol,
                action="BUY",
                confidence=round(conf, 3),
                target_weight=cfg.target_weight,
                reason=(
                    f"PairsTrading SHORT {pair.ticker_a} / LONG {pair.ticker_b} "
                    f"z={zscore:+.3f}  β={pair.hedge_ratio:.4f}  "
                    f"score={pair.robustness_score}/{len(_COINT_WINDOWS)}"
                ),
                meta={
                    **base_meta,
                    "direction":  "short_a_long_b",
                    "pair_legs": {
                        "long":  [pair.ticker_b, cfg.target_weight],
                        "short": [pair.ticker_a, cfg.target_weight],
                    },
                },
            )

        # 6c. Inside the band — no signal
        return self._hold(
            state, regime,
            reason=(
                f"PairsTrading HOLD {pair.ticker_a}/{pair.ticker_b} "
                f"z={zscore:+.3f} inside [{-cfg.zscore_entry:.1f}, "
                f"+{cfg.zscore_entry:.1f}]"
            ),
            extra_meta=base_meta,
        )

    # ── Signal factories ───────────────────────────────────────────────────────

    def _hold(
        self,
        state: MarketState,
        regime: Optional[str],
        *,
        reason: str,
        extra_meta: Optional[Dict] = None,
    ) -> AgentSignal:
        return AgentSignal(
            agent_name=self.name,
            symbol=state.symbol,
            action="HOLD",
            confidence=0.0,
            target_weight=0.0,
            reason=reason,
            meta={
                "strategy": "market_neutral",
                "regime": regime,
                **(extra_meta or {}),
            },
        )

    def _close_signal(
        self,
        state: MarketState,
        pair: ValidatedPair,
        zscore: float,
        previous_direction: str,
        base_meta: Dict,
        *,
        reason: str,
        confidence: float,
    ) -> AgentSignal:
        if previous_direction == "long_a_short_b":
            legs = {"close_long": pair.ticker_a, "close_short": pair.ticker_b}
        else:
            legs = {"close_long": pair.ticker_b, "close_short": pair.ticker_a}
        return AgentSignal(
            agent_name=self.name,
            symbol=state.symbol,
            action="SELL",
            confidence=confidence,
            target_weight=0.0,
            reason=reason,
            meta={
                **base_meta,
                "direction":          "close",
                "previous_direction": previous_direction,
                "pair_legs":          legs,
            },
        )
