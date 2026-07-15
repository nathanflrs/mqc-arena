# src/agents/macro.py
"""
MacroAgent — Rebuilt on real FRED data (Bridgewater-inspired top-down macro).

Primary signals (FRED series):
  T10Y2Y   : yield curve (10Y − 2Y). Inversion < 0 → recession leading indicator.
  BAMLH0A0HYM2 : HY OAS credit spread (bps). > 500 = stress; < 300 = benign.
  VIXCLS   : CBOE VIX daily close. > 30 = fear; < 15 = calm.
  FEDFUNDS : Fed funds effective rate. Trend (6m ΔΔ) encodes tightening/easing.

Fallback: if FRED key absent or API unreachable, falls back to ETF-proxy
momentum (SPY/GLD/TLT) — same logic as the previous MacroAgent — so the
runner never loses a signal.

Cache: logs/fred_cache/<series>_<date>.json with 4h TTL (same pattern as
news cache). FRED series update once per business day so cache is safe.

Ticker confirmation: kept from the previous version — the underlying equity
must confirm the macro signal (momentum > threshold) to avoid false positives.
"""
from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import Dict, Optional

import pandas as pd

from src.agents.base import BaseAgent, MarketState, AgentSignal
from src.data.market_data import download_ohlcv

logger = logging.getLogger(__name__)

# ── FRED series IDs ────────────────────────────────────────────────────────────

_YIELD_CURVE  = "T10Y2Y"           # 10Y-2Y spread (pct pts)
_HY_SPREAD    = "BAMLH0A0HYM2"     # HY OAS (bps)
_VIX          = "VIXCLS"           # VIX close
_FED_FUNDS    = "FEDFUNDS"         # monthly fed funds rate

_CACHE_DIR    = Path("logs/fred_cache")
_CACHE_TTL    = timedelta(hours=4)


# ── FRED client (thin wrapper + file cache) ────────────────────────────────────

class FREDClient:
    """Fetches FRED series via fredapi, caches to disk to limit API calls."""

    def __init__(self, api_key: Optional[str] = None) -> None:
        self._key = api_key or os.getenv("FRED_API_KEY", "")
        _CACHE_DIR.mkdir(parents=True, exist_ok=True)

    @property
    def available(self) -> bool:
        return bool(self._key)

    def _cache_path(self, series_id: str) -> Path:
        today = datetime.now(timezone.utc).date().isoformat()
        return _CACHE_DIR / f"{series_id}_{today}.json"

    def _load_cache(self, series_id: str) -> Optional[float]:
        p = self._cache_path(series_id)
        if not p.exists():
            return None
        try:
            data = json.loads(p.read_text())
            cached_at = datetime.fromisoformat(data["cached_at"])
            if datetime.now(timezone.utc) - cached_at > _CACHE_TTL:
                return None
            return float(data["value"])
        except Exception:
            return None

    def _save_cache(self, series_id: str, value: float) -> None:
        try:
            self._cache_path(series_id).write_text(json.dumps({
                "cached_at": datetime.now(timezone.utc).isoformat(),
                "value": value,
            }))
        except Exception as exc:
            logger.debug("FREDClient: cache write failed for %s: %s", series_id, exc)

    def latest(self, series_id: str, periods: int = 1) -> Optional[float | list[float]]:
        """
        Return the most recent observation(s) for a FRED series.
        periods=1 → float, periods>1 → list[float] (most recent first).
        Returns None on failure.
        """
        if periods == 1:
            cached = self._load_cache(series_id)
            if cached is not None:
                return cached

        if not self._key:
            logger.debug("FREDClient: no API key — cannot fetch %s", series_id)
            return None

        try:
            from fredapi import Fred
            fred  = Fred(api_key=self._key)
            s     = fred.get_series(series_id)
            s     = s.dropna()
            if s.empty:
                return None
            if periods == 1:
                val = float(s.iloc[-1])
                self._save_cache(series_id, val)
                return val
            return [float(v) for v in s.iloc[-periods:][::-1]]
        except Exception as exc:
            logger.warning("FREDClient: fetch failed for %s — %s", series_id, exc)
            return None


# ── Scoring engine ─────────────────────────────────────────────────────────────

@dataclass
class MacroIndicators:
    yield_curve: Optional[float]   # T10Y2Y pct pts (None = unavailable)
    hy_spread:   Optional[float]   # BAMLH0A0HYM2 bps
    vix:         Optional[float]   # VIXCLS
    fed_trend:   Optional[float]   # 6-month ΔΔ fedfunds (positive = tightening)


def score_fred_indicators(ind: MacroIndicators) -> tuple[int, int]:
    """
    Returns (risk_on_score, risk_off_score) from FRED indicators.
    Max 8 points each (2 per indicator).
    """
    risk_on = 0
    risk_off = 0

    # Yield curve — leading indicator with ~12m lag
    if ind.yield_curve is not None:
        if ind.yield_curve < -0.25:      risk_off += 2   # strong inversion
        elif ind.yield_curve < 0:        risk_off += 1   # mild inversion
        elif ind.yield_curve > 1.0:      risk_on  += 2   # steep = expansion
        elif ind.yield_curve > 0.5:      risk_on  += 1

    # HY credit spread — coincident, risk-appetite barometer
    if ind.hy_spread is not None:
        if ind.hy_spread > 600:          risk_off += 2   # stress / near-crisis
        elif ind.hy_spread > 450:        risk_off += 1
        elif ind.hy_spread < 300:        risk_on  += 2   # benign credit
        elif ind.hy_spread < 380:        risk_on  += 1

    # VIX — sentiment / fear gauge
    if ind.vix is not None:
        if ind.vix > 30:                 risk_off += 2
        elif ind.vix > 23:               risk_off += 1
        elif ind.vix < 15:               risk_on  += 1   # calm markets

    # Fed funds trend — policy direction
    if ind.fed_trend is not None:
        if ind.fed_trend > 0.75:         risk_off += 1   # tightening cycle
        elif ind.fed_trend < -0.50:      risk_on  += 1   # easing cycle

    return risk_on, risk_off


# ── Config ─────────────────────────────────────────────────────────────────────

@dataclass
class MacroConfig:
    # FRED thresholds inherited from score_fred_indicators()
    # ETF-proxy fallback thresholds
    spy_threshold: float = 0.05
    gld_threshold: float = 0.05
    tlt_threshold: float = 0.03
    # Signal thresholds
    risk_on_min:  int   = 4     # raised from 3 (more indicators, higher bar)
    risk_off_min: int   = 4
    # Sizing
    target_weight:       float = 0.05
    sell_confidence:     float = 0.80
    buy_base_confidence: float = 0.60
    momentum_period:     int   = 63
    ticker_momentum_threshold: float = 0.02


# ── Agent ──────────────────────────────────────────────────────────────────────

class MacroAgent(BaseAgent):
    """
    MacroAgent (Bridgewater-inspired) — rebuilt on FRED macro data.

    Signal hierarchy:
      1. FRED indicators (T10Y2Y, HY spread, VIX, Fed trend) when key is set.
      2. ETF-proxy fallback (SPY/GLD/TLT momentum) when FRED is unavailable.
      3. Ticker-specific momentum as confirmation filter (both paths).
    """
    name = "MacroAgent"

    def __init__(
        self,
        config:    Optional[MacroConfig] = None,
        preloaded: Optional[Dict[str, pd.DataFrame]] = None,
        fred_client: Optional[FREDClient] = None,
    ):
        self.cfg         = config or MacroConfig()
        self._preloaded  = preloaded or {}
        self._fred       = fred_client or FREDClient()

    # ── FRED path ─────────────────────────────────────────────────────────────

    def _fetch_fred_indicators(self) -> Optional[MacroIndicators]:
        """Fetch all FRED series. Returns None if FRED is unavailable."""
        if not self._fred.available:
            return None
        try:
            yield_curve = self._fred.latest(_YIELD_CURVE)
            hy_spread   = self._fred.latest(_HY_SPREAD)
            vix         = self._fred.latest(_VIX)

            # Fed funds trend: latest vs 6 months ago
            fed_series = self._fred.latest(_FED_FUNDS, periods=7)  # ~6 months of monthly data
            if isinstance(fed_series, list) and len(fed_series) >= 7:
                fed_trend = float(fed_series[0]) - float(fed_series[-1])
            else:
                fed_trend = None

            return MacroIndicators(
                yield_curve = float(yield_curve) if yield_curve is not None else None,
                hy_spread   = float(hy_spread)   if hy_spread   is not None else None,
                vix         = float(vix)         if vix         is not None else None,
                fed_trend   = fed_trend,
            )
        except Exception as exc:
            logger.warning("MacroAgent: FRED fetch error — %s", exc)
            return None

    # ── ETF-proxy fallback ────────────────────────────────────────────────────

    def _momentum(self, symbol: str, period: int = 63) -> float:
        """ETF proxy momentum — used as fallback when FRED is unavailable."""
        try:
            df = self._preloaded.get(symbol)
            if df is None:
                df = download_ohlcv(symbol, period="1y")
            close = pd.to_numeric(df["Close"], errors="coerce").dropna()
            if len(close) < period + 1:
                return 0.0
            return float(close.iloc[-1] / close.iloc[-period] - 1.0)
        except Exception:
            return 0.0

    def _etf_proxy_scores(self) -> tuple[int, int, dict]:
        """Compute risk_on/risk_off from ETF proxies (fallback path)."""
        mom_spy = self._momentum("SPY", period=self.cfg.momentum_period)
        mom_gld = self._momentum("GLD", period=self.cfg.momentum_period)
        mom_tlt = self._momentum("TLT", period=self.cfg.momentum_period)

        risk_on = risk_off = 0
        if mom_spy > self.cfg.spy_threshold:     risk_on  += 2
        elif mom_spy < -self.cfg.spy_threshold:  risk_off += 2
        if mom_gld > self.cfg.gld_threshold:     risk_off += 1
        elif mom_gld < 0.0:                      risk_on  += 1
        if mom_tlt > self.cfg.tlt_threshold:     risk_off += 1
        elif mom_tlt < -self.cfg.tlt_threshold:  risk_on  += 1

        extra = {"mom_spy": round(mom_spy, 4),
                 "mom_gld": round(mom_gld, 4),
                 "mom_tlt": round(mom_tlt, 4),
                 "source": "etf_proxy"}
        return risk_on, risk_off, extra

    # ── Ticker confirmation (both paths) ──────────────────────────────────────

    def _ticker_momentum(self, symbol: str, data: Optional[pd.DataFrame],
                         period: int = 63) -> float:
        try:
            df = data if data is not None else download_ohlcv(symbol, period="1y")
            close = pd.to_numeric(df["Close"], errors="coerce").dropna()
            if len(close) < period + 1:
                return 0.0
            return float(close.iloc[-1] / close.iloc[-period] - 1.0)
        except Exception:
            return 0.0

    # ── Main signal ───────────────────────────────────────────────────────────

    def generate_signal(
        self,
        state:     MarketState,
        portfolio: Dict[str, float],
        regime:    Optional[str] = None,
        data:      Optional[pd.DataFrame] = None,
    ) -> AgentSignal:

        # 1) Try FRED path
        fred_ind = self._fetch_fred_indicators()
        if fred_ind is not None:
            risk_on, risk_off = score_fred_indicators(fred_ind)
            extra_meta = {
                "source":      "fred",
                "yield_curve": fred_ind.yield_curve,
                "hy_spread":   fred_ind.hy_spread,
                "vix":         fred_ind.vix,
                "fed_trend":   fred_ind.fed_trend,
            }
        else:
            # 2) Fallback: ETF proxy
            risk_on, risk_off, extra_meta = self._etf_proxy_scores()

        # 3) Ticker confirmation
        mom_ticker = self._ticker_momentum(state.symbol, data,
                                           period=self.cfg.momentum_period)
        thr = self.cfg.ticker_momentum_threshold

        meta = {
            "regime":         regime,
            "risk_on_score":  risk_on,
            "risk_off_score": risk_off,
            "mom_ticker":     round(mom_ticker, 4),
            **extra_meta,
        }

        in_position = portfolio.get(state.symbol, 0.0) > 0

        # 4) SELL — ticker must confirm weakness
        if in_position and risk_off >= self.cfg.risk_off_min and mom_ticker < -thr:
            src = extra_meta.get("source", "etf_proxy")
            return AgentSignal(
                agent_name=self.name,
                symbol=state.symbol,
                action="SELL",
                confidence=self.cfg.sell_confidence,
                target_weight=0.0,
                reason=(
                    f"Macro RISK OFF [{src}]: off={risk_off} on={risk_on} "
                    f"ticker={mom_ticker:.2%}"
                ),
                meta=meta,
            )

        # 5) BUY — ticker must confirm strength
        if risk_on >= self.cfg.risk_on_min and regime != "bear" and mom_ticker > thr:
            conf = min(0.85, self.cfg.buy_base_confidence
                       + (0.10 if risk_on >= self.cfg.risk_on_min + 2 else 0.0))
            src = extra_meta.get("source", "etf_proxy")
            return AgentSignal(
                agent_name=self.name,
                symbol=state.symbol,
                action="BUY",
                confidence=conf,
                target_weight=self.cfg.target_weight,
                reason=(
                    f"Macro RISK ON [{src}]: on={risk_on} off={risk_off} "
                    f"ticker={mom_ticker:.2%}"
                ),
                meta=meta,
            )

        # 6) HOLD
        return AgentSignal(
            agent_name=self.name,
            symbol=state.symbol,
            action="HOLD",
            confidence=0.4,
            target_weight=0.0,
            reason=f"Macro MIXED: risk_on={risk_on} risk_off={risk_off}",
            meta=meta,
        )
