"""
Dividend Arbitrage Agent — institutional-grade implementation.

Architecture:
  DividendDataCollector    — yfinance dividend calendar + regularity scoring
  DividendPricingModel     — carry, price risk, tax, put-call parity
  DividendPositionTracker  — thread-safe persistent state (logs/dividend_positions.json)
  DividendArbitrageAgent   — arena-compatible signal generator
  generate_dividend_report — 30-day calendar formatted for Telegram

Entry window : J-7 to J-2 before ex-dividend date
Exit window  : J+1 (default), take-profit at +1.5×div, stop-loss at -2×div
"""
from __future__ import annotations

import json
import logging
import math
import threading
from dataclasses import dataclass
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import yfinance as yf

from src.agents.base import AgentSignal, BaseAgent, MarketState

# ── Module-level logger ───────────────────────────────────────────────────────
_LOG_PATH = Path("logs/dividend_arb.log")


def _configure_log() -> logging.Logger:
    log = logging.getLogger("dividend_arb")
    if not log.handlers:
        _LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
        fh = logging.FileHandler(_LOG_PATH)
        fh.setFormatter(logging.Formatter("%(asctime)s %(levelname)-8s %(message)s"))
        log.addHandler(fh)
        log.setLevel(logging.INFO)
    return log


_log = _configure_log()

# ── Constants ─────────────────────────────────────────────────────────────────
_TAX_WITHHOLDING   = 0.15    # US dividend withholding rate for LU residents
_SPREAD_PCT        = 0.0002  # one-way half-spread estimate
_COMMISSION_PCT    = 0.0005  # round-trip commission (IB tiered ~$0.005/share)
_TC_ROUND_TRIP     = 2 * _SPREAD_PCT + _COMMISSION_PCT  # 0.0009 (9 bps)
_DEFAULT_RF        = 0.05    # fallback risk-free rate when ^IRX unavailable
_DEFAULT_VOL_DAILY = 0.01    # fallback daily vol (~16% ann.) when data insufficient
_NOTIONAL_REPORT   = 10_000.0  # $ notional used in dividend calendar report


# ═══════════════════════════════════════════════════════════════════════════════
# PART 1 — DIVIDEND DATA COLLECTOR
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class DividendInfo:
    ticker: str
    ex_date: date
    dividend_amount: float       # per-share, current payment
    payment_date: Optional[date]
    record_date: Optional[date]
    current_price: float
    days_to_ex: int              # negative = past ex-date
    yield_annualized: float      # (div/price) × (365/days_to_ex)
    div_regularity: float        # 1 - CV of 3y history, clamped [0, 1]
    is_special: bool             # True if amount > 2× 3y mean


class DividendDataCollector:
    """Fetches dividend calendar and history from yfinance, with full error isolation."""

    _rf_cache: Optional[float] = None  # shared across instances within a process run

    # ── Public API ────────────────────────────────────────────────────────────

    def get_dividend_info(self, ticker: str) -> Optional[DividendInfo]:
        """
        Returns DividendInfo for the next upcoming dividend.
        Returns None silently if: no dividends, stale data, or any yfinance error.
        """
        try:
            t    = yf.Ticker(ticker)
            info = t.info

            ex_ts = info.get("exDividendDate")
            if not ex_ts:
                return None

            ex_date = date.fromtimestamp(int(ex_ts))
            today   = date.today()

            # Skip events > 1 day in the past (stale yfinance data)
            if ex_date < today - timedelta(days=1):
                return None

            div_amount = self._extract_div_amount(info)
            if div_amount <= 0:
                return None

            price = float(info.get("currentPrice") or info.get("regularMarketPrice") or 0.0)
            if price <= 0:
                return None

            days_to_ex = (ex_date - today).days
            safe_days  = max(days_to_ex, 1)  # avoid /0 on ex-date itself
            yield_ann  = (div_amount / price) * (365.0 / safe_days)

            # 3y history for regularity + special dividend detection
            regularity, is_special = self._compute_regularity(t.dividends, div_amount)

            # Payment date (best-effort — yfinance field is inconsistent)
            payment_ts = info.get("lastDividendDate")
            payment_dt = date.fromtimestamp(int(payment_ts)) if payment_ts else None

            return DividendInfo(
                ticker          = ticker,
                ex_date         = ex_date,
                dividend_amount = div_amount,
                payment_date    = payment_dt,
                record_date     = None,  # not reliably available from yfinance
                current_price   = price,
                days_to_ex      = days_to_ex,
                yield_annualized= yield_ann,
                div_regularity  = regularity,
                is_special      = is_special,
            )
        except Exception as exc:
            _log.debug("get_dividend_info(%s): %s", ticker, exc)
            return None

    def get_risk_free_rate(self) -> float:
        """Fetch 13-week T-Bill rate (^IRX). Falls back to 5%."""
        if DividendDataCollector._rf_cache is not None:
            return DividendDataCollector._rf_cache
        try:
            hist = yf.Ticker("^IRX").history(period="5d")["Close"]
            if len(hist) >= 1:
                rf = float(hist.iloc[-1]) / 100.0
                DividendDataCollector._rf_cache = rf
                return rf
        except Exception:
            pass
        return _DEFAULT_RF

    # ── Helpers ───────────────────────────────────────────────────────────────

    @staticmethod
    def _extract_div_amount(info: dict) -> float:
        """Prefer lastDividendValue, fall back to dividendRate/4."""
        v = info.get("lastDividendValue")
        if v and float(v) > 0:
            return float(v)
        annual = float(info.get("dividendRate") or 0.0)
        return annual / 4 if annual > 0 else 0.0

    @staticmethod
    def _compute_regularity(hist: pd.Series, current_amount: float) -> Tuple[float, bool]:
        """
        Returns (regularity, is_special).
        regularity = 1 - CV (coefficient of variation), clamped [0, 1].
        is_special  = current_amount > 2× historical mean.
        """
        if hist is None or len(hist) == 0:
            return 0.5, False  # neutral: no history available

        cutoff = pd.Timestamp.now(tz="UTC") - pd.DateOffset(years=3)
        try:
            recent = hist[hist.index >= cutoff]
        except Exception:
            recent = hist  # timezone-naive fallback

        if len(recent) < 2:
            return 0.5, False

        vals     = recent.values.astype(float)
        mean_div = float(np.mean(vals))
        if mean_div <= 0:
            return 0.5, False

        cv         = float(np.std(vals)) / mean_div
        regularity = float(np.clip(1.0 - cv, 0.0, 1.0))
        is_special = current_amount > 2.0 * mean_div

        return regularity, is_special


# ═══════════════════════════════════════════════════════════════════════════════
# PART 2 — DIVIDEND PRICING MODEL
# ═══════════════════════════════════════════════════════════════════════════════

class DividendPricingModel:
    """
    Net dividend edge net of carry, price risk, tax, and transaction costs.
    All monetary units are USD.
    """

    def __init__(self, risk_free_rate: float = _DEFAULT_RF):
        self.rf = risk_free_rate

    # ── Cost components ───────────────────────────────────────────────────────

    def carry_cost(self, position_size: float, days_to_ex: int) -> float:
        """Funding cost of holding until ex-date (risk-free rate × time)."""
        return position_size * self.rf * (max(days_to_ex, 1) / 365.0)

    def price_risk(self, position_size: float, vol_daily: float, days_to_ex: int) -> float:
        """95th-percentile adverse price move over the holding period."""
        return position_size * vol_daily * math.sqrt(max(days_to_ex, 1)) * 1.645

    def transaction_costs(self, position_size: float) -> float:
        """Round-trip spread + commission (9 bps total)."""
        return position_size * _TC_ROUND_TRIP

    # ── Net edge ──────────────────────────────────────────────────────────────

    def net_edge(
        self,
        div_info: DividendInfo,
        shares: float,
        position_size: float,
        vol_daily: float,
        risk_aversion: float = 0.5,
    ) -> Tuple[float, dict]:
        """
        Returns (net_edge_usd, breakdown_dict).
        net_edge > 0 → trade is profitable in expectation after all costs.
        """
        gross_div = div_info.dividend_amount * shares
        tax       = gross_div * _TAX_WITHHOLDING
        net_div   = gross_div - tax

        carry = self.carry_cost(position_size, div_info.days_to_ex)
        risk  = self.price_risk(position_size, vol_daily, div_info.days_to_ex)
        costs = self.transaction_costs(position_size)

        edge = net_div - carry - costs - risk * risk_aversion

        breakdown = {
            "gross_dividend":    round(gross_div, 4),
            "tax_withholding":   round(tax, 4),
            "net_dividend":      round(net_div, 4),
            "carry_cost":        round(carry, 4),
            "price_risk_95pct":  round(risk, 4),
            "transaction_costs": round(costs, 4),
            "risk_aversion":     risk_aversion,
            "net_edge":          round(edge, 4),
        }
        return edge, breakdown

    # ── Put-call parity check ─────────────────────────────────────────────────

    def check_put_call_parity(
        self,
        ticker: str,
        ex_date: date,
        current_price: float,
        dividend_amount: float,
    ) -> bool:
        """
        Returns True if put-call parity is violated by > 0.5% near ATM.
        C − P  ≈  S − PV(K) − PV(D)

        We flag this as informational only — we do NOT trade options directly.
        Returns False on any error or missing data.
        """
        try:
            t            = yf.Ticker(ticker)
            expirations  = t.options
            if not expirations:
                return False

            # First expiry after ex-date (so the option captures the dividend)
            target = ex_date + timedelta(days=1)
            valid  = [
                e for e in expirations
                if datetime.strptime(e, "%Y-%m-%d").date() >= target
            ]
            if not valid:
                return False

            chain = t.option_chain(valid[0])
            calls, puts = chain.calls, chain.puts
            if calls.empty or puts.empty:
                return False

            # Closest ATM strike
            atm_strike = float(
                calls.loc[(calls["strike"] - current_price).abs().idxmin(), "strike"]
            )
            c_rows = calls[calls["strike"] == atm_strike]
            p_rows = puts[puts["strike"]  == atm_strike]
            if c_rows.empty or p_rows.empty:
                return False

            C = float(c_rows["lastPrice"].iloc[0])
            P = float(p_rows["lastPrice"].iloc[0])
            K = atm_strike

            # Time to expiry
            exp_dt = datetime.strptime(valid[0], "%Y-%m-%d").date()
            T      = max((exp_dt - date.today()).days / 365.0, 1.0 / 365.0)
            PV_K   = K * math.exp(-self.rf * T)
            PV_D   = dividend_amount  # discount negligible for <30-day holding

            lhs = C - P
            rhs = current_price - PV_K - PV_D
            return abs(lhs - rhs) / current_price > 0.005

        except Exception:
            return False


# ═══════════════════════════════════════════════════════════════════════════════
# PART 4 — POSITION TRACKER (persistent, thread-safe)
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class DivPosition:
    ticker: str
    entry_date: str       # ISO date string (YYYY-MM-DD)
    entry_price: float
    ex_date: str          # ISO date string
    dividend_amount: float
    target_exit_date: str # ISO date string (ex_date + exit_days_after)
    shares: float

    def to_dict(self) -> dict:
        return {k: v for k, v in self.__dict__.items()}

    @classmethod
    def from_dict(cls, d: dict) -> "DivPosition":
        return cls(**{k: d[k] for k in cls.__dataclass_fields__})


class DividendPositionTracker:
    """
    Thread-safe JSON-backed tracker for open and closed dividend arb positions.

    Open positions: logs/dividend_positions.json
    Closed trades:  logs/dividend_positions_closed.json
    """

    _POSITIONS_PATH: Path = Path("logs/dividend_positions.json")
    _CLOSED_PATH:    Path = Path("logs/dividend_positions_closed.json")
    _lock = threading.Lock()

    def __init__(self) -> None:
        with self._lock:
            self._positions: Dict[str, DivPosition] = self._load_positions()

    # ── Persistence ───────────────────────────────────────────────────────────

    def _load_positions(self) -> Dict[str, DivPosition]:
        if not self._POSITIONS_PATH.exists():
            return {}
        try:
            raw = json.loads(self._POSITIONS_PATH.read_text())
            return {k: DivPosition.from_dict(v) for k, v in raw.items()}
        except Exception:
            return {}

    def _save_positions(self) -> None:
        self._POSITIONS_PATH.parent.mkdir(parents=True, exist_ok=True)
        data = {k: v.to_dict() for k, v in self._positions.items()}
        self._POSITIONS_PATH.write_text(json.dumps(data, indent=2))

    def _append_closed(self, entry: dict) -> None:
        closed: list = []
        if self._CLOSED_PATH.exists():
            try:
                closed = json.loads(self._CLOSED_PATH.read_text())
            except Exception:
                pass
        closed.append(entry)
        self._CLOSED_PATH.parent.mkdir(parents=True, exist_ok=True)
        self._CLOSED_PATH.write_text(json.dumps(closed, indent=2))

    # ── Position management ───────────────────────────────────────────────────

    def open_position(self, pos: DivPosition) -> None:
        with self._lock:
            self._positions[pos.ticker] = pos
            self._save_positions()
        _log.info(
            "DIV_OPEN  %s entry=%.2f ex=%s div=%.4f shares=%.1f",
            pos.ticker, pos.entry_price, pos.ex_date, pos.dividend_amount, pos.shares,
        )

    def get_position(self, ticker: str) -> Optional[DivPosition]:
        with self._lock:
            return self._positions.get(ticker)

    def remove_stale(self, ticker: str) -> None:
        """Remove tracker entry for a ticker not in portfolio (rejected buy)."""
        with self._lock:
            if ticker in self._positions:
                del self._positions[ticker]
                self._save_positions()

    def close_position(self, ticker: str, exit_price: float, reason: str) -> Optional[float]:
        """
        Removes open position and appends to closed log.
        Returns realized P&L (USD) or None if no position found.

        P&L = price change + net dividend received − round-trip costs
        """
        with self._lock:
            pos = self._positions.pop(ticker, None)
            if pos is None:
                return None
            self._save_positions()

            shares      = pos.shares
            price_pnl   = (exit_price - pos.entry_price) * shares
            net_div     = pos.dividend_amount * shares * (1 - _TAX_WITHHOLDING)
            rt_costs    = (pos.entry_price + exit_price) * shares * (_SPREAD_PCT + _COMMISSION_PCT / 2)
            pnl         = price_pnl + net_div - rt_costs

            # _append_closed inside lock — prevents concurrent writes corrupting the JSON file.
            self._append_closed({
                "ticker":          ticker,
                "entry_date":      pos.entry_date,
                "exit_date":       date.today().isoformat(),
                "entry_price":     pos.entry_price,
                "exit_price":      exit_price,
                "shares":          shares,
                "dividend_amount": pos.dividend_amount,
                "pnl":             round(pnl, 2),
                "reason":          reason,
            })

        _log.info("DIV_CLOSE %s exit=%.2f pnl=%.2f reason=%s", ticker, exit_price, pnl, reason)
        return pnl

    # ── Analytics ─────────────────────────────────────────────────────────────

    def closed_trades(self) -> list:
        if not self._CLOSED_PATH.exists():
            return []
        try:
            return json.loads(self._CLOSED_PATH.read_text())
        except Exception:
            return []

    def total_closed_pnl(self) -> float:
        return sum(float(t.get("pnl", 0.0)) for t in self.closed_trades())


# ═══════════════════════════════════════════════════════════════════════════════
# PART 3 — SIGNAL LOGIC + AGENT
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class DividendArbitrageConfig:
    min_yield_pct:      float = 0.003   # minimum per-payment yield
    entry_window_start: int   = 7       # earliest entry: J-7
    entry_window_end:   int   = 2       # latest entry: J-2 (never J-1 or J-0)
    exit_days_after:    int   = 1       # target exit at J+1
    target_weight:      float = 0.08   # max portfolio weight per position
    sell_confidence:    float = 0.90   # confidence on all exit signals
    risk_aversion:      float = 0.5    # fraction of price_risk in net_edge
    tp_multiplier:      float = 1.5    # TP: entry + tp_multiplier × div
    sl_multiplier:      float = 2.0    # SL: entry − sl_multiplier × div
    min_net_edge:       float = 0.0    # HOLD if net_edge ≤ this (USD)
    min_confidence:     float = 0.50   # HOLD if confidence < this


class DividendArbitrageAgent(BaseAgent):
    """
    Institutional dividend arbitrage for the Milan Capital arena.

    Generates BUY signals in the J-7 to J-2 window when net_edge > 0 and
    confidence ≥ 0.50. Generates SELL at J+1, TP, or SL.

    Uses meta["div_arb_priority"] = True to signal absolute override in
    selector.select_best() during the active window.

    Compatible with:
    - Regime-aware RiskManager (HOLD in "bear" regime)
    - Kelly net-of-costs framework (cost model matches _TC_ROUND_TRIP = 9 bps)
    - RiskManager.sell_only_mode (no BUY generated in bear regime)
    """

    name = "DividendArbitrageAgent"

    def __init__(self, config: Optional[DividendArbitrageConfig] = None) -> None:
        self.cfg        = config or DividendArbitrageConfig()
        self._collector = DividendDataCollector()
        self._tracker   = DividendPositionTracker()
        self._cache:    Dict[str, Optional[DividendInfo]] = {}
        self._pricing:  Optional[DividendPricingModel]   = None

    # ── Lazy singletons ───────────────────────────────────────────────────────

    def _get_pricing_model(self) -> DividendPricingModel:
        if self._pricing is None:
            rf = self._collector.get_risk_free_rate()
            self._pricing = DividendPricingModel(risk_free_rate=rf)
        return self._pricing

    def _get_div_info(self, ticker: str) -> Optional[DividendInfo]:
        if ticker not in self._cache:
            self._cache[ticker] = self._collector.get_dividend_info(ticker)
        return self._cache[ticker]

    # ── Feature helpers ───────────────────────────────────────────────────────

    @staticmethod
    def _vol_daily(data: Optional[pd.DataFrame]) -> float:
        if data is None or len(data) < 5:
            return _DEFAULT_VOL_DAILY
        try:
            rets = data["Close"].pct_change().dropna()
            return float(rets.tail(20).std()) if len(rets) >= 5 else _DEFAULT_VOL_DAILY
        except Exception:
            return _DEFAULT_VOL_DAILY

    @staticmethod
    def _volume_ratio(data: Optional[pd.DataFrame]) -> float:
        """current_volume / 20-day mean. Returns 0.0 if insufficient data."""
        if data is None or len(data) < 21:
            return 0.0
        try:
            vol = data["Volume"].astype(float)
            mean20 = float(vol.iloc[-21:-1].mean())
            return float(vol.iloc[-1]) / mean20 if mean20 > 0 else 0.0
        except Exception:
            return 0.0

    # ── Confidence scoring ────────────────────────────────────────────────────

    def _compute_confidence(
        self,
        net_edge: float,
        div_info: DividendInfo,
        options_mispricing: bool,
        vol_ratio: float,
    ) -> float:
        conf = 0.0
        conf += 0.30 if net_edge > 0 else 0.0
        conf += min(0.20, div_info.div_regularity * 0.20)
        conf += 0.15 if 3 <= div_info.days_to_ex <= 7 else 0.0
        conf += 0.15 if not div_info.is_special else 0.0
        conf += 0.10 if vol_ratio > 1.0 else 0.0
        conf += 0.10 if options_mispricing else 0.0
        return round(min(conf, 1.0), 4)

    # ── Main signal generator ─────────────────────────────────────────────────

    def generate_signal(
        self,
        state: MarketState,
        portfolio: Dict[str, float],
        regime: Optional[str] = None,
        data: Optional[pd.DataFrame] = None,
    ) -> AgentSignal:
        ticker      = state.symbol
        price       = state.price
        today       = date.today()
        in_position = portfolio.get(ticker, 0.0) > 0

        # Bear regime blocks new entries (SELL exits still allowed)
        if regime == "bear" and not in_position:
            return self._hold(ticker, "Bear regime: dividend arb paused", regime=regime)

        # Check exits for open positions
        if in_position:
            return self._check_exits(ticker, price, today, regime)

        # No open position: clean any stale tracker entry (rejected buy from last run)
        self._tracker.remove_stale(ticker)

        # Fetch dividend info
        div_info = self._get_div_info(ticker)
        if div_info is None:
            return self._hold(ticker, "No dividend data available", regime=regime)

        days = div_info.days_to_ex

        # Entry window check: J-7 to J-2 only
        if days < self.cfg.entry_window_end or days > self.cfg.entry_window_start:
            if days < self.cfg.entry_window_end:
                reason = (
                    f"J-{days}: trop proche (min J-{self.cfg.entry_window_end}, spread trop large)"
                    if days >= 0 else f"Ex-date passée ({div_info.ex_date})"
                )
            else:
                reason = (
                    f"J-{days}: hors fenêtre (max J-{self.cfg.entry_window_start})"
                )
            return self._hold(ticker, reason, regime=regime)

        # Minimum per-payment yield filter
        per_payment_yield = div_info.dividend_amount / price if price > 0 else 0.0
        if per_payment_yield < self.cfg.min_yield_pct:
            return self._hold(
                ticker,
                f"Yield per payment {per_payment_yield:.4%} < min {self.cfg.min_yield_pct:.3%}",
                regime=regime,
            )

        # Net edge calculation
        model    = self._get_pricing_model()
        vol_d    = self._vol_daily(data)
        vol_r    = self._volume_ratio(data)
        notional = 100_000.0 * self.cfg.target_weight  # proxy for a standard portfolio
        shares   = notional / price if price > 0 else 100.0

        edge, breakdown = model.net_edge(div_info, shares, notional, vol_d, self.cfg.risk_aversion)

        # Options parity check (informational — never blocks a trade)
        options_mispricing = model.check_put_call_parity(
            ticker, div_info.ex_date, price, div_info.dividend_amount
        )

        confidence = self._compute_confidence(edge, div_info, options_mispricing, vol_r)

        meta = {
            "regime":              regime,
            "ex_date":             div_info.ex_date.isoformat(),
            "days_to_ex":          days,
            "dividend_amount":     div_info.dividend_amount,
            "yield_annualized":    f"{div_info.yield_annualized:.3%}",
            "net_edge":            round(edge, 4),
            "div_regularity":      round(div_info.div_regularity, 3),
            "is_special_dividend": div_info.is_special,
            "vol_daily":           round(vol_d, 5),
            "volume_ratio":        round(vol_r, 3),
            "options_mispricing":  options_mispricing,
            "breakdown":           breakdown,
        }

        _log.info(
            "%s J-%d div=%.4f edge=%.2f conf=%.2f special=%s",
            ticker, days, div_info.dividend_amount, edge, confidence, div_info.is_special,
        )

        # HOLD: edge not positive
        if edge <= self.cfg.min_net_edge:
            _log.info(
                "%s HOLD — net_edge=%.2f (carry=%.2f risk=%.2f costs=%.2f)",
                ticker,
                breakdown["net_edge"],
                breakdown["carry_cost"],
                breakdown["price_risk_95pct"],
                breakdown["transaction_costs"],
            )
            return AgentSignal(
                agent_name   = self.name,
                symbol       = ticker,
                action       = "HOLD",
                confidence   = min(confidence, 0.45),
                target_weight= 0.0,
                reason       = (
                    f"DivArb HOLD: net_edge={edge:.2f}$ ≤ 0 "
                    f"(div_net={breakdown['net_dividend']:.2f} "
                    f"carry={breakdown['carry_cost']:.2f} "
                    f"risk={breakdown['price_risk_95pct']:.2f} "
                    f"costs={breakdown['transaction_costs']:.2f})"
                ),
                meta=meta,
            )

        # HOLD: confidence below threshold
        if confidence < self.cfg.min_confidence:
            return self._hold(
                ticker,
                f"DivArb: confidence={confidence:.2f} < {self.cfg.min_confidence:.2f}",
                regime=regime,
                meta=meta,
            )

        # BUY: in window with positive edge and sufficient confidence
        meta["div_arb_priority"] = True  # override other agents in selector
        target_exit = (
            div_info.ex_date + timedelta(days=self.cfg.exit_days_after)
        ).isoformat()

        # Optimistic position record (cleaned if buy rejected by RiskManager)
        self._tracker.open_position(DivPosition(
            ticker          = ticker,
            entry_date      = today.isoformat(),
            entry_price     = price,
            ex_date         = div_info.ex_date.isoformat(),
            dividend_amount = div_info.dividend_amount,
            target_exit_date= target_exit,
            shares          = shares,
        ))

        conviction = "fort" if confidence > 0.75 else "modéré"
        return AgentSignal(
            agent_name   = self.name,
            symbol       = ticker,
            action       = "BUY",
            confidence   = confidence,
            target_weight= self.cfg.target_weight,
            reason       = (
                f"DivArb BUY {conviction}: ex={div_info.ex_date} J-{days} "
                f"div={div_info.dividend_amount:.4f}$ "
                f"net_edge={edge:.2f}$ conf={confidence:.2f}"
            ),
            meta=meta,
        )

    # ── Exit checker ──────────────────────────────────────────────────────────

    def _check_exits(
        self,
        ticker: str,
        price: float,
        today: date,
        regime: Optional[str],
    ) -> AgentSignal:
        pos = self._tracker.get_position(ticker)

        if pos is None:
            # Orphaned position — check div_info for timing
            div_info = self._get_div_info(ticker)
            if div_info and div_info.days_to_ex < 0:
                return self._sell(ticker, price, "DivArb EXIT: ex-date passée (orphan)", regime)
            return self._hold(
                ticker, "DivArb: position ouverte, en attente ex-date", regime=regime
            )

        ex_date    = date.fromisoformat(pos.ex_date)
        days_past  = (today - ex_date).days
        div        = pos.dividend_amount

        # J+exit_days_after or later
        if days_past >= self.cfg.exit_days_after:
            self._tracker.close_position(ticker, price, "J+1_EXIT")
            return AgentSignal(
                agent_name   = self.name,
                symbol       = ticker,
                action       = "SELL",
                confidence   = self.cfg.sell_confidence,
                target_weight= 0.0,
                reason       = (
                    f"DivArb EXIT J+{days_past}: dividende capturé "
                    f"ex={pos.ex_date} entry={pos.entry_price:.2f}"
                ),
                meta={"regime": regime, "div_arb_priority": True, "ex_date": pos.ex_date},
            )

        # Take-profit: price > entry + 1.5×div (captured premium before ex-date)
        tp = pos.entry_price + self.cfg.tp_multiplier * div
        if price > tp:
            self._tracker.close_position(ticker, price, "TAKE_PROFIT")
            return AgentSignal(
                agent_name   = self.name,
                symbol       = ticker,
                action       = "SELL",
                confidence   = 0.95,
                target_weight= 0.0,
                reason       = (
                    f"DivArb TAKE PROFIT: {price:.2f} > {tp:.2f} "
                    f"(entry={pos.entry_price:.2f} + 1.5×{div:.4f})"
                ),
                meta={"regime": regime, "div_arb_priority": True},
            )

        # Stop-loss: price < entry - 2×div
        sl = pos.entry_price - self.cfg.sl_multiplier * div
        if price < sl:
            self._tracker.close_position(ticker, price, "STOP_LOSS")
            return AgentSignal(
                agent_name   = self.name,
                symbol       = ticker,
                action       = "SELL",
                confidence   = 0.95,
                target_weight= 0.0,
                reason       = (
                    f"DivArb STOP LOSS: {price:.2f} < {sl:.2f} "
                    f"(entry={pos.entry_price:.2f} - 2×{div:.4f})"
                ),
                meta={"regime": regime, "div_arb_priority": True},
            )

        # Still in valid holding period
        days_to_ex = (ex_date - today).days
        label = f"J+{abs(days_to_ex)}" if days_to_ex < 0 else f"J-{days_to_ex}"
        return self._hold(
            ticker,
            f"DivArb: position ouverte {label}, en attente J+{self.cfg.exit_days_after}",
            regime=regime,
            meta={"regime": regime, "div_arb_priority": True, "entry_price": pos.entry_price},
        )

    # ── Signal factories ──────────────────────────────────────────────────────

    def _sell(self, ticker: str, price: float, reason: str, regime: Optional[str]) -> AgentSignal:
        return AgentSignal(
            agent_name   = self.name,
            symbol       = ticker,
            action       = "SELL",
            confidence   = self.cfg.sell_confidence,
            target_weight= 0.0,
            reason       = reason,
            meta         = {"regime": regime, "div_arb_priority": True},
        )

    def _hold(
        self,
        ticker: str,
        reason: str,
        *,
        regime: Optional[str] = None,
        meta: Optional[dict] = None,
    ) -> AgentSignal:
        return AgentSignal(
            agent_name   = self.name,
            symbol       = ticker,
            action       = "HOLD",
            confidence   = 0.20,
            target_weight= 0.0,
            reason       = reason,
            meta         = meta or {"regime": regime},
        )


# ═══════════════════════════════════════════════════════════════════════════════
# PART 5 — DIVIDEND REPORT (morning briefing)
# ═══════════════════════════════════════════════════════════════════════════════

def generate_dividend_report(watchlist: List[str]) -> str:
    """
    Returns a Telegram-formatted section with the 30-day dividend calendar
    for all tickers in watchlist.

    Sorted by days_to_ex (soonest first). Includes estimated net_edge using
    a $10k notional proxy and default daily vol.
    """
    from concurrent.futures import ThreadPoolExecutor, as_completed

    collector = DividendDataCollector()
    rf        = collector.get_risk_free_rate()
    model     = DividendPricingModel(risk_free_rate=rf)
    entries:  List[DividendInfo] = []

    def _fetch(ticker: str) -> Optional[DividendInfo]:
        try:
            info = collector.get_dividend_info(ticker)
            return info if (info and 0 <= info.days_to_ex <= 30) else None
        except Exception:
            return None

    with ThreadPoolExecutor(max_workers=8) as pool:
        futures = {pool.submit(_fetch, t): t for t in watchlist}
        for fut in as_completed(futures, timeout=30):
            result = fut.result()
            if result is not None:
                entries.append(result)

    if not entries:
        return "📅 DIVIDENDES (30j)\nAucun dividende prévu sur le WATCHLIST."

    entries.sort(key=lambda x: x.days_to_ex)

    lines = ["📅 DIVIDENDES (30j)"]
    for info in entries:
        shares   = _NOTIONAL_REPORT / info.current_price if info.current_price > 0 else 100.0
        edge, _  = model.net_edge(info, shares, _NOTIONAL_REPORT, _DEFAULT_VOL_DAILY)
        opp      = "✅" if edge > 0 else "—"
        special  = " ⚠️SPÉCIAL" if info.is_special else ""
        in_window = "🎯" if 2 <= info.days_to_ex <= 7 else "  "

        lines.append(
            f"{in_window} {info.ticker:<5}  "
            f"ex:{info.ex_date.strftime('%d/%m')}  J-{info.days_to_ex:2d}  "
            f"div=${info.dividend_amount:.3f}  "
            f"{info.yield_annualized:.1%}/an  "
            f"edge=${edge:+.0f}  "
            f"{opp}{special}"
        )

    # Active window summary
    active = [e for e in entries if 2 <= e.days_to_ex <= 7]
    if active:
        best = sorted(active, key=lambda x: x.yield_annualized, reverse=True)
        lines.append(f"\n🎯 Fenêtre active: {', '.join(e.ticker for e in best)}")

    return "\n".join(lines)
