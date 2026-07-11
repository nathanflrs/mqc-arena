"""
Tests for DividendArbitrageAgent — 20 cases covering all spec requirements.

No live yfinance calls: all external I/O is monkeypatched or bypassed.
Tracker paths are redirected to tmp_path for isolation.

Design choice: cfg.risk_aversion=0.0 isolates BUY/HOLD edge tests from the
price-risk term (which uses vol_daily × sqrt(days) and can dominate for short
holding periods). Tests specifically covering price risk use explicit params.
"""
from __future__ import annotations

import json
from datetime import date, timedelta
from pathlib import Path
from typing import Optional

import pandas as pd
import pytest

from src.agents.base import AgentSignal, MarketState
from src.agents.dividend_arbitrage_agent import (
    DividendArbitrageAgent,
    DividendArbitrageConfig,
    DividendDataCollector,
    DividendInfo,
    DividendPositionTracker,
    DivPosition,
    DividendPricingModel,
    generate_dividend_report,
)
from src.arena.selector import select_best


# ── Factories ─────────────────────────────────────────────────────────────────

def _make_div_info(
    ticker: str = "AAPL",
    days_to_ex: int = 5,
    dividend_amount: float = 1.00,
    price: float = 100.0,
    regularity: float = 0.95,
    is_special: bool = False,
) -> DividendInfo:
    today   = date.today()
    ex_date = today + timedelta(days=days_to_ex)
    yield_ann = (dividend_amount / price) * (365.0 / max(days_to_ex, 1))
    return DividendInfo(
        ticker          = ticker,
        ex_date         = ex_date,
        dividend_amount = dividend_amount,
        payment_date    = None,
        record_date     = None,
        current_price   = price,
        days_to_ex      = days_to_ex,
        yield_annualized= yield_ann,
        div_regularity  = regularity,
        is_special      = is_special,
    )


def _make_state(ticker: str = "AAPL", price: float = 100.0) -> MarketState:
    return MarketState(symbol=ticker, price=price, timestamp="2026-06-24T09:00:00")


def _setup_agent(
    monkeypatch,
    tmp_path: Path,
    div_info: Optional[DividendInfo] = None,
    rf: float = 0.0,
    options_mispricing: bool = False,
    config: Optional[DividendArbitrageConfig] = None,
) -> DividendArbitrageAgent:
    """
    Create a fully mocked agent.
    - Tracker files redirected to tmp_path
    - Pricing model pre-initialized with given rf (no yfinance ^IRX call)
    - div_info injected into cache (no yfinance call)
    - check_put_call_parity mocked to fixed return value
    """
    monkeypatch.setattr(DividendPositionTracker, "_POSITIONS_PATH", tmp_path / "pos.json")
    monkeypatch.setattr(DividendPositionTracker, "_CLOSED_PATH",    tmp_path / "closed.json")

    agent          = DividendArbitrageAgent(config=config)
    agent._pricing = DividendPricingModel(risk_free_rate=rf)

    if div_info is not None:
        monkeypatch.setattr(agent, "_get_div_info", lambda t: div_info)

    monkeypatch.setattr(agent._pricing, "check_put_call_parity",
                        lambda *a, **k: options_mispricing)
    return agent


# ═══════════════════════════════════════════════════════════════════════════════
# 1. Net edge positive → BUY
# ═══════════════════════════════════════════════════════════════════════════════

def test_net_edge_positive(tmp_path, monkeypatch):
    # risk_aversion=0 removes price-risk penalty so net_div alone determines edge
    cfg      = DividendArbitrageConfig(risk_aversion=0.0)
    div_info = _make_div_info(dividend_amount=2.0, price=100.0, days_to_ex=5, regularity=0.95)
    agent    = _setup_agent(monkeypatch, tmp_path, div_info=div_info, rf=0.0, config=cfg)

    signal = agent.generate_signal(_make_state(), {}, regime="bull")
    assert signal.action == "BUY"
    assert signal.meta["net_edge"] > 0


# ═══════════════════════════════════════════════════════════════════════════════
# 2. Net edge negative → HOLD (detailed reason in log/reason)
# ═══════════════════════════════════════════════════════════════════════════════

def test_net_edge_negative(tmp_path, monkeypatch):
    # min_yield_pct=0 bypasses yield filter; high rf makes carry > net_div
    cfg      = DividendArbitrageConfig(min_yield_pct=0.0, risk_aversion=0.0)
    div_info = _make_div_info(dividend_amount=0.001, price=100.0, days_to_ex=5)
    agent    = _setup_agent(monkeypatch, tmp_path, div_info=div_info, rf=0.50, config=cfg)
    # With rf=0.50: carry = 8000*0.5*(5/365)=54.8, net_div=0.068 → edge << 0

    signal = agent.generate_signal(_make_state(), {}, regime="bull")
    assert signal.action == "HOLD"
    assert "net_edge" in signal.reason.lower()


# ═══════════════════════════════════════════════════════════════════════════════
# 3. Entry window: BUY only between J-7 and J-2
# ═══════════════════════════════════════════════════════════════════════════════

@pytest.mark.parametrize("days,expected", [
    (8,  "HOLD"),   # too early
    (7,  "BUY"),    # boundary in
    (5,  "BUY"),    # optimal window
    (2,  "BUY"),    # boundary in
    (1,  "HOLD"),   # J-1: spread too large
    (0,  "HOLD"),   # ex-date itself
])
def test_entry_window(days, expected, tmp_path, monkeypatch):
    # risk_aversion=0 ensures edge > 0 when in window (isolates window logic)
    cfg      = DividendArbitrageConfig(risk_aversion=0.0)
    div_info = _make_div_info(dividend_amount=2.0, price=100.0, days_to_ex=days, regularity=0.95)
    agent    = _setup_agent(monkeypatch, tmp_path, div_info=div_info, rf=0.0, config=cfg)

    signal = agent.generate_signal(_make_state(), {}, regime="bull")
    assert signal.action == expected, (
        f"days_to_ex={days}: expected {expected}, got {signal.action} | {signal.reason}"
    )


# ═══════════════════════════════════════════════════════════════════════════════
# 4. Auto SELL at J+1 (past ex-date, dividend captured)
# ═══════════════════════════════════════════════════════════════════════════════

def test_exit_j_plus_1(tmp_path, monkeypatch):
    monkeypatch.setattr(DividendPositionTracker, "_POSITIONS_PATH", tmp_path / "pos.json")
    monkeypatch.setattr(DividendPositionTracker, "_CLOSED_PATH",    tmp_path / "closed.json")

    agent     = DividendArbitrageAgent()
    yesterday = (date.today() - timedelta(days=1)).isoformat()

    agent._tracker.open_position(DivPosition(
        ticker          = "AAPL",
        entry_date      = (date.today() - timedelta(days=3)).isoformat(),
        entry_price     = 150.0,
        ex_date         = yesterday,
        dividend_amount = 0.25,
        target_exit_date= date.today().isoformat(),
        shares          = 53.0,
    ))

    # Portfolio shows AAPL position; ex-date was yesterday → trigger J+1 exit
    signal = agent.generate_signal(_make_state(price=150.10), {"AAPL": 100.0})
    assert signal.action == "SELL"
    assert "capturé" in signal.reason or "J+1" in signal.reason


# ═══════════════════════════════════════════════════════════════════════════════
# 5. Stop-loss: price < entry − 2×dividend_amount
# ═══════════════════════════════════════════════════════════════════════════════

def test_stop_loss(tmp_path, monkeypatch):
    monkeypatch.setattr(DividendPositionTracker, "_POSITIONS_PATH", tmp_path / "pos.json")
    monkeypatch.setattr(DividendPositionTracker, "_CLOSED_PATH",    tmp_path / "closed.json")

    agent   = DividendArbitrageAgent()
    ex_date = (date.today() + timedelta(days=2)).isoformat()

    agent._tracker.open_position(DivPosition(
        ticker="AAPL", entry_date=date.today().isoformat(), entry_price=150.0,
        ex_date=ex_date, dividend_amount=0.25,
        target_exit_date=(date.today() + timedelta(days=3)).isoformat(), shares=53.0,
    ))

    # SL trigger: 150 - 2×0.25 = 149.50; current = 149.00
    signal = agent.generate_signal(_make_state(price=149.00), {"AAPL": 100.0})
    assert signal.action == "SELL"
    assert "STOP LOSS" in signal.reason


# ═══════════════════════════════════════════════════════════════════════════════
# 6. Take-profit: price > entry + 1.5×dividend_amount
# ═══════════════════════════════════════════════════════════════════════════════

def test_take_profit(tmp_path, monkeypatch):
    monkeypatch.setattr(DividendPositionTracker, "_POSITIONS_PATH", tmp_path / "pos.json")
    monkeypatch.setattr(DividendPositionTracker, "_CLOSED_PATH",    tmp_path / "closed.json")

    agent   = DividendArbitrageAgent()
    ex_date = (date.today() + timedelta(days=2)).isoformat()

    agent._tracker.open_position(DivPosition(
        ticker="AAPL", entry_date=date.today().isoformat(), entry_price=150.0,
        ex_date=ex_date, dividend_amount=0.25,
        target_exit_date=(date.today() + timedelta(days=3)).isoformat(), shares=53.0,
    ))

    # TP trigger: 150 + 1.5×0.25 = 150.375; current = 150.50
    signal = agent.generate_signal(_make_state(price=150.50), {"AAPL": 100.0})
    assert signal.action == "SELL"
    assert "TAKE PROFIT" in signal.reason


# ═══════════════════════════════════════════════════════════════════════════════
# 7. Special dividend flag: amount > 2× historical mean
# ═══════════════════════════════════════════════════════════════════════════════

def test_special_dividend_flag():
    # Mean of 12 quarterly dividends = 0.25; current = 0.60 → 2.4× → special
    # Plain RangeIndex: tz comparison fails → fallback uses all 12 elements (tested path)
    hist = pd.Series([0.25] * 12, dtype=float)

    _, is_special = DividendDataCollector._compute_regularity(hist, current_amount=0.60)
    assert is_special is True

    # Just below 2× threshold (0.49 < 2×0.25 = 0.50) → not special
    _, not_special = DividendDataCollector._compute_regularity(hist, current_amount=0.49)
    assert not_special is False


# ═══════════════════════════════════════════════════════════════════════════════
# 8. Carry cost formula: position_size × rf × (days / 365)
# ═══════════════════════════════════════════════════════════════════════════════

def test_carry_cost_calculation():
    model    = DividendPricingModel(risk_free_rate=0.05)
    expected = 10_000.0 * 0.05 * (5.0 / 365.0)
    assert abs(model.carry_cost(10_000.0, 5) - expected) < 0.001

    # days=0 is treated as 1 (no division by zero)
    assert model.carry_cost(10_000.0, 0) > 0


# ═══════════════════════════════════════════════════════════════════════════════
# 9. Tax withholding: 15% deducted from gross dividend
# ═══════════════════════════════════════════════════════════════════════════════

def test_tax_withholding():
    div_info = _make_div_info(dividend_amount=1.00, price=100.0, days_to_ex=5)
    model    = DividendPricingModel(risk_free_rate=0.0)
    # 100 shares × $1.00 = $100 gross
    _, breakdown = model.net_edge(div_info, shares=100.0, position_size=10_000.0,
                                  vol_daily=0.0, risk_aversion=0.0)

    assert breakdown["gross_dividend"]  == 100.0
    assert abs(breakdown["tax_withholding"] - 15.0) < 0.001
    assert abs(breakdown["net_dividend"]    - 85.0) < 0.001


# ═══════════════════════════════════════════════════════════════════════════════
# 10. Confidence scoring: all 6 components verified individually
# ═══════════════════════════════════════════════════════════════════════════════

def test_confidence_scoring():
    agent = DividendArbitrageAgent()

    # All 6 components active: 0.30+0.20+0.15+0.15+0.10+0.10 = 1.00
    div_full = _make_div_info(days_to_ex=5, regularity=1.0, is_special=False)
    conf_full = agent._compute_confidence(
        net_edge=1.0, div_info=div_full, options_mispricing=True, vol_ratio=1.5
    )
    assert abs(conf_full - 1.00) < 0.001

    # C1 off (net_edge < 0): 0+0.20+0.15+0.15+0.10+0.10 = 0.70
    conf_no_edge = agent._compute_confidence(
        net_edge=-1.0, div_info=div_full, options_mispricing=True, vol_ratio=1.5
    )
    assert abs(conf_no_edge - 0.70) < 0.001

    # C4 off (special dividend): 0.30+0.20+0.15+0+0.10+0.10 = 0.85
    div_special = _make_div_info(days_to_ex=5, regularity=1.0, is_special=True)
    conf_special = agent._compute_confidence(
        net_edge=1.0, div_info=div_special, options_mispricing=True, vol_ratio=1.5
    )
    assert abs(conf_special - 0.85) < 0.001

    # C3 off (days_to_ex=2, outside [3,7]): 0.30+0.20+0+0.15+0+0 = 0.65
    div_day2 = _make_div_info(days_to_ex=2, regularity=1.0, is_special=False)
    conf_day2 = agent._compute_confidence(
        net_edge=1.0, div_info=div_day2, options_mispricing=False, vol_ratio=0.5
    )
    assert abs(conf_day2 - 0.65) < 0.001


# ═══════════════════════════════════════════════════════════════════════════════
# 11. Position persistence: survives a process restart
# ═══════════════════════════════════════════════════════════════════════════════

def test_position_persistence(tmp_path, monkeypatch):
    monkeypatch.setattr(DividendPositionTracker, "_POSITIONS_PATH", tmp_path / "pos.json")
    monkeypatch.setattr(DividendPositionTracker, "_CLOSED_PATH",    tmp_path / "closed.json")

    # First instance writes position
    tracker1 = DividendPositionTracker()
    tracker1.open_position(DivPosition(
        ticker="AAPL", entry_date="2026-06-20", entry_price=150.0,
        ex_date="2026-06-25", dividend_amount=0.25,
        target_exit_date="2026-06-26", shares=53.0,
    ))

    # Second instance (simulates restart) reads the same file
    tracker2 = DividendPositionTracker()
    pos = tracker2.get_position("AAPL")
    assert pos is not None
    assert pos.entry_price     == 150.0
    assert pos.dividend_amount == 0.25


# ═══════════════════════════════════════════════════════════════════════════════
# 12. No-dividend ticker (GLD, QQQ): graceful HOLD, no exception
# ═══════════════════════════════════════════════════════════════════════════════

def test_no_dividend_ticker(tmp_path, monkeypatch):
    agent = _setup_agent(monkeypatch, tmp_path, div_info=None)   # None = no dividend data

    signal = agent.generate_signal(_make_state(ticker="GLD"), {}, regime="bull")
    assert signal.action == "HOLD"
    assert any(kw in signal.reason.lower() for kw in ("no dividend", "data", "available"))


# ═══════════════════════════════════════════════════════════════════════════════
# 13. Telegram report format: required sections present
# ═══════════════════════════════════════════════════════════════════════════════

def test_telegram_report_format(monkeypatch):
    aapl_info = _make_div_info(ticker="AAPL", days_to_ex=5, dividend_amount=0.25, price=150.0)

    monkeypatch.setattr(DividendDataCollector, "get_dividend_info",
                        lambda self, t: aapl_info if t == "AAPL" else None)
    monkeypatch.setattr(DividendDataCollector, "get_risk_free_rate", lambda self: 0.05)

    report = generate_dividend_report(["AAPL", "GLD", "SPY"])

    assert "📅 DIVIDENDES" in report
    assert "AAPL" in report
    assert "div=$" in report
    assert "ex:" in report


# ═══════════════════════════════════════════════════════════════════════════════
# 14. Options parity flag: mispricing detected → meta["options_mispricing"] True
# ═══════════════════════════════════════════════════════════════════════════════

def test_options_parity_flag(tmp_path, monkeypatch):
    cfg      = DividendArbitrageConfig(min_yield_pct=0.0)
    div_info = _make_div_info(dividend_amount=0.001, price=100.0, days_to_ex=5)
    # Force a HOLD via negative edge; options flag should still appear in meta
    agent    = _setup_agent(monkeypatch, tmp_path, div_info=div_info, rf=0.50,
                            options_mispricing=True, config=cfg)

    signal = agent.generate_signal(_make_state(), {}, regime="bull")

    # options_mispricing is included in meta regardless of HOLD/BUY outcome
    assert signal.meta.get("options_mispricing") is True


# ═══════════════════════════════════════════════════════════════════════════════
# 15. Absolute override in select_best via div_arb_priority flag
# ═══════════════════════════════════════════════════════════════════════════════

def test_override_priority():
    div_signal = AgentSignal(
        agent_name   = "DividendArbitrageAgent",
        symbol       = "AAPL",
        action       = "BUY",
        confidence   = 0.70,
        target_weight= 0.08,
        reason       = "DivArb BUY fort",
        meta         = {"div_arb_priority": True},
    )
    # Nominally stronger signal from another agent (higher score = 0.99×0.20)
    strong = AgentSignal(
        agent_name   = "BuffettAgent",
        symbol       = "AAPL",
        action       = "BUY",
        confidence   = 0.99,
        target_weight= 0.20,
        reason       = "Strong fundamental buy",
        meta         = {},
    )
    result = select_best([strong, div_signal])
    assert result is not None
    assert result.agent_name == "DividendArbitrageAgent"


# ═══════════════════════════════════════════════════════════════════════════════
# 16. Regular dividend → regularity score close to 1.0
# ═══════════════════════════════════════════════════════════════════════════════

def test_regular_dividend_score():
    # Perfectly regular: CV=0 → regularity=1.0 (RangeIndex → tz fallback → all elements used)
    hist = pd.Series([0.25] * 12, dtype=float)
    regularity, _ = DividendDataCollector._compute_regularity(hist, current_amount=0.25)
    assert abs(regularity - 1.0) < 0.001

    # Irregular: high variance (CV ≈ 0.84) → regularity ≈ 0.16 < 0.5
    hist_irreg = pd.Series([0.10, 1.00, 0.05, 0.80, 0.15, 0.90], dtype=float)
    reg_irreg, _ = DividendDataCollector._compute_regularity(hist_irreg, current_amount=0.25)
    assert reg_irreg < 0.5


# ═══════════════════════════════════════════════════════════════════════════════
# 17. Volume ratio: below-average volume → no C5 confidence bonus
# ═══════════════════════════════════════════════════════════════════════════════

def test_volume_ratio():
    # Last day: half the average → ratio < 1.0 → no C5 bonus
    low = pd.DataFrame({
        "Close":  [100.0] * 25,
        "Volume": [1_000_000] * 24 + [500_000],
    })
    assert DividendArbitrageAgent._volume_ratio(low) < 1.0

    # Last day: 2× average → ratio > 1.0 → C5 bonus earned
    high = pd.DataFrame({
        "Close":  [100.0] * 25,
        "Volume": [1_000_000] * 24 + [2_000_000],
    })
    assert DividendArbitrageAgent._volume_ratio(high) > 1.0

    # Edge cases: None and empty DataFrame
    assert DividendArbitrageAgent._volume_ratio(None) == 0.0
    assert DividendArbitrageAgent._volume_ratio(pd.DataFrame()) == 0.0


# ═══════════════════════════════════════════════════════════════════════════════
# 18. Watchlist coverage: report runs for all tickers without exception
# ═══════════════════════════════════════════════════════════════════════════════

def test_watchlist_coverage(monkeypatch):
    from src.config import WATCHLIST

    # All tickers return None (no dividends) — report must still be well-formed
    monkeypatch.setattr(DividendDataCollector, "get_dividend_info", lambda self, t: None)
    monkeypatch.setattr(DividendDataCollector, "get_risk_free_rate", lambda self: 0.05)

    result = generate_dividend_report(WATCHLIST)
    assert isinstance(result, str)
    assert "DIVIDENDES" in result
    assert len(WATCHLIST) == 14  # sanity-check watchlist size matches spec


# ═══════════════════════════════════════════════════════════════════════════════
# 19. Bear regime (circuit breaker proxy): no BUY regardless of edge
# ═══════════════════════════════════════════════════════════════════════════════

def test_circuit_breaker_respect(tmp_path, monkeypatch):
    cfg      = DividendArbitrageConfig(risk_aversion=0.0)
    div_info = _make_div_info(dividend_amount=2.0, price=100.0, days_to_ex=5, regularity=0.95)
    agent    = _setup_agent(monkeypatch, tmp_path, div_info=div_info, rf=0.0, config=cfg)

    # No position, bear regime → agent must return HOLD (mirrors sell_only_mode behavior)
    signal = agent.generate_signal(_make_state(), {}, regime="bear")
    assert signal.action == "HOLD"
    assert "bear" in signal.reason.lower()


# ═══════════════════════════════════════════════════════════════════════════════
# 20. Full round-trip: open → J+1 exit with positive realized P&L
# ═══════════════════════════════════════════════════════════════════════════════

def test_full_round_trip(tmp_path, monkeypatch):
    monkeypatch.setattr(DividendPositionTracker, "_POSITIONS_PATH", tmp_path / "pos.json")
    monkeypatch.setattr(DividendPositionTracker, "_CLOSED_PATH",    tmp_path / "closed.json")

    tracker     = DividendPositionTracker()
    entry_price = 150.0
    shares      = 53.33
    yesterday   = (date.today() - timedelta(days=1)).isoformat()

    # ── Phase 1: simulate a buy 3 days ago ────────────────────────────────────
    tracker.open_position(DivPosition(
        ticker          = "AAPL",
        entry_date      = (date.today() - timedelta(days=3)).isoformat(),
        entry_price     = entry_price,
        ex_date         = yesterday,
        dividend_amount = 0.25,
        target_exit_date= date.today().isoformat(),
        shares          = shares,
    ))
    assert tracker.get_position("AAPL") is not None

    # ── Phase 2: J+1 exit, price slightly above entry ─────────────────────────
    # P&L = price_change + net_div - costs
    # = (150.10-150)*53.33 + 0.25*53.33*0.85 - round_trip_costs
    # = 5.33 + 11.33 - ~7.21 ≈ +9.45 > 0
    exit_price = 150.10
    pnl        = tracker.close_position("AAPL", exit_price, "J+1_EXIT")

    assert pnl is not None
    assert pnl > 0, f"Expected positive P&L, got {pnl:.2f}"
    assert tracker.get_position("AAPL") is None

    # ── Phase 3: verify closed log ────────────────────────────────────────────
    trades = tracker.closed_trades()
    assert len(trades) == 1
    assert trades[0]["ticker"] == "AAPL"
    assert trades[0]["reason"] == "J+1_EXIT"
    assert trades[0]["pnl"]    == round(pnl, 2)
    assert tracker.total_closed_pnl() > 0
