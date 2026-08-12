"""
Tests for CrossSectionalMomentumAgent (Jegadeesh-Titman cross-sectional momentum).

All tests use synthetic price series — no network access.
"""
from __future__ import annotations

from datetime import date
from typing import Dict

import numpy as np
import pandas as pd
import pytest

from src.agents.base import MarketState
from src.agents.momentum import (
    CrossSectionalMomentumAgent,
    MomentumConfig,
    _momentum_score,
)
from tests.conftest import bdate_index


# ── Helpers ───────────────────────────────────────────────────────────────────

def _close(n: int = 320, start: float = 100.0, total_return: float = 0.20) -> pd.Series:
    """Linear ramp producing a known total return over n days."""
    prices = np.linspace(start, start * (1 + total_return), n)
    idx = bdate_index(n)
    return pd.Series(prices, index=idx)


def _df(close: pd.Series) -> pd.DataFrame:
    return pd.DataFrame({"Close": close, "Open": close, "High": close,
                         "Low": close, "Volume": 1_000_000})


def _state(symbol: str = "AAPL", price: float = 150.0) -> MarketState:
    return MarketState(symbol=symbol, price=price, timestamp="2026-07-15T09:30:00Z")


def _universe(*total_returns: float, symbols: list[str] | None = None) -> Dict[str, pd.DataFrame]:
    """
    Build a synthetic universe where ticker i has the given total_return.
    Returns are monotonically ordered — easy to predict ranking.
    """
    syms = symbols or [f"T{i}" for i in range(len(total_returns))]
    return {sym: _df(_close(total_return=r)) for sym, r in zip(syms, total_returns)}


# ── _momentum_score unit tests ────────────────────────────────────────────────

class TestMomentumScore:

    def test_positive_return_gives_positive_score(self):
        close = _close(total_return=0.20)
        sc = _momentum_score(close, periods=[63], skip_days=21)
        assert sc is not None and sc > 0

    def test_negative_return_gives_negative_score(self):
        close = _close(total_return=-0.15)
        sc = _momentum_score(close, periods=[63], skip_days=21)
        assert sc is not None and sc < 0

    def test_insufficient_history_returns_none(self):
        # Only 50 days — not enough for 63+21
        close = pd.Series(np.linspace(100, 110, 50))
        sc = _momentum_score(close, periods=[63], skip_days=21)
        assert sc is None

    def test_multiple_periods_averaged(self):
        close = _close(total_return=0.20, n=320)
        sc_3m  = _momentum_score(close, [63],        skip_days=21)
        sc_avg = _momentum_score(close, [63, 126, 252], skip_days=21)
        # Can't predict exact value, but both are positive
        assert sc_avg is not None and sc_avg > 0
        assert sc_3m  is not None and sc_3m  > 0

    def test_score_is_float(self):
        close = _close()
        sc = _momentum_score(close, [63, 126], skip_days=21)
        assert isinstance(sc, float)


# ── Ranking tests ─────────────────────────────────────────────────────────────

class TestMomentumRanking:

    def test_highest_return_ranked_first(self):
        """Ticker with +40% should rank above ticker with +10%."""
        uni = _universe(0.40, 0.25, 0.10, -0.05, symbols=["A","B","C","D"])
        agent = CrossSectionalMomentumAgent(universe=uni)
        assert agent._rankings["A"]["rank"] == 1
        assert agent._rankings["D"]["rank"] == 4

    def test_top_quartile_gets_buy(self):
        """Top 25% (1 of 4 tickers) → BUY."""
        uni = _universe(0.40, 0.20, 0.10, -0.05, symbols=["A","B","C","D"])
        agent = CrossSectionalMomentumAgent(universe=uni)
        sig = agent.generate_signal(_state("A"), {})
        assert sig.action == "BUY"

    def test_non_top_gets_hold(self):
        """Tickers outside top quartile → HOLD."""
        uni = _universe(0.40, 0.20, 0.10, -0.05, symbols=["A","B","C","D"])
        agent = CrossSectionalMomentumAgent(universe=uni)
        for sym in ["B", "C", "D"]:
            sig = agent.generate_signal(_state(sym), {})
            assert sig.action == "HOLD", f"{sym} should be HOLD"

    def test_top_quartile_size_rounds_up(self):
        """ceil(0.25 × 5) = 2 — two tickers should get BUY with 5-ticker universe."""
        uni = _universe(0.50, 0.30, 0.15, 0.05, -0.10,
                        symbols=["A","B","C","D","E"])
        agent = CrossSectionalMomentumAgent(universe=uni)
        buys = [s for s in ["A","B","C","D","E"]
                if agent.generate_signal(_state(s), {}).action == "BUY"]
        assert len(buys) == 2
        assert "A" in buys
        assert "B" in buys

    def test_n_populated(self):
        uni = _universe(0.20, 0.10, 0.05, -0.05)
        agent = CrossSectionalMomentumAgent(universe=uni)
        for info in agent._rankings.values():
            assert info["n"] == 4

    def test_pct_rank_monotone(self):
        """pct_rank increases from rank 1 to rank n."""
        uni = _universe(0.40, 0.25, 0.10, -0.05, symbols=["A","B","C","D"])
        agent = CrossSectionalMomentumAgent(universe=uni)
        pcts = [agent._rankings[s]["pct_rank"] for s in ["A","B","C","D"]]
        assert pcts == sorted(pcts)

    def test_score_positive_for_ramp_up(self):
        close = _close(total_return=0.30)
        sc = _momentum_score(close, periods=[63, 126, 252], skip_days=21)
        assert sc is not None and sc > 0


# ── Signal output tests ───────────────────────────────────────────────────────

class TestMomentumSignalOutput:

    def test_buy_confidence_in_range(self):
        uni = _universe(0.40, 0.20, 0.10, -0.05, symbols=["A","B","C","D"])
        agent = CrossSectionalMomentumAgent(universe=uni)
        sig = agent.generate_signal(_state("A"), {})
        assert 0.0 < sig.confidence <= 0.90

    def test_buy_target_weight(self):
        uni = _universe(0.40, 0.20, 0.10, -0.05, symbols=["A","B","C","D"])
        agent = CrossSectionalMomentumAgent(universe=uni)
        sig = agent.generate_signal(_state("A"), {})
        assert sig.target_weight == pytest.approx(0.07)

    def test_hold_target_weight_zero(self):
        uni = _universe(0.40, 0.20, 0.10, -0.05, symbols=["A","B","C","D"])
        agent = CrossSectionalMomentumAgent(universe=uni)
        sig = agent.generate_signal(_state("D"), {})
        assert sig.target_weight == 0.0
        assert sig.confidence   == 0.0

    def test_meta_populated(self):
        uni = _universe(0.40, 0.20, 0.10, -0.05, symbols=["A","B","C","D"])
        agent = CrossSectionalMomentumAgent(universe=uni)
        sig = agent.generate_signal(_state("A"), {})
        assert sig.meta["rank"] == 1
        assert sig.meta["n"]    == 4
        assert sig.meta["is_top"] is True

    def test_reason_contains_rank(self):
        uni = _universe(0.40, 0.20, 0.10, -0.05, symbols=["A","B","C","D"])
        agent = CrossSectionalMomentumAgent(universe=uni)
        sig = agent.generate_signal(_state("A"), {})
        assert "rang 1" in sig.reason
        assert "4" in sig.reason  # total n

    def test_agent_name(self):
        agent = CrossSectionalMomentumAgent()
        assert agent.name == "CrossSectionalMomentumAgent"

    def test_confidence_rank1_gt_last_top(self):
        """Rank-1 ticker should have higher confidence than last top-quartile ticker."""
        # 8 tickers, top 25% = top 2
        returns = [0.50, 0.35, 0.20, 0.15, 0.10, 0.05, -0.05, -0.15]
        syms = [f"T{i}" for i in range(8)]
        uni = _universe(*returns, symbols=syms)
        agent = CrossSectionalMomentumAgent(universe=uni)
        sig1 = agent.generate_signal(_state("T0"), {})
        sig2 = agent.generate_signal(_state("T1"), {})
        assert sig1.confidence >= sig2.confidence


# ── Edge cases ────────────────────────────────────────────────────────────────

class TestMomentumEdgeCases:

    def test_unknown_ticker_returns_hold(self):
        uni = _universe(0.30, 0.10, 0.05, -0.05)
        agent = CrossSectionalMomentumAgent(universe=uni)
        sig = agent.generate_signal(_state("UNKNOWN"), {})
        assert sig.action == "HOLD"
        assert sig.meta["rank"] is None

    def test_empty_universe_no_rankings(self):
        agent = CrossSectionalMomentumAgent(universe={})
        assert agent._rankings == {}
        sig = agent.generate_signal(_state("AAPL"), {})
        assert sig.action == "HOLD"

    def test_too_small_universe_no_rankings(self):
        """min_universe=4: 3 tickers → no ranking."""
        uni = _universe(0.30, 0.10, -0.05)
        cfg = MomentumConfig(min_universe=4)
        agent = CrossSectionalMomentumAgent(config=cfg, universe=uni)
        assert agent._rankings == {}

    def test_short_history_ticker_excluded(self):
        """Ticker with < min_history rows is excluded; others still ranked."""
        uni = _universe(0.40, 0.20, 0.10, -0.05, 0.01, symbols=["A","B","C","D","E"])
        # Replace C with a very short series (50 days)
        short = pd.Series(np.linspace(100, 105, 50),
                          index=bdate_index(50))
        uni["C"] = _df(short)
        agent = CrossSectionalMomentumAgent(universe=uni)
        assert "C" not in agent._rankings
        # A, B, D, E (4 tickers) still have sufficient history
        assert "A" in agent._rankings
        assert "D" in agent._rankings

    def test_missing_close_column_excluded(self):
        uni = _universe(0.30, 0.10, 0.05, -0.05, symbols=["A","B","C","D"])
        # Remove Close from B
        uni["B"] = pd.DataFrame({"Open": uni["B"]["Close"]})
        agent = CrossSectionalMomentumAgent(universe=uni)
        assert "B" not in agent._rankings

    def test_set_universe_updates_rankings(self):
        """Calling set_universe again replaces old rankings."""
        uni1 = _universe(0.40, 0.20, 0.10, -0.05, symbols=["A","B","C","D"])
        uni2 = _universe(-0.40, 0.20, 0.10, 0.50, symbols=["A","B","C","D"])
        agent = CrossSectionalMomentumAgent(universe=uni1)
        assert agent._rankings["A"]["rank"] == 1  # A best in uni1

        agent.set_universe(uni2)
        assert agent._rankings["D"]["rank"] == 1  # D best in uni2
        assert agent._rankings["A"]["rank"] == 4  # A worst in uni2

    def test_no_universe_set_returns_hold(self):
        """Agent with no universe set always returns HOLD."""
        agent = CrossSectionalMomentumAgent()
        sig = agent.generate_signal(_state("AAPL"), {})
        assert sig.action == "HOLD"

    def test_custom_top_pct(self):
        """top_pct=0.50 → top 2 of 4 get BUY."""
        cfg = MomentumConfig(top_pct=0.50)
        uni = _universe(0.40, 0.20, 0.10, -0.05, symbols=["A","B","C","D"])
        agent = CrossSectionalMomentumAgent(config=cfg, universe=uni)
        buys = [s for s in ["A","B","C","D"]
                if agent.generate_signal(_state(s), {}).action == "BUY"]
        assert len(buys) == 2

    def test_custom_target_weight(self):
        cfg = MomentumConfig(target_weight=0.10, top_pct=0.25)
        uni = _universe(0.40, 0.20, 0.10, -0.05, symbols=["A","B","C","D"])
        agent = CrossSectionalMomentumAgent(config=cfg, universe=uni)
        sig = agent.generate_signal(_state("A"), {})
        assert sig.target_weight == pytest.approx(0.10)
