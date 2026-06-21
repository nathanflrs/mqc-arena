# tests/test_earnings_filter.py
"""
Tests for EarningsFilter.

All yfinance calls are mocked — no network required.
"""
from __future__ import annotations

from datetime import date, timedelta
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest

from src.risk.earnings_filter import EarningsFilter


# ── Helpers ───────────────────────────────────────────────────────────────────

def _make_earnings_df(dates: list[date]) -> pd.DataFrame:
    """Mimics the DataFrame returned by yf.Ticker.get_earnings_dates()."""
    idx = pd.DatetimeIndex([pd.Timestamp(d) for d in dates])
    return pd.DataFrame({"EPS Estimate": [None] * len(dates)}, index=idx)


def _today() -> date:
    return pd.Timestamp.today().normalize().date()


# ── EarningsFilter.should_block_buy ──────────────────────────────────────────

class TestShouldBlockBuy:

    def _filter_with_dates(self, dates: list[date]) -> EarningsFilter:
        ef = EarningsFilter(buffer_days=3)
        ef._cache["AAPL"] = pd.DatetimeIndex([pd.Timestamp(d) for d in dates])
        return ef

    def test_earnings_today_is_blocked(self):
        ef = self._filter_with_dates([_today()])
        blocked, reason = ef.should_block_buy("AAPL")
        assert blocked
        assert "today" in reason

    def test_earnings_in_1_bday_is_blocked(self):
        target = pd.bdate_range(_today(), periods=2)[-1].date()
        ef = self._filter_with_dates([target])
        blocked, reason = ef.should_block_buy("AAPL")
        assert blocked
        assert "1 bday" in reason

    def test_earnings_in_3_bdays_is_blocked(self):
        target = pd.bdate_range(_today(), periods=4)[-1].date()
        ef = self._filter_with_dates([target])
        blocked, reason = ef.should_block_buy("AAPL")
        assert blocked

    def test_earnings_in_4_bdays_is_not_blocked(self):
        target = pd.bdate_range(_today(), periods=5)[-1].date()
        ef = self._filter_with_dates([target])
        blocked, _ = ef.should_block_buy("AAPL")
        assert not blocked

    def test_earnings_in_the_past_is_not_blocked(self):
        # Use BDay offset to guarantee a genuine past business day regardless of
        # weekends — timedelta(days=1) from a Monday would give Sunday (bday_count=0)
        past_bday = (pd.Timestamp.today() - pd.offsets.BDay(4)).normalize().date()
        ef = self._filter_with_dates([past_bday])
        blocked, _ = ef.should_block_buy("AAPL")
        assert not blocked

    def test_empty_dates_never_blocks(self):
        ef = self._filter_with_dates([])
        blocked, _ = ef.should_block_buy("AAPL")
        assert not blocked

    def test_reason_string_contains_date(self):
        target = _today()
        ef = self._filter_with_dates([target])
        blocked, reason = ef.should_block_buy("AAPL")
        assert blocked
        assert str(target) in reason

    def test_multiple_dates_blocks_on_nearest(self):
        far = pd.bdate_range(_today(), periods=20)[-1].date()
        near = pd.bdate_range(_today(), periods=2)[-1].date()
        ef = self._filter_with_dates([far, near])
        blocked, _ = ef.should_block_buy("AAPL")
        assert blocked


# ── EarningsFilter.prefetch ───────────────────────────────────────────────────

class TestPrefetch:

    def test_prefetch_loads_dates_into_cache(self):
        upcoming = _today() + timedelta(days=1)
        mock_df = _make_earnings_df([upcoming])

        with patch("yfinance.Ticker") as MockTicker:
            MockTicker.return_value.get_earnings_dates.return_value = mock_df
            ef = EarningsFilter(buffer_days=3)
            ef.prefetch(["AAPL", "MSFT"])

        assert "AAPL" in ef._cache
        assert "MSFT" in ef._cache

    def test_prefetch_blocks_buy_for_symbol_with_near_earnings(self):
        upcoming = pd.bdate_range(pd.Timestamp.today(), periods=2)[-1].date()
        mock_df = _make_earnings_df([upcoming])

        with patch("yfinance.Ticker") as MockTicker:
            MockTicker.return_value.get_earnings_dates.return_value = mock_df
            ef = EarningsFilter(buffer_days=3)
            ef.prefetch(["AAPL"])

        blocked, _ = ef.should_block_buy("AAPL")
        assert blocked


# ── EarningsFilter — graceful failure ────────────────────────────────────────

class TestGracefulFailure:

    def test_yfinance_exception_does_not_block(self):
        with patch("yfinance.Ticker") as MockTicker:
            MockTicker.return_value.get_earnings_dates.side_effect = RuntimeError("network error")
            ef = EarningsFilter(buffer_days=3)
            blocked, _ = ef.should_block_buy("AAPL")
        assert not blocked  # fail-open

    def test_yfinance_returns_none_does_not_block(self):
        with patch("yfinance.Ticker") as MockTicker:
            MockTicker.return_value.get_earnings_dates.return_value = None
            ef = EarningsFilter(buffer_days=3)
            blocked, _ = ef.should_block_buy("AAPL")
        assert not blocked

    def test_yfinance_returns_empty_df_does_not_block(self):
        with patch("yfinance.Ticker") as MockTicker:
            MockTicker.return_value.get_earnings_dates.return_value = pd.DataFrame()
            ef = EarningsFilter(buffer_days=3)
            blocked, _ = ef.should_block_buy("AAPL")
        assert not blocked


# ── EarningsFilter — on-demand fetch ─────────────────────────────────────────

class TestOnDemandFetch:

    def test_fetch_on_demand_if_not_prefetched(self):
        upcoming = _today()  # today = blocked
        mock_df = _make_earnings_df([upcoming])

        with patch("yfinance.Ticker") as MockTicker:
            MockTicker.return_value.get_earnings_dates.return_value = mock_df
            ef = EarningsFilter(buffer_days=3)
            # No prefetch call
            blocked, _ = ef.should_block_buy("NVDA")

        assert blocked
        assert "NVDA" in ef._cache  # cached after first fetch

    def test_fetch_result_is_cached(self):
        with patch("yfinance.Ticker") as MockTicker:
            MockTicker.return_value.get_earnings_dates.return_value = pd.DataFrame()
            ef = EarningsFilter()
            ef.should_block_buy("TSLA")
            ef.should_block_buy("TSLA")  # second call — should NOT call yfinance again

        assert MockTicker.return_value.get_earnings_dates.call_count == 1


# ── EarningsFilter — custom buffer ───────────────────────────────────────────

class TestCustomBuffer:

    def test_buffer_0_only_blocks_on_earnings_day(self):
        target = pd.bdate_range(pd.Timestamp.today(), periods=2)[-1].date()
        ef = EarningsFilter(buffer_days=0)
        ef._cache["AAPL"] = pd.DatetimeIndex([pd.Timestamp(target)])
        blocked, _ = ef.should_block_buy("AAPL")
        assert not blocked  # 1 bday away, buffer=0 → not blocked

    def test_buffer_5_blocks_5_bdays_out(self):
        target = pd.bdate_range(pd.Timestamp.today(), periods=6)[-1].date()
        ef = EarningsFilter(buffer_days=5)
        ef._cache["AAPL"] = pd.DatetimeIndex([pd.Timestamp(target)])
        blocked, _ = ef.should_block_buy("AAPL")
        assert blocked
