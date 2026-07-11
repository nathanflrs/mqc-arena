"""
Tests for src/news/collector.py and src/news/selector.py.
22 tests covering: scoring, diversity cap, broad-ETF malus,
deduplication, threshold, cache, API failure, intraday alerts,
category detection.
"""
from __future__ import annotations

import json
import time
from datetime import datetime, timezone, timedelta
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from src.news.collector import (
    NewsCollector, NewsItem,
    _headline_hash, _categorize,
)
from src.news.selector import (
    NewsSelector, ScoredNews,
    SCORE_OPEN_POSITION, SCORE_WATCHLIST, SCORE_HIGH_IMPACT,
    SCORE_ANALYST, SCORE_RECENCY, MALUS_BROAD_ETF,
    MIN_SCORE, MAX_RESULTS, MAX_PER_TICKER,
    BROAD_MARKET_ETFS, ETF_NO_INTRADAY,
)

# ── helpers ───────────────────────────────────────────────────────────────────

def _item(ticker: str, category: str = "general", hours_ago: float = 2.0) -> NewsItem:
    dt = datetime.now(timezone.utc) - timedelta(hours=hours_ago)
    return NewsItem(
        ticker   = ticker,
        headline = f"{ticker} some news headline that is long enough",
        source   = "TestSource",
        url      = "https://example.com",
        datetime = dt,
        category = category,
    )


WATCHLIST = ["AAPL", "SPY", "QQQ", "NVDA", "MSFT", "GOOGL",
             "META", "JPM", "GS", "GLD", "TSLA", "AMD", "AMZN", "LLY"]

SEL = NewsSelector()


# ── 1. Category detection ─────────────────────────────────────────────────────

def test_categorize_earnings_keyword():
    assert _categorize("", "Company beats earnings per share estimates") == "earnings"

def test_categorize_ma_keyword():
    assert _categorize("", "Firm agreed to acquire rival for $5B") == "ma"

def test_categorize_guidance_keyword():
    assert _categorize("", "Management raises guidance for Q3") == "guidance"

def test_categorize_analyst_keyword():
    assert _categorize("", "Goldman issues downgrade on NVDA") == "analyst"

def test_categorize_general_fallback():
    assert _categorize("", "Stock moves higher in afternoon trading") == "general"


# ── 2. Headline hash ──────────────────────────────────────────────────────────

def test_headline_hash_stable():
    h1 = _headline_hash("Apple Reports Q3 Earnings Beat!")
    h2 = _headline_hash("Apple Reports Q3 Earnings Beat!")
    assert h1 == h2

def test_headline_hash_normalises_punctuation():
    h1 = _headline_hash("Apple reports Q3 earnings beat!!!")
    h2 = _headline_hash("Apple reports Q3 earnings beat")
    assert h1 == h2


# ── 3. Scoring ────────────────────────────────────────────────────────────────

def test_score_open_position_plus_high_impact():
    item = _item("AAPL", "earnings", hours_ago=20)
    top  = SEL.select_daily([item], portfolio={"AAPL": 10.0}, watchlist=WATCHLIST)
    assert len(top) == 1
    assert top[0].score == SCORE_OPEN_POSITION + SCORE_HIGH_IMPACT

def test_score_open_position_high_impact_recency():
    item = _item("AAPL", "earnings", hours_ago=6)
    top  = SEL.select_daily([item], portfolio={"AAPL": 10.0}, watchlist=WATCHLIST)
    assert top[0].score == SCORE_OPEN_POSITION + SCORE_HIGH_IMPACT + SCORE_RECENCY

def test_score_watchlist_guidance():
    item = _item("NVDA", "guidance", hours_ago=20)
    top  = SEL.select_daily([item], portfolio={}, watchlist=WATCHLIST)
    assert top[0].score == SCORE_WATCHLIST + SCORE_HIGH_IMPACT

def test_score_watchlist_analyst_below_threshold():
    # watchlist(+1) + analyst(+0.5) = 1.5 < MIN_SCORE=2.0 → filtered
    item = _item("NVDA", "analyst", hours_ago=20)
    top  = SEL.select_daily([item], portfolio={}, watchlist=WATCHLIST)
    assert len(top) == 0

def test_score_unknown_ticker_filtered():
    # No position, not on watchlist, general category, old article → score 0 < MIN_SCORE
    item = _item("ZZZZ", "general", hours_ago=20)
    top  = SEL.select_daily([item], portfolio={}, watchlist=WATCHLIST)
    assert len(top) == 0


# ── 4. Broad-market ETF malus ─────────────────────────────────────────────────

def test_broad_etf_malus_spy():
    # SPY open pos(+3) + guidance(+2) + recency(+1) + malus(-1.5) = 4.5
    item = _item("SPY", "guidance", hours_ago=6)
    top  = SEL.select_daily([item], portfolio={"SPY": 50.0}, watchlist=WATCHLIST)
    assert len(top) == 1
    assert top[0].score == pytest.approx(
        SCORE_OPEN_POSITION + SCORE_HIGH_IMPACT + SCORE_RECENCY + MALUS_BROAD_ETF
    )
    assert "broad_etf_malus" in top[0].breakdown

def test_broad_etf_malus_qqq():
    item = _item("QQQ", "earnings", hours_ago=20)
    top  = SEL.select_daily([item], portfolio={}, watchlist=WATCHLIST)
    # watchlist(+1) + earnings(+2) + malus(-1.5) = 1.5 < 2.0 → filtered
    assert len(top) == 0

def test_no_malus_for_tlt():
    # TLT is a sector ETF — no malus applied
    item = _item("TLT", "guidance", hours_ago=20)
    top  = SEL.select_daily([item], portfolio={"TLT": -30.0}, watchlist=["TLT"])
    assert len(top) == 1
    assert "broad_etf_malus" not in top[0].breakdown

def test_no_malus_for_gld():
    item = _item("GLD", "earnings", hours_ago=20)
    top  = SEL.select_daily([item], portfolio={"GLD": 10.0}, watchlist=WATCHLIST)
    assert len(top) == 1
    expected = SCORE_OPEN_POSITION + SCORE_HIGH_IMPACT
    assert top[0].score == pytest.approx(expected)


# ── 5. Recency ────────────────────────────────────────────────────────────────

def test_recency_bonus_applied_under_12h():
    item = _item("AAPL", "earnings", hours_ago=11)
    top  = SEL.select_daily([item], portfolio={}, watchlist=WATCHLIST)
    assert SCORE_RECENCY in top[0].breakdown.values()

def test_no_recency_bonus_over_12h():
    item = _item("AAPL", "earnings", hours_ago=13)
    top  = SEL.select_daily([item], portfolio={}, watchlist=WATCHLIST)
    assert "recency" not in top[0].breakdown


# ── 6. Threshold boundary ─────────────────────────────────────────────────────

def test_min_score_boundary_passes():
    # watchlist(+1) + guidance(+2) - SPY malus would fail — use non-SPY
    item = _item("AAPL", "earnings", hours_ago=20)  # watchlist(+1) + earnings(+2) = 3.0 ≥ 2.0
    top  = SEL.select_daily([item], portfolio={}, watchlist=WATCHLIST)
    assert len(top) == 1

def test_below_threshold_filtered():
    # general category: watchlist(+1) = 1.0 < 2.0
    item = _item("AAPL", "general", hours_ago=20)
    top  = SEL.select_daily([item], portfolio={}, watchlist=WATCHLIST)
    assert len(top) == 0


# ── 7. Diversity cap ──────────────────────────────────────────────────────────

def test_diversity_cap_max_two_per_ticker():
    # 6 AAPL earnings items — only 2 should make the Top-5
    items = [_item("AAPL", "earnings", hours_ago=i) for i in range(1, 7)]
    top   = SEL.select_daily(items, portfolio={"AAPL": 10.0}, watchlist=WATCHLIST)
    aapl_count = sum(1 for sn in top if sn.item.ticker == "AAPL")
    assert aapl_count == MAX_PER_TICKER

def test_diversity_fills_from_other_tickers():
    # 6 AAPL + 2 NVDA + 2 MSFT → cap allows 2 per ticker → 2+2+1 = 5 total
    # (AAPL is capped at 2, NVDA at 2, MSFT fills the 5th slot)
    aapl_items = [_item("AAPL", "earnings", hours_ago=i)   for i in range(1, 7)]
    nvda_items = [_item("NVDA", "guidance", hours_ago=i+6) for i in range(1, 3)]
    msft_items = [_item("MSFT", "earnings", hours_ago=9)]
    top = SEL.select_daily(
        aapl_items + nvda_items + msft_items,
        portfolio={"AAPL": 10.0},
        watchlist=WATCHLIST,
    )
    assert sum(1 for sn in top if sn.item.ticker == "AAPL") == MAX_PER_TICKER
    assert sum(1 for sn in top if sn.item.ticker == "NVDA") == MAX_PER_TICKER
    # No single ticker monopolises more than MAX_PER_TICKER slots
    from collections import Counter
    counts = Counter(sn.item.ticker for sn in top)
    assert max(counts.values()) <= MAX_PER_TICKER

def test_max_results_capped_at_five():
    # 10 different high-scoring items
    items = [_item(t, "earnings", hours_ago=i+1)
             for i, t in enumerate(["AAPL","NVDA","MSFT","GOOGL","META","JPM","GS","TSLA","AMD","AMZN"])]
    top = SEL.select_daily(items, portfolio={t: 10.0 for t in ["AAPL","NVDA","MSFT","GOOGL","META"]},
                           watchlist=WATCHLIST)
    assert len(top) <= MAX_RESULTS


# ── 8. Intraday alerts ────────────────────────────────────────────────────────

def test_intraday_alert_individual_stock():
    item    = _item("AAPL", "earnings", hours_ago=2)
    alerts  = SEL.check_intraday_alerts([item], portfolio={"AAPL": 10.0})
    assert len(alerts) == 1 and alerts[0].ticker == "AAPL"

def test_intraday_alert_etf_excluded():
    # SPY is in ETF_NO_INTRADAY → no alert even with open position
    item   = _item("SPY", "earnings", hours_ago=2)
    alerts = SEL.check_intraday_alerts([item], portfolio={"SPY": 100.0})
    assert len(alerts) == 0

def test_intraday_no_alert_watchlist_only():
    # NVDA in watchlist but no position → no intraday alert
    item   = _item("NVDA", "earnings", hours_ago=2)
    alerts = SEL.check_intraday_alerts([item], portfolio={})
    assert len(alerts) == 0

def test_intraday_general_category_no_alert():
    # only earnings/ma trigger intraday alerts
    item   = _item("AAPL", "general", hours_ago=2)
    alerts = SEL.check_intraday_alerts([item], portfolio={"AAPL": 10.0})
    assert len(alerts) == 0


# ── 9. Deduplication ─────────────────────────────────────────────────────────

def test_dedup_same_headline_within_ticker():
    now = datetime.now(timezone.utc)
    i1  = NewsItem("AAPL", "Apple reports record earnings beat", "S1", "", now, "earnings")
    i2  = NewsItem("AAPL", "Apple reports record earnings beat", "S2", "", now - timedelta(hours=1), "earnings")
    collector = NewsCollector.__new__(NewsCollector)
    result    = collector._dedup_cross_ticker([i1, i2])
    assert len(result) == 1

def test_dedup_cross_ticker_keeps_relevant():
    now = datetime.now(timezone.utc)
    i_aapl = NewsItem("AAPL", "Apple reports record earnings beat", "S", "", now, "earnings")
    i_spy  = NewsItem("SPY",  "Apple reports record earnings beat", "S", "", now - timedelta(seconds=1), "earnings")
    collector = NewsCollector.__new__(NewsCollector)
    result    = collector._dedup_cross_ticker([i_aapl, i_spy])
    assert len(result) == 1
    # AAPL appears in the headline "Apple..." → AAPL should be kept
    assert result[0].ticker == "AAPL"


# ── 10. API failure ───────────────────────────────────────────────────────────

def test_api_failure_returns_empty_list(tmp_path):
    collector = NewsCollector(api_key="fake-key", cache_dir=str(tmp_path))
    with patch("src.news.collector.requests.get") as mock_get:
        mock_get.side_effect = Exception("network error")
        result = collector.fetch_company_news("AAPL")
    assert result == []

def test_missing_api_key_returns_empty_list(tmp_path):
    collector = NewsCollector(api_key="", cache_dir=str(tmp_path))
    result    = collector.fetch_company_news("AAPL")
    assert result == []


# ── 11. Cache ─────────────────────────────────────────────────────────────────

def test_cache_used_when_fresh(tmp_path):
    item = NewsItem(
        ticker="AAPL", headline="Test headline for cache",
        source="S", url="", datetime=datetime.now(timezone.utc),
        category="general",
    )
    collector   = NewsCollector(api_key="test-key", cache_dir=str(tmp_path))
    today_str   = datetime.now(timezone.utc).date().isoformat()
    cache_path  = tmp_path / f"AAPL_{today_str}.json"
    cache_path.write_text(json.dumps({
        "cached_at": datetime.now(timezone.utc).isoformat(),
        "items":     [item.to_dict()],
    }))

    api_called = False
    with patch("src.news.collector.requests.get") as mock_get:
        mock_get.side_effect = lambda *a, **kw: (_ for _ in ()).throw(Exception("should not call API"))
        result = collector.fetch_company_news("AAPL")

    assert len(result) == 1
    assert result[0].headline == "Test headline for cache"


# ── 12. select_daily edge cases ───────────────────────────────────────────────

def test_select_daily_empty_input():
    top = SEL.select_daily([], portfolio={}, watchlist=WATCHLIST)
    assert top == []

def test_select_daily_returns_sorted_by_score():
    items = [
        _item("AAPL", "earnings",  hours_ago=20),   # watchlist +1 + earnings +2 = 3.0
        _item("NVDA", "guidance",  hours_ago=4),    # watchlist +1 + guidance +2 + recency +1 = 4.0
        _item("AMZN", "earnings",  hours_ago=4),    # watchlist +1 + earnings +2 + recency +1 = 4.0
    ]
    top = SEL.select_daily(items, portfolio={}, watchlist=WATCHLIST)
    scores = [sn.score for sn in top]
    assert scores == sorted(scores, reverse=True)
