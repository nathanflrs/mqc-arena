"""
NewsSelector — rule-based scoring and filtering of news items.

No LLM. Pure scoring:
  +3.0  ticker has an open position in the portfolio
  +1.0  ticker is on the watchlist
  +2.0  category is earnings | m&a | guidance
  +0.5  category is analyst upgrade/downgrade
  +1.0  recency bonus (article < 12h old)
  -1.5  broad-market ETF malus (SPY, QQQ only — sector/asset ETFs unaffected)

Diversity cap: max 2 articles per ticker in the final Top-5.
Min score to pass: 2.0 (applied AFTER malus).
Empty result is valid (no news above threshold today).
"""
from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone, timedelta

from src.news.collector import NewsItem

# ── Scoring constants ─────────────────────────────────────────────────────────

SCORE_OPEN_POSITION  = 3.0
SCORE_WATCHLIST      = 1.0
SCORE_HIGH_IMPACT    = 2.0   # earnings | ma | guidance
SCORE_ANALYST        = 0.5
SCORE_RECENCY        = 1.0   # article < RECENCY_HOURS old

# SPY and QQQ are the entire market — a position there doesn't make every
# macro article portfolio-relevant. Sector/asset ETFs (TLT, GLD…) keep
# their full score because their news IS specific to that asset class.
MALUS_BROAD_ETF      = -1.5
BROAD_MARKET_ETFS    = frozenset({"SPY", "QQQ"})

RECENCY_HOURS        = 12
MIN_SCORE            = 2.0
MAX_RESULTS          = 5
MAX_PER_TICKER       = 2     # diversity cap: no ticker monopolises the Top-5

HIGH_IMPACT_CATS     = frozenset({"earnings", "ma", "guidance"})

# Categories that warrant an intraday alert when a position is open
ALERT_CATS           = frozenset({"earnings", "ma"})

# ETFs don't have single-company earnings events — exclude from intraday alerts
# to avoid noise from broad-market articles mentioning "earnings season" etc.
ETF_NO_INTRADAY      = frozenset({"SPY","QQQ","TLT","GLD","UUP","DBC","IWM","EEM","XLK","XLF"})


@dataclass
class ScoredNews:
    item:            NewsItem
    score:           float
    breakdown:       dict   # for logging / tests


class NewsSelector:
    """Filter and rank news by portfolio relevance."""

    def select_daily(
        self,
        items:     list[NewsItem],
        portfolio: dict[str, float],  # {ticker: quantity; neg = short}
        watchlist: list[str],
    ) -> list[ScoredNews]:
        """
        Score every item, return top MAX_RESULTS with score >= MIN_SCORE.
        """
        open_pos = {t for t, q in portfolio.items() if abs(q) > 1e-6}
        wl       = set(watchlist)
        now      = datetime.now(timezone.utc)

        scored: list[ScoredNews] = []
        for item in items:
            s, breakdown = self._score(item, open_pos, wl, now)
            if s >= MIN_SCORE:
                scored.append(ScoredNews(item=item, score=s, breakdown=breakdown))

        scored.sort(key=lambda x: (-x.score, -x.item.datetime.timestamp()))
        return self._apply_diversity_cap(scored)

    def check_intraday_alerts(
        self,
        items:     list[NewsItem],
        portfolio: dict[str, float],
    ) -> list[NewsItem]:
        """
        Returns items that warrant an intraday warning event:
        open position AND high-impact category (earnings or M&A).
        """
        open_pos = {t for t, q in portfolio.items() if abs(q) > 1e-6}
        return [
            i for i in items
            if (i.ticker in open_pos
                and i.category in ALERT_CATS
                and i.ticker not in ETF_NO_INTRADAY)
        ]

    # ── Diversity cap ─────────────────────────────────────────────────────────

    @staticmethod
    def _apply_diversity_cap(scored: list[ScoredNews]) -> list[ScoredNews]:
        """Greedy pick: walk the sorted list, accept at most MAX_PER_TICKER per ticker."""
        counts: defaultdict[str, int] = defaultdict(int)
        result: list[ScoredNews]      = []
        for sn in scored:
            ticker = sn.item.ticker
            if counts[ticker] < MAX_PER_TICKER:
                result.append(sn)
                counts[ticker] += 1
            if len(result) == MAX_RESULTS:
                break
        return result

    # ── Scoring logic ─────────────────────────────────────────────────────────

    def _score(
        self,
        item:     NewsItem,
        open_pos: set[str],
        wl:       set[str],
        now:      datetime,
    ) -> tuple[float, dict]:
        score     = 0.0
        breakdown: dict[str, float] = {}

        # Relevance: open position > watchlist
        if item.ticker in open_pos:
            score += SCORE_OPEN_POSITION
            breakdown["open_position"] = SCORE_OPEN_POSITION
        elif item.ticker in wl:
            score += SCORE_WATCHLIST
            breakdown["watchlist"] = SCORE_WATCHLIST

        # Category impact
        if item.category in HIGH_IMPACT_CATS:
            score += SCORE_HIGH_IMPACT
            breakdown["high_impact"] = SCORE_HIGH_IMPACT
        elif item.category == "analyst":
            score += SCORE_ANALYST
            breakdown["analyst"] = SCORE_ANALYST

        # Recency bonus
        if now - item.datetime < timedelta(hours=RECENCY_HOURS):
            score += SCORE_RECENCY
            breakdown["recency"] = SCORE_RECENCY

        # Broad-market ETF malus: SPY/QQQ news is background noise, not
        # portfolio-specific signal — even when a position is open.
        if item.ticker in BROAD_MARKET_ETFS:
            score += MALUS_BROAD_ETF
            breakdown["broad_etf_malus"] = MALUS_BROAD_ETF

        return score, breakdown
