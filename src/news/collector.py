"""
NewsCollector — pulls company news from Finnhub.

Rate limit: 60 calls/min (free tier). Enforced with per-call delay.
Cache: one JSON file per ticker per day in logs/news_cache/.
Returns empty list on API failure — never raises.
"""
from __future__ import annotations

import hashlib
import json
import logging
import os
import re
import time
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import Optional

import requests

logger = logging.getLogger(__name__)

FINNHUB_BASE     = "https://finnhub.io/api/v1"
RATE_LIMIT_DELAY = 1.1   # seconds between calls (stays well under 60/min)


@dataclass
class NewsItem:
    ticker:         str
    headline:       str
    source:         str
    url:            str
    datetime:       datetime
    category:       str          # 'earnings' | 'ma' | 'guidance' | 'analyst' | 'general'
    headline_hash:  str = field(default="")

    def __post_init__(self) -> None:
        if not self.headline_hash:
            self.headline_hash = _headline_hash(self.headline)

    def to_dict(self) -> dict:
        d = asdict(self)
        d["datetime"] = self.datetime.isoformat()
        return d


def _headline_hash(headline: str) -> str:
    """Normalized fingerprint for deduplication: first 8 significant tokens."""
    normalized = re.sub(r"[^a-z0-9\s]", "", headline.lower())
    tokens = [t for t in normalized.split() if len(t) > 3][:8]
    return hashlib.md5(" ".join(tokens).encode()).hexdigest()[:12]


_EARNINGS_KW = ("earnings", " eps", " revenue", " profit", " quarter",
                " q1 ", " q2 ", " q3 ", " q4 ", " beats", " misses",
                "results", "per share")
_MA_KW       = ("merger", "acquisition", "acquires", "takeover",
                "agreed to acquire", "agreed to merge", "agrees to buy",
                "going private", "tender offer", "hostile bid")
_GUIDANCE_KW = ("guidance", "forecast", "outlook", "raises guidance",
                "lowers guidance", "reaffirms", "raises target",
                "lowers target")
_ANALYST_KW  = ("upgrade", "downgrade", "price target", " rating ",
                "initiates", "overweight", "underweight", "outperform",
                "underperform", "buy rating", "sell rating", "hold rating")


def _categorize(raw_category: str, headline: str) -> str:
    hl = " " + headline.lower() + " "
    # Check most-specific categories first to avoid false positives from
    # overlapping keywords (e.g. "raises guidance for Q3" should be guidance,
    # not earnings, even though " q3 " also appears in _EARNINGS_KW).
    if any(k in hl for k in _MA_KW):
        return "ma"
    if any(k in hl for k in _GUIDANCE_KW):
        return "guidance"
    if any(k in hl for k in _ANALYST_KW):
        return "analyst"
    if any(k in hl for k in _EARNINGS_KW):
        return "earnings"
    # Fallback to Finnhub native category
    cat = raw_category.lower()
    if "earnings" in cat or "ipo" in cat:
        return "earnings"
    if "merger" in cat:
        return "ma"
    return "general"


class NewsCollector:
    def __init__(
        self,
        api_key:         Optional[str] = None,
        cache_dir:       str  = "logs/news_cache",
        cache_ttl_hours: float = 6.0,
    ) -> None:
        self._api_key  = api_key or os.getenv("FINNHUB_API_KEY", "")
        self._cache    = Path(cache_dir)
        self._cache.mkdir(parents=True, exist_ok=True)
        self._ttl      = timedelta(hours=cache_ttl_hours)

        if not self._api_key:
            logger.warning("NewsCollector: FINNHUB_API_KEY not set — returning empty results")

    # ── Cache helpers ─────────────────────────────────────────────────────────

    def _cache_path(self, ticker: str, date_str: str) -> Path:
        return self._cache / f"{ticker}_{date_str}.json"

    def _load_cache(self, ticker: str, date_str: str) -> Optional[list[dict]]:
        path = self._cache_path(ticker, date_str)
        if not path.exists():
            return None
        try:
            data     = json.loads(path.read_text())
            cached_at = datetime.fromisoformat(data["cached_at"])
            if datetime.now(timezone.utc) - cached_at > self._ttl:
                return None
            return data["items"]
        except Exception:
            return None

    def _save_cache(self, ticker: str, date_str: str, items: list[dict]) -> None:
        try:
            self._cache_path(ticker, date_str).write_text(json.dumps({
                "cached_at": datetime.now(timezone.utc).isoformat(),
                "items": items,
            }))
        except Exception as exc:
            logger.debug("NewsCollector: cache write failed: %s", exc)

    # ── Public API ────────────────────────────────────────────────────────────

    def fetch_company_news(self, ticker: str, days_back: int = 1) -> list[NewsItem]:
        """Fetch news for one ticker. Returns [] on any failure."""
        if not self._api_key:
            return []

        today    = datetime.now(timezone.utc).date()
        date_str = today.isoformat()
        from_dt  = (today - timedelta(days=days_back)).isoformat()

        cached = self._load_cache(ticker, date_str)
        if cached is not None:
            return [self._from_dict(d) for d in cached]

        try:
            resp = requests.get(
                f"{FINNHUB_BASE}/company-news",
                params={"symbol": ticker, "from": from_dt, "to": date_str,
                        "token": self._api_key},
                timeout=10,
            )
            resp.raise_for_status()
            raw = resp.json()
        except Exception as exc:
            logger.warning("NewsCollector: API error for %s: %s", ticker, exc)
            return []

        items: list[NewsItem] = []
        seen: set[str]        = set()

        for entry in raw:
            try:
                headline = str(entry.get("headline", "")).strip()
                if not headline or len(headline) < 10:
                    continue
                h = _headline_hash(headline)
                if h in seen:
                    continue
                seen.add(h)
                items.append(NewsItem(
                    ticker        = ticker,
                    headline      = headline,
                    source        = str(entry.get("source", "")),
                    url           = str(entry.get("url", "")),
                    datetime      = datetime.fromtimestamp(
                                        entry.get("datetime", 0), tz=timezone.utc),
                    category      = _categorize(entry.get("category", ""), headline),
                    headline_hash = h,
                ))
            except Exception:
                continue

        self._save_cache(ticker, date_str, [i.to_dict() for i in items])
        return items

    def fetch_all(
        self,
        tickers:   list[str],
        days_back: int   = 1,
        delay:     float = RATE_LIMIT_DELAY,
    ) -> list[NewsItem]:
        """Fetch all tickers sequentially, deduplicate cross-ticker, return combined list."""
        all_items: list[NewsItem] = []
        for i, ticker in enumerate(tickers):
            items = self.fetch_company_news(ticker, days_back=days_back)
            all_items.extend(items)
            logger.debug("NewsCollector: %s → %d items", ticker, len(items))
            if i < len(tickers) - 1:
                time.sleep(delay)
        return self._dedup_cross_ticker(all_items)

    # ── Internal ──────────────────────────────────────────────────────────────

    def _dedup_cross_ticker(self, items: list[NewsItem]) -> list[NewsItem]:
        """
        Cross-ticker dedup: same headline_hash = same story.
        Keep the copy where the ticker appears in the headline; else keep the
        first-seen (newest-first sort guarantees recency).
        """
        seen: dict[str, NewsItem] = {}
        for item in sorted(items, key=lambda x: x.datetime, reverse=True):
            key = item.headline_hash
            if key not in seen:
                seen[key] = item
            else:
                existing = seen[key]
                if (existing.ticker.upper() not in item.headline.upper()
                        and item.ticker.upper() in item.headline.upper()):
                    seen[key] = item
        return list(seen.values())

    @staticmethod
    def _from_dict(d: dict) -> NewsItem:
        d = dict(d)
        d["datetime"] = datetime.fromisoformat(d["datetime"])
        return NewsItem(**d)
