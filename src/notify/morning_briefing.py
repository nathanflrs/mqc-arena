"""
Morning market briefing: RSS scraping + yfinance → dashboard (event bus).
Runs at 08h00 CET (07:00 UTC) via GitHub Actions on trading days.
No LLM — raw data formatted directly.
"""
from __future__ import annotations

from datetime import datetime

import feedparser
import yfinance as yf
from dotenv import load_dotenv

from src.config import WATCHLIST

load_dotenv()

RSS_FEEDS = [
    ("Reuters",          "🌍", "https://feeds.reuters.com/reuters/businessNews"),
    ("Financial Times",  "📰", "https://www.ft.com/rss/home"),
    ("WSJ Markets",      "💼", "https://feeds.a.dj.com/rss/RSSMarketsMain.xml"),
    ("BBC Business",     "🔵", "http://feeds.bbci.co.uk/news/business/rss.xml"),
    ("Yahoo Finance",    "💹", "https://finance.yahoo.com/news/rssindex"),
    ("Seeking Alpha",    "🔎", "https://seekingalpha.com/market_currents.xml"),
]

FUTURES_TICKERS = ["SPY", "QQQ", "NVDA", "GS", "AAPL"]
MAX_HEADLINES = 5
TITLE_MAX = 90  # tronque les titres trop longs

FR_DAYS = ["Lundi", "Mardi", "Mercredi", "Jeudi", "Vendredi", "Samedi", "Dimanche"]
FR_MONTHS = [
    "", "janvier", "février", "mars", "avril", "mai", "juin",
    "juillet", "août", "septembre", "octobre", "novembre", "décembre",
]


def _fr_date(dt: datetime) -> str:
    return f"{FR_DAYS[dt.weekday()]} {dt.day} {FR_MONTHS[dt.month]}"


def _pct(price: float, prev: float) -> str:
    if not prev:
        return "n/a"
    p = (price - prev) / prev * 100
    sign = "+" if p >= 0 else ""
    return f"{sign}{p:.1f}%"


def _build_market_section() -> str:
    lines: list[str] = []

    # Equities line
    parts: list[str] = []
    for ticker in FUTURES_TICKERS:
        try:
            fi = yf.Ticker(ticker).fast_info
            pct = _pct(fi.last_price, fi.previous_close)
            parts.append(f"{ticker} {pct}")
        except Exception:
            parts.append(f"{ticker} n/a")
    lines.append(" | ".join(parts))

    # VIX + DXY line
    extras: list[str] = []
    try:
        hist = yf.Ticker("^VIX").history(period="3d")["Close"]
        if len(hist) >= 2:
            now, prev = float(hist.iloc[-1]), float(hist.iloc[-2])
            chg = now - prev
            sign = "+" if chg >= 0 else ""
            extras.append(f"VIX {now:.1f} ({sign}{chg:.1f})")
    except Exception:
        extras.append("VIX n/a")
    try:
        fi = yf.Ticker("DX-Y.NYB").fast_info
        pct = _pct(fi.last_price, fi.previous_close)
        extras.append(f"DXY {fi.last_price:.1f} ({pct})")
    except Exception:
        extras.append("DXY n/a")
    lines.append(" | ".join(extras))

    return "\n".join(lines)


def _build_feed_section(label: str, emoji: str, url: str) -> str:
    try:
        feed = feedparser.parse(url)
        titles = [
            e.get("title", "").strip()[:TITLE_MAX]
            for e in feed.entries[:MAX_HEADLINES]
            if e.get("title", "").strip()
        ]
    except Exception:
        titles = []
    if not titles:
        return ""
    header = f"{emoji} {label.upper()}"
    bullets = "\n".join(f"• {t}" for t in titles)
    return f"{header}\n{bullets}"


def _build_news_section() -> str:
    """Top-5 scored news for today's WATCHLIST + CTA universe. Non-blocking."""
    try:
        from src.news.collector import NewsCollector
        from src.news.selector import NewsSelector

        CTA_EXTRA = ["TLT", "UUP", "DBC"]
        all_tickers = WATCHLIST + CTA_EXTRA

        collector = NewsCollector()
        selector  = NewsSelector()

        items = collector.fetch_all(all_tickers, days_back=1)
        top   = selector.select_daily(items, portfolio={}, watchlist=WATCHLIST)

        if not top:
            return ""

        cat_icon = {"earnings": "📊", "ma": "🤝", "guidance": "📈",
                    "analyst": "🔍", "general": "📰"}
        lines = ["📰 TODAY'S NEWS"]
        for sn in top:
            icon = cat_icon.get(sn.item.category, "📰")
            headline = sn.item.headline[:90].rstrip()
            lines.append(f"{icon} [{sn.item.ticker}] {headline}")
        return "\n".join(lines)
    except Exception:
        return ""


def _build_dividend_section() -> str:
    try:
        from src.agents.dividend_arbitrage_agent import generate_dividend_report
        return generate_dividend_report(WATCHLIST)
    except Exception:
        return ""


def _build_mc_line() -> str:
    """Quick MC snapshot: N=1,000 simulations over 30 days."""
    try:
        from src.analytics.monte_carlo import run_simulation
        result = run_simulation(
            n_simulations=1_000,
            horizon_days=30,
            save_path=None,  # no persistence for morning run
        )
        p50 = result.percentiles["p50"]
        sign = "+" if p50 >= 0 else ""
        var_sign = "+" if result.var_95 >= 0 else ""
        return (
            f"🎲 MC p50 (30j): {sign}{p50:.1%} | "
            f"VaR95: {var_sign}{result.var_95:.1%}"
        )
    except Exception:
        return ""


def run() -> None:
    now = datetime.now()
    date_str = _fr_date(now)

    market    = _build_market_section()
    mc_line   = _build_mc_line()
    news      = _build_news_section()
    div_cal   = _build_dividend_section()
    feeds     = [_build_feed_section(label, emoji, url) for label, emoji, url in RSS_FEEDS]
    feeds_text= "\n\n".join(s for s in feeds if s)

    market_block = market
    if mc_line:
        market_block = market + "\n" + mc_line

    sections = [
        f"☀️ Milan Capital — Morning Briefing\n{date_str} | 08:00 CET",
        f"📈 MARCHÉS PRÉ-MARKET\n{market_block}",
    ]
    if news:
        sections.append(news)
    if div_cal:
        sections.append(div_cal)
    if feeds_text:
        sections.append(feeds_text)

    message = "\n\n".join(sections)

    # Emit to event bus (dashboard) — info severity, not sent to Telegram
    try:
        from src.events.bus import get_bus, Event
        get_bus().emit(Event(
            type="briefing",
            severity="info",
            title=f"Morning Briefing — {date_str}",
            body=message[:2000],
            meta={"date": date_str},
        ))
    except Exception:
        pass  # bus unavailable (e.g. GitHub Actions without DB) — non-blocking


if __name__ == "__main__":
    run()
