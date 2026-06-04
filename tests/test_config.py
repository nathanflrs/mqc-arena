# tests/test_config.py
from __future__ import annotations

import pytest

from src.config import WATCHLIST, AGENT_PRIORITY

# TLT exclu de la watchlist active (Sharpe négatif sur tous les agents)
ACTIVE_TICKERS = {
    "AAPL", "SPY", "QQQ", "NVDA", "MSFT",
    "GOOGL", "META", "JPM", "GS", "GLD",
    "BRK-B", "JNJ", "TSLA", "AMD",
}
ARCHIVED_TICKERS = {"TLT"}  # gardés dans AGENT_PRIORITY pour référence, hors watchlist

VALID_AGENTS = {
    "BuffettAgent",
    "CitadelAgent",
    "MeanReversionAgent",
    "TrendFollowingAgent",
    "MacroAgent",
    "VolatilityAgent",
    "DividendArbitrageAgent",
    "PairsTradingAgent",
    "DummyHoldAgent",
}


def test_watchlist_contains_exactly_active_tickers():
    assert set(WATCHLIST) == ACTIVE_TICKERS


def test_watchlist_excludes_archived_tickers():
    assert not (set(WATCHLIST) & ARCHIVED_TICKERS), "TLT ne doit pas être dans WATCHLIST"


def test_watchlist_has_no_duplicates():
    assert len(WATCHLIST) == len(set(WATCHLIST))


def test_agent_priority_covers_full_watchlist():
    missing = set(WATCHLIST) - set(AGENT_PRIORITY.keys())
    assert not missing, f"Tickers sans agent assigné : {missing}"


def test_agent_priority_may_contain_archived_tickers():
    # TLT peut rester dans AGENT_PRIORITY pour référence sans être dans WATCHLIST
    for ticker in ARCHIVED_TICKERS:
        if ticker in AGENT_PRIORITY:
            assert AGENT_PRIORITY[ticker] in VALID_AGENTS


def test_agent_priority_values_are_valid_agents():
    invalid = {
        sym: agent
        for sym, agent in AGENT_PRIORITY.items()
        if agent not in VALID_AGENTS
    }
    assert not invalid, f"Agents inconnus dans AGENT_PRIORITY : {invalid}"


@pytest.mark.parametrize("ticker", sorted(ACTIVE_TICKERS))
def test_each_active_ticker_has_priority_agent(ticker):
    assert ticker in AGENT_PRIORITY, f"{ticker} manquant dans AGENT_PRIORITY"
    assert isinstance(AGENT_PRIORITY[ticker], str)
    assert len(AGENT_PRIORITY[ticker]) > 0
