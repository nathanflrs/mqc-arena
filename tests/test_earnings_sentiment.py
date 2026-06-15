# tests/test_earnings_sentiment.py
from __future__ import annotations

import json
import time
from unittest.mock import MagicMock, patch

import pytest

from src.agents.base import MarketState
from src.agents.earnings_sentiment import EarningsSentimentAgent, EarningsSentimentConfig

# ── Fixtures ──────────────────────────────────────────────────────────────────

MOCK_NEWS = [
    {"title": "Apple beats Q4 earnings, EPS $1.50 vs $1.40 expected", "publisher": "Reuters", "summary": "Apple reported strong Q4."},
    {"title": "iPhone demand strong in Asia", "publisher": "Bloomberg", "summary": ""},
]

_BUY  = '{"action":"BUY","confidence":0.78,"reason":"EPS beat + raised guidance","sentiment_score":0.8,"key_catalyst":"EPS beat Q4"}'
_HOLD = '{"action":"HOLD","confidence":0.52,"reason":"Mixed signals","sentiment_score":0.1,"key_catalyst":"no clear catalyst"}'
_SELL = '{"action":"SELL","confidence":0.72,"reason":"Guidance cut −20%","sentiment_score":-0.7,"key_catalyst":"FY guidance cut"}'


def _state(symbol: str = "AAPL", price: float = 190.0) -> MarketState:
    return MarketState(symbol=symbol, price=price, timestamp="2026-06-14T09:30:00Z")


def _resp(text: str) -> MagicMock:
    block = MagicMock()
    block.type = "text"
    block.text = text
    resp = MagicMock()
    resp.content = [block]
    return resp


# ── Tests ─────────────────────────────────────────────────────────────────────

@patch("src.agents.earnings_sentiment.yf.Ticker")
@patch("src.agents.earnings_sentiment.anthropic.Anthropic")
def test_buy_signal(mock_cls, mock_ticker):
    mock_ticker.return_value.news = MOCK_NEWS
    mock_cls.return_value.messages.create.return_value = _resp(_BUY)

    sig = EarningsSentimentAgent().generate_signal(_state(), {})

    assert sig.action == "BUY"
    assert sig.confidence == pytest.approx(0.78)
    assert sig.target_weight == pytest.approx(0.10)
    assert sig.agent_name == "EarningsSentimentAgent"


@patch("src.agents.earnings_sentiment.yf.Ticker")
@patch("src.agents.earnings_sentiment.anthropic.Anthropic")
def test_hold_signal(mock_cls, mock_ticker):
    mock_ticker.return_value.news = MOCK_NEWS
    mock_cls.return_value.messages.create.return_value = _resp(_HOLD)

    sig = EarningsSentimentAgent().generate_signal(_state(), {})

    assert sig.action == "HOLD"
    assert sig.target_weight == 0.0


@patch("src.agents.earnings_sentiment.yf.Ticker")
@patch("src.agents.earnings_sentiment.anthropic.Anthropic")
def test_sell_signal(mock_cls, mock_ticker):
    mock_ticker.return_value.news = MOCK_NEWS
    mock_cls.return_value.messages.create.return_value = _resp(_SELL)

    sig = EarningsSentimentAgent().generate_signal(_state(), {})

    assert sig.action == "SELL"
    assert sig.target_weight == 0.0


@patch("src.agents.earnings_sentiment.yf.Ticker")
@patch("src.agents.earnings_sentiment.anthropic.Anthropic")
def test_low_confidence_coerced_to_hold(mock_cls, mock_ticker):
    """BUY with confidence below min_confidence becomes HOLD."""
    low = '{"action":"BUY","confidence":0.40,"reason":"weak","sentiment_score":0.2,"key_catalyst":"analyst note"}'
    mock_ticker.return_value.news = MOCK_NEWS
    mock_cls.return_value.messages.create.return_value = _resp(low)

    sig = EarningsSentimentAgent().generate_signal(_state(), {})

    assert sig.action == "HOLD"
    assert sig.target_weight == 0.0
    assert sig.confidence == pytest.approx(0.40)  # confidence preserved in meta even after downgrade


@patch("src.agents.earnings_sentiment.yf.Ticker")
@patch("src.agents.earnings_sentiment.anthropic.Anthropic")
def test_cache_prevents_second_api_call(mock_cls, mock_ticker):
    """Second call for the same symbol on the same day must not re-call Claude."""
    mock_ticker.return_value.news = MOCK_NEWS
    mock_client = mock_cls.return_value
    mock_client.messages.create.return_value = _resp(_BUY)

    agent = EarningsSentimentAgent()
    agent.generate_signal(_state(), {})
    agent.generate_signal(_state(), {})

    assert mock_client.messages.create.call_count == 1


@patch("src.agents.earnings_sentiment.yf.Ticker")
@patch("src.agents.earnings_sentiment.anthropic.Anthropic")
def test_expired_cache_triggers_new_call(mock_cls, mock_ticker):
    """Cache entry older than TTL should trigger a fresh API call."""
    mock_ticker.return_value.news = MOCK_NEWS
    mock_client = mock_cls.return_value
    mock_client.messages.create.return_value = _resp(_BUY)

    agent = EarningsSentimentAgent(EarningsSentimentConfig(cache_ttl_hours=0.0))
    agent.generate_signal(_state(), {})
    agent.generate_signal(_state(), {})

    assert mock_client.messages.create.call_count == 2


@patch("src.agents.earnings_sentiment.yf.Ticker")
@patch("src.agents.earnings_sentiment.anthropic.Anthropic")
def test_api_error_returns_hold(mock_cls, mock_ticker):
    """Claude API exception → HOLD with error reason, never raises."""
    mock_ticker.return_value.news = MOCK_NEWS
    mock_cls.return_value.messages.create.side_effect = Exception("network error")

    sig = EarningsSentimentAgent().generate_signal(_state(), {})

    assert sig.action == "HOLD"
    assert "API error" in sig.reason


@patch("src.agents.earnings_sentiment.yf.Ticker")
@patch("src.agents.earnings_sentiment.anthropic.Anthropic")
def test_no_news_still_calls_claude(mock_cls, mock_ticker):
    """Empty news list: agent should still call Claude (passes 'No recent news available.')."""
    mock_ticker.return_value.news = []
    mock_cls.return_value.messages.create.return_value = _resp(_HOLD)

    sig = EarningsSentimentAgent().generate_signal(_state(), {})

    assert sig.agent_name == "EarningsSentimentAgent"
    mock_cls.return_value.messages.create.assert_called_once()
    call_args = mock_cls.return_value.messages.create.call_args
    prompt = call_args.kwargs["messages"][0]["content"]
    assert "No recent news available" in prompt


@patch("src.agents.earnings_sentiment.yf.Ticker")
@patch("src.agents.earnings_sentiment.anthropic.Anthropic")
def test_yfinance_error_still_calls_claude(mock_cls, mock_ticker):
    """yfinance failure → empty news → Claude still called, no crash."""
    mock_ticker.side_effect = Exception("rate limit")
    mock_cls.return_value.messages.create.return_value = _resp(_HOLD)

    sig = EarningsSentimentAgent().generate_signal(_state(), {})

    assert sig.agent_name == "EarningsSentimentAgent"


@patch("src.agents.earnings_sentiment.yf.Ticker")
@patch("src.agents.earnings_sentiment.anthropic.Anthropic")
def test_custom_min_confidence(mock_cls, mock_ticker):
    """With min_confidence=0.85, a BUY at 0.78 becomes HOLD."""
    mock_ticker.return_value.news = MOCK_NEWS
    mock_cls.return_value.messages.create.return_value = _resp(_BUY)  # confidence=0.78

    cfg = EarningsSentimentConfig(min_confidence=0.85, target_weight=0.15)
    sig = EarningsSentimentAgent(config=cfg).generate_signal(_state(), {})

    assert sig.action == "HOLD"


@patch("src.agents.earnings_sentiment.yf.Ticker")
@patch("src.agents.earnings_sentiment.anthropic.Anthropic")
def test_markdown_fenced_json_parsed(mock_cls, mock_ticker):
    """Claude wrapping JSON in ```json ... ``` must still parse correctly."""
    fenced = f"```json\n{_BUY}\n```"
    mock_ticker.return_value.news = MOCK_NEWS
    mock_cls.return_value.messages.create.return_value = _resp(fenced)

    sig = EarningsSentimentAgent().generate_signal(_state(), {})

    assert sig.action == "BUY"


@patch("src.agents.earnings_sentiment.yf.Ticker")
@patch("src.agents.earnings_sentiment.anthropic.Anthropic")
def test_regime_passed_to_prompt(mock_cls, mock_ticker):
    """Regime string must appear in the prompt sent to Claude."""
    mock_ticker.return_value.news = []
    mock_cls.return_value.messages.create.return_value = _resp(_HOLD)

    EarningsSentimentAgent().generate_signal(_state(), {}, regime="bear")

    prompt = mock_cls.return_value.messages.create.call_args.kwargs["messages"][0]["content"]
    assert "bear" in prompt
