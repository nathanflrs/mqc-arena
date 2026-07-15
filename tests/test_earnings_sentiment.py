# tests/test_earnings_sentiment.py
from __future__ import annotations

import json
import time
from unittest.mock import MagicMock, patch

import pytest

from src.agents.base import MarketState
from src.agents.earnings_sentiment import EarningsSentimentAgent, EarningsSentimentConfig

# ── Fixtures ──────────────────────────────────────────────────────────────────

def _news_item(headline: str, source: str = "Reuters", category: str = "earnings"):
    item = MagicMock()
    item.headline = headline
    item.source = source
    item.category = category
    return item


MOCK_NEWS_ITEMS = [
    _news_item("Apple beats Q4 earnings, EPS $1.50 vs $1.40 expected"),
    _news_item("iPhone demand strong in Asia", source="Bloomberg"),
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


def _mock_news(items=None):
    """Patch NewsCollector.fetch_company_news to return the given items list."""
    mock = patch("src.agents.earnings_sentiment.NewsCollector")
    return mock, items if items is not None else MOCK_NEWS_ITEMS


# ── Tests ─────────────────────────────────────────────────────────────────────

@patch("src.agents.earnings_sentiment.NewsCollector")
@patch("src.agents.earnings_sentiment.anthropic.Anthropic")
def test_buy_signal(mock_cls, mock_news_cls):
    mock_news_cls.return_value.fetch_company_news.return_value = MOCK_NEWS_ITEMS
    mock_cls.return_value.messages.create.return_value = _resp(_BUY)

    sig = EarningsSentimentAgent().generate_signal(_state(), {})

    assert sig.action == "BUY"
    assert sig.confidence == pytest.approx(0.78)
    assert sig.target_weight == pytest.approx(0.10)
    assert sig.agent_name == "EarningsSentimentAgent"


@patch("src.agents.earnings_sentiment.NewsCollector")
@patch("src.agents.earnings_sentiment.anthropic.Anthropic")
def test_hold_signal(mock_cls, mock_news_cls):
    mock_news_cls.return_value.fetch_company_news.return_value = MOCK_NEWS_ITEMS
    mock_cls.return_value.messages.create.return_value = _resp(_HOLD)

    sig = EarningsSentimentAgent().generate_signal(_state(), {})

    assert sig.action == "HOLD"
    assert sig.target_weight == 0.0


@patch("src.agents.earnings_sentiment.NewsCollector")
@patch("src.agents.earnings_sentiment.anthropic.Anthropic")
def test_sell_signal(mock_cls, mock_news_cls):
    mock_news_cls.return_value.fetch_company_news.return_value = MOCK_NEWS_ITEMS
    mock_cls.return_value.messages.create.return_value = _resp(_SELL)

    sig = EarningsSentimentAgent().generate_signal(_state(), {})

    assert sig.action == "SELL"
    assert sig.target_weight == 0.0


@patch("src.agents.earnings_sentiment.NewsCollector")
@patch("src.agents.earnings_sentiment.anthropic.Anthropic")
def test_low_confidence_coerced_to_hold(mock_cls, mock_news_cls):
    """BUY with confidence below min_confidence becomes HOLD."""
    low = '{"action":"BUY","confidence":0.40,"reason":"weak","sentiment_score":0.2,"key_catalyst":"analyst note"}'
    mock_news_cls.return_value.fetch_company_news.return_value = MOCK_NEWS_ITEMS
    mock_cls.return_value.messages.create.return_value = _resp(low)

    sig = EarningsSentimentAgent().generate_signal(_state(), {})

    assert sig.action == "HOLD"
    assert sig.target_weight == 0.0
    assert sig.confidence == pytest.approx(0.40)


@patch("src.agents.earnings_sentiment.NewsCollector")
@patch("src.agents.earnings_sentiment.anthropic.Anthropic")
def test_cache_prevents_second_api_call(mock_cls, mock_news_cls):
    """Second call for the same symbol on the same day must not re-call Claude."""
    mock_news_cls.return_value.fetch_company_news.return_value = MOCK_NEWS_ITEMS
    mock_client = mock_cls.return_value
    mock_client.messages.create.return_value = _resp(_BUY)

    agent = EarningsSentimentAgent()
    agent.generate_signal(_state(), {})
    agent.generate_signal(_state(), {})

    assert mock_client.messages.create.call_count == 1


@patch("src.agents.earnings_sentiment.NewsCollector")
@patch("src.agents.earnings_sentiment.anthropic.Anthropic")
def test_expired_cache_triggers_new_call(mock_cls, mock_news_cls):
    """Cache entry older than TTL should trigger a fresh API call."""
    mock_news_cls.return_value.fetch_company_news.return_value = MOCK_NEWS_ITEMS
    mock_client = mock_cls.return_value
    mock_client.messages.create.return_value = _resp(_BUY)

    agent = EarningsSentimentAgent(EarningsSentimentConfig(cache_ttl_hours=0.0))
    agent.generate_signal(_state(), {})
    agent.generate_signal(_state(), {})

    assert mock_client.messages.create.call_count == 2


@patch("src.agents.earnings_sentiment.NewsCollector")
@patch("src.agents.earnings_sentiment.anthropic.Anthropic")
def test_api_error_returns_hold(mock_cls, mock_news_cls):
    """Claude API exception → HOLD with error reason, never raises."""
    mock_news_cls.return_value.fetch_company_news.return_value = MOCK_NEWS_ITEMS
    mock_cls.return_value.messages.create.side_effect = Exception("network error")

    sig = EarningsSentimentAgent().generate_signal(_state(), {})

    assert sig.action == "HOLD"
    assert "API error" in sig.reason


@patch("src.agents.earnings_sentiment.NewsCollector")
@patch("src.agents.earnings_sentiment.anthropic.Anthropic")
def test_no_news_still_calls_claude(mock_cls, mock_news_cls):
    """Empty news list: agent should still call Claude (passes 'No recent news available.')."""
    mock_news_cls.return_value.fetch_company_news.return_value = []
    mock_cls.return_value.messages.create.return_value = _resp(_HOLD)

    sig = EarningsSentimentAgent().generate_signal(_state(), {})

    assert sig.agent_name == "EarningsSentimentAgent"
    mock_cls.return_value.messages.create.assert_called_once()
    call_args = mock_cls.return_value.messages.create.call_args
    prompt = call_args.kwargs["messages"][0]["content"]
    assert "No recent news available" in prompt


@patch("src.agents.earnings_sentiment.NewsCollector")
@patch("src.agents.earnings_sentiment.anthropic.Anthropic")
def test_news_collector_error_still_calls_claude(mock_cls, mock_news_cls):
    """NewsCollector failure → empty news → Claude still called, no crash."""
    mock_news_cls.return_value.fetch_company_news.side_effect = Exception("rate limit")
    mock_cls.return_value.messages.create.return_value = _resp(_HOLD)

    sig = EarningsSentimentAgent().generate_signal(_state(), {})

    assert sig.agent_name == "EarningsSentimentAgent"


@patch("src.agents.earnings_sentiment.NewsCollector")
@patch("src.agents.earnings_sentiment.anthropic.Anthropic")
def test_custom_min_confidence(mock_cls, mock_news_cls):
    """With min_confidence=0.85, a BUY at 0.78 becomes HOLD."""
    mock_news_cls.return_value.fetch_company_news.return_value = MOCK_NEWS_ITEMS
    mock_cls.return_value.messages.create.return_value = _resp(_BUY)  # confidence=0.78

    cfg = EarningsSentimentConfig(min_confidence=0.85, target_weight=0.15)
    sig = EarningsSentimentAgent(config=cfg).generate_signal(_state(), {})

    assert sig.action == "HOLD"


@patch("src.agents.earnings_sentiment.NewsCollector")
@patch("src.agents.earnings_sentiment.anthropic.Anthropic")
def test_markdown_fenced_json_parsed(mock_cls, mock_news_cls):
    """Claude wrapping JSON in ```json ... ``` must still parse correctly."""
    fenced = f"```json\n{_BUY}\n```"
    mock_news_cls.return_value.fetch_company_news.return_value = MOCK_NEWS_ITEMS
    mock_cls.return_value.messages.create.return_value = _resp(fenced)

    sig = EarningsSentimentAgent().generate_signal(_state(), {})

    assert sig.action == "BUY"


@patch("src.agents.earnings_sentiment.NewsCollector")
@patch("src.agents.earnings_sentiment.anthropic.Anthropic")
def test_regime_passed_to_prompt(mock_cls, mock_news_cls):
    """Regime string must appear in the prompt sent to Claude."""
    mock_news_cls.return_value.fetch_company_news.return_value = []
    mock_cls.return_value.messages.create.return_value = _resp(_HOLD)

    EarningsSentimentAgent().generate_signal(_state(), {}, regime="bear")

    prompt = mock_cls.return_value.messages.create.call_args.kwargs["messages"][0]["content"]
    assert "bear" in prompt


def test_missing_api_key_returns_hold_without_crash():
    """When ANTHROPIC_API_KEY is absent the SDK raises TypeError; agent must return HOLD."""
    import src.agents.earnings_sentiment as mod
    orig = mod._missing_key_warned
    mod._missing_key_warned = False
    try:
        with patch(
            "src.agents.earnings_sentiment.anthropic.Anthropic",
            side_effect=TypeError("Could not resolve authentication method"),
        ):
            sig = EarningsSentimentAgent().generate_signal(_state(), {})
        assert sig.action == "HOLD"
        assert sig.target_weight == 0.0
        assert "API error" in sig.reason
    finally:
        mod._missing_key_warned = orig


def test_missing_api_key_warning_logged_once(caplog):
    """Warning is emitted exactly once even when called for multiple tickers."""
    import src.agents.earnings_sentiment as mod
    mod._missing_key_warned = False
    try:
        with patch(
            "src.agents.earnings_sentiment.anthropic.Anthropic",
            side_effect=TypeError("Could not resolve authentication method"),
        ):
            import logging
            agent = EarningsSentimentAgent()
            with caplog.at_level(logging.WARNING, logger="src.agents.earnings_sentiment"):
                agent.generate_signal(_state("AAPL"), {})
                agent._client = None
                agent.generate_signal(_state("NVDA"), {})
        warning_count = sum(
            1 for r in caplog.records
            if "ANTHROPIC_API_KEY" in r.message
        )
        assert warning_count == 1
    finally:
        mod._missing_key_warned = False


@patch("src.agents.earnings_sentiment.NewsCollector")
@patch("src.agents.earnings_sentiment.anthropic.Anthropic")
def test_uses_haiku_not_opus(mock_cls, mock_news_cls):
    """Model must be Haiku 4.5, not Opus — cost guard."""
    mock_news_cls.return_value.fetch_company_news.return_value = MOCK_NEWS_ITEMS
    mock_cls.return_value.messages.create.return_value = _resp(_HOLD)

    EarningsSentimentAgent().generate_signal(_state(), {})

    call_kwargs = mock_cls.return_value.messages.create.call_args.kwargs
    assert "haiku" in call_kwargs.get("model", "").lower()
    assert "opus" not in call_kwargs.get("model", "").lower()


@patch("src.agents.earnings_sentiment.NewsCollector")
@patch("src.agents.earnings_sentiment.anthropic.Anthropic")
def test_no_thinking_parameter(mock_cls, mock_news_cls):
    """Extended thinking must NOT be sent to Haiku (unsupported + expensive)."""
    mock_news_cls.return_value.fetch_company_news.return_value = MOCK_NEWS_ITEMS
    mock_cls.return_value.messages.create.return_value = _resp(_HOLD)

    EarningsSentimentAgent().generate_signal(_state(), {})

    call_kwargs = mock_cls.return_value.messages.create.call_args.kwargs
    assert "thinking" not in call_kwargs
