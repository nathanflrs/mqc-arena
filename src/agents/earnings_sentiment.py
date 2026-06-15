# src/agents/earnings_sentiment.py
from __future__ import annotations

import json
import time
from dataclasses import dataclass
from datetime import date
from typing import Dict, List, Optional

import anthropic
import yfinance as yf

from src.agents.base import BaseAgent, MarketState, AgentSignal


@dataclass
class EarningsSentimentConfig:
    target_weight: float = 0.10
    min_confidence: float = 0.55
    max_news_items: int = 5
    cache_ttl_hours: float = 6.0
    model: str = "claude-opus-4-8"


class EarningsSentimentAgent(BaseAgent):
    """
    LLM-driven alternative-data agent.

    Fetches recent news via yfinance and asks Claude to produce a
    BUY/HOLD/SELL signal. Results are cached per (symbol, calendar-date)
    with a 6-hour TTL so a full Arena run incurs at most one API call per
    symbol.

    Walk-forward insight: OHLCV rule-based strategies miss catalyst-driven
    moves (earnings beats, guidance cuts, M&A). This agent fills that gap.
    """
    name = "EarningsSentimentAgent"

    def __init__(self, config: Optional[EarningsSentimentConfig] = None):
        self.cfg = config or EarningsSentimentConfig()
        self._client: Optional[anthropic.Anthropic] = None
        self._cache: Dict[str, dict] = {}
        self._cache_ts: Dict[str, float] = {}

    def _get_client(self) -> anthropic.Anthropic:
        if self._client is None:
            self._client = anthropic.Anthropic()
        return self._client

    def _cache_key(self, symbol: str) -> str:
        return f"{symbol}_{date.today()}"

    def _is_cached(self, key: str) -> bool:
        if key not in self._cache:
            return False
        age = time.time() - self._cache_ts.get(key, 0.0)
        return age < self.cfg.cache_ttl_hours * 3600

    def _fetch_news(self, symbol: str) -> List[dict]:
        try:
            ticker = yf.Ticker(symbol)
            news = ticker.news or []
            return news[: self.cfg.max_news_items]
        except Exception:
            return []

    def _format_news(self, items: List[dict]) -> str:
        if not items:
            return "No recent news available."
        lines = []
        for i, item in enumerate(items, 1):
            title = item.get("title", "")
            publisher = item.get("publisher", "")
            summary = item.get("summary", item.get("description", ""))[:200]
            lines.append(f"{i}. [{publisher}] {title}")
            if summary:
                lines.append(f"   {summary}")
        return "\n".join(lines)

    def _extract_json(self, text: str) -> dict:
        start = text.find("{")
        end = text.rfind("}") + 1
        if start < 0 or end <= start:
            raise ValueError(f"No JSON object in response: {text[:200]!r}")
        return json.loads(text[start:end])

    def _call_claude(
        self,
        symbol: str,
        price: float,
        regime: Optional[str],
        news_text: str,
    ) -> dict:
        prompt = f"""You are a quantitative analyst at a hedge fund. Analyze these recent news items for {symbol} and produce a trading signal.

News items:
{news_text}

Market context:
- Symbol: {symbol}
- Price: ${price:.2f}
- Regime: {regime or "unknown"}

Respond with ONLY valid JSON in this exact format (no markdown, no explanation):
{{
  "action": "BUY or HOLD or SELL",
  "confidence": 0.0,
  "reason": "one-line explanation of the key catalyst",
  "sentiment_score": 0.0,
  "key_catalyst": "main event driving signal"
}}

Guidelines:
- BUY: strong positive catalyst (earnings beat, revenue beat, major launch, M&A target, analyst upgrade)
- SELL: strong negative catalyst (earnings miss, guidance cut, legal/regulatory issue, sector headwind)
- HOLD: mixed signals, no clear catalyst, or insufficient information
- confidence: 0.55–0.65 mild, 0.65–0.80 moderate, >0.80 only for unambiguous catalysts
- In bear regime prefer HOLD and reduce confidence
- sentiment_score: −1.0 (very bearish) to +1.0 (very bullish)"""

        client = self._get_client()
        response = client.messages.create(
            model=self.cfg.model,
            max_tokens=512,
            thinking={"type": "adaptive"},
            messages=[{"role": "user", "content": prompt}],
        )
        text = next(
            (b.text for b in response.content if getattr(b, "type", "") == "text" and b.text),
            "",
        )
        return self._extract_json(text)

    def generate_signal(
        self,
        state: MarketState,
        portfolio: Dict[str, float],
        regime: Optional[str] = None,
        data=None,
    ) -> AgentSignal:
        cache_key = self._cache_key(state.symbol)

        if self._is_cached(cache_key):
            result = self._cache[cache_key]
        else:
            try:
                news = self._fetch_news(state.symbol)
                news_text = self._format_news(news)
                result = self._call_claude(state.symbol, state.price, regime, news_text)
                self._cache[cache_key] = result
                self._cache_ts[cache_key] = time.time()
            except Exception as e:
                return AgentSignal(
                    agent_name=self.name,
                    symbol=state.symbol,
                    action="HOLD",
                    confidence=0.1,
                    target_weight=0.0,
                    reason=f"EarningsSentiment: API error — {type(e).__name__}",
                    meta={"error": str(e), "regime": regime},
                )

        raw_action = str(result.get("action", "HOLD")).upper()
        action = raw_action if raw_action in ("BUY", "SELL", "HOLD") else "HOLD"

        confidence = float(result.get("confidence", 0.5))
        confidence = max(0.0, min(1.0, confidence))

        if confidence < self.cfg.min_confidence:
            action = "HOLD"

        target_weight = self.cfg.target_weight if action == "BUY" else 0.0

        return AgentSignal(
            agent_name=self.name,
            symbol=state.symbol,
            action=action,
            confidence=confidence,
            target_weight=target_weight,
            reason=result.get("reason", "EarningsSentiment signal"),
            meta={
                "regime": regime,
                "sentiment_score": result.get("sentiment_score", 0.0),
                "key_catalyst": result.get("key_catalyst", ""),
            },
        )
