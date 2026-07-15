"""
Tests for RiskAssistant + build_context().

All Claude API calls are mocked — no network access.
"""
from __future__ import annotations

import csv
import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from src.risk.assistant import (
    RiskAssistant,
    PortfolioContext,
    build_context,
    _build_context_block,
)


# ── Helpers ───────────────────────────────────────────────────────────────────

def _mock_client(answer: str = "Votre VaR est acceptable.") -> MagicMock:
    """Return a mock Anthropic client whose messages.create returns `answer`."""
    client = MagicMock()
    msg = MagicMock()
    msg.content = [MagicMock(text=answer)]
    client.messages.create.return_value = msg
    return client


def _write_cb(logs: Path, drawdown: float = 0.05, triggered: bool = False, level: int = 1):
    (logs / "circuit_breaker.json").write_text(json.dumps({
        "drawdown": drawdown, "triggered": triggered, "level": level,
        "peak_netliq": 100_000, "triggered_at": None,
    }))


def _write_var(logs: Path, var_99_pct: float = 0.018):
    (logs / "var_latest.json").write_text(json.dumps({
        "var_95_pct": var_99_pct * 0.7,
        "var_99_pct": var_99_pct,
        "var_95_usd": 1200.0,
        "var_99_usd": 1800.0,
        "cvar_99_pct": var_99_pct * 1.3,
        "cvar_99_usd": 2100.0,
        "portfolio_value": 100_000.0,
        "n_days": 252,
        "n_positions": 3,
        "computed_at": "2026-07-15T09:30:00+00:00",
    }))


def _write_decisions(logs: Path):
    path = logs / "decisions.csv"
    rows = [
        {"ts": "2026-07-15T09:30:00", "symbol": "AAPL", "regime": "bull",
         "winner_agent": "BuffettAgent", "action": "BUY", "confidence": "0.78",
         "reason": "Buffett: strong fundamentals"},
        {"ts": "2026-07-15T09:30:00", "symbol": "SPY", "regime": "bull",
         "winner_agent": "MacroAgent", "action": "HOLD", "confidence": "0.50",
         "reason": "MacroAgent: risk_on < threshold"},
    ]
    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)


def _write_allocator(logs: Path, netliq: float = 100_000.0):
    (logs / "allocator_cache.json").write_text(json.dumps({"netliq": netliq}))


# ── build_context tests ───────────────────────────────────────────────────────

class TestBuildContext:

    def test_empty_logs_returns_default_context(self, tmp_path):
        ctx = build_context(tmp_path)
        assert isinstance(ctx, PortfolioContext)
        assert ctx.netliq == 0.0
        assert ctx.cb_state == {}
        assert ctx.var_data == {}
        assert ctx.recent_signals == []

    def test_reads_circuit_breaker(self, tmp_path):
        _write_cb(tmp_path, drawdown=0.08, triggered=True, level=3)
        ctx = build_context(tmp_path)
        assert ctx.cb_state["drawdown"] == pytest.approx(0.08)
        assert ctx.cb_state["triggered"] is True
        assert ctx.cb_state["level"] == 3

    def test_reads_var(self, tmp_path):
        _write_var(tmp_path, var_99_pct=0.025)
        ctx = build_context(tmp_path)
        assert ctx.var_data["var_99_pct"] == pytest.approx(0.025)
        assert ctx.var_data["n_positions"] == 3

    def test_reads_signals_last_per_ticker(self, tmp_path):
        _write_decisions(tmp_path)
        ctx = build_context(tmp_path)
        symbols = {s["symbol"] for s in ctx.recent_signals}
        assert "AAPL" in symbols
        assert "SPY"  in symbols

    def test_reads_regime_from_decisions(self, tmp_path):
        _write_decisions(tmp_path)
        ctx = build_context(tmp_path)
        assert ctx.regime == "bull"

    def test_reads_netliq_from_allocator(self, tmp_path):
        _write_allocator(tmp_path, netliq=123_456.0)
        ctx = build_context(tmp_path)
        assert ctx.netliq == pytest.approx(123_456.0)

    def test_corrupted_json_skipped(self, tmp_path):
        (tmp_path / "circuit_breaker.json").write_text("INVALID JSON{{{{")
        ctx = build_context(tmp_path)
        assert ctx.cb_state == {}   # not crashed, just empty

    def test_missing_files_ok(self, tmp_path):
        ctx = build_context(tmp_path)
        assert ctx.regime == "unknown"


# ── _build_context_block tests ────────────────────────────────────────────────

class TestBuildContextBlock:

    def test_contains_regime(self):
        ctx = PortfolioContext(regime="bull")
        block = _build_context_block(ctx)
        assert "BULL" in block

    def test_contains_cb_drawdown(self):
        ctx = PortfolioContext(cb_state={"drawdown": 0.07, "triggered": False, "level": 1})
        block = _build_context_block(ctx)
        assert "7.0%" in block or "7%" in block.replace("0.07", "7%")
        assert "niveau 1" in block

    def test_sell_only_flagged(self):
        ctx = PortfolioContext(cb_state={"drawdown": 0.20, "triggered": True, "level": 3})
        block = _build_context_block(ctx)
        assert "SELL-ONLY" in block

    def test_var_present(self):
        ctx = PortfolioContext(var_data={
            "var_95_pct": 0.015, "var_95_usd": 1500,
            "var_99_pct": 0.022, "var_99_usd": 2200,
            "cvar_99_usd": 2800, "n_positions": 4, "n_days": 252,
        })
        block = _build_context_block(ctx)
        assert "2.20%" in block or "2.2" in block
        assert "4 positions" in block

    def test_signals_listed(self):
        ctx = PortfolioContext(recent_signals=[
            {"symbol": "NVDA", "winner_agent": "CitadelAgent",
             "action": "BUY", "confidence": "0.82", "reason": "momentum"},
        ])
        block = _build_context_block(ctx)
        assert "NVDA" in block
        assert "BUY"  in block
        assert "CitadelAgent" in block

    def test_no_var_message(self):
        ctx = PortfolioContext()
        block = _build_context_block(ctx)
        assert "absentes" in block.lower() or "var" in block.lower()


# ── RiskAssistant.ask tests ───────────────────────────────────────────────────

class TestRiskAssistantAsk:

    def test_returns_claude_response(self):
        client = _mock_client("Le risque est faible.")
        assistant = RiskAssistant(client=client)
        ctx = PortfolioContext(regime="bull")
        answer = assistant.ask("Quel est le risque actuel ?", ctx)
        assert answer == "Le risque est faible."

    def test_client_called_once(self):
        client = _mock_client()
        assistant = RiskAssistant(client=client)
        assistant.ask("Question test", PortfolioContext())
        assert client.messages.create.call_count == 1

    def test_system_prompt_contains_persona(self):
        client = _mock_client()
        assistant = RiskAssistant(client=client)
        assistant.ask("Question ?", PortfolioContext())
        call_kwargs = client.messages.create.call_args[1]
        assert "Milan" in call_kwargs["system"]
        assert "français" in call_kwargs["system"]

    def test_user_message_contains_question(self):
        client = _mock_client()
        assistant = RiskAssistant(client=client)
        assistant.ask("Pourquoi AAPL est HOLD ?", PortfolioContext())
        call_kwargs = client.messages.create.call_args[1]
        messages = call_kwargs["messages"]
        combined = " ".join(m["content"] for m in messages)
        assert "Pourquoi AAPL est HOLD" in combined

    def test_user_message_contains_context(self):
        client = _mock_client()
        ctx = PortfolioContext(regime="bear", cb_state={"drawdown": 0.12, "triggered": False, "level": 2})
        assistant = RiskAssistant(client=client)
        assistant.ask("Status ?", ctx)
        call_kwargs = client.messages.create.call_args[1]
        combined = " ".join(m["content"] for m in call_kwargs["messages"])
        assert "BEAR" in combined
        assert "niveau 2" in combined

    def test_empty_question_no_api_call(self):
        client = _mock_client()
        assistant = RiskAssistant(client=client)
        answer = assistant.ask("", PortfolioContext())
        client.messages.create.assert_not_called()
        assert "vide" in answer.lower() or "question" in answer.lower()

    def test_whitespace_question_no_api_call(self):
        client = _mock_client()
        assistant = RiskAssistant(client=client)
        answer = assistant.ask("   \n  ", PortfolioContext())
        client.messages.create.assert_not_called()

    def test_api_error_returns_error_message(self):
        client = MagicMock()
        client.messages.create.side_effect = Exception("rate limit")
        assistant = RiskAssistant(client=client)
        answer = assistant.ask("Question ?", PortfolioContext())
        assert "rate limit" in answer.lower() or "erreur" in answer.lower()

    def test_correct_model_used(self):
        from src.risk.assistant import _MODEL
        client = _mock_client()
        assistant = RiskAssistant(client=client)
        assistant.ask("Test ?", PortfolioContext())
        call_kwargs = client.messages.create.call_args[1]
        assert call_kwargs["model"] == _MODEL

    def test_full_context_in_prompt(self, tmp_path):
        """build_context + ask integration: real files → correct prompt."""
        _write_cb(tmp_path, drawdown=0.06, level=1)
        _write_var(tmp_path, var_99_pct=0.019)
        _write_decisions(tmp_path)
        _write_allocator(tmp_path, netliq=95_000.0)

        ctx    = build_context(tmp_path)
        client = _mock_client("Tout va bien.")
        answer = RiskAssistant(client=client).ask("Résume le portfolio.", ctx)
        assert answer == "Tout va bien."

        call_kwargs = client.messages.create.call_args[1]
        combined = " ".join(m["content"] for m in call_kwargs["messages"])
        assert "95" in combined          # netliq 95000
        assert "AAPL" in combined        # from signals
        assert "niveau 1" in combined    # CB level
