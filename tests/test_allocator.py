# tests/test_allocator.py
from __future__ import annotations

import json
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest

from src.risk.allocator import (
    AllocatorConfig,
    AllocationResult,
    DynamicAllocator,
    _rolling_sharpe,
    _weights_from_sharpes,
)


# ─── Helpers ─────────────────────────────────────────────────────────────────

def _equity(values: list[float]) -> pd.Series:
    idx = pd.date_range("2023-01-01", periods=len(values), freq="B")
    return pd.Series(values, index=idx)


def _mock_backtest_result(sharpe_value: float) -> MagicMock:
    n = 200
    base = 100_000.0
    # Simule une courbe d'equity cohérente avec le sharpe demandé
    result = MagicMock()
    result.equity_curve = _equity([base] * n)
    return result


# ─── _rolling_sharpe ─────────────────────────────────────────────────────────

def test_rolling_sharpe_flat_returns_zero():
    eq = _equity([100_000.0] * 100)
    assert _rolling_sharpe(eq, lookback=60) == 0.0


def test_rolling_sharpe_positive_trend():
    vals = [100_000 * (1.001 ** i) for i in range(300)]
    eq = _equity(vals)
    sharpe = _rolling_sharpe(eq, lookback=126)
    assert sharpe > 0


def test_rolling_sharpe_uses_only_lookback_window():
    # Courbe longue mais seule la fin est positive
    flat = [100_000.0] * 200
    upward = [100_000 * (1.001 ** i) for i in range(100)]
    eq = _equity(flat + upward)
    sharpe_full = _rolling_sharpe(eq, lookback=len(eq))
    sharpe_recent = _rolling_sharpe(eq, lookback=100)
    # Sur la fenêtre récente (toute positive), le Sharpe doit être meilleur
    assert sharpe_recent >= sharpe_full


def test_rolling_sharpe_short_series():
    eq = _equity([100_000.0] * 3)
    assert _rolling_sharpe(eq, lookback=126) == 0.0


# ─── _weights_from_sharpes ────────────────────────────────────────────────────

def test_all_negative_sharpes_return_min_weight():
    sharpes = {"A": -0.5, "B": -1.2}
    w = _weights_from_sharpes(sharpes, base_weight=0.10, min_weight=0.02, max_weight=0.25)
    assert w["A"] == 0.02
    assert w["B"] == 0.02


def test_equal_positive_sharpes_return_base_weight():
    sharpes = {"A": 1.0, "B": 1.0}
    w = _weights_from_sharpes(sharpes, base_weight=0.10, min_weight=0.02, max_weight=0.25)
    assert w["A"] == pytest.approx(0.10)
    assert w["B"] == pytest.approx(0.10)


def test_higher_sharpe_gets_higher_weight():
    sharpes = {"A": 0.5, "B": 2.0}
    w = _weights_from_sharpes(sharpes, base_weight=0.10, min_weight=0.02, max_weight=0.25)
    assert w["B"] > w["A"]


def test_weight_clamped_to_max():
    # C domine largement A et B → son poids calculé dépasse max_weight → clampé
    sharpes = {"A": 0.1, "B": 0.1, "C": 3.0}
    w = _weights_from_sharpes(sharpes, base_weight=0.10, min_weight=0.02, max_weight=0.25)
    assert w["C"] == 0.25


def test_negative_agent_gets_min_weight():
    sharpes = {"A": 1.0, "B": -0.3}
    w = _weights_from_sharpes(sharpes, base_weight=0.10, min_weight=0.02, max_weight=0.25)
    assert w["B"] == 0.02
    assert w["A"] > 0.02


# ─── AllocationResult serialisation ─────────────────────────────────────────

def test_allocation_result_round_trip():
    result = AllocationResult(
        weights={"BuffettAgent": {"AAPL": 0.12}},
        best_agent={"AAPL": "BuffettAgent"},
        sharpes={"BuffettAgent": {"AAPL": 1.2}},
        computed_at="2024-01-01T12:00:00+00:00",
    )
    restored = AllocationResult.from_dict(result.to_dict())
    assert restored.best_agent == {"AAPL": "BuffettAgent"}
    assert restored.weights["BuffettAgent"]["AAPL"] == pytest.approx(0.12)


def test_telegram_summary_contains_symbol_and_agent():
    result = AllocationResult(
        weights={"BuffettAgent": {"AAPL": 0.12}},
        best_agent={"AAPL": "BuffettAgent"},
        sharpes={"BuffettAgent": {"AAPL": 1.2}},
        computed_at="2024-01-01T12:00:00+00:00",
    )
    summary = result.telegram_summary()
    assert "AAPL" in summary
    assert "BuffettAgent" in summary
    assert "1.20" in summary


# ─── Cache TTL ────────────────────────────────────────────────────────────────

def test_cache_loaded_when_fresh(tmp_path):
    cache_file = tmp_path / "cache.json"
    from datetime import datetime, timezone
    fresh_result = AllocationResult(
        weights={"A": {"AAPL": 0.10}},
        best_agent={"AAPL": "A"},
        sharpes={"A": {"AAPL": 0.8}},
        computed_at=datetime.now(timezone.utc).isoformat(),
    )
    cache_file.write_text(json.dumps(fresh_result.to_dict()))

    cfg = AllocatorConfig(cache_ttl_hours=24.0, cache_path=str(cache_file))
    alloc = DynamicAllocator(cfg)
    loaded = alloc._load_cache()
    assert loaded is not None
    assert loaded.best_agent == {"AAPL": "A"}


def test_cache_ignored_when_stale(tmp_path):
    cache_file = tmp_path / "cache.json"
    stale_result = AllocationResult(
        weights={},
        best_agent={},
        sharpes={},
        computed_at="2020-01-01T00:00:00+00:00",
    )
    cache_file.write_text(json.dumps(stale_result.to_dict()))

    cfg = AllocatorConfig(cache_ttl_hours=24.0, cache_path=str(cache_file))
    alloc = DynamicAllocator(cfg)
    assert alloc._load_cache() is None


def test_cache_saved_after_compute(tmp_path):
    cache_file = tmp_path / "cache.json"
    cfg = AllocatorConfig(cache_ttl_hours=24.0, cache_path=str(cache_file))
    alloc = DynamicAllocator(cfg)

    mock_agent = MagicMock()
    mock_agent.name = "MockAgent"

    mock_result = MagicMock()
    mock_result.equity_curve = _equity([100_000 * (1.0005 ** i) for i in range(300)])

    with patch("src.risk.allocator.BacktestEngine") as MockEngine:
        MockEngine.return_value.run.return_value = mock_result
        data = {"AAPL": pd.DataFrame({"Close": [100.0] * 300})}
        result = alloc.compute(data, [mock_agent])

    assert cache_file.exists()
    assert result.best_agent == {"AAPL": "MockAgent"}


# ─── best_agent sélection ─────────────────────────────────────────────────────

def test_best_agent_is_highest_sharpe(tmp_path):
    cache_file = tmp_path / "cache.json"
    cfg = AllocatorConfig(cache_path=str(cache_file))
    alloc = DynamicAllocator(cfg)

    agent_a = MagicMock()
    agent_a.name = "AgentA"
    agent_b = MagicMock()
    agent_b.name = "AgentB"

    # AgentA : equity plate → Sharpe ≈ 0
    # AgentB : equity croissante → Sharpe > 0
    eq_flat = _equity([100_000.0] * 300)
    eq_up   = _equity([100_000 * (1.001 ** i) for i in range(300)])

    def fake_run(symbol, df):
        result = MagicMock()
        result.equity_curve = eq_up if symbol == "AAPL" and BacktestEngine_call_count[0] == 1 else eq_flat
        BacktestEngine_call_count[0] += 1
        return result

    BacktestEngine_call_count = [0]

    with patch("src.risk.allocator.BacktestEngine") as MockEngine:
        results_iter = iter([
            MagicMock(equity_curve=eq_flat),   # AgentA / AAPL
            MagicMock(equity_curve=eq_up),     # AgentB / AAPL
        ])
        MockEngine.return_value.run.side_effect = lambda symbol, df: next(results_iter)
        data = {"AAPL": pd.DataFrame({"Close": [100.0] * 300})}
        result = alloc.compute(data, [agent_a, agent_b])

    assert result.best_agent["AAPL"] == "AgentB"
