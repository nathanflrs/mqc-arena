"""
Tests for the 3-tier stop-loss entry price mechanism.

The stop-loss must fire even when executions.csv is empty or absent —
using IBKR avgCost and/or the plan entry price as fallbacks.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from src.arena.runner import _load_entry_prices, _write_plan_entry_prices
from src.execution.planner import OrderPlan


# ── helpers ──────────────────────────────────────────────────────────────────

def _buy(symbol: str, price: float, target_qty: float = 100.0) -> OrderPlan:
    return OrderPlan(
        symbol=symbol, action="BUY",
        target_weight=0.10, last_price=price,
        current_qty=0.0, target_qty=target_qty,
        delta_qty=target_qty, est_notional=price * target_qty,
        reason="test BUY",
    )


def _sell_close(symbol: str, price: float, qty: float = 100.0) -> OrderPlan:
    return OrderPlan(
        symbol=symbol, action="SELL",
        target_weight=0.0, last_price=price,
        current_qty=qty, target_qty=0.0,
        delta_qty=-qty, est_notional=price * qty,
        reason="test SELL close",
    )


def _sell_partial(symbol: str, price: float) -> OrderPlan:
    return OrderPlan(
        symbol=symbol, action="SELL",
        target_weight=0.05, last_price=price,
        current_qty=100.0, target_qty=50.0,   # partial — target_qty != 0
        delta_qty=-50.0, est_notional=price * 50,
        reason="test SELL partial",
    )


def _executions_csv(tmp_path: Path, rows: list[dict]) -> Path:
    p = tmp_path / "executions.csv"
    if rows:
        import csv
        headers = list(rows[0].keys())
        with open(p, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=headers)
            w.writeheader()
            w.writerows(rows)
    else:
        p.write_text("")
    return p


# ── _load_entry_prices ───────────────────────────────────────────────────────

class TestLoadEntryPrices:

    def test_empty_all_sources_returns_empty(self, tmp_path):
        result = _load_entry_prices(
            plan_prices_path=tmp_path / "nonexistent.json",
            exec_path=tmp_path / "nonexistent.csv",
        )
        assert result == {}

    def test_reads_plan_prices_json(self, tmp_path):
        """Tier 3: plan price is returned when no other source exists."""
        p = tmp_path / "ep.json"
        p.write_text(json.dumps({"AAPL": 190.0, "NVDA": 450.0}))

        result = _load_entry_prices(
            plan_prices_path=p,
            exec_path=tmp_path / "nonexistent.csv",
        )
        assert result["AAPL"] == pytest.approx(190.0)
        assert result["NVDA"] == pytest.approx(450.0)

    def test_avg_costs_overrides_plan_price(self, tmp_path):
        """Tier 2: IBKR avgCost wins over plan price."""
        p = tmp_path / "ep.json"
        p.write_text(json.dumps({"AAPL": 190.0}))

        result = _load_entry_prices(
            avg_costs={"AAPL": 185.50},
            plan_prices_path=p,
            exec_path=tmp_path / "nonexistent.csv",
        )
        assert result["AAPL"] == pytest.approx(185.50)

    def test_avg_costs_zero_does_not_override(self, tmp_path):
        """avgCost = 0 is invalid — plan price should still be used."""
        p = tmp_path / "ep.json"
        p.write_text(json.dumps({"AAPL": 190.0}))

        result = _load_entry_prices(
            avg_costs={"AAPL": 0.0},
            plan_prices_path=p,
            exec_path=tmp_path / "nonexistent.csv",
        )
        assert result["AAPL"] == pytest.approx(190.0)

    def test_executions_csv_fill_overrides_both(self, tmp_path):
        """Tier 1: actual fill price wins over avgCost and plan price."""
        p = tmp_path / "ep.json"
        p.write_text(json.dumps({"AAPL": 190.0}))
        ex = _executions_csv(tmp_path, [
            {"timestamp": "2026-07-15T09:31:00Z", "symbol": "AAPL",
             "side": "BUY", "limit_price": 191.0, "avg_fill_price": 191.25},
        ])

        result = _load_entry_prices(
            avg_costs={"AAPL": 185.50},
            plan_prices_path=p,
            exec_path=ex,
        )
        assert result["AAPL"] == pytest.approx(191.25)

    def test_executions_csv_fill_zero_yields_no_entry_price(self, tmp_path):
        """
        avg_fill_price = 0 → l'ordre n'a pas été rempli, donc il n'existe pas
        de prix de revient. Le test précédent affirmait le contraire (repli sur
        limit_price), et ce contrat était faux : le stop-loss raisonnait alors
        sur une base de coût fantôme.

        Constaté en production le 2026-08-02 — les trois lignes historiques
        d'executions.csv étaient en 'PendingSubmit', et AAPL en héritait d'un
        prix d'entrée de 255,79 $ pour un ordre jamais exécuté. Un stop-loss à
        7 % sur une base fausse déclenche une vente au mauvais moment, ou ne la
        déclenche pas quand il le faudrait.
        """
        ex = _executions_csv(tmp_path, [
            {"timestamp": "2026-07-15T09:31:00Z", "symbol": "AAPL",
             "side": "BUY", "limit_price": 191.0, "avg_fill_price": 0.0},
        ])
        result = _load_entry_prices(
            exec_path=ex, plan_prices_path=tmp_path / "absent.json"
        )
        assert "AAPL" not in result

    def test_partial_fill_still_provides_an_entry_price(self, tmp_path):
        """Un remplissage partiel est un vrai trade : son prix compte."""
        ex = _executions_csv(tmp_path, [
            {"timestamp": "2026-07-15T09:31:00Z", "symbol": "AAPL",
             "side": "BUY", "limit_price": 191.0, "avg_fill_price": 190.4},
        ])
        result = _load_entry_prices(
            exec_path=ex, plan_prices_path=tmp_path / "absent.json"
        )
        assert result["AAPL"] == pytest.approx(190.4)

    def test_empty_executions_csv_falls_back_to_plan(self, tmp_path):
        """Empty executions.csv doesn't block plan-price fallback."""
        p = tmp_path / "ep.json"
        p.write_text(json.dumps({"TSLA": 350.0}))
        ex = _executions_csv(tmp_path, [])   # empty file

        result = _load_entry_prices(plan_prices_path=p, exec_path=ex)
        assert result["TSLA"] == pytest.approx(350.0)

    def test_missing_executions_csv_falls_back_to_plan(self, tmp_path):
        """Absent executions.csv also falls back gracefully."""
        p = tmp_path / "ep.json"
        p.write_text(json.dumps({"GLD": 370.0}))

        result = _load_entry_prices(
            plan_prices_path=p,
            exec_path=tmp_path / "no_file.csv",
        )
        assert result["GLD"] == pytest.approx(370.0)

    def test_stop_loss_scenario_missing_fill(self, tmp_path):
        """
        Reproduces the original bug: executions.csv is absent (no IBKR fill),
        but _write_plan_entry_prices wrote entry_prices.json at plan approval.
        Stop-loss must still fire.
        """
        p = tmp_path / "ep.json"
        p.write_text(json.dumps({"AAPL": 200.0}))

        entry_prices = _load_entry_prices(
            plan_prices_path=p,
            exec_path=tmp_path / "executions.csv",   # doesn't exist
        )
        entry_px = entry_prices.get("AAPL")
        assert entry_px is not None, "entry price must be found from plan fallback"
        current_px = 180.0  # 10% drawdown
        pnl_pct = (current_px - entry_px) / entry_px
        stop_loss_pct = 0.07
        assert pnl_pct < -stop_loss_pct, "stop-loss should trigger"


# ── _write_plan_entry_prices ─────────────────────────────────────────────────

class TestWritePlanEntryPrices:

    def test_buy_plan_creates_entry(self, tmp_path):
        p = tmp_path / "ep.json"
        _write_plan_entry_prices([_buy("AAPL", 190.0)], path=p)
        data = json.loads(p.read_text())
        assert data["AAPL"] == pytest.approx(190.0)

    def test_full_sell_removes_entry(self, tmp_path):
        p = tmp_path / "ep.json"
        p.write_text(json.dumps({"AAPL": 190.0, "NVDA": 450.0}))
        _write_plan_entry_prices([_sell_close("AAPL", 200.0)], path=p)
        data = json.loads(p.read_text())
        assert "AAPL" not in data
        assert "NVDA" in data     # other symbols untouched

    def test_partial_sell_keeps_entry(self, tmp_path):
        """Partial SELL (target_qty != 0) must not remove the entry price."""
        p = tmp_path / "ep.json"
        p.write_text(json.dumps({"AAPL": 190.0}))
        _write_plan_entry_prices([_sell_partial("AAPL", 200.0)], path=p)
        data = json.loads(p.read_text())
        assert data["AAPL"] == pytest.approx(190.0)   # preserved

    def test_scale_in_buy_does_not_overwrite_original(self, tmp_path):
        """Second BUY (scale-in) must not shift the entry reference."""
        p = tmp_path / "ep.json"
        p.write_text(json.dumps({"AAPL": 190.0}))
        _write_plan_entry_prices([_buy("AAPL", 210.0)], path=p)  # scale-in at 210
        data = json.loads(p.read_text())
        assert data["AAPL"] == pytest.approx(190.0)   # first entry preserved

    def test_fresh_buy_after_full_close(self, tmp_path):
        """After a full close, a new BUY records the new entry."""
        p = tmp_path / "ep.json"
        p.write_text(json.dumps({"AAPL": 190.0}))
        # Close position
        _write_plan_entry_prices([_sell_close("AAPL", 200.0)], path=p)
        # Re-enter
        _write_plan_entry_prices([_buy("AAPL", 205.0)], path=p)
        data = json.loads(p.read_text())
        assert data["AAPL"] == pytest.approx(205.0)

    def test_multiple_plans_mixed(self, tmp_path):
        """BUY NVDA + SELL AAPL (close) in one call."""
        p = tmp_path / "ep.json"
        p.write_text(json.dumps({"AAPL": 190.0}))
        _write_plan_entry_prices([_buy("NVDA", 450.0), _sell_close("AAPL", 195.0)], path=p)
        data = json.loads(p.read_text())
        assert data["NVDA"] == pytest.approx(450.0)
        assert "AAPL" not in data

    def test_creates_file_when_absent(self, tmp_path):
        """Creates logs/entry_prices.json if it doesn't exist yet."""
        p = tmp_path / "ep.json"
        assert not p.exists()
        _write_plan_entry_prices([_buy("GLD", 370.0)], path=p)
        assert p.exists()
        data = json.loads(p.read_text())
        assert "GLD" in data

    def test_empty_plan_list_is_noop(self, tmp_path):
        p = tmp_path / "ep.json"
        p.write_text(json.dumps({"AAPL": 190.0}))
        _write_plan_entry_prices([], path=p)
        data = json.loads(p.read_text())
        assert data["AAPL"] == pytest.approx(190.0)
