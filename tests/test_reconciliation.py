"""
Cycle de vie des ordres et réconciliation broker (src/execution/reconciliation.py).

Avant le 2026-08-02, l'exécution plaçait les ordres, dormait 10 secondes,
journalisait le statut trouvé là, et n'y revenait jamais. Les 3 lignes réelles
d'executions.csv sont toutes en `PendingSubmit` : le système croyait détenir
des positions qu'il n'avait pas.

Ces tests couvrent la logique pure. L'adaptateur ib_insync (polling,
annulation) vit dans le runner et n'est pas testable sans gateway — d'où
l'extraction de tout ce qui peut l'être.
"""
from __future__ import annotations

import pytest

from src.execution.reconciliation import (
    ACTIVE_STATUSES, TERMINAL_STATUSES, OrderOutcome, expected_positions,
    is_terminal, reconcile, summarize, to_execution_rows,
)


def _out(symbol="AAPL", side="BUY", requested=100, filled=100.0,
         avg=150.0, limit=150.5, signal=150.0, status="Filled", **kw):
    return OrderOutcome(
        symbol=symbol, side=side, requested_qty=requested, filled_qty=filled,
        avg_fill_price=avg, limit_price=limit, signal_price=signal,
        status=status, **kw,
    )


# ── Statuts ───────────────────────────────────────────────────────────────────

class TestStatuses:
    @pytest.mark.parametrize("st", ["Filled", "Cancelled", "ApiCancelled", "Inactive"])
    def test_terminal(self, st):
        assert is_terminal(st) is True

    @pytest.mark.parametrize("st", ["PendingSubmit", "PreSubmitted", "Submitted", "ApiPending"])
    def test_active_is_not_terminal(self, st):
        assert is_terminal(st) is False

    def test_inactive_is_terminal_even_though_ib_insync_lists_it_nowhere(self):
        """
        ib_insync ne classe 'Inactive' ni dans DoneStates ni dans ActiveStates.
        S'en remettre à sa liste ferait tourner le polling jusqu'au timeout sur
        un ordre rejeté par le broker.
        """
        assert "Inactive" in TERMINAL_STATUSES
        assert "Inactive" not in ACTIVE_STATUSES

    def test_statuses_do_not_overlap(self):
        assert not (TERMINAL_STATUSES & ACTIVE_STATUSES)


# ── Classification ────────────────────────────────────────────────────────────

class TestClassification:
    def test_full_fill(self):
        o = _out(requested=100, filled=100.0)
        assert (o.is_filled, o.is_partial, o.is_unfilled) == (True, False, False)

    def test_partial_fill(self):
        o = _out(requested=100, filled=40.0)
        assert (o.is_filled, o.is_partial, o.is_unfilled) == (False, True, False)

    def test_no_fill(self):
        o = _out(requested=100, filled=0.0, avg=0.0, status="Cancelled")
        assert (o.is_filled, o.is_partial, o.is_unfilled) == (False, False, True)

    def test_signed_qty_direction(self):
        assert _out(side="BUY", filled=50.0).signed_qty == 50.0
        assert _out(side="SELL", filled=50.0).signed_qty == -50.0


# ── Slippage ──────────────────────────────────────────────────────────────────

class TestSlippage:
    def test_buy_above_signal_is_unfavourable(self):
        assert _out(side="BUY", avg=101.0, signal=100.0).slippage_bps == pytest.approx(100.0)

    def test_sell_below_signal_is_unfavourable(self):
        assert _out(side="SELL", avg=99.0, signal=100.0).slippage_bps == pytest.approx(100.0)

    def test_favourable_fill_is_negative(self):
        assert _out(side="BUY", avg=99.0, signal=100.0).slippage_bps == pytest.approx(-100.0)

    def test_unfilled_order_has_no_slippage(self):
        assert _out(filled=0.0, avg=0.0).slippage_bps == 0.0


# ── Journalisation : uniquement ce qui s'est produit ──────────────────────────

class TestExecutionRows:
    def test_unfilled_order_is_never_logged(self):
        """
        Le bug d'origine : un ordre non rempli était écrit comme un trade, et
        _load_entry_prices reprenait son limit_price comme prix de revient pour
        le stop-loss d'une position inexistante.
        """
        rows = to_execution_rows(
            [_out(filled=0.0, avg=0.0, status="Cancelled")], "p1", "2026-08-02T12:00:00Z")
        assert rows == []

    def test_partial_fill_logs_the_quantity_obtained(self):
        rows = to_execution_rows([_out(requested=100, filled=37.0)], "p1", "t")
        assert rows[0]["qty"] == 37.0
        assert rows[0]["requested_qty"] == 100
        assert rows[0]["partial"] is True

    def test_notional_uses_the_fill_price_not_the_limit(self):
        rows = to_execution_rows(
            [_out(requested=100, filled=100.0, avg=150.0, limit=155.0)], "p1", "t")
        assert rows[0]["est_notional"] == pytest.approx(15_000.0)

    def test_every_logged_row_has_a_real_price(self):
        outs = [_out(symbol="A", filled=100.0), _out(symbol="B", filled=0.0, avg=0.0),
                _out(symbol="C", filled=10.0)]
        rows = to_execution_rows(outs, "p1", "t")
        assert {r["symbol"] for r in rows} == {"A", "C"}
        assert all(r["avg_fill_price"] > 0 for r in rows)

    def test_plan_id_and_timestamp_propagate(self):
        rows = to_execution_rows([_out()], "abc123", "2026-08-02T12:00:00Z")
        assert rows[0]["plan_id"] == "abc123"
        assert rows[0]["timestamp"] == "2026-08-02T12:00:00Z"


# ── Positions attendues ───────────────────────────────────────────────────────

class TestExpectedPositions:
    def test_buy_increases(self):
        assert expected_positions({}, [_out("AAPL", "BUY", filled=100.0)]) == {"AAPL": 100.0}

    def test_sell_decreases(self):
        got = expected_positions({"AAPL": 150.0}, [_out("AAPL", "SELL", filled=50.0)])
        assert got == {"AAPL": 100.0}

    def test_full_exit_removes_the_symbol(self):
        got = expected_positions({"AAPL": 100.0}, [_out("AAPL", "SELL", filled=100.0)])
        assert "AAPL" not in got

    def test_unfilled_order_does_not_move_the_position(self):
        before = {"AAPL": 100.0}
        assert expected_positions(before, [_out("AAPL", "BUY", filled=0.0, avg=0.0)]) == before

    def test_partial_fill_moves_by_what_was_obtained(self):
        got = expected_positions({}, [_out("AAPL", "BUY", requested=100, filled=30.0)])
        assert got == {"AAPL": 30.0}

    def test_untouched_symbols_are_preserved(self):
        got = expected_positions({"MSFT": 50.0}, [_out("AAPL", "BUY", filled=10.0)])
        assert got["MSFT"] == 50.0


# ── Réconciliation ────────────────────────────────────────────────────────────

class TestReconcile:
    def test_matching_positions_are_clean(self):
        r = reconcile({"AAPL": 100.0, "MSFT": 50.0}, {"AAPL": 100.0, "MSFT": 50.0})
        assert r.is_clean and r.n_checked == 2

    def test_quantity_mismatch_is_flagged(self):
        r = reconcile({"AAPL": 100.0}, {"AAPL": 60.0})
        assert not r.is_clean
        assert r.drifts[0].delta == pytest.approx(-40.0)

    def test_position_held_but_not_expected_is_flagged(self):
        """Ordre rempli hors fenêtre de polling, ou position ouverte à la main."""
        r = reconcile({}, {"NVDA": 25.0})
        assert not r.is_clean and r.drifts[0].symbol == "NVDA"

    def test_position_expected_but_absent_is_flagged(self):
        r = reconcile({"NVDA": 25.0}, {})
        assert not r.is_clean and r.drifts[0].delta == pytest.approx(-25.0)

    def test_fractional_noise_is_tolerated(self):
        assert reconcile({"AAPL": 100.0}, {"AAPL": 100.2}).is_clean

    def test_tolerance_is_configurable(self):
        assert reconcile({"AAPL": 100.0}, {"AAPL": 103.0}, tolerance=5.0).is_clean

    def test_report_tells_the_operator_to_stop(self):
        txt = reconcile({"AAPL": 100.0}, {"AAPL": 60.0}).render()
        assert "divergent" in txt and "AAPL" in txt

    def test_end_to_end_partial_fill_reconciles(self):
        """Chaîne complète : positions initiales → remplissage partiel → contrôle."""
        before = {"AAPL": 200.0}
        outcomes = [_out("AAPL", "BUY", requested=100, filled=40.0)]
        exp = expected_positions(before, outcomes)
        assert exp == {"AAPL": 240.0}
        assert reconcile(exp, {"AAPL": 240.0}).is_clean
        assert not reconcile(exp, {"AAPL": 300.0}).is_clean


# ── Résumé ────────────────────────────────────────────────────────────────────

class TestSummary:
    def test_counts_each_category(self):
        txt = summarize([
            _out("A", filled=100.0),
            _out("B", requested=100, filled=30.0),
            _out("C", filled=0.0, avg=0.0, status="Cancelled"),
        ])
        assert "1 rempli(s)" in txt and "1 partiel(s)" in txt and "1 non rempli(s)" in txt

    def test_reports_average_slippage_on_executed_orders_only(self):
        txt = summarize([_out("A", side="BUY", avg=101.0, signal=100.0),
                         _out("B", filled=0.0, avg=0.0)])
        assert "+100.0 bps" in txt

    def test_empty_run_says_so(self):
        assert "Aucun ordre" in summarize([])


# ── Prix de revient du stop-loss ──────────────────────────────────────────────

class TestEntryPricesIgnorePhantomFills:
    """
    `_load_entry_prices` reprenait le `limit_price` quand aucun prix de
    remplissage n'était disponible. Constaté en production : les trois lignes
    d'executions.csv étaient en 'PendingSubmit', et le moteur de stop-loss
    raisonnait sur un prix de revient AAPL de 255,79 $ issu d'un ordre jamais
    exécuté.
    """

    def _write(self, tmp_path, rows, header):
        p = tmp_path / "executions.csv"
        p.write_text(header + "\n" + "\n".join(rows) + "\n")
        return p

    def test_unfilled_order_gives_no_entry_price(self, tmp_path):
        from src.arena.runner import _load_entry_prices
        ex = self._write(
            tmp_path,
            ["2026-01-26T20:26:08+00:00,AAPL,BUY,391,255.79,0.0,PendingSubmit"],
            "timestamp,symbol,side,qty,limit_price,avg_fill_price,status",
        )
        got = _load_entry_prices(exec_path=ex, plan_prices_path=tmp_path / "none.json")
        assert got == {}, "un ordre non rempli ne porte aucun prix de revient"

    def test_real_fill_is_used(self, tmp_path):
        from src.arena.runner import _load_entry_prices
        ex = self._write(
            tmp_path,
            ["2026-01-26T20:26:08+00:00,AAPL,BUY,391,255.79,254.12,Filled"],
            "timestamp,symbol,side,qty,limit_price,avg_fill_price,status",
        )
        got = _load_entry_prices(exec_path=ex, plan_prices_path=tmp_path / "none.json")
        assert got == {"AAPL": pytest.approx(254.12)}

    def test_legacy_format_without_fill_price_is_ignored(self, tmp_path):
        """Ancien schéma sans colonne avg_fill_price : aucun prix n'y est fiable."""
        from src.arena.runner import _load_entry_prices
        ex = self._write(
            tmp_path,
            ["2026-01-26T20:26:08+00:00,AAPL,BUY,391,255.79,PendingSubmit"],
            "timestamp,symbol,side,qty,limit_price,status",
        )
        got = _load_entry_prices(exec_path=ex, plan_prices_path=tmp_path / "none.json")
        assert got == {}

    def test_plan_price_still_serves_as_fallback(self, tmp_path):
        """Le prix de plan (tier 3) reste la référence quand aucun fill n'existe."""
        import json
        from src.arena.runner import _load_entry_prices
        plan = tmp_path / "entry_prices.json"
        plan.write_text(json.dumps({"AAPL": 330.31}))
        ex = self._write(
            tmp_path,
            ["2026-01-26T20:26:08+00:00,AAPL,BUY,391,255.79,0.0,Cancelled"],
            "timestamp,symbol,side,qty,limit_price,avg_fill_price,status",
        )
        got = _load_entry_prices(exec_path=ex, plan_prices_path=plan)
        assert got == {"AAPL": pytest.approx(330.31)}
