"""
Tests for the DivArb tracker timing fix.

Verifies that:
1. generate_signal() does NOT write to the tracker — only signal meta is populated.
2. When RiskManager rejects a DivArb BUY, no phantom tracker entry is created.
3. No SELL phantom is generated on the next run because of a rejected BUY.
4. When RiskManager APPROVES a DivArb BUY, the tracker IS written (via runner logic).
"""
from __future__ import annotations

import json
from datetime import date, timedelta
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from src.agents.dividend_arbitrage_agent import (
    DividendArbitrageAgent,
    DividendPositionTracker,
    DivPosition,
)
from src.agents.base import MarketState
from tests.conftest import bdate_index


# ── Helpers ──────────────────────────────────────────────────────────────────

def _tracker(tmp_path: Path) -> DividendPositionTracker:
    """Return a tracker backed by a temp file."""
    t = DividendPositionTracker.__new__(DividendPositionTracker)
    import threading
    t._lock = threading.Lock()
    t._POSITIONS_PATH = tmp_path / "dividend_positions.json"
    t._CLOSED_PATH    = tmp_path / "dividend_positions_closed.json"
    t._positions = {}
    return t


def _fake_div_info(ticker: str = "MSFT", days_to_ex: int = 3):
    from src.agents.dividend_arbitrage_agent import DividendInfo
    ex = date.today() + timedelta(days=days_to_ex)
    return DividendInfo(
        ticker          = ticker,
        ex_date         = ex,
        dividend_amount = 0.75,
        div_regularity  = 1.0,
        yield_annualized= 0.012,
        is_special      = False,
        days_to_ex      = days_to_ex,
        payment_date    = ex + timedelta(days=14),
        record_date     = None,
        current_price   = 200.0,
    )


# ── Core invariant: generate_signal does NOT write to tracker ────────────────

class TestGenerateSignalDoesNotWriteTracker:

    def test_buy_signal_leaves_tracker_empty(self, tmp_path):
        """BUY signal must NOT write to the tracker — that's runner.py's job."""
        agent = DividendArbitrageAgent()
        agent._tracker = _tracker(tmp_path)

        state = MarketState(symbol="MSFT", price=200.0,
                            timestamp=date.today().isoformat())

        import pandas as pd
        import numpy as np
        idx = bdate_index(100)
        df  = pd.DataFrame({"Close": np.linspace(400, 420, 100),
                            "Volume": [1e6]*100}, index=idx)

        with patch.object(agent, "_get_div_info", return_value=_fake_div_info("MSFT", 3)), \
             patch.object(agent, "_get_pricing_model") as mock_model:
            mock_model.return_value.net_edge.return_value = (5.0, {
                "net_edge": 5.0, "carry_cost": 1.0, "price_risk_95pct": 2.0,
                "transaction_costs": 0.5, "net_dividend": 8.5,
            })
            mock_model.return_value.check_put_call_parity.return_value = False

            sig = agent.generate_signal(state, portfolio={}, regime="bull", data=df)

        assert sig.action == "BUY", f"Expected BUY, got {sig.action}: {sig.reason}"
        # Tracker must be EMPTY — no write at signal time
        assert agent._tracker._positions == {}, \
            "generate_signal() must not write to tracker (pre-approval)"

    def test_buy_signal_meta_has_divpos_fields(self, tmp_path):
        """BUY signal meta must carry the position data for runner.py to use."""
        agent = DividendArbitrageAgent()
        agent._tracker = _tracker(tmp_path)

        state = MarketState(symbol="MSFT", price=200.0,
                            timestamp=date.today().isoformat())
        import pandas as pd, numpy as np
        idx = bdate_index(100)
        df  = pd.DataFrame({"Close": np.linspace(400, 420, 100),
                            "Volume": [1e6]*100}, index=idx)

        with patch.object(agent, "_get_div_info", return_value=_fake_div_info("MSFT", 3)), \
             patch.object(agent, "_get_pricing_model") as mock_model:
            mock_model.return_value.net_edge.return_value = (5.0, {
                "net_edge": 5.0, "carry_cost": 1.0, "price_risk_95pct": 2.0,
                "transaction_costs": 0.5, "net_dividend": 8.5,
            })
            mock_model.return_value.check_put_call_parity.return_value = False

            sig = agent.generate_signal(state, portfolio={}, regime="bull", data=df)

        assert "ex_date" in sig.meta
        assert "dividend_amount" in sig.meta
        assert "_divpos_target_exit" in sig.meta, "runner needs target_exit_date"
        assert "_divpos_shares" in sig.meta,       "runner needs shares for DivPosition"


# ── Rejection scenario: no phantom entry, no phantom SELL ────────────────────

class TestRejectionLeavesNoPhantom:

    def _make_agent_with_open_signal(self, tmp_path) -> tuple:
        """Returns (agent, signal) for a valid MSFT DivArb BUY."""
        agent = DividendArbitrageAgent()
        agent._tracker = _tracker(tmp_path)

        state = MarketState(symbol="MSFT", price=200.0,
                            timestamp=date.today().isoformat())
        import pandas as pd, numpy as np
        idx = bdate_index(100)
        df  = pd.DataFrame({"Close": np.linspace(400, 420, 100),
                            "Volume": [1e6]*100}, index=idx)

        with patch.object(agent, "_get_div_info", return_value=_fake_div_info("MSFT", 3)), \
             patch.object(agent, "_get_pricing_model") as mock_model:
            mock_model.return_value.net_edge.return_value = (5.0, {
                "net_edge": 5.0, "carry_cost": 1.0, "price_risk_95pct": 2.0,
                "transaction_costs": 0.5, "net_dividend": 8.5,
            })
            mock_model.return_value.check_put_call_parity.return_value = False
            sig = agent.generate_signal(state, portfolio={}, regime="bull", data=df)

        return agent, sig

    def test_rejected_buy_leaves_no_tracker_entry(self, tmp_path):
        """RiskManager rejects → no open_position() called → tracker stays empty."""
        agent, sig = self._make_agent_with_open_signal(tmp_path)
        # Simulate RiskManager rejection: runner.py never calls tracker.open_position()
        # because _pending_divArb[sym] is only committed for approved plans.
        # So tracker must still be empty.
        assert sig.action == "BUY"
        assert "MSFT" not in agent._tracker._positions, \
            "Rejected plan must leave no tracker entry"

    def test_no_phantom_sell_on_next_run_after_rejection(self, tmp_path):
        """
        Run 1: BUY signal generated, RiskManager rejects (tracker not written).
        Run 2: MSFT NOT in portfolio → agent must return HOLD, not phantom SELL.
        """
        agent, _ = self._make_agent_with_open_signal(tmp_path)
        # Run 1 rejected — tracker is empty (confirmed above)

        # Run 2: next day, MSFT not in portfolio (never bought)
        state2 = MarketState(symbol="MSFT", price=419.0,
                             timestamp=(date.today() + timedelta(days=1)).isoformat())
        import pandas as pd, numpy as np
        idx = bdate_index(100, date.today() + timedelta(days=1))
        df2 = pd.DataFrame({"Close": np.linspace(400, 419, 100),
                            "Volume": [1e6]*100}, index=idx)

        with patch.object(agent, "_get_div_info", return_value=_fake_div_info("MSFT", 2)):
            sig2 = agent.generate_signal(state2, portfolio={}, regime="bull", data=df2)

        assert sig2.action != "SELL", \
            f"Phantom SELL detected! Signal: {sig2.action} — {sig2.reason}"


# ── Approval scenario: tracker IS written ────────────────────────────────────

class TestApprovedBuyWritesTracker:

    def test_approved_plan_writes_tracker_entry(self, tmp_path):
        """
        Simulates what runner.py does after RiskManager approves a DivArb BUY:
        DivPosition is created from the signal meta and written to the tracker.
        """
        from src.agents.dividend_arbitrage_agent import DividendPositionTracker, DivPosition
        tracker = _tracker(tmp_path)

        # Simulate signal meta that would come from generate_signal()
        ex = (date.today() + timedelta(days=3)).isoformat()
        target_exit = (date.today() + timedelta(days=4)).isoformat()
        meta = {
            "ex_date": ex,
            "dividend_amount": 0.75,
            "_divpos_target_exit": target_exit,
            "_divpos_shares": 50.0,
        }

        # This is what runner.py does for approved plans:
        tracker.open_position(DivPosition(
            ticker           = "MSFT",
            entry_date       = date.today().isoformat(),
            entry_price      = 200.0,
            ex_date          = meta["ex_date"],
            dividend_amount  = meta["dividend_amount"],
            target_exit_date = meta["_divpos_target_exit"],
            shares           = 48.0,  # from plan.delta_qty
        ))

        assert "MSFT" in tracker._positions, "Approved plan must create tracker entry"
        pos = tracker._positions["MSFT"]
        assert pos.dividend_amount == pytest.approx(0.75)
        assert pos.shares == pytest.approx(48.0)

    def test_tracker_file_written_on_approval(self, tmp_path):
        """Approved plan writes to the JSON file on disk."""
        tracker = _tracker(tmp_path)
        ex = (date.today() + timedelta(days=3)).isoformat()
        tracker.open_position(DivPosition(
            ticker="MSFT", entry_date=date.today().isoformat(),
            entry_price=200.0, ex_date=ex, dividend_amount=0.75,
            target_exit_date=ex, shares=48.0,
        ))
        assert tracker._POSITIONS_PATH.exists(), "Tracker JSON file must be created on approval"
        data = json.loads(tracker._POSITIONS_PATH.read_text())
        assert "MSFT" in data
