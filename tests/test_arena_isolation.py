"""
Isolation des pannes d'agent (src/arena/arena.py).

Avant le 2026-08-02, `Arena.run()` appelait `agent.generate_signal()` sans
garde. Trois des douze agents dépendent du réseau (API Anthropic, SEC EDGAR,
yfinance) : une exception remontait jusqu'à la boucle principale du runner et
annulait la journée de trading entière, sur tous les symboles.

Deux exigences, la seconde plus importante que la première :
  1. une panne ne doit pas interrompre les autres agents ;
  2. une panne ne doit JAMAIS être silencieuse — un agent muet pendant un mois
     laisse le système produire des décisions d'apparence normale sur une
     information amputée.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.agents.base import AgentSignal, BaseAgent
from src.arena.arena import Arena


def _df(n: int = 300) -> pd.DataFrame:
    idx = pd.date_range("2024-01-01", periods=n, freq="B")
    close = np.linspace(100, 130, n)
    return pd.DataFrame(
        {"Open": close, "High": close * 1.01, "Low": close * 0.99,
         "Close": close, "Volume": np.full(n, 1e6)},
        index=idx,
    )


class OkAgent(BaseAgent):
    def __init__(self, name="OkAgent", action="BUY", conf=0.7):
        self.name, self._action, self._conf = name, action, conf

    def generate_signal(self, state, portfolio, regime=None, data=None):
        return AgentSignal(agent_name=self.name, symbol=state.symbol,
                           action=self._action, confidence=self._conf,
                           target_weight=0.10, reason="ok")


class BoomAgent(BaseAgent):
    """Simule un agent réseau qui tombe (timeout API, EDGAR indisponible…)."""
    name = "BoomAgent"

    def __init__(self, exc=None):
        self._exc = exc or TimeoutError("anthropic: request timed out")

    def generate_signal(self, state, portfolio, regime=None, data=None):
        raise self._exc


class NoneAgent(BaseAgent):
    """Agent qui ne lève pas mais ne retourne rien — panne plus sournoise."""
    name = "NoneAgent"

    def generate_signal(self, state, portfolio, regime=None, data=None):
        return None


# ── 1. La panne n'interrompt rien ─────────────────────────────────────────────

class TestFailureDoesNotPropagate:
    def test_one_broken_agent_does_not_kill_the_run(self):
        arena = Arena([OkAgent("A"), BoomAgent(), OkAgent("B")])
        signals = arena.run("AAPL", _df())
        assert [s.agent_name for s in signals] == ["A", "B"]

    def test_agents_after_the_failure_still_run(self):
        """Régression : l'exception coupait la boucle, pas seulement l'agent."""
        arena = Arena([BoomAgent(), OkAgent("A"), OkAgent("B"), OkAgent("C")])
        assert len(arena.run("AAPL", _df())) == 3

    @pytest.mark.parametrize("exc", [
        TimeoutError("api timeout"),
        ConnectionError("EDGAR unreachable"),
        KeyError("Close"),
        ValueError("not enough data"),
        ZeroDivisionError("division by zero"),
    ])
    def test_any_exception_type_is_contained(self, exc):
        arena = Arena([BoomAgent(exc), OkAgent("A")])
        assert len(arena.run("AAPL", _df())) == 1

    def test_agent_returning_none_is_treated_as_a_failure(self):
        arena = Arena([NoneAgent(), OkAgent("A")])
        signals = arena.run("AAPL", _df())
        assert len(signals) == 1
        assert len(arena.failures) == 1
        assert "None" in arena.failures[0].error


# ── 2. La panne est visible ───────────────────────────────────────────────────

class TestFailureIsLoud:
    def test_failure_is_recorded_with_agent_symbol_and_error(self):
        arena = Arena([BoomAgent(), OkAgent("A")])
        arena.run("NVDA", _df())
        f = arena.failures[0]
        assert f.agent_name == "BoomAgent"
        assert f.symbol == "NVDA"
        assert "TimeoutError" in f.error
        assert "timed out" in f.error

    def test_traceback_is_kept_for_diagnosis(self):
        arena = Arena([BoomAgent()])
        arena.run("AAPL", _df())
        assert "generate_signal" in arena.failures[0].traceback

    def test_failures_accumulate_across_symbols(self):
        arena = Arena([BoomAgent(), OkAgent("A")])
        for sym in ["AAPL", "MSFT", "NVDA"]:
            arena.run(sym, _df())
        assert len(arena.failures) == 3
        assert {f.symbol for f in arena.failures} == {"AAPL", "MSFT", "NVDA"}

    def test_summary_groups_one_line_per_distinct_failure(self):
        """14 symboles touchés par la même panne = 1 ligne, pas 14."""
        arena = Arena([BoomAgent(), OkAgent("A")])
        for i in range(14):
            arena.run(f"S{i}", _df())
        summary = arena.failure_summary()
        assert len(summary) == 1
        assert "BoomAgent" in summary[0]
        assert "14 symbole(s)" in summary[0]

    def test_a_healthy_arena_reports_nothing(self):
        arena = Arena([OkAgent("A"), OkAgent("B")])
        arena.run("AAPL", _df())
        assert arena.failures == []
        assert arena.failure_summary() == []


# ── 3. Refus d'arbitrer quand trop peu d'agents répondent ─────────────────────

class TestDegradedArena:
    def test_majority_down_is_flagged_degraded(self):
        arena = Arena([BoomAgent(), BoomAgent(), BoomAgent(), OkAgent("A")])
        assert arena.is_degraded(arena.run("AAPL", _df())) is True

    def test_minority_down_is_not_degraded(self):
        arena = Arena([OkAgent("A"), OkAgent("B"), OkAgent("C"), BoomAgent()])
        assert arena.is_degraded(arena.run("AAPL", _df())) is False

    @pytest.mark.parametrize("n,expected", [
        (1, 1), (2, 1), (3, 2), (4, 2), (5, 3), (7, 4), (11, 6), (12, 6), (13, 7),
    ])
    def test_threshold_rounds_up_on_odd_rosters(self, n, expected):
        """
        Régression : le seuil était calculé avec int(), qui tronque. Sur
        5 agents il tombait à 2, donc une arène amputée de 3 agents sur 5 se
        déclarait saine. Le défaut était invisible sur un effectif pair — le
        premier test de seuil utilisait 12 agents et passait.
        """
        arena = Arena([OkAgent(f"A{i}") for i in range(n)])
        assert arena.min_healthy_agents == expected

    def test_three_of_five_down_is_degraded(self):
        """Le cas concret qui a révélé le bug, sur le pipeline réel."""
        arena = Arena([OkAgent("A"), OkAgent("B"),
                       BoomAgent(), BoomAgent(), BoomAgent()])
        signals = arena.run("AAPL", _df())
        assert len(signals) == 2
        assert arena.is_degraded(signals) is True

    def test_threshold_is_configurable(self):
        arena = Arena([OkAgent(f"A{i}") for i in range(12)], min_healthy_ratio=0.75)
        assert arena.min_healthy_agents == 9

    def test_threshold_never_drops_below_one(self):
        assert Arena([OkAgent("A")], min_healthy_ratio=0.0).min_healthy_agents == 1

    def test_total_wipeout_is_degraded_not_a_crash(self):
        arena = Arena([BoomAgent(), BoomAgent()])
        signals = arena.run("AAPL", _df())
        assert signals == []
        assert arena.is_degraded(signals) is True


# ── 4. Un agent en panne ne devient pas un avis ───────────────────────────────

class TestFailedAgentDoesNotVote:
    def test_no_substitute_hold_is_injected(self):
        """
        Remplacer une panne par un HOLD de confiance nulle transformerait
        l'indisponibilité en opinion, et ferait entrer l'agent dans le quorum
        de corroboration P0(c). L'agent doit être absent, pas neutre.
        """
        arena = Arena([BoomAgent(), OkAgent("A")])
        signals = arena.run("AAPL", _df())
        assert all(s.agent_name != "BoomAgent" for s in signals)
        assert len(signals) == 1

    def test_failed_agent_cannot_reach_the_selector(self):
        from src.arena.selector import select_best
        arena = Arena([BoomAgent(), OkAgent("A", action="BUY", conf=0.8)])
        best = select_best(arena.run("AAPL", _df()))
        assert best is not None and best.agent_name == "A"
