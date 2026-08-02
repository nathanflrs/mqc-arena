# src/arena/arena.py
from __future__ import annotations

import logging
import math
import traceback
from dataclasses import dataclass
from typing import Dict, List, Optional

import pandas as pd

from src.agents.base import BaseAgent, MarketState, AgentSignal

logger = logging.getLogger(__name__)

# Fraction minimale d'agents devant répondre pour qu'un vote garde du sens.
# En dessous, l'arène n'arbitre plus entre des avis — elle enregistre l'opinion
# des rares survivants. Voir Arena.is_degraded().
DEFAULT_MIN_HEALTHY_RATIO = 0.5


@dataclass(frozen=True)
class AgentFailure:
    """Un agent qui a levé une exception sur un symbole donné."""
    agent_name: str
    symbol: str
    error: str          # type + message, une ligne
    traceback: str      # trace complète, pour le diagnostic

    @property
    def key(self) -> tuple:
        """Identité d'une panne, pour dédupliquer les alertes d'un même run."""
        return (self.agent_name, self.error)


class Arena:
    """
    L'arène compare les signaux des agents sur un même état de marché.
    Elle ne trade PAS. Elle observe, score, et log.

    Isolation des pannes
    --------------------
    Chaque agent est appelé sous garde. Une exception l'écarte du vote pour ce
    symbole, sans interrompre les autres ni le run.

    Sur douze agents, trois dépendent du réseau (API Anthropic, SEC EDGAR,
    yfinance). Sans cette garde, un timeout un mardi matin remontait jusqu'à la
    boucle principale et annulait la journée de trading complète — sur tous les
    symboles, pas seulement celui en cours.

    La panne n'est jamais silencieuse. C'est le point essentiel : un agent muet
    pendant un mois est plus dangereux qu'un agent qui plante bruyamment, parce
    que le système continue de produire des décisions qui semblent normales avec
    une information amputée. Les pannes sont donc collectées dans `failures`,
    remontées par le runner, et `is_degraded()` permet de refuser d'arbitrer
    quand trop peu d'agents ont répondu.
    """

    def __init__(
        self,
        agents: List[BaseAgent],
        *,
        min_healthy_ratio: float = DEFAULT_MIN_HEALTHY_RATIO,
    ):
        self.agents = agents
        self.min_healthy_ratio = min_healthy_ratio
        # Pannes accumulées sur toute la durée de vie de l'instance (un run).
        self.failures: List[AgentFailure] = []

    # ── Diagnostic ────────────────────────────────────────────────────────────

    @property
    def min_healthy_agents(self) -> int:
        """
        Nombre d'agents devant répondre, arrondi **au supérieur**.

        `int()` tronquait : sur 5 agents et un ratio de 0.5, le seuil tombait à
        2, si bien qu'une arène amputée de 3 agents sur 5 se déclarait saine.
        Le défaut n'était pas visible sur un effectif pair — d'où le test
        paramétré sur effectifs pairs ET impairs.
        """
        return max(1, math.ceil(len(self.agents) * self.min_healthy_ratio))

    def is_degraded(self, signals: List[AgentSignal]) -> bool:
        """Trop peu d'agents ont répondu pour que le vote soit interprétable."""
        return len(signals) < self.min_healthy_agents

    def failure_summary(self) -> List[str]:
        """Une ligne par panne distincte, quel que soit le nombre de symboles touchés."""
        by_key: Dict[tuple, List[str]] = {}
        for f in self.failures:
            by_key.setdefault(f.key, []).append(f.symbol)
        out = []
        for (agent, error), syms in by_key.items():
            out.append(f"{agent} — {error} ({len(syms)} symbole(s) : {', '.join(syms[:5])}"
                       + ("…)" if len(syms) > 5 else ")"))
        return out

    # ── Exécution ─────────────────────────────────────────────────────────────

    def run(
        self,
        symbol: str,
        data: pd.DataFrame,
        portfolio: Dict[str, float] | None = None,
        regime: str | None = None,
    ) -> List[AgentSignal]:
        """
        Retourne les signaux des agents **ayant répondu**. Un agent en panne est
        absent de la liste : il ne vote pas, et n'entre pas non plus dans le
        quorum de corroboration — ce qui est le comportement voulu. Substituer
        un HOLD de confiance nulle serait pire : la panne deviendrait un avis.
        """
        portfolio = portfolio or {}

        last_price = float(data["Close"].iloc[-1])
        state = MarketState(
            symbol=symbol,
            price=last_price,
            timestamp=str(data.index[-1]),
        )

        signals: List[AgentSignal] = []

        for agent in self.agents:
            try:
                sig = agent.generate_signal(
                    state=state,
                    portfolio=portfolio,
                    regime=regime,
                    data=data,
                )
            except Exception as exc:
                name = getattr(agent, "name", type(agent).__name__)
                failure = AgentFailure(
                    agent_name=name,
                    symbol=symbol,
                    error=f"{type(exc).__name__}: {exc}"[:200],
                    traceback=traceback.format_exc(),
                )
                self.failures.append(failure)
                logger.warning("Arena: %s a échoué sur %s — %s",
                               name, symbol, failure.error)
                continue

            if sig is None:
                name = getattr(agent, "name", type(agent).__name__)
                self.failures.append(AgentFailure(
                    agent_name=name, symbol=symbol,
                    error="a retourné None au lieu d'un AgentSignal", traceback="",
                ))
                logger.warning("Arena: %s a retourné None sur %s", name, symbol)
                continue

            signals.append(sig)

        return signals
