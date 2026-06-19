# src/risk/manager.py
from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import List

from src.broker.portfolio import PortfolioSnapshot
from src.execution.planner import OrderPlan


@dataclass
class RiskConfig:
    # Max exposition nette longue en % du net_liquidation
    max_net_long_pct: float = 0.40
    # Max notional d'un seul ordre BUY en % du net_liquidation
    max_single_position_pct: float = 0.20
    # Floor de cash à conserver en % du net_liquidation
    min_cash_pct: float = 0.30
    # Kill switch manuel — bloque tous les BUY si True
    sell_only_mode: bool = False


@dataclass
class RejectedPlan:
    plan: OrderPlan
    reason: str


@dataclass
class RiskReport:
    approved: List[OrderPlan]
    rejected: List[RejectedPlan]
    pre_trade_long_pct: float
    post_trade_long_pct: float
    sell_only_triggered: bool

    def telegram_summary(self) -> str:
        lines = [
            f"🛡 Risk Manager",
            f"  Net long (pre)  : {self.pre_trade_long_pct:.1%}",
            f"  Net long (post) : {self.post_trade_long_pct:.1%}",
            f"  Approuvés       : {len(self.approved)}",
            f"  Rejetés         : {len(self.rejected)}",
        ]
        if self.sell_only_triggered:
            lines.append("  ⚠️  SELL-ONLY MODE actif")
        for r in self.rejected:
            lines.append(f"  ✂️  {r.plan.symbol} ({r.plan.action}) → {r.reason}")
        return "\n".join(lines)


class RiskManager:
    def __init__(self, config: RiskConfig | None = None):
        self.cfg = config or RiskConfig()

    def check(self, plans: List[OrderPlan], snap: PortfolioSnapshot) -> RiskReport:
        """
        Filtre les plans selon les règles de risque portefeuille.
        Les SELL passent toujours — on ne bloque jamais la réduction du risque.
        """
        netliq = snap.net_liquidation
        approved: List[OrderPlan] = []
        rejected: List[RejectedPlan] = []

        # Exposition longue actuelle = current_qty * last_price sur tous les plans
        # (proxy valable si le watchlist couvre toutes les positions)
        current_long_notional = sum(
            p.current_qty * p.last_price for p in plans if p.current_qty > 0
        )
        projected_long_notional = current_long_notional
        projected_cash = snap.cash

        pre_trade_long_pct = current_long_notional / netliq if netliq > 0 else 0.0

        for p in plans:
            # SELL et HOLD : toujours approuvés, réduisent le risque
            if p.action in ("SELL", "HOLD"):
                approved.append(p)
                if p.action == "SELL":
                    projected_long_notional = max(0.0, projected_long_notional - p.est_notional)
                    projected_cash += p.est_notional
                continue

            # À partir d'ici : action == BUY

            # Règle 0 — Kill switch manuel
            if self.cfg.sell_only_mode:
                rejected.append(RejectedPlan(p, "SELL_ONLY_MODE actif — BUY bloqué"))
                continue

            # Règle 1 — Taille unitaire maximale
            single_pct = p.est_notional / netliq if netliq > 0 else 1.0
            if single_pct > self.cfg.max_single_position_pct:
                rejected.append(RejectedPlan(
                    p,
                    f"position unitaire {single_pct:.1%} > max {self.cfg.max_single_position_pct:.0%}",
                ))
                continue

            # Règle 2 — Exposition nette longue globale
            new_long_pct = (projected_long_notional + p.est_notional) / netliq if netliq > 0 else 1.0
            if new_long_pct > self.cfg.max_net_long_pct:
                rejected.append(RejectedPlan(
                    p,
                    f"net long post-trade {new_long_pct:.1%} > max {self.cfg.max_net_long_pct:.0%}",
                ))
                continue

            # Règle 3 — Floor de cash
            new_cash_pct = (projected_cash - p.est_notional) / netliq if netliq > 0 else 0.0
            if new_cash_pct < self.cfg.min_cash_pct:
                rejected.append(RejectedPlan(
                    p,
                    f"cash résiduel {new_cash_pct:.1%} < floor {self.cfg.min_cash_pct:.0%}",
                ))
                continue

            # Plan approuvé — mettre à jour les projections
            approved.append(p)
            projected_long_notional += p.est_notional
            projected_cash -= p.est_notional

        post_trade_long_pct = projected_long_notional / netliq if netliq > 0 else 0.0

        return RiskReport(
            approved=approved,
            rejected=rejected,
            pre_trade_long_pct=pre_trade_long_pct,
            post_trade_long_pct=post_trade_long_pct,
            sell_only_triggered=self.cfg.sell_only_mode,
        )


class DrawdownCircuitBreaker:
    """
    Monitors portfolio drawdown from the NetLiq peak.
    Triggers SELL-ONLY mode automatically when drawdown > THRESHOLD (8%).
    Reset is manual only — never automatic.
    State persists in logs/circuit_breaker.json between runs.
    """

    THRESHOLD: float = 0.08
    _STATE_PATH: Path = Path("logs/circuit_breaker.json")

    def __init__(self) -> None:
        self._state = self._load()

    def _load(self) -> dict:
        if self._STATE_PATH.exists():
            try:
                return json.loads(self._STATE_PATH.read_text())
            except (json.JSONDecodeError, OSError):
                pass
        return {
            "triggered": False,
            "peak_netliq": None,
            "current_netliq": None,
            "drawdown": 0.0,
            "triggered_at": None,
            "drawdown_at_trigger": None,
        }

    def _save(self) -> None:
        self._STATE_PATH.parent.mkdir(parents=True, exist_ok=True)
        self._STATE_PATH.write_text(json.dumps(self._state, indent=2))

    @property
    def is_triggered(self) -> bool:
        return bool(self._state["triggered"])

    @property
    def drawdown(self) -> float:
        return float(self._state.get("drawdown") or 0.0)

    @property
    def peak_netliq(self) -> float | None:
        v = self._state.get("peak_netliq")
        return float(v) if v is not None else None

    def evaluate(self, netliq: float, *, ci_mode: bool = False) -> bool:
        """
        Record the current NetLiq, update peak, compute drawdown.
        Sends a Telegram alert and activates SELL-ONLY on first breach.
        Returns True if the circuit breaker is active.
        """
        s = self._state

        if s["peak_netliq"] is None or netliq > float(s["peak_netliq"]):
            s["peak_netliq"] = netliq

        s["current_netliq"] = netliq
        peak = float(s["peak_netliq"])
        dd = (peak - netliq) / peak if peak > 0 else 0.0
        s["drawdown"] = dd

        newly_triggered = dd > self.THRESHOLD and not s["triggered"]
        if newly_triggered:
            s["triggered"] = True
            s["triggered_at"] = datetime.now(timezone.utc).isoformat()
            s["drawdown_at_trigger"] = dd

        self._save()

        if newly_triggered and not ci_mode:
            try:
                from src.notify.telegram import send_message
                send_message(
                    f"🚨 CIRCUIT BREAKER ACTIVÉ — Milan Capital\n"
                    f"Drawdown depuis pic : {dd:.1%}\n"
                    f"Peak NetLiq : ${peak:,.0f}\n"
                    f"Current NetLiq : ${netliq:,.0f}\n"
                    f"Seuil : {self.THRESHOLD:.0%}\n\n"
                    f"⛔ SELL-ONLY MODE actif automatiquement.\n"
                    f"Reset manuel requis."
                )
            except Exception:
                pass

        return bool(s["triggered"])

    def reset(self) -> None:
        """Manual reset only. Never called automatically."""
        self._state["triggered"] = False
        self._state["triggered_at"] = None
        self._state["drawdown_at_trigger"] = None
        self._save()
        print("✅ Circuit breaker reset. SELL-ONLY mode deactivated.")
        