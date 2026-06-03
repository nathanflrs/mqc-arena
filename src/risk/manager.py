# src/risk/manager.py
from __future__ import annotations

from dataclasses import dataclass, field
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
