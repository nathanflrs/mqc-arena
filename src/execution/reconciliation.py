# src/execution/reconciliation.py
"""
Milan Capital — Suivi du cycle de vie des ordres et réconciliation broker.

Ce module ne connaît pas IBKR. Il ne contient que la logique décidable :
quels ordres sont terminés, lesquels annuler, ce qu'il faut journaliser, et
comment comparer l'état interne au portefeuille réel. L'adaptateur ib_insync
vit dans le runner. Cette séparation est délibérée : le code broker n'est pas
testable sans gateway, donc tout ce qui peut en être extrait doit l'être.

Le problème corrigé
-------------------
`execute_plans_paper_ibkr` plaçait les ordres, dormait 10 secondes, journalisait
le statut courant, puis n'y revenait jamais. Conséquences :

  - Un ordre non rempli était écrit dans executions.csv comme s'il s'agissait
    d'un trade. `_load_entry_prices` reprenait alors son `limit_price` comme
    prix de revient, et le moteur de stop-loss raisonnait sur une position
    inexistante. Les 3 lignes d'executions.csv sont toutes en `PendingSubmit`.
  - Les ordres DAY non remplis n'étaient jamais annulés : ils pouvaient se
    remplir plus tard dans la séance, à un prix sans rapport avec le signal,
    sans que le système le sache.
  - Un remplissage partiel était journalisé à la quantité **demandée**, pas
    à la quantité obtenue.
  - Aucune comparaison entre l'état interne et le portefeuille IBKR réel.

Règle directrice : on ne journalise que ce qui s'est réellement passé.
Un ordre non rempli est un non-événement, pas un trade à zéro.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Optional, Tuple

# Statuts IBKR. Alignés sur ib_insync.OrderStatus (v0.9.86) et complétés par
# 'Inactive', que l'API renvoie sur un ordre rejeté et qui ne figure dans
# aucune des deux listes de la librairie — un ordre inactif n'est ni actif
# ni terminé selon ib_insync, ce qui ferait tourner le polling jusqu'au timeout.
TERMINAL_STATUSES = frozenset({"Filled", "Cancelled", "ApiCancelled", "Inactive"})
ACTIVE_STATUSES = frozenset({"ApiPending", "PendingSubmit", "PreSubmitted", "Submitted"})

# Tolérance de réconciliation, en actions. IBKR renvoie des quantités
# fractionnaires sur certains instruments ; en deçà, l'écart n'est pas un écart.
POSITION_TOLERANCE = 0.5


def is_terminal(status: str) -> bool:
    """Un ordre terminé ne bougera plus : inutile de continuer à l'interroger."""
    return status in TERMINAL_STATUSES


@dataclass
class OrderOutcome:
    """Ce qu'un ordre est devenu, une fois son cycle de vie achevé."""
    symbol: str
    side: str                    # "BUY" | "SELL"
    requested_qty: int
    filled_qty: float
    avg_fill_price: float
    limit_price: float
    signal_price: float          # prix qui a déclenché la décision
    status: str
    reason: str = ""
    cancelled_remainder: bool = False

    @property
    def is_filled(self) -> bool:
        return self.filled_qty >= self.requested_qty - 1e-9 and self.filled_qty > 0

    @property
    def is_partial(self) -> bool:
        return 0 < self.filled_qty < self.requested_qty - 1e-9

    @property
    def is_unfilled(self) -> bool:
        return self.filled_qty <= 1e-9

    @property
    def signed_qty(self) -> float:
        """Variation de position effective : positive à l'achat, négative à la vente."""
        return self.filled_qty if self.side == "BUY" else -self.filled_qty

    @property
    def slippage_bps(self) -> float:
        """
        Écart entre le prix obtenu et le prix qui a déclenché la décision.
        Positif = défavorable, dans les deux sens.
        """
        if self.avg_fill_price <= 0 or self.signal_price <= 0:
            return 0.0
        d = (self.avg_fill_price - self.signal_price) / self.signal_price
        return (d if self.side == "BUY" else -d) * 10_000.0

    def render(self) -> str:
        if self.is_unfilled:
            return (f"  ⭕ {self.symbol} {self.side} {self.requested_qty} — "
                    f"non rempli ({self.status}), annulé")
        tag = "partiel" if self.is_partial else "rempli"
        extra = f" — reliquat {self.requested_qty - self.filled_qty:.0f} annulé" if self.cancelled_remainder else ""
        return (f"  ✅ {self.symbol} {self.side} {self.filled_qty:.0f}/{self.requested_qty} "
                f"@ {self.avg_fill_price:.2f} ({tag}, slippage {self.slippage_bps:+.1f} bps){extra}")


def to_execution_rows(
    outcomes: Iterable[OrderOutcome], plan_id: str, timestamp: str,
) -> List[dict]:
    """
    Lignes à écrire dans executions.csv — **uniquement les remplissages réels**.

    Un ordre non rempli n'y figure pas. C'est ce qui empêche le moteur de
    stop-loss de reprendre le `limit_price` d'un ordre jamais exécuté comme
    prix de revient d'une position qui n'existe pas.
    """
    rows = []
    for o in outcomes:
        if o.is_unfilled:
            continue
        rows.append({
            "plan_id": plan_id,
            "timestamp": timestamp,
            "symbol": o.symbol,
            "side": o.side,
            "qty": float(o.filled_qty),          # quantité OBTENUE, pas demandée
            "requested_qty": int(o.requested_qty),
            "limit_price": round(o.limit_price, 4),
            "avg_fill_price": round(o.avg_fill_price, 4),
            "slippage_bps": round(o.slippage_bps, 2),
            "last_price": round(o.signal_price, 4),
            "est_notional": round(o.filled_qty * o.avg_fill_price, 2),
            "status": o.status,
            "partial": o.is_partial,
            "reason": o.reason,
        })
    return rows


def expected_positions(
    before: Dict[str, float], outcomes: Iterable[OrderOutcome],
) -> Dict[str, float]:
    """Positions attendues après exécution = positions initiales + quantités obtenues."""
    out = {k: float(v) for k, v in before.items()}
    for o in outcomes:
        if o.is_unfilled:
            continue
        out[o.symbol] = out.get(o.symbol, 0.0) + o.signed_qty
    return {k: v for k, v in out.items() if abs(v) > 1e-9}


@dataclass
class PositionDrift:
    symbol: str
    expected: float
    actual: float

    @property
    def delta(self) -> float:
        return self.actual - self.expected

    def render(self) -> str:
        return (f"  ⚠️  {self.symbol}: attendu {self.expected:+.0f}, "
                f"réel {self.actual:+.0f} (écart {self.delta:+.0f})")


@dataclass
class ReconciliationReport:
    drifts: List[PositionDrift] = field(default_factory=list)
    n_checked: int = 0

    @property
    def is_clean(self) -> bool:
        return not self.drifts

    def render(self) -> str:
        if self.is_clean:
            return f"🔍 Réconciliation : {self.n_checked} position(s) conformes ✅"
        lines = [f"🚨 Réconciliation : {len(self.drifts)} écart(s) sur {self.n_checked} position(s)"]
        lines += [d.render() for d in self.drifts]
        lines.append("   L'état interne et le compte broker divergent — "
                     "ne pas relancer d'exécution avant d'avoir tranché.")
        return "\n".join(lines)


def reconcile(
    expected: Dict[str, float],
    actual: Dict[str, float],
    tolerance: float = POSITION_TOLERANCE,
) -> ReconciliationReport:
    """
    Compare positions attendues et positions réellement détenues chez le broker.

    Un écart signifie qu'une hypothèse du système est fausse : ordre rempli
    hors fenêtre de polling, remplissage manqué, position ouverte à la main,
    corporate action. Dans tous les cas il faut le savoir avant le run suivant,
    qui calculerait ses deltas à partir d'un état erroné.
    """
    report = ReconciliationReport()
    for sym in sorted(set(expected) | set(actual)):
        exp = float(expected.get(sym, 0.0))
        act = float(actual.get(sym, 0.0))
        report.n_checked += 1
        if abs(act - exp) > tolerance:
            report.drifts.append(PositionDrift(symbol=sym, expected=exp, actual=act))
    return report


def summarize(outcomes: List[OrderOutcome]) -> str:
    """Résumé lisible du devenir des ordres d'un run."""
    if not outcomes:
        return "📭 Aucun ordre envoyé."
    filled = [o for o in outcomes if o.is_filled]
    partial = [o for o in outcomes if o.is_partial]
    unfilled = [o for o in outcomes if o.is_unfilled]
    lines = [
        f"📊 Devenir des ordres : {len(filled)} rempli(s), "
        f"{len(partial)} partiel(s), {len(unfilled)} non rempli(s) "
        f"sur {len(outcomes)}"
    ]
    lines += [o.render() for o in outcomes]
    done = filled + partial
    if done:
        avg = sum(o.slippage_bps for o in done) / len(done)
        lines.append(f"  Slippage moyen sur les ordres exécutés : {avg:+.1f} bps")
    return "\n".join(lines)
