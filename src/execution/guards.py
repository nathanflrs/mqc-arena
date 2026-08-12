# src/execution/guards.py
"""
Milan Capital — Garde d'exécution.

Transforme une liste de plans approuvés par le RiskManager en une liste d'ordres
effectivement envoyables au broker, en appliquant deux contraintes d'exécution :

  1. MAX_NOTIONAL_PCT — notionnel maximal d'un ordre unitaire, en % du NetLiq.
     C'est un **plafond de tranche**, pas un filtre : un plan trop gros est
     redimensionné, jamais supprimé. Une cible de 17 % se construit alors en
     plusieurs sessions au lieu de ne jamais se construire.

  2. MAX_ORDERS_PER_RUN — nombre maximal d'ordres par run.
     Ne s'applique qu'aux ordres **augmentant** le risque. Un ordre qui réduit
     le risque (sortie, stop-loss, couverture de short) n'est jamais ni
     redimensionné ni écarté par un quota : on ne rationne pas la réduction
     de risque. C'est le même principe que RiskManager.check(), où les SELL
     passent toujours.

Historique
----------
Avant le 2026-08-02, un plan dont `est_notional > max_notional` était éliminé
par un `continue` silencieux. Avec MAX_NOTIONAL_PCT=0.02 sur un NetLiq de
1.03 M$, le plafond valait 20 522 $ alors que les plans générés valaient
41 k$ à 154 k$ : **100 % des ordres étaient jetés sans trace**, et le seul
message émis annonçait « filtered / HOLD », ce qui était faux.

Ce module rend la troncature explicite, ordonnée par conviction, et auditable.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Sequence, Tuple

from src.execution.planner import OrderPlan

# Seules les stratégies directionnelles voient leur SELL plafonné aux titres
# détenus (protection contre un short accidentel). Les stratégies qui shortent
# délibérément — pairs market-neutral, CTA long/short — en sont exemptées :
# leur exposition est bornée en amont par les plafonds gross du RiskManager.
_MAY_OPEN_SHORT = ("market_neutral", "cta_trend")


@dataclass
class ExecutionCandidate:
    """Un ordre prêt à être envoyé au broker."""
    plan: OrderPlan
    side: str                  # "BUY" | "SELL"
    qty: int                   # quantité finale, après clip et redimensionnement
    limit_price: float
    risk_reducing: bool
    requested_qty: int         # quantité initialement demandée par le plan
    resized: bool = False

    @property
    def notional(self) -> float:
        return self.qty * float(self.plan.last_price)


@dataclass
class ExecutionAdjustment:
    """Trace auditable d'une modification ou d'un rejet à l'exécution."""
    symbol: str
    kind: str                  # "resized" | "clipped" | "dropped"
    reason: str
    requested_qty: int
    final_qty: int
    requested_notional: float
    final_notional: float

    def render(self) -> str:
        if self.kind == "dropped":
            return (
                f"  ✂️  {self.symbol}: écarté — {self.reason} "
                f"(demandé {self.requested_qty} × ${self.requested_notional:,.0f})"
            )
        return (
            f"  📐 {self.symbol}: {self.requested_qty} → {self.final_qty} titres "
            f"(${self.requested_notional:,.0f} → ${self.final_notional:,.0f}) — {self.reason}"
        )


@dataclass
class ExecutionPlan:
    """Résultat complet de la garde : ce qui part, ce qui a été modifié, ce qui saute."""
    candidates: List[ExecutionCandidate] = field(default_factory=list)
    adjustments: List[ExecutionAdjustment] = field(default_factory=list)
    max_notional: float = 0.0

    @property
    def n_resized(self) -> int:
        return sum(1 for a in self.adjustments if a.kind == "resized")

    @property
    def n_dropped(self) -> int:
        return sum(1 for a in self.adjustments if a.kind == "dropped")

    def render(self) -> str:
        """Rapport texte honnête : ce qui part ET ce qui a été rogné, avec les chiffres."""
        lines = [
            f"🧮 Garde d'exécution — plafond unitaire ${self.max_notional:,.0f}",
        ]
        if self.candidates:
            lines.append(f"  ✅ {len(self.candidates)} ordre(s) à envoyer :")
            for c in self.candidates:
                tag = " [réduction risque]" if c.risk_reducing else f" [conv={c.plan.confidence:.2f}]"
                resized = f" (rogné depuis {c.requested_qty})" if c.resized else ""
                lines.append(
                    f"  📤 {c.plan.symbol}: {c.side} {c.qty} @ ~{c.limit_price:.2f}"
                    f" | ${c.notional:,.0f}{resized}{tag}"
                )
        else:
            lines.append("  ⚠️  Aucun ordre envoyable.")

        if self.adjustments:
            lines.append(f"  🔧 {len(self.adjustments)} ajustement(s) :")
            lines.extend(a.render() for a in self.adjustments)
        return "\n".join(lines)


def _is_risk_reducing(current_qty: float, delta_qty: float) -> bool:
    """
    Un ordre réduit le risque s'il rapproche la position de zéro.

        long 100 → SELL 100   : |0|   < |100| → réduction
        short -50 → BUY 50    : |0|   < |-50| → réduction (couverture)
        flat 0  → BUY 100     : |100| > |0|   → augmentation
        flat 0  → SELL 100    : |100| > |0|   → augmentation (ouverture de short)
        long 100 → SELL 300   : |-200| > |100| → augmentation (reversal CTA)

    Un reversal est classé « augmentation » : il crée une exposition nouvelle
    dans le sens opposé, il doit donc passer par le plafond et la file d'attente.
    """
    return abs(current_qty + delta_qty) < abs(current_qty) - 1e-9


def build_execution_plan(
    plans: Sequence[OrderPlan],
    *,
    net_liquidation: float,
    max_notional_pct: float,
    max_orders: int,
    limit_buffer_bps: float,
) -> ExecutionPlan:
    """
    Construit la liste d'ordres envoyables.

    Ordre de traitement :
      1. Filtre les HOLD (delta nul).
      2. Plafonne les SELL directionnels aux titres détenus (anti-short accidentel).
      3. Classe chaque ordre : réduction ou augmentation de risque.
      4. Redimensionne les augmentations au plafond unitaire.
      5. Trie les augmentations par conviction décroissante, puis notionnel.
      6. Applique max_orders aux seules augmentations.
      7. Les réductions passent en tête, sans plafond ni quota.
    """
    max_notional = float(net_liquidation) * float(max_notional_pct)
    buff = float(limit_buffer_bps) / 10_000.0

    out = ExecutionPlan(max_notional=max_notional)
    reducing: List[ExecutionCandidate] = []
    increasing: List[ExecutionCandidate] = []

    for p in plans:
        dq = int(round(p.delta_qty))
        if dq == 0:
            continue  # HOLD — rien à envoyer, pas un rejet

        side = "BUY" if dq > 0 else "SELL"
        requested_qty = abs(dq)
        qty = requested_qty
        px = float(p.last_price)

        if px <= 0:
            out.adjustments.append(ExecutionAdjustment(
                symbol=p.symbol, kind="dropped",
                reason="prix de référence nul — impossible de sizer",
                requested_qty=requested_qty, final_qty=0,
                requested_notional=float(p.est_notional), final_notional=0.0,
            ))
            continue

        # ── 2. Anti-short accidentel sur les stratégies purement longues ──────
        strategy = getattr(p, "strategy", "directional")
        if side == "SELL" and strategy not in _MAY_OPEN_SHORT:
            held = int(round(p.current_qty))
            if qty > held:
                out.adjustments.append(ExecutionAdjustment(
                    symbol=p.symbol, kind="clipped",
                    reason=f"SELL directionnel plafonné aux {held} titres détenus (pas de short)",
                    requested_qty=requested_qty, final_qty=max(held, 0),
                    requested_notional=requested_qty * px, final_notional=max(held, 0) * px,
                ))
                qty = max(held, 0)
            if qty <= 0:
                continue

        signed_qty = qty if side == "BUY" else -qty
        reduces = _is_risk_reducing(float(p.current_qty), float(signed_qty))

        # ── 4. Plafond unitaire — augmentations uniquement ────────────────────
        resized = False
        if not reduces and qty * px > max_notional:
            capped = int(max_notional // px)
            if capped < 1:
                out.adjustments.append(ExecutionAdjustment(
                    symbol=p.symbol, kind="dropped",
                    reason=(
                        f"1 titre (${px:,.2f}) dépasse déjà le plafond unitaire "
                        f"${max_notional:,.0f} — augmente MAX_NOTIONAL_PCT"
                    ),
                    requested_qty=requested_qty, final_qty=0,
                    requested_notional=qty * px, final_notional=0.0,
                ))
                continue
            out.adjustments.append(ExecutionAdjustment(
                symbol=p.symbol, kind="resized",
                reason=f"plafond unitaire ${max_notional:,.0f} ({max_notional_pct:.1%} NetLiq)",
                requested_qty=qty, final_qty=capped,
                requested_notional=qty * px, final_notional=capped * px,
            ))
            qty = capped
            resized = True

        limit_price = px * (1.0 + buff) if side == "BUY" else px * (1.0 - buff)

        cand = ExecutionCandidate(
            plan=p, side=side, qty=qty, limit_price=limit_price,
            risk_reducing=reduces, requested_qty=requested_qty, resized=resized,
        )
        (reducing if reduces else increasing).append(cand)

    # ── 5. Tri par conviction : la troncature ne doit jamais être arbitraire ──
    increasing.sort(
        key=lambda c: (float(c.plan.confidence), c.notional),
        reverse=True,
    )

    # ── 6. Quota — augmentations uniquement ───────────────────────────────────
    if max_orders is not None and max_orders >= 0 and len(increasing) > max_orders:
        for c in increasing[max_orders:]:
            out.adjustments.append(ExecutionAdjustment(
                symbol=c.plan.symbol, kind="dropped",
                reason=(
                    f"quota MAX_ORDERS_PER_RUN={max_orders} atteint — "
                    f"conviction {c.plan.confidence:.2f} insuffisante ce run"
                ),
                requested_qty=c.requested_qty, final_qty=0,
                requested_notional=c.notional, final_notional=0.0,
            ))
        increasing = increasing[:max_orders]

    # ── 7. Réductions de risque en tête ───────────────────────────────────────
    out.candidates = reducing + increasing
    return out
