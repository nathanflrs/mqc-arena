# src/risk/manager.py
from __future__ import annotations

import dataclasses
import json
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from src.broker.portfolio import PortfolioSnapshot
from src.execution.planner import OrderPlan, _compute_tx_cost

# Un BUY rogné en dessous de cette fraction de sa taille initiale n'est plus
# le trade que l'agent a proposé : on préfère le rejeter proprement plutôt
# que d'envoyer un résidu qui paiera la commission sans porter la thèse.
MIN_TRIM_FRACTION: float = 0.25

# ── Regime scaling factors for max_net_long_pct ───────────────────────────────
_REGIME_SCALE: Dict[str, float] = {
    "bull_quiet":    1.00,
    "bull_volatile": 0.75,
    "sideways":      0.60,
    "bear":          0.35,
}

# ── Circuit breaker level metadata ────────────────────────────────────────────
_CB_LEVELS = {
    0: ("NORMAL",   None),
    1: ("DÉFENSIF", 0.25),   # max_net_long override
    2: ("ALERTE",   0.12),
    3: ("URGENCE",  0.00),   # sell-only
}
_CB_ICONS  = {0: "✅", 1: "🟡", 2: "🟠", 3: "🔴"}


@dataclass
class RiskConfig:
    # Max exposition nette longue en % du net_liquidation (directional only)
    max_net_long_pct: float = 0.40
    # Max notional d'un seul ordre BUY en % du net_liquidation
    max_single_position_pct: float = 0.20
    # Floor de cash à conserver en % du net_liquidation
    min_cash_pct: float = 0.30
    # Kill switch manuel — bloque tous les BUY si True
    sell_only_mode: bool = False
    # Max gross exposure (long + short combined) for market-neutral pairs positions,
    # as % of net_liquidation. Tracked separately from max_net_long_pct because
    # pairs positions are approximately beta-neutral by construction.
    max_gross_pairs_pct: float = 0.30
    # Max gross exposure for CTA trend positions (long ETFs + short ETFs combined),
    # as % of net_liquidation. Excluded from net_long budget.
    max_gross_cta_pct: float = 0.60


@dataclass
class RejectedPlan:
    plan: OrderPlan
    reason: str


@dataclass
class TrimmedPlan:
    """Un BUY réduit pour tenir dans le budget restant, au lieu d'être rejeté."""
    symbol: str
    reason: str            # contrainte qui a mordu
    original_notional: float
    final_notional: float

    def render(self) -> str:
        return (
            f"  ✂️  {self.symbol}: ${self.original_notional:,.0f} → "
            f"${self.final_notional:,.0f} — {self.reason}"
        )


def _is_risk_reducing(current_qty: float, delta_qty: float) -> bool:
    """Un ordre réduit le risque s'il rapproche la position de zéro."""
    return abs(current_qty + delta_qty) < abs(current_qty) - 1e-9


def _order_key(p: OrderPlan) -> Tuple:
    """
    Ordre d'évaluation par le RiskManager.

    1. Ce qui libère du budget passe en premier (SELL, HOLD, réductions).
       Auparavant les plans étaient évalués dans l'ordre de la WATCHLIST : un
       SELL sur LLY évalué après les BUY ne libérait sa capacité qu'une fois
       les BUY déjà rejetés faute de place. Le budget était sous-estimé.
    2. Puis par conviction décroissante. Auparavant, premier arrivé premier
       servi : NVDA (Sharpe 1.50) était rejeté parce qu'il apparaît tard dans
       la WATCHLIST, au profit de plans plus faibles évalués avant lui.
    3. Notionnel puis symbole pour un ordre totalement déterministe.
    """
    frees_budget = (
        p.action in ("SELL", "HOLD")
        or _is_risk_reducing(p.current_qty, p.delta_qty)
    )
    return (0 if frees_budget else 1, -float(p.confidence), -float(p.est_notional), p.symbol)


def _trim_to(plan: OrderPlan, allowed_notional: float) -> Optional[OrderPlan]:
    """
    Réduit un BUY pour qu'il tienne dans `allowed_notional`.
    Retourne None si le résidu n'a plus de sens (< 1 action, ou < MIN_TRIM_FRACTION
    de la taille initiale).
    """
    px = float(plan.last_price)
    if px <= 0 or allowed_notional <= 0:
        return None

    new_qty = int(allowed_notional // px)
    if new_qty < 1:
        return None
    if (new_qty * px) < MIN_TRIM_FRACTION * float(plan.est_notional):
        return None

    return dataclasses.replace(
        plan,
        target_qty=float(plan.current_qty) + new_qty,
        delta_qty=float(new_qty),
        est_notional=float(new_qty * px),
        est_cost_usd=_compute_tx_cost(new_qty, px),
        reason=f"{plan.reason} [rogné par risk manager]",
    )


@dataclass
class RiskReport:
    approved: List[OrderPlan]
    rejected: List[RejectedPlan]
    pre_trade_long_pct: float
    post_trade_long_pct: float
    sell_only_triggered: bool
    # Regime-aware context (populated by RiskManager.check)
    regime: Optional[str] = None
    regime_scale: float = 1.0
    effective_max_net_long: float = 0.0
    # Circuit breaker state at time of check
    cb_level: int = 0
    cb_level_name: str = "NORMAL"
    # Market-neutral pairs exposure (gross = |long leg| + |short leg|), as % of NAV.
    # Tracked separately from net_long because pairs positions are beta-neutral
    # by construction and must not consume the directional net-long budget.
    pairs_gross_pre: float = 0.0
    pairs_gross_post: float = 0.0
    # CTA trend gross exposure (|long ETFs| + |short ETFs|), as % of NAV.
    # Excluded from net_long. Both long and short entries checked against max_gross_cta_pct.
    cta_gross_pre: float = 0.0
    cta_gross_post: float = 0.0
    # BUY réduits pour tenir dans le budget, au lieu d'être rejetés en bloc.
    trimmed: List["TrimmedPlan"] = field(default_factory=list)
    # Positions détenues sans plan ni prix connu : exposition non valorisable,
    # donc absente du net long calculé. À surveiller, jamais à ignorer.
    unpriced_positions: List[str] = field(default_factory=list)

    def telegram_summary(self) -> str:
        lines = ["🛡 Risk Manager"]
        if self.regime:
            lines.append(
                f"  Régime          : {self.regime.upper()}"
                f" (×{self.regime_scale:.2f} → lim {self.effective_max_net_long:.0%})"
            )
        if self.cb_level > 0:
            icon = _CB_ICONS.get(self.cb_level, "")
            lines.append(f"  Circuit breaker : {icon} {self.cb_level_name}")
        lines += [
            f"  Net long (pre)  : {self.pre_trade_long_pct:.1%}",
            f"  Net long (post) : {self.post_trade_long_pct:.1%}",
        ]
        if self.pairs_gross_post > 0 or self.pairs_gross_pre > 0:
            lines.append(
                f"  Pairs gross     : {self.pairs_gross_pre:.1%} → {self.pairs_gross_post:.1%}"
                " (market-neutral, exclu du net long)"
            )
        if self.cta_gross_post > 0 or self.cta_gross_pre > 0:
            lines.append(
                f"  CTA gross       : {self.cta_gross_pre:.1%} → {self.cta_gross_post:.1%}"
                " (trend long/short, exclu du net long)"
            )
        lines += [
            f"  Approuvés       : {len(self.approved)}",
            f"  Rejetés         : {len(self.rejected)}",
        ]
        if self.trimmed:
            lines.append(f"  Rognés          : {len(self.trimmed)}")
            lines.extend(t.render() for t in self.trimmed)
        if self.unpriced_positions:
            lines.append(
                "  ⚠️  Positions non valorisées (hors net long) : "
                + ", ".join(self.unpriced_positions)
            )
        if self.sell_only_triggered:
            lines.append("  ⚠️  SELL-ONLY MODE actif")
        for r in self.rejected:
            lines.append(f"  ✂️  {r.plan.symbol} ({r.plan.action}) → {r.reason}")
        return "\n".join(lines)


class RiskManager:
    def __init__(self, config: RiskConfig | None = None):
        self.cfg = config or RiskConfig()

    def check(
        self,
        plans: List[OrderPlan],
        snap: PortfolioSnapshot,
        *,
        gmm_regime: Optional[str] = None,
        adv_map: Optional[Dict[str, float]] = None,
        cb_level: int = 0,
        price_map: Optional[Dict[str, float]] = None,
    ) -> RiskReport:
        """
        Filtre les plans selon les règles de risque portefeuille.
        Les SELL passent toujours — on ne bloque jamais la réduction du risque.

        Params:
            gmm_regime  — label GMM courant ("bull_quiet" / "bull_volatile" / "sideways" / "bear").
                          Quand fourni, ajuste max_net_long_pct via _REGIME_SCALE.
            adv_map     — {symbol: adv_10j_en_actions}. Bloque les BUY > 1 % du volume journalier.
            cb_level    — niveau circuit breaker (0‑3). Affiché dans le rapport Telegram.

        Market-neutral routing
        ----------------------
        Plans with strategy='market_neutral' (pairs trades) are routed through a
        separate gross-exposure cap (max_gross_pairs_pct) and are EXCLUDED from
        the net-long calculation. This prevents pairs positions from consuming the
        directional long budget — they are beta-neutral by construction.

        The gross pairs cap = (long leg notional + short leg notional) / netliq.
        Both legs are submitted as separate OrderPlan objects tagged
        strategy='market_neutral' by the pairs execution layer.
        """
        netliq = snap.net_liquidation

        # ── Régime : ajustement dynamique du plafond long ─────────────────────
        regime_scale = _REGIME_SCALE.get(gmm_regime or "", 1.0)
        effective_max_long = self.cfg.max_net_long_pct * regime_scale

        approved: List[OrderPlan] = []
        rejected: List[RejectedPlan] = []
        trimmed: List[TrimmedPlan] = []

        # Directional net-long (excludes market-neutral and CTA legs)
        current_long_notional = sum(
            p.current_qty * p.last_price
            for p in plans
            if p.current_qty > 0 and p.strategy not in ("market_neutral", "cta_trend")
        )

        # ── Positions détenues mais SANS plan ce run ──────────────────────────
        # La somme ci-dessus itère sur `plans`, pas sur le portefeuille. Or un
        # symbole ne produit aucun plan quand ses données manquent, ou quand
        # l'arène ne désigne aucun gagnant (cas devenu plus fréquent depuis la
        # règle de corroboration P0(c)). Son exposition devenait alors invisible :
        # le net long était **sous-estimé**, donc les plafonds trop permissifs.
        # C'est le sens dangereux de l'erreur.
        _planned = {p.symbol for p in plans}
        _non_directional = {
            p.symbol for p in plans if p.strategy in ("market_neutral", "cta_trend")
        }
        unpriced_positions: List[str] = []
        for _sym, _qty in (snap.positions or {}).items():
            if _qty <= 0 or _sym in _planned or _sym in _non_directional:
                continue
            _px = (price_map or {}).get(_sym, 0.0)
            if _px > 0:
                current_long_notional += float(_qty) * float(_px)
            else:
                # Sans prix, impossible de valoriser : on le signale plutôt que
                # de faire silencieusement comme si l'exposition n'existait pas.
                unpriced_positions.append(_sym)

        projected_long_notional = current_long_notional
        projected_cash = snap.cash

        # Évaluation ordonnée : ce qui libère du budget d'abord, puis par
        # conviction décroissante. Voir _order_key().
        plans = sorted(plans, key=_order_key)

        # Gross pairs exposure: |long leg| + |short leg| for all market-neutral positions
        current_pairs_gross = sum(
            abs(p.current_qty) * p.last_price
            for p in plans
            if p.strategy == "market_neutral"
        )
        projected_pairs_gross = current_pairs_gross

        # Gross CTA exposure: |long ETFs| + |short ETFs| for all cta_trend positions
        current_cta_gross = sum(
            abs(p.current_qty) * p.last_price
            for p in plans
            if p.strategy == "cta_trend"
        )
        projected_cta_gross = current_cta_gross

        pre_trade_long_pct   = current_long_notional / netliq if netliq > 0 else 0.0
        pairs_gross_pre      = current_pairs_gross   / netliq if netliq > 0 else 0.0
        cta_gross_pre        = current_cta_gross     / netliq if netliq > 0 else 0.0

        for p in plans:
            is_neutral = p.strategy == "market_neutral"
            is_cta     = p.strategy == "cta_trend"

            # ── CTA trend : catégorie de risque distincte ──────────────────────
            # Traité en premier pour éviter que les SELL ouvrant un short CTA
            # soient auto-approuvés par le bloc SELL/HOLD ci-dessous.
            if is_cta:
                if p.action == "HOLD":
                    approved.append(p)
                    continue

                # Clôture d'un long CTA (SELL, on était long) → toujours approuvée
                if p.action == "SELL" and p.current_qty > 0:
                    approved.append(p)
                    projected_cta_gross = max(0.0, projected_cta_gross - p.est_notional)
                    continue

                # Couverture d'un short CTA (BUY, on était short) → toujours approuvée
                if p.action == "BUY" and p.current_qty < 0:
                    approved.append(p)
                    projected_cta_gross = max(0.0, projected_cta_gross - p.est_notional)
                    continue

                # Nouvelle entrée CTA (ouverture long ou short) — kill switch + gross cap
                if self.cfg.sell_only_mode:
                    rejected.append(RejectedPlan(p, "SELL_ONLY_MODE actif — nouvelle entrée CTA bloquée"))
                    continue

                new_cta_pct = (projected_cta_gross + p.est_notional) / netliq if netliq > 0 else 1.0
                if new_cta_pct > self.cfg.max_gross_cta_pct:
                    rejected.append(RejectedPlan(
                        p,
                        f"gross CTA {new_cta_pct:.1%} > max {self.cfg.max_gross_cta_pct:.0%} (cta_trend)",
                    ))
                    continue

                approved.append(p)
                projected_cta_gross += p.est_notional
                continue

            # ── SELL / HOLD : toujours approuvés (non-CTA) ────────────────────
            if p.action in ("SELL", "HOLD"):
                approved.append(p)
                if p.action == "SELL":
                    if is_neutral:
                        # Closing a pairs leg reduces gross exposure.
                        # Cash is approximately self-funding (long proceeds offset short cover),
                        # so we leave projected_cash unchanged for market-neutral closes.
                        projected_pairs_gross = max(0.0, projected_pairs_gross - p.est_notional)
                    else:
                        projected_long_notional = max(0.0, projected_long_notional - p.est_notional)
                        projected_cash += p.est_notional
                continue

            # ── BUY ───────────────────────────────────────────────────────────

            # Règle 0 — Kill switch / sell-only (applies to all strategies)
            if self.cfg.sell_only_mode:
                rejected.append(RejectedPlan(p, "SELL_ONLY_MODE actif — BUY bloqué"))
                continue

            if is_neutral:
                # ── Market-neutral path ────────────────────────────────────────
                # Rule MN-1: gross pairs cap
                new_gross_pct = (projected_pairs_gross + p.est_notional) / netliq if netliq > 0 else 1.0
                if new_gross_pct > self.cfg.max_gross_pairs_pct:
                    rejected.append(RejectedPlan(
                        p,
                        f"gross pairs exposure {new_gross_pct:.1%} > max "
                        f"{self.cfg.max_gross_pairs_pct:.0%} (market-neutral)",
                    ))
                    continue
                # Approved — pairs trades are approximately cash-neutral (long leg
                # funded by short sale proceeds), so we do not update projected_cash.
                approved.append(p)
                projected_pairs_gross += p.est_notional
                continue

            # ── Directional path ──────────────────────────────────────────────
            #
            # Les quatre règles historiques deviennent quatre plafonds de notionnel
            # évalués ensemble. On prend le plus contraignant, on nomme celui qui
            # mord, et on **rogne** le plan au lieu de le rejeter en bloc.
            #
            # Le rejet binaire faisait sauter NVDA (conviction 0.90) pour un
            # dépassement marginal du plafond net long, alors qu'une position
            # réduite tenait parfaitement dans le budget restant.
            budgets: List[Tuple[float, str]] = [
                (
                    self.cfg.max_single_position_pct * netliq,
                    f"taille unitaire max {self.cfg.max_single_position_pct:.0%} NAV",
                ),
                (
                    effective_max_long * netliq - projected_long_notional,
                    f"plafond net long {effective_max_long:.0%}"
                    + (f" (régime {gmm_regime})" if gmm_regime else ""),
                ),
                (
                    projected_cash - self.cfg.min_cash_pct * netliq,
                    f"floor de cash {self.cfg.min_cash_pct:.0%}",
                ),
            ]

            # Liquidité : max 1 % de l'ADV, exprimé en notionnel
            if adv_map:
                adv = adv_map.get(p.symbol, 0.0)
                if adv > 0 and p.last_price > 0:
                    budgets.append((
                        0.01 * adv * p.last_price,
                        "liquidité — max 1 % de l'ADV",
                    ))

            allowed, binding = min(budgets, key=lambda b: b[0])

            if allowed <= 0:
                rejected.append(RejectedPlan(
                    p, f"budget saturé — {binding} (capacité restante nulle)"
                ))
                continue

            if p.est_notional > allowed:
                trimmed_plan = _trim_to(p, allowed)
                if trimmed_plan is None:
                    rejected.append(RejectedPlan(
                        p,
                        f"{binding} — capacité restante ${allowed:,.0f} "
                        f"< {MIN_TRIM_FRACTION:.0%} de la taille demandée "
                        f"(${p.est_notional:,.0f})",
                    ))
                    continue
                trimmed.append(TrimmedPlan(
                    symbol=p.symbol, reason=binding,
                    original_notional=float(p.est_notional),
                    final_notional=float(trimmed_plan.est_notional),
                ))
                p = trimmed_plan

            # Plan approuvé (éventuellement rogné)
            approved.append(p)
            projected_long_notional += p.est_notional
            projected_cash -= p.est_notional

        post_trade_long_pct = projected_long_notional / netliq if netliq > 0 else 0.0
        pairs_gross_post    = projected_pairs_gross   / netliq if netliq > 0 else 0.0
        cta_gross_post      = projected_cta_gross     / netliq if netliq > 0 else 0.0
        cb_level_name = _CB_LEVELS.get(cb_level, _CB_LEVELS[0])[0]

        return RiskReport(
            approved=approved,
            rejected=rejected,
            pre_trade_long_pct=pre_trade_long_pct,
            post_trade_long_pct=post_trade_long_pct,
            sell_only_triggered=self.cfg.sell_only_mode,
            regime=gmm_regime,
            regime_scale=regime_scale,
            effective_max_net_long=effective_max_long,
            cb_level=cb_level,
            cb_level_name=cb_level_name,
            pairs_gross_pre=pairs_gross_pre,
            pairs_gross_post=pairs_gross_post,
            cta_gross_pre=cta_gross_pre,
            cta_gross_post=cta_gross_post,
            trimmed=trimmed,
            unpriced_positions=unpriced_positions,
        )


class DrawdownCircuitBreaker:
    """
    Graduated drawdown protection with 3 levels.

    Level 0 — NORMAL   : drawdown ≤ 4 %    — no restriction
    Level 1 — DÉFENSIF : drawdown > 4 %    — max_net_long capped at 0.25
    Level 2 — ALERTE   : drawdown > 6 %    — max_net_long capped at 0.12,
                                               BUY blocked unless confidence > 0.85
    Level 3 — URGENCE  : drawdown > 8 %    — SELL-ONLY (sticky, manual reset required)

    Levels 0-2 are dynamic (revert automatically when drawdown recovers).
    Level 3 is sticky until manual reset().
    State persists in logs/circuit_breaker.json between runs.
    """

    _THRESHOLDS = [(0.08, 3), (0.06, 2), (0.04, 1)]   # sorted descending — single source of truth
    _STATE_PATH: Path = Path("logs/circuit_breaker.json")

    def __init__(self) -> None:
        self._state = self._load()

    # ── Persistence ───────────────────────────────────────────────────────────

    def _load(self) -> dict:
        defaults: dict = {
            "triggered": False,
            "level": 0,
            "peak_netliq": None,
            "current_netliq": None,
            "drawdown": 0.0,
            "triggered_at": None,
            "drawdown_at_trigger": None,
        }
        if self._STATE_PATH.exists():
            try:
                data = json.loads(self._STATE_PATH.read_text())
                # Migrate old JSON that lacks "level"
                if "level" not in data:
                    dd = float(data.get("drawdown") or 0.0)
                    data["level"] = 3 if data.get("triggered") else self._level_from_dd(dd)
                return {**defaults, **data}
            except (json.JSONDecodeError, OSError):
                pass
        return defaults

    def _save(self) -> None:
        self._STATE_PATH.parent.mkdir(parents=True, exist_ok=True)
        self._STATE_PATH.write_text(json.dumps(self._state, indent=2))

    # ── Helpers ───────────────────────────────────────────────────────────────

    @classmethod
    def _level_from_dd(cls, dd: float) -> int:
        for threshold, level in cls._THRESHOLDS:
            if dd > threshold:
                return level
        return 0

    # ── Properties ────────────────────────────────────────────────────────────

    @property
    def level(self) -> int:
        return int(self._state.get("level", 0))

    @property
    def level_name(self) -> str:
        return _CB_LEVELS.get(self.level, _CB_LEVELS[0])[0]

    @property
    def is_triggered(self) -> bool:
        """True only at level 3 (URGENCE — sell-only). Backward compatible."""
        return self.level >= 3

    @property
    def drawdown(self) -> float:
        return float(self._state.get("drawdown") or 0.0)

    @property
    def peak_netliq(self) -> float | None:
        v = self._state.get("peak_netliq")
        return float(v) if v is not None else None

    # ── Evaluation ────────────────────────────────────────────────────────────

    def evaluate(self, netliq: float, *, ci_mode: bool = False) -> bool:
        """
        Record NetLiq, compute drawdown, update level.
        Sends a Telegram alert when level increases.
        Returns True only at level 3 (sell-only), for backward compat.
        """
        s = self._state

        if s["peak_netliq"] is None or netliq > float(s["peak_netliq"]):
            s["peak_netliq"] = netliq

        s["current_netliq"] = netliq
        peak = float(s["peak_netliq"])
        dd = (peak - netliq) / peak if peak > 0 else 0.0
        s["drawdown"] = dd

        old_level = int(s.get("level", 0))

        # Level 3 is sticky — only manual reset() clears it
        if old_level >= 3:
            new_level = 3
        else:
            new_level = self._level_from_dd(dd)

        # Record first entry into level 3
        if new_level == 3 and old_level < 3:
            s["triggered"] = True
            s["triggered_at"] = datetime.now(timezone.utc).isoformat()
            s["drawdown_at_trigger"] = dd

        s["level"] = new_level
        self._save()

        if new_level > old_level and not ci_mode:
            self._send_level_alert(new_level, dd, netliq, peak)

        return new_level >= 3

    def _send_level_alert(self, level: int, dd: float, netliq: float, peak: float) -> None:
        icon     = _CB_ICONS.get(level, "")
        name, _  = _CB_LEVELS.get(level, ("?", None))
        cb_limit = _CB_LEVELS[level][1]

        if level == 3:
            title = f"🔴 CIRCUIT BREAKER — URGENCE — Milan Capital"
            body  = (
                f"Drawdown depuis pic : {dd:.1%}\n"
                f"Peak NetLiq : ${peak:,.0f} → ${netliq:,.0f}\n"
                f"⛔ SELL-ONLY MODE actif automatiquement.\n"
                f"Reset manuel requis."
            )
        else:
            limit_str = f"max_net_long → {cb_limit:.0%}" if cb_limit else ""
            title = f"{icon} CIRCUIT BREAKER — {name} — Milan Capital"
            body  = (
                f"Drawdown depuis pic : {dd:.1%}\n"
                f"Peak NetLiq : ${peak:,.0f} → ${netliq:,.0f}\n"
                f"{limit_str}"
            )

        # Level 1 (DÉFENSIF): avertissement dashboard uniquement.
        # Levels 2+ (ALERTE, URGENCE): critical → dashboard + Telegram.
        severity = "warning" if level == 1 else "critical"

        try:
            from src.events.bus import get_bus, Event
            get_bus().emit(Event(
                type="circuit_breaker",
                severity=severity,
                title=title,
                body=body,
                meta={"level": level, "drawdown": round(dd, 4),
                      "netliq": netliq, "peak": peak},
            ))
        except Exception:
            pass

    # ── Reset ─────────────────────────────────────────────────────────────────

    def reset(self) -> None:
        """Manual reset only. Clears level 3 (sell-only). Never called automatically."""
        self._state["triggered"] = False
        self._state["level"] = 0
        self._state["triggered_at"] = None
        self._state["drawdown_at_trigger"] = None
        self._save()
        print("✅ Circuit breaker reset. All levels cleared.")
        
