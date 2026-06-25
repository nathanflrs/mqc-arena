# src/risk/manager.py
from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional

from src.broker.portfolio import PortfolioSnapshot
from src.execution.planner import OrderPlan

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
    # Regime-aware context (populated by RiskManager.check)
    regime: Optional[str] = None
    regime_scale: float = 1.0
    effective_max_net_long: float = 0.0
    # Circuit breaker state at time of check
    cb_level: int = 0
    cb_level_name: str = "NORMAL"

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

    def check(
        self,
        plans: List[OrderPlan],
        snap: PortfolioSnapshot,
        *,
        gmm_regime: Optional[str] = None,
        adv_map: Optional[Dict[str, float]] = None,
        cb_level: int = 0,
    ) -> RiskReport:
        """
        Filtre les plans selon les règles de risque portefeuille.
        Les SELL passent toujours — on ne bloque jamais la réduction du risque.

        Params:
            gmm_regime  — label GMM courant ("bull_quiet" / "bull_volatile" / "sideways" / "bear").
                          Quand fourni, ajuste max_net_long_pct via _REGIME_SCALE.
            adv_map     — {symbol: adv_10j_en_actions}. Bloque les BUY > 1 % du volume journalier.
            cb_level    — niveau circuit breaker (0‑3). Affiché dans le rapport Telegram.
        """
        netliq = snap.net_liquidation

        # ── Régime : ajustement dynamique du plafond long ─────────────────────
        regime_scale = _REGIME_SCALE.get(gmm_regime or "", 1.0)
        effective_max_long = self.cfg.max_net_long_pct * regime_scale

        approved: List[OrderPlan] = []
        rejected: List[RejectedPlan] = []

        current_long_notional = sum(
            p.current_qty * p.last_price for p in plans if p.current_qty > 0
        )
        projected_long_notional = current_long_notional
        projected_cash = snap.cash

        pre_trade_long_pct = current_long_notional / netliq if netliq > 0 else 0.0

        for p in plans:
            # SELL et HOLD : toujours approuvés
            if p.action in ("SELL", "HOLD"):
                approved.append(p)
                if p.action == "SELL":
                    projected_long_notional = max(0.0, projected_long_notional - p.est_notional)
                    projected_cash += p.est_notional
                continue

            # À partir d'ici : action == BUY

            # Règle 0 — Kill switch / sell-only
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

            # Règle 2 — Exposition nette longue (régime-ajustée)
            new_long_pct = (projected_long_notional + p.est_notional) / netliq if netliq > 0 else 1.0
            if new_long_pct > effective_max_long:
                rejected.append(RejectedPlan(
                    p,
                    f"net long post-trade {new_long_pct:.1%} > max {effective_max_long:.0%}"
                    + (f" (régime {gmm_regime})" if gmm_regime else ""),
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

            # Règle 4 — Liquidité ADV (< 1 % du volume journalier moyen)
            if adv_map:
                adv = adv_map.get(p.symbol, 0.0)
                if adv > 0 and p.last_price > 0:
                    target_shares = p.est_notional / p.last_price
                    adv_pct = target_shares / adv
                    if adv_pct > 0.01:
                        rejected.append(RejectedPlan(
                            p,
                            f"liquidité ADV: {adv_pct:.1%} du volume journalier (max 1%)",
                        ))
                        continue

            # Plan approuvé
            approved.append(p)
            projected_long_notional += p.est_notional
            projected_cash -= p.est_notional

        post_trade_long_pct = projected_long_notional / netliq if netliq > 0 else 0.0
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

        try:
            from src.notify.telegram import send_message
            if level == 3:
                msg = (
                    f"🔴 CIRCUIT BREAKER — URGENCE — Milan Capital\n"
                    f"Drawdown depuis pic : {dd:.1%}\n"
                    f"Peak NetLiq : ${peak:,.0f} → ${netliq:,.0f}\n"
                    f"⛔ SELL-ONLY MODE actif automatiquement.\n"
                    f"Reset manuel requis."
                )
            else:
                limit_str = f"max_net_long → {cb_limit:.0%}" if cb_limit else ""
                msg = (
                    f"{icon} CIRCUIT BREAKER — {name} — Milan Capital\n"
                    f"Drawdown depuis pic : {dd:.1%}\n"
                    f"Peak NetLiq : ${peak:,.0f} → ${netliq:,.0f}\n"
                    f"{limit_str}"
                )
            send_message(msg)
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
        
