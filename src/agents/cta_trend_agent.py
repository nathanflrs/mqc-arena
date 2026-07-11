# src/agents/cta_trend_agent.py
"""
CTA Trend-Following Agent — Milan Capital
==========================================
Agent long/short multi-actifs sur un univers de 6 ETFs liquides.
Stratégie inspirée des CTA systématiques (Winton / Man AHL style) :
  - Signal dual-timeframe (mom63j + mom126j) filtré par SMA200 + ADX>20
  - Vol targeting : taille = target_vol / realized_vol (20j ann.)
  - Catégorie de risque distincte : 'cta_trend' (exclu du budget net-long)
  - Sorties : ADX<15 exit, retournement direct L→S, stop-loss 10%

STEPS 1-4 — Signal dual-timeframe + vol targeting + exits + reversals
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from src.agents.base import AgentSignal, BaseAgent, MarketState

# ── Univers CTA ────────────────────────────────────────────────────────────────
# SPY, QQQ, GLD sont dans WATCHLIST ; TLT, UUP, DBC y sont ABSENTS.
# Le runner doit télécharger TLT/UUP/DBC séparément et les passer via data=.
CTA_UNIVERSE: List[str] = ["SPY", "QQQ", "TLT", "GLD", "UUP", "DBC"]


# ── Configuration ──────────────────────────────────────────────────────────────

@dataclass
class CTATrendConfig:
    # Signal — fenêtres momentum
    mom_fast_days: int = 63    # momentum 3 mois
    mom_slow_days: int = 126   # momentum 6 mois

    # Filtre tendance
    sma_long_days: int = 200   # SMA long terme

    # ADX — requires real HLC data; absence → FLAT, pas de fallback
    adx_period: int = 14
    adx_entry: float = 20.0   # seuil entrée — en dessous → FLAT
    adx_exit: float = 15.0    # seuil sortie (Step 4)

    # Vol targeting
    vol_target_pct: float = 0.10   # 10% annualisé cible par position
    vol_lookback_days: int = 20    # fenêtre realized vol (jours de trading)
    max_position_pct: float = 0.15 # plafond 15% NAV par position

    # Stop-loss (Step 4)
    stop_loss_pct: float = 0.10    # coupe à -10% par rapport au prix d'entrée

    # Données minimales requises
    min_history: int = 230  # 200 (SMA) + 30 de marge


# ── Dataclass interne ──────────────────────────────────────────────────────────

@dataclass
class _CTASignal:
    """Résultat interne brut avant conversion en AgentSignal."""
    symbol: str
    direction: str        # 'long' | 'short' | 'flat' | 'hold'
    #   long  → BUY  (ouvrir/maintenir long)
    #   short → SELL (ouvrir/maintenir short)
    #   flat  → EXIT : fermer la position si ouverte (signal flip ou ADX effondré)
    #   hold  → HOLD : maintenir l'état courant sans trader (zone neutre ADX)
    mom_fast: float
    mom_slow: float
    adx: float
    sma200: float
    last_price: float
    realized_vol: float   # vol annualisée réalisée (0 si non calculable)
    target_weight: float  # poids signé : +N long, -N short, 0 flat/hold
    confidence: float
    reason: str
    meta: dict = field(default_factory=dict)


# ── Agent ──────────────────────────────────────────────────────────────────────

class CTATrendAgent(BaseAgent):
    """
    CTA trend-following long/short.
    Intégré dans l'Arena pour SPY/QQQ/GLD via le loop WATCHLIST standard.
    Le runner doit boucler séparément sur TLT, UUP, DBC (hors WATCHLIST).
    """

    name = "CTATrendAgent"
    CTA_UNIVERSE: List[str] = CTA_UNIVERSE

    def __init__(self, config: Optional[CTATrendConfig] = None) -> None:
        self.cfg = config or CTATrendConfig()

    # ── Indicateurs ───────────────────────────────────────────────────────────

    def _adx(self, data: pd.DataFrame) -> float:
        """
        Average Directional Index (0-100).
        Pré-condition : data contient High, Low, Close (vérifié par l'appelant).
        Retourne 0.0 en cas d'erreur de calcul.
        """
        try:
            high  = pd.to_numeric(data["High"],  errors="coerce")
            low   = pd.to_numeric(data["Low"],   errors="coerce")
            close = pd.to_numeric(data["Close"], errors="coerce")
            period = self.cfg.adx_period

            plus_dm  = high.diff().clip(lower=0)
            minus_dm = (-low.diff()).clip(lower=0)

            tr = pd.concat([
                high - low,
                (high - close.shift()).abs(),
                (low  - close.shift()).abs(),
            ], axis=1).max(axis=1)

            atr      = tr.rolling(period).mean()
            plus_di  = 100 * (plus_dm.rolling(period).mean() / atr)
            minus_di = 100 * (minus_dm.rolling(period).mean() / atr)
            dx       = 100 * (plus_di - minus_di).abs() / (plus_di + minus_di)
            return float(dx.rolling(period).mean().iloc[-1])
        except Exception:
            return 0.0

    def _realized_vol(self, close: pd.Series) -> float:
        """
        Volatilité réalisée annualisée sur les vol_lookback_days derniers jours.
        Retourne 0.0 si moins de 5 rendements disponibles.
        """
        rets = close.pct_change().dropna().tail(self.cfg.vol_lookback_days)
        if len(rets) < 5:
            return 0.0
        return float(rets.std() * np.sqrt(252))

    # ── Signal dual-timeframe + vol targeting ─────────────────────────────────

    def _compute_signal(
        self,
        symbol: str,
        data: pd.DataFrame,
        current_qty: float = 0.0,
    ) -> _CTASignal:
        """
        Étape 1 — direction :
          LONG  : mom63>0 ET mom126>0 ET price>SMA200 ET ADX≥adx_entry
          SHORT : mom63<0 ET mom126<0 ET price<SMA200 ET ADX≥adx_entry
          FLAT  : désaccord temporal, ADX faible, ou données HLC absentes

        Étape 2 — taille (vol targeting) :
          target_weight = min(vol_target / realized_vol, max_position_pct)
          signed : +weight LONG, -weight SHORT
          Si realized_vol = 0 → FLAT (impossible de sizer)

        Étape 4 — sorties :
          ADX < adx_exit (15) → FLAT avec raison "EXIT ADX" (sortie de position)
          Zone neutre adx_exit ≤ ADX < adx_entry → FLAT "zone neutre"
          Reversal L→S / S→L : la direction inversée suffit ; generate_signal +
          cta_plan_from_signal gèrent le delta en un seul ordre.
        """
        def _flat(reason: str, **kw) -> _CTASignal:
            return _CTASignal(
                symbol=symbol, direction="flat",
                mom_fast=kw.get("mom_fast", 0.0), mom_slow=kw.get("mom_slow", 0.0),
                adx=kw.get("adx", 0.0), sma200=kw.get("sma200", 0.0),
                last_price=kw.get("last_price", 0.0),
                realized_vol=0.0, target_weight=0.0,
                confidence=0.0, reason=reason,
                meta=kw.get("meta", {}),
            )

        # ── Garde données ─────────────────────────────────────────────────────
        if data is None or data.empty or "Close" not in data.columns:
            return _flat("données absentes")

        close = pd.to_numeric(data["Close"], errors="coerce").dropna()
        n = len(close)
        min_needed = max(self.cfg.mom_slow_days + 1, self.cfg.sma_long_days)
        if n < min_needed:
            return _flat(f"historique insuffisant ({n}j < {min_needed}j)")

        # ── HLC requis pour ADX — absence = FLAT, pas de fallback ─────────────
        has_hlc = all(c in data.columns for c in ("High", "Low", "Close"))
        if not has_hlc:
            return _flat("HLC absent — ADX non calculable, pas de trade")

        # ── Indicateurs ───────────────────────────────────────────────────────
        px     = float(close.iloc[-1])
        mom_f  = float(px / close.iloc[-(self.cfg.mom_fast_days + 1)] - 1.0)
        mom_s  = float(px / close.iloc[-(self.cfg.mom_slow_days + 1)] - 1.0)
        sma200 = float(close.rolling(self.cfg.sma_long_days).mean().iloc[-1])
        adx    = self._adx(data)

        meta = {
            "mom_fast":   round(mom_f,  4),
            "mom_slow":   round(mom_s,  4),
            "sma200":     round(sma200, 2),
            "adx":        round(adx,    2),
            "last_price": round(px,     2),
            "strategy":   "cta_trend",
        }
        shared = dict(mom_fast=mom_f, mom_slow=mom_s, adx=adx,
                      sma200=sma200, last_price=px, meta=meta)

        # ── ADX exit : effondrement de tendance ──────────────────────────────
        # En dessous de adx_exit (15) → sortie si en position, FLAT sinon
        if adx < self.cfg.adx_exit:
            return _flat(
                f"EXIT ADX: ADX={adx:.1f} < {self.cfg.adx_exit} — tendance effondrée",
                **shared,
            )

        # ── Zone neutre ADX ───────────────────────────────────────────────────
        # adx_exit ≤ ADX < adx_entry → maintenir position existante sans
        # ouvrir de nouvelle entrée. Direction='hold' pour distinguer du 'flat'
        # (exit) : generate_signal retourne HOLD au lieu de fermer la position.
        if adx < self.cfg.adx_entry:
            return _CTASignal(
                symbol=symbol, direction="hold",
                mom_fast=meta.get("mom_fast", 0.0),
                mom_slow=meta.get("mom_slow", 0.0),
                adx=adx, sma200=sma200, last_price=px,
                realized_vol=0.0, target_weight=0.0,
                confidence=0.0,
                reason=f"ADX={adx:.1f} zone neutre [{self.cfg.adx_exit:.0f}–{self.cfg.adx_entry:.0f}] — position maintenue, pas de nouvelle entrée",
                meta=meta,
            )

        # ── Signal dual-timeframe ─────────────────────────────────────────────
        both_up   = mom_f > 0 and mom_s > 0
        both_down = mom_f < 0 and mom_s < 0
        above_sma = px > sma200
        below_sma = px < sma200

        if both_up and above_sma:
            direction = "long"
        elif both_down and below_sma:
            direction = "short"
        else:
            return _flat(
                f"FLAT: désaccord temporal — "
                f"mom{self.cfg.mom_fast_days}j={mom_f:+.1%} "
                f"mom{self.cfg.mom_slow_days}j={mom_s:+.1%} "
                f"SMA200={sma200:.0f}",
                **shared,
            )

        # ── Vol targeting (Step 2) ────────────────────────────────────────────
        realized_vol = self._realized_vol(close)
        if realized_vol <= 0.0:
            return _flat("vol réalisée = 0 — sizing impossible", **shared)

        raw_weight    = self.cfg.vol_target_pct / realized_vol
        capped_weight = min(raw_weight, self.cfg.max_position_pct)
        signed_weight = capped_weight if direction == "long" else -capped_weight

        # ADX-scaled confidence : 0 à adx_entry, 1 à adx_entry+30
        confidence = min(1.0, (adx - self.cfg.adx_entry) / 30.0)

        meta["realized_vol"] = round(realized_vol, 4)
        meta["target_weight"] = round(signed_weight, 4)

        reason = (
            f"{'LONG' if direction == 'long' else 'SHORT'}: "
            f"mom{self.cfg.mom_fast_days}j={mom_f:+.1%} "
            f"mom{self.cfg.mom_slow_days}j={mom_s:+.1%} "
            f"SMA200={sma200:.0f} ADX={adx:.1f} "
            f"vol={realized_vol:.1%} w={signed_weight:+.1%}"
        )

        return _CTASignal(
            symbol=symbol, direction=direction,
            mom_fast=mom_f, mom_slow=mom_s,
            adx=adx, sma200=sma200, last_price=px,
            realized_vol=realized_vol, target_weight=signed_weight,
            confidence=confidence, reason=reason, meta=meta,
        )

    # ── Interface BaseAgent ────────────────────────────────────────────────────

    def generate_signal(
        self,
        state: MarketState,
        portfolio: Dict[str, float],
        regime: Optional[str] = None,
        data: Optional[pd.DataFrame] = None,
    ) -> AgentSignal:
        """
        Retourne un AgentSignal CTA pour le symbole courant.

        target_weight signé :  +N → long N% NAV  |  -N → short N% NAV  |  0 → flat
        action : LONG → BUY | SHORT → SELL | FLAT → HOLD
        """
        if state.symbol not in self.CTA_UNIVERSE:
            return AgentSignal(
                agent_name=self.name,
                symbol=state.symbol,
                action="HOLD",
                confidence=0.0,
                target_weight=0.0,
                reason="Symbole hors univers CTA",
                meta={"strategy": "cta_trend"},
            )

        # current_qty > 0 : long, < 0 : short, 0 : flat
        current_qty = float(portfolio.get(state.symbol, 0))

        raw = self._compute_signal(state.symbol, data, current_qty)

        if raw.direction == "long":
            action = "BUY"
            tw     = raw.target_weight          # positif
            reason = raw.reason
        elif raw.direction == "short":
            action = "SELL"
            tw     = raw.target_weight          # négatif
            reason = raw.reason
        elif raw.direction == "hold":
            # Zone neutre ADX : maintenir l'état sans trader
            action = "HOLD"
            tw     = 0.0
            reason = raw.reason
        else:
            # direction="flat" → EXIT : fermer la position si elle existe
            tw = 0.0
            if current_qty > 0:
                action = "SELL"                 # fermeture long
                reason = raw.reason + " → fermeture long"
            elif current_qty < 0:
                action = "BUY"                  # couverture short
                reason = raw.reason + " → couverture short"
            else:
                action = "HOLD"                 # déjà flat
                reason = raw.reason

        # Stop-loss au niveau position : délégué au runner (STOP_LOSS_PCT)
        # L'exit ADX (< adx_exit=15) couvre la sortie modèle.

        return AgentSignal(
            agent_name=self.name,
            symbol=state.symbol,
            action=action,
            confidence=raw.confidence,
            target_weight=tw,
            reason=reason,
            meta={**raw.meta, "cta_direction": raw.direction, "current_qty": current_qty},
        )
