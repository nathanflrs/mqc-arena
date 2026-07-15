# src/risk/assistant.py
"""
Risk Assistant IA — répond en français aux questions sur le portfolio.

Construit un contexte temps-réel (positions, VaR, CB, signaux, régime)
depuis les fichiers logs/ puis appelle Claude pour répondre.
"""
from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

_LOGS = Path("logs")
_MODEL = "claude-haiku-4-5-20251001"
_MAX_TOKENS = 1024
_SYSTEM_PERSONA = (
    "Tu es Milan, l'assistant IA de Milan Capital — un hedge fund 100% automatisé "
    "multi-agents. Tu réponds en français, de façon concise et professionnelle. "
    "Tu as accès aux données temps-réel du fund : positions, VaR, circuit breaker, "
    "signaux des agents et régime de marché. "
    "Réponds directement à la question posée. Si une donnée manque, dis-le clairement."
)


# ── Context dataclass ─────────────────────────────────────────────────────────

@dataclass
class PortfolioContext:
    netliq:          float                    = 0.0
    cash:            float                    = 0.0
    positions:       Dict[str, float]         = field(default_factory=dict)
    regime:          str                      = "unknown"
    cb_state:        Dict[str, Any]           = field(default_factory=dict)
    var_data:        Dict[str, Any]           = field(default_factory=dict)
    recent_signals:  List[Dict[str, Any]]     = field(default_factory=list)
    last_plan:       List[Dict[str, Any]]     = field(default_factory=list)


# ── Context builder ───────────────────────────────────────────────────────────

def build_context(logs_dir: Path = _LOGS) -> PortfolioContext:
    ctx = PortfolioContext()

    # Circuit breaker
    cb_path = logs_dir / "circuit_breaker.json"
    if cb_path.exists():
        try:
            ctx.cb_state = json.loads(cb_path.read_text())
        except Exception:
            pass

    # VaR
    var_path = logs_dir / "var_latest.json"
    if var_path.exists():
        try:
            ctx.var_data = json.loads(var_path.read_text())
        except Exception:
            pass

    # Signals (last decision per ticker)
    dec_path = logs_dir / "decisions.csv"
    if dec_path.exists():
        try:
            import csv, io
            rows = list(csv.DictReader(io.StringIO(dec_path.read_text())))
            # Keep last row per symbol
            seen: Dict[str, dict] = {}
            for r in rows:
                seen[r.get("symbol", "")] = r
            ctx.recent_signals = list(seen.values())
            # Extract regime from last row
            if rows:
                ctx.regime = rows[-1].get("regime", "unknown")
        except Exception:
            pass

    # Order plan (last 10 entries)
    plan_path = logs_dir / "order_plan.csv"
    if plan_path.exists():
        try:
            import csv, io
            rows = list(csv.DictReader(io.StringIO(plan_path.read_text())))
            ctx.last_plan = rows[-10:] if rows else []
        except Exception:
            pass

    # Allocator cache → netliq / positions approximation
    alloc_path = logs_dir / "allocator_cache.json"
    if alloc_path.exists():
        try:
            alloc = json.loads(alloc_path.read_text())
            ctx.netliq = float(alloc.get("netliq", 0.0))
        except Exception:
            pass

    return ctx


# ── Prompt builder ────────────────────────────────────────────────────────────

def _build_context_block(ctx: PortfolioContext) -> str:
    lines: List[str] = ["=== CONTEXTE PORTFOLIO ==="]

    # Portfolio summary
    lines.append(f"NetLiq: ${ctx.netliq:,.0f}" if ctx.netliq else "NetLiq: inconnu")
    lines.append(f"Régime de marché: {ctx.regime.upper()}")

    # Circuit breaker
    if ctx.cb_state:
        dd = ctx.cb_state.get("drawdown", 0.0)
        triggered = ctx.cb_state.get("triggered", False)
        level = ctx.cb_state.get("level", 0)
        lines.append(
            f"Circuit Breaker: niveau {level} | drawdown={dd:.1%}"
            + (" | SELL-ONLY ACTIF" if triggered else "")
        )
    else:
        lines.append("Circuit Breaker: données absentes")

    # VaR
    if ctx.var_data:
        v = ctx.var_data
        lines.append(
            f"VaR 1-jour: 95%={v.get('var_95_pct', 0):.2%} (${v.get('var_95_usd', 0):,.0f}) | "
            f"99%={v.get('var_99_pct', 0):.2%} (${v.get('var_99_usd', 0):,.0f}) | "
            f"CVaR 99%=${v.get('cvar_99_usd', 0):,.0f} | "
            f"{v.get('n_positions', 0)} positions × {v.get('n_days', 0)}j"
        )
    else:
        lines.append("VaR: données absentes (aucune position ou calcul non effectué)")

    # Signals
    if ctx.recent_signals:
        lines.append("\nDerniers signaux par ticker:")
        for s in ctx.recent_signals:
            sym    = s.get("symbol", "?")
            agent  = s.get("winner_agent", "?")
            action = s.get("action", "?")
            conf   = s.get("confidence", "?")
            reason = s.get("reason", "")[:80]
            lines.append(f"  {sym}: {action} | agent={agent} | conf={conf} | {reason}")
    else:
        lines.append("\nSignaux: aucun signal récent disponible")

    # Last plan
    if ctx.last_plan:
        lines.append(f"\nDernier plan d'ordres ({len(ctx.last_plan)} ligne(s)):")
        for p in ctx.last_plan[-5:]:
            sym    = p.get("symbol", "?")
            action = p.get("action", "?")
            reason = p.get("reason", "")[:60]
            lines.append(f"  {sym}: {action} | {reason}")

    return "\n".join(lines)


# ── Assistant ─────────────────────────────────────────────────────────────────

class RiskAssistant:
    """
    Appelle Claude avec le contexte portfolio pour répondre à une question.

    Inject `client` dans les tests pour éviter les appels réseau réels.
    """

    def __init__(self, client=None) -> None:
        self._client = client

    def _get_client(self):
        if self._client is None:
            import anthropic
            self._client = anthropic.Anthropic()
        return self._client

    def ask(self, question: str, ctx: PortfolioContext) -> str:
        """
        Pose une question à Claude avec le contexte portfolio.
        Retourne la réponse en texte ou un message d'erreur.
        """
        if not question.strip():
            return "Question vide — pose-moi une question sur le portfolio."

        context_block = _build_context_block(ctx)
        user_message  = f"{context_block}\n\n=== QUESTION ===\n{question}"

        try:
            client = self._get_client()
            msg = client.messages.create(
                model      = _MODEL,
                max_tokens = _MAX_TOKENS,
                system     = _SYSTEM_PERSONA,
                messages   = [{"role": "user", "content": user_message}],
            )
            return msg.content[0].text.strip()
        except Exception as exc:
            logger.warning("RiskAssistant: Claude call failed — %s", exc)
            return f"Erreur lors de l'appel à Claude : {exc}"
