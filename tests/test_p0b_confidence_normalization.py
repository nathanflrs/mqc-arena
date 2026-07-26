# tests/test_p0b_confidence_normalization.py
"""
P0(b) — Normalisation intra-agent de la confidence (min-max scaling).

Problème : chaque agent a sa propre échelle native.
  BuffettAgent : médiane 0.785 (range 0.35-0.90)
  MeanReversionAgent : médiane 0.316 (range 0.30-0.90)

Sans normalisation, BuffettAgent gagne structurellement à conviction égale.
Avec normalisation, c'est la conviction RELATIVE (rang dans l'historique) qui compte.

Corrige l'échelle, pas la calibration : conf=0.8 normalisé ne signifie pas
80% de taux de succès, juste que l'agent est à 80% de son maximum historique.
"""
from __future__ import annotations

import pytest

from src.agents.base import AgentSignal
from src.arena.normalizer import ConfidenceNormalizer
from src.arena.selector import select_best


def _sig(agent, action, confidence, target_weight=0.10, symbol="SPY"):
    return AgentSignal(
        agent_name=agent,
        symbol=symbol,
        action=action,
        confidence=confidence,
        target_weight=target_weight,
    )


# ── Tests ─────────────────────────────────────────────────────────────────────

def test_normalization_corrects_scale_bias():
    """
    HighAgent émet nativement 0.80-0.90 ; ici à 0.83 = bas de sa fourchette (30%).
    LowAgent émet nativement 0.10-0.50 ; ici à 0.40 = haut de sa fourchette (75%).

    Sans normalisation : HighAgent gagne (0.83 × tw > 0.40 × tw).
    Avec normalisation : LowAgent gagne (75% de conviction > 30%).
    """
    stats = {
        "HighAgent": (0.80, 0.90),
        "LowAgent": (0.10, 0.50),
    }
    normalizer = ConfidenceNormalizer(stats)

    signals = [
        _sig("HighAgent", "BUY", 0.83),
        _sig("LowAgent",  "BUY", 0.40),
    ]

    # Sans normalisation : HighAgent score = 0.83×0.10 = 0.083 > LowAgent 0.40×0.10 = 0.040
    winner_raw = select_best(signals)
    assert winner_raw is not None
    assert winner_raw.agent_name == "HighAgent", (
        "Sans normalisation, l'agent à haute échelle native doit dominer"
    )

    # Avec normalisation :
    #   HighAgent  conf_norm = (0.83-0.80)/(0.90-0.80) = 0.30 → score = 0.030
    #   LowAgent   conf_norm = (0.40-0.10)/(0.50-0.10) = 0.75 → score = 0.075
    signals_norm = normalizer.normalize_all(signals)
    winner_norm = select_best(signals_norm)
    assert winner_norm is not None
    assert winner_norm.agent_name == "LowAgent", (
        "Après normalisation, l'agent à conviction relative plus élevée doit gagner"
    )


def test_constant_agent_unchanged():
    """Un agent à signal constant (std=0) n'est pas normalisé — intervalle dégénéré."""
    stats = {"VariableAgent": (0.30, 0.90)}  # ConstantAgent absent → inchangé
    normalizer = ConfidenceNormalizer(stats)

    sig = _sig("ConstantAgent", "HOLD", 0.30)
    result = normalizer.normalize(sig)
    assert result.confidence == 0.30


def test_normalize_clips_to_unit_interval():
    """Un signal hors de l'intervalle historique est clampé à [0, 1]."""
    stats = {"Agent": (0.40, 0.80)}
    normalizer = ConfidenceNormalizer(stats)

    above = _sig("Agent", "BUY", 0.95)
    below = _sig("Agent", "BUY", 0.20)

    assert normalizer.normalize(above).confidence == 1.0
    assert normalizer.normalize(below).confidence == 0.0


def test_normalization_preserves_raw_signal_immutable():
    """normalize() retourne une copie — le signal original n'est pas modifié."""
    stats = {"Agent": (0.30, 0.90)}
    normalizer = ConfidenceNormalizer(stats)

    original = _sig("Agent", "BUY", 0.60)
    normalized = normalizer.normalize(original)

    assert original.confidence == 0.60
    assert normalized.confidence == pytest.approx(0.50, abs=1e-6)


def test_from_frozen_json_skips_constant_agents(tmp_path):
    """from_frozen_json() ignore les agents avec hi==lo (signal constant)."""
    import json

    stats_file = tmp_path / "normalizer_stats.json"
    stats_file.write_text(json.dumps({
        "version": "test",
        "method": "min_max",
        "agents": {
            "VolAgent": {"lo": 0.30, "hi": 0.30},   # constant → exclu
            "VarAgent": {"lo": 0.40, "hi": 0.80},   # valide
        }
    }))

    norm = ConfidenceNormalizer.from_frozen_json(str(stats_file))
    assert "VolAgent" not in norm._stats
    assert "VarAgent" in norm._stats
    assert norm._stats["VarAgent"] == (0.40, 0.80)


def test_from_frozen_json_missing_file_returns_passthrough():
    """Sans fichier gelé, le normalizer est en mode passthrough (ne crash pas)."""
    norm = ConfidenceNormalizer.from_frozen_json("/nonexistent/path.json")
    sig = AgentSignal("AnyAgent", "SPY", "BUY", 0.60, 0.10)
    assert norm.normalize(sig).confidence == 0.60
