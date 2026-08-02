# src/analysis/agent_edge.py
"""
Milan Capital — Mesure d'edge par agent, avec puissance statistique.

Pourquoi ce module existe
-------------------------
`src/analysis/edge_audit.py` posait la bonne question et refusait honnêtement de
conclure : 11 dates de marché contre 60 requises. Le moteur de replay
(`src/backtest/system_backtest.py`) permet aujourd'hui de rejouer les mêmes
agents sur ~950 séances, ce qui donne enfin la puissance nécessaire.

Deux corrections statistiques par rapport à edge_audit
------------------------------------------------------

1. **Regroupement par date.** edge_audit calculait un intervalle de Wilson sur
   le nombre de *signaux*, alors que son propre texte reconnaissait qu'« un run
   où BuffettAgent dit BUY sur 12 actifs représente 1 observation de marché,
   pas 12 ». Les actifs bougent ensemble : traiter les signaux comme
   indépendants divise l'intervalle par ~√12. On utilise ici un bootstrap
   par blocs sur les **dates**, qui respecte la corrélation intra-journalière.

2. **Hypothèse nulle = taux de base, pas 0,5.** edge_audit testait contre une
   pièce équilibrée. Or si un actif monte de plus de 30 bps 55 % des jours, un
   agent qui dit toujours BUY affiche 55 % sans contenir la moindre
   information. La référence correcte est le taux de succès **inconditionnel**
   de la même action sur le même univers et la même période. L'écart à cette
   référence est ce qu'apporte réellement l'agent — le reste est du bêta.

Ce que ce module ne fait pas
----------------------------
Il ne corrige pas le sur-apprentissage humain : les seuils des agents (RSI,
ADX, fenêtres) ont été choisis en regardant ces mêmes marchés. Un edge mesuré
ici est donc une borne haute, jamais une promesse hors échantillon.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

# Seuil de matérialité : ~la moitié d'un aller-retour IBKR large-cap.
# Un mouvement en deçà n'est pas exploitable, il est absorbé par la friction.
MATERIALITY = 0.0030          # 30 bps, en rendement log
HORIZONS: Dict[str, int] = {"H1": 1, "H5": 5, "H20": 20}

# Nombre minimal de dates indépendantes avant d'autoriser une conclusion.
# Repris tel quel d'edge_audit — le seuil était bon, c'est l'échantillon qui
# manquait.
MIN_DATES = 60
N_BOOTSTRAP = 2000
SEED = 20260802


def forward_log_returns(close: pd.Series, horizon: int) -> pd.Series:
    """log(P[t+h] / P[t]), aligné sur t. NaN sur les h dernières barres."""
    return np.log(close.shift(-horizon) / close)


def label_success(action: str, fwd: float) -> Optional[bool]:
    """BUY correct si fwd > +μ ; SELL correct si fwd < −μ. HOLD n'est pas jugé ici."""
    if fwd is None or not np.isfinite(fwd):
        return None
    if action == "BUY":
        return bool(fwd > MATERIALITY)
    if action == "SELL":
        return bool(fwd < -MATERIALITY)
    return None


def base_rates(
    data: Dict[str, pd.DataFrame],
    symbols: Sequence[str],
    dates: pd.DatetimeIndex,
    horizon: int,
) -> Tuple[float, float]:
    """
    Taux de succès inconditionnels (BUY, SELL) sur l'univers et la période.

    C'est la performance d'un agent qui dirait toujours la même chose. Tout
    agent doit être jugé par rapport à cette référence, pas par rapport à 0,5 :
    sur un marché haussier, `base_buy` peut dépasser 0,55 sans qu'aucune
    information n'y soit pour quelque chose.
    """
    ups, downs = [], []
    for sym in symbols:
        df = data.get(sym)
        if df is None:
            continue
        fwd = forward_log_returns(pd.to_numeric(df["Close"], errors="coerce"), horizon)
        fwd = fwd.reindex(dates).dropna()
        if fwd.empty:
            continue
        ups.append((fwd > MATERIALITY).to_numpy())
        downs.append((fwd < -MATERIALITY).to_numpy())
    if not ups:
        return (float("nan"), float("nan"))
    return (
        float(np.concatenate(ups).mean()),
        float(np.concatenate(downs).mean()),
    )


@dataclass
class AgentEdge:
    agent: str
    horizon: str
    n_signals: int
    n_dates: int
    hit_rate: float
    expected_rate: float          # taux de base, pondéré par le mix BUY/SELL de l'agent
    excess: float                 # hit_rate − expected_rate
    ci_lo: float                  # IC 95 % sur l'excès, bootstrap par date
    ci_hi: float
    verdict: str
    n_buy: int = 0
    n_sell: int = 0

    @property
    def is_significant(self) -> bool:
        """L'intervalle exclut zéro : l'excès n'est pas attribuable au hasard."""
        return np.isfinite(self.ci_lo) and (self.ci_lo > 0 or self.ci_hi < 0)


def _bootstrap_excess(
    df: pd.DataFrame, base_buy: float, base_sell: float, n_boot: int = N_BOOTSTRAP,
) -> Tuple[float, float]:
    """
    IC 95 % de l'excès, par rééchantillonnage **des dates** avec remise.

    Rééchantillonner les signaux traiterait 12 actifs d'un même jour comme 12
    observations. On tire des journées entières, ce qui conserve la corrélation
    inter-actifs à l'intérieur d'une date.
    """
    # On agrège UNE fois par date, puis on rééchantillonne des indices. Une
    # boucle Python sur (tirages × dates) coûtait 12 agents × 3 horizons ×
    # 2000 × ~950 opérations : plusieurs dizaines de minutes. Ici tout se joue
    # dans numpy sur trois vecteurs de longueur n_dates.
    per_date = df.groupby("date", sort=False).agg(
        hits=("success", "sum"),
        n=("success", "size"),
        n_buy=("action", lambda a: (a == "BUY").sum()),
    )
    if len(per_date) < 2:
        return (float("nan"), float("nan"))

    hits = per_date["hits"].to_numpy(dtype=float)
    counts = per_date["n"].to_numpy(dtype=float)
    n_buy = per_date["n_buy"].to_numpy(dtype=float)
    exp = n_buy * base_buy + (counts - n_buy) * base_sell

    rng = np.random.default_rng(SEED)
    idx = rng.integers(0, len(per_date), size=(n_boot, len(per_date)))

    boot_hits = hits[idx].sum(axis=1)
    boot_exp = exp[idx].sum(axis=1)
    boot_n = counts[idx].sum(axis=1)
    out = np.where(boot_n > 0, (boot_hits - boot_exp) / boot_n, np.nan)

    return (float(np.nanpercentile(out, 2.5)), float(np.nanpercentile(out, 97.5)))


def compute_agent_edge(
    signals: pd.DataFrame,
    data: Dict[str, pd.DataFrame],
    symbols: Sequence[str],
    horizons: Dict[str, int] = HORIZONS,
) -> List[AgentEdge]:
    """
    `signals` : colonnes date, symbol, agent, action, confidence.
    Retourne un AgentEdge par (agent, horizon).
    """
    results: List[AgentEdge] = []
    all_dates = pd.DatetimeIndex(sorted(signals["date"].unique()))

    for tag, h in horizons.items():
        # Rendements forward, joints aux signaux
        fwd_map = {}
        for sym in symbols:
            df = data.get(sym)
            if df is None:
                continue
            fwd_map[sym] = forward_log_returns(
                pd.to_numeric(df["Close"], errors="coerce"), h)

        s = signals.copy()
        s["fwd"] = [
            fwd_map.get(sym, pd.Series(dtype=float)).get(dt, np.nan)
            for sym, dt in zip(s["symbol"], s["date"])
        ]
        base_buy, base_sell = base_rates(data, symbols, all_dates, h)

        for agent in sorted(s["agent"].unique()):
            g = s[(s["agent"] == agent) & s["action"].isin(["BUY", "SELL"])].copy()
            g["success"] = [label_success(a, f) for a, f in zip(g["action"], g["fwd"])]
            g = g.dropna(subset=["success"])
            g["success"] = g["success"].astype(float)

            n, n_dates = len(g), g["date"].nunique()
            if n == 0:
                results.append(AgentEdge(
                    agent, tag, 0, 0, float("nan"), float("nan"), float("nan"),
                    float("nan"), float("nan"), "aucun signal directionnel"))
                continue

            n_buy = int((g["action"] == "BUY").sum())
            n_sell = int((g["action"] == "SELL").sum())
            hit = float(g["success"].mean())
            expected = (n_buy * base_buy + n_sell * base_sell) / n
            excess = hit - expected
            lo, hi = _bootstrap_excess(g, base_buy, base_sell)

            if n_dates < MIN_DATES:
                verdict = f"échantillon insuffisant ({n_dates}/{MIN_DATES} dates)"
            elif lo > 0:
                verdict = f"edge mesurable (+{excess:.1%} vs base, IC exclut 0)"
            elif hi < 0:
                verdict = f"anti-edge (−{abs(excess):.1%} vs base, IC exclut 0)"
            else:
                verdict = "indistinguable du taux de base"

            results.append(AgentEdge(
                agent=agent, horizon=tag, n_signals=n, n_dates=n_dates,
                hit_rate=hit, expected_rate=expected, excess=excess,
                ci_lo=lo, ci_hi=hi, verdict=verdict, n_buy=n_buy, n_sell=n_sell))

    return results


def calibration_curve(
    signals: pd.DataFrame,
    data: Dict[str, pd.DataFrame],
    symbols: Sequence[str],
    agent: str,
    horizon: int = 5,
    bins: Sequence[float] = (0.0, 0.55, 0.65, 0.75, 0.85, 1.01),
) -> pd.DataFrame:
    """
    Confiance émise vs taux de succès réalisé, par tranche de confiance.

    Un agent calibré produit une courbe croissante : quand il dit 0,90 il a
    plus souvent raison que quand il dit 0,60. Une courbe plate signifie que
    la confiance ne porte aucune information — l'agent peut avoir un edge
    global tout en étant incapable de dire *quand* il est fiable, ce qui rend
    toute pondération par la confiance illusoire.
    """
    fwd_map = {
        sym: forward_log_returns(
            pd.to_numeric(data[sym]["Close"], errors="coerce"), horizon)
        for sym in symbols if sym in data
    }
    g = signals[(signals["agent"] == agent)
                & signals["action"].isin(["BUY", "SELL"])].copy()
    g["fwd"] = [fwd_map.get(s, pd.Series(dtype=float)).get(d, np.nan)
                for s, d in zip(g["symbol"], g["date"])]
    g["success"] = [label_success(a, f) for a, f in zip(g["action"], g["fwd"])]
    g = g.dropna(subset=["success"])
    if g.empty:
        return pd.DataFrame(columns=["bin", "n", "n_dates", "mean_confidence", "hit_rate"])

    g["bin"] = pd.cut(g["confidence"], bins=list(bins), right=False)
    out = g.groupby("bin", observed=True).agg(
        n=("success", "size"),
        n_dates=("date", "nunique"),
        mean_confidence=("confidence", "mean"),
        hit_rate=("success", "mean"),
    ).reset_index()
    return out


def render_table(edges: List[AgentEdge], horizon: str) -> str:
    rows = [e for e in edges if e.horizon == horizon]
    rows.sort(key=lambda e: (-e.excess if np.isfinite(e.excess) else 1))
    lines = [
        f"── Edge par agent — horizon {horizon} "
        f"(succès = |rendement| > {MATERIALITY:.2%}) ──",
        f"{'Agent':<30}{'N':>6}{'dates':>7}{'taux':>8}{'base':>8}{'excès':>9}"
        f"{'IC 95%':>18}",
    ]
    for e in rows:
        if not np.isfinite(e.hit_rate):
            lines.append(f"{e.agent:<30}{'—':>6}{'—':>7}{'—':>8}{'—':>8}{'—':>9}{'—':>18}")
            continue
        ci = f"[{e.ci_lo:+.1%}, {e.ci_hi:+.1%}]"
        mark = " ✅" if e.is_significant and e.excess > 0 else (
               " ❌" if e.is_significant else "")
        lines.append(
            f"{e.agent:<30}{e.n_signals:>6}{e.n_dates:>7}{e.hit_rate:>8.1%}"
            f"{e.expected_rate:>8.1%}{e.excess:>+9.1%}{ci:>18}{mark}")
    return "\n".join(lines)
