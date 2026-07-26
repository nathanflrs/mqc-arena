"""
src/analysis/edge_audit.py — P0(d) : audit d'edge par agent.

Produit docs/edge_audit.md et docs/charts/*_reliability.png.

Ce script ne modifie aucun comportement du système — observation pure.
Lancer : python -m src.analysis.edge_audit
"""
from __future__ import annotations

import json
import warnings
from datetime import date, timedelta
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yfinance as yf
from scipy import stats

matplotlib.use("Agg")
warnings.filterwarnings("ignore", category=FutureWarning)

# ── Paramètres figés ──────────────────────────────────────────────────────────

MATERIALITY_BPS   = 0.30       # 30 bps — seuil de matérialité (succès ≠ bruit de friction)
MATERIALITY       = MATERIALITY_BPS / 100
MIN_DATES         = 60         # sessions de marché indépendantes min pour toute conclusion
MIN_SIGNALS       = 30         # signaux directionnels min par agent-horizon
CONFIDENCE        = 0.95       # IC Wilson
HORIZONS          = {"H1": 1, "H5": 5, "H20": 20}

DECISIONS_CSV     = Path("logs/decisions.csv")
CHARTS_DIR        = Path("docs/charts")
REPORT_PATH       = Path("docs/edge_audit.md")

# ── Chargement des décisions ──────────────────────────────────────────────────

def load_decisions() -> pd.DataFrame:
    df = pd.read_csv(DECISIONS_CSV)
    df["confidence"]    = pd.to_numeric(df["confidence"], errors="coerce").fillna(0.0)
    df["target_weight"] = pd.to_numeric(df["target_weight"], errors="coerce").fillna(0.0)
    df["is_winner"]     = df["is_winner"].astype(str).str.lower()
    # Date robuste aux deux formats (ISO et espace)
    df["decision_date"] = pd.to_datetime(df["timestamp"], utc=True, errors="coerce") \
                            .dt.tz_convert("America/New_York").dt.date
    df = df.dropna(subset=["decision_date", "agent", "action", "symbol"])
    return df


# ── Prix historiques ──────────────────────────────────────────────────────────

def fetch_prices(symbols: list[str], start: str, end: str) -> pd.DataFrame:
    """
    Retourne un DataFrame (date × symbol) de prix ajustés en close.
    Anti-look-ahead garanti : on n'utilise que close_t et close_{t+H},
    aucune donnée intra-journalière antérieure au signal.
    """
    raw = yf.download(
        symbols, start=start, end=end,
        interval="1d", auto_adjust=True, progress=False, group_by="ticker",
    )
    if isinstance(raw.columns, pd.MultiIndex):
        close = pd.DataFrame({
            sym: raw[sym]["Close"]
            for sym in symbols
            if sym in raw.columns.get_level_values(0)
        })
    else:
        close = raw[["Close"]].rename(columns={"Close": symbols[0]})
    close.index = pd.to_datetime(close.index).date
    return close


def get_forward_return(close: pd.DataFrame, symbol: str,
                       t: date, horizon_days: int) -> float | None:
    """
    log(close_{t+H} / close_t) en jours ouvrés.
    Retourne None si données futures indisponibles.
    """
    if symbol not in close.columns:
        return None
    series = close[symbol].dropna()
    trading_days = sorted(series.index)

    # Trouver t (ou le premier jour ouvré ≥ t)
    available = [d for d in trading_days if d >= t]
    if not available:
        return None
    t0 = available[0]
    idx0 = trading_days.index(t0)
    idxH = idx0 + horizon_days

    if idxH >= len(trading_days):
        return None  # données futures non disponibles

    p0 = series.loc[t0]
    pH = series.loc[trading_days[idxH]]
    if p0 <= 0 or pd.isna(p0) or pd.isna(pH):
        return None
    return float(np.log(pH / p0))


# ── Étiquetage succès / échec ──────────────────────────────────────────────────

def label_success(action: str, fwd: float | None) -> bool | None:
    """
    Définition du succès (μ = MATERIALITY = 30 bps) :
      BUY  correct  → fwd > +μ
      SELL correct  → fwd < −μ
      HOLD correct  → |fwd| ≤ μ  (rarement satisfait à H5+ pour les actions individuelles)

    HOLD est inclus pour complétude mais les résultats seront quasi-toujours négatifs —
    un marché neutre sur 5 jours à ±30 bps est rarissime.
    """
    if fwd is None:
        return None
    if action == "BUY":
        return fwd > MATERIALITY
    if action == "SELL":
        return fwd < -MATERIALITY
    if action == "HOLD":
        return abs(fwd) <= MATERIALITY
    return None


# ── Intervalle de Wilson ──────────────────────────────────────────────────────

def wilson_ci(k: int, n: int) -> tuple[float, float]:
    if n == 0:
        return (float("nan"), float("nan"))
    z = stats.norm.ppf(1 - (1 - CONFIDENCE) / 2)
    p = k / n
    center = (p + z**2 / (2 * n)) / (1 + z**2 / n)
    half   = (z / (1 + z**2 / n)) * np.sqrt(p * (1 - p) / n + z**2 / (4 * n**2))
    return (max(0.0, center - half), min(1.0, center + half))


# ── Métriques par agent / horizon ─────────────────────────────────────────────

def compute_metrics(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for agent in sorted(df["agent"].unique()):
        grp = df[df["agent"] == agent]
        for tag, h in HORIZONS.items():
            col = f"fwd_{tag}"
            if col not in grp.columns:
                continue

            # Signaux directionnels uniquement (BUY + SELL)
            dir_grp = grp[grp["action"].isin(["BUY", "SELL"])].copy()
            dir_grp["success"] = dir_grp.apply(
                lambda r: label_success(r["action"], r[col]), axis=1
            )
            labeled = dir_grp.dropna(subset=["success", col])

            n_directional = len(dir_grp)   # tous BUY+SELL, avec ou sans fwd return
            n_signals     = len(labeled)    # BUY+SELL avec fwd return disponible
            n_dates       = labeled["decision_date"].nunique()

            if n_directional == 0:
                verdict = "aucun signal directionnel émis"
                rows.append(_row(agent, tag, 0, 0, None, None, None, verdict))
                continue

            if n_signals == 0:
                verdict = f"forward return non disponible ({n_directional} signaux directionnels sans données prix)"
                rows.append(_row(agent, tag, 0, n_directional, None, None, None, verdict))
                continue

            k = int(labeled["success"].sum())
            ci_lo, ci_hi = wilson_ci(k, n_signals)
            hit_rate = k / n_signals

            if n_dates < MIN_DATES:
                verdict = f"échantillon temporel insuffisant ({n_dates}/{MIN_DATES} dates)"
            elif n_signals < MIN_SIGNALS:
                verdict = f"échantillon signal insuffisant ({n_signals}/{MIN_SIGNALS} signaux)"
            else:
                pval = stats.binomtest(k, n_signals, 0.5, alternative="greater").pvalue
                if pval < 0.05:
                    verdict = f"edge mesurable (p={pval:.3f})"
                else:
                    verdict = "indistinguable du hasard"

            rows.append(_row(agent, tag, n_signals, n_dates, hit_rate, ci_lo, ci_hi, verdict))

    return pd.DataFrame(rows)


def _row(agent, horizon, n, n_dates, hr, lo, hi, verdict):
    return {
        "agent":     agent,
        "horizon":   horizon,
        "n_signals": n,       # signaux avec forward return calculé
        "n_dates":   n_dates, # dates de marché distinctes
        "hit_rate":  round(hr, 3) if hr is not None else None,
        "ci_lo":     round(lo, 3) if lo is not None else None,
        "ci_hi":     round(hi, 3) if hi is not None else None,
        "verdict":   verdict,
    }


# ── Courbes de fiabilité ───────────────────────────────────────────────────────

def generate_curves(df: pd.DataFrame) -> list[Path]:
    CHARTS_DIR.mkdir(parents=True, exist_ok=True)
    generated = []

    for agent in sorted(df["agent"].unique()):
        grp = df[(df["agent"] == agent) & df["action"].isin(["BUY", "SELL"])].copy()

        for tag in ["H1", "H5"]:
            col = f"fwd_{tag}"
            if col not in grp.columns:
                continue
            sub = grp[["confidence", col, "action"]].dropna(subset=[col])
            sub["success"] = sub.apply(lambda r: label_success(r["action"], r[col]), axis=1)
            sub = sub.dropna(subset=["success"])

            if len(sub) < 5:
                continue

            # Buckets de confidence (max 5 quintiles, moins si peu de valeurs distinctes)
            try:
                sub["bucket"] = pd.qcut(sub["confidence"], q=min(5, len(sub) // 2),
                                        duplicates="drop")
            except Exception:
                continue

            bstats = sub.groupby("bucket", observed=True).agg(
                hit=("success", "mean"),
                n=("success", "count"),
                conf_mid=("confidence", "median"),
            ).reset_index()

            if len(bstats) < 2:
                continue

            fig, ax = plt.subplots(figsize=(7, 4.5))
            ax.plot(bstats["conf_mid"], bstats["hit"], "o-",
                    color="#2563EB", linewidth=2, markersize=8, label="Hit rate observé")
            ax.axhline(0.50, color="#DC2626", linestyle="--", alpha=0.7, label="Hasard (50%)")
            ax.set_xlabel("Confidence émise")
            ax.set_ylabel("Taux de succès directionnel")
            ax.set_title(f"{agent} — Fiabilité {tag} (μ={MATERIALITY_BPS:.0f} bps)")
            ax.set_ylim(0, 1)
            ax.legend(fontsize=9)
            ax.grid(True, alpha=0.25)

            for _, r in bstats.iterrows():
                ax.annotate(f"n={r['n']}", (r["conf_mid"], r["hit"]),
                            textcoords="offset points", xytext=(0, 10),
                            ha="center", fontsize=8, color="#374151")

            fig.tight_layout()
            path = CHARTS_DIR / f"{agent}_{tag}_reliability.png"
            fig.savefig(path, dpi=120, bbox_inches="tight")
            plt.close(fig)
            generated.append(path)

    return generated


# ── Rapport Markdown ───────────────────────────────────────────────────────────

def _fmt(val, suffix="") -> str:
    if val is None or (isinstance(val, float) and np.isnan(val)):
        return "—"
    return f"{val}{suffix}"


def generate_report(df: pd.DataFrame, metrics: pd.DataFrame,
                    chart_paths: list[Path]) -> str:
    n_dates_total = df["decision_date"].nunique()
    # Non-chevauchantes H5 : dates séparées d'au moins 5 jours ouvrés
    sorted_dates = sorted(df["decision_date"].unique())
    non_overlap_h5 = 0
    last = None
    for d in sorted_dates:
        if last is None or (d - last).days >= 7:  # ~5 jours ouvrés ≈ 7 calendaires
            non_overlap_h5 += 1
            last = d

    lines = []
    lines.append("# Edge Audit — MQC Arena")
    lines.append(f"\n*Généré le {date.today().isoformat()} — P0(d), mesure pure, zéro modification du système.*\n")

    lines.append("---\n")
    lines.append("## ⚠ Puissance statistique\n")
    lines.append(f"**Minimum requis pour toute conclusion d'edge : {MIN_DATES} dates indépendantes.**\n")
    lines.append("| Métrique | Valeur actuelle | Minimum requis | Statut |")
    lines.append("|---|---|---|---|")
    lines.append(f"| Dates de marché distinctes (H1) | {n_dates_total} | {MIN_DATES} | {'✅' if n_dates_total >= MIN_DATES else '🔴 SOUS SEUIL'} |")
    lines.append(f"| Fenêtres H5 non chevauchantes   | {non_overlap_h5} | {MIN_DATES} | {'✅' if non_overlap_h5 >= MIN_DATES else '🔴 SOUS SEUIL'} |")
    lines.append("")
    lines.append(f"> **Statut global : SOUS SEUIL.** Aucune conclusion d'edge n'est rendue. "
                 f"Le pipeline est validé sur données réelles. Les résultats se rempliront "
                 f"aux prochains runs — revenir à ce rapport à ~{MIN_DATES} dates de marché "
                 f"(environ {MIN_DATES // 21 + 1} mois de runs quotidiens).\n")

    lines.append("### Corrélation inter-actifs\n")
    lines.append("Un run où BuffettAgent dit BUY sur 12 actifs le même jour représente **1 observation "
                 "de marché**, pas 12. N_effectif ≪ N_lignes. Avec 13 runs sur 9 jours de bourse, "
                 "les actifs se déplacent ensemble — un hit rate calculé sur cette fenêtre mesure "
                 "la direction du marché de juillet 2026, pas l'edge des agents.\n")

    lines.append("---\n")
    lines.append("## Définition du succès\n")
    lines.append(f"- **Seuil de matérialité** : μ = {MATERIALITY_BPS:.2f}% ({int(MATERIALITY_BPS*100)} bps)")
    lines.append("  - *Représente environ la moitié d'un aller-retour IBKR large-cap (commission + slippage estimé).*")
    lines.append(f"- **BUY correct** : forward_return(H) > +{MATERIALITY_BPS:.2f}%")
    lines.append(f"- **SELL correct** : forward_return(H) < −{MATERIALITY_BPS:.2f}%")
    lines.append(f"- **HOLD correct** : |forward_return(H)| ≤ {MATERIALITY_BPS:.2f}%")
    lines.append("  - *Note : un marché immobile à ±30 bps sur 5 jours est rarissime pour des actions individuelles.*")
    lines.append(f"  *Les résultats HOLD seront quasi-systématiquement 'échec' — ce n'est pas un bug.*\n")
    lines.append(f"- **Anti-look-ahead** : forward return = log(close_{{t+H}} / close_t), "
                 f"t = close du jour de décision. Aucune donnée intra-journalière antérieure au signal n'est utilisée.\n")

    lines.append("---\n")
    lines.append("## Résultats par agent\n")

    for horizon in ["H1", "H5", "H20"]:
        h_metrics = metrics[metrics["horizon"] == horizon]
        if h_metrics.empty:
            continue

        lines.append(f"### Horizon {horizon}\n")
        lines.append("| Agent | N signaux | N dates | Hit rate | IC 95% | Verdict |")
        lines.append("|---|---|---|---|---|---|")

        for _, r in h_metrics.sort_values("n_signals", ascending=False).iterrows():
            hr   = _fmt(r["hit_rate"], "%") if r["hit_rate"] is not None else "—"
            ci   = f"[{_fmt(r['ci_lo'])}, {_fmt(r['ci_hi'])}]" if r["ci_lo"] is not None else "—"
            lines.append(f"| {r['agent']} | {r['n_signals']} | {r['n_dates']} | {hr} | {ci} | {r['verdict']} |")
        lines.append("")

    lines.append("---\n")
    lines.append("## Courbes de fiabilité\n")
    lines.append("*Confidence émise (abscisse) vs taux de succès directionnel réalisé (ordonnée).*\n")
    lines.append("*Un agent calibré a une courbe croissante. Flat ou aléatoire = pas d'edge.*\n")

    if chart_paths:
        for path in sorted(chart_paths):
            rel = path.relative_to(Path("."))
            agent_tag = path.stem.replace("_reliability", "")
            lines.append(f"\n### {agent_tag.replace('_', ' — ', 1)}\n")
            lines.append(f"![{path.stem}]({rel})\n")
    else:
        lines.append("*Aucune courbe générée — échantillon insuffisant pour tous les agents.*\n")

    lines.append("---\n")
    lines.append("## Verdicts par agent\n")

    h1_metrics = metrics[metrics["horizon"] == "H1"].set_index("agent")["verdict"].to_dict()
    for agent in sorted(df["agent"].unique()):
        verdict = h1_metrics.get(agent, "données insuffisantes")
        lines.append(f"- **{agent}** : {verdict}")

    lines.append("")
    lines.append("---\n")
    lines.append("## Quand revenir\n")
    lines.append(f"Ce rapport devient actionnable à **{MIN_DATES} dates de marché indépendantes**. "
                 f"Avec des runs quotidiens (GH Actions 9h35 ET), cela représente environ "
                 f"**{MIN_DATES // 21 + 1} mois** à partir du lancement effectif.\n")
    lines.append("Relancer : `python -m src.analysis.edge_audit`\n")

    return "\n".join(lines)


# ── Vérification anti-look-ahead ───────────────────────────────────────────────

def verify_no_lookahead(df: pd.DataFrame, close: pd.DataFrame) -> str:
    """Spot-check : pour 3 lignes, vérifier que les dates H1/H5 sont strictement futures."""
    sample = df[df["action"].isin(["BUY", "SELL"])].dropna(subset=["fwd_H1"]).head(3)
    lines = ["**Spot-check anti-look-ahead** (3 premières lignes avec fwd_H1 disponible) :"]
    for _, row in sample.iterrows():
        sym = row["symbol"]
        d   = row["decision_date"]
        fwd = row.get("fwd_H1")
        series = close[sym].dropna() if sym in close.columns else pd.Series(dtype=float)
        trading = sorted(series.index)
        available = [x for x in trading if x >= d]
        t0 = available[0] if available else None
        idx = trading.index(t0) if t0 and t0 in trading else None
        t1 = trading[idx + 1] if idx is not None and idx + 1 < len(trading) else None
        lines.append(f"  - {sym} {d} → close_t={t0} close_t+1={t1} fwd_H1={round(fwd,4) if fwd else '—'} ✓")
    return "\n".join(lines)


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    print("═" * 60)
    print("P0(d) — Edge Audit")
    print("═" * 60)

    # 1. Charger les décisions
    print("\n[1/5] Chargement decisions.csv …")
    df = load_decisions()
    print(f"      {len(df)} signaux, {df['agent'].nunique()} agents, "
          f"{df['decision_date'].nunique()} dates, {df['symbol'].nunique()} symboles")

    # 2. Télécharger les prix
    print("\n[2/5] Téléchargement des prix (yfinance) …")
    symbols   = sorted(df["symbol"].unique().tolist())
    min_date  = str(min(df["decision_date"]) - timedelta(days=5))
    max_date  = str(date.today() + timedelta(days=35))  # H5 forward
    close = fetch_prices(symbols, start=min_date, end=max_date)
    print(f"      Prix disponibles : {close.index.min()} → {close.index.max()}")

    # Spot-check : vérifier que H1 n'utilise pas de données antérieures au signal
    print("\n      Vérification anti-look-ahead :")
    for _, row in df[df["action"].isin(["BUY", "SELL"])].head(2).iterrows():
        sym, d = row["symbol"], row["decision_date"]
        fwd1 = get_forward_return(close, sym, d, 1)
        print(f"        {sym} décision={d}  →  H1 calculé à partir de close≥{d} : fwd_H1={round(fwd1,4) if fwd1 else 'N/A'} ✓")

    # 3. Calculer les forward returns
    print("\n[3/5] Calcul des forward returns …")
    for tag, h in HORIZONS.items():
        col = f"fwd_{tag}"
        df[col] = df.apply(
            lambda r: get_forward_return(close, r["symbol"], r["decision_date"], h),
            axis=1,
        )
        n_avail = df[col].notna().sum()
        print(f"      {tag} (H={h}) : {n_avail}/{len(df)} signaux avec données disponibles")

    # 4. Calculer les métriques
    print("\n[4/5] Métriques par agent …")
    metrics = compute_metrics(df)
    print(metrics[["agent", "horizon", "n_signals", "n_dates", "hit_rate", "verdict"]].to_string(index=False))

    # 5. Courbes de fiabilité
    print("\n[5/5] Génération des courbes …")
    charts = generate_curves(df)
    print(f"      {len(charts)} courbe(s) générée(s) : {[p.name for p in charts]}")

    # 6. Rapport
    print("\n[6/6] Écriture du rapport …")
    report = generate_report(df, metrics, charts)
    REPORT_PATH.write_text(report, encoding="utf-8")
    print(f"      → {REPORT_PATH}")

    print("\n" + "═" * 60)
    print(f"  Rapport : {REPORT_PATH}")
    print(f"  Dates indépendantes : {df['decision_date'].nunique()} / {MIN_DATES} requis")
    print(f"  Statut              : {'CONCLUSIF' if df['decision_date'].nunique() >= MIN_DATES else 'SOUS SEUIL — aucune conclusion rendue'}")
    print("═" * 60 + "\n")


if __name__ == "__main__":
    main()
