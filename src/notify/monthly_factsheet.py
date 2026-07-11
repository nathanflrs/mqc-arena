"""
Monthly factsheet — generates a PDF with key performance metrics
and sends it to Telegram on the 1st of each month.

Triggered by: GitHub Actions cron '0 7 1 * *' (07:00 UTC = 08:00 CEST)
Manual run:   python -m src.notify.monthly_factsheet
"""
from __future__ import annotations

from datetime import date, timedelta
from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # headless — no display required

import matplotlib.gridspec as gridspec
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.backends.backend_pdf import PdfPages

# ── Brand colours (Structured Steel) ──────────────────────────────────────────
NAVY    = "#0f1c2e"
COBALT  = "#2a5bd7"
MC_LINE = "#4f86f7"
SPY_CLR = "#7c8aa0"
BG      = "#e8ebef"
WHITE   = "#ffffff"
TEXT    = "#26303d"
TEXT_S  = "#5c6675"
GREEN   = "#27ae60"
RED     = "#c0392b"


# ── Data helpers ───────────────────────────────────────────────────────────────

def _load_equity() -> pd.DataFrame:
    path = Path("logs/portfolio_equity.csv")
    if not path.exists():
        raise FileNotFoundError("logs/portfolio_equity.csv not found")
    df = pd.read_csv(path, parse_dates=["date"])
    df["date"] = pd.to_datetime(df["date"]).dt.tz_localize(None)
    return df.set_index("date").sort_index()


def _last_month_range() -> tuple[date, date]:
    today = date.today()
    last_day = date(today.year, today.month, 1) - timedelta(days=1)
    first_day = date(last_day.year, last_day.month, 1)
    return first_day, last_day


def _compute_metrics(rets: pd.Series) -> dict:
    """Annualised metrics from a daily return series."""
    n = len(rets)
    if n < 5:
        return {}

    rf_daily = 0.04 / 252
    mean_r   = float(rets.mean())
    std_r    = float(rets.std())
    ann      = np.sqrt(252)

    # Sharpe
    sharpe = (mean_r - rf_daily) / std_r * ann if std_r > 0 else 0.0

    # Sortino (downside deviation vs risk-free)
    excess = rets - rf_daily
    downside = excess[excess < 0]
    dd_std = float(np.sqrt((downside**2).mean())) * ann if len(downside) > 0 else 0.0
    sortino = (mean_r - rf_daily) * ann / dd_std if dd_std > 0 else 0.0

    # Max drawdown
    cum = (1 + rets).cumprod()
    peak = cum.expanding().max()
    max_dd = float(((cum - peak) / peak).min())

    # Annualised return & Calmar
    total_ret = float(cum.iloc[-1] - 1)
    n_years   = n / 252
    ann_ret   = (1 + total_ret) ** (1 / n_years) - 1 if n_years > 0 else 0.0
    calmar    = ann_ret / abs(max_dd) if max_dd != 0 else 0.0

    return {
        "sharpe":    sharpe,
        "sortino":   sortino,
        "calmar":    calmar,
        "max_dd":    max_dd,
        "ann_ret":   ann_ret,
        "total_ret": total_ret,
        "vol":       std_r * ann,
    }


# ── PDF generation ─────────────────────────────────────────────────────────────

def generate_factsheet_pdf(output_path: str | None = None) -> str:
    """Build the factsheet PDF and return the path written."""
    df = _load_equity()

    portfolio = df["portfolio_equity"]
    spy_eq    = df["spy_equity"]

    rets     = portfolio.pct_change().dropna()
    spy_rets = spy_eq.pct_change().dropna()

    metrics     = _compute_metrics(rets)
    spy_metrics = _compute_metrics(spy_rets)

    # SPY correlation
    common_spy = rets.index.intersection(spy_rets.index)
    corr_spy   = float(rets.loc[common_spy].corr(spy_rets.loc[common_spy]))

    # QQQ correlation (best-effort)
    corr_qqq: float | None = None
    try:
        import yfinance as yf
        raw = yf.download("QQQ", start=df.index[0].date(), end=df.index[-1].date(),
                          progress=False, auto_adjust=True)
        if not raw.empty:
            qqq_close = raw["Close"].squeeze()
            qqq_close.index = pd.to_datetime(qqq_close.index).tz_localize(None)
            qqq_rets  = qqq_close.pct_change().dropna()
            common_q  = rets.index.intersection(qqq_rets.index)
            if len(common_q) > 20:
                corr_qqq = float(rets.loc[common_q].corr(qqq_rets.loc[common_q]))
    except Exception:
        pass

    # Monthly returns
    monthly     = portfolio.resample("ME").last().pct_change().dropna()
    spy_monthly = spy_eq.resample("ME").last().pct_change().dropna()

    # Last-month stats
    first_day, last_day = _last_month_range()
    month_label = last_day.strftime("%B %Y")
    df_month = df.loc[str(first_day):str(last_day)]
    last_month_ret: float | None = None
    if len(df_month) >= 2:
        last_month_ret = float(
            df_month["portfolio_equity"].iloc[-1] / df_month["portfolio_equity"].iloc[0] - 1
        )

    if output_path is None:
        output_path = f"logs/factsheet_{last_day.strftime('%Y-%m')}.pdf"
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)

    # ── Figure layout ──────────────────────────────────────────────────────────
    with PdfPages(output_path) as pdf:
        fig = plt.figure(figsize=(13.0, 9.5), facecolor=BG)
        gs  = gridspec.GridSpec(
            3, 3, figure=fig,
            hspace=0.50, wspace=0.32,
            top=0.85, bottom=0.07, left=0.06, right=0.97,
        )

        # ── Header (navy band) ─────────────────────────────────────────────
        header = fig.add_axes([0.0, 0.88, 1.0, 0.12])
        header.set_facecolor(NAVY)
        header.axis("off")
        # Accent bar
        header.add_patch(mpatches.Rectangle(
            (0, 0), 0.006, 1,
            color=COBALT, transform=header.transAxes, clip_on=False,
        ))
        header.text(0.014, 0.64, "MILAN CAPITAL",
                    transform=header.transAxes, fontsize=17,
                    fontweight="bold", color=WHITE, va="center")
        header.text(0.014, 0.22, "MONTHLY FACTSHEET",
                    transform=header.transAxes, fontsize=9,
                    color="#9fb0c8", va="center")
        header.text(0.98, 0.5, month_label.upper(),
                    transform=header.transAxes, fontsize=13,
                    fontweight="bold", color=MC_LINE, ha="right", va="center")

        # ── Metrics strip ──────────────────────────────────────────────────
        def pct(v: float | None) -> str:
            return f"{v:+.1%}" if v is not None else "—"
        def num(v: float | None, d: int = 2) -> str:
            return f"{v:.{d}f}" if v is not None else "—"

        metric_cols = [
            ("Sharpe",       num(metrics.get("sharpe")),    False),
            ("Sortino",      num(metrics.get("sortino")),   False),
            ("Calmar",       num(metrics.get("calmar")),    False),
            ("Max Drawdown", pct(metrics.get("max_dd")),    True),
            ("Ann. Return",  pct(metrics.get("ann_ret")),   True),
            ("Volatility",   pct(metrics.get("vol")),       False),
            ("Total Return", pct(metrics.get("total_ret")), True),
        ]

        ax_m = fig.add_subplot(gs[0, :])
        ax_m.set_facecolor(WHITE)
        ax_m.axis("off")

        xs = np.linspace(0.04, 0.96, len(metric_cols))
        for i, (label, value, is_signed_pct) in enumerate(metric_cols):
            x = xs[i]
            # Pick colour for signed percentage values
            if is_signed_pct and value not in ("—",):
                v_color = GREEN if value.startswith("+") else RED
            elif i == 0:
                v_color = COBALT
            else:
                v_color = TEXT
            ax_m.text(x, 0.70, value, transform=ax_m.transAxes,
                      fontsize=13, fontweight="bold", color=v_color,
                      ha="center", va="center", family="monospace")
            ax_m.text(x, 0.18, label, transform=ax_m.transAxes,
                      fontsize=7.5, color=TEXT_S, ha="center", va="center",
                      fontweight="bold")
            # Vertical divider
            if i > 0:
                ax_m.axvline((xs[i] + xs[i - 1]) / 2, color=BG,
                             linewidth=1.5, ymin=0.05, ymax=0.95)

        if last_month_ret is not None:
            sign_c = GREEN if last_month_ret >= 0 else RED
            ax_m.text(0.5, -0.08,
                      f"Last month ({month_label}):  {last_month_ret:+.2%}",
                      transform=ax_m.transAxes,
                      fontsize=8.5, color=sign_c, ha="center",
                      fontweight="bold", family="monospace")

        # ── Equity curve ───────────────────────────────────────────────────
        ax_eq = fig.add_subplot(gs[1, :])
        ax_eq.set_facecolor(WHITE)

        port_norm = portfolio / portfolio.iloc[0] * 100
        spy_norm  = spy_eq   / spy_eq.iloc[0]   * 100

        ax_eq.plot(port_norm.index, port_norm.values,
                   color=MC_LINE, linewidth=2.0, label="Milan Capital", zorder=3)
        ax_eq.fill_between(port_norm.index, 100, port_norm.values,
                           where=(port_norm.values >= 100),
                           alpha=0.13, color=MC_LINE, zorder=2)
        ax_eq.plot(spy_norm.index, spy_norm.values,
                   color=SPY_CLR, linewidth=1.5, linestyle="--",
                   label="SPY", zorder=2)
        ax_eq.axhline(100, color=BG, linewidth=1, zorder=1)

        ax_eq.set_ylabel("Indexed to 100", fontsize=8, color=TEXT_S)
        ax_eq.tick_params(labelsize=8, colors=TEXT_S)
        for spine in ("top", "right"):
            ax_eq.spines[spine].set_visible(False)
        for spine in ("left", "bottom"):
            ax_eq.spines[spine].set_color(BG)
        ax_eq.legend(fontsize=8, framealpha=0, labelcolor=TEXT_S)
        ax_eq.set_title("Equity Curve", fontsize=9, color=TEXT,
                        fontweight="bold", loc="left", pad=5)

        # ── Monthly returns bar ────────────────────────────────────────────
        ax_bar = fig.add_subplot(gs[2, :2])
        ax_bar.set_facecolor(WHITE)

        bar_colors = [GREEN if r >= 0 else RED for r in monthly.values]
        ax_bar.bar(monthly.index, monthly.values * 100,
                   color=bar_colors, width=20, alpha=0.85, zorder=2)
        ax_bar.axhline(0, color=TEXT, linewidth=0.7, zorder=3)
        ax_bar.set_ylabel("Return (%)", fontsize=8, color=TEXT_S)
        ax_bar.tick_params(labelsize=7, colors=TEXT_S)
        ax_bar.xaxis.set_tick_params(rotation=45)
        for spine in ("top", "right"):
            ax_bar.spines[spine].set_visible(False)
        for spine in ("left", "bottom"):
            ax_bar.spines[spine].set_color(BG)
        ax_bar.set_title("Monthly Returns", fontsize=9, color=TEXT,
                         fontweight="bold", loc="left", pad=5)

        # ── Correlations & benchmark ───────────────────────────────────────
        ax_c = fig.add_subplot(gs[2, 2])
        ax_c.set_facecolor(WHITE)
        ax_c.axis("off")
        ax_c.set_title("Benchmark & Correlation", fontsize=9, color=TEXT,
                       fontweight="bold", loc="left", pad=5)

        rows_c = [
            ("Corr vs SPY",    f"{corr_spy:.3f}"),
            ("Corr vs QQQ",    f"{corr_qqq:.3f}" if corr_qqq is not None else "n/a"),
            ("", ""),
            ("SPY Ann. Ret.",  pct(spy_metrics.get("ann_ret"))),
            ("SPY Sharpe",     num(spy_metrics.get("sharpe"))),
            ("SPY Max DD",     pct(spy_metrics.get("max_dd"))),
            ("SPY Volatility", pct(spy_metrics.get("vol"))),
        ]
        for i, (label, value) in enumerate(rows_c):
            y = 0.90 - i * 0.13
            ax_c.text(0.04, y, label, transform=ax_c.transAxes,
                      fontsize=8, color=TEXT_S, va="center")
            ax_c.text(0.96, y, value, transform=ax_c.transAxes,
                      fontsize=8, color=TEXT, va="center", ha="right",
                      fontweight="bold", family="monospace")

        # ── Disclaimer ─────────────────────────────────────────────────────
        fig.text(
            0.5, 0.012,
            "Past performance does not guarantee future results. "
            "Metrics computed on walk-forward validated out-of-sample data (logs/portfolio_equity.csv).",
            ha="center", fontsize=6.5, color=TEXT_S, style="italic",
        )

        pdf.savefig(fig, bbox_inches="tight")
        plt.close(fig)

    return output_path


# ── Entry point ────────────────────────────────────────────────────────────────

def run() -> None:
    from dotenv import load_dotenv
    load_dotenv()

    try:
        path = generate_factsheet_pdf()
        _, last_day = _last_month_range()
        month_label = last_day.strftime("%B %Y")
        print(f"✅ Factsheet generated: {path}")
        try:
            from src.events.bus import get_bus, Event
            get_bus().emit(Event(
                type="factsheet",
                severity="info",
                title=f"Monthly Factsheet — {month_label}",
                body=f"📊 Milan Capital — Monthly Factsheet {month_label}\n📁 {path}",
                meta={"month": month_label, "path": str(path)},
            ))
        except Exception:
            pass
    except Exception as e:
        print(f"❌ Monthly factsheet failed: {e}")
        raise


if __name__ == "__main__":
    run()
