# src/dashboard/app.py
from __future__ import annotations

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

import pandas as pd
import streamlit as st
import plotly.graph_objects as go
from datetime import datetime
from dotenv import load_dotenv

load_dotenv()

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))

from src.agents.buffett import BuffettAgent
from src.agents.citadel import CitadelAgent
from src.agents.mean_reversion import MeanReversionAgent
from src.agents.macro import MacroAgent
from src.agents.trend_following import TrendFollowingAgent
from src.agents.dividend_arbitrage import DividendArbitrageAgent
from src.agents.pairs_trading import PairsTradingAgent
from src.agents.volatility import VolatilityAgent
from src.agents.dummy import DummyHoldAgent
from src.arena.arena import Arena
from src.arena.selector import select_best
from src.data.market_data import download_ohlcv
from src.data.regime import detect_regime
from src.config import WATCHLIST
from src.risk.correlation import CorrelationGuard
from src.risk.live_scorer import LiveScorer

st.set_page_config(
    page_title="Milan Capital",
    page_icon="💼",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ====== GLOBAL CSS ======
st.markdown("""
<style>
[data-testid="stAppViewContainer"] { background-color: #0e1117; }
[data-testid="stSidebar"] { background-color: #12151f; border-right: 1px solid #2a2d3a; }
[data-testid="stHeader"] { background-color: #0e1117; border-bottom: 1px solid #1a1d27; }
[data-testid="metric-container"] {
    background: #1a1d27;
    border: 1px solid #2a2d3a;
    border-radius: 10px;
    padding: 12px 16px;
}
#MainMenu { visibility: hidden; }
footer { visibility: hidden; }

/* ── Pipeline native alerts ── */
div[data-testid="stAlert"] {
    border-radius: 10px !important;
    padding: 10px 12px !important;
}
/* pipeline ticker label */
.pl-ticker {
    color: #00ff88;
    font-family: monospace;
    font-weight: 700;
    font-size: 13px;
    letter-spacing: 1.4px;
    margin-bottom: 10px;
}

/* ── Order cards ── */
.o-card {
    border-radius: 10px; padding: 12px 16px; margin-bottom: 8px;
    display: flex; align-items: center; gap: 14px;
}
.o-icon  { font-size: 20px; font-weight: 900; min-width: 22px; text-align: center; }
.o-main  { flex: 1; }
.o-side  { font-weight: 700; font-size: 13px; }
.o-rsn   { font-size: 11px; color: #999; margin-top: 2px; }
.o-meta  { text-align: right; }
.o-qty   { font-size: 12px; font-weight: 600; color: #e0e0e0; }
.o-price { font-size: 10px; color: #777; margin-top: 2px; }

/* ── Sidebar ── */
.sb-card {
    background: #1a1d27; border: 1px solid #2a2d3a; border-radius: 10px;
    padding: 11px 13px; margin-bottom: 8px;
}
.sb-label { color: #555; font-size: 10px; letter-spacing: 1px; text-transform: uppercase; margin-bottom: 3px; }
.sb-val   { color: #fff; font-size: 18px; font-weight: 700; }
.sb-delta { font-size: 11px; margin-top: 2px; }
.alloc-bg   { background: #252936; border-radius: 4px; height: 5px; margin-top: 5px; }
.alloc-fill { border-radius: 4px; height: 5px; }

/* ── Section headers ── */
.sec-header {
    display: flex; align-items: center; gap: 10px; margin: 28px 0 14px 0;
}
.sec-dot   { font-size: 17px; }
.sec-title { color: #fff; font-size: 17px; font-weight: 700; }
.sec-sub   { color: #555; font-size: 12px; margin-left: 4px; }
</style>
""", unsafe_allow_html=True)


# ====== DATA LOADING ======
@st.cache_resource(ttl=300)
def load_data():
    regime_data = detect_regime("SPY")
    arena = Arena([
        DummyHoldAgent(),
        BuffettAgent(),
        CitadelAgent(),
        MeanReversionAgent(),
        MacroAgent(),
        TrendFollowingAgent(),
        DividendArbitrageAgent(),
        PairsTradingAgent(),
        VolatilityAgent(),
    ])
    results = {}
    for sym in WATCHLIST:
        df = download_ohlcv(sym)
        signals = arena.run(sym, df, regime=regime_data["regime"])
        winner = select_best(signals)
        results[sym] = {"signals": signals, "winner": winner, "df": df}
    return regime_data, results


# ====== SIDEBAR ======
with st.sidebar:
    st.markdown("""
    <div style="padding:4px 0 16px; border-bottom:1px solid #2a2d3a; margin-bottom:16px;">
        <div style="color:#00ff88; font-size:18px; font-weight:800; letter-spacing:1px;">💼 MILAN CAPITAL</div>
        <div style="color:#444; font-size:11px; margin-top:2px;">AI Multi-Agent Hedge Fund</div>
    </div>
    """, unsafe_allow_html=True)

    if st.button("🔄  Refresh", use_container_width=True, type="primary"):
        st.cache_resource.clear()
        st.rerun()

    st.markdown(
        f"<div style='color:#444; font-size:10px; text-align:center; margin:6px 0 18px;'>"
        f"Updated {datetime.now().strftime('%H:%M:%S')}</div>",
        unsafe_allow_html=True,
    )

    # Portfolio metrics from order_plan
    try:
        _df_plan = pd.read_csv(os.path.join(ROOT, "logs/order_plan.csv"))
        _n_pos  = _df_plan["symbol"].nunique() if not _df_plan.empty else 0
        _n_buy  = len(_df_plan[_df_plan["side"] == "BUY"])  if not _df_plan.empty else 0
        _n_sell = len(_df_plan[_df_plan["side"] == "SELL"]) if not _df_plan.empty else 0
        _notional = _df_plan["est_notional"].abs().sum() if not _df_plan.empty else 0.0
    except (FileNotFoundError, KeyError):
        _n_pos, _n_buy, _n_sell, _notional = 0, 0, 0, 0.0

    st.markdown(
        f"""<div style="color:#888; font-size:11px; letter-spacing:1px; text-transform:uppercase; margin-bottom:10px;">Portfolio</div>
<div class="sb-card">
    <div class="sb-label">Net Liquidation</div>
    <div class="sb-val">${_notional:,.0f}</div>
    <div class="sb-delta" style="color:#555;">paper trading</div>
</div>
<div class="sb-card">
    <div class="sb-label">Open Positions</div>
    <div class="sb-val">{_n_pos}</div>
    <div class="sb-delta">
        <span style="color:#00ff88;">▲ {_n_buy} long</span>
        &nbsp;
        <span style="color:#ff4444;">▼ {_n_sell} short</span>
    </div>
</div>""",
        unsafe_allow_html=True,
    )

    st.markdown(
        "<div style='color:#888; font-size:11px; letter-spacing:1px; text-transform:uppercase; margin:18px 0 10px;'>Allocation</div>",
        unsafe_allow_html=True,
    )
    try:
        if not _df_plan.empty and _notional > 0:
            _alloc = (
                _df_plan.groupby("symbol")["est_notional"]
                .sum().abs()
                .sort_values(ascending=False)
                .head(8)
            )
            for _sym, _val in _alloc.items():
                _pct = _val / _notional
                _side = _df_plan[_df_plan["symbol"] == _sym]["side"].iloc[0]
                _bar_color = "#00ff88" if _side == "BUY" else "#ff4444"
                st.markdown(f"""
<div style="margin-bottom:9px;">
    <div style="display:flex; justify-content:space-between; align-items:center;">
        <span style="color:#ccc; font-size:12px; font-weight:600;">{_sym}</span>
        <span style="color:#666; font-size:11px;">{_pct:.0%}</span>
    </div>
    <div class="alloc-bg">
        <div class="alloc-fill" style="background:{_bar_color}; width:{_pct*100:.1f}%;"></div>
    </div>
</div>""", unsafe_allow_html=True)
        else:
            st.markdown("<div style='color:#444; font-size:12px; padding:4px 0;'>Aucune position</div>", unsafe_allow_html=True)
    except Exception:
        st.markdown("<div style='color:#444; font-size:12px;'>—</div>", unsafe_allow_html=True)

    # ── Circuit Breaker / Drawdown ──
    st.markdown(
        "<div style='border-top:1px solid #2a2d3a; margin:18px 0 12px;'></div>"
        "<div style='color:#888; font-size:11px; letter-spacing:1px; text-transform:uppercase; margin-bottom:10px;'>Circuit Breaker</div>",
        unsafe_allow_html=True,
    )
    try:
        import json as _json
        _cb = _json.loads(open(os.path.join(ROOT, "logs/circuit_breaker.json")).read())
        _dd        = float(_cb.get("drawdown") or 0.0)
        _peak      = float(_cb.get("peak_netliq") or 0.0)
        _current   = float(_cb.get("current_netliq") or 0.0)
        _triggered = bool(_cb.get("triggered"))
        _dd_color  = "#ff4444" if _triggered else ("#ffaa00" if _dd > 0.05 else "#00ff88")
        _status_label  = "⛔ SELL-ONLY" if _triggered else "✅ OK"
        _status_color  = "#ff4444" if _triggered else "#00ff88"
        st.markdown(f"""
<div class="sb-card" style="border-color:{_dd_color}55;">
    <div class="sb-label">Drawdown depuis pic</div>
    <div class="sb-val" style="color:{_dd_color};">{_dd:.1%}</div>
    <div class="sb-delta" style="color:{_status_color}; font-weight:700;">{_status_label}</div>
</div>
<div style="display:flex; justify-content:space-between; padding:2px 2px 8px;">
    <span style="color:#555; font-size:10px;">Peak&nbsp;${_peak:,.0f}</span>
    <span style="color:#555; font-size:10px;">Current&nbsp;${_current:,.0f}</span>
</div>""", unsafe_allow_html=True)
        if _triggered:
            _at = _cb.get("triggered_at", "")[:16].replace("T", " ")
            st.markdown(
                f"<div style='background:#1a0000; border:1px solid #ff444455; border-radius:8px; "
                f"padding:8px 10px; font-size:10px; color:#ff6666;'>"
                f"Déclenché le {_at} UTC<br>Reset manuel requis.</div>",
                unsafe_allow_html=True,
            )
    except FileNotFoundError:
        st.markdown(
            "<div style='color:#444; font-size:11px; padding:4px 0;'>Aucune donnée (runner non exécuté)</div>",
            unsafe_allow_html=True,
        )
    except Exception:
        pass

    st.markdown(
        "<div style='border-top:1px solid #2a2d3a; margin:20px 0 12px;'></div>"
        "<div style='color:#444; font-size:10px; text-align:center;'>Paper Trading Only</div>",
        unsafe_allow_html=True,
    )


# ====== MAIN — LOAD DATA ======
with st.spinner("Chargement des données de marché..."):
    regime_data, results = load_data()

# ── Sidebar: Corrélation portfolio (nécessite results) ────────────────────────
with st.sidebar:
    st.markdown(
        "<div style='border-top:1px solid #2a2d3a; margin:18px 0 12px;'></div>"
        "<div style='color:#888; font-size:11px; letter-spacing:1px;"
        "text-transform:uppercase; margin-bottom:10px;'>Corrélation Portfolio</div>",
        unsafe_allow_html=True,
    )
    try:
        _buy_syms = (
            list(_df_plan[_df_plan["side"] == "BUY"]["symbol"].unique())
            if not _df_plan.empty else []
        )
        if len(_buy_syms) >= 2:
            _price_data = {s: results[s]["df"] for s in _buy_syms if s in results}
            _corr = CorrelationGuard().correlation_matrix(_buy_syms, _price_data)
            if not _corr.empty:
                _corr_masked = _corr.copy()
                for _c in _corr_masked.columns:
                    _corr_masked.loc[_c, _c] = float("nan")
                _max_c = float(_corr_masked.abs().max().max())
                _cc = "#ff4444" if _max_c >= 0.7 else ("#ffaa00" if _max_c >= 0.5 else "#00ff88")
                st.markdown(f"""
<div class="sb-card" style="border-color:{_cc}55;">
    <div class="sb-label">Max corrélation pairwise</div>
    <div class="sb-val" style="color:{_cc};">{_max_c:.2f}</div>
    <div class="sb-delta" style="color:#555;">{len(_buy_syms)} positions BUY
        {'&nbsp;⚠️ sur-concentration' if _max_c >= 0.7 else ''}</div>
</div>""", unsafe_allow_html=True)
        else:
            st.markdown(
                "<div style='color:#444; font-size:12px; padding:4px 0;'>"
                "≥ 2 positions BUY requises</div>",
                unsafe_allow_html=True,
            )
    except Exception:
        pass

regime = regime_data["regime"]


# ====== REGIME BANNER ======
_R_COLORS = {
    "bull":   ("#00ff88", "#091e13", "#00ff8833"),
    "bear":   ("#ff4444", "#1e0909", "#ff444433"),
    "choppy": ("#ffaa00", "#1e1600", "#ffaa0033"),
}
r_fg, r_bg, r_border = _R_COLORS.get(regime, ("#888", "#1a1d27", "#88888833"))
r_emoji = {"bull": "🟢", "bear": "🔴", "choppy": "🟡"}.get(regime, "⚪")

st.markdown(f"""
<div style="background:{r_bg}; border:1px solid {r_border}; border-radius:12px; padding:16px 24px; margin-bottom:22px; display:flex; align-items:center; justify-content:space-between; flex-wrap:wrap; gap:14px;">
    <div>
        <div style="color:{r_fg}; font-size:20px; font-weight:800; letter-spacing:1px;">{r_emoji}&nbsp; {regime.upper()} MARKET</div>
        <div style="color:#555; font-size:11px; margin-top:3px;">Regime détecté • SPY SMA crossover + vol filter</div>
    </div>
    <div style="display:flex; gap:28px;">
        <div>
            <div style="color:#555; font-size:10px; text-transform:uppercase; letter-spacing:.8px;">SPY</div>
            <div style="color:#fff; font-weight:700; font-size:15px;">${regime_data['price']}</div>
        </div>
        <div>
            <div style="color:#555; font-size:10px; text-transform:uppercase; letter-spacing:.8px;">SMA 50</div>
            <div style="color:#fff; font-weight:700; font-size:15px;">${regime_data['sma50']}</div>
        </div>
        <div>
            <div style="color:#555; font-size:10px; text-transform:uppercase; letter-spacing:.8px;">SMA 200</div>
            <div style="color:#fff; font-weight:700; font-size:15px;">${regime_data['sma200']}</div>
        </div>
        <div>
            <div style="color:#555; font-size:10px; text-transform:uppercase; letter-spacing:.8px;">Volatilité</div>
            <div style="color:{r_fg}; font-weight:700; font-size:15px;">{regime_data['vol_regime'].upper()}</div>
        </div>
    </div>
</div>
""", unsafe_allow_html=True)


# ====== PIPELINE HELPERS (Plotly JARVIS) ======
_SHORT_NAMES = {
    "BuffettAgent":           "Buffett",
    "CitadelAgent":           "Citadel",
    "MeanReversionAgent":     "MeanRev",
    "MacroAgent":             "Macro",
    "TrendFollowingAgent":    "Trend",
    "DividendArbitrageAgent": "DivArb",
    "PairsTradingAgent":      "Pairs",
    "VolatilityAgent":        "Vol",
    "DummyHoldAgent":         "Dummy",
}

_A_BORDER = {"BUY": "#00ff88", "SELL": "#ff4444", "HOLD": "#3a3d4a"}
_A_FILL   = {"BUY": "#001a0d", "SELL": "#1a0000", "HOLD": "#1a1d27"}
_A_TEXT   = {"BUY": "#00ff88", "SELL": "#ff4444", "HOLD": "#888888"}


def _build_pipeline_figure(sym: str, signals: list, winner) -> go.Figure:
    nodes = [{"top": "MARKET", "mid": sym, "bot": "LIVE", "action": "DATA", "win": False, "kind": "src"}]
    for sig in signals:
        is_win = winner is not None and sig.agent_name == winner.agent_name
        nodes.append({
            "top":    _SHORT_NAMES.get(sig.agent_name, sig.agent_name.replace("Agent", "")),
            "mid":    sig.action,
            "bot":    f"{sig.confidence:.0%}",
            "action": sig.action,
            "win":    is_win,
            "kind":   "agent",
        })
    cio_action = winner.action if (winner and winner.agent_name != "DummyHoldAgent") else "HOLD"
    nodes.append({"top": "CIO", "mid": cio_action, "bot": "FINAL", "action": cio_action, "win": False, "kind": "cio"})

    N = len(nodes)
    nw, nh, y_c = 0.38, 0.27, 0.5

    # layer='below' is critical — without it shapes are drawn ABOVE scatter traces
    shapes, annotations = [], []
    for i, nd in enumerate(nodes):
        if nd["kind"] == "src":
            border, fill, lw = "#2a6aaa", "#0d1a2e", 1.5
        elif nd["win"]:
            border, fill, lw = "#00ff88", "#003322", 3.0
        else:
            border = _A_BORDER.get(nd["action"], "#3a3d4a")
            fill   = _A_FILL.get(nd["action"], "#1a1d27")
            lw     = 1.5

        shapes.append(dict(
            type="rect", layer="below",
            x0=i - nw, y0=y_c - nh, x1=i + nw, y1=y_c + nh,
            line=dict(color=border, width=lw),
            fillcolor=fill,
        ))
        if nd["win"]:
            annotations.append(dict(
                x=i, y=y_c + nh + 0.10,
                text="★ WINNER",
                showarrow=False,
                font=dict(color="#00ff88", size=8, family="monospace"),
                xanchor="center",
            ))

    fig = go.Figure()

    # Connector lines (drawn before text so text stays on top)
    lx, ly = [], []
    for i in range(N - 1):
        lx += [i + nw + 0.01, i + 1 - nw - 0.01, None]
        ly += [y_c, y_c, None]
    fig.add_trace(go.Scatter(x=lx, y=ly, mode="lines",
        line=dict(color="rgba(0,255,136,0.6)", width=1.5),
        showlegend=False, hoverinfo="skip"))

    fig.add_trace(go.Scatter(
        x=[i + 1 - nw for i in range(N - 1)],
        y=[y_c] * (N - 1),
        mode="markers",
        marker=dict(symbol="triangle-right", size=6, color="rgba(0,255,136,0.6)"),
        showlegend=False, hoverinfo="skip"))

    # One text trace per node — single HTML string, centered in the rectangle
    for i, nd in enumerate(nodes):
        mid_col = (
            "#2a9aff" if nd["kind"] == "src"
            else "#00ff88" if nd["win"]
            else _A_TEXT.get(nd["action"], "#cccccc")
        )
        label = (
            f"<span style='color:#777777;font-size:9px'>{nd['top']}</span><br>"
            f"<b><span style='color:{mid_col};font-size:12px'>{nd['mid']}</span></b><br>"
            f"<span style='color:#555555;font-size:8px'>{nd['bot']}</span>"
        )
        fig.add_trace(go.Scatter(
            x=[i], y=[y_c],
            mode="text",
            text=[label],
            textposition="middle center",
            showlegend=False,
            hoverinfo="skip",
        ))

    fig.update_layout(
        shapes=shapes, annotations=annotations,
        paper_bgcolor="#0e1117", plot_bgcolor="#0e1117",
        margin=dict(l=8, r=8, t=22, b=8),
        height=180,
        xaxis=dict(range=[-0.6, N - 0.4], showgrid=False, zeroline=False,
                   showticklabels=False, visible=False),
        yaxis=dict(range=[0.05, 1.0], showgrid=False, zeroline=False,
                   showticklabels=False, visible=False),
        showlegend=False, hovermode=False,
    )
    return fig


# ====== PIPELINE SECTION ======
st.markdown("""
<div class="sec-header">
    <div class="sec-dot" style="color:#00ff88;">⬡</div>
    <div class="sec-title">Agent Pipeline</div>
    <div class="sec-sub">• live signals</div>
</div>
""", unsafe_allow_html=True)

for sym in WATCHLIST:
    data    = results[sym]
    signals = [s for s in data["signals"] if s.agent_name != "DummyHoldAgent"]
    winner  = data["winner"]
    st.markdown(f"<p class='pl-ticker'>◈ &nbsp;{sym}</p>", unsafe_allow_html=True)
    st.plotly_chart(_build_pipeline_figure(sym, signals, winner), use_container_width=True)
    st.markdown("<div style='height:4px;'></div>", unsafe_allow_html=True)


# ====== ORDER PLAN ======
st.markdown("""
<div class="sec-header">
    <div class="sec-dot" style="color:#ffaa00;">◎</div>
    <div class="sec-title">Order Plan</div>
</div>
""", unsafe_allow_html=True)

try:
    df_plan = pd.read_csv(os.path.join(ROOT, "logs/order_plan.csv"))
    if not df_plan.empty:
        for _, row in df_plan.tail(10).iterrows():
            side     = row["side"]
            symbol   = row["symbol"]
            delta    = row["delta_qty"]
            price    = row["last_price"]
            notional = row["est_notional"]
            reason   = str(row["reason"])[:65]
            if side == "BUY":
                icon, fg, bg = "▲", "#00ff88", "#091e13"
            elif side == "SELL":
                icon, fg, bg = "▼", "#ff4444", "#1e0909"
            else:
                icon, fg, bg = "●", "#888",    "#1a1d27"
            st.markdown(f"""
<div class="o-card" style="background:{bg}; border:1px solid {fg}33;">
    <div class="o-icon" style="color:{fg};">{icon}</div>
    <div class="o-main">
        <div class="o-side" style="color:{fg};">{side}&nbsp;
            <span style="color:#e0e0e0;">{symbol}</span>
        </div>
        <div class="o-rsn">{reason}</div>
    </div>
    <div class="o-meta">
        <div class="o-qty">{delta:+.0f} shares</div>
        <div class="o-price">@ ${price:.2f}&nbsp;•&nbsp;est. ${notional:.0f}</div>
    </div>
</div>""", unsafe_allow_html=True)
    else:
        st.markdown("<div style='color:#555; padding:20px; text-align:center; background:#1a1d27; border-radius:10px;'>Aucun order plan disponible.</div>", unsafe_allow_html=True)
except FileNotFoundError:
    st.markdown("<div style='color:#555; padding:20px; text-align:center; background:#1a1d27; border-radius:10px;'>Lancez d'abord le runner pour générer un order plan.</div>", unsafe_allow_html=True)


# ====== EQUITY CURVE ======
st.markdown("""
<div class="sec-header">
    <div class="sec-dot" style="color:#00ff88;">↗</div>
    <div class="sec-title">Equity Curve</div>
    <div class="sec-sub">• paper trading</div>
</div>
""", unsafe_allow_html=True)

try:
    df_exec = pd.read_csv(os.path.join(ROOT, "logs/executions.csv"))
    if not df_exec.empty and "timestamp" in df_exec.columns:
        df_exec["timestamp"] = pd.to_datetime(df_exec["timestamp"])
        df_exec = df_exec.sort_values("timestamp")
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=df_exec["timestamp"],
            y=df_exec["est_notional"].cumsum(),
            mode="lines",
            line=dict(color="#00ff88", width=2.5),
            fill="tozeroy",
            fillcolor="rgba(0,255,136,0.06)",
            name="Notional cumulé",
        ))
        fig.update_layout(
            paper_bgcolor="#0e1117",
            plot_bgcolor="#1a1d27",
            font=dict(color="#aaa"),
            xaxis=dict(gridcolor="#2a2d3a", linecolor="#2a2d3a", zerolinecolor="#2a2d3a"),
            yaxis=dict(gridcolor="#2a2d3a", linecolor="#2a2d3a", zerolinecolor="#2a2d3a", tickprefix="$"),
            margin=dict(l=0, r=0, t=10, b=0),
            height=280,
            showlegend=False,
        )
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.markdown("<div style='color:#555; padding:24px; text-align:center; background:#1a1d27; border-radius:10px;'>Approuvez un order plan pour voir l'equity curve.</div>", unsafe_allow_html=True)
except FileNotFoundError:
    st.markdown("<div style='color:#555; padding:24px; text-align:center; background:#1a1d27; border-radius:10px;'>Pas encore d'exécutions.</div>", unsafe_allow_html=True)


# ====== BACKTEST RESULTS ======
st.markdown("""
<div class="sec-header">
    <div class="sec-dot" style="color:#aa88ff;">⬟</div>
    <div class="sec-title">Backtest Results</div>
    <div class="sec-sub">• 3 ans</div>
</div>
""", unsafe_allow_html=True)

try:
    df_bt = pd.read_csv(os.path.join(ROOT, "logs/backtest_results.csv"))
    if not df_bt.empty:
        best = df_bt.sort_values("sharpe_ratio", ascending=False).iloc[0]
        b1, b2, b3, b4 = st.columns(4)
        with b1:
            st.metric("🏆 Best Agent",   best["agent"].replace("Agent", ""))
        with b2:
            st.metric("📈 Best Return",  f"{best['total_return']:+.1%}")
        with b3:
            st.metric("⚡ Best Sharpe",  f"{best['sharpe_ratio']:.2f}")
        with b4:
            st.metric("📉 Max Drawdown", f"{best['max_drawdown']:.1%}")

        st.markdown("<br>", unsafe_allow_html=True)

        df_display = df_bt.copy().sort_values("sharpe_ratio", ascending=False)
        df_display["total_return"]      = df_display["total_return"].map("{:+.1%}".format)
        df_display["annualized_return"] = df_display["annualized_return"].map("{:+.1%}".format)
        df_display["sharpe_ratio"]      = df_display["sharpe_ratio"].map("{:.2f}".format)
        df_display["max_drawdown"]      = df_display["max_drawdown"].map("{:.1%}".format)
        df_display["win_rate"]          = df_display["win_rate"].map("{:.0%}".format)
        df_display.columns = ["Agent", "Symbole", "Return Total", "Return Ann.", "Sharpe", "Max DD", "Win Rate", "Trades"]
        st.dataframe(df_display, use_container_width=True, hide_index=True)

        st.markdown("<br>", unsafe_allow_html=True)

        fig2 = go.Figure()
        for sym_bt in df_bt["symbol"].unique():
            sub = df_bt[df_bt["symbol"] == sym_bt]
            fig2.add_trace(go.Bar(
                name=sym_bt,
                x=sub["agent"].str.replace("Agent", "", regex=False),
                y=sub["sharpe_ratio"],
            ))
        fig2.update_layout(
            barmode="group",
            paper_bgcolor="#0e1117",
            plot_bgcolor="#1a1d27",
            font=dict(color="#aaa"),
            xaxis=dict(gridcolor="#2a2d3a", linecolor="#2a2d3a"),
            yaxis=dict(gridcolor="#2a2d3a", linecolor="#2a2d3a", title="Sharpe Ratio"),
            legend=dict(bgcolor="#1a1d27", bordercolor="#2a2d3a"),
            margin=dict(l=0, r=0, t=40, b=0),
            height=360,
            title=dict(text="Sharpe Ratio par Agent & Symbole", font=dict(color="#e0e0e0", size=14)),
        )
        fig2.add_hline(
            y=0, line_dash="dash", line_color="#ff4444", opacity=0.5,
            annotation_text="Seuil 0", annotation_font_color="#ff4444",
        )
        st.plotly_chart(fig2, use_container_width=True)

    else:
        st.markdown("<div style='color:#555; padding:24px; text-align:center; background:#1a1d27; border-radius:10px;'>Lancez d'abord le backtest.</div>", unsafe_allow_html=True)
except FileNotFoundError:
    st.markdown("<div style='color:#555; padding:24px; text-align:center; background:#1a1d27; border-radius:10px;'>Lancez d'abord : python -m src.backtest.run_backtest</div>", unsafe_allow_html=True)


# ====== CORRELATION HEATMAP ======
st.markdown("""
<div class="sec-header">
    <div class="sec-dot" style="color:#ffaa00;">⬡</div>
    <div class="sec-title">Corrélation Portfolio</div>
    <div class="sec-sub">• 60 jours glissants | seuil BUY bloqué ≥ 0.70</div>
</div>
""", unsafe_allow_html=True)

try:
    _all_syms = [s for s in WATCHLIST if s in results]
    _price_data_all = {s: results[s]["df"] for s in _all_syms}
    _corr_full = CorrelationGuard(lookback_days=60).correlation_matrix(_all_syms, _price_data_all)
    if not _corr_full.empty:
        _fig_corr = go.Figure(go.Heatmap(
            z=_corr_full.values,
            x=list(_corr_full.columns),
            y=list(_corr_full.index),
            colorscale=[
                [0.0, "#1a0000"], [0.35, "#0e1117"],
                [0.5, "#1a1d27"],
                [0.65, "#001a0d"], [1.0, "#00ff88"],
            ],
            zmin=-1, zmax=1,
            text=_corr_full.round(2).values,
            texttemplate="%{text}",
            textfont=dict(size=9, color="#ccc"),
            hovertemplate="%{y} / %{x}: %{z:.2f}<extra></extra>",
        ))
        _fig_corr.update_layout(
            paper_bgcolor="#0e1117", plot_bgcolor="#1a1d27",
            font=dict(color="#aaa", size=10),
            margin=dict(l=0, r=0, t=10, b=0),
            height=400,
        )
        st.plotly_chart(_fig_corr, use_container_width=True)
        st.caption("Corrélation des rendements journaliers sur 60 jours. "
                   "Rouge = corrélation négative, Vert = positive.")
    else:
        st.markdown("<div style='color:#555; padding:20px; text-align:center; background:#1a1d27; "
                    "border-radius:10px;'>Données insuffisantes.</div>", unsafe_allow_html=True)
except Exception as _e:
    st.markdown(f"<div style='color:#555; padding:20px;'>Erreur heatmap : {_e}</div>",
                unsafe_allow_html=True)


# ====== TEARSHEET ======
st.markdown("""
<div class="sec-header">
    <div class="sec-dot" style="color:#aa88ff;">📊</div>
    <div class="sec-title">Tearsheet PnL Attribution</div>
    <div class="sec-sub">• par agent • live round-trips</div>
</div>
""", unsafe_allow_html=True)

try:
    import glob as _glob
    _sheets = sorted(_glob.glob(os.path.join(ROOT, "logs/tearsheet_*.csv")))
    if _sheets:
        _df_tear = pd.read_csv(_sheets[-1])
        _week_label = os.path.basename(_sheets[-1]).replace("tearsheet_", "").replace(".csv", "")
        st.caption(f"Semaine {_week_label} — {len(_sheets)} tearsheet(s) disponibles")
        c1, c2, c3, c4 = st.columns(4)
        _best_row = _df_tear.sort_values("sharpe", ascending=False).iloc[0]
        with c1:
            st.metric("🤖 Meilleur agent", _best_row["agent"].replace("Agent", ""))
        with c2:
            st.metric("⚡ Sharpe", f"{_best_row['sharpe']:.2f}")
        with c3:
            st.metric("✅ Win rate", f"{_best_row['win_rate']:.0%}")
        with c4:
            st.metric("📉 Max DD", f"{_best_row['max_drawdown']:.1%}")
        st.markdown("<br>", unsafe_allow_html=True)
        _df_tear_disp = _df_tear.copy().sort_values("sharpe", ascending=False)
        _df_tear_disp["win_rate"]        = _df_tear_disp["win_rate"].map("{:.0%}".format)
        _df_tear_disp["avg_return_pct"]  = _df_tear_disp["avg_return_pct"].map("{:+.2%}".format)
        _df_tear_disp["total_pnl_pct"]   = _df_tear_disp["total_pnl_pct"].map("{:+.1%}".format)
        _df_tear_disp["max_drawdown"]    = _df_tear_disp["max_drawdown"].map("{:.1%}".format)
        _df_tear_disp["sharpe"]          = _df_tear_disp["sharpe"].map("{:.2f}".format)
        _df_tear_disp["avg_holding_days"]= _df_tear_disp["avg_holding_days"].map("{:.0f}j".format)
        _df_tear_disp.columns = [
            "Agent", "Trades", "Win Rate", "Avg/Trade",
            "PnL Total", "Max DD", "Sharpe", "Hold Moy",
        ]
        st.dataframe(_df_tear_disp, use_container_width=True, hide_index=True)
    else:
        _scorer = LiveScorer()
        _metrics = _scorer.compute_agent_metrics()
        if _metrics:
            st.info("Aucun tearsheet CSV disponible — génération live depuis les logs.")
            _rows = [m.to_dict() for m in sorted(_metrics.values(),
                                                   key=lambda x: x.sharpe, reverse=True)]
            st.dataframe(pd.DataFrame(_rows), use_container_width=True, hide_index=True)
        else:
            st.markdown(
                "<div style='color:#555; padding:24px; text-align:center; background:#1a1d27; "
                "border-radius:10px;'>Aucun round-trip enregistré. Exécutez des ordres d'abord.</div>",
                unsafe_allow_html=True,
            )
except Exception as _e:
    st.markdown(f"<div style='color:#555; padding:20px;'>Erreur tearsheet : {_e}</div>",
                unsafe_allow_html=True)


# ====== FOOTER ======
st.markdown("""
<div style="border-top:1px solid #2a2d3a; margin-top:36px; padding-top:14px; text-align:center; color:#444; font-size:11px;">
    Milan Capital &nbsp;•&nbsp; AI Multi-Agent Hedge Fund &nbsp;•&nbsp; Paper Trading Only
</div>
""", unsafe_allow_html=True)
