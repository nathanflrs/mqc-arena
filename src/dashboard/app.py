# src/dashboard/app.py
from __future__ import annotations

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

import pandas as pd
import streamlit as st
import plotly.graph_objects as go
from dotenv import load_dotenv

load_dotenv()

from src.agents.buffett import BuffettAgent
from src.agents.citadel import CitadelAgent
from src.agents.mean_reversion import MeanReversionAgent
from src.agents.macro import MacroAgent
from src.agents.trend_following import TrendFollowingAgent
from src.agents.dummy import DummyHoldAgent
from src.arena.arena import Arena
from src.arena.selector import select_best
from src.data.market_data import download_ohlcv
from src.data.regime import detect_regime
from src.config import WATCHLIST

st.set_page_config(
    page_title="Milan Capital",
    page_icon="💼",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ====== HEADER ======
col1, col2 = st.columns([3, 1])
with col1:
    st.markdown("# 💼 Milan Capital")
    st.markdown("*AI-Powered Multi-Agent Hedge Fund*")
with col2:
    st.markdown("<br>", unsafe_allow_html=True)
    if st.button("🔄 Refresh", use_container_width=True):
        st.cache_data.clear()
        st.rerun()

st.divider()


# ====== DATA LOADING ======
@st.cache_data(ttl=300)
def load_data():
    regime_data = detect_regime("SPY")
    arena = Arena([
        DummyHoldAgent(),
        BuffettAgent(),
        CitadelAgent(),
        MeanReversionAgent(),
        MacroAgent(),
        TrendFollowingAgent(),
    ])
    results = {}
    for sym in WATCHLIST:
        df = download_ohlcv(sym)
        signals = arena.run(sym, df, regime=regime_data["regime"])
        winner = select_best(signals)
        results[sym] = {"signals": signals, "winner": winner, "df": df}
    return regime_data, results


with st.spinner("Chargement des données..."):
    regime_data, results = load_data()

regime = regime_data["regime"]


# ====== REGIME ======
regime_emoji = {"bull": "🟢", "bear": "🔴", "choppy": "🟡"}
r1, r2, r3, r4, r5 = st.columns(5)
with r1:
    st.metric("RÉGIME", f"{regime_emoji.get(regime,'')} {regime.upper()}")
with r2:
    st.metric("SPY", f"${regime_data['price']}")
with r3:
    st.metric("SMA50", f"${regime_data['sma50']}")
with r4:
    st.metric("SMA200", f"${regime_data['sma200']}")
with r5:
    st.metric("VOLATILITÉ", regime_data['vol_regime'].upper())

st.divider()


# ====== HELPERS ======
def render_agent_card(col, sig, is_winner):
    name = sig.agent_name.replace("Agent", "")
    label = "⭐ WINNER\n" if is_winner else ""
    text = f"{label}**{name}**\n\n{sig.action} — conf: {sig.confidence:.0%}"
    with col:
        if sig.action == "BUY":
            st.success(text)
        elif sig.action == "SELL":
            st.error(text)
        else:
            st.info(text)


def render_cio_card(col, winner):
    with col:
        if winner and winner.agent_name != "DummyHoldAgent":
            wname = winner.agent_name.replace("Agent", "")
            text = f"🏆 **CIO DECISION**\n\n**{winner.action}** — {wname}"
            if winner.action == "BUY":
                st.success(text)
            elif winner.action == "SELL":
                st.error(text)
            else:
                st.info(text)
        else:
            st.info("🏆 **CIO DECISION**\n\n⛔ NO TRADE")


# ====== PIPELINE ======
st.markdown("### 🔄 Pipeline des Agents")

for sym in WATCHLIST:
    data = results[sym]
    signals = [s for s in data["signals"] if s.agent_name != "DummyHoldAgent"]
    winner = data["winner"]

    st.markdown(f"#### 📊 {sym}")
    cols = st.columns(len(signals) + 2)

    with cols[0]:
        st.info(f"📈 **{sym}**\n\nMARKET DATA")

    for i, sig in enumerate(signals):
        is_winner = winner is not None and sig.agent_name == winner.agent_name
        render_agent_card(cols[i + 1], sig, is_winner)

    render_cio_card(cols[-1], winner)
    st.markdown("---")


# ====== ORDER PLAN ======
st.markdown("### 📋 Order Plan")

try:
    df_plan = pd.read_csv("logs/order_plan.csv")
    if not df_plan.empty:
        for _, row in df_plan.tail(10).iterrows():
            side = row["side"]
            symbol = row["symbol"]
            delta = row["delta_qty"]
            price = row["last_price"]
            notional = row["est_notional"]
            reason = str(row["reason"])[:60]
            text = f"**{side}** {symbol} | delta: {delta:+.0f} shares | @ ${price:.2f} | est. ${notional:.0f} | _{reason}_"
            if side == "BUY":
                st.success(text)
            elif side == "SELL":
                st.error(text)
            else:
                st.info(text)
    else:
        st.info("Aucun order plan disponible.")
except FileNotFoundError:
    st.info("Lancez d'abord le runner pour générer un order plan.")


# ====== EQUITY CURVE ======
st.divider()
st.markdown("### 📈 Equity Curve (Paper)")

try:
    df_exec = pd.read_csv("logs/executions.csv")
    if not df_exec.empty and "timestamp" in df_exec.columns:
        df_exec["timestamp"] = pd.to_datetime(df_exec["timestamp"])
        df_exec = df_exec.sort_values("timestamp")
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=df_exec["timestamp"],
            y=df_exec["est_notional"].cumsum(),
            mode="lines+markers",
            line=dict(color="#00ff88", width=2),
            fill="tozeroy",
            fillcolor="rgba(0,255,136,0.05)",
            name="Notional cumulé",
        ))
        fig.update_layout(
            paper_bgcolor="#0e1117",
            plot_bgcolor="#1a1d27",
            font=dict(color="#fff"),
            xaxis=dict(gridcolor="#333"),
            yaxis=dict(gridcolor="#333"),
            margin=dict(l=0, r=0, t=0, b=0),
            height=300,
        )
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("Pas encore d'exécutions. Approuvez un order plan pour voir l'equity curve.")
except FileNotFoundError:
    st.info("Pas encore d'exécutions.")


# ====== BACKTEST RESULTS ======
st.divider()
st.markdown("### 🧪 Résultats Backtest (3 ans)")

try:
    df_bt = pd.read_csv("logs/backtest_results.csv")
    if not df_bt.empty:

        # Métriques clés en haut
        best = df_bt.sort_values("sharpe_ratio", ascending=False).iloc[0]
        b1, b2, b3, b4 = st.columns(4)
        with b1:
            st.metric("🏆 Meilleur Agent", best["agent"].replace("Agent", ""))
        with b2:
            st.metric("📈 Meilleur Return", f"{best['total_return']:+.1%}")
        with b3:
            st.metric("⚡ Meilleur Sharpe", f"{best['sharpe_ratio']:.2f}")
        with b4:
            st.metric("📉 Son Drawdown", f"{best['max_drawdown']:.1%}")

        st.markdown("<br>", unsafe_allow_html=True)

        # Tableau complet
        df_display = df_bt.copy()
        df_display = df_display.sort_values("sharpe_ratio", ascending=False)
        df_display["total_return"] = df_display["total_return"].map("{:+.1%}".format)
        df_display["annualized_return"] = df_display["annualized_return"].map("{:+.1%}".format)
        df_display["sharpe_ratio"] = df_display["sharpe_ratio"].map("{:.2f}".format)
        df_display["max_drawdown"] = df_display["max_drawdown"].map("{:.1%}".format)
        df_display["win_rate"] = df_display["win_rate"].map("{:.0%}".format)
        df_display.columns = ["Agent", "Symbole", "Return Total", "Return Annualisé", "Sharpe", "Max Drawdown", "Win Rate", "Nb Trades"]
        st.dataframe(df_display, use_container_width=True, hide_index=True)

        st.markdown("<br>", unsafe_allow_html=True)

        # Graphique barres — Sharpe par agent/symbole
        fig2 = go.Figure()
        for sym in df_bt["symbol"].unique():
            sub = df_bt[df_bt["symbol"] == sym]
            fig2.add_trace(go.Bar(
                name=sym,
                x=sub["agent"].str.replace("Agent", ""),
                y=sub["sharpe_ratio"],
            ))

        fig2.update_layout(
            barmode="group",
            paper_bgcolor="#0e1117",
            plot_bgcolor="#1a1d27",
            font=dict(color="#fff"),
            xaxis=dict(gridcolor="#333"),
            yaxis=dict(gridcolor="#333", title="Sharpe Ratio"),
            legend=dict(bgcolor="#1a1d27"),
            margin=dict(l=0, r=0, t=30, b=0),
            height=350,
            title=dict(text="Sharpe Ratio par Agent et Symbole", font=dict(color="#fff")),
        )
        fig2.add_hline(y=0, line_dash="dash", line_color="#ff4444", annotation_text="Seuil 0")
        st.plotly_chart(fig2, use_container_width=True)

    else:
        st.info("Lancez d'abord le backtest.")
except FileNotFoundError:
    st.info("Lancez d'abord : python -m src.backtest.run_backtest")


# ====== FOOTER ======
st.divider()
st.caption("Milan Capital — AI Multi-Agent Hedge Fund — Paper Trading Only")
