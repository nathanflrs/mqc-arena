from __future__ import annotations

import dataclasses
import json
import os
import uuid
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import date, datetime, timezone
from pathlib import Path

import pandas as pd

from src.config import (
    WATCHLIST,
    AGENT_PRIORITY,
    EXECUTION_ENABLED,
    MAX_ORDERS_PER_RUN,
    MAX_NOTIONAL_PCT,
    LIMIT_BUFFER_BPS,
    IBKR_PORT,
    IBKR_CLIENT_ID,
    RISK_MAX_NET_LONG_PCT,
    RISK_MAX_SINGLE_POSITION_PCT,
    RISK_MIN_CASH_PCT,
    RISK_SELL_ONLY_MODE,
    STOP_LOSS_PCT,
)
from src.arena.arena import Arena
from src.arena.normalizer import ConfidenceNormalizer
from src.arena.selector import select_best
from src.agents.buffett import BuffettAgent
from src.agents.citadel import CitadelAgent
from src.agents.dummy import DummyHoldAgent
from src.agents.mean_reversion import MeanReversionAgent
from src.agents.macro import MacroAgent
from src.agents.trend_following import TrendFollowingAgent
from src.agents.dividend_arbitrage_agent import (
    DividendArbitrageAgent, DividendPositionTracker, DivPosition,
)
from src.agents.pairs_trading import PairsTradingAgent
from src.agents.volatility import VolatilityAgent
from src.agents.earnings_sentiment import EarningsSentimentAgent
from src.agents.cta_trend_agent import CTATrendAgent, CTA_UNIVERSE
from src.agents.insider_buy import InsiderBuyAgent
from src.agents.momentum import CrossSectionalMomentumAgent
from src.agents.base import MarketState
from src.broker.ibkr import connect_ibkr
from src.broker.portfolio import fetch_account_snapshot
from src.data.market_data import download_ohlcv, get_last_close_1d, normalize_ohlcv
from src.regime.detector import GMMRegimeDetector
from src.execution.planner import plan_from_signal, cta_plan_from_signal, pairs_plans_from_signal, OrderPlan
from src.execution.logger import log_order_plan, log_execution, log_decisions
from src.risk.manager import RiskConfig, RiskManager, DrawdownCircuitBreaker
from src.risk.allocator import AllocatorConfig, DynamicAllocator
from src.risk.correlation import CorrelationGuard
from src.risk.live_scorer import LiveScorer
from src.risk.earnings_filter import EarningsFilter
from src.risk.vol_sizing import vol_adjusted_weight
from src.risk.var_engine import HistoricalVaREngine, VaRConfig


def execute_plans_paper_ibkr(ib, snap, plans, plan_id: str) -> None:
    from ib_insync import LimitOrder, Stock
    import os

    exec_log = "logs/executions.csv"
    if os.path.exists(exec_log):
        df_exec = pd.read_csv(exec_log)
        if "plan_id" in df_exec.columns:
            if plan_id in df_exec["plan_id"].astype(str).values:
                _msg = f"❌ Plan {plan_id} already executed. Aborting."
                print(_msg)
                try:
                    from src.events.bus import get_bus, Event
                    get_bus().emit(Event(type="execution", severity="info",
                                        title="Plan déjà exécuté", body=_msg,
                                        meta={"plan_id": plan_id}))
                except Exception:
                    pass
                return

    max_notional = snap.net_liquidation * MAX_NOTIONAL_PCT
    buff = LIMIT_BUFFER_BPS / 10000.0

    candidates: list[tuple[object, str, int, float]] = []
    for p in plans:
        dq = int(round(p.delta_qty))
        if dq == 0:
            continue

        side = "BUY" if dq > 0 else "SELL"
        qty = abs(dq)

        if side == "SELL" and getattr(p, "strategy", "directional") != "market_neutral":
            # Directional SELL: cap at current holdings to prevent accidental shorts.
            qty = min(qty, int(round(p.current_qty)))
        # Market-neutral SELL (strategy='market_neutral'): no cap.
        # IBKR paper trading handles SELL on a non-held position as a short sale.
        # The RiskManager's gross pairs cap enforces position limits upstream.
        if qty <= 0:
            continue

        if float(p.est_notional) > max_notional:
            continue

        if side == "BUY":
            limit_price = float(p.last_price) * (1.0 + buff)
        else:
            limit_price = float(p.last_price) * (1.0 - buff)

        candidates.append((p, side, qty, limit_price))

    candidates = candidates[:MAX_ORDERS_PER_RUN]

    if not candidates:
        print("No orders after guards.")
        try:
            from src.events.bus import get_bus, Event
            get_bus().emit(Event(type="execution", severity="info",
                                title="Aucun ordre — filtrés / HOLD",
                                body="Milan Capital — Execution: no orders after guards (filtered / HOLD).",
                                meta={"plan_id": plan_id}))
        except Exception:
            pass
        return

    _exec_start = (
        "🚀 AUTO-EXEC — sending PAPER orders (LIMIT)\n"
        f"plan_id={plan_id}\n"
        f"max_orders={MAX_ORDERS_PER_RUN} | max_notional/order={max_notional:.0f} | buffer={LIMIT_BUFFER_BPS}bps"
    )
    print(_exec_start)
    try:
        from src.events.bus import get_bus, Event
        get_bus().emit(Event(type="execution", severity="info",
                            title=f"AUTO-EXEC démarré — {plan_id}",
                            body=_exec_start, meta={"plan_id": plan_id}))
    except Exception:
        pass

    Path("logs").mkdir(parents=True, exist_ok=True)
    placed: list[tuple] = []

    # ── 1. Place all orders ──────────────────────────────────────────────────
    for p, side, qty, limit_price in candidates:
        contract = Stock(p.symbol, "SMART", "USD")
        order = LimitOrder(side, qty, round(limit_price, 2))
        order.tif = "DAY"
        order.outsideRth = False
        trade = ib.placeOrder(contract, order)
        msg = f"📤 PAPER {p.symbol}: {side} {qty} @ LMT {order.lmtPrice} | status={trade.orderStatus.status}"
        print(msg)
        try:
            from src.events.bus import get_bus, Event
            get_bus().emit(Event(type="execution", severity="info",
                                title=f"Ordre PAPER — {p.symbol} {side}",
                                body=msg, meta={"plan_id": plan_id, "symbol": p.symbol, "side": side}))
        except Exception:
            pass
        placed.append((p, side, qty, float(order.lmtPrice), trade))

    # ── 2. Wait for IBKR paper fills (fills almost instantly on paper) ───────
    ib.sleep(10)

    # ── 3. Log real fill prices instead of theoretical limit prices ──────────
    ts = datetime.now(timezone.utc).replace(microsecond=0).isoformat()
    exec_rows = []
    for p, side, qty, limit_price, trade in placed:
        fill_px = trade.orderStatus.avgFillPrice
        fill_qty = trade.orderStatus.filled
        status = trade.orderStatus.status

        avg_fill = float(fill_px) if fill_px and float(fill_px) > 0 else 0.0
        actual_qty = int(fill_qty) if fill_qty and int(fill_qty) > 0 else qty
        last_px = float(p.last_price)

        # Slippage vs signal price (positive = unfavourable)
        if avg_fill > 0 and last_px > 0:
            if side == "BUY":
                slippage_bps = (avg_fill - last_px) / last_px * 10_000
            else:
                slippage_bps = (last_px - avg_fill) / last_px * 10_000
        else:
            slippage_bps = 0.0

        exec_rows.append({
            "plan_id": plan_id,
            "timestamp": ts,
            "symbol": p.symbol,
            "side": side,
            "qty": actual_qty,
            "limit_price": round(limit_price, 4),   # original limit order price
            "avg_fill_price": round(avg_fill, 4),    # actual IBKR fill (0 = not filled yet)
            "slippage_bps": round(slippage_bps, 2),  # vs signal price
            "last_price": round(last_px, 4),
            "est_notional": float(p.est_notional),
            "target_weight": float(p.target_weight),
            "reason": str(p.reason),
            "status": status,
        })

    log_execution(exec_rows)


def _load_entry_prices(
    avg_costs: dict[str, float] | None = None,
    plan_prices_path: str | Path = "logs/entry_prices.json",
    exec_path: str | Path = "logs/executions.csv",
) -> dict[str, float]:
    """
    3-tier lookup for position entry prices — used by the stop-loss engine.

    Tier 3 (lowest):  logs/entry_prices.json  — plan signal price, written at BUY
                      approval. Always available when a plan was created.
    Tier 2:           avg_costs               — IBKR-reported average cost basis.
                      Most precise when IBKR is connected.
    Tier 1 (highest): logs/executions.csv     — actual broker fill price.
                      Overrides when the fill was recorded.

    Each tier overwrites the previous so the highest-fidelity price wins.
    Returns {} when no source has data (stop-loss skips that symbol with a warning).
    """
    result: dict[str, float] = {}

    # Tier 3: plan signal price (persistent fallback)
    plan_path = Path(plan_prices_path)
    if plan_path.exists():
        try:
            result.update(json.loads(plan_path.read_text()))
        except Exception:
            pass

    # Tier 2: IBKR avgCost from live portfolio (overrides plan price)
    if avg_costs:
        result.update({k: v for k, v in avg_costs.items() if v > 0})

    # Tier 1: executions.csv actual fill price (highest priority)
    ex = Path(exec_path)
    if ex.exists():
        try:
            df = pd.read_csv(ex)
            buys = df[df["side"] == "BUY"].sort_values("timestamp")
            if "avg_fill_price" in buys.columns:
                price_col = buys["avg_fill_price"].where(
                    buys["avg_fill_price"] > 0, buys["limit_price"]
                )
                buys = buys.copy()
                buys["_price"] = price_col
                result.update(buys.groupby("symbol")["_price"].last().to_dict())
            else:
                result.update(buys.groupby("symbol")["limit_price"].last().to_dict())
        except Exception:
            pass

    return result


def _write_plan_entry_prices(
    approved_plans: list[OrderPlan],
    path: str | Path = "logs/entry_prices.json",
) -> None:
    """
    Persist entry prices for approved BUY plans so stop-loss has a reference
    even when IBKR fills are delayed or absent.

    - BUY plan:  record last_price as provisional entry (setdefault — first BUY wins,
                 scale-ins don't shift the reference).
    - SELL plan to target_qty=0: remove symbol (position fully closed).
    """
    p = Path(path)
    existing: dict[str, float] = {}
    if p.exists():
        try:
            existing = json.loads(p.read_text())
        except Exception:
            pass

    for plan in approved_plans:
        if plan.action == "BUY" and plan.last_price > 0:
            existing.setdefault(plan.symbol, plan.last_price)
        elif plan.action == "SELL" and plan.target_qty == 0.0:
            existing.pop(plan.symbol, None)

    try:
        p.write_text(json.dumps(existing, indent=2))
    except Exception as exc:
        print(f"⚠️  _write_plan_entry_prices: {exc}")


def _run_intraday_news_check(positions: dict[str, float]) -> None:
    """
    Fetch latest news for open individual-stock positions and emit a warning
    event for any earnings or M&A article found today.
    Non-blocking — any failure is silently swallowed.
    """
    try:
        from src.news.collector import NewsCollector
        from src.news.selector import NewsSelector
        from src.events.bus import get_bus, Event

        CTA_EXTRA   = ["TLT", "UUP", "DBC"]
        all_tickers = WATCHLIST + CTA_EXTRA

        collector = NewsCollector(cache_ttl_hours=6.0)
        selector  = NewsSelector()

        items  = collector.fetch_all(all_tickers, days_back=1)
        alerts = selector.check_intraday_alerts(items, positions)

        cat_icon = {"earnings": "📊", "ma": "🤝", "guidance": "📈",
                    "analyst": "🔍", "general": "📰"}
        for item in alerts:
            icon = cat_icon.get(item.category, "📰")
            get_bus().emit(Event(
                type     = "news",
                severity = "warning",
                title    = f"{icon} {item.category.upper()} — {item.ticker}",
                body     = item.headline,
                meta     = {
                    "ticker":   item.ticker,
                    "category": item.category,
                    "source":   item.source,
                    "url":      item.url,
                },
            ))
    except Exception as exc:
        print(f"⚠️  Intraday news check failed: {exc}")


def _send_post_execution_report(
    plans_sent: list,
    risk_report,
    corr_blocks: list,
    cb: DrawdownCircuitBreaker,
    snap,
    plan_id: str,
    regime: str,
) -> None:
    lines = [
        "📋 RAPPORT POST-EXÉCUTION — Milan Capital",
        f"plan_id={plan_id} | {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}",
        "",
    ]

    # Ordres envoyés
    if plans_sent:
        lines.append(f"✅ Ordres envoyés ({len(plans_sent)}) :")
        for p in plans_sent:
            side = "BUY" if p.delta_qty > 0 else ("SELL" if p.delta_qty < 0 else "HOLD")
            lines.append(
                f"  📤 {p.symbol}: {side} {abs(int(round(p.delta_qty)))} @ ~{p.last_price:.2f}"
                f" | ~${p.est_notional:.0f}"
            )
    else:
        lines.append("✅ Aucun ordre envoyé (plan vide après filtrage)")
    lines.append("")

    # Bloqués par le risk manager
    if risk_report.rejected:
        lines.append(f"🛡 Bloqués par risk manager ({len(risk_report.rejected)}) :")
        for r in risk_report.rejected:
            lines.append(f"  ✂️  {r.plan.symbol} ({r.plan.action}) → {r.reason}")
        lines.append("")

    # Bloqués par corrélation
    if corr_blocks:
        lines.append(f"🔗 Bloqués par corrélation ({len(corr_blocks)}) :")
        for b in corr_blocks:
            lines.append(f"  ✂️  {b['symbol']} (r={b['max_corr']:.2f} ↔ {b['correlated_with']})")
        lines.append("")

    # Sharpe et drawdown
    try:
        scorer = LiveScorer()
        metrics = scorer.compute_agent_metrics()
        if metrics:
            all_trips = scorer.get_roundtrips()
            from src.risk.live_scorer import _sharpe_from_roundtrips
            portfolio_sharpe = _sharpe_from_roundtrips(all_trips) if len(all_trips) >= 3 else None
            best = max(metrics.values(), key=lambda m: m.sharpe)
            if portfolio_sharpe is not None:
                lines.append(f"📈 Sharpe portefeuille : {portfolio_sharpe:.2f} | Meilleur : {best.agent} ({best.sharpe:.2f})")
            else:
                lines.append(f"📈 Sharpe : n/a (<3 round-trips) | Meilleur agent : {best.agent} ({best.sharpe:.2f})")
        else:
            lines.append("📈 Sharpe : n/a (aucun round-trip enregistré)")
    except Exception as e:
        lines.append(f"📈 Sharpe : erreur ({e})")

    cb_status = "🚨 ACTIF" if cb.is_triggered else "✅ inactif"
    lines.append(f"📉 Drawdown depuis pic : {cb.drawdown:.1%} | Circuit breaker : {cb_status}")
    lines.append("")

    # Résumé 5 lignes max
    n_sent = len(plans_sent)
    n_risk = len(risk_report.rejected)
    n_corr = len(corr_blocks)
    lines.append("─── Résumé ───")
    lines.append(f"Régime : {regime.upper()} | NetLiq : ${snap.net_liquidation:,.0f}")
    lines.append(f"Ordres : {n_sent} envoyés | {n_risk} bloqués risk | {n_corr} bloqués corrél.")
    lines.append(f"Drawdown : {cb.drawdown:.1%} | Circuit breaker : {cb_status}")

    body = "\n".join(lines)[:2000]
    try:
        from src.events.bus import get_bus, Event
        get_bus().emit(Event(
            type="execution",
            severity="info",
            title=f"Rapport post-exécution — {plan_id}",
            body=body,
            meta={"plan_id": plan_id, "regime": regime, "n_sent": len(plans_sent)},
        ))
    except Exception as e:
        print(f"⚠️  Post-execution event bus emit failed: {e}")


def main() -> None:
    print("✅ Runner started (IBKR READ-ONLY + ORDER PLAN)")

    ci_mode = os.getenv("CI", "").lower() == "true"
    if ci_mode:
        print("ℹ️  CI mode detected — Telegram disabled, no approval required.")

    ib = None
    ibkr_ok = False
    try:
        ib = connect_ibkr(port=IBKR_PORT, client_id=IBKR_CLIENT_ID)
        ibkr_ok = True
    except Exception as e:
        msg = f"⚠️ IBKR unavailable ({e}). Running in analysis-only mode (no execution)."
        print(msg)
        if not ci_mode:
            try:
                from src.events.bus import get_bus, Event
                get_bus().emit(Event(
                    type="system",
                    severity="critical",
                    title="IBKR unavailable — analysis-only mode",
                    body=msg,
                    meta={},
                ))
            except Exception:
                pass

    try:
        if ibkr_ok:
            snap = fetch_account_snapshot(ib)
            print(f"✅ IBKR connected | NetLiq={snap.net_liquidation:.2f} | Cash={snap.cash:.2f}")
            _ec_path = Path("logs/equity_curve.csv")
            _ec_header = not _ec_path.exists()
            with _ec_path.open("a") as _ec_f:
                if _ec_header:
                    _ec_f.write("date,netliq\n")
                _ec_f.write(f"{datetime.now(timezone.utc).date().isoformat()},{snap.net_liquidation:.2f}\n")
        else:
            from src.broker.portfolio import PortfolioSnapshot
            snap = PortfolioSnapshot(net_liquidation=100_000.0, cash=100_000.0, positions={})
            print("⚠️  IBKR offline — using paper snapshot (NetLiq=100 000, no positions)")
        print(f"Positions: {snap.positions if snap.positions else '{}'}")

        # ====== CIRCUIT BREAKER ======
        cb = DrawdownCircuitBreaker()
        # Only evaluate with real IBKR data — the offline fallback (100K) would
        # produce a spurious 90%+ drawdown against a real peak and falsely trigger.
        if ibkr_ok:
            cb.evaluate(snap.net_liquidation, ci_mode=ci_mode)
        if cb.level > 0:
            _cb_icon = {1: "🟡", 2: "🟠", 3: "🔴"}.get(cb.level, "")
            msg = (
                f"{_cb_icon} Circuit breaker — {cb.level_name} "
                f"(DD={cb.drawdown:.1%} depuis pic ${cb.peak_netliq:,.0f})"
                + (" — SELL-ONLY forcé." if cb.is_triggered else "")
            )
            print(msg)
            if not ci_mode:
                try:
                    from src.events.bus import get_bus, Event
                    get_bus().emit(Event(type="alert", severity="info",
                                        title=f"Circuit Breaker — Niveau {cb.level}",
                                        body=msg, meta={"cb_level": cb.level, "drawdown": cb.drawdown}))
                except Exception:
                    pass

        # ── Intraday news alerts (individual-stock positions only) ──────────
        if not ci_mode and snap.positions:
            _run_intraday_news_check(snap.positions)

        # Batch download — un seul appel réseau, thread-safe, ~10-15s.
        import yfinance as _yf
        _raw_batch = _yf.download(
            list(WATCHLIST),
            period="2y",
            interval="1d",
            auto_adjust=True,
            progress=False,
            group_by="ticker",
        )
        all_data: dict = {}
        for sym in WATCHLIST:
            try:
                _df_sym = _raw_batch[sym].copy() if isinstance(_raw_batch.columns, pd.MultiIndex) else _raw_batch.copy()
                all_data[sym] = normalize_ohlcv(_df_sym)
            except Exception as _e:
                print(f"⚠️  {sym}: données manquantes ({_e})")

        # Arena créée APRÈS all_data — MacroAgent reçoit SPY/GLD pré-chargés
        # pour éviter 2 appels réseau redondants par run.
        # CrossSectionalMomentumAgent reçoit tout all_data pour le ranking JT.
        arena = Arena([
            DummyHoldAgent(),
            BuffettAgent(),
            CitadelAgent(),
            MeanReversionAgent(),
            MacroAgent(preloaded={
                "SPY": all_data.get("SPY"),
                "GLD": all_data.get("GLD"),
            }),
            TrendFollowingAgent(),
            DividendArbitrageAgent(),
            PairsTradingAgent(),
            VolatilityAgent(),
            EarningsSentimentAgent(),
            InsiderBuyAgent(),
            CrossSectionalMomentumAgent(universe=all_data),
        ])

        _gmm_detector = GMMRegimeDetector()
        _spy_df       = all_data["SPY"]
        regime_data   = _gmm_detector.get_regime(_spy_df)
        regime        = regime_data["regime"]
        gmm_kelly_scale = GMMRegimeDetector.regime_kelly_scale(
            regime_data.get("gmm_regime"),
            regime_data.get("proba", {}),
        )
        _regime_line = (
            f"\n🌍 RÉGIME DÉTECTÉ : {regime.upper()} | "
            f"SPY={regime_data['price']} | "
            f"SMA50={regime_data['sma50']} | "
            f"SMA200={regime_data['sma200']} | "
            f"Vol={regime_data['vol_regime']}"
        )
        if regime_data.get("gmm_available"):
            _regime_line += (
                f"\n   GMM: {regime_data['gmm_regime'].upper()}"
                f" (p={regime_data['proba'].get(regime_data['gmm_regime'], 0):.0%})"
                f" | Kelly scale: {gmm_kelly_scale:.2f}"
            )
        else:
            _regime_line += "\n   GMM: not fitted — run: python -m src.regime.detector"
        print(_regime_line)

        plan_id = uuid.uuid4().hex[:8]
        plans = []
        _decisions_summary: list[dict] = []
        _pending_divArb: dict[str, dict] = {}  # sym → signal meta, populated before RiskManager

        # ====== VAR TEMPS RÉEL ======
        _var_engine = HistoricalVaREngine(VaRConfig(lookback_days=252, risk_budget_pct=0.03))
        _var_result = _var_engine.compute(snap.positions, all_data, snap.net_liquidation)
        if _var_result is not None:
            _var_line = (
                f"\n📊 VaR 1-jour — 95%: ${_var_result.var_95_usd:,.0f} ({_var_result.var_95_pct:.2%}) | "
                f"99%: ${_var_result.var_99_usd:,.0f} ({_var_result.var_99_pct:.2%}) | "
                f"CVaR 99%: ${_var_result.cvar_99_usd:,.0f} | "
                f"{_var_result.n_positions} position(s) × {_var_result.n_days}j"
            )
            print(_var_line)
            if _var_engine.exceeds_budget(_var_result, snap.net_liquidation) and not ci_mode:
                try:
                    from src.events.bus import get_bus, Event
                    get_bus().emit(Event(
                        type="alert",
                        severity="critical",
                        title=f"VaR 99% dépasse le budget risque",
                        body=(
                            f"🚨 VaR 1-jour 99% = ${_var_result.var_99_usd:,.0f} "
                            f"({_var_result.var_99_usd / snap.net_liquidation:.2%} du portfolio) "
                            f"> seuil 3%.\n"
                            f"CVaR 99% = ${_var_result.cvar_99_usd:,.0f}\n"
                            f"Positions: {_var_result.n_positions} | Historique: {_var_result.n_days}j"
                        ),
                        meta={
                            "var_99_usd": _var_result.var_99_usd,
                            "var_99_pct": _var_result.var_99_pct,
                            "cvar_99_usd": _var_result.cvar_99_usd,
                            "budget_pct": _var_engine.cfg.risk_budget_pct,
                        },
                    ))
                except Exception as _ve:
                    print(f"⚠️  VaR alert emit failed: {_ve}")

        # ADV 10 jours pour filtre liquidité (shares, calculé depuis Volume OHLCV)
        adv_map: dict[str, float] = {
            sym: float(all_data[sym]["Volume"].tail(10).mean())
            for sym in WATCHLIST
            if sym in all_data and len(all_data[sym]) >= 10
        }

        # ====== EARNINGS FILTER — pré-chargement ======
        earnings_filter = EarningsFilter(buffer_days=3)
        earnings_filter.prefetch(WATCHLIST)

        alloc_agents = [BuffettAgent(), CitadelAgent(), MeanReversionAgent(), TrendFollowingAgent()]
        alloc_result = DynamicAllocator(AllocatorConfig()).compute(all_data, alloc_agents)

        # ====== KELLY WEIGHTS (depuis live round-trips) ======
        _live_scorer = LiveScorer()
        kelly_weights: dict[str, float] = _live_scorer.compute_kelly_weights(min_trades=5)
        if kelly_weights:
            print(f"\n📐 Kelly demi-fraction : {kelly_weights}")

        print("\n" + alloc_result.telegram_summary())

        # P0(b) : normalisation intra-agent (min-max sur tout l'historique)
        # Corrige les biais d'échelle entre agents — pas une calibration.
        # Les signaux bruts sont conservés pour le logging ; seule la sélection utilise les valeurs normalisées.
        _normalizer = ConfidenceNormalizer.from_frozen_json()

        for sym in WATCHLIST:
            if sym not in all_data:
                print(f"⚠️  {sym} ignoré — données manquantes")
                continue
            df = all_data[sym]

            # Priority dynamique (remplace AGENT_PRIORITY statique, fallback si symbole absent)
            priority = alloc_result.best_agent.get(sym) or AGENT_PRIORITY.get(sym)

            signals = arena.run(sym, df, portfolio=snap.positions, regime=regime)

            # Normalisation P0(b) : on sélectionne sur confidences normalisées,
            # mais winner est récupéré depuis les signaux bruts (pour circuit breaker, logs).
            _signals_norm = _normalizer.normalize_all(signals)
            _winner_norm = select_best(_signals_norm, priority_agent=priority)
            winner = (
                next((s for s in signals if s.agent_name == _winner_norm.agent_name), None)
                if _winner_norm else None
            )

            print(f"\n=== RUN {sym} | regime={regime} | priority={priority} ===")
            for s in signals:
                print(" -", s)
            print("🏆 WINNER:", winner)

            log_decisions(
                signals,
                symbol=sym,
                regime=regime,
                plan_id=plan_id,
                winner_agent=winner.agent_name if winner else None,
            )

            if winner is None:
                _decisions_summary.append({"symbol": sym, "agent": "NONE", "action": "HOLD"})
                continue

            # Ajuste le target_weight : DynamicAllocator en priorité, puis Kelly si dispo
            dynamic_weight = alloc_result.weights.get(winner.agent_name, {}).get(sym)
            if dynamic_weight is not None:
                winner = dataclasses.replace(winner, target_weight=dynamic_weight)

            # Kelly blending : blend progressif Kelly ↔ dynamic au fur et à mesure des trades
            # α = min(1, n_trades/20) — 100% Kelly à partir de 20 round-trips par agent
            kelly_w = kelly_weights.get(winner.agent_name)
            if kelly_w is not None and kelly_w > 0:
                kelly_w_scaled = round(kelly_w * gmm_kelly_scale, 4)
                n_trips = _live_scorer.get_n_trades(winner.agent_name, sym)
                alpha = min(1.0, n_trips / 20)
                blended_w = round((1 - alpha) * winner.target_weight + alpha * kelly_w_scaled, 4)
                print(
                    f"   📐 Kelly {winner.agent_name}/{sym}: "
                    f"dynamic={winner.target_weight:.3f} kelly={kelly_w:.3f} "
                    f"×{gmm_kelly_scale:.2f}(GMM)={kelly_w_scaled:.3f} "
                    f"α={alpha:.2f} → {blended_w:.3f} ({n_trips} trips)"
                )
                winner = dataclasses.replace(winner, target_weight=blended_w)

            # ── Vol-adjusted sizing ──────────────────────────────────────────
            if winner.action == "BUY":
                adj_w, vol_reason = vol_adjusted_weight(df, winner.target_weight)
                if vol_reason:
                    print(f"   Vol sizing {sym}: {vol_reason}")
                    winner = dataclasses.replace(winner, target_weight=adj_w)

            # ── Earnings filter — block BUY within 3 bdays of earnings ──────
            if winner.action == "BUY":
                blocked, earn_reason = earnings_filter.should_block_buy(sym)
                if blocked:
                    msg = f"Earnings filter: {sym} BUY blocked — {earn_reason}"
                    print(msg)
                    if not ci_mode:
                        try:
                            from src.events.bus import get_bus, Event
                            get_bus().emit(Event(type="alert", severity="info",
                                                title=f"Earnings filter — {sym} bloqué",
                                                body=f"📅 {msg}",
                                                meta={"symbol": sym}))
                        except Exception:
                            pass
                    winner = dataclasses.replace(
                        winner,
                        action="HOLD",
                        reason=f"EARNINGS_BLOCK: {earn_reason}",
                    )

            # CB niveau 2 (ALERTE) : bloquer BUY si confiance < 0.85
            if cb.level == 2 and winner.action == "BUY" and winner.confidence < 0.85:
                winner = dataclasses.replace(
                    winner,
                    action="HOLD",
                    reason=f"CB_ALERTE: DD {cb.drawdown:.1%} — conf {winner.confidence:.2f} < 0.85",
                )

            last_px = get_last_close_1d(df)
            current_qty = snap.positions.get(sym, 0.0)

            if winner.agent_name == "PairsTradingAgent":
                # Build a prices dict with both legs; fetch partner price on demand
                meta = winner.meta or {}
                pair_legs = meta.get("pair_legs", {})
                direction = meta.get("direction", "")
                if direction in ("long_a_short_b", "short_a_long_b"):
                    leg_tickers = {pair_legs["long"][0], pair_legs["short"][0]}
                elif direction == "close":
                    leg_tickers = {pair_legs.get("close_long", ""), pair_legs.get("close_short", "")}
                else:
                    leg_tickers = set()
                prices: dict[str, float] = {}
                for t in leg_tickers:
                    if t in all_data:
                        prices[t] = get_last_close_1d(all_data[t])
                    else:
                        try:
                            _partner_df = download_ohlcv(t)
                            all_data[t] = _partner_df
                            prices[t] = get_last_close_1d(_partner_df)
                        except Exception as exc:
                            print(f"⚠️  PairsTradingAgent: no price data for {t} — {exc}")
                pair_plans = pairs_plans_from_signal(
                    winner,
                    net_liquidation=snap.net_liquidation,
                    prices=prices,
                    current_qtys=snap.positions,
                )
                plans.extend(pair_plans)
                if not pair_plans:
                    print(f"⚠️  PairsTradingAgent: signal for {sym} produced 0 plans — skipped")
            else:
                plan = plan_from_signal(
                    winner,
                    net_liquidation=snap.net_liquidation,
                    last_price=last_px,
                    current_qty=current_qty,
                )
                plans.append(plan)
                # DivArb: capture pending position data — written to tracker only after
                # RiskManager confirms the plan (prevents phantom SELL on next run).
                if winner.agent_name == "DividendArbitrageAgent" and winner.action == "BUY":
                    _pending_divArb[sym] = winner.meta or {}
            _decisions_summary.append({"symbol": sym, "agent": winner.agent_name, "action": winner.action})

        # ====== STOP-LOSS PAR POSITION ======
        entry_prices = _load_entry_prices(avg_costs=snap.avg_costs)
        for sym, qty in snap.positions.items():
            if qty <= 0 or sym not in all_data:
                continue
            entry_px = entry_prices.get(sym)
            if entry_px is None:
                print(f"⚠️  Stop-loss {sym}: aucun prix d'entrée connu (fill absent, avgCost=0, plan non enregistré) — ignoré")
                continue
            current_px = get_last_close_1d(all_data[sym])
            pnl_pct = (current_px - entry_px) / entry_px
            if pnl_pct < -STOP_LOSS_PCT:
                # Retire tout plan existant pour ce symbole et injecte un SELL forcé
                plans = [p for p in plans if p.symbol != sym]
                plans.append(OrderPlan(
                    symbol=sym,
                    action="SELL",
                    target_weight=0.0,
                    last_price=current_px,
                    current_qty=float(qty),
                    target_qty=0.0,
                    delta_qty=-float(qty),
                    est_notional=float(qty) * current_px,
                    reason=f"STOP-LOSS: {pnl_pct:.1%} < -{STOP_LOSS_PCT:.0%}",
                ))
                msg = f"🛑 STOP-LOSS {sym}: {pnl_pct:.1%} (entrée ${entry_px:.2f} → ${current_px:.2f})"
                print(msg)
                if not ci_mode:
                    try:
                        from src.events.bus import get_bus, Event
                        get_bus().emit(Event(type="alert", severity="critical",
                                            title=f"STOP-LOSS déclenché — {sym}",
                                            body=msg,
                                            meta={"symbol": sym, "pnl_pct": pnl_pct}))
                    except Exception:
                        pass

        # ====== CTA TREND — boucle séparée (TLT/UUP/DBC hors WATCHLIST) ======
        # SPY/QQQ/GLD sont dans WATCHLIST et traités par l'Arena ci-dessus.
        # Seuls TLT/UUP/DBC n'ont pas de couverture Arena et nécessitent ce loop.
        _cta_agent  = CTATrendAgent()
        _CTA_EXTRA  = [s for s in CTA_UNIVERSE if s not in WATCHLIST]  # TLT, UUP, DBC
        cta_data: dict = dict(all_data)
        for _sym in _CTA_EXTRA:
            if _sym not in cta_data:
                try:
                    cta_data[_sym] = download_ohlcv(_sym)
                except Exception as _exc:
                    print(f"⚠️  CTA: téléchargement {_sym} échoué: {_exc}")

        print("\n====== CTA TREND SIGNALS ======")
        for _sym in _CTA_EXTRA:
            _df_cta = cta_data.get(_sym)
            if _df_cta is None or _df_cta.empty:
                print(f"   {_sym}: données absentes — HOLD")
                continue
            _px_cta  = get_last_close_1d(_df_cta)
            _state   = MarketState(symbol=_sym, price=_px_cta,
                                   timestamp=str(_df_cta.index[-1]))
            _sig_cta = _cta_agent.generate_signal(
                _state, snap.positions, regime=regime, data=_df_cta
            )
            _plan_cta = cta_plan_from_signal(
                _sig_cta,
                net_liquidation=snap.net_liquidation,
                last_price=_px_cta,
                current_qty=float(snap.positions.get(_sym, 0.0)),
            )
            print(
                f"   {_sym}: {_sig_cta.action} | tw={_sig_cta.target_weight:+.3f} | "
                f"conf={_sig_cta.confidence:.2f} | {_sig_cta.reason[:80]}"
            )
            plans.append(_plan_cta)
            _decisions_summary.append({
                "symbol": _sym, "agent": "CTATrendAgent", "action": _sig_cta.action
            })

        print("\n====== ORDER PLAN (NO EXECUTION) ======")
        if not plans:
            print("No plans.")
        else:
            for p in plans:
                print(
                    f"{p.symbol}: {p.action} | tgt_w={p.target_weight:.2f} | "
                    f"px={p.last_price:.2f} | cur={p.current_qty:.0f} -> tgt={p.target_qty:.0f} "
                    f"| delta={p.delta_qty:+.0f} | est$={p.est_notional:.0f} | {p.reason}"
                )

        # ====== RISK MANAGER ======
        # CB graduated response : levels 1/2 override max_net_long et bloquent le scaling régime.
        # Level 3 (URGENCE) → sell-only. Les levels 0-2 laissent RiskManager appliquer
        # le scaling régime GMM (sauf si CB est actif — CB prend alors le contrôle total).
        _cb_long_override = {0: None, 1: 0.25, 2: 0.12, 3: 0.0}[cb.level]
        if _cb_long_override is not None:
            risk_max_long       = _cb_long_override
            gmm_regime_for_risk = None   # CB prime sur le scaling régime
        else:
            risk_max_long       = RISK_MAX_NET_LONG_PCT
            gmm_regime_for_risk = regime_data.get("gmm_regime")

        risk_cfg = RiskConfig(
            max_net_long_pct=risk_max_long,
            max_single_position_pct=RISK_MAX_SINGLE_POSITION_PCT,
            min_cash_pct=RISK_MIN_CASH_PCT,
            sell_only_mode=RISK_SELL_ONLY_MODE or cb.is_triggered,
        )
        risk_report = RiskManager(risk_cfg).check(
            plans,
            snap,
            gmm_regime=gmm_regime_for_risk,
            adv_map=adv_map,
            cb_level=cb.level,
        )

        if risk_report.rejected:
            print(f"⚠️  Risk manager: {len(risk_report.rejected)} plan(s) rejeté(s)")
            for r in risk_report.rejected:
                print(f"   ✂️  {r.plan.symbol} ({r.plan.action}): {r.reason}")

        plans = risk_report.approved

        # ====== CORRELATION GUARD ======
        corr_guard = CorrelationGuard(threshold=0.7, lookback_days=60)
        plans, corr_blocks = corr_guard.filter_plans(plans, snap, all_data)
        if corr_blocks:
            for block in corr_blocks:
                msg = (
                    f"⚠️  Corrélation: {block['symbol']} bloqué "
                    f"(r={block['max_corr']:.2f} avec {block['correlated_with']})"
                )
                print(msg)
                if not ci_mode:
                    try:
                        from src.events.bus import get_bus, Event
                        get_bus().emit(Event(type="alert", severity="info",
                                            title=f"Corrélation — {block['symbol']} bloqué",
                                            body=msg,
                                            meta={"symbol": block["symbol"], "corr": block["max_corr"]}))
                    except Exception:
                        pass

        # Persist BUY plan prices so stop-loss has a reference even without IBKR fills
        _write_plan_entry_prices(plans)

        # Record DivArb positions AFTER RiskManager + CorrelationGuard — only for approved plans.
        # This prevents phantom SELL signals when a DivArb BUY is rejected.
        if _pending_divArb:
            _divArb_tracker = DividendPositionTracker()
            for _plan in plans:
                if _plan.symbol in _pending_divArb and _plan.action == "BUY":
                    _dmeta = _pending_divArb[_plan.symbol]
                    try:
                        _divArb_tracker.open_position(DivPosition(
                            ticker           = _plan.symbol,
                            entry_date       = date.today().isoformat(),
                            entry_price      = _plan.last_price,
                            ex_date          = _dmeta["ex_date"],
                            dividend_amount  = _dmeta["dividend_amount"],
                            target_exit_date = _dmeta.get("_divpos_target_exit", ""),
                            shares           = float(_plan.delta_qty),
                        ))
                    except Exception as _de:
                        print(f"⚠️  DivArb tracker write failed for {_plan.symbol}: {_de}")

        log_order_plan(plans, plan_id=plan_id)

        if not ci_mode:
            try:
                from src import track_record
                digest = track_record.append_daily_entry(
                    plan_id=plan_id,
                    regime=regime,
                    decisions_summary=_decisions_summary,
                    netliq=snap.net_liquidation,
                )
                print(f"✅ Track record signed: {digest[:16]}…")
            except Exception as _tr_err:
                print(f"⚠️  Track record: {_tr_err}")

        if not plans:
            msg = f"Milan Capital — ORDER PLAN ready\nplan_id={plan_id}\nNo plans."
            print(msg)
            if not ci_mode:
                try:
                    from src.events.bus import get_bus, Event
                    get_bus().emit(Event(type="execution", severity="info",
                                        title=f"Plan vide — {plan_id}",
                                        body=msg, meta={"plan_id": plan_id}))
                except Exception:
                    pass
            return

        lines = []
        for p in plans:
            side = "BUY" if p.delta_qty > 0 else ("SELL" if p.delta_qty < 0 else "HOLD")
            lines.append(f"{p.symbol}: {side} | delta={p.delta_qty:+.0f} | est$={p.est_notional:.0f}")

        plan_summary = (
            f"🌍 Régime: {regime.upper()} | Vol: {regime_data['vol_regime']}\n"
            "Milan Capital — ORDER PLAN ready\n"
            f"plan_id={plan_id}\n\n"
            + "\n".join(lines)
            + "\n\n"
            + alloc_result.telegram_summary()
            + "\n\n"
            + risk_report.telegram_summary()
        )
        print(plan_summary)

        if ci_mode:
            print("ℹ️  CI mode — order plan logged to logs/order_plan.csv. No Telegram, no execution.")
            return

        execution_enabled = EXECUTION_ENABLED and ibkr_ok
        if not execution_enabled:
            reason = "IBKR unavailable" if not ibkr_ok else "EXECUTION_ENABLED=false"
            _msg = f"🧯 {reason} → NO orders sent. plan_id={plan_id}"
            print(_msg)
            try:
                from src.events.bus import get_bus, Event
                get_bus().emit(Event(
                    type="execution",
                    severity="info",
                    title=f"Exécution désactivée — {reason}",
                    body=_msg,
                    meta={"plan_id": plan_id, "reason": reason},
                ))
            except Exception:
                pass
            print(f"🧯 {reason} → NO orders sent.")
            return

        execute_plans_paper_ibkr(ib, snap, plans, plan_id)
        _send_post_execution_report(plans, risk_report, corr_blocks, cb, snap, plan_id, regime)

    finally:
        if ib is not None:
            ib.disconnect()
        print("\n✅ Done.")


if __name__ == "__main__":
    main()
    