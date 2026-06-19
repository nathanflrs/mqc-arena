from __future__ import annotations

import dataclasses
import os
import uuid
from datetime import datetime, timezone
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
from src.arena.selector import select_best
from src.agents.buffett import BuffettAgent
from src.agents.citadel import CitadelAgent
from src.agents.dummy import DummyHoldAgent
from src.agents.mean_reversion import MeanReversionAgent
from src.agents.macro import MacroAgent
from src.agents.trend_following import TrendFollowingAgent
from src.agents.dividend_arbitrage import DividendArbitrageAgent
from src.agents.pairs_trading import PairsTradingAgent
from src.agents.volatility import VolatilityAgent
from src.agents.earnings_sentiment import EarningsSentimentAgent
from src.broker.ibkr import connect_ibkr
from src.broker.portfolio import fetch_account_snapshot
from src.data.market_data import download_ohlcv, get_last_close_1d
from src.data.regime import detect_regime
from src.execution.planner import plan_from_signal, OrderPlan
from src.execution.logger import log_order_plan, log_execution, log_decisions
from src.notify.telegram import send_message
from src.risk.manager import RiskConfig, RiskManager, DrawdownCircuitBreaker
from src.risk.allocator import AllocatorConfig, DynamicAllocator
from src.risk.correlation import CorrelationGuard
from src.risk.live_scorer import LiveScorer


def execute_plans_paper_ibkr(ib, snap, plans, plan_id: str) -> None:
    from ib_insync import LimitOrder, Stock
    import os

    exec_log = "logs/executions.csv"
    if os.path.exists(exec_log):
        df_exec = pd.read_csv(exec_log)
        if "plan_id" in df_exec.columns:
            if plan_id in df_exec["plan_id"].astype(str).values:
                send_message(f"❌ Plan {plan_id} already executed. Aborting.")
                print(f"❌ Plan {plan_id} already executed. Aborting.")
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

        if side == "SELL":
            qty = min(qty, int(round(p.current_qty)))
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
        send_message("Milan Capital — Execution: no orders after guards (filtered / HOLD).")
        print("No orders after guards.")
        return

    send_message(
        "🚀 AUTO-EXEC — sending PAPER orders (LIMIT)\n"
        f"plan_id={plan_id}\n"
        f"max_orders={MAX_ORDERS_PER_RUN} | max_notional/order={max_notional:.0f} | buffer={LIMIT_BUFFER_BPS}bps"
    )

    Path("logs").mkdir(parents=True, exist_ok=True)
    placed: list[tuple] = []

    # ── 1. Place all orders ──────────────────────────────────────────────────
    for p, side, qty, limit_price in candidates:
        contract = Stock(p.symbol, "SMART", "USD")
        order = LimitOrder(side, qty, round(limit_price, 2))
        trade = ib.placeOrder(contract, order)
        msg = f"📤 PAPER {p.symbol}: {side} {qty} @ LMT {order.lmtPrice} | status={trade.orderStatus.status}"
        print(msg)
        send_message(msg)
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


def _load_entry_prices() -> dict[str, float]:
    """Returns the last recorded BUY fill price per symbol from executions.csv.
    Prefers avg_fill_price when available (new schema), falls back to limit_price."""
    path = Path("logs/executions.csv")
    if not path.exists():
        return {}
    try:
        df = pd.read_csv(path)
        buys = df[df["side"] == "BUY"].sort_values("timestamp")
        if "avg_fill_price" in buys.columns:
            # Use fill price when > 0, otherwise fall back to limit_price
            price_col = buys["avg_fill_price"].where(
                buys["avg_fill_price"] > 0, buys["limit_price"]
            )
            buys = buys.copy()
            buys["_price"] = price_col
            return buys.groupby("symbol")["_price"].last().to_dict()
        return buys.groupby("symbol")["limit_price"].last().to_dict()
    except Exception:
        return {}


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

    try:
        send_message("\n".join(lines)[:4096])
    except Exception as e:
        print(f"⚠️  Post-execution Telegram report failed: {e}")


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
            send_message(msg)

    try:
        if ibkr_ok:
            snap = fetch_account_snapshot(ib)
            print(f"✅ IBKR connected | NetLiq={snap.net_liquidation:.2f} | Cash={snap.cash:.2f}")
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
        if cb.is_triggered:
            msg = (
                f"🚨 Circuit breaker actif — drawdown={cb.drawdown:.1%} depuis pic "
                f"(${cb.peak_netliq:,.0f}). SELL-ONLY mode forcé."
            )
            print(msg)
            if not ci_mode:
                send_message(msg)

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
            EarningsSentimentAgent(),
        ])

        regime_data = detect_regime("SPY")
        regime = regime_data["regime"]
        print(
            f"\n🌍 RÉGIME DÉTECTÉ : {regime.upper()} | "
            f"SPY={regime_data['price']} | "
            f"SMA50={regime_data['sma50']} | "
            f"SMA200={regime_data['sma200']} | "
            f"Vol={regime_data['vol_regime']}"
        )

        plan_id = uuid.uuid4().hex[:8]
        plans = []

        # ====== ALLOCATION DYNAMIQUE ======
        # Télécharge toutes les données en une passe (réutilisées dans la boucle)
        all_data = {sym: download_ohlcv(sym) for sym in WATCHLIST}

        alloc_agents = [BuffettAgent(), CitadelAgent(), MeanReversionAgent(), TrendFollowingAgent()]
        alloc_result = DynamicAllocator(AllocatorConfig()).compute(all_data, alloc_agents)

        # ====== KELLY WEIGHTS (depuis live round-trips) ======
        _live_scorer = LiveScorer()
        kelly_weights: dict[str, float] = _live_scorer.compute_kelly_weights(min_trades=5)
        if kelly_weights:
            print(f"\n📐 Kelly demi-fraction : {kelly_weights}")

        print("\n" + alloc_result.telegram_summary())

        for sym in WATCHLIST:
            df = all_data[sym]

            # Priority dynamique (remplace AGENT_PRIORITY statique, fallback si symbole absent)
            priority = alloc_result.best_agent.get(sym) or AGENT_PRIORITY.get(sym)

            signals = arena.run(sym, df, regime=regime)
            winner = select_best(signals, priority_agent=priority)

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
                continue

            # Ajuste le target_weight : DynamicAllocator en priorité, puis Kelly si dispo
            dynamic_weight = alloc_result.weights.get(winner.agent_name, {}).get(sym)
            if dynamic_weight is not None:
                winner = dataclasses.replace(winner, target_weight=dynamic_weight)

            # Kelly blending : blend progressif Kelly ↔ dynamic au fur et à mesure des trades
            # α = min(1, n_trades/20) — 100% Kelly à partir de 20 round-trips par agent
            kelly_w = kelly_weights.get(winner.agent_name)
            if kelly_w is not None and kelly_w > 0:
                n_trips = _live_scorer.get_n_trades(winner.agent_name, sym)
                alpha = min(1.0, n_trips / 20)
                blended_w = round((1 - alpha) * winner.target_weight + alpha * kelly_w, 4)
                print(
                    f"   📐 Kelly {winner.agent_name}/{sym}: "
                    f"dynamic={winner.target_weight:.3f} kelly={kelly_w:.3f} "
                    f"α={alpha:.2f} → {blended_w:.3f} ({n_trips} trips)"
                )
                winner = dataclasses.replace(winner, target_weight=blended_w)

            last_px = get_last_close_1d(df)
            current_qty = snap.positions.get(sym, 0.0)

            plan = plan_from_signal(
                winner,
                net_liquidation=snap.net_liquidation,
                last_price=last_px,
                current_qty=current_qty,
            )
            plans.append(plan)

        # ====== STOP-LOSS PAR POSITION ======
        entry_prices = _load_entry_prices()
        for sym, qty in snap.positions.items():
            if qty <= 0 or sym not in all_data:
                continue
            entry_px = entry_prices.get(sym)
            if entry_px is None:
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
                    send_message(msg)

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
        risk_cfg = RiskConfig(
            max_net_long_pct=RISK_MAX_NET_LONG_PCT,
            max_single_position_pct=RISK_MAX_SINGLE_POSITION_PCT,
            min_cash_pct=RISK_MIN_CASH_PCT,
            sell_only_mode=RISK_SELL_ONLY_MODE or cb.is_triggered,
        )
        risk_report = RiskManager(risk_cfg).check(plans, snap)

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
                    send_message(msg)

        log_order_plan(plans, plan_id=plan_id)

        if not plans:
            msg = f"Milan Capital — ORDER PLAN ready\nplan_id={plan_id}\nNo plans."
            print(msg)
            if not ci_mode:
                send_message(msg)
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
            send_message(f"🧯 {reason} → NO orders sent. plan_id={plan_id}")
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
    