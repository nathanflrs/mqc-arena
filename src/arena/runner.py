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
    TELEGRAM_APPROVAL_TIMEOUT,
    RISK_MAX_NET_LONG_PCT,
    RISK_MAX_SINGLE_POSITION_PCT,
    RISK_MIN_CASH_PCT,
    RISK_SELL_ONLY_MODE,
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
from src.broker.ibkr import connect_ibkr
from src.broker.portfolio import fetch_account_snapshot
from src.data.market_data import download_ohlcv, get_last_close_1d
from src.data.regime import detect_regime
from src.execution.planner import plan_from_signal
from src.execution.logger import log_order_plan, log_execution, log_decisions
from src.notify.telegram import drain_updates, send_message, wait_for_approval
from src.risk.manager import RiskConfig, RiskManager
from src.risk.allocator import AllocatorConfig, DynamicAllocator


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
        "✅ APPROVED — sending PAPER orders (LIMIT)\n"
        f"plan_id={plan_id}\n"
        f"max_orders={MAX_ORDERS_PER_RUN} | max_notional/order={max_notional:.0f} | buffer={LIMIT_BUFFER_BPS}bps"
    )

    ts = datetime.now(timezone.utc).replace(microsecond=0).isoformat()
    Path("logs").mkdir(parents=True, exist_ok=True)

    exec_rows = []
    for p, side, qty, limit_price in candidates:
        contract = Stock(p.symbol, "SMART", "USD")
        order = LimitOrder(side, qty, round(limit_price, 2))
        trade = ib.placeOrder(contract, order)

        status = trade.orderStatus.status
        msg = f"📤 PAPER {p.symbol}: {side} {qty} @ LMT {order.lmtPrice} | status={status}"
        print(msg)
        send_message(msg)

        exec_rows.append({
            "plan_id": plan_id,
            "timestamp": ts,
            "symbol": p.symbol,
            "side": side,
            "qty": qty,
            "limit_price": float(order.lmtPrice),
            "last_price": float(p.last_price),
            "est_notional": float(p.est_notional),
            "target_weight": float(p.target_weight),
            "reason": str(p.reason),
            "status": status,
        })

    log_execution(exec_rows)
    send_message("✅ Execution run complete. Logged to logs/executions.csv")


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

            # Ajuste le target_weight du winner selon le Sharpe rolling de l'agent
            dynamic_weight = alloc_result.weights.get(winner.agent_name, {}).get(sym)
            if dynamic_weight is not None:
                winner = dataclasses.replace(winner, target_weight=dynamic_weight)

            last_px = get_last_close_1d(df)
            current_qty = snap.positions.get(sym, 0.0)

            plan = plan_from_signal(
                winner,
                net_liquidation=snap.net_liquidation,
                last_price=last_px,
                current_qty=current_qty,
            )
            plans.append(plan)

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
            sell_only_mode=RISK_SELL_ONLY_MODE,
        )
        risk_report = RiskManager(risk_cfg).check(plans, snap)

        if risk_report.rejected:
            print(f"⚠️  Risk manager: {len(risk_report.rejected)} plan(s) rejeté(s)")
            for r in risk_report.rejected:
                print(f"   ✂️  {r.plan.symbol} ({r.plan.action}): {r.reason}")

        plans = risk_report.approved

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
            print("ℹ️  CI mode — order plan logged to logs/order_plan.csv. No Telegram, no approval, no execution.")
            return

        send_message(plan_summary + "\n\nReply APPROVE / REJECT (15 min)")

        last_id = drain_updates()
        approved, _ = wait_for_approval(
            plan_id=plan_id,
            timeout_seconds=TELEGRAM_APPROVAL_TIMEOUT,
            last_update_id=last_id,
        )

        if not approved:
            send_message(f"❌ Not approved (or timeout). plan_id={plan_id}. No execution.")
            print("❌ Not approved (or timeout). Exiting safely.")
            return

        print("✅ Approved.")

        execution_enabled = EXECUTION_ENABLED and ibkr_ok
        if not execution_enabled:
            reason = "IBKR unavailable" if not ibkr_ok else "EXECUTION_ENABLED=false"
            send_message(f"🧯 Approved but {reason} → NO orders sent. plan_id={plan_id}")
            print(f"🧯 {reason} → NO orders sent.")
            return

        execute_plans_paper_ibkr(ib, snap, plans, plan_id)

    finally:
        if ib is not None:
            ib.disconnect()
        print("\n✅ Done.")


if __name__ == "__main__":
    main()
    