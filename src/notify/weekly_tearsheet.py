"""
Weekly tearsheet runner — emits PnL attribution report to the event bus (dashboard).
Triggered every Monday at 07:15 UTC via GitHub Actions.
"""
from __future__ import annotations

from dotenv import load_dotenv

load_dotenv()

from src.risk.live_scorer import LiveScorer


def _emit_portfolio_performance(scorer: LiveScorer) -> None:
    perf = scorer.compute_portfolio_performance()
    if perf is None:
        return
    sign_p = "+" if perf.portfolio_return >= 0 else ""
    sign_s = "+" if perf.spy_return >= 0 else ""
    sign_a = "+" if perf.alpha >= 0 else ""
    body = (
        f"📈 Performance portefeuille (depuis le {perf.first_trade_date})\n"
        f"Portefeuille : {sign_p}{perf.portfolio_return:.1%}\n"
        f"SPY B&H      : {sign_s}{perf.spy_return:.1%}\n"
        f"Alpha        : {sign_a}{perf.alpha:.1%}\n"
        f"({perf.n_trades} round-trips)"
    )
    try:
        from src.events.bus import get_bus, Event
        get_bus().emit(Event(
            type="performance",
            severity="info",
            title="Performance portefeuille — Tearsheet",
            body=body,
            meta={"portfolio_return": perf.portfolio_return, "alpha": perf.alpha},
        ))
    except Exception:
        pass


def _emit_drift_alerts(scorer: LiveScorer) -> None:
    alerts = scorer.compute_drift_alerts()
    if not alerts:
        return
    lines = ["⚠️ Dérive Sharpe détectée (OOS vs Live) :"]
    for a in alerts:
        lines.append(
            f"  {a.agent.replace('Agent', '')}: "
            f"OOS={a.oos_sharpe:.2f} | Live={a.live_sharpe:.2f} | Écart={a.drift:.2f}"
        )
    body = "\n".join(lines)
    try:
        from src.events.bus import get_bus, Event
        get_bus().emit(Event(
            type="alert",
            severity="info",
            title="Dérive Sharpe détectée — Tearsheet",
            body=body,
            meta={"n_agents": len(alerts)},
        ))
    except Exception:
        pass


def _emit_dividend_arb_pnl() -> None:
    from src.agents.dividend_arbitrage_agent import DividendPositionTracker

    tracker = DividendPositionTracker()
    trades  = tracker.closed_trades()
    if not trades:
        return

    total_pnl = tracker.total_closed_pnl()
    n         = len(trades)
    winners   = sum(1 for t in trades if float(t.get("pnl", 0)) > 0)
    win_rate  = winners / n if n > 0 else 0.0
    sign      = "+" if total_pnl >= 0 else ""
    body = (
        f"📊 Dividend Arbitrage Performance\n"
        f"Trades: {n}  |  Win rate: {win_rate:.0%}\n"
        f"P&L total: {sign}${total_pnl:,.2f}"
    )
    try:
        from src.events.bus import get_bus, Event
        get_bus().emit(Event(
            type="performance",
            severity="info",
            title="Dividend Arbitrage — Tearsheet",
            body=body,
            meta={"total_pnl": total_pnl, "n_trades": n, "win_rate": win_rate},
        ))
    except Exception:
        pass


def _send_monte_carlo() -> None:
    try:
        from src.analytics.monte_carlo import run_simulation, MonteCarloReporter

        result = run_simulation(
            n_simulations=10_000,
            horizon_days=90,
            save_path="logs/monte_carlo_latest.json",
        )
        msg = MonteCarloReporter().format_tearsheet_section(result)

        # Emit to event bus (dashboard) — info only, not Telegram
        try:
            from src.events.bus import get_bus, Event
            get_bus().emit(Event(
                type="monte_carlo",
                severity="info",
                title="Monte Carlo Simulation — 90j N=10,000",
                body=msg,
                meta={
                    "var_95": result.var_95,
                    "prob_positive": result.prob_positive,
                    "median_return": result.median_return,
                },
            ))
        except Exception:
            pass
    except Exception as exc:
        import logging
        logging.getLogger(__name__).warning("Monte Carlo tearsheet échoué: %s", exc)


def _emit_tearsheet_event(scorer: "LiveScorer") -> None:
    """Emit a tearsheet info event to the dashboard (no Telegram)."""
    try:
        from src.events.bus import get_bus, Event
        from datetime import date
        get_bus().emit(Event(
            type="tearsheet",
            severity="info",
            title=f"Weekly Tearsheet — {date.today().isoformat()}",
            body="Tearsheet hebdomadaire généré. Voir l'onglet Performance pour les détails.",
            meta={},
        ))
    except Exception:
        pass


def run() -> None:
    scorer = LiveScorer()
    scorer.send_weekly_tearsheet()
    _emit_portfolio_performance(scorer)
    _emit_drift_alerts(scorer)
    _emit_dividend_arb_pnl()
    _send_monte_carlo()
    _emit_tearsheet_event(scorer)


if __name__ == "__main__":
    run()
