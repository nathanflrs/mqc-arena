"""
Weekly tearsheet runner — sends PnL attribution report to Telegram.
Triggered every Monday at 07:15 UTC via GitHub Actions.
"""
from __future__ import annotations

from dotenv import load_dotenv

load_dotenv()

from src.risk.live_scorer import LiveScorer


def _send_portfolio_performance(scorer: LiveScorer) -> None:
    from src.notify.telegram import send_message
    perf = scorer.compute_portfolio_performance()
    if perf is None:
        return
    sign_p = "+" if perf.portfolio_return >= 0 else ""
    sign_s = "+" if perf.spy_return >= 0 else ""
    sign_a = "+" if perf.alpha >= 0 else ""
    send_message(
        f"📈 Performance portefeuille (depuis le {perf.first_trade_date})\n"
        f"Portefeuille : {sign_p}{perf.portfolio_return:.1%}\n"
        f"SPY B&H      : {sign_s}{perf.spy_return:.1%}\n"
        f"Alpha        : {sign_a}{perf.alpha:.1%}\n"
        f"({perf.n_trades} round-trips)"
    )


def _send_drift_alerts(scorer: LiveScorer) -> None:
    from src.notify.telegram import send_message
    alerts = scorer.compute_drift_alerts()
    if not alerts:
        return
    lines = ["⚠️ Dérive Sharpe détectée (OOS vs Live) :"]
    for a in alerts:
        lines.append(
            f"  {a.agent.replace('Agent', '')}: "
            f"OOS={a.oos_sharpe:.2f} | Live={a.live_sharpe:.2f} | Écart={a.drift:.2f}"
        )
    send_message("\n".join(lines))


def run() -> None:
    scorer = LiveScorer()
    scorer.send_weekly_tearsheet()
    _send_portfolio_performance(scorer)
    _send_drift_alerts(scorer)


if __name__ == "__main__":
    run()
