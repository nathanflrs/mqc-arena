"""
Weekly tearsheet runner — sends PnL attribution report to Telegram.
Triggered every Monday at 07:15 UTC via GitHub Actions.
"""
from __future__ import annotations

from dotenv import load_dotenv

load_dotenv()

from src.risk.live_scorer import LiveScorer


def run() -> None:
    LiveScorer().send_weekly_tearsheet()


if __name__ == "__main__":
    run()
