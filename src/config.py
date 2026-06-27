# src/config.py
from __future__ import annotations

import os
from dotenv import load_dotenv

load_dotenv()

# ====== WATCHLIST ======
WATCHLIST = [
    "AAPL", "SPY", "QQQ", "NVDA", "MSFT",
    "GOOGL", "META", "JPM", "GS", "GLD",
    "TSLA", "AMD", "AMZN", "LLY",
]

# ====== AGENT PRIORITY PAR SYMBOLE ======
# Backtest 3 ans — meilleur Sharpe par symbole (run_backtest.py)
# Note: DividendArbitrageAgent (poids initial 1.0) n'a pas d'entrée ici car
# il utilise un override absolu via meta["div_arb_priority"] dans selector.select_best()
# pendant sa fenêtre J-7→J+1 — aucun autre mécanisme de priorité n'est nécessaire.
AGENT_PRIORITY = {
    "AAPL": "BuffettAgent",
    "SPY": "BuffettAgent",
    "QQQ": "CitadelAgent",
    "NVDA": "BuffettAgent",
    "MSFT": "MeanReversionAgent",
    "GOOGL": "CitadelAgent",
    "META": "CitadelAgent",
    "JPM": "MeanReversionAgent",
    "GS": "BuffettAgent",
    "GLD": "BuffettAgent",
    "TSLA": "BuffettAgent",
    "AMD": "MeanReversionAgent",
    "AMZN": "TrendFollowingAgent",
    "LLY": "CitadelAgent",
}

# ====== EXECUTION ======
EXECUTION_ENABLED = os.getenv("EXECUTION_ENABLED", "false").lower() == "true"
MAX_ORDERS_PER_RUN = int(os.getenv("MAX_ORDERS_PER_RUN", "1"))
MAX_NOTIONAL_PCT = float(os.getenv("MAX_NOTIONAL_PCT", "0.02"))
LIMIT_BUFFER_BPS = int(os.getenv("LIMIT_BUFFER_BPS", "10"))

# ====== BROKER ======
IBKR_PORT = int(os.getenv("IBKR_PORT", "7497"))
IBKR_CLIENT_ID = int(os.getenv("IBKR_CLIENT_ID", "1"))

# ====== RISK ======
MAX_LEVERAGE = float(os.getenv("MAX_LEVERAGE", "1.0"))
MIN_SCORE_THRESHOLD = float(os.getenv("MIN_SCORE_THRESHOLD", "0.02"))

# ====== RISK MANAGER ======
RISK_MAX_NET_LONG_PCT = float(os.getenv("RISK_MAX_NET_LONG_PCT", "0.60"))
RISK_MAX_SINGLE_POSITION_PCT = float(os.getenv("RISK_MAX_SINGLE_POSITION_PCT", "0.20"))
RISK_MIN_CASH_PCT = float(os.getenv("RISK_MIN_CASH_PCT", "0.30"))
RISK_SELL_ONLY_MODE = os.getenv("RISK_SELL_ONLY_MODE", "false").lower() == "true"
STOP_LOSS_PCT = float(os.getenv("STOP_LOSS_PCT", "0.07"))  # 7% par position

# ====== TELEGRAM ======
TELEGRAM_APPROVAL_TIMEOUT = int(os.getenv("TELEGRAM_APPROVAL_TIMEOUT", "900"))