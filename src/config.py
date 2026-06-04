# src/config.py
from __future__ import annotations

import os
from dotenv import load_dotenv

load_dotenv()

# ====== WATCHLIST ======
WATCHLIST = [
    "AAPL", "SPY", "QQQ", "NVDA", "MSFT",
    "GOOGL", "META", "JPM", "GS", "GLD",
    "BRK-B", "JNJ", "TSLA", "AMD",
]

# ====== AGENT PRIORITY PAR SYMBOLE ======
# Backtest 3 ans — meilleur Sharpe par symbole (run_backtest.py)
AGENT_PRIORITY = {
    "AAPL":  "MeanReversionAgent",   # Sharpe=0.81, Return=+38%
    "SPY":   "BuffettAgent",          # Sharpe=0.71, Return=+32%
    "QQQ":   "BuffettAgent",          # Sharpe=0.79, Return=+38%
    "NVDA":  "MeanReversionAgent",   # Sharpe=1.28, Return=+98%
    "MSFT":  "MeanReversionAgent",   # Sharpe=0.22, Return=+14%
    "GOOGL": "TrendFollowingAgent",  # Sharpe=1.36, Return=+98%
    "META":  "MeanReversionAgent",   # Sharpe=0.61, Return=+35%
    "JPM":   "MeanReversionAgent",   # Sharpe=1.09, Return=+33%
    "GS":    "BuffettAgent",          # Sharpe=1.62, Return=+141%
    "GLD":   "BuffettAgent",          # Sharpe=1.20, Return=+73%
    "TLT":   "CitadelAgent",          # Sharpe=-0.64 (meilleur disponible — tous négatifs)
    "BRK-B": "MeanReversionAgent",   # Sharpe=0.38, Return=+13%
    "JNJ":   "TrendFollowingAgent",  # Sharpe=1.56, Return=+45%
    "TSLA":  "TrendFollowingAgent",  # Sharpe=0.49, Return=+37%
    "AMD":   "CitadelAgent",          # Sharpe=1.31, Return=+163%
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

# ====== TELEGRAM ======
TELEGRAM_APPROVAL_TIMEOUT = int(os.getenv("TELEGRAM_APPROVAL_TIMEOUT", "900"))