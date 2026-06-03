# src/config.py
from __future__ import annotations

import os
from dotenv import load_dotenv

load_dotenv()

# ====== WATCHLIST ======
# MSFT retiré — backtest montre pertes sur tous les agents
WATCHLIST = ["AAPL", "SPY"]

# ====== AGENT PRIORITY PAR SYMBOLE ======
# Basé sur les résultats du backtest (meilleur Sharpe)
AGENT_PRIORITY = {
    "AAPL": "MeanReversionAgent",   # Sharpe=0.81, Return=+38%
    "SPY":  "BuffettAgent",          # Sharpe=0.68, Return=+31%
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
RISK_MAX_NET_LONG_PCT = float(os.getenv("RISK_MAX_NET_LONG_PCT", "0.40"))
RISK_MAX_SINGLE_POSITION_PCT = float(os.getenv("RISK_MAX_SINGLE_POSITION_PCT", "0.20"))
RISK_MIN_CASH_PCT = float(os.getenv("RISK_MIN_CASH_PCT", "0.30"))
RISK_SELL_ONLY_MODE = os.getenv("RISK_SELL_ONLY_MODE", "false").lower() == "true"

# ====== TELEGRAM ======
TELEGRAM_APPROVAL_TIMEOUT = int(os.getenv("TELEGRAM_APPROVAL_TIMEOUT", "900"))