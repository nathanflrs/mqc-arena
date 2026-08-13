# src/agents/registry.py
"""
Single source of truth for all live trading agents.
Update REAL_AGENTS here when adding or retiring a strategy.
- DummyHoldAgent is excluded (benchmark baseline, not a strategy).
- CrossSectionalMomentumAgent retiré le 2026-08-13 (edge négatif prouvé,
  voir docs/verdicts_agents.md).
- CTATrendAgent retiré le 2026-08-13 : son univers est composé à 100 %
  d'ETF américains, que la réglementation européenne PRIIPs interdit à un
  particulier résidant dans l'UE. Voir docs/verdicts_agents.md.
Consumed by /api/public-meta — never used in trading logic.
"""
from src.agents.buffett import BuffettAgent
from src.agents.citadel import CitadelAgent
from src.agents.mean_reversion import MeanReversionAgent
from src.agents.macro import MacroAgent
from src.agents.trend_following import TrendFollowingAgent
from src.agents.dividend_arbitrage_agent import DividendArbitrageAgent
from src.agents.pairs_trading import PairsTradingAgent
from src.agents.volatility import VolatilityAgent
from src.agents.earnings_sentiment import EarningsSentimentAgent
from src.agents.insider_buy import InsiderBuyAgent

REAL_AGENTS: tuple = (
    BuffettAgent,
    CitadelAgent,
    MeanReversionAgent,
    MacroAgent,
    TrendFollowingAgent,
    DividendArbitrageAgent,
    PairsTradingAgent,
    VolatilityAgent,
    EarningsSentimentAgent,
    InsiderBuyAgent,
)
