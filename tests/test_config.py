# tests/test_config.py
from __future__ import annotations

import pytest

from src.config import WATCHLIST, AGENT_PRIORITY

# Univers TRADABLE. Les ETF américains (SPY, QQQ, GLD, TLT, UUP, DBC) en ont
# été retirés le 2026-08-13 : IBKR les refuse à un particulier résidant dans
# l'UE, faute de document d'information réglementaire (PRIIPs/KID). SPY et GLD
# restent téléchargés comme données — voir DATA_ONLY — mais ne sont plus
# achetables.
ACTIVE_TICKERS = {
    "AAPL", "NVDA", "MSFT", "GOOGL", "META",
    "JPM", "GS", "TSLA", "AMD", "AMZN", "LLY",
}
# Retirés de l'univers tradable, pour des raisons différentes :
#   TLT                    — Sharpe négatif sur tous les agents
#   SPY, QQQ, GLD          — ETF, interdits à l'achat depuis l'UE
ARCHIVED_TICKERS = {"TLT", "SPY", "QQQ", "GLD"}

VALID_AGENTS = {
    "BuffettAgent",
    "CitadelAgent",
    "MeanReversionAgent",
    "TrendFollowingAgent",
    "MacroAgent",
    "VolatilityAgent",
    "DividendArbitrageAgent",
    "PairsTradingAgent",
    "DummyHoldAgent",
    "InsiderBuyAgent",
    "CrossSectionalMomentumAgent",
}


def test_watchlist_contains_exactly_active_tickers():
    assert set(WATCHLIST) == ACTIVE_TICKERS


def test_watchlist_excludes_archived_tickers():
    assert not (set(WATCHLIST) & ARCHIVED_TICKERS), "TLT ne doit pas être dans WATCHLIST"


def test_watchlist_has_no_duplicates():
    assert len(WATCHLIST) == len(set(WATCHLIST))


def test_agent_priority_covers_full_watchlist():
    missing = set(WATCHLIST) - set(AGENT_PRIORITY.keys())
    assert not missing, f"Tickers sans agent assigné : {missing}"


def test_agent_priority_may_contain_archived_tickers():
    # TLT peut rester dans AGENT_PRIORITY pour référence sans être dans WATCHLIST
    for ticker in ARCHIVED_TICKERS:
        if ticker in AGENT_PRIORITY:
            assert AGENT_PRIORITY[ticker] in VALID_AGENTS


def test_agent_priority_values_are_valid_agents():
    invalid = {
        sym: agent
        for sym, agent in AGENT_PRIORITY.items()
        if agent not in VALID_AGENTS
    }
    assert not invalid, f"Agents inconnus dans AGENT_PRIORITY : {invalid}"


@pytest.mark.parametrize("ticker", sorted(ACTIVE_TICKERS))
def test_each_active_ticker_has_priority_agent(ticker):
    assert ticker in AGENT_PRIORITY, f"{ticker} manquant dans AGENT_PRIORITY"
    assert isinstance(AGENT_PRIORITY[ticker], str)
    assert len(AGENT_PRIORITY[ticker]) > 0


def test_env_example_mirrors_config_defaults():
    """
    Le gabarit publié ne doit jamais contredire les défauts du code.

    Écrit le 2026-08-15 après incident : le `.env.example` ajouté au dépôt
    public annonçait un stop loss à 15 % contre 7 % dans `src/config.py`, un
    plancher de cash à 20 % contre 30 %, et une taille d'ordre cinq fois trop
    grande. Un gabarit qui desserre silencieusement une limite de risque est
    pire que pas de gabarit du tout : il est copié tel quel, et personne ne
    relit un fichier d'exemple.
    """
    import re
    from pathlib import Path
    import src.config as cfg

    tpl = dict(re.findall(r"^([A-Z_]+)=(.*)$",
                          Path(".env.example").read_text(), re.M))
    numeriques = [
        "RISK_MAX_NET_LONG_PCT", "RISK_MAX_SINGLE_POSITION_PCT",
        "RISK_MIN_CASH_PCT", "STOP_LOSS_PCT", "MAX_LEVERAGE",
        "MAX_ORDERS_PER_RUN", "MAX_NOTIONAL_PCT", "LIMIT_BUFFER_BPS",
    ]
    ecarts = []
    for k in numeriques:
        code = getattr(cfg, k, None)
        if code is None or k not in tpl or tpl[k].strip() == "":
            continue
        if abs(float(tpl[k]) - float(code)) > 1e-9:
            ecarts.append(f"{k} : gabarit {tpl[k]} ≠ code {code}")
    assert not ecarts, "\n".join(ecarts)
