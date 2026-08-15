# Milan Capital — MQC Arena

An autonomous multi-agent equity trading system, and — more importantly — the
measurement apparatus built to find out whether any of it actually works.

**It runs unattended on a dedicated server every trading day.** It connects to
Interactive Brokers, polls ten strategy agents, filters through risk layers,
places limit orders, reconciles fills against expected positions, measures its
own slippage, and pushes a notification. Nobody presses a button.

**No strategy in it has demonstrated an edge.** Five hypotheses were
pre-registered and tested; five were rejected. That result is the point of this
repository, not a footnote to it.

> Paper trading only. This is a research system, not an investment product, and
> nothing here is financial advice.

---

## What was actually found

| Hypothesis | Pre-registered | Result |
|---|---|---|
| Cross-sectional momentum (Jegadeesh-Titman), long-only | — | No measurable edge; retired |
| Cross-sectional momentum, both legs, S&P 500 | ✅ | +0.42 % / +0.53 % over 20 days, CI crosses zero on both epochs |
| Mean reversion out-of-sample (2010-2019) | ✅ | Does not reproduce |
| Market-regime conditioning | ✅ | Refuted — the market changed, not the strategy mix |
| Accruals anomaly (Sloan 1996), point-in-time | ✅ | Rejected on its own stated criterion |

Every verdict is dated and written up in [`docs/verdicts_agents.md`](docs/verdicts_agents.md),
including the reasoning that turned out to be wrong.

### Two published results were retracted

On 2026-08-14 the confidence intervals in this repository were found to be **too
narrow**, and two "significant" findings did not survive correction.

A 20-day forward return measured every session shares 19 days of market history
with the previous one. Resampling those sessions independently counts the same
information twenty times and divides the interval by the square root of a
fictitious sample size. On the long/short momentum test, 1,364 "observations"
were worth 69 independent ones.

The fix is a **circular block bootstrap** ([`src/analysis/agent_edge.py`](src/analysis/agent_edge.py)):
contiguous blocks the length of the return window, with the series closed into a
ring so that every observation carries equal weight. The first attempt at the
fix was itself defective — it under-sampled the edges of each series by a factor
of 23 — which showed up as a sample mean falling *outside* its own confidence
interval.

Both the original claims and the corrections are in the repository. Nothing was
quietly edited.

---

## How the system is built

```
  market data          agents                arbitration          risk              execution
 ┌────────────┐   ┌──────────────────┐   ┌───────────────┐   ┌────────────┐   ┌──────────────┐
 │ yfinance   │──▶│ 10 strategies    │──▶│ selector      │──▶│ position   │──▶│ IBKR limit   │
 │ SEC EDGAR  │   │ + 1 benchmark    │   │ (production)  │   │ sizing     │   │ orders       │
 │ Finnhub    │   │ each proposes    │   │ consensus     │   │ drawdown   │   │ fill polling │
 │ IBKR       │   │ BUY/SELL/HOLD    │   │ (observation) │   │ VaR, corr. │   │ reconcile    │
 └────────────┘   └──────────────────┘   └───────────────┘   └────────────┘   └──────────────┘
```

**Ten agents, ten stated philosophies:** quality screening (Buffett), short-term
multifactor (Citadel), RSI/Bollinger mean reversion, moving-average + ADX trend
following, regime-aware macro, dividend ex-date arbitrage, cointegration pairs,
VIX-extreme volatility, SEC Form 4 insider buying, and news sentiment. An
eleventh agent buys and holds — the benchmark every other agent has to beat.

**The arbitration is being replaced, and here is why.** The production selector
ranks signals on `confidence × target_weight` — two constants hand-written in
each agent's source file. Measured calibration shows that number predicts
nothing (flat for Buffett, *inverted* for trend following). The only agent whose
confidence is informative is the one that almost never speaks and never wins.
[`src/arena/consensus.py`](src/arena/consensus.py) aggregates instead: equal
weights until an edge is demonstrated, HOLD counts as abstention, and position
size is damped by √(participation). It runs in observation alongside the old
mechanism and executes nothing.

**What the replacement revealed matters more than the replacement.** Replayed
over 13 live sessions, it produces 89 % identical decisions. Agents that speak
agree 96 % of the time, and 98.4 % of all action proposals are buys. The arena
never had anything to arbitrate: ten philosophies on eleven megacaps in a bull
market are one opinion repeated ten times.

---

## Methodology

Three corrections separate this from a naive backtest, each made after finding
the previous approach wrong.

**The null hypothesis is the unconditional base rate, not a coin flip.** In a
rising market, an agent that always says BUY scores well above 50 % while
carrying no information at all. Every "excess" reported here is measured against
the same action's base rate on the same universe over the same period.

**Confidence intervals cluster on dates, not signals.** A session where one
agent says BUY on twelve names is one market observation, not twelve.

**Overlapping windows are handled by block bootstrap.** See above.

**Point-in-time data throughout.** S&P 500 membership is reconstructed from
Wikipedia revision history ([`src/data/universe.py`](src/data/universe.py)) so
that survivorship bias cannot enter. SEC fundamentals are dated by **filing
date**, not fiscal period end ([`src/data/sec_fundamentals.py`](src/data/sec_fundamentals.py)) —
using the first filing rather than the last, because companies restate prior
years as comparatives in every subsequent report.

**Hypotheses are pre-registered.** [`docs/hypothese_01_accruals.md`](docs/hypothese_01_accruals.md)
was written on 2026-08-12, before any measurement, with its rejection criterion
stated in advance. It was rejected on that criterion.

---

## Things a backtest cannot teach you

The system was rejected by Interactive Brokers on its first live order:

```
Error 201: Order rejected — Customer Ineligible
This product does not have a KID in a language approved for your country
```

EU **PRIIPs** regulation: a retail investor resident in the EU cannot buy a US
ETF, because the issuer publishes no Key Information Document in the local
language. The trend-following agent's universe was 100 % US ETFs. It was retired
and the tradable universe was restructured into `WATCHLIST` (what can be bought)
and `DATA_ONLY` (what is read but never held).

No amount of historical simulation would have surfaced this. It took a real
order to a real broker.

---

## Repository layout

```
src/
  agents/       10 live strategies + 1 benchmark, plus retired agents kept
                for historical replay (removing them would erase the record
                of what was tried)
  arena/        signal arbitration — selector (live), consensus (observation)
  analysis/     edge measurement, bootstrap, calibration curves
  data/         point-in-time universe, SEC fundamentals, price quality filter
  regime/       GMM market-regime detection
  risk/         position sizing, allocator, VaR, correlation, circuit breaker
  execution/    order planning, guards, IBKR fills, reconciliation, run lock
  broker/       Interactive Brokers connection and portfolio snapshot
  backtest/     full-system replay and walk-forward validation
  dashboard/    authenticated web dashboard (FastAPI)
  notify/       Web Push notifications
docs/           methodology, dated verdicts, pre-registered hypotheses
scripts/        research: edge measurement, hypothesis tests, replays
tests/          995 tests
deploy/         server provisioning, IB Gateway, systemd units
```

## Running it

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env          # then fill in your own credentials
python -m pytest tests/ -q
```

Research scripts are standalone and never place orders:

```bash
python -m scripts.measure_agent_edge          # regenerates docs/agent_edge.md
python -m scripts.test_hypothese_01_accruals  # replays the accruals hypothesis
python -m scripts.replay_consensus            # old vs new arbitration
```

Live execution requires **two independent locks** to be open — `ReadOnlyApi=no`
on the IB Gateway and `EXECUTION_ENABLED=true` in the environment. Closing
either one is sufficient to stop all orders.

## Stack

Python 3.12 · pandas · numpy · statsmodels · scikit-learn · yfinance ·
ib_insync · FastAPI · pywebpush · pytest · systemd

---

## What this repository does not claim

- **No demonstrated edge.** Zero surviving positive results.
- **13 live sessions** on a paper account. There is no track record.
- **Six of ten agents have never been measured** — their inputs (news, FRED
  revisions, insider filings) cannot be reconstructed point-in-time, so no
  historical replay is possible. They still make live decisions.
- **Agent thresholds were tuned on the same data used to judge them.** Any edge
  measured in this repository is an upper bound, not a forecast.
- **Portfolio construction ignores correlation.** Five megacaps bought on the
  same day are one bet placed five times, and nothing in the system notices.
- **Large parts of the code were written with AI assistance.** The research
  direction, the decisions, and the verdicts are mine.

Internal documentation in `docs/` is in French.
