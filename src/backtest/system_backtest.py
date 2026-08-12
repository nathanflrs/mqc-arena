# src/backtest/system_backtest.py
"""
Milan Capital — Backtest du système réel.

Ce module rejoue **le pipeline de décision de production**, jour par jour, sur
données historiques :

    Arena (N agents) → normalisation → sélecteur + corroboration
    → allocateur → vol-sizing → planner → stop-loss → RiskManager
    → CorrelationGuard → garde d'exécution → remplissage avec coûts

À distinguer de `src/backtest/engine.py`, qui simule **un** agent sur **un**
symbole avec 95 % du capital, sans arène, sans risk manager et sans plafond
d'exécution. Les Sharpe walk-forward produits par ce dernier décrivent un
système qui n'a jamais été déployé.

────────────────────────────────────────────────────────────────────────────
LIMITES CONNUES — à lire avant de citer un chiffre produit par ce module
────────────────────────────────────────────────────────────────────────────

1. Couverture d'agents partielle. Six agents sur treize sont rejouables ;
   les autres dépendent de données non reconstituables point-in-time
   (voir EXCLUDED_AGENTS). Les agents rejouables représentent 78 % des
   victoires d'arène historiques (128 sur 164, logs/decisions.csv).

2. Look-ahead structurel, neutralisé par défaut mais mesurable :
     - `AGENT_PRIORITY` (config.py) a été calibré sur un backtest de la
       période entière. L'utiliser revient à savoir dès 2022 quel agent
       gagnera sur chaque symbole jusqu'en 2026.
     - `DynamicAllocator` lit en priorité `logs/walkforward_summary.csv`,
       calculé lui aussi sur toute la période.
   `use_agent_priority` et `allocator_mode` valent donc False/"off" par
   défaut. Lancer les deux modes et comparer : l'écart mesure exactement ce
   que la connaissance du futur apporte au système.

3. Les bornes du normaliseur (`logs/normalizer_stats.json`) ont été gelées
   sur une fenêtre de juillet 2026. Fuite considérée faible — ces bornes
   décrivent l'amplitude intrinsèque de la formule de confiance de chaque
   agent, pas la performance des symboles — mais elle n'est pas nulle.

4. Prix ajustés yfinance (`auto_adjust=True`) : l'historique est réajusté
   rétroactivement à chaque split ou dividende. Deux exécutions à deux dates
   ne sont pas strictement comparables. Un cache figé et daté reste à faire.

Aucune de ces limites n'est cachée dans le résultat : `SystemBacktestResult`
les transporte dans `caveats` et `render()` les imprime.
"""
from __future__ import annotations

import dataclasses
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence

import numpy as np
import pandas as pd

from src.agents.base import MarketState
from src.agents.buffett import BuffettAgent
from src.agents.citadel import CitadelAgent
from src.agents.cta_trend_agent import CTATrendAgent, CTA_UNIVERSE
from src.agents.dummy import DummyHoldAgent
from src.agents.mean_reversion import MeanReversionAgent
from src.agents.momentum import CrossSectionalMomentumAgent
from src.agents.trend_following import TrendFollowingAgent
from src.arena.arena import Arena
from src.arena.normalizer import ConfidenceNormalizer
from src.arena.selector import select_best
from src.data.regime import detect_regime
from src.execution.guards import build_execution_plan
from src.execution.planner import (
    OrderPlan, _compute_tx_cost, cta_plan_from_signal, plan_from_signal,
)
from src.risk.correlation import CorrelationGuard
from src.risk.manager import RiskConfig, RiskManager
from src.risk.vol_sizing import vol_adjusted_weight
from src.broker.portfolio import PortfolioSnapshot


# Agents écartés du replay, avec le motif. Toute réintégration exige de
# reconstituer la donnée en point-in-time, pas seulement de brancher l'API.
EXCLUDED_AGENTS: Dict[str, str] = {
    "MacroAgent":
        "FRED — les séries macro sont révisées après publication ; l'historique "
        "téléchargeable aujourd'hui n'est pas celui qu'on voyait à l'époque.",
    "EarningsSentimentAgent":
        "LLM sur des news datées — pas d'archive de news point-in-time, et "
        "rejouer un LLM sur des articles postérieurs est du look-ahead pur.",
    "InsiderBuyAgent":
        "SEC EDGAR en direct ; nécessiterait l'archive des Form 4 avec leur "
        "date de dépôt réelle (≠ date de transaction).",
    "DividendArbitrageAgent":
        "Calendrier d'ex-dates via yfinance — non reconstituable tel qu'il "
        "était connu à la date de décision.",
    "PairsTradingAgent":
        "Télécharge ses partenaires à la volée ; la cointégration devrait être "
        "réestimée sur fenêtre glissante à chaque date.",
    "VolatilityAgent":
        "Nécessite ^VIX ; signal par ailleurs classé dégénéré (100 % global, "
        "aucune discrimination entre symboles — audit P6).",
}


@dataclass
class SystemBacktestConfig:
    initial_capital: float = 1_000_000.0
    # Historique minimal avant la première décision. Doit couvrir l'agent le
    # plus exigeant — CrossSectionalMomentumAgent demande 280 séances (252 de
    # formation + 21 de skip + marge). En dessous, cet agent reste muet et le
    # backtest mesure un système amputé sans le dire.
    warmup_days: int = 300

    # ── Contrôles de look-ahead (voir docstring du module) ────────────────────
    use_agent_priority: bool = False
    allocator_mode: str = "off"     # "off" | "walkforward" (look-ahead assumé)

    # ── Exécution ─────────────────────────────────────────────────────────────
    max_notional_pct: float = 0.05
    max_orders_per_run: int = 5
    limit_buffer_bps: float = 5.0
    # "limit" : l'ordre ne se remplit que si l'ouverture de J+1 respecte la
    #           limite — un gap défavorable annule l'ordre, comme en réel.
    # "open"  : remplissage inconditionnel à l'ouverture de J+1 (optimiste).
    fill_model: str = "limit"

    # ── Risque ────────────────────────────────────────────────────────────────
    risk: RiskConfig = field(default_factory=lambda: RiskConfig(
        max_net_long_pct=0.60, max_single_position_pct=0.20, min_cash_pct=0.30,
    ))
    stop_loss_pct: float = 0.07
    correlation_threshold: float = 0.70

    # ── Sélecteur ─────────────────────────────────────────────────────────────
    qualified_voters: Optional[set] = None
    abstain_threshold: float = 0.25
    min_quorum: int = 2


@dataclass
class Fill:
    date: pd.Timestamp
    symbol: str
    side: str
    qty: int
    price: float
    cost_usd: float
    agent: str
    reason: str


@dataclass
class SystemBacktestResult:
    equity: pd.Series
    benchmark: pd.Series
    fills: List[Fill]
    daily: pd.DataFrame
    caveats: List[str]
    n_orders_sent: int
    n_orders_filled: int

    # ── Métriques ─────────────────────────────────────────────────────────────
    @property
    def total_return(self) -> float:
        return float(self.equity.iloc[-1] / self.equity.iloc[0] - 1.0)

    @property
    def benchmark_return(self) -> float:
        return float(self.benchmark.iloc[-1] / self.benchmark.iloc[0] - 1.0)

    @property
    def alpha(self) -> float:
        return self.total_return - self.benchmark_return

    @property
    def years(self) -> float:
        return len(self.equity) / 252.0

    @property
    def cagr(self) -> float:
        return float((1.0 + self.total_return) ** (1.0 / self.years) - 1.0) if self.years > 0 else 0.0

    @property
    def benchmark_cagr(self) -> float:
        return float((1.0 + self.benchmark_return) ** (1.0 / self.years) - 1.0) if self.years > 0 else 0.0

    def _sharpe(self, s: pd.Series, rf: float = 0.04) -> float:
        r = s.pct_change().dropna()
        sd = r.std()
        return float((r.mean() - rf / 252) / sd * np.sqrt(252)) if sd and sd > 0 else 0.0

    @property
    def sharpe(self) -> float:
        return self._sharpe(self.equity)

    @property
    def benchmark_sharpe(self) -> float:
        return self._sharpe(self.benchmark)

    @staticmethod
    def _mdd(s: pd.Series) -> float:
        return float((s / s.cummax() - 1.0).min())

    @property
    def max_drawdown(self) -> float:
        return self._mdd(self.equity)

    @property
    def benchmark_max_drawdown(self) -> float:
        return self._mdd(self.benchmark)

    @property
    def fill_rate(self) -> float:
        return self.n_orders_filled / self.n_orders_sent if self.n_orders_sent else 0.0

    @property
    def total_costs(self) -> float:
        return sum(f.cost_usd for f in self.fills)

    def render(self) -> str:
        pct = lambda x: f"{x:+.1%}"
        lines = [
            "═══ BACKTEST SYSTÈME — Milan Capital ═══",
            f"Période : {self.equity.index[0].date()} → {self.equity.index[-1].date()}"
            f"  ({self.years:.2f} ans, {len(self.equity)} séances)",
            "",
            f"{'':<22}{'Milan Capital':>16}{'SPY':>14}",
            f"{'Rendement total':<22}{pct(self.total_return):>16}{pct(self.benchmark_return):>14}",
            f"{'CAGR':<22}{pct(self.cagr):>16}{pct(self.benchmark_cagr):>14}",
            f"{'Sharpe':<22}{self.sharpe:>16.2f}{self.benchmark_sharpe:>14.2f}",
            f"{'Max drawdown':<22}{pct(self.max_drawdown):>16}{pct(self.benchmark_max_drawdown):>14}",
            "",
            f"Alpha vs SPY        : {pct(self.alpha)}",
            f"Ordres envoyés      : {self.n_orders_sent}"
            f"  |  remplis : {self.n_orders_filled} ({self.fill_rate:.0%})",
            f"Coûts de transaction: ${self.total_costs:,.0f}"
            f"  ({self.total_costs / self.equity.iloc[0] * 10_000:.0f} bps du capital initial)",
        ]
        if self.caveats:
            lines += ["", "⚠️  Réserves attachées à ce résultat :"]
            lines += [f"   • {c}" for c in self.caveats]
        return "\n".join(lines)


# ──────────────────────────────────────────────────────────────────────────────


def _replayable_agents(universe: Dict[str, pd.DataFrame]) -> List:
    """Agents ne dépendant que de l'OHLCV fourni — aucun appel réseau."""
    return [
        DummyHoldAgent(),
        BuffettAgent(),
        CitadelAgent(),
        MeanReversionAgent(),
        TrendFollowingAgent(),
        CrossSectionalMomentumAgent(universe=universe),
    ]


def _agent_priority_map(enabled: bool) -> Dict[str, str]:
    if not enabled:
        return {}
    from src.config import AGENT_PRIORITY
    return dict(AGENT_PRIORITY)


def run_system_backtest(
    data: Dict[str, pd.DataFrame],
    symbols: Sequence[str],
    benchmark_symbol: str = "SPY",
    cfg: Optional[SystemBacktestConfig] = None,
    cta_symbols: Optional[Sequence[str]] = None,
    verbose: bool = True,
) -> SystemBacktestResult:
    """
    Rejoue le pipeline de production jour par jour.

    Séquence stricte, sans look-ahead :
      1. Les ordres décidés en J-1 se remplissent à l'OUVERTURE de J.
      2. Le portefeuille est valorisé à la CLÔTURE de J.
      3. Les décisions de J n'utilisent que les données jusqu'à la clôture de J.
      4. Les ordres produits sont mis en file pour J+1.

    Un ordre ne peut donc jamais se remplir au prix qui l'a déclenché.
    """
    cfg = cfg or SystemBacktestConfig()
    cta_symbols = list(cta_symbols or [])

    dates = data[benchmark_symbol].index
    agents = _replayable_agents(data)
    momentum = next(a for a in agents if isinstance(a, CrossSectionalMomentumAgent))
    arena = Arena(agents)
    cta_agent = CTATrendAgent()

    normalizer = ConfidenceNormalizer.from_frozen_json()
    priority_map = _agent_priority_map(cfg.use_agent_priority)
    corr_guard = CorrelationGuard(threshold=cfg.correlation_threshold, lookback_days=60)
    risk_mgr = RiskManager(cfg.risk)

    cash = cfg.initial_capital
    positions: Dict[str, float] = {}
    entry_px: Dict[str, float] = {}
    pending: List = []                # candidats décidés en J-1
    fills: List[Fill] = []
    equity_rows, daily_rows = [], []
    n_sent = n_filled = 0

    all_syms = list(symbols) + cta_symbols
    start = max(cfg.warmup_days, 1)

    for i in range(start, len(dates)):
        today = dates[i]

        # ── 1. Remplissage des ordres de J-1, à l'ouverture de J ──────────────
        for cand in pending:
            sym = cand.plan.symbol
            df = data.get(sym)
            if df is None or today not in df.index:
                continue
            row = df.loc[today]
            open_px = float(row.get("Open", row["Close"]))
            if not np.isfinite(open_px) or open_px <= 0:
                open_px = float(row["Close"])

            if cfg.fill_model == "limit":
                # Un gap au-delà de la limite annule l'ordre (DAY order non exécuté).
                if cand.side == "BUY" and open_px > cand.limit_price:
                    continue
                if cand.side == "SELL" and open_px < cand.limit_price:
                    continue

            qty = cand.qty
            cost = _compute_tx_cost(qty, open_px)
            signed = qty if cand.side == "BUY" else -qty
            notional = qty * open_px

            if cand.side == "BUY" and notional + cost > cash:
                continue                      # pas de levier implicite

            cash -= signed * open_px + cost
            positions[sym] = positions.get(sym, 0.0) + signed
            if abs(positions[sym]) < 1e-9:
                positions.pop(sym, None)
                entry_px.pop(sym, None)
            elif cand.side == "BUY":
                entry_px.setdefault(sym, open_px)

            n_filled += 1
            fills.append(Fill(today, sym, cand.side, qty, open_px, cost,
                              cand.plan.reason[:60], cand.plan.reason))
        pending = []

        # ── 2. Valorisation à la clôture de J ─────────────────────────────────
        def close_of(sym: str) -> float:
            df = data.get(sym)
            if df is None or today not in df.index:
                return 0.0
            return float(df.loc[today, "Close"])

        prices = {s: close_of(s) for s in all_syms}
        nav = cash + sum(q * prices.get(s, 0.0) for s, q in positions.items())
        equity_rows.append((today, nav))

        if nav <= 0:
            break

        window = {s: data[s].iloc[: i + 1] for s in all_syms if s in data}
        snap = PortfolioSnapshot(net_liquidation=nav, cash=cash, positions=dict(positions))

        # ── 3. Régime, calculé sur l'historique disponible uniquement ─────────
        regime = detect_regime(df=window[benchmark_symbol])["regime"]

        # Les classements momentum sont figés à l'appel : sans ce recalcul sur
        # la fenêtre tronquée, l'agent verrait tout l'historique futur.
        momentum.set_universe({s: window[s] for s in symbols if s in window})

        plans: List[OrderPlan] = []

        # ── 4. Arène → sélection → sizing → plan, par symbole ─────────────────
        for sym in symbols:
            df_sym = window.get(sym)
            if df_sym is None or len(df_sym) < 200:
                continue

            signals = arena.run(sym, df_sym, portfolio=positions, regime=regime)
            winner_norm = select_best(
                normalizer.normalize_all(signals),
                priority_agent=priority_map.get(sym),
                qualified_voters=cfg.qualified_voters,
                abstain_threshold=cfg.abstain_threshold,
                min_quorum=cfg.min_quorum,
            )
            if winner_norm is None:
                continue
            winner = next(s for s in signals if s.agent_name == winner_norm.agent_name)

            if winner.action == "BUY":
                adj_w, _ = vol_adjusted_weight(df_sym, winner.target_weight)
                winner = dataclasses.replace(winner, target_weight=adj_w)

            plan = plan_from_signal(
                winner, net_liquidation=nav,
                last_price=prices[sym], current_qty=positions.get(sym, 0.0),
            )
            if plan is not None:
                plans.append(plan)

        # ── 5. Stop-loss — remplace tout plan existant sur le symbole ─────────
        for sym, qty in list(positions.items()):
            if qty <= 0 or sym not in entry_px or prices.get(sym, 0.0) <= 0:
                continue
            pnl = (prices[sym] - entry_px[sym]) / entry_px[sym]
            if pnl < -cfg.stop_loss_pct:
                plans = [p for p in plans if p.symbol != sym]
                plans.append(OrderPlan(
                    symbol=sym, action="SELL", target_weight=0.0,
                    last_price=prices[sym], current_qty=float(qty), target_qty=0.0,
                    delta_qty=-float(qty), est_notional=float(qty) * prices[sym],
                    reason=f"STOP-LOSS {pnl:.1%}", confidence=1.0,
                    est_cost_usd=_compute_tx_cost(qty, prices[sym]),
                ))

        # ── 6. Poche CTA ──────────────────────────────────────────────────────
        for sym in cta_symbols:
            df_cta = window.get(sym)
            if df_cta is None or len(df_cta) < 200 or prices.get(sym, 0.0) <= 0:
                continue
            sig = cta_agent.generate_signal(
                MarketState(symbol=sym, price=prices[sym], timestamp=str(today)),
                positions, regime=regime, data=df_cta,
            )
            plans.append(cta_plan_from_signal(
                sig, net_liquidation=nav, last_price=prices[sym],
                current_qty=float(positions.get(sym, 0.0)),
            ))

        if not plans:
            daily_rows.append((today, nav, regime, 0, 0, 0))
            continue

        # ── 7. Risk manager → corrélation → garde d'exécution ─────────────────
        report = risk_mgr.check(plans, snap, adv_map=None, price_map=prices)
        approved, _ = corr_guard.filter_plans(report.approved, snap, window)
        exec_plan = build_execution_plan(
            approved, net_liquidation=nav,
            max_notional_pct=cfg.max_notional_pct,
            max_orders=cfg.max_orders_per_run,
            limit_buffer_bps=cfg.limit_buffer_bps,
        )

        pending = exec_plan.candidates
        n_sent += len(pending)
        daily_rows.append((today, nav, regime, len(plans),
                           len(report.rejected), len(pending)))

        if verbose and i % 100 == 0:
            print(f"  {today.date()}  NAV=${nav:,.0f}  {regime:<7} "
                  f"pos={len(positions):2d}  ordres={len(pending)}")

    eq = pd.Series([v for _, v in equity_rows],
                   index=pd.DatetimeIndex([d for d, _ in equity_rows]), name="equity")

    bench_close = data[benchmark_symbol]["Close"].reindex(eq.index).ffill()
    bench = bench_close / bench_close.iloc[0] * cfg.initial_capital
    bench.name = "benchmark"

    caveats = [
        f"{len(EXCLUDED_AGENTS)} agents sur 13 exclus du replay "
        f"(données non reconstituables point-in-time) — les agents rejouables "
        f"représentent 78 % des victoires d'arène historiques.",
        "Prix ajustés yfinance : historique réajusté rétroactivement, résultat "
        "non strictement reproductible à une autre date.",
        "Bornes du normaliseur gelées sur une fenêtre de juillet 2026.",
    ]
    if cfg.use_agent_priority:
        caveats.append(
            "🔴 AGENT_PRIORITY ACTIF — calibré sur toute la période : le système "
            "connaît d'avance le meilleur agent par symbole. Résultat NON publiable."
        )
    if cfg.allocator_mode != "off":
        caveats.append(
            "🔴 Allocateur actif sur Sharpe walk-forward calculés sur toute la "
            "période — look-ahead. Résultat NON publiable."
        )
    if cfg.fill_model == "open":
        caveats.append(
            "Remplissage inconditionnel à l'ouverture : ignore les gaps "
            "défavorables qui annuleraient l'ordre à cours limité."
        )

    return SystemBacktestResult(
        equity=eq, benchmark=bench, fills=fills,
        daily=pd.DataFrame(daily_rows, columns=[
            "date", "nav", "regime", "n_plans", "n_rejected", "n_orders"]).set_index("date"),
        caveats=caveats, n_orders_sent=n_sent, n_orders_filled=n_filled,
    )
