"""
Milan Capital — Fama-French / Carhart Factor Analysis Engine

Answers one question per agent: genuine alpha or disguised beta?

    from src.analytics.factor_analysis import (
        FactorDataLoader, AgentReturnSeriesBuilder,
        FactorRegression, FactorReporter,
    )

    loader   = FactorDataLoader(cache_dir="logs/factor_cache")
    factors  = loader.load_factors(model="carhart")
    builder  = AgentReturnSeriesBuilder()
    series   = builder.build_all_agents()
    reg      = FactorRegression(model="carhart")
    results  = reg.run_all(series, factors)
    reporter = FactorReporter()
    print(reporter.format_console_table(results))
"""
from __future__ import annotations

import logging
import math
import warnings
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import Dict, Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# ── Constants ─────────────────────────────────────────────────────────────────

TRADING_DAYS_PER_YEAR = 252
MIN_OBS_DEFAULT       = 60

# Kenneth French dataset identifiers
_FF3_DAILY    = "F-F_Research_Data_Factors_Daily"
_MOM_DAILY    = "F-F_Momentum_Factor_Daily"
_FF5_DAILY    = "F-F_Research_Data_5_Factors_2x3_Daily"

_CARHART_FACTORS = ["Mkt-RF", "SMB", "HML", "Mom"]
_FF5_FACTORS     = ["Mkt-RF", "SMB", "HML", "RMW", "CMA"]

_LIVE_MIN_DAYS = 60  # minimum live days before auto switches to walkforward


# ─────────────────────────────────────────────────────────────────────────────
# Dataclass
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class RegressionResult:
    agent_name:          str
    model:               str          # 'carhart' | 'ff5'
    alpha_daily:         float
    alpha_annualized:    float
    alpha_tstat:         float
    alpha_pvalue:        float
    alpha_significant:   bool         # |t| > 1.96
    betas:               Dict[str, float]
    beta_tstats:         Dict[str, float]
    beta_pvalues:        Dict[str, float]
    r_squared:           float
    adj_r_squared:       float
    n_observations:      int
    residual_vol_annual: float        # idiosyncratic vol annualised
    information_ratio:   float        # alpha_annual / residual_vol_annual
    source:              str          # 'live' | 'walkforward'
    insufficient_data:   bool
    interpretation:      str
    timestamp:           str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )

    def to_dict(self) -> dict:
        d = asdict(self)
        return d


# ─────────────────────────────────────────────────────────────────────────────
# Part 1 — FactorDataLoader
# ─────────────────────────────────────────────────────────────────────────────

class FactorDataLoader:
    """
    Downloads Fama-French daily factors from the Kenneth French Data Library.
    Caches locally as parquet; re-downloads only when cache is older than 24h.
    """

    CACHE_TTL_HOURS = 24

    def __init__(self, cache_dir: str = "logs/factor_cache") -> None:
        self._cache_dir = Path(cache_dir)
        self._cache_dir.mkdir(parents=True, exist_ok=True)

    # ── Public ────────────────────────────────────────────────────────────────

    def load_factors(self, model: str = "carhart") -> pd.DataFrame:
        """
        Return a daily DataFrame indexed by date with columns:
          Carhart : Mkt-RF, SMB, HML, Mom, RF
          FF5     : Mkt-RF, SMB, HML, RMW, CMA, RF
        All values are decimal returns (NOT percent).
        """
        model = model.lower()
        if model not in {"carhart", "ff5"}:
            raise ValueError(f"model must be 'carhart' or 'ff5', got {model!r}")

        cache_file = self._cache_dir / f"factors_{model}.parquet"

        if self._cache_valid(cache_file):
            logger.info("FactorDataLoader: using cached %s factors", model)
            return pd.read_parquet(cache_file)

        try:
            df = self._download(model)
            df.to_parquet(cache_file)
            logger.info("FactorDataLoader: downloaded and cached %s factors (%d rows)", model, len(df))
            return df
        except Exception as exc:
            if cache_file.exists():
                warnings.warn(
                    f"Factor download failed ({exc}); using stale cache from {cache_file}",
                    RuntimeWarning, stacklevel=2,
                )
                return pd.read_parquet(cache_file)
            raise RuntimeError(
                f"Cannot download factors and no cache exists at {cache_file}. "
                f"Original error: {exc}"
            ) from exc

    # ── Private ───────────────────────────────────────────────────────────────

    def _cache_valid(self, path: Path) -> bool:
        if not path.exists():
            return False
        age_hours = (datetime.now().timestamp() - path.stat().st_mtime) / 3600
        return age_hours < self.CACHE_TTL_HOURS

    def _download(self, model: str) -> pd.DataFrame:
        ff3  = self._download_ff_factors()
        if model == "carhart":
            mom = self._download_momentum()
            df  = ff3.join(mom, how="inner")
            return df[["Mkt-RF", "SMB", "HML", "Mom", "RF"]]
        else:  # ff5
            ff5 = self._download_ff5()
            return ff5[["Mkt-RF", "SMB", "HML", "RMW", "CMA", "RF"]]

    @staticmethod
    def _to_datetime_index(df: pd.DataFrame) -> pd.DataFrame:
        """Convert PeriodIndex or integer index to DatetimeIndex."""
        idx = df.index
        if hasattr(idx, "to_timestamp"):
            df.index = idx.to_timestamp()
        else:
            df.index = pd.to_datetime(idx, errors="coerce")
        df.index.name = "date"
        df.index = df.index.normalize()
        return df

    def _download_ff_factors(self) -> pd.DataFrame:
        from pandas_datareader.famafrench import FamaFrenchReader
        reader = FamaFrenchReader(_FF3_DAILY, start="2018-01-01")
        raw    = reader.read()
        df     = raw[0].copy()
        df     = self._to_datetime_index(df)
        df     = df / 100.0  # percent → decimal
        return df.dropna()

    def _download_momentum(self) -> pd.DataFrame:
        from pandas_datareader.famafrench import FamaFrenchReader
        reader = FamaFrenchReader(_MOM_DAILY, start="2018-01-01")
        raw    = reader.read()
        df     = raw[0].copy()
        # Column might be "Mom" or "WML"
        col    = df.columns[0]
        df     = df.rename(columns={col: "Mom"})
        df     = self._to_datetime_index(df)
        df     = df / 100.0
        return df[["Mom"]].dropna()

    def _download_ff5(self) -> pd.DataFrame:
        from pandas_datareader.famafrench import FamaFrenchReader
        reader = FamaFrenchReader(_FF5_DAILY, start="2018-01-01")
        raw    = reader.read()
        df     = raw[0].copy()
        df     = self._to_datetime_index(df)
        df     = df / 100.0
        return df.dropna()


# ─────────────────────────────────────────────────────────────────────────────
# Part 2 — AgentReturnSeriesBuilder
# ─────────────────────────────────────────────────────────────────────────────

class AgentReturnSeriesBuilder:
    """
    Builds daily return series per agent from:
      - walkforward OOS returns (preferred, more observations)
      - live executions (round-trips from decisions.csv + executions.csv)

    Days without a position count as 0.
    """

    def __init__(
        self,
        decisions_path:   str = "logs/decisions.csv",
        executions_path:  str = "logs/executions.csv",
        walkforward_path: str = "logs/walkforward_results.csv",
    ) -> None:
        self._dec_path  = Path(decisions_path)
        self._exc_path  = Path(executions_path)
        self._wf_path   = Path(walkforward_path)

    # ── Public ────────────────────────────────────────────────────────────────

    def build_daily_returns(
        self, agent_name: str, source: str = "auto"
    ) -> pd.Series:
        """
        Returns a pd.Series of daily returns indexed by date (tz-naive),
        with zeros on non-trading/flat days.
        source : 'auto' | 'walkforward' | 'live'
        """
        if source == "walkforward":
            return self._from_walkforward(agent_name)
        if source == "live":
            return self._from_live_executions(agent_name)

        # auto: try live first
        live = self._from_live_executions(agent_name)
        n_live = (live != 0).sum()
        if n_live >= _LIVE_MIN_DAYS:
            return live
        warnings.warn(
            f"{agent_name}: only {n_live} live observation days (< {_LIVE_MIN_DAYS}); "
            "falling back to walkforward.",
            RuntimeWarning, stacklevel=2,
        )
        return self._from_walkforward(agent_name)

    def build_all_agents(self, source: str = "auto") -> Dict[str, pd.Series]:
        """Build daily return series for all agents found in walkforward results."""
        agents = self._available_agents()
        result = {}
        for agent in agents:
            try:
                s = self.build_daily_returns(agent, source=source)
                if len(s) > 0:
                    result[agent] = s
            except Exception as exc:
                logger.warning("AgentReturnSeriesBuilder: %s failed — %s", agent, exc)
        return result

    # ── Private — walkforward ─────────────────────────────────────────────────

    def _from_walkforward(self, agent_name: str) -> pd.Series:
        """
        Convert walkforward period returns to a daily series.

        Each OOS window has a total period return. We convert to an implied
        daily return via geometric compounding, then fill every trading day
        in that window. Overlapping windows for the same date are averaged.
        Zero-fill days outside any window.
        """
        if not self._wf_path.exists():
            return pd.Series(dtype=float, name=agent_name)

        df = pd.read_csv(self._wf_path)
        df = df[df["agent"] == agent_name].copy()
        if df.empty:
            return pd.Series(dtype=float, name=agent_name)

        # Average OOS return across symbols per window
        df["test_start"] = pd.to_datetime(df["test_start"], errors="coerce")
        df["test_end"]   = pd.to_datetime(df["test_end"],   errors="coerce")
        df = df.dropna(subset=["test_start", "test_end"])

        # Aggregate across symbols: mean OOS return per (agent, window)
        agg = (
            df.groupby("window")
            .agg(
                test_start  = ("test_start",  "first"),
                test_end    = ("test_end",    "first"),
                oos_return  = ("oos_return",  "mean"),
            )
            .reset_index()
        )

        # Build a DatetimeIndex covering all windows
        all_dates = pd.bdate_range(
            start=agg["test_start"].min(),
            end=agg["test_end"].max(),
        )
        daily_accumulator = pd.DataFrame(index=all_dates)
        daily_accumulator["sum"]   = 0.0
        daily_accumulator["count"] = 0

        for _, row in agg.iterrows():
            window_dates = pd.bdate_range(start=row["test_start"], end=row["test_end"])
            n = len(window_dates)
            if n == 0:
                continue
            # Geometric daily return for this window
            r_total = float(row["oos_return"])
            r_daily = (1.0 + r_total) ** (1.0 / n) - 1.0
            mask = daily_accumulator.index.isin(window_dates)
            daily_accumulator.loc[mask, "sum"]   += r_daily
            daily_accumulator.loc[mask, "count"] += 1

        # Average overlapping windows; zero where no coverage
        daily_ret = np.where(
            daily_accumulator["count"] > 0,
            daily_accumulator["sum"] / daily_accumulator["count"],
            0.0,
        )
        series = pd.Series(daily_ret, index=all_dates, name=agent_name)
        series.index = series.index.normalize()  # date only, tz-naive
        return series

    # ── Private — live executions ─────────────────────────────────────────────

    def _from_live_executions(self, agent_name: str) -> pd.Series:
        """
        Reconstruct daily returns from BUY/SELL executions.
        Return is booked on the SELL (exit) date.
        Flat days → 0.
        """
        from src.risk.live_scorer import LiveScorer, LiveScorerConfig
        scorer = LiveScorer(LiveScorerConfig(
            decisions_path=str(self._dec_path),
            executions_path=str(self._exc_path),
        ))
        scorer._load()
        trips = [t for t in scorer._roundtrips if t.agent == agent_name]

        if not trips:
            return pd.Series(dtype=float, name=agent_name)

        # Book return on exit date; same-day trades sum
        exit_returns: Dict[pd.Timestamp, list] = {}
        for t in trips:
            date = t.exit_date.normalize().tz_localize(None)
            exit_returns.setdefault(date, []).append(t.return_pct)

        # Span from first entry to last exit, business days
        all_dates = pd.bdate_range(
            start=min(t.entry_date.normalize().tz_localize(None) for t in trips),
            end=max(t.exit_date.normalize().tz_localize(None) for t in trips),
        )
        daily = pd.Series(0.0, index=all_dates, name=agent_name)
        for date, rets in exit_returns.items():
            if date in daily.index:
                daily.loc[date] = float(np.mean(rets))

        daily.index = daily.index.normalize()
        return daily

    # ── Private — helpers ─────────────────────────────────────────────────────

    def _available_agents(self) -> list[str]:
        if not self._wf_path.exists():
            return []
        df = pd.read_csv(self._wf_path, usecols=["agent"])
        return list(df["agent"].unique())


# ─────────────────────────────────────────────────────────────────────────────
# Part 3 — FactorRegression
# ─────────────────────────────────────────────────────────────────────────────

class FactorRegression:
    """
    OLS regression with Newey-West HAC standard errors.
    Regresses excess agent returns on Carhart 4-factor or FF5 model.
    """

    def __init__(self, model: str = "carhart", min_observations: int = MIN_OBS_DEFAULT) -> None:
        if model not in {"carhart", "ff5"}:
            raise ValueError(f"model must be 'carhart' or 'ff5'")
        self.model = model
        self.min_observations = min_observations

    def run(self, agent_returns: pd.Series, factors: pd.DataFrame) -> RegressionResult:
        import statsmodels.api as sm

        agent_name   = agent_returns.name or "unknown"
        factor_cols  = _CARHART_FACTORS if self.model == "carhart" else _FF5_FACTORS

        # Align on common dates (inner join)
        agent_idx = agent_returns.index
        if hasattr(agent_idx, "tz") and agent_idx.tz is not None:
            agent_returns = agent_returns.copy()
            agent_returns.index = agent_idx.tz_localize(None)

        factors_aligned = factors.copy()
        factors_aligned.index = pd.to_datetime(factors_aligned.index).normalize()
        if hasattr(factors_aligned.index, "tz") and factors_aligned.index.tz is not None:
            factors_aligned.index = factors_aligned.index.tz_localize(None)

        agent_returns.index = pd.to_datetime(agent_returns.index).normalize()

        combined = pd.concat(
            [agent_returns.rename("R_agent"), factors_aligned],
            axis=1,
            join="inner",
        ).dropna()

        T = len(combined)
        source = _infer_source(agent_returns)

        if T < self.min_observations:
            warnings.warn(
                f"{agent_name}: only {T} observations after alignment "
                f"(minimum is {self.min_observations}). Verdict skipped.",
                RuntimeWarning, stacklevel=2,
            )
            return _insufficient_result(agent_name, self.model, T, source)

        # ── Série dégénérée : refuser plutôt que fabriquer un alpha ───────────
        #
        # Constaté le 2026-08-17 sur BuffettAgent : 854 « rendements
        # quotidiens » ne portaient que 13 valeurs distinctes, parce que
        # `_from_walkforward` remplit chaque jour d'une fenêtre avec le MÊME
        # rendement implicite. La régression annonçait alors un alpha de
        # +18.5 % annualisé, un t de 14.8, une p-value de 9×10⁻⁵⁰ et un
        # Information Ratio de 21.4 — tous produits par une volatilité
        # résiduelle de 0.885 % par an, contre 31 % pour AAPL seul.
        #
        # La cause est de fond : un rendement de période ne contient PAS
        # l'information du chemin quotidien. Savoir qu'une stratégie a gagné
        # 20 % en six mois ne dit pas si elle est montée en ligne droite ou
        # passée par −40 %. Remplir les jours d'une constante invente le chemin
        # le plus lisse possible, puis la régression mesure qu'il est lisse.
        #
        # Le nombre de valeurs distinctes est donc la borne haute de l'effectif
        # réellement indépendant. On l'exige au même niveau que T.
        n_distinct = int(combined["R_agent"].round(12).nunique())
        if n_distinct < self.min_observations:
            warnings.warn(
                f"{agent_name}: {T} observations mais seulement {n_distinct} "
                f"valeurs distinctes — série dégénérée, régression refusée.",
                RuntimeWarning, stacklevel=2,
            )
            return _degenerate_result(agent_name, self.model, T, n_distinct, source)

        # Excess return of agent over Rf
        y = combined["R_agent"] - combined["RF"]
        X = sm.add_constant(combined[factor_cols])

        # Newey-West HAC lag selection: L ≈ floor(4·(T/100)^(2/9))
        nw_lags = max(1, math.floor(4 * (T / 100) ** (2 / 9)))

        res = sm.OLS(y, X).fit(
            cov_type="HAC",
            cov_kwds={"maxlags": nw_lags},
        )

        alpha_daily = float(res.params["const"])
        alpha_ann   = alpha_daily * TRADING_DAYS_PER_YEAR
        alpha_t     = float(res.tvalues["const"])
        alpha_p     = float(res.pvalues["const"])

        betas       = {f: float(res.params[f])  for f in factor_cols}
        beta_t      = {f: float(res.tvalues[f]) for f in factor_cols}
        beta_p      = {f: float(res.pvalues[f]) for f in factor_cols}

        # Residual (idiosyncratic) vol, annualised
        resid_vol_ann = float(res.resid.std() * math.sqrt(TRADING_DAYS_PER_YEAR))
        ir = (alpha_ann / resid_vol_ann) if resid_vol_ann > 1e-12 else 0.0

        verdict = _generate_verdict(
            agent_name=agent_name,
            alpha_ann=alpha_ann,
            alpha_t=alpha_t,
            ir=ir,
            r2=float(res.rsquared),
            betas=betas,
            beta_t=beta_t,
            beta_p=beta_p,
            factor_cols=factor_cols,
        )

        return RegressionResult(
            agent_name        = agent_name,
            model             = self.model,
            alpha_daily       = alpha_daily,
            alpha_annualized  = alpha_ann,
            alpha_tstat       = alpha_t,
            alpha_pvalue      = alpha_p,
            alpha_significant = abs(alpha_t) > 1.96,
            betas             = betas,
            beta_tstats       = beta_t,
            beta_pvalues      = beta_p,
            r_squared         = float(res.rsquared),
            adj_r_squared     = float(res.rsquared_adj),
            n_observations    = T,
            residual_vol_annual = resid_vol_ann,
            information_ratio = ir,
            source            = source,
            insufficient_data = False,
            interpretation    = verdict,
        )

    def run_all(
        self,
        agent_series: Dict[str, pd.Series],
        factors: pd.DataFrame,
    ) -> Dict[str, RegressionResult]:
        results = {}
        for agent, series in agent_series.items():
            try:
                results[agent] = self.run(series, factors)
            except Exception as exc:
                logger.error("FactorRegression: %s failed — %s", agent, exc)
        return results


# ─────────────────────────────────────────────────────────────────────────────
# Part 4 — Verdict generation
# ─────────────────────────────────────────────────────────────────────────────

def _generate_verdict(
    agent_name: str,
    alpha_ann: float,
    alpha_t: float,
    ir: float,
    r2: float,
    betas: dict,
    beta_t: dict,
    beta_p: dict,
    factor_cols: list[str],
) -> str:
    # Dominant factor: highest |beta| that is significant (|t|>1.96)
    sig_betas = {
        f: abs(betas[f]) for f in factor_cols
        if abs(beta_t.get(f, 0)) > 1.96
    }
    dominant_factor_str = ""
    if sig_betas:
        dom_f = max(sig_betas, key=sig_betas.__getitem__)
        dom_b = betas[dom_f]
        dominant_factor_str = (
            f" Exposition dominante : {dom_f} (β={dom_b:+.2f})"
            " — rendement principalement factoriel."
        )

    if abs(alpha_t) > 1.96 and alpha_ann > 0:
        return (
            f"VRAI ALPHA : skill confirmé, α annualisé = {alpha_ann:+.1%}, "
            f"IR = {ir:.2f}. Agent à conserver et améliorer."
            f"{dominant_factor_str}"
        )
    elif abs(alpha_t) > 1.96 and alpha_ann < 0:
        return (
            f"ALPHA NÉGATIF : l'agent détruit de la valeur après ajustement factoriel "
            f"(α = {alpha_ann:+.1%}, t={alpha_t:.2f}). Coupe prioritaire."
            f"{dominant_factor_str}"
        )
    else:
        return (
            f"PAS D'ALPHA PROUVÉ : le rendement s'explique par l'exposition factorielle. "
            f"R²={r2:.1%} (|t|={abs(alpha_t):.2f} < 1.96). "
            f"Candidat à la coupe ou à revoir."
            f"{dominant_factor_str}"
        )


def _insufficient_result(
    agent_name: str, model: str, n: int, source: str
) -> RegressionResult:
    nan = float("nan")
    return RegressionResult(
        agent_name        = agent_name,
        model             = model,
        alpha_daily       = nan,
        alpha_annualized  = nan,
        alpha_tstat       = nan,
        alpha_pvalue      = nan,
        alpha_significant = False,
        betas             = {},
        beta_tstats       = {},
        beta_pvalues      = {},
        r_squared         = nan,
        adj_r_squared     = nan,
        n_observations    = n,
        residual_vol_annual = nan,
        information_ratio = nan,
        source            = source,
        insufficient_data = True,
        interpretation    = (
            f"DONNÉES INSUFFISANTES : {n} obs < {MIN_OBS_DEFAULT}. "
            "Verdict impossible, accumuler plus de track record."
        ),
    )


def _degenerate_result(
    agent_name: str, model: str, n: int, n_distinct: int, source: str
) -> RegressionResult:
    """
    Verdict refusé : la série de rendements ne porte pas assez d'information
    indépendante pour être régressée.

    On renvoie des NaN plutôt qu'un zéro : un alpha nul serait une affirmation
    (« cet agent n'a pas d'avantage »), alors qu'on ne sait rien du tout.
    """
    nan = float("nan")
    return RegressionResult(
        agent_name        = agent_name,
        model             = model,
        alpha_daily       = nan,
        alpha_annualized  = nan,
        alpha_tstat       = nan,
        alpha_pvalue      = nan,
        alpha_significant = False,
        betas             = {},
        beta_tstats       = {},
        beta_pvalues      = {},
        r_squared         = nan,
        adj_r_squared     = nan,
        n_observations    = n,
        residual_vol_annual = nan,
        information_ratio = nan,
        source            = source,
        insufficient_data = True,
        interpretation    = (
            f"VERDICT REFUSÉ — série dégénérée : {n} lignes pour seulement "
            f"{n_distinct} valeurs distinctes. Les rendements de fenêtre "
            f"walk-forward sont étalés en constantes sur chaque jour, ce qui "
            f"écrase la volatilité et fabrique un alpha significatif à partir "
            f"de rien. Un rendement de période ne contient pas le chemin "
            f"quotidien : l'information n'existe pas. Il faut {MIN_OBS_DEFAULT} "
            f"observations RÉELLEMENT distinctes — en pratique, du live."
        ),
    )


def _infer_source(series: pd.Series) -> str:
    """Heuristic: if series has uniform-ish daily returns it came from walkforward."""
    nonzero = series[series != 0]
    if len(nonzero) < 5:
        return "unknown"
    # Live series tend to have a much lower fraction of non-zero days
    nonzero_ratio = len(nonzero) / len(series)
    return "walkforward" if nonzero_ratio > 0.4 else "live"


# ─────────────────────────────────────────────────────────────────────────────
# Part 5 — FactorReporter
# ─────────────────────────────────────────────────────────────────────────────

class FactorReporter:

    # ── Console table ─────────────────────────────────────────────────────────

    def format_console_table(self, results: Dict[str, RegressionResult]) -> str:
        if not results:
            return "No results to display."

        rows = sorted(
            results.values(),
            key=lambda r: (r.insufficient_data, -(r.alpha_annualized if not math.isnan(r.alpha_annualized) else -999)),
        )

        header = (
            f"{'Agent':<26} {'α ann':>7} {'t-stat':>7} {'sig':>4} "
            f"{'β_mkt':>6} {'β_mom':>6} {'R²':>5} {'IR':>6} "
            f"{'N':>5}  Verdict"
        )
        sep = "─" * 120

        lines = [sep, header, sep]
        for r in rows:
            if r.insufficient_data:
                lines.append(
                    f"{r.agent_name:<26} {'—':>7} {'—':>7} {'—':>4} "
                    f"{'—':>6} {'—':>6} {'—':>5} {'—':>6} "
                    f"{r.n_observations:>5}  DONNÉES INSUFFISANTES"
                )
                continue

            sig_mark = "✓" if r.alpha_significant else " "
            mkt_b    = r.betas.get("Mkt-RF", float("nan"))
            mom_b    = r.betas.get("Mom",   r.betas.get("MOM", float("nan")))
            verdict_short = r.interpretation.split(".")[0][:60]

            lines.append(
                f"{r.agent_name:<26} "
                f"{r.alpha_annualized:>+7.1%} "
                f"{r.alpha_tstat:>7.2f} "
                f"{sig_mark:>4} "
                f"{mkt_b:>6.2f} "
                f"{mom_b:>6.2f} "
                f"{r.r_squared:>5.1%} "
                f"{r.information_ratio:>6.2f} "
                f"{r.n_observations:>5}  "
                f"{verdict_short}"
            )
        lines.append(sep)
        lines.append(f"Model: {next(iter(results.values())).model.upper()} | "
                     f"HAC Newey-West SE | α sig at |t|>1.96")
        return "\n".join(lines)

    # ── Telegram ──────────────────────────────────────────────────────────────

    def format_telegram(self, results: Dict[str, RegressionResult]) -> str:
        alpha_pos, alpha_none, alpha_neg, insuf = [], [], [], []

        for r in results.values():
            if r.insufficient_data:
                insuf.append(r)
            elif r.alpha_significant and r.alpha_annualized > 0:
                alpha_pos.append(r)
            elif r.alpha_significant and r.alpha_annualized < 0:
                alpha_neg.append(r)
            else:
                alpha_none.append(r)

        lines = [f"📊 *Factor Analysis — Milan Capital* ({datetime.now().strftime('%Y-%m-%d')})"]
        lines.append(f"Model: {next(iter(results.values())).model.upper()} | Carhart 4-factor")
        lines.append("")

        if alpha_pos:
            lines.append("✅ *VRAI ALPHA*")
            for r in sorted(alpha_pos, key=lambda x: -x.alpha_annualized):
                lines.append(
                    f"  • {r.agent_name}: α={r.alpha_annualized:+.1%}, "
                    f"t={r.alpha_tstat:.2f}, IR={r.information_ratio:.2f}"
                )

        if alpha_none:
            lines.append("\n⚠️ *BETA DÉGUISÉ* (alpha non prouvé)")
            for r in sorted(alpha_none, key=lambda x: -(x.alpha_annualized or 0)):
                dom = _dominant_factor_name(r)
                lines.append(
                    f"  • {r.agent_name}: α={r.alpha_annualized:+.1%}, "
                    f"R²={r.r_squared:.0%}{', exp. ' + dom if dom else ''}"
                )

        if alpha_neg:
            lines.append("\n❌ *ALPHA NÉGATIF* (coupe prioritaire)")
            for r in sorted(alpha_neg, key=lambda x: x.alpha_annualized):
                lines.append(
                    f"  • {r.agent_name}: α={r.alpha_annualized:+.1%}, "
                    f"t={r.alpha_tstat:.2f}"
                )

        if insuf:
            lines.append("\n⏳ *DONNÉES INSUFFISANTES*")
            for r in insuf:
                lines.append(f"  • {r.agent_name}: {r.n_observations} obs")

        return "\n".join(lines)

    # ── Save ──────────────────────────────────────────────────────────────────

    def save_json(self, results: Dict[str, RegressionResult], path: str) -> None:
        import json
        out = {name: r.to_dict() for name, r in results.items()}
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            json.dump(out, f, indent=2, default=str)
        logger.info("Factor analysis saved to %s", path)

    def save_csv(self, results: Dict[str, RegressionResult], path: str) -> None:
        rows = []
        for r in results.values():
            row = {
                "agent":               r.agent_name,
                "model":               r.model,
                "alpha_annualized":    r.alpha_annualized,
                "alpha_tstat":         r.alpha_tstat,
                "alpha_pvalue":        r.alpha_pvalue,
                "alpha_significant":   r.alpha_significant,
                "r_squared":           r.r_squared,
                "adj_r_squared":       r.adj_r_squared,
                "n_observations":      r.n_observations,
                "residual_vol_annual": r.residual_vol_annual,
                "information_ratio":   r.information_ratio,
                "source":              r.source,
                "insufficient_data":   r.insufficient_data,
                "interpretation":      r.interpretation,
                "timestamp":           r.timestamp,
            }
            for factor, beta in r.betas.items():
                row[f"beta_{factor}"] = beta
            rows.append(row)
        df = pd.DataFrame(rows)
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(path, index=False)
        logger.info("Factor analysis CSV saved to %s", path)


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _dominant_factor_name(r: RegressionResult) -> str:
    sig = {
        f: abs(b)
        for f, b in r.betas.items()
        if abs(r.beta_tstats.get(f, 0)) > 1.96
    }
    if not sig:
        return ""
    return max(sig, key=sig.__getitem__)
