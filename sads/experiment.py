"""Rolling one-day forecasts, chronological tuning, and research-question tables."""
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import json
from importlib import metadata as package_metadata
from pathlib import Path
import platform
import subprocess
from typing import NamedTuple, TypeAlias, TypedDict

import numpy as np
import pandas as pd
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure
from scipy.stats import norm
from threadpoolctl import threadpool_limits

from . import constants as c
from .data import align_signals, fingerprint, load_returns, stress_releases
from .models import (
    bayesian_variance,
    ewma_variance_path,
    pca_path,
    sample_variance,
    shrinkage_variance,
    tail_forecast,
)
from .selection import adaptive_path, best_rank, regime_rank, trailing_optimal_path
from .statistics import (
    MCSResult,
    RankRegimeResult,
    coverage,
    fz0,
    gw_test,
    hac_covariance_mean,
    is_significant_improvement,
    model_confidence_set,
    rank_regime_test,
)
from .types import (
    BoolArray,
    Distribution,
    FloatArray,
    PathLike,
    StressSignal,
    TailForecast,
)

JSONValue: TypeAlias = (
    str | int | float | bool | None | list["JSONValue"] | dict[str, "JSONValue"]
)


class CandidateForecasts(NamedTuple):
    """Aligned metadata, per-rank forecasts/losses, and baseline variances."""

    metadata: pd.DataFrame
    risk: TailForecast
    loss: FloatArray
    shrinkage: FloatArray
    sample: FloatArray
    ewma: FloatArray
    bayes: FloatArray


class RobustnessResult(TypedDict):
    """Joint assessment of full-sample, holdout, and crisis-deletion comparisons."""

    conclusion: str
    full_sample_improvement: bool
    all_three_episodes_available: bool
    all_holdout_improvements: bool
    all_deletion_improvements: bool


class SignalEvaluation(TypedDict):
    """Rank effects, model confidence set, and robustness conclusions."""

    rank_regime_tests: dict[str, RankRegimeResult]
    model_confidence_set: MCSResult
    robustness: dict[str, RobustnessResult]


class SignalResult(SignalEvaluation):
    """Forecast tables and statistical results for one stress signal."""

    forecasts: pd.DataFrame
    summary: pd.DataFrame
    coverage: pd.DataFrame
    comparisons: pd.DataFrame
    crisis_holdout: pd.DataFrame
    crisis_deletion: pd.DataFrame


@dataclass
class Config:
    """Data paths and study parameters; all window lengths count trading dates.

    calibration counts completed forecasts reserved for rank initialization.
    publication_lag_days is a calendar-day assumption when release dates are
    absent. bootstrap_block is the stationary bootstrap's mean block length.
    """

    returns_path: PathLike = c.RETURNS_PATH
    stress_path: PathLike = c.STRESS_PATH
    release_calendar: PathLike | None = c.RELEASE_CALENDAR
    publication_lag_days: int = c.PUBLICATION_LAG_DAYS
    start: str = c.SAMPLE_START
    end: str | None = c.SAMPLE_END
    window: int = c.ESTIMATION_WINDOW
    calibration: int = c.CALIBRATION_WINDOW
    max_rank: int = c.MAX_RANK
    selection_window: int = c.SELECTION_WINDOW
    min_regime: int = c.MIN_REGIME_OBSERVATIONS
    volatility_window: int = c.VOLATILITY_WINDOW
    distribution: Distribution = c.RETURN_DISTRIBUTION
    student_df: float = c.STUDENT_DF
    bayes_prior_shape: float = c.BAYES_PRIOR_SHAPE
    bayes_prior_scale: float = c.BAYES_PRIOR_SCALE
    bootstrap_replications: int = c.BOOTSTRAP_REPLICATIONS
    bootstrap_block: int = c.BOOTSTRAP_BLOCK_LENGTH
    hac_lags: int | None = c.HAC_LAGS
    seed: int = c.RANDOM_SEED

    def validate(self) -> None:
        """Check sample-size, rank, and distribution constraints before estimation."""
        if self.window < c.MIN_WINDOW or self.calibration < c.MIN_WINDOW:
            raise ValueError("window and calibration must be >=20")
        if not 1 <= self.max_rank < min(c.N_INDUSTRIES, self.window):
            raise ValueError("max_rank must be in [1, min(49,window)-1]")
        if (
            self.selection_window < c.MIN_WINDOW
            or self.min_regime < 1
            or self.volatility_window < c.MIN_VOLATILITY_WINDOW
        ):
            raise ValueError("Invalid selection or volatility window")
        if (
            self.bootstrap_replications < c.MIN_BOOTSTRAP_REPLICATIONS
            or self.bootstrap_block < 1
        ):
            raise ValueError("Use >=99 bootstrap replications and block length >=1")
        if (
            self.distribution not in (c.NORMAL_DISTRIBUTION, c.STUDENT_DISTRIBUTION)
            or self.student_df <= 2
        ):
            raise ValueError(
                "Invalid return distribution or Student degrees of freedom"
            )
        if (
            not np.isfinite(self.bayes_prior_shape) or self.bayes_prior_shape <= 1
            or not np.isfinite(self.bayes_prior_scale) or self.bayes_prior_scale <= 0
        ):
            raise ValueError("Bayesian prior requires finite shape > 1 and scale > 0")


def candidate_forecasts(
    returns: pd.DataFrame, signals: pd.DataFrame, cfg: Config
) -> CandidateForecasts:
    """Estimate every rank's one-day tail forecast on rolling prior returns.

    returns and signals share the trading-date index. Retained dates need
    complete signals and W preceding return rows. Risk and loss arrays have
    shape (forecast dates, max_rank + 1), including calibration forecasts.
    """
    ranks = range(cfg.max_rank + 1)
    values = returns.to_numpy()
    ewma = ewma_variance_path(values.mean(axis=1), cfg.window)
    sample = []
    bayes = []
    rows, covariances, bn, absorption, shrinkage = [], [], [], [], []
    for i in range(cfg.window, len(returns)):
        row = signals.iloc[i]
        if not np.isfinite(row[list(c.SIGNAL_COLUMNS)].to_numpy(dtype=float)).all():
            continue
        x = values[i - cfg.window : i]
        variance, bn_index, absorbed = pca_path(x, ranks)
        covariances.append(variance)
        bn.append(bn_index)
        absorption.append(absorbed)
        rows.append(i)
        shrinkage.append(shrinkage_variance(x))
        sample.append(sample_variance(x))
        bayes.append(bayesian_variance(x, cfg.bayes_prior_shape, cfg.bayes_prior_scale))
    if len(rows) < cfg.calibration + c.MIN_EVALUATION_OBSERVATIONS:
        raise ValueError(
            "Need at least calibration+20 candidate forecasts after the rolling window"
        )
    metadata = signals.iloc[rows].copy()
    metadata["return"] = returns.iloc[rows].mean(axis=1)
    metadata["bn_rank"] = bn
    metadata["absorption_ratio"] = absorption
    metadata["estimation_end"] = returns.index[np.asarray(rows) - 1]
    risk = tail_forecast(np.asarray(covariances), cfg.distribution, cfg.student_df)
    loss = fz0(metadata["return"].to_numpy()[:, None], risk[1], risk[2])
    return CandidateForecasts(
        metadata, risk, loss, np.asarray(shrinkage), np.asarray(sample), ewma[rows],
        np.asarray(bayes),
    )


def make_forecast_frame(
    metadata: pd.DataFrame,
    risk: TailForecast,
    loss: FloatArray,
    shrinkage: FloatArray,
    cfg: Config,
    signal: StressSignal,
    sample: FloatArray,
    ewma: FloatArray,
    bayes: FloatArray,
) -> tuple[pd.DataFrame, BoolArray]:
    """Select model ranks and assemble one row per evaluation date and model.

    Candidate arrays must share metadata's row order. Calibration rows are
    excluded from the returned table; the regime mask includes all dates.
    """
    if signal == c.STLFSI_SIGNAL:
        stress, threshold = metadata.stress, metadata.stress_threshold
    else:
        stress, threshold = metadata.realized_vol, metadata.vol_threshold
    regimes = (stress > threshold).to_numpy()
    fixed = best_rank(loss[: cfg.calibration])
    selections = {
        c.FIXED_PCA: np.full(len(loss), fixed),
        c.BAI_NG: metadata.bn_rank.to_numpy(),
        c.ROLLING_PCA: trailing_optimal_path(loss, cfg.calibration, cfg.selection_window),
        c.SADS: adaptive_path(
            loss, regimes, cfg.calibration, cfg.selection_window, cfg.min_regime
        ),
    }
    indices = np.arange(cfg.calibration, len(loss))
    base = metadata.iloc[indices].copy()
    base["elevated"] = regimes[indices]
    base["trailing_optimal_rank"] = trailing_optimal_path(loss, cfg.calibration)[
        indices
    ]
    base["stress_signal"] = np.asarray(stress)[indices]
    base["regime_threshold"] = np.asarray(threshold)[indices]

    parts = []
    variances = {
        c.LEDOIT_WOLF: shrinkage, c.SAMPLE_VAR: sample, c.EWMA: ewma,
        c.BAYES_VAR: bayes,
    }
    for name in (*selections, *variances):
        part = base.copy()
        part["model"] = name
        if name in variances:
            part["rank"] = np.nan
            forecasts = tail_forecast(
                variances[name][indices], cfg.distribution, cfg.student_df
            )
        else:
            chosen = selections[name][indices].astype(int)
            part["rank"] = chosen
            forecasts = tuple(values[indices, chosen] for values in risk)
        for column, values in zip(c.RISK_COLUMNS, forecasts):
            part[column] = values
        part["fz0"] = fz0(part["return"], part.var_025, part.es_025)
        parts.append(part)
    return pd.concat(parts).reset_index(), regimes


def _loss_panel(forecasts: pd.DataFrame) -> pd.DataFrame:
    """Align model FZ0 losses on the forecast-date index."""
    return forecasts.pivot(index="date", columns="model", values="fz0").sort_index()


def evaluate(
    forecasts: pd.DataFrame, cfg: Config
) -> tuple[
    pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, RankRegimeResult], MCSResult
]:
    """Summarize FZ0, VaR calibration, GW comparisons, rank effects, and the MCS.

    Forecast rows are ordered by trading date within each model. GW loss
    differences use SADS minus each comparator; negative values favor SADS.
    """
    summaries, calibrations, comparisons = [], [], []
    pivot = _loss_panel(forecasts)
    reference = forecasts[forecasts.model == c.SADS].set_index("date").loc[pivot.index]
    for name, group in forecasts.groupby("model", sort=True):
        group = group.sort_values("date")
        mean = group.fz0.mean()
        se = float(np.sqrt(max(hac_covariance_mean(group.fz0, cfg.hac_lags)[0, 0], 0)))
        margin = norm.ppf(1 - c.SIGNIFICANCE_LEVEL / 2) * se
        summaries.append(
            dict(
                model=name,
                n=len(group),
                mean_fz0=mean,
                std_fz0=group.fz0.std(ddof=1),
                hac_se_fz0=se,
                hac_ci_lower=mean - margin,
                hac_ci_upper=mean + margin,
                violation_rate_01=(group["return"] < group.var_01).mean(),
                violation_rate_025=(group["return"] < group.var_025).mean(),
                elevated_fz0=group.loc[group.elevated, "fz0"].mean(),
                lower_fz0=group.loc[~group.elevated, "fz0"].mean(),
                mean_rank=group["rank"].mean(),
            )
        )
        # Christoffersen transitions follow the full trading calendar.
        for alpha, column in c.VAR_COVERAGE_LEVELS:
            calibrations.append(
                dict(
                    model=name,
                    alpha=alpha,
                    **coverage(group["return"], group[column], alpha),
                )
            )
    instruments = np.column_stack([np.ones(len(pivot)), reference.elevated.astype(int)])
    comparisons_spec = (
        (c.FULL_SAMPLE, c.GW_CONSTANT, None, None),
        (c.FULL_SAMPLE, c.GW_STRESS_INSTRUMENT, None, instruments),
        (c.ELEVATED_STRESS, c.GW_CONSTANT, reference.elevated.to_numpy(), None),
    )
    for other in pivot.columns:
        if other == c.SADS:
            continue
        difference = (pivot[c.SADS] - pivot[other]).to_numpy()
        for regime, test, mask, instrument in comparisons_spec:
            result = gw_test(difference, mask, cfg.hac_lags, instrument)
            margin = norm.ppf(1 - c.SIGNIFICANCE_LEVEL / 2) * result["se"]
            comparisons.append(
                dict(
                    model_a=c.SADS,
                    model_b=other,
                    regime=regime,
                    test=test,
                    **result,
                    ci_lower=result["mean_difference"] - margin,
                    ci_upper=result["mean_difference"] + margin,
                    significant_improvement=bool(
                        is_significant_improvement(
                            result["mean_difference"], result["p_value"]
                        )
                    ),
                )
            )
    rq2 = {
        name: rank_regime_test(reference[column], reference.elevated, cfg.hac_lags)
        for name, column in c.RANK_DIAGNOSTICS
    }
    mcs = model_confidence_set(
        pivot.to_numpy(),
        list(pivot.columns),
        replications=cfg.bootstrap_replications,
        block_length=cfg.bootstrap_block,
        seed=cfg.seed,
    )
    return (
        pd.DataFrame(summaries),
        pd.DataFrame(calibrations),
        pd.DataFrame(comparisons),
        rq2,
        mcs,
    )


def crisis_checks(
    metadata: pd.DataFrame, loss: FloatArray, regimes: BoolArray, cfg: Config
) -> pd.DataFrame:
    """Fit ranks before each crisis and evaluate them on the held-out episode."""
    rows = []
    available = np.arange(len(loss)) >= cfg.calibration
    dates = metadata.index
    for episode, (start, end) in c.CRISES.items():
        heldout = np.asarray((dates >= start) & (dates <= end)) & available
        train = np.asarray(dates < start)
        if (
            heldout.sum() < c.MIN_EVALUATION_OBSERVATIONS
            or train.sum() < cfg.calibration
        ):
            continue
        fixed = best_rank(loss[train])
        low = regime_rank(loss[train], ~regimes[train], cfg.min_regime)
        high = regime_rank(loss[train], regimes[train], cfg.min_regime)
        chosen = np.where(regimes, high, low)
        adaptive_loss = loss[np.arange(len(loss)), chosen]
        fixed_loss = loss[:, fixed]
        bn_loss = loss[np.arange(len(loss)), metadata.bn_rank.to_numpy()]
        rolling_rank = best_rank(loss[train][-cfg.selection_window :])
        for name, baseline in (
            (c.FIXED_PCA, fixed_loss),
            (c.BAI_NG, bn_loss),
            (c.ROLLING_PCA, loss[:, rolling_rank]),
        ):
            rows.append(
                dict(
                    episode=episode,
                    baseline=name,
                    procedure=c.CRISIS_HOLDOUT_PROCEDURE,
                    fixed_rank=fixed,
                    rolling_rank=rolling_rank,
                    low_rank=low if (~regimes).any() else np.nan,
                    high_rank=high if regimes.any() else np.nan,
                    **gw_test(adaptive_loss - baseline, heldout, cfg.hac_lags),
                )
            )
    return pd.DataFrame(rows, columns=c.HOLDOUT_COLUMNS)


def deletion_checks(forecasts: pd.DataFrame, cfg: Config) -> pd.DataFrame:
    """Retest main forecast differences with each crisis year excluded in turn."""
    pivot = _loss_panel(forecasts)
    rows = []
    for episode, (start, end) in c.CRISES.items():
        inside = np.asarray((pivot.index >= start) & (pivot.index <= end))
        if not inside.any():
            continue
        for baseline in c.MATCHED_BASELINES:
            result = gw_test(
                (pivot[c.SADS] - pivot[baseline]).to_numpy(), ~inside, cfg.hac_lags
            )
            rows.append(
                dict(
                    excluded_episode=episode,
                    baseline=baseline,
                    significant_improvement=bool(
                        is_significant_improvement(
                            result["mean_difference"], result["p_value"]
                        )
                    ),
                    **result,
                )
            )
    return pd.DataFrame(rows, columns=c.DELETION_COLUMNS)


def robustness_summary(
    comparisons: pd.DataFrame, holdout: pd.DataFrame, deletion: pd.DataFrame
) -> dict[str, RobustnessResult]:
    """Require significant improvement in the full sample and every crisis check."""
    results: dict[str, RobustnessResult] = {}
    for baseline in c.MATCHED_BASELINES:
        full = comparisons[
            (comparisons.model_b == baseline)
            & (comparisons.regime == c.FULL_SAMPLE)
            & (comparisons.test == c.GW_CONSTANT)
        ].iloc[0]
        full_improvement = bool(
            is_significant_improvement(full.mean_difference, full.p_value)
        )
        held = holdout[holdout.baseline == baseline] if len(holdout) else pd.DataFrame()
        deleted = (
            deletion[deletion.baseline == baseline] if len(deletion) else pd.DataFrame()
        )
        complete = len(held) == len(c.CRISES) and len(deleted) == len(c.CRISES)
        holdout_pass = bool(
            len(held)
            and is_significant_improvement(held.mean_difference, held.p_value).all()
        )
        deletion_pass = bool(len(deleted) and deleted.significant_improvement.all())
        if not full_improvement:
            conclusion = c.NO_FULL_SAMPLE_IMPROVEMENT
        elif not complete:
            conclusion = c.INCOMPLETE_ROBUSTNESS
        elif holdout_pass and deletion_pass:
            conclusion = c.ROBUST_IMPROVEMENT
        else:
            conclusion = c.NOT_ROBUST
        results[baseline] = dict(
            conclusion=conclusion,
            full_sample_improvement=full_improvement,
            all_three_episodes_available=complete,
            all_holdout_improvements=holdout_pass,
            all_deletion_improvements=deletion_pass,
        )
    return results


def _json_value(value: object) -> JSONValue:
    """Convert numerical results to JSON values, representing undefined tests as null."""
    if isinstance(value, dict):
        return {str(k): _json_value(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_value(v) for v in value]
    if isinstance(value, np.bool_):
        return bool(value)
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, (float, np.floating)):
        return float(value) if np.isfinite(value) else None
    if isinstance(value, Path):
        return str(value)
    if value is None or isinstance(value, (str, int, bool)):
        return value
    raise TypeError(f"Cannot serialize {type(value).__name__} as JSON")


def write_json(path: PathLike, value: object) -> None:
    """Save settings or statistical results as JSON with finite numeric values."""
    Path(path).write_text(
        json.dumps(_json_value(value), indent=c.JSON_INDENT, allow_nan=False) + "\n"
    )


def runtime_settings() -> dict[str, object]:
    """Collect execution environment metadata without requiring git or Linux."""
    packages = {}
    for name in c.PACKAGE_NAMES:
        try:
            packages[name] = package_metadata.version(name)
        except package_metadata.PackageNotFoundError:
            packages[name] = "unavailable"
    try:
        commit = subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=c.PROJECT_DIR,
            stderr=subprocess.DEVNULL,
            text=True,
            timeout=5,
        ).strip()
    except (OSError, subprocess.SubprocessError):
        commit = "unavailable"
    cpu = platform.processor()
    if not cpu:
        try:
            cpu = next(
                line.split(":", 1)[1].strip()
                for line in Path("/proc/cpuinfo").read_text().splitlines()
                if line.startswith("model name")
            )
        except (OSError, StopIteration):
            cpu = "unavailable"
    return dict(
        python=platform.python_version(),
        packages=packages,
        git_commit=commit,
        platform=platform.platform(),
        cpu=cpu,
    )


def plot_results(forecasts: pd.DataFrame, output: PathLike) -> None:
    """Save rank paths, SADS tail forecasts, and cumulative model scores."""
    reference = forecasts[forecasts.model == c.SADS].set_index("date")
    figure = Figure(figsize=c.FIGURE_SIZE)
    FigureCanvasAgg(figure)
    axes = figure.subplots(3, 1, sharex=True)
    for name in c.PCA_MODELS:
        frame = forecasts[forecasts.model == name].set_index("date")
        axes[0].plot(
            frame.index, frame["rank"], label=name, linewidth=c.RANK_LINE_WIDTH
        )
        axes[2].plot(
            frame.index, frame.fz0.cumsum(), label=name, linewidth=c.SCORE_LINE_WIDTH
        )
    axes[0].set_ylabel(c.RANK_AXIS_LABEL)
    axes[0].legend(ncol=c.LEGEND_COLUMNS)
    for column, label, width, alpha in c.TAIL_PLOT_SERIES:
        axes[1].plot(
            reference.index,
            reference[column],
            label=label,
            linewidth=width,
            alpha=alpha,
        )
    axes[1].legend(ncol=c.LEGEND_COLUMNS)
    axes[1].set_ylabel(c.RETURN_AXIS_LABEL)
    axes[2].set_ylabel(c.SCORE_AXIS_LABEL)
    figure.tight_layout()
    figure.savefig(Path(output) / c.DIAGNOSTICS_FILENAME, dpi=c.FIGURE_DPI)


def run(cfg: Config, output: PathLike) -> dict[StressSignal, SignalResult]:
    """Run both stress signals and save their tables, figures, and study settings.

    Return results keyed by stlfsi and realized_vol, containing the same
    forecast tables and statistical results saved under the output directory.
    """
    cfg.validate()
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    returns = load_returns(cfg.returns_path, cfg.start, cfg.end)
    releases = stress_releases(
        cfg.stress_path, cfg.release_calendar, cfg.publication_lag_days
    )
    signals = align_signals(returns, releases, cfg.volatility_window)
    results: dict[StressSignal, SignalResult] = {}
    with threadpool_limits(limits=c.NUMERICAL_THREADS):
        metadata, risk, loss, shrinkage, sample, ewma, bayes = candidate_forecasts(
            returns, signals, cfg
        )
        for signal in c.STRESS_SIGNALS:
            folder = output / signal
            folder.mkdir(exist_ok=True)
            forecasts, regimes = make_forecast_frame(
                metadata, risk, loss, shrinkage, cfg, signal, sample, ewma, bayes
            )
            summary, calibration, comparisons, rq2, mcs = evaluate(forecasts, cfg)
            holdout = crisis_checks(metadata, loss, regimes, cfg)
            deletion = deletion_checks(forecasts, cfg)
            evaluation = SignalEvaluation(
                rank_regime_tests=rq2,
                model_confidence_set=mcs,
                robustness=robustness_summary(comparisons, holdout, deletion),
            )
            write_json(folder / c.EVALUATION_FILENAME, evaluation)
            plot_results(forecasts, folder)
            result = SignalResult(
                forecasts=forecasts,
                summary=summary,
                coverage=calibration,
                comparisons=comparisons,
                crisis_holdout=holdout,
                crisis_deletion=deletion,
                **evaluation,
            )
            for name, table in result.items():
                if isinstance(table, pd.DataFrame):
                    table.to_csv(folder / f"{name}.csv", index=False)
            results[signal] = result

    inputs = [cfg.returns_path, cfg.stress_path]
    if cfg.release_calendar:
        inputs.append(cfg.release_calendar)
    write_json(
        output / c.SETTINGS_FILENAME,
        dict(
            parameters=asdict(cfg),
            environment=runtime_settings(),
            inputs=[fingerprint(path) for path in inputs],
            executed_at_utc=datetime.now(timezone.utc).isoformat(),
            evaluation_start=str(metadata.index[cfg.calibration].date()),
            evaluation_end=str(metadata.index[-1].date()),
            publication_timing=c.TIMING_WITH_CALENDAR
            if cfg.release_calendar
            else c.TIMING_WITH_LAG,
            interpretation=c.STUDY_INTERPRETATION,
        ),
    )
    return results
