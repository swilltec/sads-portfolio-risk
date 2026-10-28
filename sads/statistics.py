"""Forecast scoring, calibration tests, and model comparison."""

from collections.abc import Sequence
from typing import TypedDict

import numpy as np
from numpy.typing import ArrayLike
from scipy.special import xlogy
from scipy.stats import chi2, norm

from . import constants as c
from .types import BoolArray, FloatArray, IntArray


class GWResult(TypedDict):
    """Mean loss difference and its HAC Wald test."""

    n: int
    mean_difference: float
    se: float
    statistic: float
    p_value: float


class RankRegimeResult(TypedDict):
    """Elevated-minus-lower-stress rank effect with HAC inference."""

    high_minus_low: float
    se: float
    p_value: float


class CoverageResult(TypedDict):
    """Violation frequency and likelihood-ratio calibration statistics."""

    n: int
    violations: int
    violation_rate: float
    kupiec_lr: float
    kupiec_p: float
    independence_lr: float
    independence_p: float
    conditional_coverage_p: float


class MCSStep(TypedDict):
    """One equal-predictive-ability test in the MCS elimination sequence."""

    models: list[str]
    p_value: float
    adjusted_p: float
    eliminated: str | None


class MCSResult(TypedDict):
    """Surviving models and the MCS elimination trace."""

    included: list[str]
    alpha: float
    steps: list[MCSStep]


def fz0(
    y: ArrayLike, var: ArrayLike, es: ArrayLike, alpha: float = c.ES_ALPHA
) -> FloatArray:
    """Compute joint VaR–ES FZ0 loss, broadcasting observations and forecasts.

    Forecasts use lower-return-tail signs with ES <= VaR and ES < 0.
    Lower loss indicates better joint forecasts at probability alpha.
    """
    y, var, es = np.broadcast_arrays(y, var, es)
    if (
        np.any(es >= 0)
        or np.any(es > var)
        or not all(np.isfinite(a).all() for a in (y, var, es))
    ):
        raise ValueError(
            "FZ0 requires finite lower-tail forecasts with ES <= VaR and ES < 0"
        )
    return np.asarray(
        -((y <= var) * (var - y)) / (alpha * es) + var / es + np.log(-es) - 1
    )


def is_significant_improvement(difference: ArrayLike, p_value: ArrayLike) -> BoolArray:
    """Identify lower expected loss with significance at the study's test level."""
    return np.asarray(
        (np.asarray(difference) < 0) & (np.asarray(p_value) < c.SIGNIFICANCE_LEVEL)
    )


def hac_covariance_mean(x: ArrayLike, lags: int | None = None) -> FloatArray:
    """Estimate covariance of the sample mean with Bartlett HAC weights.

    Rows are consecutive dates and columns are moment series. A vector input
    gives a 1×1 matrix. The default bandwidth is floor(4(n/100)^(2/9)).
    """
    values = np.asarray(x, dtype=float)
    if values.ndim == 1:
        values = values[:, None]
    n = len(values)
    if lags is None:
        lags = int(
            np.floor(
                c.HAC_BANDWIDTH_SCALE
                * (n / c.HAC_REFERENCE_SIZE) ** c.HAC_BANDWIDTH_EXPONENT
            )
        )
    lags = min(max(lags, 0), n - 1)
    centered = values - values.mean(axis=0)
    covariance = centered.T @ centered / n
    for lag in range(1, lags + 1):
        gamma = centered[lag:].T @ centered[:-lag] / n
        covariance += (1 - lag / (lags + 1)) * (gamma + gamma.T)
    return covariance / n


def gw_test(
    difference: ArrayLike,
    mask: ArrayLike | None = None,
    lags: int | None = None,
    instruments: ArrayLike | None = None,
) -> GWResult:
    """Test equal predictive ability from loss A minus loss B using HAC moments.

    Negative differences favor A. A mask selects dates while preserving their
    spacing through zero-weight moments. Instruments have one row per date
    and must be known at the forecast origin; the default is a constant.
    Tests with fewer than 10 selected dates or singular covariance return NaN.
    """
    d = np.asarray(difference, dtype=float)
    selected = (
        np.ones(len(d), dtype=bool) if mask is None else np.asarray(mask, dtype=bool)
    )
    n = int(selected.sum())
    result: GWResult = dict(
        n=n, mean_difference=np.nan, se=np.nan, statistic=np.nan, p_value=np.nan
    )
    if n < c.MIN_TEST_OBSERVATIONS:
        return result
    moments = d * (selected / selected.mean())
    covariance = hac_covariance_mean(moments, lags)[0, 0]
    result["mean_difference"] = float(moments.mean())
    result["se"] = float(np.sqrt(max(covariance, 0)))
    h = (
        np.ones((len(d), 1))
        if instruments is None
        else np.asarray(instruments, dtype=float)
    )
    if h.ndim == 1:
        h = h[:, None]
    g = moments[:, None] * h
    mean, cov = g.mean(axis=0), hac_covariance_mean(g, lags)
    if np.linalg.matrix_rank(cov) == cov.shape[0]:
        statistic = float(mean @ np.linalg.solve(cov, mean))
        result["statistic"] = statistic
        result["p_value"] = float(chi2.sf(statistic, h.shape[1]))
    return result


def rank_regime_test(
    ranks: ArrayLike, elevated: ArrayLike, lags: int | None = None
) -> RankRegimeResult:
    """Regress ranks on an intercept and stress indicator with HAC covariance.

    At least 10 dates in each regime are required. The reported coefficient
    measures the elevated-minus-lower-stress difference in mean rank.
    """
    y, stress = np.asarray(ranks, dtype=float), np.asarray(elevated, dtype=bool)
    if min(stress.sum(), (~stress).sum()) < c.MIN_TEST_OBSERVATIONS:
        return dict(high_minus_low=np.nan, se=np.nan, p_value=np.nan)
    design = np.column_stack([np.ones(len(y)), stress])
    beta = np.linalg.lstsq(design, y, rcond=None)[0]
    residual = y - design @ beta
    bread = np.linalg.inv(design.T @ design / len(y))
    covariance = bread @ hac_covariance_mean(design * residual[:, None], lags) @ bread
    se = np.sqrt(max(covariance[1, 1], 0))
    p = (
        2 * norm.sf(abs(beta[1] / se))
        if se
        else (1.0 if abs(beta[1]) < c.ZERO_EFFECT_TOLERANCE else 0.0)
    )
    return dict(high_minus_low=float(beta[1]), se=float(se), p_value=float(p))


def _bernoulli_ll(
    successes: ArrayLike, failures: ArrayLike, probability: ArrayLike
) -> float:
    """Evaluate a Bernoulli log likelihood, including probabilities zero and one."""
    return float(
        xlogy(successes, probability) + xlogy(failures, 1 - np.asarray(probability))
    )


def coverage(y: ArrayLike, var: ArrayLike, alpha: float) -> CoverageResult:
    """Run Kupiec and Christoffersen tests on consecutive VaR violations.

    A violation is a return strictly below VaR. Independence and conditional
    coverage p-values are NaN when a previous-violation state is unobserved.
    """
    hits = np.asarray(y) < np.asarray(var)
    n, violations = len(hits), int(hits.sum())
    p = violations / n
    lr_uc = max(
        0.0,
        -2
        * (
            _bernoulli_ll(violations, n - violations, alpha)
            - _bernoulli_ll(violations, n - violations, p)
        ),
    )
    counts = np.zeros((2, 2), dtype=int)
    for previous, current in zip(hits[:-1], hits[1:]):
        counts[int(previous), int(current)] += 1
    if np.any(counts.sum(axis=1) == 0):
        lr_ind = np.nan
    else:
        pooled = counts[:, 1].sum() / counts.sum()
        null = _bernoulli_ll(counts[:, 1].sum(), counts[:, 0].sum(), pooled)
        alternative = sum(
            _bernoulli_ll(row[1], row[0], row[1] / row.sum()) for row in counts
        )
        lr_ind = max(0.0, -2 * (null - alternative))
    return dict(
        n=n,
        violations=violations,
        violation_rate=p,
        kupiec_lr=lr_uc,
        kupiec_p=float(chi2.sf(lr_uc, 1)),
        independence_lr=lr_ind,
        independence_p=float(chi2.sf(lr_ind, 1)),
        conditional_coverage_p=float(chi2.sf(lr_uc + lr_ind, 2)),
    )


def stationary_indices(
    n: int, replications: int, block_length: int, rng: np.random.Generator
) -> IntArray:
    """Draw circular stationary-bootstrap indices with a given mean block length."""
    indices = np.empty((replications, n), dtype=int)
    indices[:, 0] = rng.integers(n, size=replications)
    for i in range(1, n):
        restart = rng.random(replications) < 1 / block_length
        indices[:, i] = np.where(
            restart, rng.integers(n, size=replications), (indices[:, i - 1] + 1) % n
        )
    return indices


def model_confidence_set(
    losses: ArrayLike,
    names: Sequence[str],
    alpha: float = c.SIGNIFICANCE_LEVEL,
    replications: int = c.BOOTSTRAP_REPLICATIONS,
    block_length: int = c.BOOTSTRAP_BLOCK_LENGTH,
    seed: int = c.RANDOM_SEED,
) -> MCSResult:
    """Construct the Hansen–Lunde–Nason MCS using the range statistic.

    losses has shape (dates, models), with columns matching names. Stationary
    bootstrap differences are recentered under equal predictive ability.
    Sequential tests eliminate the largest standardized relative loss until
    non-rejection at alpha; confidence is 1-alpha.
    """
    values = np.asarray(losses, dtype=float)
    if (
        replications < c.MIN_BOOTSTRAP_REPLICATIONS
        or len(values) < c.MIN_EVALUATION_OBSERVATIONS
        or block_length < 1
    ):
        raise ValueError(
            "MCS needs >=99 replications, >=20 observations and positive block length"
        )
    indices = stationary_indices(
        len(values), replications, block_length, np.random.default_rng(seed)
    )
    boot_means = np.stack([values[index].mean(axis=0) for index in indices])
    means = values.mean(axis=0)
    active = list(range(len(names)))
    steps: list[MCSStep] = []
    adjusted = 0.0
    while len(active) > 1:
        observed = means[active, None] - means[None, active]
        boot = boot_means[:, active, None] - boot_means[:, None, active]
        se = np.maximum(boot.std(axis=0, ddof=1), c.STANDARD_ERROR_FLOOR)
        statistic = np.max(np.abs(observed / se))
        simulated = np.max(np.abs((boot - observed) / se), axis=(1, 2))
        p = float((1 + np.sum(simulated >= statistic)) / (replications + 1))
        adjusted = max(adjusted, p)
        worst = int(np.argmax(np.max(observed / se, axis=1)))
        steps.append(
            dict(
                models=[names[i] for i in active],
                p_value=p,
                adjusted_p=adjusted,
                eliminated=names[active[worst]] if p < alpha else None,
            )
        )
        if p >= alpha:
            break
        active.pop(worst)
    return dict(included=[names[i] for i in active], alpha=alpha, steps=steps)
