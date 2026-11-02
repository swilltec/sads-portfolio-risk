"""Factor covariance estimators and portfolio tail forecasts."""

from collections.abc import Iterable

import numpy as np
from numpy.typing import ArrayLike
from scipy.stats import norm, t
from sklearn.covariance import LedoitWolf

from . import constants as c
from .types import Distribution, FloatArray, TailForecast


def pca_path(x: FloatArray, ranks: Iterable[int]) -> tuple[FloatArray, int, float]:
    """Estimate equal-weight portfolio variance for each candidate PCA rank.

    x is an observations-by-assets return matrix. Each covariance retains the
    diagonal residual variance. Return candidate variances, the column index
    minimizing Bai–Ng ICp2, and the leading eigenvalue's variance share.
    """
    n_obs, n_assets = x.shape
    centered = x - x.mean(axis=0)
    covariance = centered.T @ centered / n_obs
    eigenvalues, eigenvectors = np.linalg.eigh(covariance)
    eigenvalues, eigenvectors = eigenvalues[::-1].clip(0), eigenvectors[:, ::-1]
    weights = np.full(n_assets, 1 / n_assets)
    common_variance = np.cumsum(eigenvalues * (eigenvectors.T @ weights) ** 2)
    diagonal_variance = np.cumsum(
        eigenvalues * ((eigenvectors**2).T @ (weights**2))
    )
    base = np.diag(covariance) @ (weights**2)
    variances, criterion = [], []
    penalty = (n_assets + n_obs) / (n_assets * n_obs) * np.log(min(n_assets, n_obs))
    for rank in ranks:
        if not 0 <= rank < min(n_obs, n_assets):
            raise ValueError("Candidate ranks must be below min(window, assets)")
        # Replace the retained components' diagonal risk with their full covariance.
        contribution = (
            common_variance[rank - 1] - diagonal_variance[rank - 1] if rank else 0
        )
        variances.append(base + contribution)
        residual = eigenvalues[rank:].sum() / n_assets
        criterion.append(np.log(max(residual, np.finfo(float).tiny)) + rank * penalty)
    return (
        np.maximum(variances, c.VARIANCE_FLOOR),
        int(np.argmin(criterion)),
        float(eigenvalues[0] / eigenvalues.sum()),
    )


def shrinkage_variance(x: FloatArray) -> float:
    """Estimate equal-weight portfolio variance with Ledoit–Wolf shrinkage."""
    covariance = LedoitWolf().fit(x).covariance_
    return float(covariance.sum() / x.shape[1] ** 2)


def sample_variance(x: FloatArray) -> float:
    """Centered equal-weight portfolio variance, using the PCA divisor W."""
    return float(np.var(x.mean(axis=1), ddof=0))


def bayesian_variance(
    x: FloatArray,
    prior_shape: float = c.BAYES_PRIOR_SHAPE,
    prior_scale: float = c.BAYES_PRIOR_SCALE,
) -> float:
    """Posterior mean variance under a zero-mean Gaussian working likelihood.

    Use an inverse-gamma prior with density proportional to
    v**(-shape-1) * exp(-scale/v). Defaults have prior mean variance 0.0001
    (1% daily volatility), fixed independently of observed returns. Each call
    updates this same prior from the supplied estimation window only.
    """
    if (
        not np.isfinite(prior_shape) or prior_shape <= 1
        or not np.isfinite(prior_scale) or prior_scale <= 0
    ):
        raise ValueError("Bayesian prior requires finite shape > 1 and scale > 0")
    portfolio = x.mean(axis=1)
    shape = prior_shape + len(portfolio) / 2
    scale = prior_scale + float(portfolio @ portfolio) / 2
    return scale / (shape - 1)


def ewma_variance_path(
    portfolio: FloatArray, window: int, decay: float = c.EWMA_DECAY
) -> FloatArray:
    """Return variance forecasts at t using only returns through t-1.

    Initialize at t=window with centered first-window variance. Earlier entries
    are undefined; subsequent updates use squared returns under zero mean.
    """
    values = np.asarray(portfolio, dtype=float)
    if values.ndim != 1 or not 1 <= window < len(values) or not 0 < decay < 1:
        raise ValueError("EWMA requires a vector, 1 <= window < n, and 0 < decay < 1")
    variances = np.full(len(values), np.nan)
    variances[window] = np.var(values[:window], ddof=0)
    for i in range(window + 1, len(values)):
        variances[i] = decay * variances[i - 1] + (1 - decay) * values[i - 1] ** 2
    return variances


def tail_forecast(
    variance: ArrayLike,
    distribution: Distribution = c.RETURN_DISTRIBUTION,
    df: float = c.STUDENT_DF,
) -> TailForecast:
    """Map variance to 1% VaR, 2.5% VaR, and 2.5% ES under zero mean.

    Outputs retain the input shape, with negative values for lower return
    tails. Student-t quantiles are rescaled to unit variance; scalar inputs
    return zero-dimensional arrays.
    """
    scale = np.sqrt(np.maximum(variance, c.VARIANCE_FLOOR))
    if distribution == c.NORMAL_DISTRIBUTION:
        q01, q025 = norm.ppf(c.VAR_ALPHA), norm.ppf(c.ES_ALPHA)
        es = -norm.pdf(q025) / c.ES_ALPHA
    elif distribution == c.STUDENT_DISTRIBUTION and df > 2:
        adjustment = np.sqrt((df - 2) / df)
        q = t.ppf(c.ES_ALPHA, df)
        q01, q025 = t.ppf(c.VAR_ALPHA, df) * adjustment, q * adjustment
        es = -(df + q * q) / (df - 1) * t.pdf(q, df) / c.ES_ALPHA * adjustment
    else:
        raise ValueError("Use normal or student with df > 2")
    return np.asarray(scale * q01), np.asarray(scale * q025), np.asarray(scale * es)
