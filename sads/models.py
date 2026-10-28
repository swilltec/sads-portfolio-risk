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
