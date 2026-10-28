import numpy as np
from scipy.stats import norm
from sads.models import pca_path, tail_forecast
from sads.statistics import fz0


def test_pca_variance_matches_explicit_factor_plus_residual_covariance() -> None:
    rng = np.random.default_rng(12)
    x = rng.normal(size=(150, 8)) @ rng.normal(size=(8, 8))
    variances, _, _ = pca_path(x, [0, 1, 4, 7])
    covariance = np.cov(x, rowvar=False, ddof=0)
    eigenvalues, vectors = np.linalg.eigh(covariance)
    eigenvalues, vectors = eigenvalues[::-1], vectors[:, ::-1]
    weights = np.ones(8) / 8
    for actual, rank in zip(variances, [0, 1, 4, 7]):
        common = (vectors[:, :rank] * eigenvalues[:rank]) @ vectors[:, :rank].T
        estimate = common + np.diag(np.diag(covariance - common))
        np.testing.assert_allclose(actual, weights @ estimate @ weights, rtol=1e-12)


def test_bai_ng_recovers_strong_factor_on_synthetic_panel() -> None:
    rng = np.random.default_rng(32)
    x = rng.normal(size=(1000, 1)) @ np.ones((1, 49)) + rng.normal(
        scale=0.1, size=(1000, 49)
    )
    _, rank, _ = pca_path(x, list(range(11)))
    assert rank == 1


def test_normal_tails_and_student_ordering() -> None:
    v01, v025, es = tail_forecast(0.04, "normal")
    np.testing.assert_allclose(v025, 0.2 * norm.ppf(0.025))
    assert es < v025 < 0 and v01 < v025
    v01, v025, es = tail_forecast(0.04, "student", 6)
    assert es < v025 < 0 and v01 < v025


def test_fz0_rewards_true_distribution_in_expectation() -> None:
    rng = np.random.default_rng(7)
    y = rng.normal(0, 0.01, 300000)
    _, q, es = tail_forecast(0.0001, "normal")
    true = fz0(y, q, es).mean()
    assert true < fz0(y, 0.5 * q, 0.5 * es).mean()
    assert true < fz0(y, 2 * q, 2 * es).mean()
