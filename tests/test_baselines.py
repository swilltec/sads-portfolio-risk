import numpy as np
import pandas as pd

from sads import constants as c
from sads.experiment import Config, candidate_forecasts, crisis_checks, make_forecast_frame
from sads.models import ewma_variance_path, sample_variance, tail_forecast
from sads.statistics import fz0


def test_sample_variance_matches_full_sample_covariance() -> None:
    x = np.random.default_rng(81).normal(size=(60, 49))
    weights = np.full(49, 1 / 49)
    expected = weights @ np.cov(x, rowvar=False, ddof=0) @ weights
    np.testing.assert_allclose(sample_variance(x), expected, rtol=1e-12)


def test_ewma_matches_hand_recursion_and_excludes_current_return() -> None:
    portfolio = np.array([1., 3., 2., -4., 5., 6.])
    # Initial centered variance is 1; first update uses return[2] = 2.
    expected = [np.nan, np.nan, 1., 1.18, 2.0692, 3.445048]
    np.testing.assert_allclose(ewma_variance_path(portfolio, 2), expected)
    changed = portfolio.copy()
    changed[3:] *= 100
    np.testing.assert_allclose(
        ewma_variance_path(portfolio, 2)[:4], ewma_variance_path(changed, 2)[:4]
    )


def synthetic_candidates():
    rng = np.random.default_rng(51)
    n, ranks = 100, 4
    index = pd.bdate_range("2000-01-03", periods=n, name="date")
    metadata = pd.DataFrame({
        "stress": np.ones(n), "stress_threshold": np.zeros(n),
        "realized_vol": np.ones(n), "vol_threshold": np.zeros(n),
        "return": rng.normal(0, .01, n), "bn_rank": np.zeros(n, dtype=int),
    }, index=index)
    risk = tail_forecast(rng.uniform(.00005, .0003, (n, ranks)))
    loss = fz0(metadata["return"].to_numpy()[:, None], risk[1], risk[2])
    variance = np.full(n, .0001)
    cfg = Config(calibration=20, selection_window=25, min_regime=5)
    return metadata, risk, loss, variance, cfg


def test_rolling_pca_model_uses_prior_losses_and_equals_single_regime_sads() -> None:
    metadata, risk, loss, variance, cfg = synthetic_candidates()
    original, _ = make_forecast_frame(
        metadata, risk, loss, variance, cfg, c.STLFSI_SIGNAL, variance, variance, variance
    )
    rolling = original[original.model == c.ROLLING_PCA].reset_index(drop=True)
    sads = original[original.model == c.SADS].reset_index(drop=True)
    np.testing.assert_array_equal(rolling["rank"], sads["rank"])
    np.testing.assert_allclose(rolling[list(c.RISK_COLUMNS)], sads[list(c.RISK_COLUMNS)])
    for i, rank in enumerate(rolling["rank"], start=cfg.calibration):
        assert rank == np.argmin(loss[max(0, i - cfg.selection_window):i].mean(axis=0))
    changed = loss.copy()
    changed[60:] += [1000, -1000, 200, 50]
    perturbed, _ = make_forecast_frame(
        metadata, risk, changed, variance, cfg, c.STLFSI_SIGNAL, variance, variance, variance
    )
    selected = perturbed[perturbed.model == c.ROLLING_PCA].reset_index(drop=True)
    np.testing.assert_array_equal(rolling["rank"][:41], selected["rank"][:41])


def test_variance_baselines_align_across_missing_signal_dates() -> None:
    rng = np.random.default_rng(12)
    dates = pd.bdate_range("2000-01-01", periods=100, name="date")
    returns = pd.DataFrame(rng.normal(scale=.01, size=(100, 49)), index=dates)
    signals = pd.DataFrame(1., index=dates, columns=c.SIGNAL_COLUMNS)
    signals.loc[dates[21:25], "stress"] = np.nan
    cfg = Config(window=20, calibration=20, max_rank=3)
    candidates = candidate_forecasts(returns, signals, cfg)
    expected = ewma_variance_path(returns.mean(axis=1).to_numpy(), cfg.window)
    rows = dates.get_indexer(candidates.metadata.index)
    np.testing.assert_allclose(candidates.ewma, expected[rows])
    np.testing.assert_allclose(candidates.sample, [
        sample_variance(returns.iloc[i-20:i].to_numpy()) for i in rows
    ])
    changed = returns.copy()
    changed.iloc[70:] *= 100
    perturbed = candidate_forecasts(changed, signals, cfg)
    np.testing.assert_allclose(
        candidates.ewma[candidates.metadata.index <= dates[70]],
        perturbed.ewma[perturbed.metadata.index <= dates[70]],
    )


def test_rolling_holdout_rank_uses_only_trailing_precrisis_losses() -> None:
    dates = pd.bdate_range("2007-01-01", "2008-12-31", name="date")
    metadata = pd.DataFrame({"bn_rank": np.zeros(len(dates), dtype=int)}, index=dates)
    loss = np.zeros((len(dates), 3))
    loss[:, 0] = -5
    pre = np.flatnonzero(dates < "2008-01-01")
    loss[pre[-20:], 1] = -10
    cfg = Config(calibration=20, selection_window=20)
    regimes = np.zeros(len(dates), dtype=bool)
    original = crisis_checks(metadata, loss, regimes, cfg)
    row = original[original.baseline == c.ROLLING_PCA].iloc[0]
    assert row.rolling_rank == 1 and row.fixed_rank == 0
    changed = loss.copy()
    changed[dates >= "2008-01-01", 2] = -1000
    perturbed = crisis_checks(metadata, changed, regimes, cfg)
    np.testing.assert_array_equal(original.rolling_rank, perturbed.rolling_rank)
    assert set(original.baseline) == set(c.MATCHED_BASELINES)
