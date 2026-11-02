import numpy as np
import pandas as pd
import pytest

from sads import constants as c
from sads.experiment import Config, candidate_forecasts, make_forecast_frame
from sads.models import bayesian_variance, tail_forecast


def test_bayesian_variance_matches_hand_computed_posterior() -> None:
    portfolio = np.array([.01, -.02, .03])
    x = np.repeat(portfolio[:, None], 49, axis=1)
    # IG(2, .0001) -> IG(3.5, .0008); mean = .0008 / 2.5.
    np.testing.assert_allclose(bayesian_variance(x), .00032)
    # The zero-mean likelihood uses raw squared returns, not centered variance.
    constant = np.full((2, 49), .01)
    np.testing.assert_allclose(bayesian_variance(constant), .0001)
    assert bayesian_variance(np.zeros((20, 49))) > 0


@pytest.mark.parametrize("shape,scale", [(1, .001), (2, 0), (np.nan, .001), (2, np.inf)])
def test_bayesian_prior_rejects_invalid_parameters(shape, scale) -> None:
    with pytest.raises(ValueError, match="Bayesian prior"):
        bayesian_variance(np.zeros((20, 49)), shape, scale)
    with pytest.raises(ValueError, match="Bayesian prior"):
        Config(bayes_prior_shape=shape, bayes_prior_scale=scale).validate()


@pytest.mark.parametrize("distribution", [c.NORMAL_DISTRIBUTION, c.STUDENT_DISTRIBUTION])
def test_bayesian_forecasts_use_only_prior_window_and_shared_tails(distribution) -> None:
    rng = np.random.default_rng(97)
    dates = pd.bdate_range("2000-01-03", periods=100, name="date")
    returns = pd.DataFrame(rng.normal(scale=.01, size=(100, 49)), index=dates)
    signals = pd.DataFrame(1., index=dates, columns=c.SIGNAL_COLUMNS)
    signals.loc[dates[25:29], "stress"] = np.nan
    cfg = Config(
        window=20, calibration=20, max_rank=3, distribution=distribution,
        bayes_prior_shape=3., bayes_prior_scale=.0002,
    )
    candidates = candidate_forecasts(returns, signals, cfg)
    forecasts, _ = make_forecast_frame(
        candidates.metadata, candidates.risk, candidates.loss, candidates.shrinkage,
        cfg, c.STLFSI_SIGNAL, candidates.sample, candidates.ewma, candidates.bayes,
    )
    portfolio = returns.mean(axis=1).to_numpy()
    positions = dates.get_indexer(candidates.metadata.index)
    expected = np.array([
        (.0002 + .5 * np.sum(portfolio[i-20:i] ** 2)) / (3 + 20/2 - 1)
        for i in positions
    ])
    np.testing.assert_allclose(candidates.bayes, expected)
    bayes = forecasts[forecasts.model == c.BAYES_VAR]
    assert bayes["rank"].isna().all()
    tails = tail_forecast(expected[cfg.calibration:], distribution, cfg.student_df)
    for column, values in zip(c.RISK_COLUMNS, tails):
        np.testing.assert_allclose(bayes[column], values)
    # A return changed at t must leave the forecast at t (and earlier) intact.
    changed = returns.copy()
    changed.iloc[70:] *= 100
    perturbed = candidate_forecasts(changed, signals, cfg)
    before = candidates.metadata.index <= dates[70]
    np.testing.assert_allclose(candidates.bayes[before], perturbed.bayes[before])
    assert not np.allclose(candidates.bayes[~before], perturbed.bayes[~before])
