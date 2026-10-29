from pathlib import Path

import numpy as np
import pandas as pd
from sads.data import align_signals, stress_releases
from sads.selection import adaptive_path, trailing_optimal_path
from sads.experiment import Config, candidate_forecasts, crisis_checks


def test_publication_date_is_strict_and_threshold_weekly(tmp_path: Path) -> None:
    path = tmp_path / "stress.csv"
    path.write_text("observation_date,STLFSI4\n2000-01-07,1\n2000-01-14,3\n")
    releases = stress_releases(path, lag_days=7)
    dates = pd.date_range("2000-01-13", periods=12)
    returns = pd.DataFrame(np.ones((12, 49)), index=dates)
    signals = align_signals(returns, releases, 2)
    assert pd.isna(signals.loc["2000-01-14", "stress"])
    assert signals.loc["2000-01-21", "stress"] == 1
    assert signals.loc["2000-01-22", "stress"] == 3
    assert signals.loc["2000-01-22", "stress_threshold"] == 2


def test_rank_choices_do_not_see_current_or_future_losses() -> None:
    rng = np.random.default_rng(4)
    loss = rng.normal(size=(160, 5))
    regimes = rng.random(160) > 0.5
    selected = adaptive_path(loss, regimes, 30, min_regime=5)
    optimum = trailing_optimal_path(loss, 30)
    perturbed = loss.copy()
    perturbed[100:] += np.array([10000, 3000, -1000, 5, 17])
    np.testing.assert_array_equal(
        selected[:101], adaptive_path(perturbed, regimes, 30, min_regime=5)[:101]
    )
    np.testing.assert_array_equal(
        optimum[:101], trailing_optimal_path(perturbed, 30)[:101]
    )


def test_return_forecasts_are_prefix_invariant() -> None:
    rng = np.random.default_rng(12)
    dates = pd.bdate_range("2000-01-01", periods=100)
    returns = pd.DataFrame(rng.normal(scale=0.01, size=(100, 49)), index=dates)
    releases = pd.DataFrame(
        {
            "observation_date": [pd.Timestamp("1999-01-01")],
            "release_date": [pd.Timestamp("1999-01-08")],
            "STLFSI4": [1.0],
            "stress_threshold": [1.0],
        }
    )
    cfg = Config(window=20, calibration=20, max_rank=3, volatility_window=5)
    signals = align_signals(returns, releases, 5)
    a = candidate_forecasts(returns, signals, cfg)
    changed = returns.copy()
    changed.iloc[70:] *= 100
    b = candidate_forecasts(changed, align_signals(changed, releases, 5), cfg)
    np.testing.assert_allclose(a[1][1][:51], b[1][1][:51])
    np.testing.assert_allclose(a[0].realized_vol.iloc[:51], b[0].realized_vol.iloc[:51])


def test_supplied_release_calendar_controls_availability(tmp_path: Path) -> None:
    stress = tmp_path / "stress.csv"
    calendar = tmp_path / "releases.csv"
    stress.write_text("observation_date,STLFSI4\n2000-01-07,1\n")
    calendar.write_text("observation_date,release_date\n2000-01-07,2000-01-20\n")
    dates = pd.date_range("2000-01-19", periods=3)
    returns = pd.DataFrame(np.zeros((3, 49)), index=dates)
    signals = align_signals(returns, stress_releases(stress, calendar), 2)
    assert pd.isna(signals.loc["2000-01-20", "stress"])
    assert signals.loc["2000-01-21", "stress"] == 1


def test_crisis_holdout_ranks_exclude_episode_and_later_losses() -> None:
    rng = np.random.default_rng(41)
    dates = pd.bdate_range("2007-01-01", "2009-12-31")
    metadata = pd.DataFrame({"bn_rank": np.ones(len(dates), dtype=int)}, index=dates)
    losses = rng.normal(size=(len(dates), 4))
    regimes = np.arange(len(dates)) % 2 == 0
    cfg = Config(calibration=20, min_regime=5)
    original = crisis_checks(metadata, losses, regimes, cfg)
    changed = losses.copy()
    changed[dates >= "2008-01-01"] += [1000, -1000, 500, 800]
    perturbed = crisis_checks(metadata, changed, regimes, cfg)
    columns = ["episode", "baseline", "fixed_rank", "low_rank", "high_rank"]
    pd.testing.assert_frame_equal(original[columns], perturbed[columns])
    assert original.procedure.eq("pre_crisis_holdout").all()
