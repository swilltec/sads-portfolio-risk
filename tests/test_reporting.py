import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import norm

from sads import constants as c
from sads.experiment import Config, evaluate, run
from sads.statistics import hac_covariance_mean

MODELS = (*c.PCA_MODELS, c.LEDOIT_WOLF, c.SAMPLE_VAR, c.EWMA, c.BAYES_VAR)


def reporting_forecasts() -> pd.DataFrame:
    rng = np.random.default_rng(72)
    n = 120
    dates = pd.bdate_range("2000-01-03", periods=n)
    losses = rng.normal(size=n)
    parts = []
    for model in MODELS:
        parts.append(pd.DataFrame({
            "date": dates, "model": model,
            "fz0": losses + (rng.normal(2, .1, n) if model == c.SAMPLE_VAR else 0),
            "elevated": np.arange(n) % 2 == 0,
            "rank": 2 if model in c.PCA_MODELS else np.nan,
            "return": rng.normal(0, .01, n),
            "var_01": -.023, "var_025": -.02, "es_025": -.03,
            "trailing_optimal_rank": 1, "bn_rank": 1,
        }))
    return pd.concat(parts, ignore_index=True)


def test_summary_and_comparisons_on_synthetic_losses() -> None:
    forecasts = reporting_forecasts()
    cfg = Config(bootstrap_replications=99, hac_lags=2)
    summary, coverage, comparisons, _, mcs = evaluate(forecasts, cfg)
    for row in summary.itertuples():
        group = forecasts[forecasts.model == row.model]
        assert row.n == 120
        np.testing.assert_allclose(row.std_fz0, np.std(group.fz0, ddof=1))
        se = np.sqrt(hac_covariance_mean(group.fz0, cfg.hac_lags)[0, 0])
        np.testing.assert_allclose(row.hac_se_fz0, se)
        np.testing.assert_allclose(row.hac_ci_lower, group.fz0.mean() - norm.ppf(.975)*se)
        np.testing.assert_allclose(row.hac_ci_upper, group.fz0.mean() + norm.ppf(.975)*se)
        assert row.hac_ci_lower <= row.mean_fz0 <= row.hac_ci_upper
        assert np.isnan(row.mean_rank) if row.model not in c.PCA_MODELS else row.mean_rank == 2
        for alpha, rate in ((.01, row.violation_rate_01), (.025, row.violation_rate_025)):
            expected = coverage[(coverage.model == row.model) & (coverage.alpha == alpha)]
            assert rate == expected.iloc[0].violation_rate
    full = comparisons[(comparisons.regime == c.FULL_SAMPLE) & (comparisons.test == c.GW_CONSTANT)].set_index("model_b")
    assert bool(full.loc[c.SAMPLE_VAR, "significant_improvement"])
    assert not bool(full.loc[c.FIXED_PCA, "significant_improvement"])
    assert np.isnan(full.loc[c.FIXED_PCA, "p_value"])
    np.testing.assert_allclose(comparisons.ci_lower, comparisons.mean_difference - norm.ppf(.975)*comparisons.se)
    np.testing.assert_allclose(comparisons.ci_upper, comparisons.mean_difference + norm.ppf(.975)*comparisons.se)
    assert set(mcs["steps"][0]["models"]) == set(MODELS)


def test_run_writes_all_outputs_for_every_model(tmp_path: Path) -> None:
    rng = np.random.default_rng(82)
    dates = pd.bdate_range("2007-09-03", periods=220)
    names = ",".join(f"industry{i}" for i in range(49))
    rows = "\n".join(
        date.strftime("%Y%m%d") + "," + ",".join(map(str, values))
        for date, values in zip(dates, rng.normal(0, 1, (len(dates), 49)))
    )
    returns_path = tmp_path / c.RETURNS_PATH.name
    returns_path.write_text(f"Preamble\nAverage Value Weighted Returns -- Daily\n,{names}\n{rows}\n\n")
    stress_path = tmp_path / c.STRESS_PATH.name
    weeks = pd.date_range("2007-08-03", dates[-1], freq="W-FRI")
    pd.DataFrame({"observation_date": weeks, "STLFSI4": rng.normal(size=len(weeks))}).to_csv(stress_path, index=False)
    cfg = Config(
        returns_path=returns_path, stress_path=stress_path, window=20, calibration=20,
        max_rank=3, selection_window=25, min_regime=5, volatility_window=5,
        bootstrap_replications=99,
    )
    output = tmp_path / "results"
    results = run(cfg, output)
    settings = json.loads((output / c.SETTINGS_FILENAME).read_text())
    assert settings["parameters"]["bayes_prior_shape"] == cfg.bayes_prior_shape
    assert settings["parameters"]["bayes_prior_scale"] == cfg.bayes_prior_scale
    environment = settings["environment"]
    assert set(environment) == {"python", "packages", "git_commit", "platform", "cpu"}
    assert set(environment["packages"]) == set(c.PACKAGE_NAMES)
    for signal in c.STRESS_SIGNALS:
        folder = output / signal
        assert {p.name for p in folder.iterdir()} == {
            "forecasts.csv", "summary.csv", "coverage.csv", "comparisons.csv",
            "crisis_holdout.csv", "crisis_deletion.csv", c.EVALUATION_FILENAME,
            c.DIAGNOSTICS_FILENAME,
        }
        for name in ("forecasts", "summary", "coverage"):
            frame = pd.read_csv(folder / f"{name}.csv")
            assert set(frame.model) == set(MODELS)
        forecasts = results[signal]["forecasts"]
        assert forecasts.groupby("model").date.apply(list).map(tuple).nunique() == 1
        assert forecasts.loc[~forecasts.model.isin(c.PCA_MODELS), "rank"].isna().all()
        comparisons = pd.read_csv(folder / "comparisons.csv")
        assert set(comparisons.model_b) == set(MODELS) - {c.SADS}
        for name in ("crisis_holdout", "crisis_deletion"):
            assert set(pd.read_csv(folder / f"{name}.csv").baseline) == set(c.MATCHED_BASELINES)
        evaluation = json.loads((folder / c.EVALUATION_FILENAME).read_text())
        assert set(evaluation["model_confidence_set"]["steps"][0]["models"]) == set(MODELS)
        assert set(evaluation["robustness"]) == set(c.MATCHED_BASELINES)
