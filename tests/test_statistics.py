import numpy as np
import pandas as pd

from sads.experiment import robustness_summary
from sads import constants as c
from sads.statistics import coverage, gw_test, model_confidence_set, rank_regime_test


def test_coverage_handles_no_violations_without_inventing_independence() -> None:
    result = coverage(np.zeros(200), np.full(200, -1), 0.01)
    assert np.isfinite(result["kupiec_p"])
    assert np.isnan(result["independence_p"])


def test_gw_detects_difference_and_reports_direction() -> None:
    difference = np.random.default_rng(22).normal(-0.4, 0.2, 1000)
    result = gw_test(difference)
    assert result["mean_difference"] < -0.35 and result["p_value"] < 0.001
    assert np.isnan(gw_test(np.zeros(100))["p_value"])


def test_rank_regime_effect_has_correct_sign() -> None:
    elevated = np.tile([False, True], 100)
    ranks = np.random.default_rng(1).normal(size=200) + elevated * 3
    result = rank_regime_test(ranks, elevated)
    assert result["high_minus_low"] > 2 and result["p_value"] < 0.001


def test_mcs_eliminates_clearly_inferior_model_reproducibly() -> None:
    rng = np.random.default_rng(4)
    losses = np.column_stack([rng.normal(size=300), rng.normal(5, 1, 300)])
    result = model_confidence_set(losses, ["good", "bad"], replications=199)
    assert result["included"] == ["good"]
    assert result == model_confidence_set(losses, ["good", "bad"], replications=199)


def test_robustness_requires_every_crisis_check_to_pass() -> None:
    comparisons = pd.DataFrame(
        [
            dict(
                model_b=b,
                regime="all",
                test="GW_constant",
                mean_difference=-0.1,
                p_value=0.01,
            )
            for b in c.MATCHED_BASELINES
        ]
    )
    holdout = pd.DataFrame(
        [
            dict(baseline=b, mean_difference=-0.1, p_value=p)
            for b in c.MATCHED_BASELINES
            for p in (0.01, 0.01, 0.3)
        ]
    )
    deletion = pd.DataFrame(
        [
            dict(baseline=b, significant_improvement=True)
            for b in c.MATCHED_BASELINES
            for _ in range(3)
        ]
    )
    result = robustness_summary(comparisons, holdout, deletion)
    assert result["fixed_pca"]["conclusion"] == "not_robust"
    assert result["bai_ng"]["conclusion"] == "not_robust"
    assert result[c.ROLLING_PCA]["conclusion"] == "not_robust"
