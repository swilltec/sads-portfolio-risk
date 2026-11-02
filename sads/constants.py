from pathlib import Path
from typing import Final

from .types import Distribution, StressSignal

PROJECT_DIR: Final = Path(__file__).resolve().parent.parent
DATA_DIR: Final = PROJECT_DIR / "data"
OUTPUT_DIR: Final = PROJECT_DIR / "results"
RETURNS_PATH: Final = DATA_DIR / "49_Industry_Portfolios_Daily.csv"
STRESS_PATH: Final = DATA_DIR / "STLFSI4.csv"
RELEASE_CALENDAR: Final[Path | None] = None

SAMPLE_START: Final = "1994-01-01"
SAMPLE_END: Final[str | None] = None
ESTIMATION_WINDOW: Final = 252
CALIBRATION_WINDOW: Final = 252
MAX_RANK: Final = 10
SELECTION_WINDOW: Final = 1260
MIN_REGIME_OBSERVATIONS: Final = 30
VOLATILITY_WINDOW: Final = 21
RANK_LOOKBACK: Final = 252
PUBLICATION_LAG_DAYS: Final = 7
NORMAL_DISTRIBUTION: Final[Distribution] = "normal"
STUDENT_DISTRIBUTION: Final[Distribution] = "student"
RETURN_DISTRIBUTION: Final = STUDENT_DISTRIBUTION
STUDENT_DF: Final = 6.0
EWMA_DECAY: Final = 0.94
BAYES_PRIOR_SHAPE: Final = 2.0
BAYES_PRIOR_SCALE: Final = 0.0001
BOOTSTRAP_REPLICATIONS: Final = 999
BOOTSTRAP_BLOCK_LENGTH: Final = 20
HAC_LAGS: Final[int | None] = None
RANDOM_SEED: Final = 42
SIGNIFICANCE_LEVEL: Final = 0.05
VAR_ALPHA: Final = 0.01
ES_ALPHA: Final = 0.025

N_INDUSTRIES: Final = 49
FRENCH_VALUE_WEIGHTED_HEADER: Final = "Average Value Weighted Returns"
FRENCH_DATE_PATTERN: Final = r"^\s*\d{8},"
FRENCH_DATE_FORMAT: Final = "%Y%m%d"
FRENCH_MISSING_RETURNS: Final = (-99.99, -999.0)
PERCENT_SCALE: Final = 100
STRESS_SERIES: Final = "STLFSI4"
STRESS_MISSING_VALUE: Final = "."
STLFSI_SIGNAL: Final[StressSignal] = "stlfsi"
VOLATILITY_SIGNAL: Final[StressSignal] = "realized_vol"
STRESS_SIGNALS: Final = (STLFSI_SIGNAL, VOLATILITY_SIGNAL)
SIGNAL_COLUMNS: Final = ("stress", "stress_threshold", "realized_vol", "vol_threshold")

MIN_WINDOW: Final = 20
MIN_EVALUATION_OBSERVATIONS: Final = 20
MIN_TEST_OBSERVATIONS: Final = 10
MIN_VOLATILITY_WINDOW: Final = 2
MIN_BOOTSTRAP_REPLICATIONS: Final = 99
VARIANCE_FLOOR: Final = 1e-16
STANDARD_ERROR_FLOOR: Final = 1e-12
ZERO_EFFECT_TOLERANCE: Final = 1e-12
HAC_BANDWIDTH_SCALE: Final = 4
HAC_REFERENCE_SIZE: Final = 100
HAC_BANDWIDTH_EXPONENT: Final = 2 / 9
NUMERICAL_THREADS: Final = 1

FIXED_PCA: Final = "fixed_pca"
BAI_NG: Final = "bai_ng"
SADS: Final = "sads"
LEDOIT_WOLF: Final = "ledoit_wolf"
ROLLING_PCA: Final = "rolling_pca"
SAMPLE_VAR: Final = "sample_var"
EWMA: Final = "ewma"
BAYES_VAR: Final = "bayes_var"
MATCHED_BASELINES: Final = (FIXED_PCA, BAI_NG, ROLLING_PCA)
PCA_MODELS: Final = (SADS, FIXED_PCA, BAI_NG, ROLLING_PCA)
PACKAGE_NAMES: Final = (
    "numpy", "pandas", "scipy", "scikit-learn", "matplotlib", "threadpoolctl", "pytest"
)
RISK_COLUMNS: Final = ("var_01", "var_025", "es_025")
VAR_COVERAGE_LEVELS: Final = ((VAR_ALPHA, RISK_COLUMNS[0]), (ES_ALPHA, RISK_COLUMNS[1]))
RANK_DIAGNOSTICS: Final = (
    ("trailing_loss_optimal", "trailing_optimal_rank"),
    (BAI_NG, "bn_rank"),
    (SADS, "rank"),
)
FULL_SAMPLE: Final = "all"
ELEVATED_STRESS: Final = "elevated"
CRISIS_HOLDOUT_PROCEDURE: Final = "pre_crisis_holdout"
NO_FULL_SAMPLE_IMPROVEMENT: Final = "no_significant_full_sample_improvement"
INCOMPLETE_ROBUSTNESS: Final = "robustness_not_fully_evaluable"
ROBUST_IMPROVEMENT: Final = "robust_across_predeclared_checks"
NOT_ROBUST: Final = "not_robust"
TIMING_WITH_CALENDAR: Final = "provided_calendar"
TIMING_WITH_LAG: Final = "assumed_calendar_lag"
STUDY_INTERPRETATION: Final = "pseudo-real-time"
GW_CONSTANT: Final = "GW_constant"
GW_STRESS_INSTRUMENT: Final = "GW_stress_instrument"
GW_COLUMNS: Final = ("n", "mean_difference", "se", "statistic", "p_value")
CRISES: Final = {
    "gfc_2008": ("2008-01-01", "2008-12-31"),
    "covid_2020": ("2020-01-01", "2020-12-31"),
    "rates_2022": ("2022-01-01", "2022-12-31"),
}
HOLDOUT_COLUMNS: Final = (
    "episode",
    "baseline",
    "procedure",
    "fixed_rank",
    "rolling_rank",
    "low_rank",
    "high_rank",
    *GW_COLUMNS,
)
DELETION_COLUMNS: Final = (
    "excluded_episode",
    "baseline",
    "significant_improvement",
    *GW_COLUMNS,
)

SETTINGS_FILENAME: Final = "settings.json"
EVALUATION_FILENAME: Final = "evaluation.json"
DIAGNOSTICS_FILENAME: Final = "diagnostics.png"
JSON_INDENT: Final = 2
FIGURE_SIZE: Final = (12, 9)
FIGURE_DPI: Final = 160
RANK_LINE_WIDTH: Final = 0.8
SCORE_LINE_WIDTH: Final = 0.9
LEGEND_COLUMNS: Final = 3
TAIL_PLOT_SERIES: Final = (
    ("return", "portfolio return", 0.5, 0.5),
    (RISK_COLUMNS[1], "2.5% VaR", 0.7, 1.0),
    (RISK_COLUMNS[2], "2.5% ES", 0.7, 1.0),
)
RANK_AXIS_LABEL: Final = "Retained rank"
RETURN_AXIS_LABEL: Final = "Daily return"
SCORE_AXIS_LABEL: Final = "Cumulative FZ0 (lower is better)"
