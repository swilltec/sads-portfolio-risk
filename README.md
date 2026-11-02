# Stress-adaptive dimensionality selection

One-day VaR and expected shortfall forecasting for an equal-weighted portfolio of the 49 French industry portfolios. SADS selects the PCA rank from market stress and past forecast losses, alongside fixed-rank PCA, rolling loss-optimized PCA, rolling Bai–Ng, Ledoit–Wolf, sample variance, EWMA, and Bayesian variance benchmarks.

## Run

Use Python **3.11.7** and the exact dependency versions in `requirements.txt`. Study defaults are in `sads/constants.py`.

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python scripts/download_data.py
python main.py
python -m pytest -q
```

The downloader validates both inputs before replacing existing files. If access is blocked, it exits with a clear error; no substitute data or results are generated. The source files are the [French 49 Industry Portfolios daily CSV ZIP](https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/ftp/49_Industry_Portfolios_daily_CSV.zip) and [FRED STLFSI4 CSV](https://fred.stlouisfed.org/graph/fredgraph.csv?id=STLFSI4), saved as `data/49_Industry_Portfolios_Daily.csv` and `data/STLFSI4.csv`.

Override paths or dates with `python main.py --data-dir data --output results --start 1994-01-01 --end 2025-12-31`. The downloader also accepts `--data-dir`.

For notebook analysis:

```python
from main import PARAMETERS, OUTPUT
from sads.experiment import run

results = run(PARAMETERS, OUTPUT)
results["stlfsi"]["comparisons"]
```

## Method

- **Data:** daily value-weighted industry returns, converted from percent to decimals. Each industry receives portfolio weight 1/49.
- **Estimation:** a 252-day rolling window, candidate ranks 0–10, and PCA covariance with diagonal residual risk. Each forecast uses returns through the preceding trading day.
- **Risk forecasts:** zero conditional mean and a unit-variance Student-t density with df=6; VaR at 1% and 2.5%, and ES at 2.5%.
- **Calibration:** the first 252 completed forecasts select the fixed PCA rank. Evaluation starts after estimation and calibration.
- **SADS:** separate ranks for elevated and lower stress, selected by mean FZ0 over up to 1,260 previous forecasts. Regimes with fewer than 30 observations use the best rank over that historical window.
- **Rolling PCA ablation:** minimum mean FZ0 over the same preceding 1,260 forecasts, without regime information, sharing SADS estimation, calibration, candidates, and evaluation dates.
- **Volatility baselines:** sample variance uses centered equal-weighted portfolio returns over the preceding estimation window with divisor W, matching the PCA sample covariance. RiskMetrics EWMA uses lambda=0.94, initialized with centered variance of the first estimation window, then updated with squared portfolio returns through t-1 on every trading date. Both use the same tail distribution and df as SADS.
- **Bayesian variance (`bayes_var`):** for portfolio returns r in the preceding estimation window, use a zero-mean Gaussian working likelihood and inverse-gamma prior with shape a=2 and scale b=0.0001. The prior density is proportional to v^(-a-1) exp(-b/v), with mean b/(a-1)=0.0001 (1% daily volatility). Posterior shape is a+W/2, scale is b+sum(r²)/2, and the reported variance is their ratio scale/(shape-1). Reset to the same fixed prior for each rolling window; do not repeatedly update overlapping windows. The prior is not fitted to evaluation data. Config fields `bayes_prior_shape` and `bayes_prior_scale` are saved in settings. VaR/ES use this posterior mean through the same configured tail distribution and df as the other baselines; they do not integrate over posterior uncertainty. This is a Bayesian variance estimator, with the Gaussian working likelihood distinct from the default Student-t tail mapping. It enters FZ0, coverage, GW comparisons, and MCS; like the other non-PCA baselines, it has no selected rank and is outside the matched PCA crisis checks.
- **Stress:** available STLFSI4 above its expanding weekly mean. The experiment repeats with 21-day trailing portfolio volatility above its expanding daily mean.
- **Evaluation:** FZ0, Giacomini–White tests with HAC covariance, Kupiec and Christoffersen coverage tests, and a 95% Model Confidence Set. Lower FZ0 and negative SADS-minus-baseline differences favor SADS.
- **Rank analysis:** HAC estimates of elevated-minus-lower-stress rank differences for SADS, Bai–Ng, and the rank minimizing FZ0 over the preceding 252 forecasts.
- **Crisis holdouts:** ranks are fitted on forecasts completed before each of 2008, 2020, and 2022, then held fixed during that episode. The rolling-PCA baseline uses only the last selection-window pre-crisis losses; existing SADS and fixed-rank holdout calibration is unchanged. Separate deletion checks omit each crisis year when comparing the main forecast losses.

STLFSI4 availability uses an assumed seven-calendar-day publication lag, with same-day releases excluded. Supply `release_calendar` with `observation_date,release_date` columns for historical publication dates. The supplied STLFSI4 history contains revised and back-filled values and is **not real-time vintage data**; the volatility-proxy analysis uses trailing returns. Individual comparison p-values are unadjusted.

## Files

`main.py` runs the study using the defaults in `sads/constants.py`. The `sads/` modules contain data alignment, covariance models, rank selection, statistical tests, and the experiment loop. Run the numerical and timing checks with `python -m pytest -q`.

`results/settings.json` records parameters, input hashes, evaluation dates, Python/package versions, git commit, platform, and CPU. Each signal's output directory contains forecast, summary, coverage, comparison, crisis-holdout, and crisis-deletion CSVs; `evaluation.json` contains rank tests, the MCS, and robustness results. `diagnostics.png` plots ranks, tail forecasts, and cumulative PCA-model FZ0. The tables are also returned as pandas objects.

`summary.csv` includes observation count, mean and sample standard deviation of FZ0, HAC standard error and normal 95% confidence interval, regime means, 1%/2.5% violation rates, and mean rank (undefined for non-PCA models). `comparisons.csv` reports SADS-minus-baseline loss differences, normal 95% HAC intervals, and `significant_improvement` when the difference is negative and the reported GW p-value is below 0.05. Stress-instrument rows use the joint GW test p-value; their interval still describes the mean loss difference. Singular or undersized tests retain undefined inference and do not indicate significant improvement.

## References

- [French Data Library](https://mba.tuck.dartmouth.edu/pages/faculty/Ken.French/data_library.html)
- [FRED STLFSI4](https://fred.stlouisfed.org/series/STLFSI4)
- [Patton, Ziegel & Chen (2019), FZ0](https://public.econ.duke.edu/~ap172/Patton_Ziegel_Chen_JoE_2019.pdf)
- [Bai–Ng ICp2, NBER lecture](https://conference.nber.org/confer/2008/si2008/TSE/Lecture11.pdf)
- [Inverse-gamma prior parameterization](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.invgamma.html)
