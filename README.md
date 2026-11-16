# Stress-adaptive dimensionality selection

One-day VaR and expected shortfall forecasting for an equal-weighted portfolio of the 49 French industry portfolios. SADS selects the PCA rank from market stress and past forecast losses, alongside fixed-rank PCA, rolling loss-optimized PCA, rolling Bai–Ng, Ledoit–Wolf, sample variance, EWMA, and Bayesian variance benchmarks.

## Run

Use a Python version compatible with the dependencies in `requirements.txt` and study defaults are in `sads/constants.py`.

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python scripts/download_data.py
python main.py
python -m pytest -q # To run tests (Optional)
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

## Current results

Snapshot from the saved run on **2026-11-15** (America/New_York), using source commit `4a825d8`. This run used a 252-day estimation window, 252 calibration forecasts, PCA ranks 0–10, a 1,260-forecast selection window, and Student-t tails with df=6. MCS uses 999 stationary-bootstrap replications, mean block length 20, and seed 42.

The tables below summarize the saved [STLFSI4 results](results/stlfsi/summary.csv) and [volatility-signal results](results/realized_vol/summary.csv). Lower mean FZ0 is better. Confidence intervals are normal 95% HAC intervals for each model's mean loss; paired comparisons below test loss differences directly. Only SADS forecasts depend on the stress signal, so each other model appears once.

| Model | Mean FZ0 | 95% HAC CI | 1% VaR violations | 2.5% VaR violations |
| --- | ---: | --- | ---: | ---: |
| EWMA | -3.579723 | [-3.662325, -3.497121] | 1.486% | 3.399% |
| SADS (STLFSI4) | -3.406841 | [-3.546601, -3.267082] | 1.629% | 3.244% |
| SADS (trailing volatility) | -3.407967 | [-3.547701, -3.268233] | 1.629% | 3.257% |
| Bayesian variance | -3.406816 | [-3.546894, -3.266737] | 1.616% | 3.283% |
| Bai–Ng PCA | -3.406627 | [-3.546415, -3.266839] | 1.629% | 3.257% |
| Rolling loss-optimized PCA | -3.406597 | [-3.546444, -3.266751] | 1.629% | 3.244% |
| Fixed PCA | -3.406115 | [-3.546088, -3.266143] | 1.629% | 3.257% |
| Sample variance | -3.403905 | [-3.544907, -3.262903] | 1.641% | 3.296% |
| Ledoit–Wolf | -3.394132 | [-3.540981, -3.247282] | 1.732% | 3.425% |

Full-sample constant-instrument Giacomini–White comparisons use **SADS minus baseline**: negative differences favor SADS. The p-values below are two-sided HAC tests and are unadjusted for multiple comparisons. Stress-instrument joint tests are reported separately in the CSVs and do not by themselves establish a significant mean-loss improvement.

| Baseline | STLFSI4 loss difference | p-value | Volatility-signal loss difference | p-value |
| --- | ---: | ---: | ---: | ---: |
| EWMA | +0.172882 | 0.0004183 | +0.171756 | 0.0004591 |
| Bayesian variance | -0.000026 | 0.9818 | -0.001151 | 0.2528 |
| Bai–Ng PCA | -0.000214 | 0.7887 | -0.001340 | 0.001994 |
| Rolling loss-optimized PCA | -0.000244 | 0.2911 | -0.001369 | 0.09735 |
| Fixed PCA | -0.000726 | 0.3528 | -0.001851 | 4.512e-05 |
| Sample variance | -0.002936 | 0.006565 | -0.004062 | 8.988e-06 |
| Ledoit–Wolf | -0.012710 | 0.001837 | -0.013835 | 0.0006317 |

- **EWMA has the lowest observed mean FZ0** and is the sole survivor of the 95% MCS for both signals. The paired full-sample tests favor EWMA over SADS (p<0.001 for both signals).
- **Stress conditioning has limited evidence of an incremental gain:** SADS does not significantly outperform rolling loss-optimized PCA at 5% for either signal (p=0.2911 and 0.09735). Its difference from Bayesian variance is also insignificant (p=0.9818 and 0.2528).
- SADS beats Ledoit–Wolf and sample variance on the full-sample tests for both signals. The volatility-signal version also beats fixed PCA and Bai–Ng, but **no matched PCA comparison passes all predeclared crisis holdout/deletion checks**. See the saved [STLFSI4 evaluation](results/stlfsi/evaluation.json) and [volatility-signal evaluation](results/realized_vol/evaluation.json).
- **Coverage remains inadequate:** every model exceeds the nominal 1% and 2.5% violation rates, and all Kupiec and conditional-coverage tests reject at 5% in both signal runs. EWMA's better relative loss therefore does not establish adequate absolute VaR calibration. See the [STLFSI4 coverage tests](results/stlfsi/coverage.csv) and [volatility-signal coverage tests](results/realized_vol/coverage.csv).

**Provenance and limitations:** these are pseudo-real-time results with an assumed seven-calendar-day STLFSI4 release lag. STLFSI4 history is revised/back-filled, not real-time vintages. The saved run used Python 3.11.7 with NumPy 1.26.0, pandas 2.2.3, SciPy 1.11.4, scikit-learn 1.7.0, matplotlib 3.9.2, threadpoolctl 3.6.0, and pytest 7.4.0, exactly matching the package pins in `requirements.txt`. Full parameters, package versions, input SHA-256 hashes, and execution time are in [settings.json](results/settings.json). Data and generated outputs are ignored by git; the relative result links become available after running the study locally, and a future download may contain revised observations.

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
