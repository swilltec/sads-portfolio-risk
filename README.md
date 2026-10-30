# Stress-adaptive dimensionality selection

One-day VaR and expected shortfall forecasting for an equal-weighted portfolio of the 49 French industry portfolios. SADS selects the PCA rank from market stress and past forecast losses, alongside fixed-rank PCA, rolling Bai–Ng, and Ledoit–Wolf benchmarks.

## Run

Set the data paths, output directory, and study defaults in `sads/constants.py`.

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python main.py
```

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
- **Stress:** available STLFSI4 above its expanding weekly mean. The experiment repeats with 21-day trailing portfolio volatility above its expanding daily mean.
- **Evaluation:** FZ0, Giacomini–White tests with HAC covariance, Kupiec and Christoffersen coverage tests, and a 95% Model Confidence Set. Lower FZ0 and negative SADS-minus-baseline differences favor SADS.
- **Rank analysis:** HAC estimates of elevated-minus-lower-stress rank differences for SADS, Bai–Ng, and the rank minimizing FZ0 over the preceding 252 forecasts.
- **Crisis holdouts:** ranks are fitted on forecasts completed before each of 2008, 2020, and 2022, then held fixed during that episode. Separate deletion checks omit each crisis year when comparing the main forecast losses.

STLFSI4 availability uses an assumed seven-calendar-day publication lag, with same-day releases excluded. Supply `release_calendar` with `observation_date,release_date` columns for historical publication dates. The supplied index history contains revised and back-filled values; the volatility-proxy analysis uses trailing returns. Individual comparison p-values are unadjusted.

## Files

`main.py` runs the study using the defaults in `sads/constants.py`. The `sads/` modules contain data alignment, covariance models, rank selection, statistical tests, and the experiment loop. Run the numerical and timing checks with `python -m pytest -q`.

`results/settings.json` records parameters, input hashes, and evaluation dates. Each signal's output directory contains forecast, summary, coverage, comparison, crisis-holdout, and crisis-deletion CSVs; `evaluation.json` contains rank tests, the MCS, and robustness results. `diagnostics.png` plots ranks, tail forecasts, and cumulative FZ0. The tables are also returned as pandas objects.

## References

- [French Data Library](https://mba.tuck.dartmouth.edu/pages/faculty/Ken.French/data_library.html)
- [FRED STLFSI4](https://fred.stlouisfed.org/series/STLFSI4)
- [Patton, Ziegel & Chen (2019), FZ0](https://public.econ.duke.edu/~ap172/Patton_Ziegel_Chen_JoE_2019.pdf)
- [Bai–Ng ICp2, NBER lecture](https://conference.nber.org/confer/2008/si2008/TSE/Lecture11.pdf)
