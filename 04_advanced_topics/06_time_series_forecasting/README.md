# Time Series Forecasting

**🇬🇧 English | [🇮🇩 Indonesian version](README.id.md)**

🏠 Back to Advanced Topics: `../README.md`

Daily outlet sales are not a bag of independent rows. Today's revenue depends on
yesterday, last week's same weekday, the payday cycle, holidays and a slow
business trend. Random train/test splits and random K-fold cross-validation
therefore leak information from the future and produce optimistic scores.

This module builds a small, transparent forecasting toolkit with **NumPy only**
and then shows how the same ideas map to scikit-learn and statsmodels. The
running example is synthetic Indonesian outlet sales with a linear trend,
weekly seasonality, *gajian* (payday) spikes around the 25th to the 1st, and
fixed-date holiday effects.

## 📚 Table of Contents

- [Why Time Series Is Different](#-why-time-series-is-different)
- [The Synthetic Outlet Dataset](#-the-synthetic-outlet-dataset)
- [Forecasting Setup and Leakage](#-forecasting-setup-and-leakage)
- [Baseline Forecasts](#-baseline-forecasts)
- [Simple Exponential Smoothing](#-simple-exponential-smoothing)
- [Additive Holt-Winters](#-additive-holt-winters)
- [Autoregressive Models](#-autoregressive-models)
- [Lag Features and LagRegression](#-lag-features-and-lagregression)
- [Forecast Metrics](#-forecast-metrics)
- [Ordered Split and Rolling-Origin Backtesting](#-ordered-split-and-rolling-origin-backtesting)
- [Quick Start](#-quick-start)
- [Scikit-Learn and statsmodels Examples](#-scikit-learn-and-statsmodels-examples)
- [Notebook Exercises](#-notebook-exercises)
- [Files in This Directory](#-files-in-this-directory)
- [When to Use Each Method](#-when-to-use-each-method)
- [Advantages and Limitations](#-advantages-and-limitations)
- [Retail Applications](#-retail-applications)
- [Related Modules](#-related-modules)

## 🧠 Why Time Series Is Different

A standard supervised-learning row is assumed to be exchangeable: shuffling the
rows does not change the problem. A time series is ordered. At time `t` you may
only use information from `t-1`, `t-2`, ...; the future is unknown.

Three consequences drive every design decision in this module:

1. **Split chronologically.** Training data must be older than test data.
2. **Build features from the past only.** A feature for target `y[t]` may use
   `y[t-1]`, `y[t-2]`, ..., but never `y[t]` or `y[t+1]`.
3. **Backtest with a moving origin.** Evaluate many future windows instead of a
   single split, always retraining only on data available at that origin.

The goal is not just a low error on one window. The goal is a procedure that
would have produced an honest forecast at the moment the forecast was needed.

## 🏪 The Synthetic Outlet Dataset

`generate_outlet_sales(n_days, seed)` returns `(dates, values)`:

- `dates` is a `numpy.datetime64[D]` array of consecutive calendar days,
- `values` is a positive daily-sales vector with two decimal places.

The same seed always produces exactly the same series. The generator combines
five components:

```
sales[t] = (base[t] * weekly[weekday(t)] + payday[t]) * (1 + holiday[t]) + noise[t]
```

| Component | Shape | Purpose |
|-----------|-------|---------|
| `base[t] = 900 + 0.55 * t` | Linear trend | Slow growth in the business |
| `weekly` | 7 multipliers | Weekend lift, Saturday busiest |
| `payday` | Additive spikes | Gajian window from the 25th to the 2nd |
| `holiday` | Multiplicative lift | New Year, Labour Day, Independence Day, Christmas |
| `noise` | Gaussian | Day-to-day randomness |

### Weekly seasonality

The weekday multipliers are:

| Day | Mon | Tue | Wed | Thu | Fri | Sat | Sun |
|-----|-----|-----|-----|-----|-----|-----|-----|
| Factor | 0.92 | 0.94 | 0.97 | 1.00 | 1.08 | 1.16 | 1.05 |

This creates a repeating period of seven days, which is why `season=7` is the
default throughout the module.

### Payday (gajian) spikes

Salaries in Indonesia are commonly paid on the 25th. The generator adds an
additive boost by day of month:

```
25th: +300   26th: +250   27th: +175   28th: +120
29th: +80    30th: +50    31st: +30
 1st: +260    2nd: +150    3rd: +80     4th: +40
```

The result is a large spending window at the end of the month, a short quiet
period, and a second bump when the new month starts. This pattern is *not*
captured by a plain weekly seasonal model, which is why extra calendar features
are valuable in production.

### Holiday effects

Fixed-date holidays receive a multiplicative lift, with smaller lifts on the day
before and after:

- New Year (1 January),
- Labour Day (1 May),
- Independence Day (17 August),
- Christmas (25 December),
- New Year's Eve (31 December).

Lunar holidays such as Eid al-Fitr move every year. A real system should join an
official holiday calendar instead of hard-coding them, but the fixed-date
version is enough to demonstrate the modelling idea.

## ⚙️ Forecasting Setup and Leakage

The module uses a single ordered split for the final test window:

```python
from implementation import generate_outlet_sales, time_train_test_split

dates, sales = generate_outlet_sales(n_days=365, seed=42)
train, test = time_train_test_split(sales, test_size=30)

# train contains the first 335 days, test the last 30 days.
# The two slices are contiguous and never shuffled.
```

`time_train_test_split` accepts either an integer test size or a float fraction:

```python
train, test = time_train_test_split(sales, test_size=0.2)  # ceil(20%) test
```

A shuffled split would put future days into the training set, letting the model
learn from the exact period it is supposed to forecast. That is leakage. The
same logic applies to features:

```python
# WRONG: feature for y[t] contains y[t] itself
X_bad = sales[t]

# RIGHT: feature for y[t] contains only earlier observations
X_good = sales[t - 7]
```

The `make_lag_features` helper enforces this rule and is covered below.

## 📈 Baseline Forecasts

Always establish a baseline before fitting a complex model. If Holt-Winters or
gradient boosting cannot beat a seasonal naive forecast, the extra complexity is
not justified.

### 1. Naive forecast

```
y_hat[T+h] = y[T]
```

The last observed value is repeated for every future step. This is the
random-walk baseline and is surprisingly hard to beat on noisy financial or
demand series.

```python
from implementation import naive_forecast

prediction = naive_forecast(train, horizon=30)
```

### 2. Seasonal naive forecast

```
y_hat[T+h] = y[T+h-m]
```

The value from the same season one period ago is repeated. With daily data and
`m = 7`, Monday is forecast from last Monday, Tuesday from last Tuesday, and so
on. It is a strong baseline whenever weekly or annual seasonality dominates.

```python
from implementation import seasonal_naive_forecast

prediction = seasonal_naive_forecast(train, horizon=30, season=7)
```

### 3. Moving-average forecast

```
y_hat[T+h] = mean(y[T-w+1], ..., y[T])
```

A flat forecast equal to the mean of the last `w` observations. It smooths
noise and captures the current level, but it ignores both trend and seasonality.
It is most useful for level-only series or as a quick sanity check.

```python
from implementation import moving_average_forecast

prediction = moving_average_forecast(train, horizon=30, window=7)
```

## 🌊 Simple Exponential Smoothing

Simple exponential smoothing (SES) keeps one state, the **level** `L_t`:

```
L_0 = y_0
L_t = alpha * y_t + (1 - alpha) * L_{t-1}
y_hat[T+h] = L_T
```

`alpha` in `(0, 1]` controls the memory:

- **Small alpha** (for example 0.1): slow reaction, very smooth level, good for
  stable noisy series.
- **Large alpha** (for example 0.9): fast reaction, close to the naive forecast,
  good when the level genuinely changes.

SES captures level shifts but cannot represent a trend or a repeating weekly
pattern. Its forecast is always flat.

```python
from implementation import SimpleExpSmoothing

model = SimpleExpSmoothing(alpha=0.3).fit(train)
prediction = model.forecast(horizon=30)

print(model.level_)          # final level
print(model.fitted_values_)  # one-step-ahead in-sample fits
print(model.residuals_)
```

The class exposes `forecast(h)` and the alias `predict(h)`. Calling either before
`fit` raises a clear error.

## ❄️ Additive Holt-Winters

Holt-Winters extends SES with a **trend** `T_t` and a **seasonal** component
`S_t` with one value per phase. For period `m` and additive seasonality:

```
y_hat_t = L_{t-1} + T_{t-1} + S_{t-m}

L_t = alpha * (y_t - S_{t-m}) + (1 - alpha) * (L_{t-1} + T_{t-1})
T_t = beta  * (L_t - L_{t-1})       + (1 - beta)  * T_{t-1}
S_t = gamma * (y_t - L_t)           + (1 - gamma) * S_{t-m}
```

The seasonal component is additive: it is a number of sales added to the current
level, not a percentage multiplier. That matches a series where the weekend
difference stays roughly constant as the business grows.

### Initialisation

The implementation uses the first two complete seasons:

```
L_0 = mean(first season)
T_0 = (mean(second season) - mean(first season)) / m
S_i = first_season[i] - L_0       (centred to mean zero)
```

At least two full seasons are required, so daily data with `m = 7` needs at
least 14 observations. A longer history gives much more stable estimates.

### Forecasting

After fitting, the h-step forecast is:

```
y_hat[T+h] = L_T + h * T_T + S[(T+h-1) mod m]
```

The level is extrapolated with the final trend, and the seasonal state for the
appropriate phase is added back.

```python
from implementation import HoltWinters

model = HoltWinters(
    alpha=0.3,
    beta=0.05,
    gamma=0.3,
    season_length=7,
).fit(train)

prediction = model.forecast(30)
print(model.level_, model.trend_, model.seasonal_)
```

Practical guidance for the three smoothing factors:

| Parameter | Too small | Too large |
|-----------|-----------|-----------|
| `alpha` | Level lags behind changes | Level chases noise |
| `beta` | Trend is too rigid | Trend oscillates |
| `gamma` | Seasonal pattern is frozen | Seasonal pattern absorbs noise |

The class deliberately uses fixed factors rather than an optimiser, so the
update equations remain visible. In production, tune them on a validation window
or use statsmodels' optimised implementation.

## 🔁 Autoregressive Models

An `AR(p)` model predicts the next value from its own previous `p` values:

```
y_t = c + phi_1 y_{t-1} + ... + phi_p y_{t-p} + e_t
```

`ARModel(p)` estimates the intercept and coefficients by ordinary least squares.
For observations `t = p, ..., n-1`, the design matrix is:

```
[ 1  y[t-1]  y[t-2]  ...  y[t-p] ]
```

The target column is `y[p:]`. Using `numpy.linalg.lstsq` avoids explicitly
forming the normal equations and is numerically more stable.

```python
from implementation import ARModel

model = ARModel(p=7).fit(train)
print(model.intercept_)
print(model.coef_)          # phi_1 ... phi_7
print(model.forecast(30))   # recursive multi-step forecast
```

### Recursive multi-step forecasting

A one-step prediction only needs known history. For step two, the model uses its
own step-one prediction as the most recent lag; for step three it uses the
step-one and step-two predictions, and so on:

```
y_hat[T+1] = c + phi_1 y[T]     + phi_2 y[T-1] + ...
y_hat[T+2] = c + phi_1 y_hat[T+1] + phi_2 y[T] + ...
```

This is convenient but errors compound. A stable AR model decays towards its
long-run mean; an unstable one can explode. Always inspect multi-step forecasts
instead of trusting them blindly.

`ARModel` does not enforce stationarity. If the estimated roots lie outside the
unit circle, the recursive forecast may grow without bound. For production,
consider differencing, regularization or a restricted/stationary parameterisation.

## 🧱 Lag Features and LagRegression

Tree-based and linear supervised models need a flat feature matrix. The bridge
from a time series to that matrix is lag features.

### `make_lag_features(y, lags, rolling_windows)`

For target time `t`, the helper builds:

- one column per lag `k`: `y[t-k]`,
- for each rolling window `w`: `mean(y[t-w:t])` and `std(y[t-w:t])`.

The first usable target is `max(max(lags), max(rolling_windows))`, because all
requested history must exist. Lag 0 is rejected because it would equal the
target and leak the answer.

```python
from implementation import make_lag_features

X, target, index = make_lag_features(
    train,
    lags=(1, 2, 3, 7),
    rolling_windows=(7, 14),
    return_index=True,
)

# Row i predicts train[index[i]] using only train[:index[i]].
print(X.shape, target.shape, index[:5])
```

The function returns `(X, target)` by default. With `return_index=True` it also
returns the integer target indices, which keeps feature rows aligned with dates
and ordered splits.

Why the rolling windows are safe: for target `t`, the slice is `y[t-w:t]`, which
ends immediately before `t`. It does not include `y[t]`. A common mistake is
`y[t-w+1:t+1]`, which includes the target and inflates validation scores.

### `LagRegression`

`LagRegression` puts ordinary least squares on top of these features:

```
y_t = c + sum_k phi_k y[t-k]
        + sum_w (a_w * rolling_mean_w[t] + b_w * rolling_std_w[t])
```

```python
from implementation import LagRegression

model = LagRegression(
    lags=(1, 2, 3, 4, 5, 6, 7, 14),
    rolling_windows=(7, 14),
).fit(train)

prediction = model.forecast(30)
print(model.feature_names_)
print(model.coef_)
```

Forecasting is recursive: each prediction is appended to the history before the
next feature row is computed. `feature_names_` documents the column order:
`lag_1`, `lag_2`, ..., `rolling_mean_7`, `rolling_std_7`, and so on.

For a fair comparison with gradient boosting, `sklearn_example.py` evaluates
one-step-ahead predictions on held-out targets. That measures feature quality
without conflating it with recursive error compounding. The from-scratch
`LagRegression.forecast` demonstrates the harder multi-step task.

## 📏 Forecast Metrics

Let `y_i` be the actual value and `y_hat_i` the forecast for `i = 1 ... n`.

| Metric | Formula | Unit | Notes |
|--------|---------|------|-------|
| MAE | `mean(|y_i - y_hat_i|)` | same as sales | Easy to explain |
| RMSE | `sqrt(mean((y_i - y_hat_i)^2))` | same as sales | Penalises large errors |
| MAPE | `100 * mean(|y_i - y_hat_i| / |y_i|)` | percent | Undefined at zero actuals |
| SMAPE | `100 * mean(2|y_i - y_hat_i| / (|y_i| + |y_hat_i|))` | percent (0-200) | Symmetric, bounded |

```python
from implementation import mae, rmse, mape, smape

print(mae(test, prediction))
print(rmse(test, prediction))
print(mape(test, prediction))
print(smape(test, prediction))
```

### Choosing a metric

- **MAE** is the most interpretable when all days matter equally.
- **RMSE** is useful when large misses are disproportionately expensive (for
  example, stockouts or overstock).
- **MAPE** is common in retail reporting but explodes when actual sales are near
  zero. The implementation skips zero actuals and returns `inf` only when every
  actual is zero and the predictions are not.
- **SMAPE** stays bounded when values are small, which is useful for low-sales
  days, but it is asymmetric in a different way and should not be compared
  blindly across datasets.

Report more than one metric. A model can win on MAE while losing on RMSE because
it makes a few very large errors.

## 🔄 Ordered Split and Rolling-Origin Backtesting

A single train/test split can be lucky or unlucky. Walk-forward validation
answers a more realistic question: *if the forecast had been made at each origin,
how well would the model have done?*

### `time_train_test_split`

```python
train, test = time_train_test_split(sales, test_size=30)
```

No shuffle, no overlap. The last training observation is immediately before the
first test observation. For a float `test_size`, the test length is rounded up.

### `rolling_origin_backtest`

```python
from implementation import rolling_origin_backtest, seasonal_naive_forecast

folds = rolling_origin_backtest(
    sales,
    lambda history, h: seasonal_naive_forecast(history, h, season=7),
    initial=250,
    horizon=30,
    step=30,
)

print(len(folds))          # 3 complete folds for 365 observations
print(folds[0]["mae"])
```

The origins are `initial, initial + step, initial + 2*step, ...` while
`origin + horizon <= n`. With `n = 365`, `initial = 250`, `horizon = 30` and
`step = 30`, the origins are 250, 280 and 310, so there are three folds. The
general fold count is:

```
floor((n - initial - horizon) / step) + 1
```

Each fold dictionary contains:

| Key | Meaning |
|-----|---------|
| `fold` | Zero-based fold number |
| `train_start`, `train_end` | Training window `[0, train_end)` |
| `test_start`, `test_end` | Test window `[test_start, test_end)` |
| `actual`, `predicted` | Arrays of length `horizon` |
| `mae`, `rmse`, `mape`, `smape` | Fold-level metrics |

`model_fn` is flexible. It can be:

```python
# 1. A function returning predictions
lambda history, h: naive_forecast(history, h)

# 2. A factory returning a fitted model with forecast/predict
lambda history: HoltWinters(season_length=7).fit(history)

# 3. Any callable returning a prediction array
lambda history: seasonal_naive_forecast(history, 30, season=7)
```

The backtester never trains on observations at or after the test start, so it is
safe for model selection. Still keep a final untouched test window or a later
period for the final report.

### TimeSeriesSplit

For supervised lag models, scikit-learn's `TimeSeriesSplit` provides the same
expanding-window idea:

```python
from sklearn.model_selection import TimeSeriesSplit

splitter = TimeSeriesSplit(n_splits=5)
for train_index, validation_index in splitter.split(X_train):
    ...
```

Validation rows always come after training rows. Do not use `KFold`,
`ShuffleSplit` or `train_test_split(shuffle=True)` on time-ordered data.

## 🚀 Quick Start

From this directory:

```bash
python implementation.py
python sklearn_example.py
```

Minimal comparison:

```python
import numpy as np

from implementation import (
    ARModel,
    HoltWinters,
    LagRegression,
    SimpleExpSmoothing,
    generate_outlet_sales,
    mae,
    naive_forecast,
    seasonal_naive_forecast,
    time_train_test_split,
)

dates, sales = generate_outlet_sales(n_days=365, seed=42)
train, test = time_train_test_split(sales, test_size=30)
horizon = len(test)

models = {
    "Naive": naive_forecast(train, horizon),
    "Seasonal naive": seasonal_naive_forecast(train, horizon, season=7),
    "SES": SimpleExpSmoothing(alpha=0.3).fit(train).forecast(horizon),
    "Holt-Winters": HoltWinters(season_length=7).fit(train).forecast(horizon),
    "AR(7)": ARModel(7).fit(train).forecast(horizon),
    "LagRegression": LagRegression(
        lags=(1, 2, 3, 7), rolling_windows=(7,)
    ).fit(train).forecast(horizon),
}

for name, prediction in models.items():
    print(name, round(mae(test, prediction), 2))
```

## 🤖 Scikit-Learn and statsmodels Examples

`sklearn_example.py` turns the series into a supervised problem and runs:

- `GradientBoostingRegressor`,
- `RandomForestRegressor`,
- both evaluated with `TimeSeriesSplit` on the training window,
- a final one-step-ahead test window,
- the from-scratch `LagRegression` and seasonal-naive baselines,
- statsmodels `ExponentialSmoothing` **only if statsmodels is importable**.

The optional block is guarded by a `try/except ImportError` and prints a clear
skip message instead of failing:

```text
statsmodels is not installed - skipping this optional comparison.
Install it with: pip install statsmodels
```

The feature matrix includes lags 1--7 and 14 plus rolling means and standard
deviations over 7 and 14 days. Keeping lag 14 lets the model see two full weeks
of history.

## 📓 Notebook Exercises

`exercise.ipynb` contains ten code cells and nine guided exercises with worked
solutions:

1. Explore trend, weekdays, payday and holiday structure.
2. Build an ordered split and check it is chronological.
3. Compare naive, seasonal-naive and moving-average forecasts.
4. Fit SES and Holt-Winters and inspect the learned states.
5. Recover known AR(2) coefficients from simulated data.
6. Prove that lag features do not leak future values.
7. Fit and recursively forecast with `LagRegression`.
8. Count and inspect rolling-origin folds.
9. Rank models on the test window and discuss model selection.

The notebook runs top to bottom from either this directory or the repository
root and uses only NumPy plus this module.

## 📁 Files in This Directory

| File | Purpose |
|------|---------|
| `implementation.py` | NumPy-only data generator, models, features, backtest and metrics |
| `sklearn_example.py` | Gradient boosting, random forest, `TimeSeriesSplit`, optional statsmodels |
| `exercise.ipynb` | Guided exercises with solutions |
| `README.md` | English documentation (this file) |
| `README.id.md` | Indonesian documentation |
| `../../tests/test_time_series.py` | Automated tests for leakage, accuracy and metrics |

## 🎯 When to Use Each Method

| Method | Use when | Avoid when |
|--------|----------|------------|
| Naive | Very short horizon, noisy random walk | Clear trend or seasonality |
| Seasonal naive | Strong stable weekly/annual pattern | Pattern changes quickly |
| Moving average | Level-only, noisy series | Trend or seasonality present |
| SES | Level shifts, no trend/season | Weekly cycles matter |
| Holt-Winters | Trend + stable additive seasonality | Multiplicative seasonality, many covariates |
| AR(p) | Short-memory autocorrelation, stationary series | Strong trend/seasonality without differencing |
| LagRegression | Many lags, rolling summaries, interpretable baseline | Strong non-linearity, long memory |
| Gradient boosting | Non-linear feature interactions, enough history | Tiny data, need extrapolation |
| Random forest | Robust non-linear baseline | Extrapolating a trend beyond training range |

Tree models cannot extrapolate a linear trend outside the range seen in training.
For a growing business, detrend the series, include calendar features, or combine
a tree model with a trend model.

## ⚖️ Advantages and Limitations

### Advantages

- **Transparent:** every state update and coefficient is visible.
- **Fast:** all models fit in milliseconds on a year of daily data.
- **Strong baselines:** seasonal naive and Holt-Winters are hard to beat.
- **Honest evaluation:** ordered splits and rolling origins prevent leakage.
- **Composable:** lag features connect classical forecasting to modern ML.

### Limitations

- Only univariate series are modelled; promotions, price and foot traffic are
  not included.
- Holt-Winters uses fixed smoothing factors rather than maximum likelihood.
- `ARModel` and `LagRegression` can produce unstable recursive forecasts when
  coefficients imply non-stationarity.
- Fixed-date holidays ignore moving lunar holidays such as Eid al-Fitr.
- No prediction intervals, probabilistic forecasts or hierarchical reconciliation.
- No automatic seasonality detection, differencing or stationarity tests.

These are deliberate scope choices. The module teaches the mechanics and the
validation discipline; production systems can then use statsmodels, Prophet,
ARIMA/SARIMAX, gradient boosting or deep sequence models.

## 🏬 Retail Applications

- **Daily sales planning:** forecast the next one to four weeks per outlet.
- **Staff scheduling:** align shifts with weekday and payday demand.
- **Inventory replenishment:** order more before the gajian window and holidays.
- **Promotion evaluation:** compare actual sales with a counterfactual forecast.
- **Budgeting:** aggregate outlet forecasts into a regional plan.
- **Anomaly detection:** flag days that deviate strongly from the forecast.

In the SPG retail scenario, the payday pattern is especially important: a model
that only knows weekly seasonality systematically under-forecasts the end of the
month. Adding day-of-month features, a payday indicator or a separate payday
component usually improves accuracy substantially.

## 🔗 Related Modules

- [Model Evaluation](../03_model_evaluation/) for metrics and cross-validation.
- [XGBoost](../04_xgboost/) and [CNN & Computer Vision](../05_cnn_computer_vision/)
  for the broader advanced-topics track.
- [Linear Regression](../../01_supervised_learning/01_linear_regression/) for the
  least-squares foundation used by `ARModel` and `LagRegression`.
- [Random Forest](../../01_supervised_learning/04_random_forest/) and
  [Gradient Boosting](../../01_supervised_learning/09_gradient_boosting/) for the
  supervised models in `sklearn_example.py`.

**Happy Learning! 🎓** | *[🇮🇩 Indonesian version](README.id.md)*
