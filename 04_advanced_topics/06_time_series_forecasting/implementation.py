"""
Time Series Forecasting from Scratch
====================================

A compact, dependency-free (NumPy only) toolkit for univariate daily sales
forecasting.  The module covers the classical baselines and models that are
used before reaching for heavier machine-learning libraries:

- synthetic outlet sales with trend, weekly seasonality, *gajian* (payday)
  spikes and fixed-date holiday effects,
- naive, seasonal-naive and moving-average baselines,
- simple exponential smoothing (SES),
- additive Holt-Winters (level + trend + seasonality),
- autoregressive models ``AR(p)`` estimated with ordinary least squares,
- leakage-free lag/rolling feature construction and a recursive
  ``LagRegression`` model,
- an ordered train/test split and walk-forward rolling-origin backtesting,
- MAE, RMSE, MAPE and SMAPE metrics.

Everything is implemented with NumPy so the mechanics stay visible.  The
script at the bottom prints a small end-to-end comparison.

Author: ML-Algo Repository
"""

from __future__ import annotations

import inspect
from typing import Callable, Iterable, List, Optional, Sequence, Tuple, Union

import numpy as np


# ---------------------------------------------------------------------------
# Validation helpers
# ---------------------------------------------------------------------------
def _as_1d_float(values: Sequence[float], name: str = "y") -> np.ndarray:
    """Return ``values`` as a finite one-dimensional float array."""
    arr = np.asarray(values, dtype=float)
    if arr.ndim == 0:
        arr = arr.reshape(1)
    else:
        arr = arr.ravel()
    if arr.size == 0:
        raise ValueError(f"{name} must not be empty")
    if not np.all(np.isfinite(arr)):
        raise ValueError(f"{name} contains NaN or infinite values")
    return arr


def _positive_int(value: int, name: str, minimum: int = 1) -> int:
    """Validate an integer scalar and return it as a Python ``int``."""
    if isinstance(value, bool) or not isinstance(value, (int, np.integer)):
        raise ValueError(f"{name} must be an integer")
    value = int(value)
    if value < minimum:
        raise ValueError(f"{name} must be at least {minimum}")
    return value


def _int_sequence(values: Union[int, Iterable[int]], name: str) -> Tuple[int, ...]:
    """Normalize an int or an iterable of ints to a sorted unique tuple."""
    if isinstance(values, (int, np.integer)) and not isinstance(values, bool):
        values = (int(values),)
    try:
        items = [int(v) for v in values]
    except TypeError as exc:
        raise ValueError(f"{name} must be an int or an iterable of ints") from exc
    if not items:
        raise ValueError(f"{name} must contain at least one value")
    if any(v < 1 for v in items):
        raise ValueError(f"{name} values must be positive (no lag 0)")
    return tuple(sorted(set(items)))


def _check_forecast_horizon(horizon: int) -> int:
    return _positive_int(horizon, "horizon")


def _check_is_fitted(model: object, attribute: str, class_name: str) -> None:
    if not hasattr(model, attribute):
        raise ValueError(
            f"This {class_name} instance is not fitted yet. Call fit first."
        )


# ---------------------------------------------------------------------------
# Synthetic outlet data
# ---------------------------------------------------------------------------
# Monday ... Sunday multipliers used by the data generator.  Indonesian
# outlets usually see a weekend lift, with Saturday as the busiest day.
_WEEKLY_FACTORS = np.array(
    [0.92, 0.94, 0.97, 1.00, 1.08, 1.16, 1.05], dtype=float
)

# Fixed-date holidays: (month, day, relative lift).  The generator also adds a
# smaller lift on the day before and after each holiday.
_FIXED_HOLIDAYS = (
    (1, 1, 0.90),    # New Year
    (5, 1, 0.45),    # Labour Day
    (8, 17, 0.65),   # Independence Day
    (12, 25, 0.85),  # Christmas
    (12, 31, 0.35),  # New Year's Eve
)

# Additive payday boosts by day of month.  Salaries in Indonesia are commonly
# paid on the 25th, so the spike starts around the 25th, decays, and gets a
# second, smaller bump on the 1st when the new month begins.
_PAYDAY_BOOSTS = {
    25: 300.0,
    26: 250.0,
    27: 175.0,
    28: 120.0,
    29: 80.0,
    30: 50.0,
    31: 30.0,
    1: 260.0,
    2: 150.0,
    3: 80.0,
    4: 40.0,
}


def generate_outlet_sales(
    n_days: int,
    seed: int,
    start: str = "2023-01-01",
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generate reproducible synthetic daily outlet sales.

    The series combines five effects:

    1. **Linear trend** -- sales grow slowly over time.
    2. **Weekly seasonality** -- weekends are busier than weekdays.
    3. **Payday (gajian) spikes** -- a large bump around the 25th, a decaying
       tail through the end of the month, and a second bump on the 1st--4th.
    4. **Holiday effects** -- fixed-date holidays plus small neighbour effects.
    5. **Gaussian noise** -- day-to-day randomness.

    Parameters
    ----------
    n_days : int
        Number of consecutive daily observations. Must be positive.
    seed : int
        Seed for ``numpy.random.default_rng``. The same seed always returns
        exactly the same series.
    start : str, default="2023-01-01"
        First calendar date.

    Returns
    -------
    dates : ndarray of dtype datetime64[D]
        Consecutive calendar dates with shape ``(n_days,)``.
    values : ndarray of shape (n_days,)
        Daily sales values (positive floats, rounded to two decimals).
    """
    n_days = _positive_int(n_days, "n_days", minimum=1)
    if not isinstance(seed, (int, np.integer)) or isinstance(seed, bool):
        raise ValueError("seed must be an integer")

    rng = np.random.default_rng(int(seed))

    start_date = np.datetime64(start, "D")
    dates = start_date + np.arange(n_days, dtype="timedelta64[D]")

    t = np.arange(n_days, dtype=float)
    epoch_days = dates.astype("datetime64[D]").astype(np.int64)
    month_start = dates.astype("datetime64[M]")
    day_of_month = (dates - month_start).astype(np.int64) + 1
    month = month_start.astype(np.int64) % 12 + 1

    # 1970-01-01 was a Thursday, so adding 3 maps Monday to 0.
    day_of_week = (epoch_days + 3) % 7

    # Base level with a gentle upward trend, e.g. ~900 to ~1200 in a year.
    base = 900.0 + 0.55 * t

    # Weekly pattern is multiplicative so the seasonality grows with the level.
    values = base * _WEEKLY_FACTORS[day_of_week]

    # Payday (gajian) additive boosts.
    payday = np.zeros(n_days, dtype=float)
    for day, boost in _PAYDAY_BOOSTS.items():
        payday[day_of_month == day] = boost
    values = values + payday

    # Holiday effects: exact date plus smaller lifts one day before/after.
    holiday_lift = np.zeros(n_days, dtype=float)
    first_year = int(dates[0].astype("datetime64[Y]").astype(int)) + 1970
    last_year = int(dates[-1].astype("datetime64[Y]").astype(int)) + 1970
    for year in range(first_year, last_year + 1):
        for holiday_month, holiday_day, effect in _FIXED_HOLIDAYS:
            holiday_date = np.datetime64(
                f"{year:04d}-{holiday_month:02d}-{holiday_day:02d}", "D"
            )
            for offset, weight in ((-1, 0.25), (0, 1.0), (1, 0.15)):
                target = holiday_date + np.timedelta64(offset, "D")
                holiday_lift[dates == target] += effect * weight
    values = values * (1.0 + holiday_lift)

    # Day-to-day Gaussian noise with a constant standard deviation.
    noise = rng.normal(0.0, 28.0, size=n_days)
    values = values + noise

    values = np.clip(values, 1.0, None)
    return dates, np.round(values, 2)


# ---------------------------------------------------------------------------
# Baseline forecasts
# ---------------------------------------------------------------------------
def naive_forecast(y: Sequence[float], horizon: int = 1) -> np.ndarray:
    """
    Repeat the last observed value for every future step.

    This is the standard random-walk baseline: ``y_hat[T+h] = y[T]``.
    """
    y = _as_1d_float(y)
    horizon = _check_forecast_horizon(horizon)
    return np.full(horizon, y[-1], dtype=float)


def seasonal_naive_forecast(
    y: Sequence[float],
    horizon: int = 1,
    season: int = 7,
) -> np.ndarray:
    """
    Repeat the value from the same season (phase) one period ago.

    With ``season=7`` on daily data, the forecast for Monday is last Monday's
    value, the forecast for Tuesday is last Tuesday's value, and so on.  The
    function requires at least one complete season of history.
    """
    y = _as_1d_float(y)
    horizon = _check_forecast_horizon(horizon)
    season = _positive_int(season, "season")
    if y.size < season:
        raise ValueError("seasonal_naive_forecast needs at least one full season")

    last_season = y[-season:]
    repeats = int(np.ceil(horizon / season))
    return np.tile(last_season, repeats)[:horizon].astype(float, copy=True)


def moving_average_forecast(
    y: Sequence[float],
    horizon: int = 1,
    window: int = 7,
) -> np.ndarray:
    """
    Forecast a flat line equal to the mean of the last ``window`` values.

    A moving average is a strong smoothing baseline, but it cannot reproduce
    seasonality or a trend.  It is most useful for noisy, level-only series.
    """
    y = _as_1d_float(y)
    horizon = _check_forecast_horizon(horizon)
    window = _positive_int(window, "window")
    if y.size < window:
        raise ValueError("moving_average_forecast needs at least 'window' values")
    level = float(np.mean(y[-window:]))
    return np.full(horizon, level, dtype=float)


# ---------------------------------------------------------------------------
# Simple exponential smoothing
# ---------------------------------------------------------------------------
class SimpleExpSmoothing:
    """
    Simple exponential smoothing (SES) with a fixed smoothing factor.

    The level is updated recursively::

        level_t = alpha * y_t + (1 - alpha) * level_{t-1}

    and the h-step forecast is the last level, repeated::

        y_hat[T+h] = level_T

    Parameters
    ----------
    alpha : float, default=0.3
        Smoothing factor in ``(0, 1]``. Larger values react faster to recent
        observations; smaller values produce a smoother level.

    Attributes
    ----------
    level_ : float
        Final level after fitting.
    fitted_values_ : ndarray
        One-step-ahead in-sample fitted values.
    residuals_ : ndarray
        ``y - fitted_values_``.
    n_obs_ : int
        Number of observations seen during fitting.
    """

    def __init__(self, alpha: float = 0.3):
        if not 0.0 < float(alpha) <= 1.0:
            raise ValueError("alpha must be in (0, 1]")
        self.alpha = float(alpha)

    def fit(self, y: Sequence[float]) -> "SimpleExpSmoothing":
        """Fit the level from a one-dimensional series."""
        y = _as_1d_float(y)
        alpha = self.alpha

        fitted = np.empty(y.size, dtype=float)
        level = float(y[0])
        fitted[0] = level

        for t in range(1, y.size):
            fitted[t] = level
            level = alpha * y[t] + (1.0 - alpha) * level

        self.level_ = level
        self.fitted_values_ = fitted
        self.residuals_ = y - fitted
        self.n_obs_ = y.size
        return self

    def forecast(self, horizon: int = 1) -> np.ndarray:
        """Return the last level repeated ``horizon`` times."""
        _check_is_fitted(self, "level_", "SimpleExpSmoothing")
        horizon = _check_forecast_horizon(horizon)
        return np.full(horizon, self.level_, dtype=float)

    def predict(self, horizon: int = 1) -> np.ndarray:
        """Alias for :meth:`forecast`."""
        return self.forecast(horizon)


# ---------------------------------------------------------------------------
# Additive Holt-Winters
# ---------------------------------------------------------------------------
class HoltWinters:
    """
    Additive Holt-Winters exponential smoothing.

    The model maintains three states -- level ``L``, trend ``T`` and a seasonal
    component ``S`` with one value per phase.  The additive update equations
    are::

        y_hat_t = L_{t-1} + T_{t-1} + S_{t-m}
        L_t     = alpha * (y_t - S_{t-m}) + (1 - alpha) * (L_{t-1} + T_{t-1})
        T_t     = beta  * (L_t - L_{t-1}) + (1 - beta) * T_{t-1}
        S_t     = gamma * (y_t - L_t) + (1 - gamma) * S_{t-m}

    where ``m`` is the seasonal period (7 for daily data with a weekly cycle).

    Note: this is the classic textbook form, where the seasonal update uses the
    *new* level ``L_t``.  statsmodels' ``ExponentialSmoothing`` uses the
    error-correction form ``S_t = gamma * (y_t - L_{t-1} - T_{t-1}) + ...``,
    so with identical parameters and initial states the two give slightly
    different (but equally valid) results.  Parameters are fixed here, not
    optimised, so Holt-Winters is not automatically the best model.

    Parameters
    ----------
    alpha, beta, gamma : float
        Level, trend and seasonal smoothing factors in ``[0, 1]``.
    season_length : int, default=7
        Number of observations in one seasonal cycle. ``season`` and
        ``seasonal_period`` / ``seasonal_periods`` are accepted as aliases.
    seasonal : {"additive"}, default="additive"
        Kept for API clarity; only the additive model is implemented.

    Attributes
    ----------
    level_ : float
        Final level.
    trend_ : float
        Final slope (change per day).
    seasonal_ : ndarray of shape (season_length,)
        Final seasonal component for each phase.
    fitted_values_ : ndarray
        In-sample one-step-ahead fitted values.
    residuals_ : ndarray
        ``y - fitted_values_``.
    n_obs_ : int
        Number of observations used in fitting.
    """

    def __init__(
        self,
        alpha: float = 0.3,
        beta: float = 0.05,
        gamma: float = 0.3,
        season_length: int = 7,
        season: Optional[int] = None,
        seasonal_period: Optional[int] = None,
        seasonal_periods: Optional[int] = None,
        seasonal: str = "additive",
    ):
        for name, value in (("alpha", alpha), ("beta", beta), ("gamma", gamma)):
            if not 0.0 <= float(value) <= 1.0:
                raise ValueError(f"{name} must be in [0, 1]")
        if str(seasonal).lower() != "additive":
            raise ValueError("Only the additive Holt-Winters model is supported")

        if season is not None:
            season_length = season
        if seasonal_period is not None:
            season_length = seasonal_period
        if seasonal_periods is not None:
            season_length = seasonal_periods

        self.alpha = float(alpha)
        self.beta = float(beta)
        self.gamma = float(gamma)
        self.season_length = _positive_int(season_length, "season_length", minimum=2)
        self.seasonal = "additive"

    def fit(self, y: Sequence[float]) -> "HoltWinters":
        """Fit level, trend and seasonal states."""
        y = _as_1d_float(y)
        m = self.season_length
        if y.size < 2 * m:
            raise ValueError(
                "HoltWinters needs at least two full seasons of observations"
            )

        alpha, beta, gamma = self.alpha, self.beta, self.gamma

        # Initialise from the first two seasons.  The first season's mean is
        # the initial level; the change between season means gives the trend.
        first_season = y[:m]
        second_season = y[m : 2 * m]
        level = float(np.mean(first_season))
        trend = float((np.mean(second_season) - level) / m)

        seasonal = first_season - level
        seasonal = seasonal - np.mean(seasonal)

        fitted = np.empty(y.size, dtype=float)
        for t in range(m):
            fitted[t] = level + trend * t + seasonal[t]

        for t in range(m, y.size):
            phase = t % m
            fitted[t] = level + trend + seasonal[phase]

            new_level = alpha * (y[t] - seasonal[phase]) + (1.0 - alpha) * (
                level + trend
            )
            new_trend = beta * (new_level - level) + (1.0 - beta) * trend
            new_seasonal = gamma * (y[t] - new_level) + (1.0 - gamma) * seasonal[
                phase
            ]

            level = new_level
            trend = new_trend
            seasonal[phase] = new_seasonal

        self.level_ = level
        self.trend_ = trend
        self.seasonal_ = seasonal.copy()
        self.fitted_values_ = fitted
        self.residuals_ = y - fitted
        self.n_obs_ = y.size
        return self

    def forecast(self, horizon: int = 1) -> np.ndarray:
        """Forecast ``horizon`` future observations with the additive model."""
        _check_is_fitted(self, "level_", "HoltWinters")
        horizon = _check_forecast_horizon(horizon)
        m = self.season_length
        steps = np.arange(1, horizon + 1, dtype=float)
        phases = (self.n_obs_ + np.arange(horizon)) % m
        return self.level_ + steps * self.trend_ + self.seasonal_[phases]

    def predict(self, horizon: int = 1) -> np.ndarray:
        """Alias for :meth:`forecast`."""
        return self.forecast(horizon)


# ---------------------------------------------------------------------------
# Autoregressive model
# ---------------------------------------------------------------------------
class ARModel:
    """
    Autoregressive model ``AR(p)`` estimated by ordinary least squares.

    The model is::

        y_t = c + phi_1 y_{t-1} + ... + phi_p y_{t-p} + e_t

    The design matrix includes a leading column of ones, so the intercept is
    estimated together with the autoregressive coefficients.  Multi-step
    forecasts are **recursive**: each prediction is appended to the history and
    used as a lag for the next step.

    Parameters
    ----------
    p : int, default=1
        Number of autoregressive lags.

    Attributes
    ----------
    intercept_ : float
        Estimated constant ``c``.
    coef_ : ndarray of shape (p,)
        Estimated ``phi_1 ... phi_p``.
    params_ : ndarray of shape (p + 1,)
        ``[intercept_, coef_[0], ..., coef_[p - 1]]``.
    fitted_values_ : ndarray of shape (n - p,)
        In-sample fitted values aligned with ``y[p:]``.
    residuals_ : ndarray of shape (n - p,)
        In-sample residuals.
    history_ : ndarray
        Copy of the training series, kept for recursive forecasting.
    """

    def __init__(self, p: int = 1):
        self.p = _positive_int(p, "p")

    def _design_matrix(self, y: np.ndarray) -> np.ndarray:
        p = self.p
        n = y.size
        columns = [np.ones(n - p, dtype=float)]
        for lag in range(1, p + 1):
            columns.append(y[p - lag : n - lag])
        return np.column_stack(columns)

    def fit(self, y: Sequence[float]) -> "ARModel":
        """Estimate the intercept and autoregressive coefficients."""
        y = _as_1d_float(y)
        p = self.p
        if y.size <= p + 1:
            raise ValueError("ARModel needs more observations than p + 1")

        X = self._design_matrix(y)
        target = y[p:]
        params, _, _, _ = np.linalg.lstsq(X, target, rcond=None)

        self.intercept_ = float(params[0])
        self.coef_ = params[1:].copy()
        self.params_ = params.copy()
        self.fitted_values_ = X @ params
        self.residuals_ = target - self.fitted_values_
        self.history_ = y.copy()
        self.n_obs_ = y.size
        return self

    def forecast(self, horizon: int = 1) -> np.ndarray:
        """Recursively forecast ``horizon`` steps beyond the training data."""
        _check_is_fitted(self, "history_", "ARModel")
        horizon = _check_forecast_horizon(horizon)

        history = list(self.history_)
        predictions = []
        for _ in range(horizon):
            features = [history[-lag] for lag in range(1, self.p + 1)]
            value = self.intercept_ + float(np.dot(self.coef_, features))
            predictions.append(value)
            history.append(value)
        return np.asarray(predictions, dtype=float)

    def predict(self, horizon: int = 1) -> np.ndarray:
        """Alias for :meth:`forecast`."""
        return self.forecast(horizon)


# ---------------------------------------------------------------------------
# Leakage-free lag features
# ---------------------------------------------------------------------------
def _lag_feature_names(
    lags: Tuple[int, ...], rolling_windows: Tuple[int, ...]
) -> List[str]:
    names = [f"lag_{lag}" for lag in lags]
    for window in rolling_windows:
        names.append(f"rolling_mean_{window}")
        names.append(f"rolling_std_{window}")
    return names


def make_lag_features(
    y: Sequence[float],
    lags: Union[int, Iterable[int]] = (1, 2, 3, 7),
    rolling_windows: Union[int, Iterable[int]] = (7,),
    return_index: bool = False,
    windows: Optional[Union[int, Iterable[int]]] = None,
):
    """
    Build supervised lag and rolling features without look-ahead leakage.

    Row ``i`` corresponds to target time ``t = start + i``.  Every feature for
    that row uses observations strictly **before** ``t``:

    - ``lag_k = y[t - k]`` for each requested lag ``k``,
    - ``rolling_mean_w = mean(y[t - w : t])``,
    - ``rolling_std_w = std(y[t - w : t])`` (population standard deviation).

    The first usable target is ``max(max(lags), max(rolling_windows))`` so all
    requested history is available.  The target itself is never used as a
    feature.

    Parameters
    ----------
    y : array-like of shape (n,)
        Univariate time series.
    lags : int or iterable of int, default=(1, 2, 3, 7)
        Positive lag offsets. Lag 0 is rejected because it would leak the
        target.
    rolling_windows : int or iterable of int, default=(7,)
        Positive rolling-window lengths. Each window contributes a mean and a
        standard deviation column.
    return_index : bool, default=False
        If ``True``, also return the integer target indices. This is useful
        when the caller needs to align features with dates or an ordered
        train/test split.
    windows : int or iterable of int, optional
        Alias for ``rolling_windows``; takes precedence when supplied.

    Returns
    -------
    X : ndarray of shape (n_valid, n_features)
        Feature matrix.
    target : ndarray of shape (n_valid,)
        ``y[start:]``, aligned with the rows of ``X``.
    index : ndarray of shape (n_valid,)
        Target indices ``start, ..., n - 1``. Only returned when
        ``return_index=True``.
    """
    y = _as_1d_float(y)
    lags = _int_sequence(lags, "lags")

    # ``windows`` is a convenience alias for ``rolling_windows``.
    if windows is not None:
        rolling_windows = windows

    if isinstance(rolling_windows, (int, np.integer)) and not isinstance(
        rolling_windows, bool
    ):
        rolling_windows = (int(rolling_windows),)
    else:
        rolling_windows = tuple(int(w) for w in rolling_windows)
    if any(w < 1 for w in rolling_windows):
        raise ValueError("rolling window lengths must be positive")
    rolling_windows = tuple(sorted(set(rolling_windows)))

    max_lag = max(lags)
    max_window = max(rolling_windows, default=0)
    start = max(max_lag, max_window)
    if start >= y.size:
        raise ValueError(
            "Series is too short for the requested lags and rolling windows"
        )

    n_rows = y.size - start
    n_columns = len(lags) + 2 * len(rolling_windows)
    X = np.empty((n_rows, n_columns), dtype=float)

    column = 0
    for lag in lags:
        X[:, column] = y[start - lag : y.size - lag]
        column += 1

    if rolling_windows:
        from numpy.lib.stride_tricks import sliding_window_view

        for window in rolling_windows:
            windows = sliding_window_view(y, window)
            # windows[i] = y[i:i+window].  For target t the window is
            # y[t-window:t], i.e. windows[t-window].
            past = windows[start - window : y.size - window]
            X[:, column] = np.mean(past, axis=1)
            column += 1
            X[:, column] = np.std(past, axis=1)
            column += 1

    target = y[start:].copy()
    index = np.arange(start, y.size, dtype=np.int64)

    if return_index:
        return X, target, index
    return X, target


class LagRegression:
    """
    Linear regression on leakage-free lag and rolling features.

    The model is a transparent supervised-learning baseline::

        y_t = c + sum_k phi_k * y_{t-k}
                + sum_w (a_w * rolling_mean_w + b_w * rolling_std_w)

    Coefficients are estimated with ``numpy.linalg.lstsq``.  Forecasting is
    recursive: each one-step prediction is appended to the history before the
    next feature row is built.

    Parameters
    ----------
    lags : int or iterable of int, default=(1, 2, 3, 7)
        Autoregressive lag offsets.
    rolling_windows : int or iterable of int, default=(7,)
        Rolling-window lengths for mean and standard-deviation features.
    windows : int or iterable of int, optional
        Alias for ``rolling_windows``; takes precedence when supplied.

    Attributes
    ----------
    intercept_ : float
        Fitted intercept.
    coef_ : ndarray of shape (n_features,)
        Fitted feature coefficients.
    feature_names_ : list of str
        Column names in the same order as ``coef_``.
    history_ : ndarray
        Copy of the training series.
    fitted_values_ : ndarray
        In-sample fitted targets.
    residuals_ : ndarray
        In-sample residuals.
    """

    def __init__(
        self,
        lags: Union[int, Iterable[int]] = (1, 2, 3, 7),
        rolling_windows: Union[int, Iterable[int]] = (7,),
        windows: Optional[Union[int, Iterable[int]]] = None,
    ):
        self._lags_input = lags
        self._rolling_input = (
            rolling_windows if windows is None else windows
        )

    def _resolved_features(self) -> Tuple[Tuple[int, ...], Tuple[int, ...]]:
        lags = _int_sequence(self._lags_input, "lags")
        if isinstance(self._rolling_input, (int, np.integer)) and not isinstance(
            self._rolling_input, bool
        ):
            windows = (int(self._rolling_input),)
        else:
            windows = tuple(int(w) for w in self._rolling_input)
        if any(w < 1 for w in windows):
            raise ValueError("rolling window lengths must be positive")
        return lags, tuple(sorted(set(windows)))

    def fit(self, y: Sequence[float]) -> "LagRegression":
        """Fit the regression on all usable lagged rows."""
        y = _as_1d_float(y)
        lags, windows = self._resolved_features()
        X, target = make_lag_features(y, lags=lags, rolling_windows=windows)

        design = np.column_stack([np.ones(X.shape[0], dtype=float), X])
        params, _, _, _ = np.linalg.lstsq(design, target, rcond=None)

        self.lags_ = lags
        self.rolling_windows_ = windows
        self.intercept_ = float(params[0])
        self.coef_ = params[1:].copy()
        self.feature_names_ = _lag_feature_names(lags, windows)
        self.fitted_values_ = design @ params
        self.residuals_ = target - self.fitted_values_
        self.history_ = y.copy()
        self.n_obs_ = y.size
        return self

    def _row_features(self, values: np.ndarray, t: int) -> np.ndarray:
        features = [values[t - lag] for lag in self.lags_]
        for window in self.rolling_windows_:
            past = values[t - window : t]
            features.append(float(np.mean(past)))
            features.append(float(np.std(past)))
        return np.asarray(features, dtype=float)

    def forecast(self, horizon: int = 1) -> np.ndarray:
        """Recursively forecast ``horizon`` steps."""
        _check_is_fitted(self, "history_", "LagRegression")
        horizon = _check_forecast_horizon(horizon)

        values = np.concatenate([self.history_, np.empty(horizon, dtype=float)])
        predictions = np.empty(horizon, dtype=float)
        for step in range(horizon):
            t = self.history_.size + step
            features = self._row_features(values, t)
            prediction = self.intercept_ + float(np.dot(self.coef_, features))
            values[t] = prediction
            predictions[step] = prediction
        return predictions

    def predict(self, horizon: int = 1) -> np.ndarray:
        """Alias for :meth:`forecast`."""
        return self.forecast(horizon)


# ---------------------------------------------------------------------------
# Ordered split and backtesting
# ---------------------------------------------------------------------------
def time_train_test_split(
    y: Sequence[float],
    test_size: Union[int, float] = 0.2,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Split a time series into an ordered train/test pair **without shuffling**.

    The training set always contains the oldest observations and the test set
    the newest.  ``test_size`` may be:

    - an integer: exact number of test observations,
    - a float in ``(0, 1)``: fraction of observations, rounded up.

    Parameters
    ----------
    y : array-like of shape (n,)
        Time-ordered series.
    test_size : int or float, default=0.2
        Test-set size.

    Returns
    -------
    train, test : ndarray
        Consecutive, non-overlapping slices. The last training timestamp is
        immediately before the first test timestamp.
    """
    y = _as_1d_float(y)
    n = y.size

    if isinstance(test_size, (float, np.floating)) and not isinstance(
        test_size, bool
    ):
        if not 0.0 < float(test_size) < 1.0:
            raise ValueError("float test_size must be in (0, 1)")
        n_test = max(1, int(np.ceil(n * float(test_size))))
    else:
        n_test = _positive_int(test_size, "test_size")

    if n_test >= n:
        raise ValueError("test_size must leave at least one training observation")

    return y[:-n_test].copy(), y[-n_test:].copy()


def _forecast_from_model_fn(
    model_fn: Callable, train: np.ndarray, horizon: int
) -> np.ndarray:
    """Call a user-supplied model function and normalize its output."""
    # Prefer the (train, horizon) convention.  Fall back to (train) only when
    # the callable does not accept two arguments.
    try:
        signature = inspect.signature(model_fn)
        positional = [
            parameter
            for parameter in signature.parameters.values()
            if parameter.kind
            in (
                inspect.Parameter.POSITIONAL_ONLY,
                inspect.Parameter.POSITIONAL_OR_KEYWORD,
            )
        ]
        has_varargs = any(
            parameter.kind == inspect.Parameter.VAR_POSITIONAL
            for parameter in signature.parameters.values()
        )
    except (TypeError, ValueError):
        positional = []
        has_varargs = True

    if has_varargs or len(positional) >= 2:
        result = model_fn(train, horizon)
    else:
        result = model_fn(train)

    if hasattr(result, "forecast") and callable(result.forecast):
        result = result.forecast(horizon)
    elif hasattr(result, "predict") and callable(result.predict):
        result = result.predict(horizon)

    predictions = np.asarray(result, dtype=float).ravel()
    if predictions.size != horizon:
        raise ValueError(
            "model_fn must return exactly 'horizon' predictions; "
            f"got {predictions.size}"
        )
    if not np.all(np.isfinite(predictions)):
        raise ValueError("model_fn returned NaN or infinite predictions")
    return predictions


def rolling_origin_backtest(
    y: Sequence[float],
    model_fn: Callable,
    initial: int,
    horizon: int,
    step: int = 1,
) -> List[dict]:
    """
    Walk-forward validation with a rolling origin.

    Fold ``k`` trains on ``y[:start]`` and evaluates on the next ``horizon``
    observations.  The origin then moves forward by ``step`` and the process is
    repeated.  Future observations are never used for training.

    ``model_fn`` may be either:

    - ``model_fn(train, horizon) -> predictions``,
    - ``model_fn(train) -> fitted model`` where the model exposes
      ``forecast(horizon)`` or ``predict(horizon)``,
    - ``model_fn(train) -> predictions``.

    Parameters
    ----------
    y : array-like of shape (n,)
        Full time-ordered series.
    model_fn : callable
        Forecast strategy, as described above.
    initial : int
        Number of observations in the first training window.
    horizon : int
        Forecast horizon for every fold.
    step : int, default=1
        How far the origin advances between folds.

    Returns
    -------
    folds : list of dict
        One dictionary per fold with keys ``fold``, ``train_start``,
        ``train_end``, ``test_start``, ``test_end``, ``actual``, ``predicted``,
        ``mae``, ``rmse``, ``mape`` and ``smape``.  The number of folds is
        ``floor((n - initial - horizon) / step) + 1`` when at least one full
        fold fits.
    """
    y = _as_1d_float(y)
    initial = _positive_int(initial, "initial")
    horizon = _positive_int(horizon, "horizon")
    step = _positive_int(step, "step")

    if initial + horizon > y.size:
        raise ValueError(
            "initial + horizon must not exceed the length of the series"
        )

    folds = []
    fold_number = 0
    for start in range(initial, y.size - horizon + 1, step):
        train = y[:start]
        actual = y[start : start + horizon]
        predicted = _forecast_from_model_fn(model_fn, train, horizon)

        folds.append(
            {
                "fold": fold_number,
                "train_start": 0,
                "train_end": start,
                "test_start": start,
                "test_end": start + horizon,
                "actual": actual.copy(),
                "predicted": predicted.copy(),
                "mae": mae(actual, predicted),
                "rmse": rmse(actual, predicted),
                "mape": mape(actual, predicted),
                "smape": smape(actual, predicted),
            }
        )
        fold_number += 1

    return folds


# ---------------------------------------------------------------------------
# Forecast metrics
# ---------------------------------------------------------------------------
def _aligned_arrays(
    y_true: Sequence[float], y_pred: Sequence[float]
) -> Tuple[np.ndarray, np.ndarray]:
    actual = _as_1d_float(y_true, "y_true")
    predicted = _as_1d_float(y_pred, "y_pred")
    if actual.size != predicted.size:
        raise ValueError("y_true and y_pred must have the same length")
    return actual, predicted


def mae(y_true: Sequence[float], y_pred: Sequence[float]) -> float:
    """Mean absolute error: ``mean(|y - y_hat|)``."""
    actual, predicted = _aligned_arrays(y_true, y_pred)
    return float(np.mean(np.abs(actual - predicted)))


def rmse(y_true: Sequence[float], y_pred: Sequence[float]) -> float:
    """Root mean squared error: ``sqrt(mean((y - y_hat)^2))``."""
    actual, predicted = _aligned_arrays(y_true, y_pred)
    return float(np.sqrt(np.mean((actual - predicted) ** 2)))


def mape(y_true: Sequence[float], y_pred: Sequence[float]) -> float:
    """
    Mean absolute percentage error, returned in percent.

    Terms with ``y_true == 0`` are skipped because the percentage is
    undefined.  If every actual value is zero, the result is ``0.0`` when the
    predictions are also all zero and ``inf`` otherwise.
    """
    actual, predicted = _aligned_arrays(y_true, y_pred)
    denominator = np.abs(actual)
    nonzero = denominator > 0.0

    if not np.any(nonzero):
        if np.allclose(predicted, 0.0):
            return 0.0
        return float("inf")

    percentage = np.abs((actual[nonzero] - predicted[nonzero]) / denominator[nonzero])
    return float(np.mean(percentage) * 100.0)


def smape(y_true: Sequence[float], y_pred: Sequence[float]) -> float:
    """
    Symmetric mean absolute percentage error, returned in percent (0--200).

    Uses ``100 * mean(2|y - y_hat| / (|y| + |y_hat|))``.  A term where both
    actual and prediction are zero contributes zero error.
    """
    actual, predicted = _aligned_arrays(y_true, y_pred)
    denominator = np.abs(actual) + np.abs(predicted)
    numerator = 2.0 * np.abs(actual - predicted)
    terms = np.zeros_like(denominator)
    valid = denominator > 0.0
    terms[valid] = numerator[valid] / denominator[valid]
    return float(np.mean(terms) * 100.0)


# ---------------------------------------------------------------------------
# End-to-end demo
# ---------------------------------------------------------------------------
def _demo() -> None:
    """Run a small forecast comparison on synthetic outlet sales."""
    print("=" * 78)
    print("TIME SERIES FORECASTING: From-Scratch Demo")
    print("Synthetic Indonesian outlet sales")
    print("=" * 78)

    dates, sales = generate_outlet_sales(n_days=365, seed=42)
    print(f"Generated {sales.size} daily observations")
    print(f"Date range: {dates[0]} to {dates[-1]}")
    print(f"Sales range: {sales.min():.2f} to {sales.max():.2f}")
    print(f"Mean sales: {sales.mean():.2f}")

    horizon = 30
    train, test = time_train_test_split(sales, test_size=horizon)
    print(
        f"\nOrdered split: {train.size} train observations, "
        f"{test.size} test observations"
    )

    forecasts = {
        "Naive": naive_forecast(train, horizon),
        "Seasonal naive (7)": seasonal_naive_forecast(
            train, horizon, season=7
        ),
        "Moving average (7)": moving_average_forecast(
            train, horizon, window=7
        ),
        "Simple exp. smoothing": SimpleExpSmoothing(alpha=0.3)
        .fit(train)
        .forecast(horizon),
        "Holt-Winters additive": HoltWinters(
            alpha=0.3, beta=0.05, gamma=0.3, season_length=7
        )
        .fit(train)
        .forecast(horizon),
        "AR(7)": ARModel(p=7).fit(train).forecast(horizon),
        "Lag regression": LagRegression(
            lags=(1, 2, 3, 4, 5, 6, 7), rolling_windows=(7,)
        )
        .fit(train)
        .forecast(horizon),
    }

    print("\nTest-set metrics (lower is better):")
    print(f"{'Model':<26} {'MAE':>10} {'RMSE':>10} {'MAPE %':>10} {'SMAPE %':>10}")
    print("-" * 70)
    for name, prediction in forecasts.items():
        print(
            f"{name:<26} {mae(test, prediction):>10.2f} "
            f"{rmse(test, prediction):>10.2f} "
            f"{mape(test, prediction):>10.2f} "
            f"{smape(test, prediction):>10.2f}"
        )

    print("\nRolling-origin backtest (seasonal naive, initial=250, horizon=30):")
    folds = rolling_origin_backtest(
        sales,
        lambda history, h: seasonal_naive_forecast(history, h, season=7),
        initial=250,
        horizon=30,
        step=30,
    )
    print(f"Number of folds: {len(folds)}")
    for fold in folds:
        print(
            f"  fold {fold['fold']}: train_end={fold['train_end']:>3}, "
            f"test=[{fold['test_start']}, {fold['test_end']}), "
            f"MAE={fold['mae']:.2f}"
        )

    print("\nDone. See the README and exercise notebook for guided practice.")


if __name__ == "__main__":
    _demo()
