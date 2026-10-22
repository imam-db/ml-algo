"""
Tests for the time-series forecasting module.

The tests exercise the public API in ``implementation.py``: synthetic data
generation, ordered splits, leakage-free lag features, classical forecasters,
AR coefficient recovery, metrics and walk-forward backtesting.
"""

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pytest

MODULE_DIR = (
    Path(__file__).resolve().parent.parent
    / "04_advanced_topics"
    / "06_time_series_forecasting"
)
sys.path.insert(0, str(MODULE_DIR))

# Several test modules export a module named ``implementation``.  Loading this
# one under a unique name keeps the suite independent of collection order while
# still following the sys.path-based import style used by the other tests.
_SPEC = importlib.util.spec_from_file_location(
    "time_series_forecasting_implementation",
    MODULE_DIR / "implementation.py",
)
_IMPL = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_IMPL)

ARModel = _IMPL.ARModel
HoltWinters = _IMPL.HoltWinters
LagRegression = _IMPL.LagRegression
SimpleExpSmoothing = _IMPL.SimpleExpSmoothing
generate_outlet_sales = _IMPL.generate_outlet_sales
mae = _IMPL.mae
mape = _IMPL.mape
make_lag_features = _IMPL.make_lag_features
moving_average_forecast = _IMPL.moving_average_forecast
naive_forecast = _IMPL.naive_forecast
rmse = _IMPL.rmse
rolling_origin_backtest = _IMPL.rolling_origin_backtest
seasonal_naive_forecast = _IMPL.seasonal_naive_forecast
smape = _IMPL.smape
time_train_test_split = _IMPL.time_train_test_split


def _weekly_series(n=210, seed=0):
    """Deterministic trend + weekly seasonal series with small noise."""
    rng = np.random.default_rng(seed)
    t = np.arange(n, dtype=float)
    weekly = 40.0 * np.sin(2.0 * np.pi * t / 7.0)
    harmonic = 15.0 * np.cos(4.0 * np.pi * t / 7.0)
    return 500.0 + 0.5 * t + weekly + harmonic + rng.normal(0.0, 5.0, n)


class TestDataGeneration:
    def test_shapes_dates_and_reproducibility(self):
        dates, values = generate_outlet_sales(n_days=400, seed=42)
        dates_again, values_again = generate_outlet_sales(400, 42)
        _, values_other = generate_outlet_sales(400, 7)

        assert dates.shape == (400,)
        assert values.shape == (400,)
        assert np.issubdtype(dates.dtype, np.datetime64)
        np.testing.assert_array_equal(dates, dates_again)
        np.testing.assert_allclose(values, values_again)
        assert not np.allclose(values, values_other)
        assert np.all(np.diff(dates).astype(int) == 1)
        assert np.all(np.isfinite(values))
        assert np.all(values > 0)

    def test_contains_weekly_and_payday_structure(self):
        dates, values = generate_outlet_sales(n_days=365, seed=1)

        # Average by weekday: the weekend factor should lift Sat/Sun above
        # midweek observations.
        epoch_days = dates.astype("datetime64[D]").astype(np.int64)
        weekday = (epoch_days + 3) % 7
        weekday_means = np.array(
            [values[weekday == day].mean() for day in range(7)]
        )
        assert weekday_means[5] > weekday_means[2]

        # Average by day of month: the payday window should be elevated.
        month_start = dates.astype("datetime64[M]")
        day_of_month = (dates - month_start).astype(np.int64) + 1
        payday = np.isin(day_of_month, [25, 26, 1, 2])
        ordinary = np.isin(day_of_month, [10, 11, 12, 13])
        assert values[payday].mean() > values[ordinary].mean()


class TestOrderedSplit:
    def test_split_is_chronological_and_disjoint(self):
        y = np.arange(100, dtype=float)
        train, test = time_train_test_split(y, test_size=20)

        assert train.shape == (80,)
        assert test.shape == (20,)
        np.testing.assert_array_equal(train, y[:80])
        np.testing.assert_array_equal(test, y[80:])
        assert train[-1] < test[0]
        assert len(np.intersect1d(train, test)) == 0

    def test_fractional_split_uses_ceiling(self):
        y = np.arange(101, dtype=float)
        train, test = time_train_test_split(y, test_size=0.2)
        assert test.size == 21
        assert train.size == 80
        np.testing.assert_array_equal(test, y[-21:])

    def test_split_rejects_bad_sizes(self):
        y = np.arange(10, dtype=float)
        with pytest.raises(ValueError):
            time_train_test_split(y, test_size=10)
        with pytest.raises(ValueError):
            time_train_test_split(y, test_size=0.0)


class TestLagFeatures:
    def test_first_row_uses_only_past_values(self):
        y = np.arange(1, 61, dtype=float)
        X, target = make_lag_features(
            y, lags=(1, 2, 7), rolling_windows=(7,)
        )

        start = 7
        assert X.shape == (60 - start, 5)
        assert target[0] == y[start]
        np.testing.assert_allclose(X[0, :3], [y[6], y[5], y[0]])
        assert X[0, 3] == pytest.approx(np.mean(y[0:7]))
        assert X[0, 4] == pytest.approx(np.std(y[0:7]))

    def test_no_look_ahead_leakage(self):
        y = np.arange(1, 81, dtype=float)
        X, target, index = make_lag_features(
            y, lags=(1, 3, 7), rolling_windows=(7, 14), return_index=True
        )

        changed = y.copy()
        changed[40:] = changed[40:] + 10_000.0
        X_changed, target_changed, _ = make_lag_features(
            changed,
            lags=(1, 3, 7),
            rolling_windows=(7, 14),
            return_index=True,
        )

        # Rows whose target is before the changed region must be identical.
        safe = target < 40
        np.testing.assert_allclose(X[safe], X_changed[safe])
        np.testing.assert_allclose(target[safe], target_changed[safe])

        # Features must only reference strictly smaller timestamps.
        assert np.all(index >= 14)
        np.testing.assert_allclose(X[:, 0], y[index - 1])

    def test_lag_zero_is_rejected(self):
        with pytest.raises(ValueError):
            make_lag_features(np.arange(20.0), lags=(0, 1))


class TestBaselines:
    def test_naive_repeats_last_value(self):
        y = np.array([1.0, 2.0, 3.0])
        np.testing.assert_allclose(naive_forecast(y, 4), [3.0] * 4)

    def test_seasonal_naive_repeats_the_season(self):
        y = np.arange(14, dtype=float)
        prediction = seasonal_naive_forecast(y, horizon=10, season=7)
        np.testing.assert_allclose(
            prediction, [7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 7.0, 8.0, 9.0]
        )

    def test_moving_average_is_flat_mean(self):
        y = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        prediction = moving_average_forecast(y, horizon=3, window=3)
        np.testing.assert_allclose(prediction, [4.0, 4.0, 4.0])


class TestSmoothingModels:
    def test_simple_exp_smoothing_updates_level(self):
        y = np.array([10.0, 12.0, 14.0, 16.0])
        model = SimpleExpSmoothing(alpha=0.5).fit(y)
        # level: 10 -> 11 -> 12.5 -> 14.25
        assert model.level_ == pytest.approx(14.25)
        np.testing.assert_allclose(model.forecast(3), [14.25] * 3)
        assert model.fitted_values_.shape == y.shape

    def test_holt_winters_and_seasonal_naive_beat_naive(self):
        y = _weekly_series()
        train, test = y[:-28], y[-28:]

        naive_error = mae(test, naive_forecast(train, 28))
        seasonal_error = mae(
            test, seasonal_naive_forecast(train, 28, season=7)
        )

        model = HoltWinters(
            alpha=0.3, beta=0.05, gamma=0.3, season_length=7
        ).fit(train)
        hw_prediction = model.forecast(28)
        hw_error = mae(test, hw_prediction)

        assert seasonal_error < naive_error
        assert hw_error < naive_error
        assert hw_prediction.shape == (28,)
        assert np.all(np.isfinite(hw_prediction))
        assert model.seasonal_.shape == (7,)


class TestARModel:
    def test_recovers_known_ar2_coefficients(self):
        rng = np.random.default_rng(42)
        n = 6000
        y = np.zeros(n)
        true_intercept = 5.0
        true_coefficients = np.array([0.6, -0.25])
        for t in range(2, n):
            y[t] = (
                true_intercept
                + true_coefficients[0] * y[t - 1]
                + true_coefficients[1] * y[t - 2]
                + rng.normal(0.0, 0.5)
            )

        model = ARModel(p=2).fit(y)

        assert model.intercept_ == pytest.approx(true_intercept, abs=0.15)
        np.testing.assert_allclose(
            model.coef_, true_coefficients, atol=0.05
        )
        assert model.coef_.shape == (2,)

        forecast = model.forecast(12)
        assert forecast.shape == (12,)
        assert np.all(np.isfinite(forecast))

    def test_fit_uses_intercept_and_lags(self):
        y = np.array([2.0, 4.0, 6.0, 8.0, 10.0, 12.0])
        model = ARModel(1).fit(y)
        assert model.intercept_ == pytest.approx(2.0, abs=1e-8)
        assert model.coef_[0] == pytest.approx(1.0, abs=1e-8)


class TestLagRegression:
    def test_fit_and_recursive_forecast(self):
        y = _weekly_series(n=180)
        model = LagRegression(
            lags=(1, 2, 3, 7), rolling_windows=(7,)
        ).fit(y)

        assert model.coef_.shape == (6,)
        assert len(model.feature_names_) == 6
        prediction = model.forecast(14)
        assert prediction.shape == (14,)
        assert np.all(np.isfinite(prediction))


class TestMetrics:
    def test_hand_computed_metrics(self):
        actual = np.array([100.0, 200.0, 300.0])
        predicted = np.array([110.0, 190.0, 330.0])

        assert mae(actual, predicted) == pytest.approx(50.0 / 3.0)
        assert rmse(actual, predicted) == pytest.approx(np.sqrt(1100.0 / 3.0))
        assert mape(actual, predicted) == pytest.approx(25.0 / 3.0)

        expected_smape = (
            2.0 * 10.0 / 210.0
            + 2.0 * 10.0 / 390.0
            + 2.0 * 30.0 / 630.0
        ) / 3.0 * 100.0
        assert smape(actual, predicted) == pytest.approx(expected_smape)

    def test_perfect_forecast_scores_zero(self):
        y = np.array([10.0, 20.0, 30.0])
        assert mae(y, y) == 0.0
        assert rmse(y, y) == 0.0
        assert mape(y, y) == 0.0
        assert smape(y, y) == 0.0

    def test_zero_actuals_are_handled(self):
        assert mape([0.0, 100.0], [0.0, 110.0]) == pytest.approx(10.0)
        assert mape([0.0, 0.0], [0.0, 0.0]) == 0.0


class TestBacktesting:
    def test_fold_count_and_walk_forward_boundaries(self):
        y = np.arange(100, dtype=float)

        def model_fn(train, horizon):
            return np.full(horizon, train[-1])

        folds = rolling_origin_backtest(
            y, model_fn, initial=60, horizon=10, step=5
        )

        # Origins: 60, 65, 70, 75, 80, 85, 90 -> seven complete folds.
        assert len(folds) == 7
        starts = [fold["test_start"] for fold in folds]
        assert starts == [60, 65, 70, 75, 80, 85, 90]
        for fold in folds:
            assert fold["actual"].shape == (10,)
            assert fold["predicted"].shape == (10,)
            assert fold["train_end"] == fold["test_start"]
            assert fold["test_end"] == fold["test_start"] + 10
            assert fold["mae"] >= 0.0

    def test_accepts_fitted_model_factory(self):
        y = _weekly_series(n=160)

        def model_fn(train):
            return ARModel(2).fit(train)

        folds = rolling_origin_backtest(
            y, model_fn, initial=120, horizon=7, step=7
        )
        assert len(folds) == 5
        assert all(fold["predicted"].shape == (7,) for fold in folds)
