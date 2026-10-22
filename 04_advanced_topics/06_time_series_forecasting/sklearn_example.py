"""
Time Series Forecasting: scikit-learn Examples
==============================================

This script shows how to turn a univariate sales series into a supervised
learning problem and evaluate the models with **time-series cross-validation**:

1. Build leakage-free lag and rolling features with ``make_lag_features``.
2. Use ``TimeSeriesSplit`` so every validation fold comes after its training
   fold (never shuffle time series).
3. Compare ``GradientBoostingRegressor`` and ``RandomForestRegressor`` on a
   one-step-ahead test window.
4. Add the from-scratch ``LagRegression`` baseline.
5. Fit statsmodels' additive ``ExponentialSmoothing`` when statsmodels is
   available; otherwise print a clear skip message.

The script is headless-safe and only prints results.

Author: ML-Algo Repository
"""

from __future__ import annotations

import os
import sys

import numpy as np
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.model_selection import TimeSeriesSplit

# Import the shared implementation from this directory.
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from implementation import (  # noqa: E402
    LagRegression,
    generate_outlet_sales,
    make_lag_features,
    mae,
    mape,
    rmse,
    seasonal_naive_forecast,
    smape,
    time_train_test_split,
)


LAGS = (1, 2, 3, 4, 5, 6, 7, 14)
ROLLING_WINDOWS = (7, 14)


def build_supervised_dataset(sales, n_train, lags=LAGS, rolling_windows=ROLLING_WINDOWS):
    """Create aligned feature matrices for ordered train/test targets."""
    X, target, target_index = make_lag_features(
        sales,
        lags=lags,
        rolling_windows=rolling_windows,
        return_index=True,
    )
    train_mask = target_index < n_train
    X_train = X[train_mask]
    y_train = target[train_mask]
    X_test = X[~train_mask]
    y_test = target[~train_mask]
    return X_train, y_train, X_test, y_test


def cross_validate_model(model_factory, X_train, y_train, n_splits=5):
    """Run expanding-window TimeSeriesSplit and return fold MAE values."""
    splitter = TimeSeriesSplit(n_splits=n_splits)
    fold_mae = []
    for train_index, validation_index in splitter.split(X_train):
        model = model_factory()
        model.fit(X_train[train_index], y_train[train_index])
        prediction = model.predict(X_train[validation_index])
        fold_mae.append(mae(y_train[validation_index], prediction))
    return np.asarray(fold_mae, dtype=float)


def evaluate_model(name, model, X_train, y_train, X_test, y_test):
    """Fit a model on the training window and print its test metrics."""
    model.fit(X_train, y_train)
    prediction = model.predict(X_test)
    print(
        f"{name:<28} {mae(y_test, prediction):>9.2f} "
        f"{rmse(y_test, prediction):>9.2f} "
        f"{mape(y_test, prediction):>9.2f} "
        f"{smape(y_test, prediction):>9.2f}"
    )
    return prediction


def example_1_supervised_features(sales, n_train):
    """Show the feature layout and the no-shuffle time-series split."""
    print("\n" + "=" * 78)
    print("Example 1: Leakage-free supervised features")
    print("=" * 78)

    X_train, y_train, X_test, y_test = build_supervised_dataset(sales, n_train)
    print(f"Lags:                {LAGS}")
    print(f"Rolling windows:     {ROLLING_WINDOWS}")
    print(f"Training rows:       {X_train.shape[0]} x {X_train.shape[1]}")
    print(f"Test rows:           {X_test.shape[0]} x {X_test.shape[1]}")
    print(
        "Every feature row uses only observations strictly before its target "
        "timestamp."
    )

    splitter = TimeSeriesSplit(n_splits=5)
    print("\nTimeSeriesSplit fold sizes (train -> validation):")
    for fold, (train_index, validation_index) in enumerate(
        splitter.split(X_train), start=1
    ):
        print(
            f"  fold {fold}: {train_index.size:>4} -> "
            f"{validation_index.size:>3} observations"
        )

    print("\nFirst feature row (lag_1 is the most recent known value):")
    print(np.round(X_train[0], 2))


def example_2_gradient_boosting(X_train, y_train, X_test, y_test):
    """Evaluate GradientBoostingRegressor with TimeSeriesSplit."""
    print("\n" + "=" * 78)
    print("Example 2: GradientBoostingRegressor")
    print("=" * 78)

    def factory():
        return GradientBoostingRegressor(
            n_estimators=200,
            learning_rate=0.05,
            max_depth=3,
            min_samples_leaf=5,
            random_state=42,
        )

    fold_mae = cross_validate_model(factory, X_train, y_train, n_splits=5)
    print("TimeSeriesSplit validation MAE by fold:")
    print("  " + ", ".join(f"{value:.2f}" for value in fold_mae))
    print(f"  mean={fold_mae.mean():.2f}, std={fold_mae.std():.2f}")

    print("\nFinal test window:")
    print(f"{'Model':<28} {'MAE':>9} {'RMSE':>9} {'MAPE %':>9} {'SMAPE %':>9}")
    print("-" * 68)
    evaluate_model("GradientBoosting", factory(), X_train, y_train, X_test, y_test)


def example_3_random_forest(X_train, y_train, X_test, y_test):
    """Evaluate RandomForestRegressor with TimeSeriesSplit."""
    print("\n" + "=" * 78)
    print("Example 3: RandomForestRegressor")
    print("=" * 78)

    def factory():
        return RandomForestRegressor(
            n_estimators=300,
            max_depth=None,
            min_samples_leaf=2,
            random_state=42,
            n_jobs=-1,
        )

    fold_mae = cross_validate_model(factory, X_train, y_train, n_splits=5)
    print("TimeSeriesSplit validation MAE by fold:")
    print("  " + ", ".join(f"{value:.2f}" for value in fold_mae))
    print(f"  mean={fold_mae.mean():.2f}, std={fold_mae.std():.2f}")

    print("\nFinal test window:")
    print(f"{'Model':<28} {'MAE':>9} {'RMSE':>9} {'MAPE %':>9} {'SMAPE %':>9}")
    print("-" * 68)
    evaluate_model("RandomForest", factory(), X_train, y_train, X_test, y_test)


def example_4_from_scratch_baselines(train, test):
    """Compare simple baselines and the NumPy LagRegression model."""
    print("\n" + "=" * 78)
    print("Example 4: From-scratch baselines on the same horizon")
    print("=" * 78)

    horizon = test.size
    lag_model = LagRegression(
        lags=LAGS, rolling_windows=ROLLING_WINDOWS
    ).fit(train)

    models = {
        "Seasonal naive (7)": seasonal_naive_forecast(
            train, horizon, season=7
        ),
        "LagRegression (NumPy)": lag_model.forecast(horizon),
    }

    print(f"{'Model':<28} {'MAE':>9} {'RMSE':>9} {'MAPE %':>9} {'SMAPE %':>9}")
    print("-" * 68)
    for name, prediction in models.items():
        print(
            f"{name:<28} {mae(test, prediction):>9.2f} "
            f"{rmse(test, prediction):>9.2f} "
            f"{mape(test, prediction):>9.2f} "
            f"{smape(test, prediction):>9.2f}"
        )


def example_5_statsmodels(train, test):
    """Fit statsmodels Holt-Winters when the optional package is installed."""
    print("\n" + "=" * 78)
    print("Example 5: statsmodels ExponentialSmoothing (optional)")
    print("=" * 78)

    try:
        from statsmodels.tsa.holtwinters import ExponentialSmoothing
    except ImportError:
        print(
            "statsmodels is not installed - skipping this optional comparison. "
            "Install it with: pip install statsmodels"
        )
        return None

    horizon = test.size
    model = ExponentialSmoothing(
        train,
        trend="add",
        seasonal="add",
        seasonal_periods=7,
        initialization_method="estimated",
    )
    fitted = model.fit(optimized=True)
    prediction = np.asarray(fitted.forecast(horizon), dtype=float)

    print(f"{'Model':<28} {'MAE':>9} {'RMSE':>9} {'MAPE %':>9} {'SMAPE %':>9}")
    print("-" * 68)
    print(
        f"{'statsmodels Holt-Winters':<28} {mae(test, prediction):>9.2f} "
        f"{rmse(test, prediction):>9.2f} "
        f"{mape(test, prediction):>9.2f} "
        f"{smape(test, prediction):>9.2f}"
    )
    return prediction


def main():
    """Run all scikit-learn and optional statsmodels examples."""
    print("=" * 78)
    print("TIME SERIES FORECASTING: scikit-learn Examples")
    print("Synthetic Indonesian outlet sales")
    print("=" * 78)

    dates, sales = generate_outlet_sales(n_days=730, seed=42)
    horizon = 30
    train, test = time_train_test_split(sales, test_size=horizon)
    n_train = train.size

    print(f"Observations: {sales.size} ({dates[0]} to {dates[-1]})")
    print(f"Ordered split: {n_train} train / {test.size} test")
    print("Cross-validation: expanding-window TimeSeriesSplit, never shuffled")

    example_1_supervised_features(sales, n_train)
    X_train, y_train, X_test, y_test = build_supervised_dataset(sales, n_train)
    example_2_gradient_boosting(X_train, y_train, X_test, y_test)
    example_3_random_forest(X_train, y_train, X_test, y_test)
    example_4_from_scratch_baselines(train, test)
    example_5_statsmodels(train, test)

    print("\nKey takeaway: keep the test window in the future, build features "
          "only from the past, and use TimeSeriesSplit rather than random K-fold.")


if __name__ == "__main__":
    main()
