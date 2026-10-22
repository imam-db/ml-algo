"""
Tests for the XGBoost implementation and tutorial helpers.
"""

import importlib.util
import os
import sys

import numpy as np
import pytest
from sklearn.datasets import make_regression
from sklearn.utils import Bunch

# The implementation imports xgboost at module import time.
xgb = pytest.importorskip("xgboost")

# Keep plots headless when the tutorial functions are exercised.
os.environ.setdefault("MPLBACKEND", "Agg")

# Load implementation module (same pattern as the other tests in this folder).
spec = importlib.util.spec_from_file_location(
    "xgboost_impl",
    os.path.join(os.path.dirname(__file__), '..', '04_advanced_topics',
                 '04_xgboost', 'implementation.py')
)
xgboost_impl = importlib.util.module_from_spec(spec)
spec.loader.exec_module(xgboost_impl)


@pytest.fixture(scope="module")
def classification_result():
    """Train the classification demo once and share the fitted model."""
    return xgboost_impl.xgboost_classification_demo(show=False)


def test_implementation_exposes_tutorial_api():
    """The implementation should expose its datasets and demo entry points."""
    expected = {
        "create_synthetic_datasets",
        "xgboost_classification_demo",
        "xgboost_regression_demo",
        "hyperparameter_tuning_demo",
        "cross_validation_demo",
        "main",
    }
    assert expected.issubset(set(dir(xgboost_impl)))


def test_create_synthetic_datasets_shapes():
    """Synthetic classification and regression datasets have expected shapes."""
    (X_clf, y_clf), (X_reg, y_reg) = xgboost_impl.create_synthetic_datasets()

    assert X_clf.shape == (1000, 20)
    assert y_clf.shape == (1000,)
    assert set(np.unique(y_clf)) == {0, 1}

    assert X_reg.shape == (1000, 10)
    assert y_reg.shape == (1000,)
    assert np.issubdtype(y_reg.dtype, np.floating)
    assert np.std(y_reg) > 0


def test_create_synthetic_datasets_is_reproducible():
    """The synthetic data generator uses a fixed random seed."""
    (X_clf_1, y_clf_1), (X_reg_1, y_reg_1) = xgboost_impl.create_synthetic_datasets()
    (X_clf_2, y_clf_2), (X_reg_2, y_reg_2) = xgboost_impl.create_synthetic_datasets()

    np.testing.assert_allclose(X_clf_1, X_clf_2)
    np.testing.assert_allclose(y_clf_1, y_clf_2)
    np.testing.assert_allclose(X_reg_1, X_reg_2)
    np.testing.assert_allclose(y_reg_1, y_reg_2)


def test_classification_demo_returns_fitted_model(classification_result):
    """The classification demo returns a trained classifier."""
    model, feature_importance = classification_result

    assert isinstance(model, xgb.XGBClassifier)
    assert hasattr(model, "classes_")
    assert len(model.classes_) == 2
    assert model.get_booster() is not None
    assert feature_importance.shape == (30, 2)


def test_classification_demo_feature_importance(classification_result):
    """Feature importance is reported for every breast-cancer feature."""
    _, feature_importance = classification_result

    assert list(feature_importance.columns) == ["feature", "importance"]
    assert feature_importance["importance"].notna().all()
    assert (feature_importance["importance"] >= 0).all()
    assert feature_importance["importance"].sum() == pytest.approx(1.0, abs=1e-5)
    # The demo sorts importances in descending order.
    assert feature_importance["importance"].is_monotonic_decreasing


def test_classification_demo_predictions_are_valid(classification_result):
    """The fitted classifier outputs valid labels and probabilities."""
    model, _ = classification_result
    data = xgboost_impl.load_breast_cancer()
    X_test = data.data[:50]

    predictions = model.predict(X_test)
    probabilities = model.predict_proba(X_test)

    assert set(np.unique(predictions)).issubset({0, 1})
    assert probabilities.shape == (50, 2)
    np.testing.assert_allclose(probabilities.sum(axis=1), 1.0, rtol=1e-5)


def test_regression_demo_with_synthetic_housing_data(monkeypatch):
    """The regression demo trains and reports importances on supplied housing data."""
    X, y = make_regression(n_samples=400, n_features=8, noise=0.1, random_state=42)
    fake_housing = Bunch(
        data=X,
        target=y,
        feature_names=[f"feature_{i}" for i in range(X.shape[1])],
        DESCR="Synthetic housing data for tests",
    )
    monkeypatch.setattr(
        xgboost_impl, "fetch_california_housing", lambda: fake_housing
    )

    model, feature_importance = xgboost_impl.xgboost_regression_demo(show=False)

    assert isinstance(model, xgb.XGBRegressor)
    assert feature_importance.shape == (X.shape[1], 2)
    assert list(feature_importance.columns) == ["feature", "importance"]
    assert (feature_importance["importance"] >= 0).all()
    assert feature_importance["importance"].sum() == pytest.approx(1.0, abs=1e-5)

    predictions = model.predict(X)
    assert predictions.shape == (len(X),)
    assert np.isfinite(predictions).all()
    # The synthetic target is learnable, so the fitted model should explain it.
    assert model.score(X, y) > 0.5


def test_cross_validation_demo_returns_rmse_columns(monkeypatch):
    """Native XGBoost cross-validation returns the expected RMSE statistics."""
    _, (X_reg, y_reg) = xgboost_impl.create_synthetic_datasets()
    small = ((None, None), (X_reg[:160], y_reg[:160]))
    monkeypatch.setattr(xgboost_impl, "create_synthetic_datasets", lambda: small)

    cv_results = xgboost_impl.cross_validation_demo(show=False)

    assert not cv_results.empty
    assert "train-rmse-mean" in cv_results.columns
    assert "test-rmse-mean" in cv_results.columns
    assert np.isfinite(cv_results["test-rmse-mean"]).all()
    assert (cv_results["test-rmse-mean"] >= 0).all()


def test_main_dispatches_all_demos_headless(monkeypatch):
    """`--no_show` is forwarded to every demo so CI does not open windows."""
    calls = []

    def fake_demo(name, result):
        def demo(show=True):
            calls.append((name, show))
            return result
        return demo

    monkeypatch.setattr(
        xgboost_impl, "xgboost_classification_demo",
        fake_demo("classification", (None, None))
    )
    monkeypatch.setattr(
        xgboost_impl, "xgboost_regression_demo",
        fake_demo("regression", (None, None))
    )
    monkeypatch.setattr(
        xgboost_impl, "hyperparameter_tuning_demo",
        fake_demo("tuning", (None, None))
    )
    monkeypatch.setattr(
        xgboost_impl, "cross_validation_demo",
        fake_demo("cross_validation", None)
    )
    monkeypatch.setattr(sys, "argv", ["implementation.py", "--no_show"])

    xgboost_impl.main()

    assert calls == [
        ("classification", False),
        ("regression", False),
        ("tuning", False),
        ("cross_validation", False),
    ]


def test_main_defaults_to_showing_plots(monkeypatch):
    """Without `--no_show`, demos are allowed to display plots."""
    calls = []

    def fake_demo(name, result):
        def demo(show=True):
            calls.append((name, show))
            return result
        return demo

    monkeypatch.setattr(
        xgboost_impl, "xgboost_classification_demo",
        fake_demo("classification", (None, None))
    )
    monkeypatch.setattr(
        xgboost_impl, "xgboost_regression_demo",
        fake_demo("regression", (None, None))
    )
    monkeypatch.setattr(
        xgboost_impl, "hyperparameter_tuning_demo",
        fake_demo("tuning", (None, None))
    )
    monkeypatch.setattr(
        xgboost_impl, "cross_validation_demo",
        fake_demo("cross_validation", None)
    )
    monkeypatch.setattr(sys, "argv", ["implementation.py"])

    xgboost_impl.main()

    assert calls == [
        ("classification", True),
        ("regression", True),
        ("tuning", True),
        ("cross_validation", True),
    ]
