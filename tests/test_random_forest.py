"""
Tests for Random Forest implementation
Compares from-scratch implementation with sklearn
"""
import numpy as np
import pytest
from sklearn.ensemble import RandomForestClassifier as SklearnRFC
from sklearn.ensemble import RandomForestRegressor as SklearnRFR
from sklearn.model_selection import train_test_split
from sklearn.datasets import make_classification, make_regression
import sys
import os
import importlib.util

# Load implementation module
spec = importlib.util.spec_from_file_location(
    "random_forest_impl",
    os.path.join(os.path.dirname(__file__), '..', '01_supervised_learning', '04_random_forest', 'implementation.py')
)
random_forest_impl = importlib.util.module_from_spec(spec)
spec.loader.exec_module(random_forest_impl)
RandomForestClassifier = random_forest_impl.RandomForestClassifier
RandomForestRegressor = random_forest_impl.RandomForestRegressor


def test_random_forest_classifier_accuracy():
    """Test RF classifier accuracy is reasonable"""
    # Generate sample data
    X, y = make_classification(n_samples=300, n_features=6, n_informative=4, 
                               n_redundant=2, random_state=42)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Train our implementation
    our_model = RandomForestClassifier(
        n_estimators=50,
        max_depth=8,
        random_state=42,
        verbose=0
    )
    our_model.fit(X_train, y_train)
    
    # Train sklearn
    sklearn_model = SklearnRFC(
        n_estimators=50,
        max_depth=8,
        random_state=42
    )
    sklearn_model.fit(X_train, y_train)
    
    # Compare accuracy
    our_accuracy = our_model.score(X_test, y_test)
    sklearn_accuracy = sklearn_model.score(X_test, y_test)
    
    # Should be reasonably close
    assert abs(our_accuracy - sklearn_accuracy) < 0.15
    assert our_accuracy > 0.7


def test_random_forest_oob_score():
    """Test OOB score is close to sklearn"""
    # Generate sample data
    X, y = make_classification(n_samples=200, n_features=5, random_state=42)
    
    # Train our implementation
    our_model = RandomForestClassifier(
        n_estimators=50,
        max_depth=6,
        bootstrap=True,
        oob_score=True,
        random_state=42,
        verbose=0
    )
    our_model.fit(X, y)
    
    # Train sklearn
    sklearn_model = SklearnRFC(
        n_estimators=50,
        max_depth=6,
        bootstrap=True,
        oob_score=True,
        random_state=42
    )
    sklearn_model.fit(X, y)
    
    # OOB scores should be close (within 0.15)
    print(f"Our OOB: {our_model.oob_score_}, sklearn OOB: {sklearn_model.oob_score_}")
    assert abs(our_model.oob_score_ - sklearn_model.oob_score_) < 0.15
    
    # OOB score should be reasonable
    assert our_model.oob_score_ > 0.6


def test_random_forest_feature_importances():
    """Test feature importances are non-zero and sum to 1"""
    X, y = make_classification(n_samples=200, n_features=5, random_state=42)
    
    model = RandomForestClassifier(
        n_estimators=30,
        max_depth=5,
        random_state=42,
        verbose=0
    )
    model.fit(X, y)
    
    # Feature importances should exist
    assert model.feature_importances_ is not None
    assert len(model.feature_importances_) == 5
    
    # Should have at least some non-zero importances
    assert np.sum(model.feature_importances_ > 0) > 0
    
    # Should sum to 1
    importance_sum = np.sum(model.feature_importances_)
    assert importance_sum == pytest.approx(1.0, abs=1e-6) or importance_sum == 0


def test_random_forest_regressor():
    """Test RF regressor is implemented and works"""
    # Generate sample data
    X, y = make_regression(n_samples=200, n_features=5, noise=10, random_state=42)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Train our implementation
    our_model = RandomForestRegressor(
        n_estimators=50,
        max_depth=8,
        max_features=1.0,  # sklearn default for regression
        random_state=42,
        verbose=0
    )
    our_model.fit(X_train, y_train)
    
    # Make predictions
    predictions = our_model.predict(X_test)
    
    # Check predictions are numeric and reasonable
    assert predictions.shape == (len(X_test),)
    assert np.all(np.isfinite(predictions))
    
    # Check R² is reasonable
    r2 = our_model.score(X_test, y_test)
    assert r2 > 0.7
    
    # Compare with sklearn
    sklearn_model = SklearnRFR(
        n_estimators=50,
        max_depth=8,
        max_features=1.0,
        random_state=42
    )
    sklearn_model.fit(X_train, y_train)
    sklearn_r2 = sklearn_model.score(X_test, y_test)
    
    # Should be reasonably close
    assert abs(r2 - sklearn_r2) < 0.2


def test_random_forest_regressor_feature_importances():
    """Test RF regressor feature importances"""
    X, y = make_regression(n_samples=150, n_features=4, random_state=42)
    
    model = RandomForestRegressor(
        n_estimators=30,
        max_depth=5,
        random_state=42,
        verbose=0
    )
    model.fit(X, y)
    
    # Feature importances should exist and be valid
    assert model.feature_importances_ is not None
    assert len(model.feature_importances_) == 4
    assert np.sum(model.feature_importances_ > 0) > 0
    
    importance_sum = np.sum(model.feature_importances_)
    assert importance_sum == pytest.approx(1.0, abs=1e-6) or importance_sum == 0


def test_random_state_zero_determinism():
    """Test that random_state=0 works (not treated as None)"""
    X, y = make_classification(n_samples=100, n_features=4, random_state=42)
    
    # Train with random_state=0 twice
    model1 = RandomForestClassifier(n_estimators=10, random_state=0, verbose=0)
    model1.fit(X, y)
    pred1 = model1.predict(X[:10])
    
    model2 = RandomForestClassifier(n_estimators=10, random_state=0, verbose=0)
    model2.fit(X, y)
    pred2 = model2.predict(X[:10])
    
    # Predictions should be identical
    np.testing.assert_array_equal(pred1, pred2)
