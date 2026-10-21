"""
Tests for Linear Regression implementation
Compares from-scratch implementation with sklearn
"""
import numpy as np
import pytest
from sklearn.linear_model import LinearRegression as SklearnLinearRegression
from sklearn.model_selection import train_test_split
from sklearn.datasets import make_regression
import sys
import os
import importlib.util

# Load implementation module
spec = importlib.util.spec_from_file_location(
    "linear_regression_impl",
    os.path.join(os.path.dirname(__file__), '..', '01_supervised_learning', '01_linear_regression', 'implementation.py')
)
linear_regression_impl = importlib.util.module_from_spec(spec)
spec.loader.exec_module(linear_regression_impl)
LinearRegressionFromScratch = linear_regression_impl.LinearRegressionFromScratch


def test_linear_regression_normal_equation():
    """Test normal equation matches sklearn"""
    # Generate sample data
    X, y = make_regression(n_samples=100, n_features=3, noise=10, random_state=42)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Train our implementation
    our_model = LinearRegressionFromScratch(method='normal_equation')
    our_model.fit(X_train, y_train)
    
    # Train sklearn
    sklearn_model = SklearnLinearRegression()
    sklearn_model.fit(X_train, y_train)
    
    # Compare coefficients (allow small tolerance)
    np.testing.assert_allclose(our_model.weights, sklearn_model.coef_, rtol=1e-3, atol=1e-3)
    np.testing.assert_allclose(our_model.bias, sklearn_model.intercept_, rtol=1e-3, atol=1e-3)
    
    # Compare R² scores
    our_r2 = our_model.score(X_test, y_test)
    sklearn_r2 = sklearn_model.score(X_test, y_test)
    assert abs(our_r2 - sklearn_r2) < 0.01


def test_linear_regression_gradient_descent():
    """Test gradient descent gives reasonable results"""
    # Generate sample data
    X, y = make_regression(n_samples=100, n_features=2, noise=5, random_state=42)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Scale features for gradient descent
    from sklearn.preprocessing import StandardScaler
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Train our implementation with gradient descent
    our_model = LinearRegressionFromScratch(
        method='gradient_descent', 
        learning_rate=0.1, 
        n_iterations=1000
    )
    our_model.fit(X_train_scaled, y_train)
    
    # Check R² is reasonable
    our_r2 = our_model.score(X_test_scaled, y_test)
    assert our_r2 > 0.9  # Should fit well on simple data


def test_linear_regression_constant_y():
    """Test R² calculation guards against constant y"""
    X = np.random.randn(50, 2)
    y = np.ones(50) * 5.0  # Constant y
    
    model = LinearRegressionFromScratch(method='normal_equation')
    model.fit(X, y)
    
    # Should not crash and should return 0
    r2 = model.score(X, y)
    assert r2 == 0.0


def test_refit_determinism_with_random_state_zero():
    """Test that refitting with random_state=0 gives same results"""
    X, y = make_regression(n_samples=50, n_features=2, noise=1, random_state=42)
    
    # First fit
    model1 = LinearRegressionFromScratch(method='normal_equation')
    model1.fit(X, y)
    pred1 = model1.predict(X)
    
    # Second fit
    model2 = LinearRegressionFromScratch(method='normal_equation')
    model2.fit(X, y)
    pred2 = model2.predict(X)
    
    # Should be identical (deterministic)
    np.testing.assert_array_equal(pred1, pred2)
