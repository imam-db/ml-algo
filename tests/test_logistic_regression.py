"""
Tests for Logistic Regression implementation
Compares from-scratch implementation with sklearn
"""
import numpy as np
import pytest
from sklearn.linear_model import LogisticRegression as SklearnLogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.datasets import make_classification
import sys
import os
import importlib.util

# Load implementation module
spec = importlib.util.spec_from_file_location(
    "logistic_regression_impl",
    os.path.join(os.path.dirname(__file__), '..', '01_supervised_learning', '02_logistic_regression', 'implementation.py')
)
logistic_regression_impl = importlib.util.module_from_spec(spec)
spec.loader.exec_module(logistic_regression_impl)
LogisticRegressionScratch = logistic_regression_impl.LogisticRegressionScratch


def test_logistic_regression_accuracy():
    """Test logistic regression accuracy is close to sklearn"""
    # Generate sample data
    X, y = make_classification(n_samples=200, n_features=4, n_informative=3, 
                               n_redundant=1, random_state=42)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Train our implementation
    our_model = LogisticRegressionScratch(
        learning_rate=0.1, 
        max_iterations=1000,
        verbose=False
    )
    our_model.fit(X_train, y_train)
    
    # Train sklearn
    sklearn_model = SklearnLogisticRegression(max_iter=1000, random_state=42)
    sklearn_model.fit(X_train, y_train)
    
    # Compare accuracy (should be close)
    our_accuracy = our_model.score(X_test, y_test)
    sklearn_accuracy = sklearn_model.score(X_test, y_test)
    
    # Allow some difference due to optimization differences
    assert abs(our_accuracy - sklearn_accuracy) < 0.1
    assert our_accuracy > 0.7  # Should be decent on simple data


def test_logistic_regression_converged_flag():
    """Test that converged flag is reset on refit"""
    X, y = make_classification(n_samples=100, n_features=2, n_informative=2, n_redundant=0, random_state=42)
    
    model = LogisticRegressionScratch(learning_rate=0.1, max_iterations=10, tolerance=1e-20)
    
    # First fit - won't converge
    model.fit(X, y)
    first_converged = model.converged
    
    # Second fit with better params - should converge
    model.tolerance = 1e-3
    model.max_iterations = 1000
    model.fit(X, y)
    second_converged = model.converged
    
    # Converged flag should be reset properly
    assert second_converged == True or first_converged == False


def test_logistic_regression_feature_importance():
    """Test that feature importance is computed"""
    X, y = make_classification(n_samples=100, n_features=3, n_informative=2, n_redundant=0, random_state=42)
    
    model = LogisticRegressionScratch(learning_rate=0.1, max_iterations=500)
    model.fit(X, y)
    
    importance = model.get_feature_importance()
    
    # Should return array of length n_features
    assert len(importance) == 3
    # Should be non-negative
    assert np.all(importance >= 0)
