"""
Tests for Decision Tree implementation
Compares from-scratch implementation with sklearn
"""
import numpy as np
import pytest
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.model_selection import train_test_split
from sklearn.datasets import make_classification, make_regression
import sys
import os
import importlib.util

# Load implementation module
spec = importlib.util.spec_from_file_location(
    "decision_tree_impl",
    os.path.join(os.path.dirname(__file__), '..', '01_supervised_learning', '03_decision_trees', 'implementation.py')
)
decision_tree_impl = importlib.util.module_from_spec(spec)
spec.loader.exec_module(decision_tree_impl)
DecisionTreeScratch = decision_tree_impl.DecisionTreeScratch


def test_decision_tree_classification_accuracy():
    """Test decision tree classifier accuracy"""
    # Generate sample data
    X, y = make_classification(n_samples=200, n_features=4, n_informative=3, 
                               n_redundant=1, random_state=42)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Train our implementation
    our_model = DecisionTreeScratch(
        task_type='classification',
        criterion='gini',
        max_depth=5,
        random_state=42
    )
    our_model.fit(X_train, y_train)
    
    # Train sklearn
    sklearn_model = DecisionTreeClassifier(
        criterion='gini',
        max_depth=5,
        random_state=42
    )
    sklearn_model.fit(X_train, y_train)
    
    # Compare accuracy
    our_accuracy = our_model.score(X_test, y_test)
    sklearn_accuracy = sklearn_model.score(X_test, y_test)
    
    # Should be reasonably close
    assert abs(our_accuracy - sklearn_accuracy) < 0.15
    assert our_accuracy > 0.6


def test_decision_tree_string_labels():
    """Test that string labels work correctly"""
    # Generate data with string labels
    X, y_numeric = make_classification(n_samples=100, n_features=3, n_informative=2, n_redundant=0, random_state=42)
    y_string = np.array(['cat' if val == 0 else 'dog' for val in y_numeric])
    
    X_train, X_test, y_train, y_test = train_test_split(X, y_string, test_size=0.2, random_state=42)
    
    # Train model
    model = DecisionTreeScratch(
        task_type='classification',
        max_depth=4,
        random_state=42
    )
    model.fit(X_train, y_train)
    
    # Predict
    predictions = model.predict(X_test)
    
    # Check predictions are strings
    assert predictions.dtype.kind in ['U', 'S', 'O']  # Unicode, bytes, or object
    assert set(predictions).issubset({'cat', 'dog'})
    
    # Check accuracy
    accuracy = model.score(X_test, y_test)
    assert accuracy > 0.5


def test_decision_tree_refit_resets_state():
    """Test that tree_depth_ is reset on refit"""
    X, y = make_classification(n_samples=100, n_features=2, n_informative=2, n_redundant=0, random_state=42)
    
    model = DecisionTreeScratch(task_type='classification', max_depth=5, random_state=42)
    
    # First fit
    model.fit(X, y)
    first_depth = model.tree_depth_
    
    # Second fit with different depth
    model.max_depth = 2
    model.fit(X, y)
    second_depth = model.tree_depth_
    
    # Second depth should be smaller
    assert second_depth <= 2
    assert second_depth < first_depth or first_depth <= 2


def test_decision_tree_regression():
    """Test regression tree with sensible default criterion"""
    X, y = make_regression(n_samples=100, n_features=3, noise=10, random_state=42)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Should default to mse for regression
    model = DecisionTreeScratch(
        task_type='regression',
        max_depth=5,
        random_state=42
    )
    model.fit(X_train, y_train)
    
    # Check it uses mse
    assert model.criterion == 'mse'
    
    # Check reasonable R²
    r2 = model.score(X_test, y_test)
    assert r2 > 0.5


def test_decision_tree_feature_importances():
    """Test that feature importances are computed and sum to 1"""
    X, y = make_classification(n_samples=100, n_features=4, random_state=42)
    
    model = DecisionTreeScratch(
        task_type='classification',
        max_depth=4,
        random_state=42
    )
    model.fit(X, y)
    
    # Feature importances should exist
    assert model.feature_importances_ is not None
    assert len(model.feature_importances_) == 4
    
    # Should be non-negative
    assert np.all(model.feature_importances_ >= 0)
    
    # Should sum to 1 (or 0 if no splits)
    importance_sum = np.sum(model.feature_importances_)
    assert importance_sum == pytest.approx(1.0, abs=1e-6) or importance_sum == 0
