"""
Tests for Gradient Boosting implementation
Compares from-scratch implementation with sklearn
"""
import numpy as np
import pytest
from sklearn.ensemble import GradientBoostingRegressor as SklearnGBR
from sklearn.ensemble import GradientBoostingClassifier as SklearnGBC
from sklearn.model_selection import train_test_split
from sklearn.datasets import make_classification, make_regression
import sys
import os
import importlib.util

# Load implementation module
spec = importlib.util.spec_from_file_location(
    "gradient_boosting_impl",
    os.path.join(os.path.dirname(__file__), '..', '01_supervised_learning', 
                 '09_gradient_boosting', 'implementation.py')
)
gradient_boosting_impl = importlib.util.module_from_spec(spec)
spec.loader.exec_module(gradient_boosting_impl)
GradientBoostingRegressor = gradient_boosting_impl.GradientBoostingRegressor
GradientBoostingClassifier = gradient_boosting_impl.GradientBoostingClassifier


@pytest.mark.slow
def test_gradient_boosting_regressor_monotonic_loss():
    """Test that training loss decreases monotonically for regressor"""
    # Generate sample data
    X, y = make_regression(n_samples=200, n_features=5, noise=10, random_state=42)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Train our implementation
    gb = GradientBoostingRegressor(
        n_estimators=50,
        learning_rate=0.1,
        max_depth=3,
        random_state=42,
        verbose=0
    )
    gb.fit(X_train, y_train)
    
    # Check that training loss decreases (or stays same) at each iteration
    train_losses = np.array(gb.train_losses_)
    
    # Training loss should generally decrease
    # Allow for small increases due to subsampling or numerical issues
    loss_diffs = np.diff(train_losses)
    
    # At least 90% of iterations should show improvement or stay same
    non_increasing = np.sum(loss_diffs <= 0)
    improvement_ratio = non_increasing / len(loss_diffs)
    
    assert improvement_ratio >= 0.9, \
        f"Training loss should decrease monotonically, but only {improvement_ratio:.1%} improved"
    
    # First loss should be higher than last loss
    assert train_losses[0] > train_losses[-1], \
        "First training loss should be higher than final loss"


@pytest.mark.slow
def test_gradient_boosting_classifier_monotonic_loss():
    """Test that training loss decreases monotonically for classifier"""
    # Generate sample data
    X, y = make_classification(n_samples=300, n_features=8, n_informative=6,
                               n_redundant=2, random_state=42)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Train our implementation
    gb = GradientBoostingClassifier(
        n_estimators=50,
        learning_rate=0.1,
        max_depth=3,
        random_state=42,
        verbose=0
    )
    gb.fit(X_train, y_train)
    
    # Check that training loss decreases
    train_losses = np.array(gb.train_losses_)
    loss_diffs = np.diff(train_losses)
    
    # At least 90% should improve
    non_increasing = np.sum(loss_diffs <= 0)
    improvement_ratio = non_increasing / len(loss_diffs)
    
    assert improvement_ratio >= 0.9, \
        f"Training loss should decrease monotonically, but only {improvement_ratio:.1%} improved"
    
    assert train_losses[0] > train_losses[-1], \
        "First training loss should be higher than final loss"


@pytest.mark.slow
def test_gradient_boosting_regressor_vs_sklearn():
    """Test that our regressor achieves similar performance to sklearn"""
    # Generate sample data
    X, y = make_regression(n_samples=300, n_features=6, noise=15, random_state=42)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Train our implementation
    our_gb = GradientBoostingRegressor(
        n_estimators=80,
        learning_rate=0.1,
        max_depth=3,
        random_state=42,
        verbose=0
    )
    our_gb.fit(X_train, y_train)
    our_pred = our_gb.predict(X_test)
    our_mse = np.mean((y_test - our_pred) ** 2)
    
    # Train sklearn
    sklearn_gb = SklearnGBR(
        n_estimators=80,
        learning_rate=0.1,
        max_depth=3,
        random_state=42,
        verbose=0
    )
    sklearn_gb.fit(X_train, y_train)
    sklearn_pred = sklearn_gb.predict(X_test)
    sklearn_mse = np.mean((y_test - sklearn_pred) ** 2)
    
    # Our implementation should be reasonably close to sklearn
    # Allow 30% difference since implementation details differ
    relative_diff = abs(our_mse - sklearn_mse) / sklearn_mse
    
    assert relative_diff < 0.3, \
        f"Our MSE ({our_mse:.2f}) differs too much from sklearn ({sklearn_mse:.2f}), " \
        f"relative diff: {relative_diff:.1%}"
    
    # Both should achieve reasonable R²
    our_r2 = 1 - np.sum((y_test - our_pred) ** 2) / np.sum((y_test - np.mean(y_test)) ** 2)
    sklearn_r2 = 1 - np.sum((y_test - sklearn_pred) ** 2) / np.sum((y_test - np.mean(y_test)) ** 2)
    
    assert our_r2 > 0.7, f"Our R² should be > 0.7, got {our_r2:.3f}"
    assert sklearn_r2 > 0.7, f"Sklearn R² should be > 0.7, got {sklearn_r2:.3f}"


@pytest.mark.slow
def test_gradient_boosting_classifier_vs_sklearn():
    """Test that our classifier achieves similar performance to sklearn"""
    # Generate sample data
    X, y = make_classification(n_samples=400, n_features=10, n_informative=8,
                               n_redundant=2, random_state=42)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Train our implementation
    our_gb = GradientBoostingClassifier(
        n_estimators=80,
        learning_rate=0.1,
        max_depth=3,
        random_state=42,
        verbose=0
    )
    our_gb.fit(X_train, y_train)
    our_pred = our_gb.predict(X_test)
    our_acc = np.mean(our_pred == y_test)
    
    # Train sklearn
    sklearn_gb = SklearnGBC(
        n_estimators=80,
        learning_rate=0.1,
        max_depth=3,
        random_state=42,
        verbose=0
    )
    sklearn_gb.fit(X_train, y_train)
    sklearn_pred = sklearn_gb.predict(X_test)
    sklearn_acc = np.mean(sklearn_pred == y_test)
    
    # Accuracies should be close
    acc_diff = abs(our_acc - sklearn_acc)
    
    assert acc_diff < 0.1, \
        f"Our accuracy ({our_acc:.3f}) differs too much from sklearn ({sklearn_acc:.3f})"
    
    # Both should achieve reasonable accuracy
    assert our_acc > 0.7, f"Our accuracy should be > 0.7, got {our_acc:.3f}"
    assert sklearn_acc > 0.7, f"Sklearn accuracy should be > 0.7, got {sklearn_acc:.3f}"


@pytest.mark.slow
def test_early_stopping_regressor():
    """Test that early stopping works correctly for regressor"""
    # Generate data with more noise to trigger early stopping
    X, y = make_regression(n_samples=200, n_features=10, noise=50, random_state=123)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Train with early stopping
    gb_early = GradientBoostingRegressor(
        n_estimators=100,
        learning_rate=0.2,
        max_depth=5,
        early_stopping_rounds=5,
        validation_fraction=0.3,
        random_state=42,
        verbose=0
    )
    gb_early.fit(X_train, y_train)
    
    # Should have validation losses recorded when early stopping is enabled
    assert len(gb_early.val_losses_) > 0, "Validation losses should be recorded"
    
    # Number of trees should match number of stored trees
    assert len(gb_early.trees_) == gb_early.n_estimators_, \
        f"Number of trees ({len(gb_early.trees_)}) should match n_estimators_ ({gb_early.n_estimators_})"
    
    # n_estimators_ should be set and reasonable
    assert gb_early.n_estimators_ > 0, "n_estimators_ should be positive"
    assert gb_early.n_estimators_ <= 100, "n_estimators_ should not exceed max"


@pytest.mark.slow
def test_early_stopping_classifier():
    """Test that early stopping works correctly for classifier"""
    # Generate data with more features to make early stopping more likely
    X, y = make_classification(n_samples=300, n_features=20, n_informative=10,
                               n_redundant=5, random_state=123)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Train with early stopping
    gb_early = GradientBoostingClassifier(
        n_estimators=100,
        learning_rate=0.2,
        max_depth=5,
        early_stopping_rounds=5,
        validation_fraction=0.3,
        random_state=42,
        verbose=0
    )
    gb_early.fit(X_train, y_train)
    
    # Should have validation losses recorded when early stopping is enabled
    assert len(gb_early.val_losses_) > 0, "Validation losses should be recorded"
    
    # Number of trees should match
    assert len(gb_early.trees_) == gb_early.n_estimators_, \
        f"Number of trees ({len(gb_early.trees_)}) should match n_estimators_ ({gb_early.n_estimators_})"
    
    # n_estimators_ should be set and reasonable
    assert gb_early.n_estimators_ > 0, "n_estimators_ should be positive"
    assert gb_early.n_estimators_ <= 100, "n_estimators_ should not exceed max"


@pytest.mark.slow
def test_regressor_improves_with_more_trees():
    """Test that adding more trees improves performance"""
    # Generate data
    X, y = make_regression(n_samples=300, n_features=6, noise=15, random_state=42)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Train with few trees
    gb_few = GradientBoostingRegressor(
        n_estimators=10,
        learning_rate=0.1,
        max_depth=3,
        random_state=42,
        verbose=0
    )
    gb_few.fit(X_train, y_train)
    pred_few = gb_few.predict(X_test)
    mse_few = np.mean((y_test - pred_few) ** 2)
    
    # Train with more trees
    gb_many = GradientBoostingRegressor(
        n_estimators=100,
        learning_rate=0.1,
        max_depth=3,
        random_state=42,
        verbose=0
    )
    gb_many.fit(X_train, y_train)
    pred_many = gb_many.predict(X_test)
    mse_many = np.mean((y_test - pred_many) ** 2)
    
    # More trees should perform better
    assert mse_many < mse_few, \
        f"More trees should improve performance: MSE(10 trees)={mse_few:.2f}, " \
        f"MSE(100 trees)={mse_many:.2f}"


@pytest.mark.slow
def test_subsample_effect():
    """Test that subsample parameter works"""
    # Generate data
    X, y = make_regression(n_samples=300, n_features=6, noise=10, random_state=42)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Train with subsample
    gb_subsample = GradientBoostingRegressor(
        n_estimators=50,
        learning_rate=0.1,
        max_depth=3,
        subsample=0.5,
        random_state=42,
        verbose=0
    )
    gb_subsample.fit(X_train, y_train)
    pred_subsample = gb_subsample.predict(X_test)
    
    # Should still make reasonable predictions
    mse = np.mean((y_test - pred_subsample) ** 2)
    r2 = 1 - np.sum((y_test - pred_subsample) ** 2) / np.sum((y_test - np.mean(y_test)) ** 2)
    
    assert r2 > 0.6, f"Subsampling should still achieve reasonable R² > 0.6, got {r2:.3f}"


def test_classifier_binary_only():
    """Test that classifier only accepts binary classification"""
    # Generate multi-class data
    X = np.random.randn(100, 5)
    y = np.random.randint(0, 3, 100)  # 3 classes
    
    gb = GradientBoostingClassifier(n_estimators=10, random_state=42)
    
    with pytest.raises(ValueError, match="binary classification"):
        gb.fit(X, y)


@pytest.mark.slow
def test_predict_proba_sums_to_one():
    """Test that predicted probabilities sum to 1"""
    # Generate data
    X, y = make_classification(n_samples=200, n_features=5, random_state=42)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Train classifier
    gb = GradientBoostingClassifier(
        n_estimators=50,
        learning_rate=0.1,
        max_depth=3,
        random_state=42,
        verbose=0
    )
    gb.fit(X_train, y_train)
    
    # Get probabilities
    proba = gb.predict_proba(X_test)
    
    # Check shape
    assert proba.shape == (len(X_test), 2), "Probabilities should have shape (n_samples, 2)"
    
    # Check that probabilities sum to 1
    prob_sums = np.sum(proba, axis=1)
    np.testing.assert_allclose(prob_sums, 1.0, rtol=1e-5, 
                                err_msg="Probabilities should sum to 1")
    
    # Check that probabilities are in valid range
    assert np.all(proba >= 0), "Probabilities should be >= 0"
    assert np.all(proba <= 1), "Probabilities should be <= 1"


def test_staged_predict_regressor():
    """Test staged_predict for regressor"""
    # Generate data
    X, y = make_regression(n_samples=100, n_features=4, noise=5, random_state=42)
    
    gb = GradientBoostingRegressor(
        n_estimators=10,
        learning_rate=0.1,
        max_depth=2,
        random_state=42,
        verbose=0
    )
    gb.fit(X, y)
    
    # Get staged predictions
    staged_preds = list(gb.staged_predict(X[:5]))
    
    # Should have n_estimators + 1 stages (including initial prediction)
    assert len(staged_preds) == gb.n_estimators_ + 1, \
        f"Should have {gb.n_estimators_ + 1} stages, got {len(staged_preds)}"
    
    # Each prediction should have correct shape
    for pred in staged_preds:
        assert pred.shape == (5,), f"Prediction shape should be (5,), got {pred.shape}"


@pytest.mark.slow
def test_learning_rate_effect():
    """Test that learning rate affects convergence speed"""
    # Generate data
    X, y = make_regression(n_samples=200, n_features=5, noise=10, random_state=42)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # High learning rate should converge faster (fewer trees needed)
    gb_high_lr = GradientBoostingRegressor(
        n_estimators=20,
        learning_rate=0.5,
        max_depth=3,
        random_state=42,
        verbose=0
    )
    gb_high_lr.fit(X_train, y_train)
    final_loss_high = gb_high_lr.train_losses_[-1]
    
    # Low learning rate needs more trees to reach similar loss
    gb_low_lr = GradientBoostingRegressor(
        n_estimators=20,
        learning_rate=0.01,
        max_depth=3,
        random_state=42,
        verbose=0
    )
    gb_low_lr.fit(X_train, y_train)
    final_loss_low = gb_low_lr.train_losses_[-1]
    
    # High learning rate should achieve lower training loss with same number of trees
    assert final_loss_high < final_loss_low * 0.8, \
        f"High LR should converge faster: loss_high={final_loss_high:.2f}, " \
        f"loss_low={final_loss_low:.2f}"


@pytest.mark.slow
def test_max_depth_effect():
    """Test that max_depth affects model complexity"""
    # Generate data
    X, y = make_regression(n_samples=200, n_features=5, noise=10, random_state=42)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Shallow trees
    gb_shallow = GradientBoostingRegressor(
        n_estimators=50,
        learning_rate=0.1,
        max_depth=1,
        random_state=42,
        verbose=0
    )
    gb_shallow.fit(X_train, y_train)
    train_mse_shallow = gb_shallow.train_losses_[-1]
    
    # Deeper trees
    gb_deep = GradientBoostingRegressor(
        n_estimators=50,
        learning_rate=0.1,
        max_depth=5,
        random_state=42,
        verbose=0
    )
    gb_deep.fit(X_train, y_train)
    train_mse_deep = gb_deep.train_losses_[-1]
    
    # Deeper trees should achieve lower training error
    assert train_mse_deep < train_mse_shallow, \
        f"Deeper trees should fit training data better: " \
        f"shallow={train_mse_shallow:.2f}, deep={train_mse_deep:.2f}"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
