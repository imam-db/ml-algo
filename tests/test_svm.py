"""
Tests for Support Vector Machine implementation
Compares from-scratch linear and kernel SVMs with sklearn
"""
import numpy as np
import pytest
from sklearn.datasets import make_circles
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
import sys
import os
import importlib.util

# Load implementation module
spec = importlib.util.spec_from_file_location(
    "svm_impl",
    os.path.join(os.path.dirname(__file__), '..', '01_supervised_learning',
                 '10_svm', 'implementation.py')
)
svm_impl = importlib.util.module_from_spec(spec)
spec.loader.exec_module(svm_impl)
LinearSVM = svm_impl.LinearSVM
KernelSVM = svm_impl.KernelSVM
generate_outlet_data = svm_impl.generate_outlet_data


def _outlet_split(n_samples=320, random_state=42):
    """Generate outlet data and return a stratified train/test split."""
    X, y = generate_outlet_data(n_samples=n_samples, random_state=random_state)
    return train_test_split(
        X, y, test_size=0.25, random_state=random_state, stratify=y
    )


def test_generate_outlet_data_reproducible_and_valid():
    """Test that the synthetic data generator is deterministic and valid"""
    X1, y1 = generate_outlet_data(n_samples=200, random_state=42)
    X2, y2 = generate_outlet_data(n_samples=200, random_state=42)
    X_other, y_other = generate_outlet_data(n_samples=200, random_state=7)

    assert X1.shape == (200, 3), f"Expected X shape (200, 3), got {X1.shape}"
    assert y1.shape == (200,), f"Expected y shape (200,), got {y1.shape}"

    # Same seed must produce exactly the same dataset.
    np.testing.assert_allclose(X1, X2)
    np.testing.assert_array_equal(y1, y2)

    # A different seed should produce a different sample.
    assert not np.array_equal(y1, y_other)

    # Labels and features must be valid.
    assert set(np.unique(y1)).issubset({0, 1})
    assert len(np.unique(y1)) == 2, "Both outlet classes should be present"
    assert np.all(np.isfinite(X1))
    assert np.all(X1[:, 0] > 0), "Daily sales should be positive"
    assert np.all(X1[:, 1] >= 250) and np.all(X1[:, 1] <= 1800)
    assert np.all(X1[:, 2] >= 6) and np.all(X1[:, 2] <= 35)


def test_linear_svm_fit_predict_and_score():
    """Test the full LinearSVM API on 0/1 labels"""
    X_train, X_test, y_train, y_test = _outlet_split()

    model = LinearSVM(lambda_=1e-3, n_epochs=60, random_state=42)
    returned = model.fit(X_train, y_train)

    assert returned is model, "fit should return self"
    assert np.array_equal(model.classes_, np.array([0, 1]))

    predictions = model.predict(X_test)
    scores = model.decision_function(X_test)

    assert predictions.shape == y_test.shape
    assert scores.shape == y_test.shape
    assert set(np.unique(predictions)).issubset({0, 1})

    # The decision function sign must agree with the predicted label.
    np.testing.assert_array_equal(predictions, (scores >= 0).astype(int))

    # Score must equal manual accuracy and the model must beat chance.
    manual_accuracy = np.mean(predictions == y_test)
    assert model.score(X_test, y_test) == pytest.approx(manual_accuracy)
    assert manual_accuracy > 0.75, \
        f"Linear SVM should achieve > 0.75 accuracy, got {manual_accuracy:.3f}"

    # Learned parameters and training history.
    assert model.w.shape == (X_train.shape[1],)
    assert np.isscalar(model.b) or np.ndim(model.b) == 0
    assert len(model.train_losses_) == model.n_epochs
    assert model.n_iter_ == 60


def test_linear_svm_supports_minus_one_one_labels():
    """Test that LinearSVM accepts -1/1 labels and returns the original labels"""
    X_train, X_test, y_train, y_test = _outlet_split()
    y_train_pm = np.where(y_train == 1, 1, -1)
    y_test_pm = np.where(y_test == 1, 1, -1)

    model = LinearSVM(lambda_=1e-3, n_epochs=60, random_state=42)
    model.fit(X_train, y_train_pm)

    predictions = model.predict(X_test)

    assert np.array_equal(model.classes_, np.array([-1, 1]))
    assert set(np.unique(predictions)).issubset({-1, 1})
    assert model.score(X_test, y_test_pm) > 0.75


def test_linear_svm_regularization_controls_weight_norm():
    """Test that stronger L2 regularization shrinks the weight vector"""
    X_train, _, y_train, _ = _outlet_split()

    strong_reg = LinearSVM(lambda_=1e-2, n_epochs=50, random_state=42)
    weak_reg = LinearSVM(lambda_=1e-4, n_epochs=50, random_state=42)
    strong_reg.fit(X_train, y_train)
    weak_reg.fit(X_train, y_train)

    assert np.linalg.norm(strong_reg.w) < np.linalg.norm(weak_reg.w), \
        "Larger lambda_ should produce a smaller weight norm"


def test_linear_svm_rejects_non_binary_labels():
    """Test that LinearSVM rejects multi-class targets"""
    rng = np.random.default_rng(42)
    X = rng.normal(size=(90, 3))
    y = np.arange(90) % 3  # Three classes

    model = LinearSVM(n_epochs=5, random_state=42)
    with pytest.raises(ValueError, match="binary"):
        model.fit(X, y)


def test_kernel_svm_supported_kernels():
    """Test that all three kernels fit and predict reasonably"""
    X_train, X_test, y_train, y_test = _outlet_split()

    for kernel in ["linear", "rbf", "poly"]:
        model = KernelSVM(
            C=10.0,
            kernel=kernel,
            gamma="scale",
            degree=3,
            max_iter=100,
            random_state=42,
        )
        model.fit(X_train, y_train)

        predictions = model.predict(X_test)
        scores = model.decision_function(X_test)
        accuracy = model.score(X_test, y_test)

        assert predictions.shape == y_test.shape
        assert scores.shape == y_test.shape
        assert set(np.unique(predictions)).issubset({0, 1})
        np.testing.assert_array_equal(predictions, (scores >= 0).astype(int))
        assert accuracy > 0.75, \
            f"{kernel} kernel should achieve > 0.75 accuracy, got {accuracy:.3f}"


def test_kernel_svm_rbf_handles_nonlinear_data():
    """Test that the RBF kernel outperforms a linear kernel on circles"""
    X, y = make_circles(n_samples=180, noise=0.08, factor=0.4, random_state=42)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=42, stratify=y
    )

    linear_svm = KernelSVM(
        C=10.0, kernel="linear", max_iter=100, random_state=42
    )
    linear_svm.fit(X_train, y_train)
    linear_accuracy = linear_svm.score(X_test, y_test)

    rbf_svm = KernelSVM(
        C=10.0, kernel="rbf", gamma=1.0, max_iter=100, random_state=42
    )
    rbf_svm.fit(X_train, y_train)
    rbf_accuracy = rbf_svm.score(X_test, y_test)

    assert rbf_accuracy > 0.9, \
        f"RBF SVM should learn circles, got accuracy {rbf_accuracy:.3f}"
    assert rbf_accuracy > linear_accuracy + 0.2, \
        "RBF should clearly outperform a linear boundary on concentric circles"


def test_kernel_svm_support_vectors_and_dual_coefficients():
    """Test alpha bounds, support-vector bookkeeping and dual coefficients"""
    X_train, X_test, y_train, y_test = _outlet_split()
    C = 1.0

    model = KernelSVM(
        C=C, kernel="rbf", gamma="scale", max_iter=100, random_state=42
    )
    model.fit(X_train, y_train)

    assert model.n_support_ > 0
    assert model.n_support_ == len(model.support_)
    assert model.n_support_ == len(model.dual_coef_)
    assert model.support_vectors_.shape == (model.n_support_, X_train.shape[1])

    # Box constraints must hold for every dual variable.
    assert np.all(model.alpha_ >= 0.0)
    assert np.all(model.alpha_ <= C + 1e-9)

    # Non-support vectors must have zero coefficients.
    non_support = np.setdiff1d(np.arange(len(X_train)), model.support_)
    np.testing.assert_allclose(model.alpha_[non_support], 0.0, atol=1e-12)

    # dual_coef_ = alpha_i * y_i for the support vectors.
    y_pm = np.where(y_train == 1, 1.0, -1.0)
    expected_dual = model.alpha_[model.support_] * y_pm[model.support_]
    np.testing.assert_allclose(model.dual_coef_, expected_dual)

    assert model.intercept_ == pytest.approx(model.b_)
    assert model.score(X_test, y_test) > 0.75


def test_kernel_svm_rejects_invalid_kernel():
    """Test that an unsupported kernel name raises a clear error"""
    with pytest.raises(ValueError, match="kernel"):
        KernelSVM(kernel="sigmoid")


def test_kernel_svm_matches_sklearn_accuracy():
    """Test that our SMO solver is close to sklearn's SVC on outlet data"""
    X_train, X_test, y_train, y_test = _outlet_split(n_samples=400)

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    our_svm = KernelSVM(
        C=10.0,
        kernel="rbf",
        gamma="scale",
        max_iter=100,
        random_state=42,
    )
    our_svm.fit(X_train, y_train)
    our_accuracy = our_svm.score(X_test, y_test)

    sklearn_svm = SVC(C=10.0, kernel="rbf", gamma="scale")
    sklearn_svm.fit(X_train_scaled, y_train)
    sklearn_accuracy = sklearn_svm.score(X_test_scaled, y_test)

    assert our_accuracy > 0.75, \
        f"Our RBF SVM accuracy should be > 0.75, got {our_accuracy:.3f}"
    assert sklearn_accuracy > 0.75, \
        f"Sklearn RBF SVM accuracy should be > 0.75, got {sklearn_accuracy:.3f}"
    assert abs(our_accuracy - sklearn_accuracy) < 0.15, \
        f"Accuracy gap too large: ours={our_accuracy:.3f}, " \
        f"sklearn={sklearn_accuracy:.3f}"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
