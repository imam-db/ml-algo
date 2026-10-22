"""
Support Vector Machines from Scratch

This module implements two binary Support Vector Machine classifiers using
only NumPy:

- ``LinearSVM``: a linear SVM trained with the Pegasos algorithm
  (subgradient descent on hinge loss + L2 regularization).
- ``KernelSVM``: a kernel SVM trained with a simplified Sequential Minimal
  Optimization (SMO) solver. Supported kernels are ``linear``, ``rbf`` and
  ``poly``.

Both estimators accept labels in either 0/1 or -1/1 form and preserve the
original label values in their predictions. Features are standardized
internally by default because SVMs are sensitive to feature scales.

The demo at the bottom uses synthetic outlet data (daily sales, foot traffic
and staff count) to classify outlets as high- or low-performing.

Author: ML-Algo Repository
"""

from __future__ import annotations

from typing import Literal, Optional, Tuple, Union

import numpy as np


OUTLET_FEATURE_NAMES = ("daily_sales", "foot_traffic", "staff_count")


# ---------------------------------------------------------------------------
# Small validation helpers
# ---------------------------------------------------------------------------
def _as_2d_float(X: np.ndarray) -> np.ndarray:
    """Convert X to a finite 2-D float array."""
    X = np.asarray(X, dtype=float)
    if X.ndim == 1:
        X = X.reshape(1, -1)
    if X.ndim != 2:
        raise ValueError("X must be a 2-D array of shape (n_samples, n_features)")
    if not np.all(np.isfinite(X)):
        raise ValueError("X contains NaN or infinite values")
    return X


def _validate_binary_y(y: np.ndarray) -> np.ndarray:
    """Validate a binary target vector and return it as a 1-D array."""
    y = np.asarray(y)
    if y.ndim != 1:
        y = y.ravel()
    if y.shape[0] == 0:
        raise ValueError("y must not be empty")
    return y


# ---------------------------------------------------------------------------
# Linear SVM: Pegasos
# ---------------------------------------------------------------------------
class LinearSVM:
    """
    Linear Support Vector Machine trained with Pegasos.

    The model minimizes the L2-regularized hinge-loss objective::

        J(w, b) = (lambda_ / 2) * ||w||^2
                  + (1 / n) * sum_i max(0, 1 - y_i * (w . x_i + b))

    where ``y_i`` is in {-1, +1}. Pegasos performs stochastic subgradient
    descent. At iteration ``t`` the learning rate is ``eta_t = 1 / (lambda_ * t)``.
    If a sample violates the margin, the update is::

        w <- (1 - eta_t * lambda_) * w + eta_t * y_i * x_i
        b <- b + eta_t * y_i

    Otherwise only the weight decay step is applied.

    Parameters
    ----------
    lambda_ : float, default=1e-3
        L2 regularization strength. Must be positive. Smaller values give a
        wider margin and larger weight norms.
    n_epochs : int, default=50
        Number of passes over the training data.
    random_state : int or None, default=None
        Seed used for the sample shuffling inside Pegasos.
    standardize : bool, default=True
        Standardize features using the training mean and standard deviation.
    verbose : int, default=0
        If greater than zero, print the training loss after each epoch.

    Attributes
    ----------
    classes_ : ndarray of shape (2,)
        Original class labels in sorted order.
    w : ndarray of shape (n_features,)
        Weight vector in the internally standardized feature space.
    b : float
        Bias term.
    coef_ : ndarray of shape (n_features,)
        Alias for ``w`` (scikit-learn style).
    intercept_ : float
        Alias for ``b`` (scikit-learn style).
    mean_ : ndarray of shape (n_features,)
        Per-feature mean used for standardization (zero if disabled).
    scale_ : ndarray of shape (n_features,)
        Per-feature standard deviation (one if disabled).
    train_losses_ : list of float
        Full SVM objective after each epoch.
    n_iter_ : int
        Number of epochs actually run.
    """

    def __init__(
        self,
        lambda_: float = 1e-3,
        n_epochs: int = 50,
        random_state: Optional[int] = None,
        standardize: bool = True,
        verbose: int = 0,
    ):
        if lambda_ <= 0:
            raise ValueError("lambda_ must be positive")
        if n_epochs < 1:
            raise ValueError("n_epochs must be at least 1")

        self.lambda_ = float(lambda_)
        self.n_epochs = int(n_epochs)
        self.random_state = random_state
        self.standardize = bool(standardize)
        self.verbose = int(verbose)

    def _check_is_fitted(self) -> None:
        if not hasattr(self, "classes_"):
            raise ValueError("This LinearSVM instance is not fitted yet. Call fit first.")

    def fit(self, X: np.ndarray, y: np.ndarray) -> "LinearSVM":
        """
        Fit the linear SVM with Pegasos.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training features.
        y : array-like of shape (n_samples,)
            Binary labels (0/1, -1/1, or any two distinct values).

        Returns
        -------
        self : LinearSVM
            The fitted estimator.
        """
        X = _as_2d_float(X)
        y = _validate_binary_y(y)
        if X.shape[0] != y.shape[0]:
            raise ValueError("X and y must contain the same number of samples")

        self.classes_ = np.unique(y)
        if self.classes_.size != 2:
            raise ValueError(
                "LinearSVM only supports binary classification; "
                f"got {self.classes_.size} classes"
            )

        # Encode the second class as +1 and the first class as -1.
        y_pm = np.where(y == self.classes_[1], 1.0, -1.0)

        # Feature scaling is important for margin-based methods.
        if self.standardize:
            self.mean_ = X.mean(axis=0)
            self.scale_ = X.std(axis=0)
            self.scale_[self.scale_ == 0] = 1.0
        else:
            self.mean_ = np.zeros(X.shape[1], dtype=float)
            self.scale_ = np.ones(X.shape[1], dtype=float)

        X_scaled = (X - self.mean_) / self.scale_
        n_samples, n_features = X_scaled.shape

        rng = np.random.default_rng(self.random_state)
        w = np.zeros(n_features, dtype=float)
        b = 0.0
        t = 0
        self.train_losses_ = []

        for epoch in range(self.n_epochs):
            for i in rng.permutation(n_samples):
                t += 1
                eta = 1.0 / (self.lambda_ * t)

                margin = y_pm[i] * (X_scaled[i] @ w + b)
                if margin < 1.0:
                    # Subgradient step on the hinge loss plus decay from L2.
                    w = (1.0 - eta * self.lambda_) * w + eta * y_pm[i] * X_scaled[i]
                    b += eta * y_pm[i]
                else:
                    # Correct side of the margin: only the L2 decay applies.
                    w = (1.0 - eta * self.lambda_) * w

            # Full objective (regularized hinge loss).
            margins = 1.0 - y_pm * (X_scaled @ w + b)
            loss = 0.5 * self.lambda_ * (w @ w) + np.mean(np.maximum(0.0, margins))
            self.train_losses_.append(float(loss))

            if self.verbose > 0:
                print(f"Epoch {epoch + 1}/{self.n_epochs} - objective: {loss:.6f}")

        self.w = w
        self.b = float(b)
        self.coef_ = self.w
        self.intercept_ = self.b
        self.n_iter_ = self.n_epochs
        return self

    def decision_function(self, X: np.ndarray) -> np.ndarray:
        """
        Signed distance to the separating hyperplane.

        Positive values predict ``classes_[1]``; negative values predict
        ``classes_[0]``.
        """
        self._check_is_fitted()
        X = _as_2d_float(X)
        if X.shape[1] != self.w.shape[0]:
            raise ValueError(
                f"X has {X.shape[1]} features, but the model was fitted with "
                f"{self.w.shape[0]} features"
            )
        X_scaled = (X - self.mean_) / self.scale_
        return X_scaled @ self.w + self.b

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict the original class labels."""
        self._check_is_fitted()
        scores = self.decision_function(X)
        return np.where(scores >= 0.0, self.classes_[1], self.classes_[0])

    def score(self, X: np.ndarray, y: np.ndarray) -> float:
        """Return the mean accuracy on the given data."""
        y = np.asarray(y)
        return float(np.mean(self.predict(X) == y))


# ---------------------------------------------------------------------------
# Kernel SVM: simplified SMO
# ---------------------------------------------------------------------------
class KernelSVM:
    """
    Kernel Support Vector Machine trained with simplified SMO.

    The dual problem solved is::

        max_a  sum_i a_i - 0.5 * sum_i sum_j a_i a_j y_i y_j K(x_i, x_j)
        s.t.   0 <= a_i <= C,  sum_i a_i y_i = 0

    A simplified SMO loop visits every alpha, selects a second alpha using
    the maximal-error heuristic (with a fallback search), and updates the pair
    analytically with box constraints. No working-set cache or shrinking is
    used, which keeps the implementation compact and readable.

    Parameters
    ----------
    C : float, default=1.0
        Regularization parameter. Larger C allows fewer margin violations.
    kernel : {'rbf', 'linear', 'poly'}, default='rbf'
        Kernel function to use.
    gamma : {'scale', 'auto'} or float, default='scale'
        Kernel coefficient for RBF and polynomial kernels.
        - 'scale': 1 / (n_features * X.var())
        - 'auto': 1 / n_features
    degree : int, default=3
        Degree of the polynomial kernel.
    coef0 : float, default=1.0
        Independent term in the polynomial kernel.
    tol : float, default=1e-3
        KKT violation tolerance used for convergence checks.
    max_iter : int, default=100
        Maximum number of full passes over the training set.
    random_state : int or None, default=None
        Seed used to shuffle the alpha traversal order.
    standardize : bool, default=True
        Standardize features using the training mean and standard deviation.
    verbose : int, default=0
        If greater than zero, print progress after each pass.

    Attributes
    ----------
    classes_ : ndarray of shape (2,)
        Original class labels in sorted order.
    alpha_ : ndarray of shape (n_samples,)
        Dual coefficients for every training sample.
    dual_coef_ : ndarray of shape (n_support,)
        Signed dual coefficients ``alpha_i * y_i`` for the support vectors.
    support_ : ndarray of shape (n_support,)
        Indices of the support vectors in the training set.
    support_vectors_ : ndarray of shape (n_support, n_features)
        Support vectors in the original (unscaled) feature space.
    n_support_ : int
        Number of support vectors.
    b_ : float
        Bias term (intercept).
    intercept_ : float
        Alias for ``b_``.
    n_iter_ : int
        Number of full SMO passes performed.
    """

    _VALID_KERNELS = ("linear", "rbf", "poly")

    def __init__(
        self,
        C: float = 1.0,
        kernel: Literal["linear", "rbf", "poly"] = "rbf",
        gamma: Union[str, float] = "scale",
        degree: int = 3,
        coef0: float = 1.0,
        tol: float = 1e-3,
        max_iter: int = 100,
        random_state: Optional[int] = None,
        standardize: bool = True,
        verbose: int = 0,
    ):
        if C <= 0:
            raise ValueError("C must be positive")
        if kernel not in self._VALID_KERNELS:
            raise ValueError(
                f"kernel must be one of {self._VALID_KERNELS}, got {kernel!r}"
            )
        if not isinstance(gamma, str) and gamma <= 0:
            raise ValueError("gamma must be positive")
        if isinstance(gamma, str) and gamma not in ("scale", "auto"):
            raise ValueError("gamma must be 'scale', 'auto', or a positive float")
        if degree < 1:
            raise ValueError("degree must be at least 1")
        if tol <= 0:
            raise ValueError("tol must be positive")
        if max_iter < 1:
            raise ValueError("max_iter must be at least 1")

        self.C = float(C)
        self.kernel = kernel
        self.gamma = gamma
        self.degree = int(degree)
        self.coef0 = float(coef0)
        self.tol = float(tol)
        self.max_iter = int(max_iter)
        self.random_state = random_state
        self.standardize = bool(standardize)
        self.verbose = int(verbose)

    def _check_is_fitted(self) -> None:
        if not hasattr(self, "classes_"):
            raise ValueError("This KernelSVM instance is not fitted yet. Call fit first.")

    def _resolve_gamma(self, X: np.ndarray) -> float:
        """Resolve 'scale'/'auto' or validate a numeric gamma."""
        if isinstance(self.gamma, str):
            if self.gamma == "auto":
                return 1.0 / X.shape[1]
            variance = float(np.var(X))
            if variance <= 0.0:
                return 1.0
            return 1.0 / (X.shape[1] * variance)
        return float(self.gamma)

    def _kernel_matrix(self, A: np.ndarray, B: np.ndarray) -> np.ndarray:
        """Compute the kernel matrix between A and B."""
        if self.kernel == "linear":
            return A @ B.T

        if self.kernel == "rbf":
            # Squared Euclidean distances, avoiding an (n, m, d) tensor.
            a_norm = np.sum(A * A, axis=1)[:, None]
            b_norm = np.sum(B * B, axis=1)[None, :]
            sq_dist = a_norm + b_norm - 2.0 * (A @ B.T)
            sq_dist = np.maximum(sq_dist, 0.0)
            return np.exp(-self._gamma * sq_dist)

        # Polynomial kernel.
        return (self._gamma * (A @ B.T) + self.coef0) ** self.degree

    def fit(self, X: np.ndarray, y: np.ndarray) -> "KernelSVM":
        """
        Fit the kernel SVM using simplified SMO.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training features.
        y : array-like of shape (n_samples,)
            Binary labels (0/1, -1/1, or any two distinct values).

        Returns
        -------
        self : KernelSVM
            The fitted estimator.
        """
        X = _as_2d_float(X)
        y = _validate_binary_y(y)
        if X.shape[0] != y.shape[0]:
            raise ValueError("X and y must contain the same number of samples")

        self.classes_ = np.unique(y)
        if self.classes_.size != 2:
            raise ValueError(
                "KernelSVM only supports binary classification; "
                f"got {self.classes_.size} classes"
            )

        y_pm = np.where(y == self.classes_[1], 1.0, -1.0)

        if self.standardize:
            self.mean_ = X.mean(axis=0)
            self.scale_ = X.std(axis=0)
            self.scale_[self.scale_ == 0] = 1.0
        else:
            self.mean_ = np.zeros(X.shape[1], dtype=float)
            self.scale_ = np.ones(X.shape[1], dtype=float)

        X_scaled = (X - self.mean_) / self.scale_
        self._gamma = self._resolve_gamma(X_scaled)

        n_samples = X_scaled.shape[0]
        K = self._kernel_matrix(X_scaled, X_scaled)
        alpha = np.zeros(n_samples, dtype=float)
        b = 0.0
        rng = np.random.default_rng(self.random_state)
        n_iter = 0

        for iteration in range(self.max_iter):
            n_changed = 0

            for i in rng.permutation(n_samples):
                yi = y_pm[i]
                ai = alpha[i]
                # Error of sample i: f(x_i) - y_i.
                Ei = K[i] @ (alpha * y_pm) + b - yi

                # KKT violation test for the box-constrained dual.
                violates = (
                    (yi * Ei < -self.tol and ai < self.C)
                    or (yi * Ei > self.tol and ai > 0.0)
                )
                if not violates:
                    continue

                # Errors for all samples, used for second-alpha selection.
                E = K @ (alpha * y_pm) + b - y_pm
                error_diff = np.abs(Ei - E)
                error_diff[i] = -np.inf
                candidate_order = np.argsort(-error_diff)

                # Try the maximal-error partner first; if its box constraints
                # are degenerate, fall back to the next-best partners.
                for j in candidate_order:
                    j = int(j)
                    yj = y_pm[j]
                    aj = alpha[j]

                    if yi != yj:
                        L = max(0.0, aj - ai)
                        H = min(self.C, self.C + aj - ai)
                    else:
                        L = max(0.0, ai + aj - self.C)
                        H = min(self.C, ai + aj)
                    if L >= H:
                        continue

                    eta = 2.0 * K[i, j] - K[i, i] - K[j, j]
                    if eta >= -1e-12:
                        continue

                    aj_new = aj - yj * (Ei - E[j]) / eta
                    aj_new = min(H, max(L, aj_new))
                    if abs(aj_new - aj) < 1e-10:
                        continue

                    ai_new = ai + yi * yj * (aj - aj_new)
                    # Numerical guard; the pair update normally respects the box.
                    ai_new = min(self.C, max(0.0, ai_new))

                    b1 = (
                        b
                        - Ei
                        - yi * (ai_new - ai) * K[i, i]
                        - yj * (aj_new - aj) * K[i, j]
                    )
                    b2 = (
                        b
                        - E[j]
                        - yi * (ai_new - ai) * K[i, j]
                        - yj * (aj_new - aj) * K[j, j]
                    )

                    alpha[i] = ai_new
                    alpha[j] = aj_new
                    if 0.0 < ai_new < self.C:
                        b = b1
                    elif 0.0 < aj_new < self.C:
                        b = b2
                    else:
                        b = 0.5 * (b1 + b2)

                    n_changed += 1
                    break

            n_iter = iteration + 1
            if self.verbose > 0:
                f = K @ (alpha * y_pm) + b
                acc = np.mean((f >= 0.0) == (y_pm > 0.0))
                print(
                    f"Iteration {n_iter}/{self.max_iter} - "
                    f"pair updates: {n_changed} - train accuracy: {acc:.4f}"
                )
            if n_changed == 0:
                break

        support = np.where(alpha > 1e-8)[0]
        self.alpha_ = alpha
        self.alphas_ = alpha
        self.support_ = support
        self.support_vectors_ = X[support].copy()
        self.dual_coef_ = alpha[support] * y_pm[support]
        self.n_support_ = int(support.size)
        self.b_ = float(b)
        self.intercept_ = self.b_
        self.b = self.b_
        self.n_iter_ = n_iter

        # Store the scaled support vectors used by the decision function.
        self._support_scaled = X_scaled[support].copy()

        # For a linear kernel we can also expose a primal weight vector.
        if self.kernel == "linear":
            self.coef_ = self.dual_coef_ @ self._support_scaled
        else:
            self.coef_ = None

        return self

    def decision_function(self, X: np.ndarray) -> np.ndarray:
        """
        Evaluate the kernel expansion.

        Positive values predict ``classes_[1]``; negative values predict
        ``classes_[0]``.
        """
        self._check_is_fitted()
        X = _as_2d_float(X)
        if X.shape[1] != self.mean_.shape[0]:
            raise ValueError(
                f"X has {X.shape[1]} features, but the model was fitted with "
                f"{self.mean_.shape[0]} features"
            )

        X_scaled = (X - self.mean_) / self.scale_
        if self.n_support_ == 0:
            return np.full(X_scaled.shape[0], self.b_, dtype=float)

        K = self._kernel_matrix(X_scaled, self._support_scaled)
        return K @ self.dual_coef_ + self.b_

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict the original class labels."""
        self._check_is_fitted()
        scores = self.decision_function(X)
        return np.where(scores >= 0.0, self.classes_[1], self.classes_[0])

    def score(self, X: np.ndarray, y: np.ndarray) -> float:
        """Return the mean accuracy on the given data."""
        y = np.asarray(y)
        return float(np.mean(self.predict(X) == y))


# ---------------------------------------------------------------------------
# Synthetic outlet data
# ---------------------------------------------------------------------------
def generate_outlet_data(
    n_samples: int = 300,
    random_state: int = 42,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generate synthetic outlet-performance data for binary classification.

    Features
    --------
    daily_sales : float
        Total revenue for the day.
    foot_traffic : float
        Number of visitors during the day.
    staff_count : float
        Number of staff members working that day.

    Target
    ------
    y : ndarray of shape (n_samples,)
        ``1`` for a high-performing outlet and ``0`` for a low-performing
        outlet. Performance is based on sales per staff member relative to a
        traffic-dependent benchmark. Because the benchmark is quadratic in
        foot traffic, the class boundary is non-linear in the raw features.

    Parameters
    ----------
    n_samples : int, default=300
        Number of outlets to generate.
    random_state : int, default=42
        Seed for the random number generator.

    Returns
    -------
    X : ndarray of shape (n_samples, 3)
        Columns are ``daily_sales``, ``foot_traffic`` and ``staff_count``.
    y : ndarray of shape (n_samples,)
        Binary performance labels.
    """
    if n_samples < 1:
        raise ValueError("n_samples must be at least 1")

    rng = np.random.default_rng(random_state)

    # Foot traffic is the main driver of both revenue and the performance bar.
    foot_traffic = np.clip(rng.normal(900.0, 250.0, n_samples), 250.0, 1800.0)
    staff_count = rng.integers(6, 36, n_samples).astype(float)

    # Sales per staff member: a traffic effect with a mild non-linear shape.
    sales_per_staff = (
        55.0
        + 0.035 * foot_traffic
        + 10.0 * np.sin(foot_traffic / 350.0)
        + rng.normal(0.0, 7.0, n_samples)
    )
    daily_sales = sales_per_staff * staff_count

    # High traffic raises expectations: the benchmark grows quadratically.
    benchmark = 88.0 + 0.00004 * (foot_traffic - 900.0) ** 2
    score = sales_per_staff - benchmark
    y = (score > 0.0).astype(int)

    X = np.column_stack([daily_sales, foot_traffic, staff_count])
    return X, y


# ---------------------------------------------------------------------------
# Demo
# ---------------------------------------------------------------------------
def _train_test_split(
    X: np.ndarray,
    y: np.ndarray,
    test_size: float = 0.25,
    random_state: int = 42,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Small dependency-free shuffled train/test split."""
    rng = np.random.default_rng(random_state)
    indices = rng.permutation(len(X))
    n_test = max(1, int(len(X) * test_size))
    test_idx, train_idx = indices[:n_test], indices[n_test:]
    return X[train_idx], X[test_idx], y[train_idx], y[test_idx]


def demo():
    """Train both SVMs on synthetic outlet data and print the results."""
    print("=" * 66)
    print("Support Vector Machine Demo: Outlet Performance")
    print("=" * 66)

    X, y = generate_outlet_data(n_samples=320, random_state=42)
    X_train, X_test, y_train, y_test = _train_test_split(
        X, y, test_size=0.25, random_state=42
    )

    print(f"\nDataset: {len(X_train)} training / {len(X_test)} test outlets")
    print(f"Features: {', '.join(OUTLET_FEATURE_NAMES)}")
    print(f"High-performing share (train): {np.mean(y_train):.1%}")
    print(
        "Feature ranges (train): "
        + ", ".join(
            f"{name}=[{X_train[:, i].min():.0f}, {X_train[:, i].max():.0f}]"
            for i, name in enumerate(OUTLET_FEATURE_NAMES)
        )
    )

    # Linear SVM -----------------------------------------------------------
    print("\n" + "-" * 66)
    print("Linear SVM (Pegasos: hinge loss + L2)")
    print("-" * 66)
    linear = LinearSVM(lambda_=1e-3, n_epochs=60, random_state=42)
    linear.fit(X_train, y_train)

    linear_train_acc = linear.score(X_train, y_train)
    linear_test_acc = linear.score(X_test, y_test)
    print(f"Training accuracy: {linear_train_acc:.4f}")
    print(f"Test accuracy:     {linear_test_acc:.4f}")
    print(f"Weight norm ||w||: {np.linalg.norm(linear.w):.4f}")
    print(f"Bias b:            {linear.b:.4f}")

    # Kernel SVM -----------------------------------------------------------
    print("\n" + "-" * 66)
    print("Kernel SVM (simplified SMO, RBF kernel)")
    print("-" * 66)
    kernel_svm = KernelSVM(
        C=10.0,
        kernel="rbf",
        gamma="scale",
        max_iter=100,
        random_state=42,
    )
    kernel_svm.fit(X_train, y_train)

    kernel_train_acc = kernel_svm.score(X_train, y_train)
    kernel_test_acc = kernel_svm.score(X_test, y_test)
    print(f"Training accuracy: {kernel_train_acc:.4f}")
    print(f"Test accuracy:     {kernel_test_acc:.4f}")
    print(f"Support vectors:   {kernel_svm.n_support_} / {len(X_train)}")
    print(f"Bias b:            {kernel_svm.b_:.4f}")

    # A few sample predictions --------------------------------------------
    print("\n" + "-" * 66)
    print("Sample test predictions")
    print("-" * 66)
    print(
        f"{'daily_sales':>12} {'foot_traffic':>13} {'staff':>7} "
        f"{'actual':>7} {'linear':>7} {'rbf':>6} {'rbf_score':>10}"
    )
    print("-" * 70)
    for i in range(min(8, len(X_test))):
        scores = kernel_svm.decision_function(X_test[i : i + 1])[0]
        print(
            f"{X_test[i, 0]:>12.0f} {X_test[i, 1]:>13.0f} {X_test[i, 2]:>7.0f} "
            f"{y_test[i]:>7d} {linear.predict(X_test[i : i + 1])[0]:>7d} "
            f"{kernel_svm.predict(X_test[i : i + 1])[0]:>6d} {scores:>10.3f}"
        )

    print("\nInterpretation: a positive decision score means the outlet is")
    print("classified as high-performing. The RBF SVM can bend its boundary")
    print("around the non-linear traffic-dependent performance benchmark.")
    print("=" * 66)


if __name__ == "__main__":
    demo()
