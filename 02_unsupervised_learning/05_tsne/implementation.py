"""
t-SNE from Scratch
==================

This module implements t-distributed Stochastic Neighbor Embedding (t-SNE) with
only NumPy. It includes:

- ``compute_pairwise_affinities``: per-point Gaussian bandwidth calibration with
  a binary search on ``beta = 1 / (2 * sigma**2)`` so that every conditional
  distribution matches the requested perplexity,
- ``joint_probabilities``: symmetrization of the conditional probabilities,
- ``q_distribution``: the heavy-tailed Student-t similarity in the embedding,
- ``kl_divergence`` and ``kl_gradient``: the objective and its exact gradient,
- :class:`TSNE`: gradient descent with early exaggeration, momentum and gains,
- ``trustworthiness`` and ``knn_preservation``: neighborhood-quality helpers,
- ``generate_customer_segments``: reproducible synthetic outlet/customer data
  with known segments in 10-20 dimensions.

The demo at the bottom embeds a synthetic customer-segment dataset.

Author: ML-Algo Repository
"""

from __future__ import annotations

from typing import List, Optional, Sequence, Tuple, Union

import numpy as np


FEATURE_NAMES = (
    "monthly_spend",
    "visit_frequency",
    "avg_basket_size",
    "days_since_last_visit",
    "product_diversity",
    "promo_sensitivity",
    "weekend_share",
    "online_share",
    "loyalty_score",
    "basket_value_std",
    "fresh_product_share",
    "snack_share",
)

SEGMENT_NAMES = (
    "Budget families",
    "Convenience shoppers",
    "Premium loyalists",
    "Promo hunters",
    "Weekend bulk buyers",
)


# ---------------------------------------------------------------------------
# Distance and probability helpers
# ---------------------------------------------------------------------------
def _as_2d_float_array(X: np.ndarray, name: str = "X") -> np.ndarray:
    """Return ``X`` as a finite 2-D float array."""
    array = np.asarray(X, dtype=float)
    if array.ndim != 2:
        raise ValueError(f"{name} must be a 2-D array")
    if array.shape[0] < 1:
        raise ValueError(f"{name} must contain at least one sample")
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must contain only finite values")
    return array


def pairwise_squared_distances(X: np.ndarray) -> np.ndarray:
    """Return the matrix of squared Euclidean distances between rows of ``X``."""
    X = _as_2d_float_array(X)
    squared_norms = np.sum(X * X, axis=1)
    distances = (
        squared_norms[:, None] + squared_norms[None, :] - 2.0 * (X @ X.T)
    )
    # Floating-point cancellation can produce tiny negative values.
    np.maximum(distances, 0.0, out=distances)
    np.fill_diagonal(distances, 0.0)
    return distances


def _conditional_row(distances: np.ndarray, beta: float, exclude: int) -> np.ndarray:
    """Conditional probabilities for one point at precision ``beta``."""
    # Subtracting the smallest off-diagonal distance keeps the nearest
    # non-self weight at 1 and prevents underflow for large beta. The shift
    # cancels exactly in the normalization.
    mask = np.ones(distances.shape[0], dtype=bool)
    mask[exclude] = False
    if np.any(mask):
        shift = float(np.min(distances[mask]))
    else:  # pragma: no cover - only possible for a one-point dataset
        shift = 0.0
    shifted = distances - shift
    weights = np.exp(-beta * shifted)
    weights[exclude] = 0.0

    total = float(weights.sum())
    if not np.isfinite(total) or total <= 0.0:
        # Degenerate fallback: distribute mass over the non-self points.
        weights = np.ones_like(distances)
        weights[exclude] = 0.0
        total = float(weights.sum())
    return weights / total


def _entropy(p: np.ndarray) -> float:
    """Shannon entropy of a probability vector, ignoring exact zeros."""
    positive = p[p > 0.0]
    return float(-np.sum(positive * np.log(positive)))


def compute_pairwise_affinities(
    X: np.ndarray,
    perplexity: float = 30.0,
    tol: float = 1e-5,
    max_iter: int = 200,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute conditional Gaussian affinities with a per-point binary search.

    For every point ``i`` the method searches for a precision ``beta_i`` (with
    ``sigma_i = 1 / sqrt(beta_i)``) such that the conditional distribution

    ``p_{j|i} = exp(-beta_i * ||x_i - x_j||^2) / sum_{k != i} exp(...)``

    has entropy ``H(P_i) = -sum_j p_{j|i} log p_{j|i}`` equal to
    ``log(perplexity)``. Equivalently, the effective number of neighbors is
    ``exp(H) = perplexity``.

    Parameters
    ----------
    X : array of shape (n_samples, n_features)
        Input data.
    perplexity : float, default=30.0
        Target perplexity. Must satisfy ``0 < perplexity < n_samples``.
    tol : float, default=1e-5
        Entropy tolerance used to stop the binary search.
    max_iter : int, default=200
        Maximum number of bisection iterations per point.

    Returns
    -------
    conditional_P : ndarray of shape (n_samples, n_samples)
        Row-stochastic conditional probabilities with a zero diagonal.
    sigmas : ndarray of shape (n_samples,)
        Gaussian bandwidth ``sigma_i`` found for each point.

    Notes
    -----
    A small ``beta`` gives a wide Gaussian and many effective neighbors (high
    entropy); a large ``beta`` gives a narrow Gaussian and few neighbors (low
    entropy). The binary search brackets ``beta`` in ``[1e-20, 1e20]``, which
    covers the useful range even for strongly varying distance scales.
    """
    X = _as_2d_float_array(X)
    n_samples = X.shape[0]

    perplexity = float(perplexity)
    if not np.isfinite(perplexity) or perplexity <= 1.0:
        raise ValueError("perplexity must be a finite number greater than 1")
    if perplexity >= n_samples:
        raise ValueError("perplexity must be smaller than the number of samples")
    if tol <= 0.0:
        raise ValueError("tol must be positive")
    if not isinstance(max_iter, (int, np.integer)) or max_iter < 1:
        raise ValueError("max_iter must be a positive integer")

    distances = pairwise_squared_distances(X)
    target_entropy = float(np.log(perplexity))
    conditional_P = np.zeros((n_samples, n_samples), dtype=float)
    sigmas = np.ones(n_samples, dtype=float)

    for i in range(n_samples):
        row = distances[i]
        beta_min = 1e-20
        beta_max = 1e20
        beta = 1.0
        p = _conditional_row(row, beta, i)
        entropy = _entropy(p)

        for _ in range(int(max_iter)):
            if abs(entropy - target_entropy) <= tol:
                break

            if entropy > target_entropy:
                # Too many effective neighbors: sharpen the distribution.
                beta_min = beta
                if beta_max >= 1e20:
                    beta = beta * 2.0
                else:
                    beta = 0.5 * (beta + beta_max)
            else:
                # Too few effective neighbors: widen the distribution.
                beta_max = beta
                if beta_min <= 1e-20:
                    beta = beta * 0.5
                else:
                    beta = 0.5 * (beta + beta_min)

            beta = float(np.clip(beta, beta_min, beta_max))
            p = _conditional_row(row, beta, i)
            entropy = _entropy(p)

        # Recompute at the final beta so probabilities and sigma agree.
        p = _conditional_row(row, beta, i)
        conditional_P[i] = p
        sigmas[i] = 1.0 / np.sqrt(beta)

    return conditional_P, sigmas


def joint_probabilities(conditional_P: np.ndarray) -> np.ndarray:
    """
    Symmetrize conditional affinities into a joint distribution.

    ``p_ij = (p_{j|i} + p_{i|j}) / (2 * n)`` makes the joint matrix symmetric,
    and the diagonal is set to zero because a point is not its own neighbor.
    Tiny probabilities are floored at ``1e-12`` before the final normalization
    to keep logarithms well behaved.

    Parameters
    ----------
    conditional_P : array of shape (n_samples, n_samples)
        Row-stochastic conditional probabilities, typically the first output of
        :func:`compute_pairwise_affinities`.

    Returns
    -------
    ndarray of shape (n_samples, n_samples)
        Symmetric joint probabilities that sum to 1.
    """
    conditional_P = np.asarray(conditional_P, dtype=float)
    if conditional_P.ndim != 2 or conditional_P.shape[0] != conditional_P.shape[1]:
        raise ValueError("conditional_P must be a square matrix")
    if not np.all(np.isfinite(conditional_P)):
        raise ValueError("conditional_P must contain only finite values")
    if np.any(conditional_P < 0.0):
        raise ValueError("conditional_P must be non-negative")

    n_samples = conditional_P.shape[0]
    joint = (conditional_P + conditional_P.T) / (2.0 * n_samples)
    np.fill_diagonal(joint, 0.0)
    joint = np.maximum(joint, 1e-12)
    np.fill_diagonal(joint, 0.0)

    total = float(joint.sum())
    if total <= 0.0:
        raise ValueError("conditional_P does not contain any probability mass")
    return joint / total


def q_distribution(Y: np.ndarray) -> np.ndarray:
    """
    Student-t similarities between the rows of the low-dimensional embedding.

    ``q_ij = (1 + ||y_i - y_j||^2)^-1 / sum_{k != l} (1 + ||y_k - y_l||^2)^-1``

    The heavy tail (one degree of freedom) is what allows t-SNE to model
    moderately dissimilar points without forcing them infinitely far apart.

    Parameters
    ----------
    Y : array of shape (n_samples, n_components)
        Low-dimensional embedding.

    Returns
    -------
    ndarray of shape (n_samples, n_samples)
        Symmetric joint probabilities with a zero diagonal that sum to 1.
    """
    Y = _as_2d_float_array(Y, name="Y")
    distances = pairwise_squared_distances(Y)
    weights = 1.0 / (1.0 + distances)
    np.fill_diagonal(weights, 0.0)

    total = float(weights.sum())
    if total <= 0.0 or not np.isfinite(total):
        raise ValueError("Could not compute a valid Student-t distribution")
    return weights / total


def kl_divergence(P: np.ndarray, Q: np.ndarray) -> float:
    """
    Kullback-Leibler divergence ``KL(P || Q) = sum_ij p_ij log(p_ij / q_ij)``.

    Terms with ``p_ij = 0`` contribute nothing. A positive probability in ``P``
    paired with zero probability in ``Q`` makes the divergence infinite.
    """
    P = np.asarray(P, dtype=float)
    Q = np.asarray(Q, dtype=float)
    if P.shape != Q.shape:
        raise ValueError("P and Q must have the same shape")
    if P.ndim != 2 or P.shape[0] != P.shape[1]:
        raise ValueError("P and Q must be square matrices")

    mask = P > 0.0
    if np.any(Q[mask] <= 0.0):
        return float("inf")
    if not np.any(mask):
        return 0.0
    return float(np.sum(P[mask] * np.log(P[mask] / Q[mask])))


def kl_gradient(P: np.ndarray, Q: np.ndarray, Y: np.ndarray) -> np.ndarray:
    """
    Exact gradient of the t-SNE KL divergence with respect to ``Y``.

    The gradient is

    ``dKL/dy_i = 4 * sum_j (p_ij - q_ij) * (y_i - y_j)
                 / (1 + ||y_i - y_j||^2)``.

    Parameters
    ----------
    P : array of shape (n_samples, n_samples)
        Symmetric joint probabilities in the high-dimensional space.
    Q : array of shape (n_samples, n_samples)
        Student-t similarities of the current embedding.
    Y : array of shape (n_samples, n_components)
        Current embedding.

    Returns
    -------
    ndarray of shape (n_samples, n_components)
        Gradient of ``KL(P || Q)`` with respect to ``Y``.
    """
    P = np.asarray(P, dtype=float)
    Q = np.asarray(Q, dtype=float)
    Y = _as_2d_float_array(Y, name="Y")

    if P.shape != Q.shape:
        raise ValueError("P and Q must have the same shape")
    if P.shape[0] != Y.shape[0]:
        raise ValueError("P, Q and Y must describe the same number of samples")

    distances = pairwise_squared_distances(Y)
    weights = (P - Q) / (1.0 + distances)
    np.fill_diagonal(weights, 0.0)

    # 4 * (row_sum_i * y_i - sum_j w_ij * y_j)
    row_sums = np.sum(weights, axis=1)
    return 4.0 * (row_sums[:, None] * Y - weights @ Y)


# ---------------------------------------------------------------------------
# t-SNE estimator
# ---------------------------------------------------------------------------
class TSNE:
    """
    t-SNE with exact O(n^2) affinities and gradient descent.

    Parameters
    ----------
    perplexity : float, default=30.0
        Target effective number of neighbors. Smaller values emphasize local
        structure; larger values emphasize global structure.
    learning_rate : float or "auto", default=200.0
        Gradient-descent step size. ``"auto"`` uses
        ``max(n_samples / early_exaggeration / 4, 50)``.
    n_iter : int, default=1000
        Number of gradient-descent iterations.
    early_exaggeration : float, default=12.0
        Multiplier applied to ``P`` during the first
        ``early_exaggeration_iter`` iterations. It encourages tight clusters
        with clear gaps early in optimization.
    early_exaggeration_iter : int, default=250
        Number of iterations that use early exaggeration.
    momentum : float, default=0.5
        Momentum before early exaggeration ends.
    final_momentum : float, default=0.8
        Momentum after early exaggeration ends.
    min_gain : float, default=0.01
        Lower bound for the per-coordinate gain factors.
    init : {"pca", "random"}, default="pca"
        Initialization strategy for the embedding.
    random_state : int or None, default=None
        Seed used by the random initialization.
    tol : float, default=1e-5
        Entropy tolerance for the perplexity calibration.
    max_iter : int or None, default=None
        Optional alias for ``n_iter`` (matching scikit-learn's parameter name).
        When provided, it overrides ``n_iter``.

    Attributes
    ----------
    embedding_ : ndarray of shape (n_samples, 2)
        The learned two-dimensional embedding.
    P_ : ndarray of shape (n_samples, n_samples)
        Symmetric joint probabilities in the input space.
    Q_ : ndarray of shape (n_samples, n_samples)
        Student-t similarities at the final embedding.
    sigmas_ : ndarray of shape (n_samples,)
        Per-point Gaussian bandwidths from the perplexity calibration.
    kl_history_ : list of float
        Raw (non-exaggerated) KL divergence recorded once per iteration; the
        final entry is the divergence of the returned embedding.
    n_iter_ : int
        Number of iterations actually run.
    """

    def __init__(
        self,
        perplexity: float = 30.0,
        learning_rate: Union[float, str] = 200.0,
        n_iter: int = 1000,
        early_exaggeration: float = 12.0,
        early_exaggeration_iter: int = 250,
        momentum: float = 0.5,
        final_momentum: float = 0.8,
        min_gain: float = 0.01,
        init: str = "pca",
        random_state: Optional[int] = None,
        tol: float = 1e-5,
        max_iter: Optional[int] = None,
    ):
        self.perplexity = float(perplexity)
        self.learning_rate = learning_rate
        if max_iter is not None:
            if not isinstance(max_iter, (int, np.integer)) or max_iter < 1:
                raise ValueError("max_iter must be a positive integer")
            n_iter = int(max_iter)
        self.n_iter = n_iter
        self.early_exaggeration = float(early_exaggeration)
        self.early_exaggeration_iter = early_exaggeration_iter
        self.momentum = float(momentum)
        self.final_momentum = float(final_momentum)
        self.min_gain = float(min_gain)
        self.init = init
        self.random_state = random_state
        self.tol = float(tol)

        self._validate_parameters()

        self.embedding_: Optional[np.ndarray] = None
        self.P_: Optional[np.ndarray] = None
        self.Q_: Optional[np.ndarray] = None
        self.sigmas_: Optional[np.ndarray] = None
        self.kl_history_: List[float] = []
        self.n_iter_: int = 0

    def _validate_parameters(self) -> None:
        if self.perplexity <= 1.0 or not np.isfinite(self.perplexity):
            raise ValueError("perplexity must be greater than 1")
        if self.early_exaggeration < 1.0:
            raise ValueError("early_exaggeration must be at least 1")
        if not isinstance(self.n_iter, (int, np.integer)) or self.n_iter < 1:
            raise ValueError("n_iter must be a positive integer")
        if (
            not isinstance(self.early_exaggeration_iter, (int, np.integer))
            or self.early_exaggeration_iter < 0
        ):
            raise ValueError("early_exaggeration_iter must be a non-negative integer")
        if not 0.0 <= self.momentum < 1.0:
            raise ValueError("momentum must be in [0, 1)")
        if not 0.0 <= self.final_momentum < 1.0:
            raise ValueError("final_momentum must be in [0, 1)")
        if self.min_gain <= 0.0:
            raise ValueError("min_gain must be positive")
        if self.init not in {"pca", "random"}:
            raise ValueError("init must be 'pca' or 'random'")
        if isinstance(self.learning_rate, str):
            if self.learning_rate != "auto":
                raise ValueError("learning_rate must be a positive number or 'auto'")
        elif not np.isfinite(self.learning_rate) or self.learning_rate <= 0.0:
            raise ValueError("learning_rate must be positive")

    def _initial_embedding(self, X: np.ndarray) -> np.ndarray:
        n_samples, n_features = X.shape
        if self.init == "pca" and n_samples >= 2 and n_features >= 1:
            centered = X - X.mean(axis=0, keepdims=True)
            try:
                U, singular_values, _ = np.linalg.svd(
                    centered, full_matrices=False
                )
            except np.linalg.LinAlgError:  # pragma: no cover - rare fallback
                U = np.zeros((n_samples, 2))
                singular_values = np.zeros(2)
            n_components = min(2, U.shape[1], len(singular_values))
            Y = np.zeros((n_samples, 2), dtype=float)
            Y[:, :n_components] = U[:, :n_components] * singular_values[:n_components]
        else:
            rng = np.random.default_rng(self.random_state)
            Y = rng.normal(0.0, 1.0, size=(n_samples, 2))

        # Scale down so the initial Student-t distribution is nearly uniform.
        Y = Y - Y.mean(axis=0, keepdims=True)
        scale = float(np.std(Y))
        if not np.isfinite(scale) or scale < 1e-12:
            scale = 1.0
        Y = Y / scale * 1e-4
        return Y

    def _resolve_learning_rate(self, n_samples: int) -> float:
        if self.learning_rate == "auto":
            return max(n_samples / self.early_exaggeration / 4.0, 50.0)
        return float(self.learning_rate)

    def fit(self, X: np.ndarray, y: Optional[np.ndarray] = None) -> "TSNE":
        """Fit the embedding and return ``self``. ``y`` is ignored."""
        self.fit_transform(X, y=y)
        return self

    def fit_transform(
        self, X: np.ndarray, y: Optional[np.ndarray] = None
    ) -> np.ndarray:
        """
        Learn a two-dimensional embedding of ``X``.

        Parameters
        ----------
        X : array of shape (n_samples, n_features)
            Input data. ``n_samples`` must be larger than ``perplexity``.
        y : ignored
            Present for scikit-learn API compatibility.

        Returns
        -------
        ndarray of shape (n_samples, 2)
            The learned embedding.
        """
        X = _as_2d_float_array(X)
        n_samples = X.shape[0]
        if self.perplexity >= n_samples:
            raise ValueError(
                "perplexity must be smaller than the number of samples"
            )

        conditional_P, sigmas = compute_pairwise_affinities(
            X, perplexity=self.perplexity, tol=self.tol
        )
        P = joint_probabilities(conditional_P)

        rng = np.random.default_rng(self.random_state)
        if self.init == "random":
            Y = rng.normal(0.0, 1.0, size=(n_samples, 2)) * 1e-4
        else:
            Y = self._initial_embedding(X)

        learning_rate = self._resolve_learning_rate(n_samples)
        gains = np.ones_like(Y)
        update = np.zeros_like(Y)
        kl_history: List[float] = []
        exaggeration_iterations = int(self.early_exaggeration_iter)
        if exaggeration_iterations >= int(self.n_iter):
            # Very short runs still need a non-exaggerated refinement phase.
            exaggeration_iterations = max(1, int(self.n_iter) // 4)
        exaggeration_iterations = min(exaggeration_iterations, int(self.n_iter))

        for iteration in range(int(self.n_iter)):
            if iteration < exaggeration_iterations:
                P_effective = P * self.early_exaggeration
                momentum = self.momentum
            else:
                P_effective = P
                momentum = self.final_momentum

            Q = q_distribution(Y)
            kl_history.append(kl_divergence(P, Q))

            gradient = kl_gradient(P_effective, Q, Y)

            # Adaptive gains: increase a coordinate's step when the gradient
            # keeps its sign, shrink it when the gradient flips direction.
            same_direction = gradient * update > 0.0
            gains = np.where(
                same_direction, gains * 0.8, gains + 0.2
            )
            np.maximum(gains, self.min_gain, out=gains)

            update = momentum * update - learning_rate * gains * gradient
            Y = Y + update

            # Remove translation drift: only relative positions matter.
            Y = Y - Y.mean(axis=0, keepdims=True)

        Q_final = q_distribution(Y)
        if kl_history:
            # Replace the last pre-update value with the KL of the final map so
            # the history always has exactly ``n_iter`` entries.
            kl_history[-1] = kl_divergence(P, Q_final)

        self.embedding_ = Y
        self.P_ = P
        self.Q_ = Q_final
        self.sigmas_ = sigmas
        self.kl_history_ = kl_history
        self.n_iter_ = int(self.n_iter)
        return Y

    @property
    def kl_history(self) -> List[float]:
        """Alias for :attr:`kl_history_`."""
        return self.kl_history_

    @property
    def kl_divergence_(self) -> List[float]:
        """Alias for :attr:`kl_history_` (the recorded objective values)."""
        return self.kl_history_

    @property
    def embedding(self) -> Optional[np.ndarray]:
        """Alias for :attr:`embedding_`."""
        return self.embedding_


# ---------------------------------------------------------------------------
# Neighborhood-quality helpers
# ---------------------------------------------------------------------------
def _validate_neighborhood_args(
    X: np.ndarray, Y: np.ndarray, n_neighbors: int
) -> Tuple[np.ndarray, np.ndarray, int]:
    X = _as_2d_float_array(X)
    Y = _as_2d_float_array(Y, name="Y")
    if X.shape[0] != Y.shape[0]:
        raise ValueError("X and Y must contain the same number of samples")
    n_samples = X.shape[0]
    if not isinstance(n_neighbors, (int, np.integer)) or n_neighbors < 1:
        raise ValueError("n_neighbors must be a positive integer")
    k = min(int(n_neighbors), n_samples - 1)
    if k < 1:
        raise ValueError("at least two samples are required")
    return X, Y, k


def trustworthiness(
    X: np.ndarray, Y: np.ndarray, n_neighbors: int = 5
) -> float:
    """
    Trustworthiness of a low-dimensional embedding.

    A value near 1 means that points which are close in ``Y`` were also close in
    ``X``. The score penalizes low-dimensional neighbors that were far away in
    the original space:

    ``T = 1 - 2 / (n k (2n - 3k - 1)) * sum_i sum_{j in U_i^k} (r_ij - k)``

    where ``U_i^k`` is the set of ``k`` nearest neighbors in ``Y`` that are not
    among the ``k`` nearest neighbors in ``X`` and ``r_ij`` is the one-based
    rank of ``j`` in the high-dimensional neighbor list of ``i``.
    """
    X, Y, k = _validate_neighborhood_args(X, Y, n_neighbors)
    n_samples = X.shape[0]

    X_distances = pairwise_squared_distances(X)
    Y_distances = pairwise_squared_distances(Y)
    np.fill_diagonal(X_distances, np.inf)
    np.fill_diagonal(Y_distances, np.inf)

    X_order = np.argsort(X_distances, axis=1)
    Y_order = np.argsort(Y_distances, axis=1)

    # Rank of each point in the high-dimensional ordering of every other point.
    X_rank = np.empty_like(X_order)
    rows = np.arange(n_samples)[:, None]
    X_rank[rows, X_order] = np.arange(1, n_samples + 1)[None, :]  # 1-based ranks

    penalty = 0.0
    for i in range(n_samples):
        high_neighbors = set(X_order[i, :k].tolist())
        for j in Y_order[i, :k]:
            j = int(j)
            if j not in high_neighbors:
                penalty += float(X_rank[i, j] - k)

    denominator = n_samples * k * (2.0 * n_samples - 3.0 * k - 1.0)
    if denominator <= 0.0:
        return 1.0
    score = 1.0 - 2.0 * penalty / denominator
    return float(np.clip(score, 0.0, 1.0))


def knn_preservation(
    X: np.ndarray, Y: np.ndarray, n_neighbors: int = 5
) -> float:
    """
    Fraction of high-dimensional k-nearest neighbors preserved in ``Y``.

    The score is the average overlap between the ``k`` nearest neighbors in
    ``X`` and the ``k`` nearest neighbors in ``Y``. It ranges from 0 to 1, with
    1 meaning every neighborhood is perfectly preserved.
    """
    X, Y, k = _validate_neighborhood_args(X, Y, n_neighbors)

    X_distances = pairwise_squared_distances(X)
    Y_distances = pairwise_squared_distances(Y)
    np.fill_diagonal(X_distances, np.inf)
    np.fill_diagonal(Y_distances, np.inf)

    X_neighbors = np.argsort(X_distances, axis=1)[:, :k]
    Y_neighbors = np.argsort(Y_distances, axis=1)[:, :k]

    overlap = 0
    for i in range(X.shape[0]):
        overlap += len(set(X_neighbors[i].tolist()) & set(Y_neighbors[i].tolist()))
    return float(overlap / (X.shape[0] * k))


# ---------------------------------------------------------------------------
# Synthetic outlet / customer segments
# ---------------------------------------------------------------------------
def generate_customer_segments(
    n_samples: int = 300,
    n_features: int = 12,
    n_clusters: int = 5,
    cluster_std: float = 0.7,
    separation: float = 3.0,
    random_state: Optional[int] = 42,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generate synthetic outlet/customer segment features with known clusters.

    Each cluster represents a shopping segment (budget families, convenience
    shoppers, premium loyalists, promo hunters and weekend bulk buyers). The
    feature profiles are sampled around segment-specific centers in a
    ``10-20`` dimensional space, which makes the data suitable for comparing
    t-SNE, PCA, K-Means and other unsupervised methods.

    Parameters
    ----------
    n_samples : int, default=300
        Number of customers/outlets.
    n_features : int, default=12
        Number of behavioural features. Values between 10 and 20 are intended.
    n_clusters : int, default=5
        Number of latent segments.
    cluster_std : float, default=0.7
        Within-cluster standard deviation.
    separation : float, default=3.0
        Scale of the random offset between cluster centers.
    random_state : int or None, default=42
        Seed for reproducibility.

    Returns
    -------
    X : ndarray of shape (n_samples, n_features)
        Feature matrix with non-negative values.
    labels : ndarray of shape (n_samples,)
        Integer segment label for every row.
    """
    if not isinstance(n_samples, (int, np.integer)) or n_samples < 1:
        raise ValueError("n_samples must be a positive integer")
    if not isinstance(n_features, (int, np.integer)) or n_features < 2:
        raise ValueError("n_features must be an integer of at least 2")
    if not isinstance(n_clusters, (int, np.integer)) or n_clusters < 2:
        raise ValueError("n_clusters must be an integer of at least 2")
    if n_clusters > n_samples:
        raise ValueError("n_clusters cannot exceed n_samples")
    if cluster_std <= 0.0 or separation <= 0.0:
        raise ValueError("cluster_std and separation must be positive")

    rng = np.random.default_rng(random_state)

    # Segment profiles live on a positive scale, with random offsets that make
    # the groups clearly distinguishable in the original space.
    base_profiles = rng.uniform(3.0, 8.0, size=(n_clusters, n_features))
    offsets = rng.normal(0.0, separation, size=(n_clusters, n_features))
    centers = np.clip(base_profiles + offsets, 0.5, None)

    # Balanced cluster sizes, then shuffle so labels are not ordered.
    labels = np.arange(n_samples) % n_clusters
    rng.shuffle(labels)

    noise = rng.normal(0.0, cluster_std, size=(n_samples, n_features))
    feature_scales = rng.uniform(0.8, 1.2, size=n_features)
    X = (centers[labels] + noise) * feature_scales
    X = np.clip(X, 0.0, None)
    return X, labels


# Friendly aliases for different notebook/demo naming conventions.
generate_outlet_customer_data = generate_customer_segments
generate_outlet_customer_segments = generate_customer_segments
make_customer_segments = generate_customer_segments
generate_customer_data = generate_customer_segments
generate_segment_data = generate_customer_segments


# ---------------------------------------------------------------------------
# Demo
# ---------------------------------------------------------------------------
def _nearest_centroid_accuracy(Y: np.ndarray, labels: np.ndarray) -> float:
    """Simple unsupervised sanity check: purity of nearest-centroid assignments."""
    unique = np.unique(labels)
    centroids = np.vstack([Y[labels == label].mean(axis=0) for label in unique])
    distances = pairwise_squared_distances(np.vstack([Y, centroids]))
    assignment = np.argmin(distances[: len(Y), len(Y) :], axis=1)
    predicted = unique[assignment]
    return float(np.mean(predicted == labels))


def _demo() -> None:
    print("t-SNE - Customer Segment Demo")
    print("=" * 72)

    X, labels = generate_customer_segments(
        n_samples=240, n_features=12, n_clusters=5, random_state=42
    )
    print(f"Samples        : {X.shape[0]}")
    print(f"Features       : {X.shape[1]}")
    print(f"Known segments : {len(np.unique(labels))}")
    print(f"Feature sample : {np.round(X[0], 2)}")

    model = TSNE(
        perplexity=30.0,
        learning_rate=200.0,
        n_iter=600,
        early_exaggeration=12.0,
        early_exaggeration_iter=200,
        random_state=42,
    )
    embedding = model.fit_transform(X)

    print("\nEmbedding")
    print(f"  shape            : {embedding.shape}")
    print(f"  KL first         : {model.kl_history_[0]:.4f}")
    print(f"  KL final         : {model.kl_history_[-1]:.4f}")
    print(f"  trustworthiness  : {trustworthiness(X, embedding, n_neighbors=10):.3f}")
    print(f"  kNN preservation : {knn_preservation(X, embedding, n_neighbors=10):.3f}")
    print(f"  segment purity   : {_nearest_centroid_accuracy(embedding, labels):.3f}")
    print(f"  sigma range      : {model.sigmas_.min():.3f} - {model.sigmas_.max():.3f}")

    print("\nSegment sizes")
    for index, name in enumerate(SEGMENT_NAMES[: len(np.unique(labels))]):
        count = int(np.sum(labels == index))
        print(f"  {name:22s}: {count}")

    print("\nDone. Compare perplexity values to see local vs global structure!")


if __name__ == "__main__":
    _demo()
