"""
Tests for the from-scratch t-SNE implementation.

The tests verify the three mathematical pillars of t-SNE:

1. the per-point binary search matches the requested perplexity,
2. the joint probabilities are symmetric and normalized,
3. the analytic KL gradient agrees with finite differences,

and then check that optimization reduces the KL divergence and preserves local
neighborhoods on a synthetic dataset with known customer segments.

The suite uses small samples and short runs so that it stays fast.
"""

import importlib.util
import os

import numpy as np
import pytest


# Load implementation.py exactly like tests/test_association_rules.py does.
spec = importlib.util.spec_from_file_location(
    "tsne_impl",
    os.path.join(
        os.path.dirname(__file__),
        "..",
        "02_unsupervised_learning",
        "05_tsne",
        "implementation.py",
    ),
)
ts = importlib.util.module_from_spec(spec)
spec.loader.exec_module(ts)


def row_entropies(P):
    """Entropy of every row of a conditional probability matrix."""
    return np.array(
        [-np.sum(row[row > 0] * np.log(row[row > 0])) for row in P]
    )


# ---------------------------------------------------------------------------
# Perplexity calibration and probability matrices
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("perplexity", [5.0, 10.0, 20.0])
def test_perplexity_calibration_reaches_target_entropy(perplexity):
    rng = np.random.default_rng(42)
    X = rng.normal(size=(60, 6))

    conditional_P, sigmas = ts.compute_pairwise_affinities(
        X, perplexity=perplexity, tol=1e-6
    )

    assert conditional_P.shape == (60, 60)
    assert sigmas.shape == (60,)
    # Every row is a proper distribution over the other points.
    assert np.allclose(conditional_P.sum(axis=1), 1.0, atol=1e-10)
    assert np.allclose(np.diag(conditional_P), 0.0)
    assert np.all(conditional_P >= 0.0)
    # exp(entropy) must equal the requested perplexity.
    entropies = row_entropies(conditional_P)
    assert np.allclose(entropies, np.log(perplexity), atol=1e-4)
    assert np.all(np.isfinite(sigmas)) and np.all(sigmas > 0.0)


def test_perplexity_validation():
    X = np.random.default_rng(0).normal(size=(10, 3))
    with pytest.raises(ValueError, match="perplexity"):
        ts.compute_pairwise_affinities(X, perplexity=1.0)
    with pytest.raises(ValueError, match="perplexity"):
        ts.compute_pairwise_affinities(X, perplexity=10.0)


def test_joint_probabilities_are_symmetric_and_normalized():
    rng = np.random.default_rng(7)
    X = rng.normal(size=(40, 5))

    conditional_P, _ = ts.compute_pairwise_affinities(X, perplexity=8.0)
    joint_P = ts.joint_probabilities(conditional_P)

    assert joint_P.shape == (40, 40)
    assert np.allclose(joint_P, joint_P.T, atol=1e-15)
    assert np.isclose(joint_P.sum(), 1.0, atol=1e-12)
    assert np.allclose(np.diag(joint_P), 0.0)
    off_diagonal = ~np.eye(40, dtype=bool)
    assert np.all(joint_P[off_diagonal] > 0.0)


def test_q_distribution_is_symmetric_and_normalized():
    rng = np.random.default_rng(3)
    Y = rng.normal(size=(25, 2))

    Q = ts.q_distribution(Y)

    assert Q.shape == (25, 25)
    assert np.allclose(Q, Q.T, atol=1e-15)
    assert np.isclose(Q.sum(), 1.0, atol=1e-12)
    assert np.allclose(np.diag(Q), 0.0)
    # The Student-t kernel is heavy-tailed: far pairs still get some mass.
    off_diagonal = ~np.eye(25, dtype=bool)
    assert np.all(Q[off_diagonal] > 0.0)


# ---------------------------------------------------------------------------
# Gradient check
# ---------------------------------------------------------------------------
def test_analytic_gradient_matches_numerical_gradient():
    rng = np.random.default_rng(11)
    n_samples = 10
    X = rng.normal(size=(n_samples, 4))

    conditional_P, _ = ts.compute_pairwise_affinities(X, perplexity=4.0)
    P = ts.joint_probabilities(conditional_P)
    Y = rng.normal(size=(n_samples, 2)) * 0.3

    Q = ts.q_distribution(Y)
    analytic = ts.kl_gradient(P, Q, Y)

    epsilon = 1e-6
    numerical = np.zeros_like(Y)
    for i in range(n_samples):
        for dimension in range(Y.shape[1]):
            Y_plus = Y.copy()
            Y_minus = Y.copy()
            Y_plus[i, dimension] += epsilon
            Y_minus[i, dimension] -= epsilon
            loss_plus = ts.kl_divergence(P, ts.q_distribution(Y_plus))
            loss_minus = ts.kl_divergence(P, ts.q_distribution(Y_minus))
            numerical[i, dimension] = (loss_plus - loss_minus) / (2.0 * epsilon)

    assert np.allclose(analytic, numerical, rtol=1e-6, atol=1e-8)
    assert np.max(np.abs(analytic - numerical)) < 1e-6


def test_kl_divergence_properties():
    rng = np.random.default_rng(5)
    X = rng.normal(size=(15, 3))
    Y = rng.normal(size=(15, 2))

    conditional_P, _ = ts.compute_pairwise_affinities(X, perplexity=5.0)
    P = ts.joint_probabilities(conditional_P)
    Q = ts.q_distribution(Y)

    assert ts.kl_divergence(P, P) == pytest.approx(0.0, abs=1e-12)
    assert ts.kl_divergence(P, Q) > 0.0


# ---------------------------------------------------------------------------
# Optimization and neighborhood preservation
# ---------------------------------------------------------------------------
def test_kl_divergence_decreases_over_iterations():
    X, labels = ts.generate_customer_segments(
        n_samples=120, n_features=10, n_clusters=4, random_state=42
    )
    model = ts.TSNE(
        perplexity=15.0,
        n_iter=300,
        early_exaggeration_iter=100,
        random_state=42,
    )
    embedding = model.fit_transform(X)

    assert embedding.shape == (120, 2)
    assert np.all(np.isfinite(embedding))
    assert len(model.kl_history_) == model.n_iter_
    assert np.all(np.isfinite(model.kl_history_))
    assert model.kl_history_[-1] < model.kl_history_[0]
    # A meaningful optimization should cut the initial divergence substantially.
    assert model.kl_history_[-1] < 0.7 * model.kl_history_[0]
    assert model.P_.shape == (120, 120)
    assert model.sigmas_.shape == (120,)
    assert np.isclose(model.P_.sum(), 1.0, atol=1e-12)


def test_tsne_separates_known_customer_segments():
    X, labels = ts.generate_customer_segments(
        n_samples=150, n_features=12, n_clusters=5, random_state=42
    )
    model = ts.TSNE(
        perplexity=20.0,
        n_iter=350,
        early_exaggeration_iter=120,
        random_state=42,
    )
    embedding = model.fit_transform(X)

    trust = ts.trustworthiness(X, embedding, n_neighbors=10)
    preservation = ts.knn_preservation(X, embedding, n_neighbors=10)

    assert trust > 0.9
    assert preservation > 0.4
    # Points from the same segment should be much closer than other segments.
    same_segment = labels[:, None] == labels[None, :]
    np.fill_diagonal(same_segment, False)
    other_segment = ~same_segment & ~np.eye(len(labels), dtype=bool)
    distances = ts.pairwise_squared_distances(embedding)
    assert distances[same_segment].mean() < distances[other_segment].mean()


def test_trustworthiness_helpers_on_identity_embedding():
    X, _ = ts.generate_customer_segments(
        n_samples=80, n_features=10, n_clusters=4, random_state=3
    )

    assert ts.trustworthiness(X, X, n_neighbors=5) == pytest.approx(1.0)
    assert ts.knn_preservation(X, X, n_neighbors=5) == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# Synthetic generator and validation
# ---------------------------------------------------------------------------
def test_generate_customer_segments_is_reproducible_and_valid():
    X_first, labels_first = ts.generate_customer_segments(
        n_samples=100, n_features=12, n_clusters=4, random_state=7
    )
    X_second, labels_second = ts.generate_customer_segments(
        n_samples=100, n_features=12, n_clusters=4, random_state=7
    )
    X_other, labels_other = ts.generate_customer_segments(
        n_samples=100, n_features=12, n_clusters=4, random_state=8
    )

    assert X_first.shape == (100, 12)
    assert labels_first.shape == (100,)
    assert np.array_equal(X_first, X_second)
    assert np.array_equal(labels_first, labels_second)
    assert not np.array_equal(X_first, X_other)
    assert set(np.unique(labels_first)) == {0, 1, 2, 3}
    assert np.all(np.isfinite(X_first))
    assert np.all(X_first >= 0.0)
    # The generator advertises outlet/customer features in roughly 10-20 dims.
    assert 10 <= X_first.shape[1] <= 20


def test_generate_customer_segments_validation():
    with pytest.raises(ValueError, match="n_samples"):
        ts.generate_customer_segments(n_samples=0)
    with pytest.raises(ValueError, match="n_features"):
        ts.generate_customer_segments(n_features=1)
    with pytest.raises(ValueError, match="n_clusters"):
        ts.generate_customer_segments(n_samples=10, n_clusters=20)


def test_tsne_parameter_validation():
    X = np.random.default_rng(1).normal(size=(30, 4))

    with pytest.raises(ValueError, match="perplexity"):
        ts.TSNE(perplexity=1.0).fit_transform(X)
    with pytest.raises(ValueError, match="perplexity"):
        ts.TSNE(perplexity=30.0).fit_transform(X)
    with pytest.raises(ValueError, match="learning_rate"):
        ts.TSNE(perplexity=5.0, learning_rate="fast")
    with pytest.raises(ValueError, match="init"):
        ts.TSNE(perplexity=5.0, init="spectral")
    with pytest.raises(ValueError, match="n_iter"):
        ts.TSNE(perplexity=5.0, n_iter=0)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])


def test_trustworthiness_matches_sklearn():
    sk_manifold = pytest.importorskip("sklearn.manifold")
    rng = np.random.default_rng(7)
    X = rng.normal(size=(60, 6))
    Y = X[:, :2] + 0.5 * rng.normal(size=(60, 2))
    for k in (3, 8):
        expected = sk_manifold.trustworthiness(X, Y, n_neighbors=k)
        assert ts.trustworthiness(X, Y, n_neighbors=k) == pytest.approx(expected, abs=1e-10)
