"""
Tests for unsupervised learning implementations.

Compares from-scratch implementations against scikit-learn:
- K-Means: inertia, label agreement (adjusted rand score), determinism, silhouette
- PCA: explained variance, components (up to sign), reconstruction
"""

import numpy as np
import pytest

# Check if sklearn is available
try:
    from sklearn.cluster import KMeans as SKLearnKMeans
    from sklearn.decomposition import PCA as SKLearnPCA
    from sklearn.metrics import adjusted_rand_score, silhouette_score
    from sklearn.datasets import make_blobs
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False

# Import our implementations
import sys
from pathlib import Path

# Add module directories to path
workspace = Path(__file__).parent.parent
sys.path.insert(0, str(workspace / "02_unsupervised_learning" / "01_kmeans"))
sys.path.insert(0, str(workspace / "02_unsupervised_learning" / "04_pca"))

from implementation import KMeansFromScratch, silhouette_score as our_silhouette_score
from implementation import PCAFromScratch


# ============================================================================
# K-Means Tests
# ============================================================================

@pytest.mark.skipif(not SKLEARN_AVAILABLE, reason="scikit-learn not available")
def test_kmeans_inertia_close_to_sklearn():
    """Test that K-Means inertia is close to sklearn on blob data."""
    # Generate simple blob data
    X, y_true = make_blobs(n_samples=300, centers=4, n_features=2, 
                          random_state=42, cluster_std=1.0)
    
    # Fit both implementations
    our_kmeans = KMeansFromScratch(n_clusters=4, random_state=42, n_init=10, 
                                   init='k-means++', max_iter=300)
    our_kmeans.fit(X)
    
    sklearn_kmeans = SKLearnKMeans(n_clusters=4, random_state=42, n_init=10,
                                   init='k-means++', max_iter=300)
    sklearn_kmeans.fit(X)
    
    # Compare inertia (should be very close)
    inertia_diff = abs(our_kmeans.inertia_ - sklearn_kmeans.inertia_)
    inertia_rel_diff = inertia_diff / sklearn_kmeans.inertia_
    
    print(f"\nK-Means Inertia Comparison:")
    print(f"  Our implementation: {our_kmeans.inertia_:.2f}")
    print(f"  sklearn: {sklearn_kmeans.inertia_:.2f}")
    print(f"  Relative difference: {inertia_rel_diff:.4f}")
    
    # Allow up to 5% relative difference (due to random initialization differences)
    assert inertia_rel_diff < 0.05, f"Inertia differs by {inertia_rel_diff:.2%}"


@pytest.mark.skipif(not SKLEARN_AVAILABLE, reason="scikit-learn not available")
def test_kmeans_labels_match_sklearn():
    """Test that K-Means labels match sklearn up to permutation."""
    # Generate simple blob data
    X, y_true = make_blobs(n_samples=300, centers=4, n_features=2,
                          random_state=42, cluster_std=1.0)
    
    # Fit both implementations
    our_kmeans = KMeansFromScratch(n_clusters=4, random_state=42, n_init=10)
    our_labels = our_kmeans.fit_predict(X)
    
    sklearn_kmeans = SKLearnKMeans(n_clusters=4, random_state=42, n_init=10)
    sklearn_labels = sklearn_kmeans.fit_predict(X)
    
    # Compare labels using adjusted rand score (accounts for label permutation)
    ari = adjusted_rand_score(our_labels, sklearn_labels)
    
    print(f"\nK-Means Label Agreement:")
    print(f"  Adjusted Rand Index: {ari:.4f}")
    
    # Should be very high agreement (> 0.95)
    assert ari > 0.95, f"Label agreement too low: {ari:.4f}"


@pytest.mark.skipif(not SKLEARN_AVAILABLE, reason="scikit-learn not available")
def test_kmeans_determinism():
    """Test that K-Means is deterministic with random_state=0."""
    X, _ = make_blobs(n_samples=200, centers=3, n_features=2, random_state=42)
    
    # Fit twice with same random state
    kmeans1 = KMeansFromScratch(n_clusters=3, random_state=0, n_init=5)
    kmeans1.fit(X)
    
    kmeans2 = KMeansFromScratch(n_clusters=3, random_state=0, n_init=5)
    kmeans2.fit(X)
    
    # Should produce identical results
    assert np.allclose(kmeans1.cluster_centers_, kmeans2.cluster_centers_), \
        "K-Means not deterministic with same random_state"
    assert np.array_equal(kmeans1.labels_, kmeans2.labels_), \
        "K-Means labels not deterministic with same random_state"
    assert kmeans1.inertia_ == kmeans2.inertia_, \
        "K-Means inertia not deterministic with same random_state"
    
    print(f"\n✓ K-Means is deterministic with random_state=0")


@pytest.mark.skipif(not SKLEARN_AVAILABLE, reason="scikit-learn not available")
def test_silhouette_score_close_to_sklearn():
    """Test that our silhouette score matches sklearn."""
    # Generate blob data
    X, _ = make_blobs(n_samples=200, centers=3, n_features=2, 
                     random_state=42, cluster_std=1.5)
    
    # Cluster
    kmeans = KMeansFromScratch(n_clusters=3, random_state=42)
    labels = kmeans.fit_predict(X)
    
    # Compare silhouette scores
    our_score = our_silhouette_score(X, labels)
    sklearn_score = silhouette_score(X, labels)
    
    print(f"\nSilhouette Score Comparison:")
    print(f"  Our implementation: {our_score:.4f}")
    print(f"  sklearn: {sklearn_score:.4f}")
    print(f"  Difference: {abs(our_score - sklearn_score):.4f}")
    
    # Should be very close (within 0.01)
    assert abs(our_score - sklearn_score) < 0.01, \
        f"Silhouette scores differ: ours={our_score:.4f}, sklearn={sklearn_score:.4f}"


# ============================================================================
# PCA Tests
# ============================================================================

@pytest.mark.skipif(not SKLEARN_AVAILABLE, reason="scikit-learn not available")
def test_pca_explained_variance_ratio():
    """Test that PCA explained variance ratio matches sklearn."""
    # Generate correlated data
    rng = np.random.default_rng(42)
    n_samples = 200
    x1 = rng.normal(0, 1, n_samples)
    x2 = 2 * x1 + rng.normal(0, 0.5, n_samples)
    x3 = rng.normal(0, 0.3, n_samples)
    X = np.column_stack([x1, x2, x3])
    
    # Fit both implementations
    our_pca = PCAFromScratch(n_components=3)
    our_pca.fit(X)
    
    sklearn_pca = SKLearnPCA(n_components=3)
    sklearn_pca.fit(X)
    
    # Compare explained variance ratio
    print(f"\nPCA Explained Variance Ratio:")
    print(f"  Our implementation: {our_pca.explained_variance_ratio_}")
    print(f"  sklearn: {sklearn_pca.explained_variance_ratio_}")
    
    assert np.allclose(our_pca.explained_variance_ratio_, 
                      sklearn_pca.explained_variance_ratio_, 
                      rtol=1e-3, atol=1e-4), \
        "Explained variance ratios don't match"


@pytest.mark.skipif(not SKLEARN_AVAILABLE, reason="scikit-learn not available")
def test_pca_components_up_to_sign():
    """Test that PCA components match sklearn up to sign."""
    # Generate data
    rng = np.random.default_rng(42)
    X = rng.normal(0, 1, (100, 5))
    
    # Fit both implementations
    our_pca = PCAFromScratch(n_components=3)
    our_pca.fit(X)
    
    sklearn_pca = SKLearnPCA(n_components=3)
    sklearn_pca.fit(X)
    
    # Components can differ by sign, so check absolute values
    print(f"\nPCA Components Comparison:")
    
    for i in range(3):
        # Check if components match (same or opposite sign)
        same_sign = np.allclose(our_pca.components_[i], 
                               sklearn_pca.components_[i], 
                               rtol=1e-3, atol=1e-4)
        opposite_sign = np.allclose(our_pca.components_[i], 
                                   -sklearn_pca.components_[i], 
                                   rtol=1e-3, atol=1e-4)
        
        assert same_sign or opposite_sign, \
            f"Component {i} doesn't match (even accounting for sign)"
        
        print(f"  PC{i+1}: {'✓ matches' if same_sign or opposite_sign else '✗ differs'}")


@pytest.mark.skipif(not SKLEARN_AVAILABLE, reason="scikit-learn not available")
def test_pca_reconstruction_with_all_components():
    """Test that PCA inverse_transform reconstructs perfectly with all components."""
    # Generate data
    rng = np.random.default_rng(42)
    X = rng.normal(0, 1, (50, 5))
    
    # Fit PCA with all components
    pca = PCAFromScratch(n_components=5)
    X_transformed = pca.fit_transform(X)
    X_reconstructed = pca.inverse_transform(X_transformed)
    
    # Reconstruction should be nearly perfect
    mse = np.mean((X - X_reconstructed) ** 2)
    
    print(f"\nPCA Reconstruction (all components):")
    print(f"  MSE: {mse:.10f}")
    
    assert mse < 1e-10, f"Reconstruction error too high: {mse}"


@pytest.mark.skipif(not SKLEARN_AVAILABLE, reason="scikit-learn not available")
def test_pca_transform_matches_sklearn():
    """Test that PCA transform produces similar results to sklearn."""
    # Generate data
    rng = np.random.default_rng(42)
    X = rng.normal(0, 1, (100, 5))
    
    # Fit both implementations
    our_pca = PCAFromScratch(n_components=2)
    our_transformed = our_pca.fit_transform(X)
    
    sklearn_pca = SKLearnPCA(n_components=2)
    sklearn_transformed = sklearn_pca.fit_transform(X)
    
    # Transformed data should be close (up to sign)
    # Check each component separately
    for i in range(2):
        # Check correlation (accounts for sign flip)
        corr = np.corrcoef(our_transformed[:, i], sklearn_transformed[:, i])[0, 1]
        abs_corr = abs(corr)
        
        print(f"\nPCA Transform Correlation (PC{i+1}): {corr:.4f}")
        
        assert abs_corr > 0.99, \
            f"Transform correlation too low for PC{i+1}: {abs_corr:.4f}"


# ============================================================================
# Edge Cases and Special Tests
# ============================================================================

def test_kmeans_empty_cluster_handling():
    """Test that K-Means handles empty clusters gracefully."""
    # Create data where some clusters might become empty
    rng = np.random.default_rng(42)
    X = rng.normal(0, 1, (50, 2))
    
    # Try to fit with more clusters than natural groups
    kmeans = KMeansFromScratch(n_clusters=10, random_state=42, n_init=3)
    kmeans.fit(X)
    
    # Should complete without errors
    assert kmeans.labels_ is not None
    assert len(np.unique(kmeans.labels_)) <= 10
    
    print(f"\n✓ K-Means handled potential empty clusters")


def test_pca_n_components_none():
    """Test that PCA with n_components=None keeps all components."""
    X = np.random.randn(50, 5)
    
    pca = PCAFromScratch(n_components=None)
    pca.fit(X)
    
    assert pca.n_components_ == 5, \
        f"Expected 5 components, got {pca.n_components_}"
    
    print(f"\n✓ PCA with n_components=None keeps all components")


def test_pca_explained_variance_sums_to_one():
    """Test that explained variance ratios sum to 1.0."""
    X = np.random.randn(100, 5)
    
    pca = PCAFromScratch()
    pca.fit(X)
    
    total_variance = pca.explained_variance_ratio_.sum()
    
    print(f"\nTotal explained variance ratio: {total_variance:.6f}")
    
    assert abs(total_variance - 1.0) < 1e-6, \
        f"Variance ratios don't sum to 1: {total_variance}"


if __name__ == "__main__":
    # Run tests manually
    if SKLEARN_AVAILABLE:
        print("Running K-Means tests...")
        test_kmeans_inertia_close_to_sklearn()
        test_kmeans_labels_match_sklearn()
        test_kmeans_determinism()
        test_silhouette_score_close_to_sklearn()
        
        print("\n" + "="*60)
        print("Running PCA tests...")
        test_pca_explained_variance_ratio()
        test_pca_components_up_to_sign()
        test_pca_reconstruction_with_all_components()
        test_pca_transform_matches_sklearn()
        
        print("\n" + "="*60)
        print("Running edge case tests...")
        test_kmeans_empty_cluster_handling()
        test_pca_n_components_none()
        test_pca_explained_variance_sums_to_one()
        
        print("\n" + "="*60)
        print("✓ All tests passed!")
    else:
        print("scikit-learn not available, skipping tests")
