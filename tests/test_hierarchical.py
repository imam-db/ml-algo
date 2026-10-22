"""
Tests for Hierarchical (Agglomerative) Clustering implementation.
"""

import numpy as np
import pytest
import importlib.util
import sys

# Import hierarchical implementation
spec = importlib.util.spec_from_file_location(
    "hierarchical_implementation",
    "02_unsupervised_learning/03_hierarchical/implementation.py"
)
hierarchical_module = importlib.util.module_from_spec(spec)
sys.modules["hierarchical_implementation"] = hierarchical_module
spec.loader.exec_module(hierarchical_module)

AgglomerativeClustering = hierarchical_module.AgglomerativeClustering
generate_hierarchical_data = hierarchical_module.generate_hierarchical_data


class TestAgglomerativeClustering:
    """Test Agglomerative Clustering functionality."""
    
    def test_initialization(self):
        """Test initialization with different parameters."""
        hc = AgglomerativeClustering(n_clusters=3, linkage='average')
        assert hc.n_clusters == 3
        assert hc.linkage == 'average'
        assert hc.labels_ is None
    
    def test_invalid_linkage(self):
        """Test that invalid linkage raises error."""
        with pytest.raises(ValueError):
            AgglomerativeClustering(n_clusters=2, linkage='invalid')
    
    def test_fit_basic(self):
        """Test basic fitting."""
        X = np.array([
            [0, 0], [0.5, 0],  # Cluster 1
            [5, 5], [5.5, 5],  # Cluster 2
        ])
        
        hc = AgglomerativeClustering(n_clusters=2, linkage='single')
        hc.fit(X)
        
        assert hc.labels_ is not None
        assert len(hc.labels_) == len(X)
        assert hc.n_clusters_ == 2
        assert len(np.unique(hc.labels_)) == 2
    
    def test_fit_predict(self):
        """Test fit_predict method."""
        X = np.array([
            [0, 0], [0.5, 0],
            [5, 5], [5.5, 5],
        ])
        
        hc = AgglomerativeClustering(n_clusters=2, linkage='average')
        labels = hc.fit_predict(X)
        
        assert len(labels) == len(X)
        assert len(np.unique(labels)) == 2
    
    def test_single_linkage(self):
        """Test single linkage method."""
        X = generate_hierarchical_data(n_samples=100, random_state=42)
        
        hc = AgglomerativeClustering(n_clusters=4, linkage='single')
        labels = hc.fit_predict(X)
        
        assert len(labels) == 100
        assert len(np.unique(labels)) == 4
        assert hc.n_clusters_ == 4
    
    def test_complete_linkage(self):
        """Test complete linkage method."""
        X = generate_hierarchical_data(n_samples=100, random_state=42)
        
        hc = AgglomerativeClustering(n_clusters=4, linkage='complete')
        labels = hc.fit_predict(X)
        
        assert len(labels) == 100
        assert len(np.unique(labels)) == 4
        assert hc.n_clusters_ == 4
    
    def test_average_linkage(self):
        """Test average linkage method."""
        X = generate_hierarchical_data(n_samples=100, random_state=42)
        
        hc = AgglomerativeClustering(n_clusters=4, linkage='average')
        labels = hc.fit_predict(X)
        
        assert len(labels) == 100
        assert len(np.unique(labels)) == 4
        assert hc.n_clusters_ == 4
    
    def test_different_n_clusters(self):
        """Test with different numbers of clusters."""
        X = generate_hierarchical_data(n_samples=50, random_state=42)
        
        for n in [2, 3, 5]:
            hc = AgglomerativeClustering(n_clusters=n, linkage='average')
            labels = hc.fit_predict(X)
            
            assert len(np.unique(labels)) == n
            assert hc.n_clusters_ == n
    
    def test_dendrogram_data(self):
        """Test that dendrogram data is recorded."""
        X = generate_hierarchical_data(n_samples=20, random_state=42)
        
        hc = AgglomerativeClustering(n_clusters=3, linkage='average')
        hc.fit(X)
        
        # Should have n_samples - n_clusters merges
        expected_merges = 20 - 3
        assert len(hc.dendrogram_data_) == expected_merges
        
        # Each merge should have required fields
        for merge in hc.dendrogram_data_:
            assert 'cluster1' in merge
            assert 'cluster2' in merge
            assert 'distance' in merge
            assert 'size' in merge
    
    def test_deterministic_behavior(self):
        """Test that results are deterministic."""
        X = generate_hierarchical_data(n_samples=50, random_state=42)
        
        hc1 = AgglomerativeClustering(n_clusters=4, linkage='average')
        labels1 = hc1.fit_predict(X)
        
        hc2 = AgglomerativeClustering(n_clusters=4, linkage='average')
        labels2 = hc2.fit_predict(X)
        
        # Should produce identical results
        np.testing.assert_array_equal(labels1, labels2)
    
    def test_n_clusters_equals_n_samples(self):
        """Test when n_clusters equals n_samples."""
        X = np.array([[0, 0], [1, 1], [2, 2]])
        
        hc = AgglomerativeClustering(n_clusters=3, linkage='average')
        labels = hc.fit_predict(X)
        
        # Each sample should be its own cluster
        assert len(np.unique(labels)) == 3
    
    def test_n_clusters_greater_than_n_samples(self):
        """Test error when n_clusters > n_samples."""
        X = np.array([[0, 0], [1, 1]])
        
        hc = AgglomerativeClustering(n_clusters=5, linkage='average')
        
        with pytest.raises(ValueError):
            hc.fit(X)
    
    def test_linkage_methods_differ(self):
        """Test that different linkage methods can produce different results."""
        X = generate_hierarchical_data(n_samples=50, random_state=42)
        
        hc_single = AgglomerativeClustering(n_clusters=3, linkage='single')
        labels_single = hc_single.fit_predict(X)
        
        hc_complete = AgglomerativeClustering(n_clusters=3, linkage='complete')
        labels_complete = hc_complete.fit_predict(X)
        
        # Labels might differ (not guaranteed, but likely)
        # At least check both completed successfully
        assert len(labels_single) == len(labels_complete) == 50


class TestDataGeneration:
    """Test synthetic data generation."""
    
    def test_generate_hierarchical_data_shape(self):
        """Test generated data has correct shape."""
        X = generate_hierarchical_data(n_samples=150, random_state=42)
        assert X.shape == (150, 2)
    
    def test_generate_hierarchical_data_reproducibility(self):
        """Test data generation is reproducible with seed."""
        X1 = generate_hierarchical_data(n_samples=100, random_state=42)
        X2 = generate_hierarchical_data(n_samples=100, random_state=42)
        np.testing.assert_array_equal(X1, X2)
    
    def test_generate_hierarchical_data_different_seeds(self):
        """Test different seeds produce different data."""
        X1 = generate_hierarchical_data(n_samples=100, random_state=42)
        X2 = generate_hierarchical_data(n_samples=100, random_state=43)
        assert not np.array_equal(X1, X2)


class TestEdgeCases:
    """Test edge cases and error handling."""
    
    def test_two_samples(self):
        """Test with only two samples."""
        X = np.array([[0, 0], [1, 1]])
        
        hc = AgglomerativeClustering(n_clusters=1, linkage='average')
        labels = hc.fit_predict(X)
        
        assert len(labels) == 2
        # Should merge into 1 cluster
        assert len(np.unique(labels)) == 1
    
    def test_identical_points(self):
        """Test with identical points."""
        X = np.array([[0, 0], [0, 0], [0, 0]])
        
        hc = AgglomerativeClustering(n_clusters=1, linkage='average')
        labels = hc.fit_predict(X)
        
        assert len(labels) == 3
        # All should be in same cluster
        assert len(np.unique(labels)) == 1
    
    def test_single_cluster_output(self):
        """Test creating a single cluster."""
        X = generate_hierarchical_data(n_samples=30, random_state=42)
        
        hc = AgglomerativeClustering(n_clusters=1, linkage='average')
        labels = hc.fit_predict(X)
        
        # All samples in one cluster
        assert np.all(labels == labels[0])


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
