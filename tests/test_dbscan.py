"""
Tests for DBSCAN clustering implementation.
"""

import numpy as np
import pytest
import importlib.util
import sys

# Import DBSCAN implementation
spec = importlib.util.spec_from_file_location(
    "dbscan_implementation",
    "02_unsupervised_learning/02_dbscan/implementation.py"
)
dbscan_module = importlib.util.module_from_spec(spec)
sys.modules["dbscan_implementation"] = dbscan_module
spec.loader.exec_module(dbscan_module)

DBSCAN = dbscan_module.DBSCAN
generate_clustering_data = dbscan_module.generate_clustering_data


class TestDBSCAN:
    """Test DBSCAN clustering functionality."""
    
    def test_initialization(self):
        """Test DBSCAN initialization."""
        dbscan = DBSCAN(eps=0.5, min_samples=5)
        assert dbscan.eps == 0.5
        assert dbscan.min_samples == 5
        assert dbscan.metric == 'euclidean'
        assert dbscan.labels_ is None
    
    def test_fit_basic(self):
        """Test basic DBSCAN fitting."""
        # Simple 2-cluster data
        X = np.array([
            [0, 0], [0.5, 0], [0, 0.5],  # Cluster 1
            [5, 5], [5.5, 5], [5, 5.5],  # Cluster 2
        ])
        
        dbscan = DBSCAN(eps=1.0, min_samples=2)
        dbscan.fit(X)
        
        assert dbscan.labels_ is not None
        assert len(dbscan.labels_) == len(X)
        assert dbscan.n_clusters_ >= 2
    
    def test_fit_predict(self):
        """Test fit_predict method."""
        X = np.array([
            [0, 0], [0.5, 0], [0, 0.5],
            [5, 5], [5.5, 5], [5, 5.5],
        ])
        
        dbscan = DBSCAN(eps=1.0, min_samples=2)
        labels = dbscan.fit_predict(X)
        
        assert len(labels) == len(X)
        assert dbscan.n_clusters_ >= 1
    
    def test_noise_detection(self):
        """Test that DBSCAN can detect noise points."""
        # Clusters with outlier
        X = np.array([
            [0, 0], [0.5, 0], [0, 0.5],  # Cluster
            [10, 10],  # Outlier
        ])
        
        dbscan = DBSCAN(eps=1.0, min_samples=2)
        labels = dbscan.fit_predict(X)
        
        # Should have at least one noise point (-1)
        assert -1 in labels
        # Outlier should be noise
        assert labels[3] == -1
    
    def test_core_sample_identification(self):
        """Test that core samples are correctly identified."""
        X = np.array([
            [0, 0], [0.5, 0], [0, 0.5], [0.5, 0.5],  # Dense cluster
        ])
        
        dbscan = DBSCAN(eps=1.0, min_samples=2)
        dbscan.fit(X)
        
        assert dbscan.core_sample_indices_ is not None
        assert len(dbscan.core_sample_indices_) > 0
    
    def test_different_eps_values(self):
        """Test behavior with different eps values."""
        X = generate_clustering_data(n_samples=100, random_state=42)
        
        # Small eps: more clusters/noise
        dbscan_small = DBSCAN(eps=0.5, min_samples=5)
        dbscan_small.fit(X)
        
        # Large eps: fewer clusters
        dbscan_large = DBSCAN(eps=3.0, min_samples=5)
        dbscan_large.fit(X)
        
        # Small eps should have more or equal clusters/noise
        assert dbscan_small.n_clusters_ >= dbscan_large.n_clusters_ or \
               np.sum(dbscan_small.labels_ == -1) >= np.sum(dbscan_large.labels_ == -1)
    
    def test_different_min_samples(self):
        """Test behavior with different min_samples values."""
        X = generate_clustering_data(n_samples=100, random_state=42)
        
        # Low min_samples: fewer noise points
        dbscan_low = DBSCAN(eps=1.5, min_samples=2)
        dbscan_low.fit(X)
        
        # High min_samples: more noise points
        dbscan_high = DBSCAN(eps=1.5, min_samples=10)
        dbscan_high.fit(X)
        
        # Higher min_samples should create more noise
        assert np.sum(dbscan_high.labels_ == -1) >= np.sum(dbscan_low.labels_ == -1)
    
    def test_deterministic_behavior(self):
        """Test that DBSCAN gives consistent results."""
        X = generate_clustering_data(n_samples=100, random_state=42)
        
        dbscan1 = DBSCAN(eps=1.5, min_samples=5)
        labels1 = dbscan1.fit_predict(X)
        
        dbscan2 = DBSCAN(eps=1.5, min_samples=5)
        labels2 = dbscan2.fit_predict(X)
        
        # Should produce identical results
        np.testing.assert_array_equal(labels1, labels2)
    
    def test_single_cluster(self):
        """Test with data that forms a single cluster."""
        X = np.random.randn(50, 2) * 0.5
        
        dbscan = DBSCAN(eps=2.0, min_samples=5)
        dbscan.fit(X)
        
        # Should find 1 cluster
        assert dbscan.n_clusters_ == 1
    
    def test_all_noise(self):
        """Test with widely separated points (all noise)."""
        X = np.array([
            [0, 0], [10, 10], [20, 20], [30, 30]
        ])
        
        dbscan = DBSCAN(eps=1.0, min_samples=2)
        labels = dbscan.fit_predict(X)
        
        # All should be noise
        assert np.all(labels == -1)
        assert dbscan.n_clusters_ == 0


class TestDataGeneration:
    """Test synthetic data generation."""
    
    def test_generate_clustering_data_shape(self):
        """Test generated data has correct shape."""
        X = generate_clustering_data(n_samples=300, random_state=42)
        assert X.shape == (300, 2)
    
    def test_generate_clustering_data_reproducibility(self):
        """Test data generation is reproducible with seed."""
        X1 = generate_clustering_data(n_samples=100, random_state=42)
        X2 = generate_clustering_data(n_samples=100, random_state=42)
        np.testing.assert_array_equal(X1, X2)
    
    def test_generate_clustering_data_different_seeds(self):
        """Test different seeds produce different data."""
        X1 = generate_clustering_data(n_samples=100, random_state=42)
        X2 = generate_clustering_data(n_samples=100, random_state=43)
        assert not np.array_equal(X1, X2)


class TestEdgeCases:
    """Test edge cases and error handling."""
    
    def test_empty_data(self):
        """Test with empty array."""
        X = np.array([]).reshape(0, 2)
        dbscan = DBSCAN(eps=1.0, min_samples=2)
        dbscan.fit(X)
        assert len(dbscan.labels_) == 0
    
    def test_single_sample(self):
        """Test with single sample."""
        X = np.array([[0, 0]])
        dbscan = DBSCAN(eps=1.0, min_samples=1)
        dbscan.fit(X)
        assert len(dbscan.labels_) == 1
    
    def test_min_samples_larger_than_data(self):
        """Test when min_samples is larger than dataset."""
        X = np.array([[0, 0], [1, 1], [2, 2]])
        dbscan = DBSCAN(eps=5.0, min_samples=10)
        labels = dbscan.fit_predict(X)
        
        # All should be noise
        assert np.all(labels == -1)
        assert dbscan.n_clusters_ == 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
