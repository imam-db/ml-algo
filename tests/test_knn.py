"""
Tests for K-Nearest Neighbors implementation.
"""

import pytest
import numpy as np
import sys
import os

# Import KNN implementation directly
import importlib.util
knn_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '01_supervised_learning', '07_knn', 'implementation.py'))
spec = importlib.util.spec_from_file_location("knn_implementation", knn_path)
knn_impl = importlib.util.module_from_spec(spec)
spec.loader.exec_module(knn_impl)

KNNClassifier = knn_impl.KNNClassifier
KNNRegressor = knn_impl.KNNRegressor
choose_k_elbow_method = knn_impl.choose_k_elbow_method


@pytest.fixture
def classification_data():
    """Generate simple classification data."""
    np.random.seed(42)
    # Class 0: centered at (0, 0)
    X0 = np.random.randn(50, 2) * 0.5
    # Class 1: centered at (3, 3)
    X1 = np.random.randn(50, 2) * 0.5 + 3
    
    X = np.vstack([X0, X1])
    y = np.array([0] * 50 + [1] * 50)
    
    return X, y


@pytest.fixture
def regression_data():
    """Generate simple regression data."""
    np.random.seed(42)
    X = np.random.rand(100, 1) * 10
    y = 2 * X.ravel() + 5 + np.random.randn(100) * 2
    
    return X, y


class TestKNNClassifier:
    """Tests for KNN Classifier."""
    
    def test_classifier_initialization(self):
        """Test KNN classifier initialization."""
        knn = KNNClassifier(n_neighbors=5, weights='uniform', metric='euclidean')
        
        assert knn.n_neighbors == 5
        assert knn.weights == 'uniform'
        assert knn.metric == 'euclidean'
        assert knn.X_train is None
        assert knn.y_train is None
    
    def test_classifier_fit(self, classification_data):
        """Test fitting KNN classifier."""
        X, y = classification_data
        knn = KNNClassifier(n_neighbors=3)
        knn.fit(X, y)
        
        assert knn.X_train is not None
        assert knn.y_train is not None
        assert knn.X_train.shape == X.shape
        assert knn.y_train.shape == y.shape
    
    def test_classifier_predict(self, classification_data):
        """Test KNN classifier predictions."""
        X, y = classification_data
        knn = KNNClassifier(n_neighbors=3)
        knn.fit(X, y)
        
        predictions = knn.predict(X)
        
        assert predictions.shape == y.shape
        assert np.all(np.isin(predictions, [0, 1]))
        
        # Should achieve good accuracy on training data
        accuracy = np.mean(predictions == y)
        assert accuracy > 0.9
    
    def test_classifier_predict_proba(self, classification_data):
        """Test probability predictions."""
        X, y = classification_data
        knn = KNNClassifier(n_neighbors=5)
        knn.fit(X, y)
        
        probas = knn.predict_proba(X)
        
        assert probas.shape == (len(X), 2)
        assert np.all(probas >= 0)
        assert np.all(probas <= 1)
        # Each row should sum to 1
        np.testing.assert_array_almost_equal(probas.sum(axis=1), np.ones(len(X)))
    
    def test_classifier_score(self, classification_data):
        """Test scoring."""
        X, y = classification_data
        knn = KNNClassifier(n_neighbors=5)
        knn.fit(X, y)
        
        score = knn.score(X, y)
        
        assert 0 <= score <= 1
        assert score > 0.9  # Should be high on training data
    
    def test_distance_metrics(self, classification_data):
        """Test different distance metrics."""
        X, y = classification_data
        
        for metric in ['euclidean', 'manhattan', 'minkowski']:
            knn = KNNClassifier(n_neighbors=5, metric=metric)
            knn.fit(X, y)
            predictions = knn.predict(X)
            
            assert predictions.shape == y.shape
            accuracy = np.mean(predictions == y)
            assert accuracy > 0.8
    
    def test_weighting_schemes(self, classification_data):
        """Test uniform vs distance weighting."""
        X, y = classification_data
        
        for weights in ['uniform', 'distance']:
            knn = KNNClassifier(n_neighbors=5, weights=weights)
            knn.fit(X, y)
            predictions = knn.predict(X)
            
            assert predictions.shape == y.shape
            accuracy = np.mean(predictions == y)
            assert accuracy > 0.8
    
    def test_different_k_values(self, classification_data):
        """Test with different k values."""
        X, y = classification_data
        
        for k in [1, 3, 5, 7, 9]:
            knn = KNNClassifier(n_neighbors=k)
            knn.fit(X, y)
            score = knn.score(X, y)
            
            assert 0 <= score <= 1
            assert score > 0.7


class TestKNNRegressor:
    """Tests for KNN Regressor."""
    
    def test_regressor_initialization(self):
        """Test KNN regressor initialization."""
        knn = KNNRegressor(n_neighbors=5, weights='uniform', metric='euclidean')
        
        assert knn.n_neighbors == 5
        assert knn.weights == 'uniform'
        assert knn.metric == 'euclidean'
    
    def test_regressor_fit(self, regression_data):
        """Test fitting KNN regressor."""
        X, y = regression_data
        knn = KNNRegressor(n_neighbors=5)
        knn.fit(X, y)
        
        assert knn.X_train is not None
        assert knn.y_train is not None
    
    def test_regressor_predict(self, regression_data):
        """Test KNN regressor predictions."""
        X, y = regression_data
        knn = KNNRegressor(n_neighbors=5)
        knn.fit(X, y)
        
        predictions = knn.predict(X)
        
        assert predictions.shape == y.shape
        assert np.all(np.isfinite(predictions))
    
    def test_regressor_score(self, regression_data):
        """Test R² scoring."""
        X, y = regression_data
        knn = KNNRegressor(n_neighbors=5)
        knn.fit(X, y)
        
        r2 = knn.score(X, y)
        
        # R² should be high on training data
        assert r2 > 0.7
        assert r2 <= 1.0
    
    def test_regressor_distance_weighting(self, regression_data):
        """Test distance weighting improves predictions."""
        X, y = regression_data
        
        knn_uniform = KNNRegressor(n_neighbors=5, weights='uniform')
        knn_uniform.fit(X, y)
        score_uniform = knn_uniform.score(X, y)
        
        knn_distance = KNNRegressor(n_neighbors=5, weights='distance')
        knn_distance.fit(X, y)
        score_distance = knn_distance.score(X, y)
        
        # Distance weighting should be at least as good
        assert score_distance >= score_uniform - 0.1


class TestKSelection:
    """Tests for k selection methods."""
    
    def test_choose_k_classification(self, classification_data):
        """Test k selection for classification."""
        X, y = classification_data
        
        # Split into train and validation
        split = int(0.7 * len(X))
        X_train, X_val = X[:split], X[split:]
        y_train, y_val = y[:split], y[split:]
        
        k_values, scores = choose_k_elbow_method(
            X_train, y_train, X_val, y_val,
            k_range=range(1, 11),
            task='classification'
        )
        
        assert len(k_values) == len(scores)
        assert len(k_values) == 10
        assert all(0 <= s <= 1 for s in scores)
    
    def test_choose_k_regression(self, regression_data):
        """Test k selection for regression."""
        X, y = regression_data
        
        split = int(0.7 * len(X))
        X_train, X_val = X[:split], X[split:]
        y_train, y_val = y[:split], y[split:]
        
        k_values, scores = choose_k_elbow_method(
            X_train, y_train, X_val, y_val,
            k_range=range(1, 11),
            task='regression'
        )
        
        assert len(k_values) == len(scores)
        assert len(k_values) == 10


class TestEdgeCases:
    """Tests for edge cases."""
    
    def test_single_neighbor(self, classification_data):
        """Test k=1 (nearest neighbor)."""
        X, y = classification_data
        knn = KNNClassifier(n_neighbors=1)
        knn.fit(X, y)
        
        # With k=1 on training data, should get 100% accuracy
        predictions = knn.predict(X)
        accuracy = np.mean(predictions == y)
        assert accuracy == 1.0
    
    def test_binary_classification(self):
        """Test with binary classification."""
        np.random.seed(42)
        X = np.random.randn(50, 2)
        y = (X[:, 0] + X[:, 1] > 0).astype(int)
        
        knn = KNNClassifier(n_neighbors=5)
        knn.fit(X, y)
        
        predictions = knn.predict(X)
        accuracy = np.mean(predictions == y)
        assert accuracy > 0.7
    
    def test_multiclass_classification(self):
        """Test with 3 classes."""
        np.random.seed(42)
        X0 = np.random.randn(30, 2)
        X1 = np.random.randn(30, 2) + 3
        X2 = np.random.randn(30, 2) + [3, -3]
        
        X = np.vstack([X0, X1, X2])
        y = np.array([0]*30 + [1]*30 + [2]*30)
        
        knn = KNNClassifier(n_neighbors=5)
        knn.fit(X, y)
        
        predictions = knn.predict(X)
        accuracy = np.mean(predictions == y)
        assert accuracy > 0.7
        
        # Test probabilities
        probas = knn.predict_proba(X)
        assert probas.shape == (len(X), 3)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
