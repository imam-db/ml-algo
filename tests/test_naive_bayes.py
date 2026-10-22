"""
Tests for Naive Bayes implementations.
"""

import pytest
import numpy as np
import sys
import os
import importlib.util

# Import naive bayes implementation directly
nb_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '01_supervised_learning', '06_naive_bayes', 'implementation.py'))
spec = importlib.util.spec_from_file_location("nb_implementation", nb_path)
nb_impl = importlib.util.module_from_spec(spec)
spec.loader.exec_module(nb_impl)

GaussianNaiveBayes = nb_impl.GaussianNaiveBayes
MultinomialNaiveBayes = nb_impl.MultinomialNaiveBayes


@pytest.fixture
def gaussian_data():
    """Generate data for Gaussian Naive Bayes."""
    np.random.seed(42)
    # Class 0: mean=[0, 0], var=1
    X0 = np.random.randn(50, 2)
    # Class 1: mean=[3, 3], var=1
    X1 = np.random.randn(50, 2) + 3
    
    X = np.vstack([X0, X1])
    y = np.array([0] * 50 + [1] * 50)
    
    return X, y


@pytest.fixture
def multinomial_data():
    """Generate count data for Multinomial Naive Bayes."""
    np.random.seed(42)
    n_samples = 90
    n_features = 10
    
    X = []
    y = []
    
    for label in range(3):
        for _ in range(n_samples // 3):
            # Generate count data with class-specific patterns
            counts = np.random.poisson(lam=2, size=n_features)
            if label == 0:
                counts[:3] *= 3  # Class 0 has more of first 3 features
            elif label == 1:
                counts[3:6] *= 3  # Class 1 has more of middle 3
            else:
                counts[6:9] *= 3  # Class 2 has more of last 3
            
            X.append(counts)
            y.append(label)
    
    return np.array(X), np.array(y)


class TestGaussianNaiveBayes:
    """Tests for Gaussian Naive Bayes."""
    
    def test_initialization(self):
        """Test initialization."""
        gnb = GaussianNaiveBayes(var_smoothing=1e-9)
        
        assert gnb.var_smoothing == 1e-9
        assert gnb.classes is None
        assert gnb.means is None
        assert gnb.variances is None
    
    def test_fit(self, gaussian_data):
        """Test fitting."""
        X, y = gaussian_data
        gnb = GaussianNaiveBayes()
        gnb.fit(X, y)
        
        assert gnb.classes is not None
        assert gnb.means is not None
        assert gnb.variances is not None
        assert len(gnb.classes) == 2
        assert gnb.means.shape == (2, 2)
        assert gnb.variances.shape == (2, 2)
    
    def test_predict(self, gaussian_data):
        """Test predictions."""
        X, y = gaussian_data
        gnb = GaussianNaiveBayes()
        gnb.fit(X, y)
        
        predictions = gnb.predict(X)
        
        assert predictions.shape == y.shape
        assert np.all(np.isin(predictions, [0, 1]))
        
        # Should achieve good accuracy
        accuracy = np.mean(predictions == y)
        assert accuracy > 0.8
    
    def test_predict_proba(self, gaussian_data):
        """Test probability predictions."""
        X, y = gaussian_data
        gnb = GaussianNaiveBayes()
        gnb.fit(X, y)
        
        probas = gnb.predict_proba(X)
        
        assert probas.shape == (len(X), 2)
        assert np.all(probas >= 0)
        assert np.all(probas <= 1)
        np.testing.assert_array_almost_equal(probas.sum(axis=1), np.ones(len(X)))
    
    def test_predict_log_proba(self, gaussian_data):
        """Test log probability predictions."""
        X, y = gaussian_data
        gnb = GaussianNaiveBayes()
        gnb.fit(X, y)
        
        log_probas = gnb.predict_log_proba(X)
        
        assert log_probas.shape == (len(X), 2)
        assert np.all(log_probas <= 0)  # Log probabilities should be negative
    
    def test_score(self, gaussian_data):
        """Test accuracy scoring."""
        X, y = gaussian_data
        gnb = GaussianNaiveBayes()
        gnb.fit(X, y)
        
        score = gnb.score(X, y)
        
        assert 0 <= score <= 1
        assert score > 0.8
    
    def test_multiclass(self):
        """Test with 3 classes."""
        np.random.seed(42)
        X0 = np.random.randn(30, 2)
        X1 = np.random.randn(30, 2) + [3, 0]
        X2 = np.random.randn(30, 2) + [0, 3]
        
        X = np.vstack([X0, X1, X2])
        y = np.array([0]*30 + [1]*30 + [2]*30)
        
        gnb = GaussianNaiveBayes()
        gnb.fit(X, y)
        
        predictions = gnb.predict(X)
        accuracy = np.mean(predictions == y)
        assert accuracy > 0.7
        
        probas = gnb.predict_proba(X)
        assert probas.shape == (len(X), 3)


class TestMultinomialNaiveBayes:
    """Tests for Multinomial Naive Bayes."""
    
    def test_initialization(self):
        """Test initialization."""
        mnb = MultinomialNaiveBayes(alpha=1.0)
        
        assert mnb.alpha == 1.0
        assert mnb.classes is None
    
    def test_fit(self, multinomial_data):
        """Test fitting."""
        X, y = multinomial_data
        mnb = MultinomialNaiveBayes()
        mnb.fit(X, y)
        
        assert mnb.classes is not None
        assert mnb.class_priors is not None
        assert mnb.feature_log_probs is not None
        assert len(mnb.classes) == 3
    
    def test_predict(self, multinomial_data):
        """Test predictions."""
        X, y = multinomial_data
        mnb = MultinomialNaiveBayes()
        mnb.fit(X, y)
        
        predictions = mnb.predict(X)
        
        assert predictions.shape == y.shape
        assert np.all(np.isin(predictions, [0, 1, 2]))
        
        # Should achieve reasonable accuracy
        accuracy = np.mean(predictions == y)
        assert accuracy > 0.6
    
    def test_predict_proba(self, multinomial_data):
        """Test probability predictions."""
        X, y = multinomial_data
        mnb = MultinomialNaiveBayes()
        mnb.fit(X, y)
        
        probas = mnb.predict_proba(X)
        
        assert probas.shape == (len(X), 3)
        assert np.all(probas >= 0)
        assert np.all(probas <= 1)
        np.testing.assert_array_almost_equal(probas.sum(axis=1), np.ones(len(X)))
    
    def test_smoothing_effect(self, multinomial_data):
        """Test effect of different alpha values."""
        X, y = multinomial_data
        
        # Try different smoothing parameters
        for alpha in [0.1, 1.0, 10.0]:
            mnb = MultinomialNaiveBayes(alpha=alpha)
            mnb.fit(X, y)
            
            predictions = mnb.predict(X)
            accuracy = np.mean(predictions == y)
            
            # Should still make reasonable predictions
            assert 0 <= accuracy <= 1
            assert accuracy > 0.3
    
    def test_score(self, multinomial_data):
        """Test scoring."""
        X, y = multinomial_data
        mnb = MultinomialNaiveBayes()
        mnb.fit(X, y)
        
        score = mnb.score(X, y)
        
        assert 0 <= score <= 1
        assert score > 0.5


class TestEdgeCases:
    """Test edge cases."""
    
    def test_gaussian_with_perfect_separation(self):
        """Test Gaussian NB with perfectly separable classes."""
        np.random.seed(42)
        X0 = np.random.randn(30, 2) - 5
        X1 = np.random.randn(30, 2) + 5
        
        X = np.vstack([X0, X1])
        y = np.array([0]*30 + [1]*30)
        
        gnb = GaussianNaiveBayes()
        gnb.fit(X, y)
        
        predictions = gnb.predict(X)
        accuracy = np.mean(predictions == y)
        
        # Should achieve near-perfect accuracy
        assert accuracy > 0.95
    
    def test_multinomial_with_zero_counts(self):
        """Test Multinomial NB with some zero counts (Laplace smoothing)."""
        np.random.seed(42)
        X = np.array([
            [10, 0, 0, 5],
            [0, 10, 5, 0],
            [9, 1, 0, 4],
            [1, 9, 4, 1]
        ])
        y = np.array([0, 1, 0, 1])
        
        mnb = MultinomialNaiveBayes(alpha=1.0)
        mnb.fit(X, y)
        
        # Should handle zero counts with Laplace smoothing
        predictions = mnb.predict(X)
        assert len(predictions) == len(y)
    
    def test_single_feature(self):
        """Test with single feature."""
        np.random.seed(42)
        X = np.random.randn(100, 1)
        y = (X.ravel() > 0).astype(int)
        
        gnb = GaussianNaiveBayes()
        gnb.fit(X, y)
        
        predictions = gnb.predict(X)
        accuracy = np.mean(predictions == y)
        assert accuracy > 0.7


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
