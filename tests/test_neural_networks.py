"""
Tests for Neural Networks implementation
"""

import numpy as np
import pytest
from pathlib import Path
import sys

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent / '01_supervised_learning' / '08_neural_networks'))

from implementation import (
    MLPClassifier, ReLU, Sigmoid, Softmax,
    gradient_check, generate_spg_outlet_data
)


class TestActivations:
    """Test activation functions"""
    
    def test_relu_forward(self):
        x = np.array([-2, -1, 0, 1, 2])
        expected = np.array([0, 0, 0, 1, 2])
        np.testing.assert_array_equal(ReLU.forward(x), expected)
    
    def test_relu_backward(self):
        x = np.array([-2, -1, 0, 1, 2])
        expected = np.array([0, 0, 0, 1, 1])
        np.testing.assert_array_equal(ReLU.backward(x), expected)
    
    def test_sigmoid_forward(self):
        x = np.array([0])
        result = Sigmoid.forward(x)
        assert np.isclose(result[0], 0.5)
    
    def test_sigmoid_backward(self):
        x = np.array([0])
        result = Sigmoid.backward(x)
        assert np.isclose(result[0], 0.25)
    
    def test_softmax_forward(self):
        x = np.array([[1, 2, 3]])
        result = Softmax.forward(x)
        assert np.isclose(np.sum(result), 1.0)
        assert np.all(result >= 0)
        assert np.all(result <= 1)


class TestMLPClassifier:
    """Test MLP Classifier"""
    
    def test_initialization(self):
        mlp = MLPClassifier(hidden_layers=[64, 32], random_state=42)
        assert mlp.hidden_layers == [64, 32]
        assert mlp.activation == 'relu'
    
    def test_fit_binary(self):
        # Simple binary classification
        np.random.seed(42)
        X = np.random.randn(100, 5)
        y = (X[:, 0] + X[:, 1] > 0).astype(int)
        
        mlp = MLPClassifier(
            hidden_layers=[16],
            learning_rate=0.1,
            max_iter=200,
            random_state=42
        )
        mlp.fit(X, y)
        
        # Should learn this simple pattern
        accuracy = mlp.score(X, y)
        assert accuracy > 0.7
    
    def test_fit_multiclass(self):
        X, y = generate_spg_outlet_data(n_samples=200, random_state=42)
        
        mlp = MLPClassifier(
            hidden_layers=[32, 16],
            learning_rate=0.01,
            max_iter=300,
            random_state=42
        )
        mlp.fit(X, y)
        
        accuracy = mlp.score(X, y)
        assert accuracy > 0.5  # Better than random (33%)
    
    def test_predict_proba(self):
        X, y = generate_spg_outlet_data(n_samples=100, random_state=42)
        
        mlp = MLPClassifier(hidden_layers=[16], random_state=42)
        mlp.fit(X, y)
        
        proba = mlp.predict_proba(X[:10])
        
        # Check properties
        assert proba.shape == (10, 3)
        assert np.allclose(np.sum(proba, axis=1), 1.0)
        assert np.all(proba >= 0)
        assert np.all(proba <= 1)
    
    def test_predict(self):
        X, y = generate_spg_outlet_data(n_samples=100, random_state=42)
        
        mlp = MLPClassifier(hidden_layers=[16], random_state=42)
        mlp.fit(X, y)
        
        predictions = mlp.predict(X[:10])
        
        assert predictions.shape == (10,)
        assert np.all((predictions >= 0) & (predictions < 3))
    
    def test_deterministic_with_seed(self):
        X, y = generate_spg_outlet_data(n_samples=100, random_state=42)
        
        mlp1 = MLPClassifier(hidden_layers=[16], random_state=42, max_iter=100)
        mlp1.fit(X, y)
        pred1 = mlp1.predict(X)
        
        mlp2 = MLPClassifier(hidden_layers=[16], random_state=42, max_iter=100)
        mlp2.fit(X, y)
        pred2 = mlp2.predict(X)
        
        np.testing.assert_array_equal(pred1, pred2)
    
    def test_loss_decreases(self):
        X, y = generate_spg_outlet_data(n_samples=200, random_state=42)
        
        mlp = MLPClassifier(
            hidden_layers=[32],
            learning_rate=0.05,
            max_iter=500,
            random_state=42
        )
        mlp.fit(X, y)
        
        # Loss should generally decrease
        assert len(mlp.loss_history_) > 0
        assert mlp.loss_history_[-1] < mlp.loss_history_[0]


class TestGradientCheck:
    """Test gradient checking"""
    
    def test_gradient_check_passes(self):
        # Use generated data to ensure all classes present
        X, y = generate_spg_outlet_data(n_samples=50, n_classes=3, random_state=42)
        
        mlp = MLPClassifier(
            hidden_layers=[8],
            learning_rate=0.01,
            max_iter=50,
            random_state=42
        )
        mlp.fit(X, y)
        
        max_diff, rel_diff = gradient_check(mlp, X[:5], y[:5], epsilon=1e-5)
        
        # Gradient check should pass (small relative error)
        assert rel_diff < 1e-3  # Relaxed threshold for numerical stability


class TestDataGeneration:
    """Test synthetic data generation"""
    
    def test_generate_data_shape(self):
        X, y = generate_spg_outlet_data(n_samples=100, n_features=10, n_classes=3)
        
        assert X.shape == (100, 10)
        assert y.shape == (100,)
    
    def test_generate_data_classes(self):
        X, y = generate_spg_outlet_data(n_samples=100, n_classes=3, random_state=42)
        
        unique_classes = np.unique(y)
        assert len(unique_classes) == 3
        assert np.all(unique_classes == np.array([0, 1, 2]))
    
    def test_generate_data_deterministic(self):
        X1, y1 = generate_spg_outlet_data(n_samples=100, random_state=42)
        X2, y2 = generate_spg_outlet_data(n_samples=100, random_state=42)
        
        np.testing.assert_array_equal(X1, X2)
        np.testing.assert_array_equal(y1, y2)
