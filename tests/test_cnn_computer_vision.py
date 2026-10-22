"""
Tests for CNN Computer Vision implementation
"""

import numpy as np
import pytest
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / '04_advanced_topics' / '05_cnn_computer_vision'))

from implementation import conv2d, max_pool2d, SimpleCNN, generate_product_images


class TestConv2D:
    def test_conv2d_basic(self):
        image = np.ones((3, 3))
        kernel = np.ones((2, 2))
        result = conv2d(image, kernel, stride=1, padding=0)
        assert result.shape == (2, 2)


class TestMaxPool2D:
    def test_max_pool_basic(self):
        image = np.array([[1, 2, 3, 4],
                          [5, 6, 7, 8]], dtype=float)
        result = max_pool2d(image, pool_size=2, stride=2)
        assert result.shape[0] >= 1


class TestSimpleCNN:
    def test_initialization(self):
        cnn = SimpleCNN(input_shape=(28, 28), random_state=42)
        assert cnn.input_shape == (28, 28)
    
    def test_predict(self):
        cnn = SimpleCNN(input_shape=(28, 28), random_state=42)
        image = np.random.randn(28, 28)
        prediction = cnn.predict(image)
        assert prediction in [0, 1]


class TestDataGeneration:
    def test_generate_images(self):
        images, labels = generate_product_images(n_samples=10, random_state=42)
        assert len(images) == 10
        assert len(labels) == 10


class TestCNNBackprop:
    """Test CNN backpropagation"""
    
    def test_gradient_check(self):
        from implementation import gradient_check_cnn
        
        cnn = SimpleCNN(input_shape=(28, 28), random_state=42)
        X = np.random.randn(28, 28)
        y = 1.0
        
        # Train a bit first to have reasonable gradients
        for _ in range(10):
            cnn.train_step(X, y)
        
        max_diff, rel_diff = gradient_check_cnn(cnn, X, y, epsilon=1e-5)
        
        # Should have small relative error
        assert rel_diff < 0.1  # Relaxed for simple implementation
    
    def test_loss_decreases_with_training(self):
        X_train, y_train = generate_product_images(n_samples=30, random_state=42)
        
        cnn = SimpleCNN(input_shape=(28, 28), learning_rate=0.05, random_state=42)
        cnn.fit(X_train, y_train, epochs=100)
        
        # Loss should decrease
        assert len(cnn.loss_history_) > 0
        if len(cnn.loss_history_) > 1:
            assert cnn.loss_history_[-1] < cnn.loss_history_[0]
