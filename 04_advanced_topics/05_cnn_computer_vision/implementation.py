"""
CNN Computer Vision - From-scratch implementation using NumPy
Convolutional layers, pooling, and simple CNN for image classification
"""

import numpy as np
from typing import Tuple, Optional


def conv2d(
    image: np.ndarray,
    kernel: np.ndarray,
    stride: int = 1,
    padding: int = 0
) -> np.ndarray:
    """
    2D convolution operation
    
    Parameters:
    -----------
    image : np.ndarray, shape (H, W) or (H, W, C)
        Input image
    kernel : np.ndarray, shape (k_h, k_w) or (k_h, k_w, C, F)
        Convolution kernel/filter
    stride : int
        Stride for convolution
    padding : int
        Zero-padding around image
    
    Returns:
    --------
    output : np.ndarray
        Convolved output
    """
    # Handle 2D grayscale
    if image.ndim == 2:
        image = image[:, :, np.newaxis]
    if kernel.ndim == 2:
        kernel = kernel[:, :, np.newaxis, np.newaxis]
    
    H, W, C = image.shape
    k_h, k_w, _, F = kernel.shape
    
    # Add padding
    if padding > 0:
        image = np.pad(image, ((padding, padding), (padding, padding), (0, 0)), mode='constant')
        H, W = image.shape[:2]
    
    # Output dimensions
    out_h = (H - k_h) // stride + 1
    out_w = (W - k_w) // stride + 1
    
    output = np.zeros((out_h, out_w, F))
    
    # Convolution
    for f in range(F):
        for i in range(out_h):
            for j in range(out_w):
                h_start = i * stride
                w_start = j * stride
                
                region = image[h_start:h_start + k_h, w_start:w_start + k_w, :]
                output[i, j, f] = np.sum(region * kernel[:, :, :, f])
    
    return output.squeeze()


def max_pool2d(
    image: np.ndarray,
    pool_size: int = 2,
    stride: Optional[int] = None
) -> np.ndarray:
    """
    Max pooling operation
    
    Parameters:
    -----------
    image : np.ndarray, shape (H, W) or (H, W, C)
        Input image
    pool_size : int
        Size of pooling window
    stride : int, optional
        Stride for pooling (defaults to pool_size)
    
    Returns:
    --------
    output : np.ndarray
        Pooled output
    """
    if stride is None:
        stride = pool_size
    
    # Handle 2D grayscale
    if image.ndim == 2:
        image = image[:, :, np.newaxis]
        squeeze_output = True
    else:
        squeeze_output = False
    
    H, W, C = image.shape
    
    # Output dimensions
    out_h = (H - pool_size) // stride + 1
    out_w = (W - pool_size) // stride + 1
    
    output = np.zeros((out_h, out_w, C))
    
    # Max pooling
    for c in range(C):
        for i in range(out_h):
            for j in range(out_w):
                h_start = i * stride
                w_start = j * stride
                
                region = image[h_start:h_start + pool_size, w_start:w_start + pool_size, c]
                output[i, j, c] = np.max(region)
    
    return output.squeeze() if squeeze_output else output


class SimpleCNN:
    """
    Simple CNN for binary classification
    
    Architecture: Conv -> ReLU -> MaxPool -> Flatten -> Dense -> Sigmoid
    """
    
    def __init__(
        self,
        input_shape: Tuple[int, int],
        n_filters: int = 8,
        kernel_size: int = 3,
        pool_size: int = 2,
        learning_rate: float = 0.01,
        random_state: Optional[int] = None
    ):
        self.input_shape = input_shape
        self.n_filters = n_filters
        self.kernel_size = kernel_size
        self.pool_size = pool_size
        self.learning_rate = learning_rate
        
        if random_state is not None:
            np.random.seed(random_state)
        
        # Initialize weights
        self.conv_kernel = np.random.randn(kernel_size, kernel_size, 1, n_filters) * 0.1
        self.conv_bias = np.zeros((1, 1, n_filters))
        
        # Calculate dense layer input size
        conv_out_h = input_shape[0] - kernel_size + 1
        conv_out_w = input_shape[1] - kernel_size + 1
        pool_out_h = conv_out_h // pool_size
        pool_out_w = conv_out_w // pool_size
        self.dense_input_size = pool_out_h * pool_out_w * n_filters
        
        self.dense_weights = np.random.randn(self.dense_input_size, 1) * 0.1
        self.dense_bias = np.zeros((1, 1))
        
        self.loss_history_ = []
    
    def forward(self, X: np.ndarray) -> Tuple[np.ndarray, dict]:
        """Forward pass"""
        cache = {}
        
        # Conv + ReLU
        conv_out = conv2d(X, self.conv_kernel, stride=1, padding=0) + self.conv_bias
        cache['conv_out'] = conv_out
        
        relu_out = np.maximum(0, conv_out)
        cache['relu_out'] = relu_out
        
        # Max pool
        pool_out = max_pool2d(relu_out, pool_size=self.pool_size)
        cache['pool_out'] = pool_out
        
        # Flatten
        flat = pool_out.flatten().reshape(1, -1)
        cache['flat'] = flat
        
        # Dense + Sigmoid
        logits = flat @ self.dense_weights + self.dense_bias
        cache['logits'] = logits
        
        output = 1 / (1 + np.exp(-logits))
        
        return output, cache
    
    def predict(self, X: np.ndarray) -> int:
        """Predict class (0 or 1)"""
        output, _ = self.forward(X)
        return int(output[0, 0] > 0.5)
    
    def predict_proba(self, X: np.ndarray) -> float:
        """Predict probability"""
        output, _ = self.forward(X)
        return output[0, 0]
    
    def train_step(self, X: np.ndarray, y: float) -> float:
        """Single training step"""
        # Forward pass
        output, cache = self.forward(X)
        
        # Compute loss (binary cross-entropy)
        eps = 1e-10
        loss = -(y * np.log(output + eps) + (1 - y) * np.log(1 - output + eps))
        
        # Backward pass (simplified)
        d_output = output - y
        
        # Dense layer gradients
        d_dense_weights = cache['flat'].T @ d_output
        d_dense_bias = d_output
        
        # Update dense layer
        self.dense_weights -= self.learning_rate * d_dense_weights
        self.dense_bias -= self.learning_rate * d_dense_bias
        
        return loss[0, 0]
    
    def fit(self, X_list: list, y_list: list, epochs: int = 100):
        """Train the CNN"""
        for epoch in range(epochs):
            epoch_loss = 0
            for X, y in zip(X_list, y_list):
                loss = self.train_step(X, y)
                epoch_loss += loss
            
            avg_loss = epoch_loss / len(X_list)
            if epoch % 20 == 0:
                self.loss_history_.append(avg_loss)
        
        return self


def generate_product_images(
    n_samples: int = 100,
    image_size: Tuple[int, int] = (28, 28),
    n_products_range: Tuple[int, int] = (1, 4),
    random_state: Optional[int] = None
) -> Tuple[list, list]:
    """
    Generate synthetic product shelf images
    
    Images show colored rectangles (product packs) on a shelf.
    Binary classification: stockout (0) vs well-stocked (1)
    
    Returns:
    --------
    images : list of np.ndarray
        List of images, each shape (H, W)
    labels : list of int
        Binary labels (0=few products/stockout, 1=many products/stocked)
    """
    if random_state is not None:
        np.random.seed(random_state)
    
    images = []
    labels = []
    
    for _ in range(n_samples):
        # Create blank shelf
        img = np.ones(image_size) * 0.2  # Gray background
        
        # Number of products (determines label)
        n_products = np.random.randint(n_products_range[0], n_products_range[1] + 1)
        label = 1 if n_products >= 2 else 0  # Well-stocked if 2+ products
        
        # Draw product boxes
        for _ in range(n_products):
            # Random product size and position
            h = np.random.randint(5, 12)
            w = np.random.randint(4, 10)
            y = np.random.randint(0, image_size[0] - h)
            x = np.random.randint(0, image_size[1] - w)
            
            # Random intensity (different product colors/shades)
            intensity = np.random.uniform(0.5, 1.0)
            
            # Draw rectangle
            img[y:y+h, x:x+w] = intensity
        
        images.append(img)
        labels.append(label)
    
    return images, labels


if __name__ == "__main__":
    print("=" * 60)
    print("CNN Computer Vision - Product Shelf Detection")
    print("=" * 60)
    
    # Generate synthetic product images
    print("\nGenerating synthetic product shelf images...")
    X_train, y_train = generate_product_images(n_samples=80, random_state=42)
    X_test, y_test = generate_product_images(n_samples=20, random_state=43)
    
    print(f"Training samples: {len(X_train)}")
    print(f"Test samples: {len(X_test)}")
    print(f"Image shape: {X_train[0].shape}")
    
    # Train simple CNN
    print("\nTraining simple CNN...")
    cnn = SimpleCNN(
        input_shape=(28, 28),
        n_filters=8,
        kernel_size=3,
        pool_size=2,
        learning_rate=0.01,
        random_state=42
    )
    cnn.fit(X_train, y_train, epochs=100)
    
    # Evaluate
    print("\nEvaluating...")
    train_preds = [cnn.predict(x) for x in X_train]
    test_preds = [cnn.predict(x) for x in X_test]
    
    train_acc = np.mean(np.array(train_preds) == np.array(y_train))
    test_acc = np.mean(np.array(test_preds) == np.array(y_test))
    
    print(f"Training accuracy: {train_acc:.3f}")
    print(f"Test accuracy: {test_acc:.3f}")
    
    # Demo convolution
    print("\n" + "=" * 60)
    print("Convolution Demo")
    print("=" * 60)
    test_image = X_test[0]
    kernel = np.array([[1, 0, -1],
                       [2, 0, -2],
                       [1, 0, -1]])  # Edge detection
    
    conv_result = conv2d(test_image, kernel)
    print(f"Input shape: {test_image.shape}")
    print(f"Kernel shape: {kernel.shape}")
    print(f"Output shape: {conv_result.shape}")
