"""
Neural Networks - From-scratch implementation using NumPy
Multi-Layer Perceptron with various activation functions
"""

import numpy as np
from typing import Tuple, List, Optional


class Activation:
    """Base class for activation functions"""
    
    @staticmethod
    def forward(x: np.ndarray) -> np.ndarray:
        raise NotImplementedError
    
    @staticmethod
    def backward(x: np.ndarray) -> np.ndarray:
        raise NotImplementedError


class ReLU(Activation):
    """Rectified Linear Unit activation"""
    
    @staticmethod
    def forward(x: np.ndarray) -> np.ndarray:
        return np.maximum(0, x)
    
    @staticmethod
    def backward(x: np.ndarray) -> np.ndarray:
        return (x > 0).astype(float)


class Sigmoid(Activation):
    """Sigmoid activation"""
    
    @staticmethod
    def forward(x: np.ndarray) -> np.ndarray:
        return 1 / (1 + np.exp(-np.clip(x, -500, 500)))
    
    @staticmethod
    def backward(x: np.ndarray) -> np.ndarray:
        s = Sigmoid.forward(x)
        return s * (1 - s)


class Softmax(Activation):
    """Softmax activation for multi-class classification"""
    
    @staticmethod
    def forward(x: np.ndarray) -> np.ndarray:
        exp_x = np.exp(x - np.max(x, axis=1, keepdims=True))
        return exp_x / np.sum(exp_x, axis=1, keepdims=True)
    
    @staticmethod
    def backward(x: np.ndarray) -> np.ndarray:
        # For softmax with cross-entropy, gradient is computed together
        return np.ones_like(x)


class MLPClassifier:
    """
    Multi-Layer Perceptron Classifier
    
    Parameters:
    -----------
    hidden_layers : list of int
        Number of neurons in each hidden layer
    activation : str, default='relu'
        Activation function ('relu', 'sigmoid')
    learning_rate : float, default=0.01
        Learning rate for gradient descent
    max_iter : int, default=1000
        Maximum number of iterations
    batch_size : int, default=32
        Mini-batch size for SGD
    random_state : int, optional
        Random seed for reproducibility
    """
    
    def __init__(
        self,
        hidden_layers: List[int] = [64, 32],
        activation: str = 'relu',
        learning_rate: float = 0.01,
        max_iter: int = 1000,
        batch_size: int = 32,
        random_state: Optional[int] = None
    ):
        self.hidden_layers = hidden_layers
        self.activation = activation
        self.learning_rate = learning_rate
        self.max_iter = max_iter
        self.batch_size = batch_size
        self.random_state = random_state
        
        # Initialize activation function
        if activation == 'relu':
            self.activation_fn = ReLU()
        elif activation == 'sigmoid':
            self.activation_fn = Sigmoid()
        else:
            raise ValueError(f"Unknown activation: {activation}")
        
        # Model parameters (initialized during fit)
        self.weights_: List[np.ndarray] = []
        self.biases_: List[np.ndarray] = []
        self.loss_history_: List[float] = []
        self.n_classes_: int = 0
        
        if random_state is not None:
            np.random.seed(random_state)
    
    def _initialize_weights(self, n_features: int, n_classes: int):
        """Initialize weights using He initialization"""
        layers = [n_features] + self.hidden_layers + [n_classes]
        
        self.weights_ = []
        self.biases_ = []
        
        for i in range(len(layers) - 1):
            # He initialization for ReLU
            scale = np.sqrt(2.0 / layers[i])
            W = np.random.randn(layers[i], layers[i + 1]) * scale
            b = np.zeros((1, layers[i + 1]))
            
            self.weights_.append(W)
            self.biases_.append(b)
    
    def _forward_pass(self, X: np.ndarray) -> Tuple[List[np.ndarray], List[np.ndarray]]:
        """Forward propagation through the network"""
        activations = [X]
        z_values = []
        
        for i, (W, b) in enumerate(zip(self.weights_, self.biases_)):
            z = activations[-1] @ W + b
            z_values.append(z)
            
            if i < len(self.weights_) - 1:
                # Hidden layers
                a = self.activation_fn.forward(z)
            else:
                # Output layer (softmax)
                a = Softmax.forward(z)
            
            activations.append(a)
        
        return activations, z_values
    
    def _backward_pass(
        self,
        X: np.ndarray,
        y: np.ndarray,
        activations: List[np.ndarray],
        z_values: List[np.ndarray]
    ) -> Tuple[List[np.ndarray], List[np.ndarray]]:
        """Backpropagation to compute gradients"""
        m = X.shape[0]
        
        # One-hot encode labels
        y_one_hot = np.zeros((m, self.n_classes_))
        y_one_hot[np.arange(m), y] = 1
        
        # Gradients
        dW = []
        db = []
        
        # Output layer gradient (softmax + cross-entropy)
        delta = activations[-1] - y_one_hot
        
        # Backpropagate through layers
        for i in range(len(self.weights_) - 1, -1, -1):
            # Compute weight and bias gradients
            dW_i = activations[i].T @ delta / m
            db_i = np.sum(delta, axis=0, keepdims=True) / m
            
            dW.insert(0, dW_i)
            db.insert(0, db_i)
            
            # Compute delta for previous layer
            if i > 0:
                delta = (delta @ self.weights_[i].T) * self.activation_fn.backward(z_values[i - 1])
        
        return dW, db
    
    def _compute_loss(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """Compute cross-entropy loss"""
        m = y_true.shape[0]
        # Clip to avoid log(0)
        y_pred_clipped = np.clip(y_pred, 1e-10, 1 - 1e-10)
        loss = -np.sum(y_true * np.log(y_pred_clipped)) / m
        return loss
    
    def fit(self, X: np.ndarray, y: np.ndarray) -> 'MLPClassifier':
        """Train the neural network"""
        n_samples, n_features = X.shape
        self.n_classes_ = len(np.unique(y))
        
        # Initialize weights
        self._initialize_weights(n_features, self.n_classes_)
        
        # Training loop
        for epoch in range(self.max_iter):
            # Shuffle data
            indices = np.random.permutation(n_samples)
            X_shuffled = X[indices]
            y_shuffled = y[indices]
            
            # Mini-batch gradient descent
            for i in range(0, n_samples, self.batch_size):
                X_batch = X_shuffled[i:i + self.batch_size]
                y_batch = y_shuffled[i:i + self.batch_size]
                
                # Forward pass
                activations, z_values = self._forward_pass(X_batch)
                
                # Backward pass
                dW, db = self._backward_pass(X_batch, y_batch, activations, z_values)
                
                # Update weights
                for j in range(len(self.weights_)):
                    self.weights_[j] -= self.learning_rate * dW[j]
                    self.biases_[j] -= self.learning_rate * db[j]
            
            # Compute loss on full dataset (for monitoring)
            if epoch % 100 == 0:
                activations, _ = self._forward_pass(X)
                y_one_hot = np.zeros((n_samples, self.n_classes_))
                y_one_hot[np.arange(n_samples), y] = 1
                loss = self._compute_loss(y_one_hot, activations[-1])
                self.loss_history_.append(loss)
        
        return self
    
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Predict class probabilities"""
        activations, _ = self._forward_pass(X)
        return activations[-1]
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict class labels"""
        proba = self.predict_proba(X)
        return np.argmax(proba, axis=1)
    
    def score(self, X: np.ndarray, y: np.ndarray) -> float:
        """Compute accuracy score"""
        predictions = self.predict(X)
        return np.mean(predictions == y)


def gradient_check(
    model: MLPClassifier,
    X: np.ndarray,
    y: np.ndarray,
    epsilon: float = 1e-5
) -> Tuple[float, float]:
    """
    Numerical gradient checking
    
    Returns:
    --------
    max_diff : float
        Maximum difference between numerical and analytical gradients
    relative_diff : float
        Relative difference
    """
    # Forward pass to get analytical gradients
    activations, z_values = model._forward_pass(X)
    dW_analytical, db_analytical = model._backward_pass(X, y, activations, z_values)
    
    # Flatten analytical gradients
    analytical_grad = np.concatenate([dw.flatten() for dw in dW_analytical] + 
                                     [db.flatten() for db in db_analytical])
    
    # Compute numerical gradients
    numerical_grad = []
    
    # Check weights
    for layer_idx, W in enumerate(model.weights_):
        for i in range(W.shape[0]):
            for j in range(W.shape[1]):
                # Save original value
                original = W[i, j]
                
                # Compute f(x + epsilon)
                W[i, j] = original + epsilon
                activations_plus, _ = model._forward_pass(X)
                y_one_hot = np.zeros((X.shape[0], model.n_classes_))
                y_one_hot[np.arange(X.shape[0]), y] = 1
                loss_plus = model._compute_loss(y_one_hot, activations_plus[-1])
                
                # Compute f(x - epsilon)
                W[i, j] = original - epsilon
                activations_minus, _ = model._forward_pass(X)
                loss_minus = model._compute_loss(y_one_hot, activations_minus[-1])
                
                # Numerical gradient
                grad = (loss_plus - loss_minus) / (2 * epsilon)
                numerical_grad.append(grad)
                
                # Restore original value
                W[i, j] = original
    
    # Check biases
    for layer_idx, b in enumerate(model.biases_):
        for i in range(b.shape[1]):
            original = b[0, i]
            
            b[0, i] = original + epsilon
            activations_plus, _ = model._forward_pass(X)
            loss_plus = model._compute_loss(y_one_hot, activations_plus[-1])
            
            b[0, i] = original - epsilon
            activations_minus, _ = model._forward_pass(X)
            loss_minus = model._compute_loss(y_one_hot, activations_minus[-1])
            
            grad = (loss_plus - loss_minus) / (2 * epsilon)
            numerical_grad.append(grad)
            
            b[0, i] = original
    
    numerical_grad = np.array(numerical_grad)
    
    # Compute differences
    diff = np.abs(analytical_grad - numerical_grad)
    max_diff = np.max(diff)
    relative_diff = max_diff / (np.abs(analytical_grad).max() + np.abs(numerical_grad).max() + 1e-8)
    
    return max_diff, relative_diff


def generate_spg_outlet_data(
    n_samples: int = 1000,
    n_features: int = 10,
    n_classes: int = 3,
    random_state: Optional[int] = None
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generate synthetic SPG outlet stockout prediction dataset
    
    Features: sales, SKU count, visit frequency, OOS rate, etc.
    Classes: 0=Low risk, 1=Medium risk, 2=High risk stockout
    """
    if random_state is not None:
        np.random.seed(random_state)
    
    X = np.random.randn(n_samples, n_features)
    
    # Create non-linear decision boundaries
    feature_importance = np.array([3, 2, 1.5, 1, 0.8, 0.6, 0.4, 0.3, 0.2, 0.1])
    decision_function = X @ feature_importance[:n_features]
    decision_function += 0.5 * np.sin(X[:, 0] * 2)  # Non-linearity
    
    # Create classes based on thresholds
    y = np.zeros(n_samples, dtype=int)
    y[decision_function > 0.5] = 1
    y[decision_function > 1.5] = 2
    
    return X, y


if __name__ == "__main__":
    # Example usage
    print("=" * 60)
    print("Neural Networks - MLP Classification")
    print("=" * 60)
    
    # Generate synthetic SPG outlet data
    X, y = generate_spg_outlet_data(n_samples=500, random_state=42)
    
    # Split data
    split_idx = int(0.8 * len(X))
    X_train, X_test = X[:split_idx], X[split_idx:]
    y_train, y_test = y[:split_idx], y[split_idx:]
    
    # Train MLP
    print("\nTraining MLP with 2 hidden layers...")
    mlp = MLPClassifier(
        hidden_layers=[32, 16],
        activation='relu',
        learning_rate=0.01,
        max_iter=500,
        batch_size=32,
        random_state=42
    )
    mlp.fit(X_train, y_train)
    
    # Evaluate
    train_acc = mlp.score(X_train, y_train)
    test_acc = mlp.score(X_test, y_test)
    
    print(f"\nResults:")
    print(f"  Training accuracy: {train_acc:.4f}")
    print(f"  Test accuracy: {test_acc:.4f}")
    print(f"  Final loss: {mlp.loss_history_[-1]:.4f}")
    
    # Gradient check (on small sample)
    print("\nGradient check (numerical vs analytical):")
    X_small = X_train[:10]
    y_small = y_train[:10]
    max_diff, rel_diff = gradient_check(mlp, X_small, y_small)
    print(f"  Max difference: {max_diff:.2e}")
    print(f"  Relative difference: {rel_diff:.2e}")
    print(f"  {'✓ PASS' if rel_diff < 1e-5 else '✗ FAIL'}")
