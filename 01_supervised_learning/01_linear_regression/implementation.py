"""
Linear Regression Implementation from Scratch
============================================

Implementasi Linear Regression menggunakan NumPy dengan dua metode optimasi:
1. Normal Equation (Closed-form solution)
2. Gradient Descent (Iterative approach)
"""

import numpy as np
import matplotlib.pyplot as plt
from typing import Tuple, Optional

class LinearRegressionFromScratch:
    """
    Linear Regression implementation from scratch using NumPy
    
    Parameters:
    -----------
    learning_rate : float, default=0.01
        Learning rate untuk gradient descent
    n_iterations : int, default=1000
        Jumlah iterasi untuk gradient descent
    method : str, default='normal_equation'
        Metode optimasi: 'normal_equation' atau 'gradient_descent'
    """
    
    def __init__(self, learning_rate: float = 0.01, n_iterations: int = 1000, 
                 method: str = 'normal_equation'):
        self.learning_rate = learning_rate
        self.n_iterations = n_iterations
        self.method = method
        self.weights = None
        self.bias = None
        self.cost_history = []
    
    def _add_bias_term(self, X: np.ndarray) -> np.ndarray:
        """Menambahkan bias term (intercept) ke feature matrix"""
        return np.column_stack([np.ones(X.shape[0]), X])
    
    def _compute_cost(self, X: np.ndarray, y: np.ndarray) -> float:
        """Menghitung Mean Squared Error cost function"""
        n_samples = X.shape[0]
        predictions = self.predict(X)
        cost = (1 / (2 * n_samples)) * np.sum((predictions - y) ** 2)
        return cost
    
    def fit_normal_equation(self, X: np.ndarray, y: np.ndarray) -> None:
        """
        Training menggunakan Normal Equation (Closed-form solution)
        θ = (X^T * X)^(-1) * X^T * y
        """
        # Menambahkan bias term
        X_with_bias = self._add_bias_term(X)
        
        # Normal Equation formula
        # θ = (X^T * X)^(-1) * X^T * y
        try:
            theta = np.linalg.inv(X_with_bias.T @ X_with_bias) @ X_with_bias.T @ y
        except np.linalg.LinAlgError:
            # Jika matrix singular, gunakan pseudo-inverse
            theta = np.linalg.pinv(X_with_bias.T @ X_with_bias) @ X_with_bias.T @ y
        
        # Pisahkan bias dan weights
        self.bias = theta[0]
        self.weights = theta[1:] if len(theta) > 1 else np.array([theta[1]])
    
    def fit_gradient_descent(self, X: np.ndarray, y: np.ndarray) -> None:
        """
        Training menggunakan Gradient Descent
        """
        n_samples, n_features = X.shape
        
        # Initialize parameters
        self.weights = np.zeros(n_features)
        self.bias = 0
        self.cost_history = []
        
        # Gradient Descent
        for i in range(self.n_iterations):
            # Forward pass: compute predictions
            predictions = X @ self.weights + self.bias
            
            # Compute cost
            cost = (1 / (2 * n_samples)) * np.sum((predictions - y) ** 2)
            self.cost_history.append(cost)
            
            # Compute gradients
            dw = (1 / n_samples) * X.T @ (predictions - y)
            db = (1 / n_samples) * np.sum(predictions - y)
            
            # Update parameters
            self.weights -= self.learning_rate * dw
            self.bias -= self.learning_rate * db
    
    def fit(self, X: np.ndarray, y: np.ndarray) -> 'LinearRegressionFromScratch':
        """
        Training model dengan metode yang dipilih
        
        Parameters:
        -----------
        X : np.ndarray, shape (n_samples, n_features)
            Training data
        y : np.ndarray, shape (n_samples,)
            Target values
        """
        # Validasi input
        X = np.asarray(X)
        y = np.asarray(y)
        
        if X.ndim == 1:
            X = X.reshape(-1, 1)
        
        if self.method == 'normal_equation':
            self.fit_normal_equation(X, y)
        elif self.method == 'gradient_descent':
            self.fit_gradient_descent(X, y)
        else:
            raise ValueError("Method harus 'normal_equation' atau 'gradient_descent'")
        
        return self
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Membuat prediksi
        
        Parameters:
        -----------
        X : np.ndarray, shape (n_samples, n_features)
            Input data
            
        Returns:
        --------
        predictions : np.ndarray, shape (n_samples,)
            Predicted values
        """
        if self.weights is None or self.bias is None:
            raise ValueError("Model belum di-training. Panggil fit() terlebih dahulu.")
        
        X = np.asarray(X)
        if X.ndim == 1:
            X = X.reshape(-1, 1)
        
        return X @ self.weights + self.bias
    
    def score(self, X: np.ndarray, y: np.ndarray) -> float:
        """
        Menghitung R² score (coefficient of determination)
        
        R² = 1 - (SS_res / SS_tot)
        """
        y_pred = self.predict(X)
        ss_res = np.sum((y - y_pred) ** 2)
        ss_tot = np.sum((y - np.mean(y)) ** 2)
        return 1 - (ss_res / ss_tot)
    
    def plot_cost_history(self) -> None:
        """Plot cost function history untuk gradient descent"""
        if not self.cost_history:
            print("Cost history tidak tersedia. Gunakan method='gradient_descent'")
            return
        
        plt.figure(figsize=(10, 6))
        plt.plot(self.cost_history)
        plt.title('Cost Function History')
        plt.xlabel('Iterations')
        plt.ylabel('Cost (MSE)')
        plt.grid(True)
        plt.show()

def generate_sample_data(n_samples: int = 100, noise: float = 0.1, 
                        random_state: int = 42) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generate sample data untuk testing
    """
    np.random.seed(random_state)
    X = np.linspace(0, 10, n_samples).reshape(-1, 1)
    y = 2 * X.ravel() + 3 + np.random.normal(0, noise, n_samples)
    return X, y

def compare_methods():
    """
    Membandingkan Normal Equation vs Gradient Descent
    """
    # Generate sample data
    X, y = generate_sample_data(n_samples=100, noise=1.0)
    
    # Normal Equation
    model_normal = LinearRegressionFromScratch(method='normal_equation')
    model_normal.fit(X, y)
    
    # Gradient Descent
    model_gd = LinearRegressionFromScratch(
        method='gradient_descent', 
        learning_rate=0.01, 
        n_iterations=1000
    )
    model_gd.fit(X, y)
    
    # Predictions
    X_test = np.linspace(0, 10, 50).reshape(-1, 1)
    y_pred_normal = model_normal.predict(X_test)
    y_pred_gd = model_gd.predict(X_test)
    
    # Plot results
    plt.figure(figsize=(15, 5))
    
    # Plot 1: Normal Equation
    plt.subplot(1, 3, 1)
    plt.scatter(X, y, alpha=0.6, label='Data points')
    plt.plot(X_test, y_pred_normal, 'r-', linewidth=2, label='Normal Equation')
    plt.title(f'Normal Equation\\nR² = {model_normal.score(X, y):.4f}')
    plt.xlabel('X')
    plt.ylabel('y')
    plt.legend()
    plt.grid(True)
    
    # Plot 2: Gradient Descent
    plt.subplot(1, 3, 2)
    plt.scatter(X, y, alpha=0.6, label='Data points')
    plt.plot(X_test, y_pred_gd, 'g-', linewidth=2, label='Gradient Descent')
    plt.title(f'Gradient Descent\\nR² = {model_gd.score(X, y):.4f}')
    plt.xlabel('X')
    plt.ylabel('y')
    plt.legend()
    plt.grid(True)
    
    # Plot 3: Cost History
    plt.subplot(1, 3, 3)
    plt.plot(model_gd.cost_history)
    plt.title('Cost Function History')
    plt.xlabel('Iterations')
    plt.ylabel('Cost (MSE)')
    plt.grid(True)
    
    plt.tight_layout()
    plt.show()
    
    # Print parameters
    print("\\n" + "="*50)
    print("COMPARISON RESULTS")
    print("="*50)
    print(f"Normal Equation:")
    print(f"  Weights: {model_normal.weights}")
    print(f"  Bias: {model_normal.bias:.4f}")
    print(f"  R² Score: {model_normal.score(X, y):.4f}")
    
    print(f"\\nGradient Descent:")
    print(f"  Weights: {model_gd.weights}")
    print(f"  Bias: {model_gd.bias:.4f}")
    print(f"  R² Score: {model_gd.score(X, y):.4f}")

if __name__ == "__main__":
    print("Linear Regression from Scratch - Demo")
    print("="*40)
    
    # Generate sample data
    X, y = generate_sample_data()
    print(f"Generated data: {X.shape[0]} samples, {X.shape[1]} features")
    
    # Test both methods
    compare_methods()