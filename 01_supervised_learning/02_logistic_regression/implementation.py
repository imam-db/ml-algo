"""
Logistic Regression Implementation from Scratch

This module provides a complete implementation of Logistic Regression
without using scikit-learn, demonstrating the core concepts and mathematics.

Author: ML Learning Repository
Date: 2024
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from typing import Tuple, Optional
import warnings
warnings.filterwarnings('ignore')


class LogisticRegressionScratch:
    """
    Logistic Regression implementation from scratch
    
    Parameters:
    -----------
    learning_rate : float, default=0.01
        Learning rate for gradient descent
    max_iterations : int, default=1000
        Maximum number of iterations for gradient descent
    tolerance : float, default=1e-6
        Tolerance for convergence
    fit_intercept : bool, default=True
        Whether to fit an intercept term
    verbose : bool, default=False
        Whether to print training progress
    """
    
    def __init__(self, learning_rate: float = 0.01, max_iterations: int = 1000,
                 tolerance: float = 1e-6, fit_intercept: bool = True,
                 verbose: bool = False):
        self.learning_rate = learning_rate
        self.max_iterations = max_iterations
        self.tolerance = tolerance
        self.fit_intercept = fit_intercept
        self.verbose = verbose
        
        # Will be set during training
        self.weights = None
        self.cost_history = []
        self.n_features = None
        self.converged = False
        
    def _add_intercept(self, X: np.ndarray) -> np.ndarray:
        """Add bias column to the feature matrix"""
        intercept = np.ones((X.shape[0], 1))
        return np.concatenate((intercept, X), axis=1)
    
    def _sigmoid(self, z: np.ndarray) -> np.ndarray:
        """
        Sigmoid activation function
        
        σ(z) = 1 / (1 + e^(-z))
        
        Parameters:
        -----------
        z : np.ndarray
            Linear combination of features and weights
            
        Returns:
        --------
        np.ndarray
            Sigmoid values between 0 and 1
        """
        # Clip z to prevent overflow
        z = np.clip(z, -500, 500)
        return 1 / (1 + np.exp(-z))
    
    def _cost_function(self, h: np.ndarray, y: np.ndarray) -> float:
        """
        Calculate the logistic regression cost function (log-likelihood)
        
        J(θ) = -(1/m) * Σ[y*log(h) + (1-y)*log(1-h)]
        
        Parameters:
        -----------
        h : np.ndarray
            Predicted probabilities
        y : np.ndarray
            True labels
            
        Returns:
        --------
        float
            Cost value
        """
        m = y.shape[0]
        
        # Add small epsilon to prevent log(0)
        epsilon = 1e-15
        h = np.clip(h, epsilon, 1 - epsilon)
        
        cost = -(1/m) * np.sum(y * np.log(h) + (1 - y) * np.log(1 - h))
        return cost
    
    def _gradient(self, X: np.ndarray, h: np.ndarray, y: np.ndarray) -> np.ndarray:
        """
        Calculate the gradient of the cost function
        
        ∇J(θ) = (1/m) * X^T * (h - y)
        
        Parameters:
        -----------
        X : np.ndarray
            Feature matrix
        h : np.ndarray
            Predicted probabilities
        y : np.ndarray
            True labels
            
        Returns:
        --------
        np.ndarray
            Gradient vector
        """
        m = y.shape[0]
        return (1/m) * X.T.dot(h - y)
    
    def fit(self, X: np.ndarray, y: np.ndarray) -> 'LogisticRegressionScratch':
        """
        Train the logistic regression model
        
        Parameters:
        -----------
        X : np.ndarray
            Feature matrix of shape (m, n)
        y : np.ndarray
            Target vector of shape (m,)
            
        Returns:
        --------
        self : LogisticRegressionScratch
            Fitted model
        """
        # Convert to numpy arrays
        X = np.array(X)
        y = np.array(y)
        
        # Store number of samples and features
        m, n = X.shape
        self.n_features = n
        
        # Add intercept term if requested
        if self.fit_intercept:
            X = self._add_intercept(X)
            n += 1
        
        # Initialize weights randomly
        np.random.seed(42)
        self.weights = np.random.normal(0, 0.01, size=n)
        
        # Clear cost history
        self.cost_history = []
        previous_cost = float('inf')
        
        # Gradient descent
        for i in range(self.max_iterations):
            # Forward pass
            z = X.dot(self.weights)
            h = self._sigmoid(z)
            
            # Calculate cost
            cost = self._cost_function(h, y)
            self.cost_history.append(cost)
            
            # Calculate gradient
            gradient = self._gradient(X, h, y)
            
            # Update weights
            self.weights -= self.learning_rate * gradient
            
            # Check for convergence
            if abs(previous_cost - cost) < self.tolerance:
                self.converged = True
                if self.verbose:
                    print(f"Converged after {i+1} iterations")
                break
            
            previous_cost = cost
            
            # Print progress
            if self.verbose and (i % 100 == 0):
                print(f"Iteration {i}, Cost: {cost:.6f}")
        
        if not self.converged and self.verbose:
            print(f"Did not converge after {self.max_iterations} iterations")
        
        return self
    
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """
        Predict class probabilities
        
        Parameters:
        -----------
        X : np.ndarray
            Feature matrix
            
        Returns:
        --------
        np.ndarray
            Predicted probabilities
        """
        if self.weights is None:
            raise ValueError("Model must be trained before making predictions")
        
        X = np.array(X)
        
        # Add intercept if it was used during training
        if self.fit_intercept:
            X = self._add_intercept(X)
        
        # Calculate probabilities
        z = X.dot(self.weights)
        return self._sigmoid(z)
    
    def predict(self, X: np.ndarray, threshold: float = 0.5) -> np.ndarray:
        """
        Make binary predictions
        
        Parameters:
        -----------
        X : np.ndarray
            Feature matrix
        threshold : float, default=0.5
            Decision threshold
            
        Returns:
        --------
        np.ndarray
            Binary predictions
        """
        probabilities = self.predict_proba(X)
        return (probabilities >= threshold).astype(int)
    
    def score(self, X: np.ndarray, y: np.ndarray) -> float:
        """
        Calculate accuracy score
        
        Parameters:
        -----------
        X : np.ndarray
            Feature matrix
        y : np.ndarray
            True labels
            
        Returns:
        --------
        float
            Accuracy score
        """
        y_pred = self.predict(X)
        return np.mean(y_pred == y)
    
    def plot_cost_history(self):
        """Plot the cost function over iterations"""
        if not self.cost_history:
            print("No cost history available. Train the model first.")
            return
        
        plt.figure(figsize=(10, 6))
        plt.plot(self.cost_history, 'b-', linewidth=2)
        plt.title('Cost Function Over Iterations', fontsize=16)
        plt.xlabel('Iteration', fontsize=14)
        plt.ylabel('Cost', fontsize=14)
        plt.grid(True, alpha=0.3)
        plt.show()
    
    def get_feature_importance(self) -> np.ndarray:
        """
        Get feature importance based on absolute weights
        
        Returns:
        --------
        np.ndarray
            Feature importance scores
        """
        if self.weights is None:
            raise ValueError("Model must be trained first")
        
        # Exclude intercept if it was used
        weights = self.weights[1:] if self.fit_intercept else self.weights
        return np.abs(weights)


def plot_decision_boundary(X: np.ndarray, y: np.ndarray, model: LogisticRegressionScratch, 
                          title: str = "Logistic Regression Decision Boundary"):
    """
    Plot decision boundary for 2D data
    
    Parameters:
    -----------
    X : np.ndarray
        2D feature matrix
    y : np.ndarray
        Binary labels
    model : LogisticRegressionScratch
        Trained model
    title : str
        Plot title
    """
    if X.shape[1] != 2:
        print("Can only plot decision boundary for 2D data")
        return
    
    # Create a mesh
    h = 0.02
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                         np.arange(y_min, y_max, h))
    
    # Make predictions on the mesh
    mesh_points = np.c_[xx.ravel(), yy.ravel()]
    Z = model.predict_proba(mesh_points)
    Z = Z.reshape(xx.shape)
    
    # Create the plot
    plt.figure(figsize=(12, 8))
    
    # Plot the contour and training examples
    plt.contourf(xx, yy, Z, levels=50, alpha=0.6, cmap='RdYlBu')
    plt.colorbar(label='Probability')
    
    # Plot decision boundary
    plt.contour(xx, yy, Z, levels=[0.5], colors='black', linestyles='--', linewidths=2)
    
    # Plot data points
    scatter = plt.scatter(X[:, 0], X[:, 1], c=y, cmap='RdYlBu', edgecolors='black')
    plt.colorbar(scatter, label='Class')
    
    plt.title(title, fontsize=16)
    plt.xlabel('Feature 1', fontsize=14)
    plt.ylabel('Feature 2', fontsize=14)
    plt.grid(True, alpha=0.3)
    plt.show()


def generate_sample_data(n_samples: int = 1000, noise: float = 0.1) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generate sample dataset for binary classification
    
    Parameters:
    -----------
    n_samples : int
        Number of samples to generate
    noise : float
        Amount of noise to add
        
    Returns:
    --------
    tuple
        (X, y) where X is features and y is labels
    """
    np.random.seed(42)
    
    # Generate two clusters
    X1 = np.random.multivariate_normal([2, 2], [[1, 0.5], [0.5, 1]], n_samples//2)
    X2 = np.random.multivariate_normal([-1, -1], [[1, -0.3], [-0.3, 1]], n_samples//2)
    
    # Combine the clusters
    X = np.vstack((X1, X2))
    y = np.hstack((np.ones(n_samples//2), np.zeros(n_samples//2)))
    
    # Add noise
    X += np.random.normal(0, noise, X.shape)
    
    # Shuffle the data
    indices = np.random.permutation(n_samples)
    return X[indices], y[indices]


def demonstrate_logistic_regression():
    """Demonstrate the logistic regression implementation"""
    print("=" * 60)
    print("LOGISTIC REGRESSION FROM SCRATCH DEMONSTRATION")
    print("=" * 60)
    
    # Generate sample data
    print("\n1. Generating sample data...")
    X, y = generate_sample_data(n_samples=1000, noise=0.1)
    print(f"   Generated {X.shape[0]} samples with {X.shape[1]} features")
    print(f"   Class distribution: {np.bincount(y.astype(int))}")
    
    # Split data
    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Create and train model
    print("\n2. Training logistic regression model...")
    model = LogisticRegressionScratch(
        learning_rate=0.1,
        max_iterations=1000,
        tolerance=1e-6,
        verbose=True
    )
    
    model.fit(X_train, y_train)
    
    # Make predictions
    print("\n3. Making predictions...")
    train_accuracy = model.score(X_train, y_train)
    test_accuracy = model.score(X_test, y_test)
    
    print(f"   Training accuracy: {train_accuracy:.4f}")
    print(f"   Test accuracy: {test_accuracy:.4f}")
    
    # Show feature importance
    importance = model.get_feature_importance()
    print(f"\n4. Feature importance:")
    for i, imp in enumerate(importance):
        print(f"   Feature {i+1}: {imp:.4f}")
    
    # Plot results
    print("\n5. Plotting results...")
    
    # Plot cost history
    model.plot_cost_history()
    
    # Plot decision boundary
    plot_decision_boundary(X_test, y_test, model, "Logistic Regression Decision Boundary")
    
    # Compare with sklearn
    print("\n6. Comparing with scikit-learn...")
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import accuracy_score
    
    sklearn_model = LogisticRegression(random_state=42)
    sklearn_model.fit(X_train, y_train)
    sklearn_pred = sklearn_model.predict(X_test)
    sklearn_accuracy = accuracy_score(y_test, sklearn_pred)
    
    print(f"   Our implementation accuracy: {test_accuracy:.4f}")
    print(f"   Scikit-learn accuracy: {sklearn_accuracy:.4f}")
    print(f"   Difference: {abs(test_accuracy - sklearn_accuracy):.4f}")
    
    return model, X_test, y_test


if __name__ == "__main__":
    # Run demonstration
    model, X_test, y_test = demonstrate_logistic_regression()
    
    print("\n" + "=" * 60)
    print("Try experimenting with:")
    print("- Different learning rates")
    print("- Different datasets")
    print("- Feature scaling")
    print("- Regularization (L1/L2)")
    print("=" * 60)