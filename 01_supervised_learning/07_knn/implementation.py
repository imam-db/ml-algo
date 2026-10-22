"""
K-Nearest Neighbors (KNN) - Implementation from Scratch

Both classifier and regressor with multiple distance metrics.
"""

import numpy as np
from typing import Literal, Optional, Callable
from collections import Counter
import matplotlib.pyplot as plt


class KNNBase:
    """
    Base class for KNN classifier and regressor.
    """
    
    def __init__(
        self, 
        n_neighbors: int = 5,
        weights: Literal['uniform', 'distance'] = 'uniform',
        metric: Literal['euclidean', 'manhattan', 'minkowski'] = 'euclidean',
        p: int = 2
    ):
        """
        Initialize KNN model.
        
        Args:
            n_neighbors: Number of neighbors to use
            weights: Weight function ('uniform' or 'distance')
            metric: Distance metric to use
            p: Power parameter for Minkowski metric
        """
        self.n_neighbors = n_neighbors
        self.weights = weights
        self.metric = metric
        self.p = p
        
        self.X_train = None
        self.y_train = None
    
    def _euclidean_distance(self, x1: np.ndarray, x2: np.ndarray) -> float:
        """Calculate Euclidean distance."""
        return np.sqrt(np.sum((x1 - x2) ** 2))
    
    def _manhattan_distance(self, x1: np.ndarray, x2: np.ndarray) -> float:
        """Calculate Manhattan distance."""
        return np.sum(np.abs(x1 - x2))
    
    def _minkowski_distance(self, x1: np.ndarray, x2: np.ndarray) -> float:
        """Calculate Minkowski distance."""
        return np.power(np.sum(np.abs(x1 - x2) ** self.p), 1/self.p)
    
    def _get_distance_function(self) -> Callable:
        """Get the distance function based on metric."""
        if self.metric == 'euclidean':
            return self._euclidean_distance
        elif self.metric == 'manhattan':
            return self._manhattan_distance
        elif self.metric == 'minkowski':
            return self._minkowski_distance
        else:
            raise ValueError(f"Unknown metric: {self.metric}")
    
    def fit(self, X: np.ndarray, y: np.ndarray) -> 'KNNBase':
        """
        Store training data (lazy learning).
        
        Args:
            X: Training features, shape (n_samples, n_features)
            y: Training labels, shape (n_samples,)
            
        Returns:
            self
        """
        self.X_train = np.array(X)
        self.y_train = np.array(y)
        return self
    
    def _get_neighbors(self, x: np.ndarray) -> tuple:
        """
        Get k nearest neighbors for a sample.
        
        Args:
            x: Sample to find neighbors for
            
        Returns:
            indices: Indices of k nearest neighbors
            distances: Distances to k nearest neighbors
        """
        distance_func = self._get_distance_function()
        
        # Calculate distances to all training samples
        distances = np.array([distance_func(x, x_train) for x_train in self.X_train])
        
        # Get indices of k nearest neighbors
        k_indices = np.argsort(distances)[:self.n_neighbors]
        k_distances = distances[k_indices]
        
        return k_indices, k_distances


class KNNClassifier(KNNBase):
    """
    K-Nearest Neighbors Classifier.
    """
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict class labels for samples.
        
        Args:
            X: Features to predict, shape (n_samples, n_features)
            
        Returns:
            Predicted labels, shape (n_samples,)
        """
        predictions = [self._predict_single(x) for x in X]
        return np.array(predictions)
    
    def _predict_single(self, x: np.ndarray):
        """Predict class label for a single sample."""
        k_indices, k_distances = self._get_neighbors(x)
        k_labels = self.y_train[k_indices]
        
        if self.weights == 'uniform':
            # Simple majority vote
            most_common = Counter(k_labels).most_common(1)
            return most_common[0][0]
        else:  # distance weighting
            # Weight by inverse distance (add small epsilon to avoid division by zero)
            weights = 1 / (k_distances + 1e-8)
            
            # Weighted vote
            unique_labels = np.unique(k_labels)
            weighted_votes = {}
            
            for label in unique_labels:
                mask = k_labels == label
                weighted_votes[label] = np.sum(weights[mask])
            
            return max(weighted_votes, key=weighted_votes.get)
    
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """
        Predict class probabilities for samples.
        
        Args:
            X: Features to predict, shape (n_samples, n_features)
            
        Returns:
            Class probabilities, shape (n_samples, n_classes)
        """
        probas = [self._predict_proba_single(x) for x in X]
        return np.array(probas)
    
    def _predict_proba_single(self, x: np.ndarray) -> np.ndarray:
        """Predict class probabilities for a single sample."""
        k_indices, k_distances = self._get_neighbors(x)
        k_labels = self.y_train[k_indices]
        
        classes = np.unique(self.y_train)
        n_classes = len(classes)
        probas = np.zeros(n_classes)
        
        if self.weights == 'uniform':
            # Simple frequency
            for i, c in enumerate(classes):
                probas[i] = np.sum(k_labels == c) / self.n_neighbors
        else:  # distance weighting
            weights = 1 / (k_distances + 1e-8)
            total_weight = np.sum(weights)
            
            for i, c in enumerate(classes):
                mask = k_labels == c
                probas[i] = np.sum(weights[mask]) / total_weight
        
        return probas
    
    def score(self, X: np.ndarray, y: np.ndarray) -> float:
        """
        Calculate accuracy score.
        
        Args:
            X: Test features
            y: True labels
            
        Returns:
            Accuracy score
        """
        predictions = self.predict(X)
        return np.mean(predictions == y)


class KNNRegressor(KNNBase):
    """
    K-Nearest Neighbors Regressor.
    """
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict target values for samples.
        
        Args:
            X: Features to predict, shape (n_samples, n_features)
            
        Returns:
            Predicted values, shape (n_samples,)
        """
        predictions = [self._predict_single(x) for x in X]
        return np.array(predictions)
    
    def _predict_single(self, x: np.ndarray) -> float:
        """Predict target value for a single sample."""
        k_indices, k_distances = self._get_neighbors(x)
        k_values = self.y_train[k_indices]
        
        if self.weights == 'uniform':
            # Simple average
            return np.mean(k_values)
        else:  # distance weighting
            # Weighted average by inverse distance
            weights = 1 / (k_distances + 1e-8)
            return np.sum(weights * k_values) / np.sum(weights)
    
    def score(self, X: np.ndarray, y: np.ndarray) -> float:
        """
        Calculate R² score.
        
        Args:
            X: Test features
            y: True values
            
        Returns:
            R² score
        """
        predictions = self.predict(X)
        ss_res = np.sum((y - predictions) ** 2)
        ss_tot = np.sum((y - np.mean(y)) ** 2)
        return 1 - (ss_res / ss_tot)


def choose_k_elbow_method(X_train, y_train, X_val, y_val, 
                          k_range=range(1, 31), 
                          task='classification'):
    """
    Choose optimal k using elbow method.
    
    Args:
        X_train, y_train: Training data
        X_val, y_val: Validation data
        k_range: Range of k values to try
        task: 'classification' or 'regression'
        
    Returns:
        k_values: List of k values tested
        scores: Corresponding scores
    """
    scores = []
    
    for k in k_range:
        if task == 'classification':
            model = KNNClassifier(n_neighbors=k)
        else:
            model = KNNRegressor(n_neighbors=k)
        
        model.fit(X_train, y_train)
        score = model.score(X_val, y_val)
        scores.append(score)
    
    return list(k_range), scores


def plot_k_selection(k_values, scores, task='classification'):
    """Plot scores vs k for optimal k selection."""
    fig, ax = plt.subplots(figsize=(10, 6))
    
    ax.plot(k_values, scores, marker='o', linewidth=2, markersize=8)
    
    # Mark best k
    best_k_idx = np.argmax(scores)
    best_k = k_values[best_k_idx]
    best_score = scores[best_k_idx]
    
    ax.plot(best_k, best_score, 'r*', markersize=20, label=f'Best k={best_k}')
    
    metric = 'Accuracy' if task == 'classification' else 'R²'
    ax.set_xlabel('Number of Neighbors (k)', fontsize=12)
    ax.set_ylabel(f'{metric} Score', fontsize=12)
    ax.set_title(f'KNN {task.capitalize()}: Choosing Optimal k', 
                fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=10)
    
    plt.tight_layout()
    return fig


if __name__ == "__main__":
    from sklearn.datasets import make_classification, make_regression
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import StandardScaler
    
    print("="*60)
    print("KNN CLASSIFIER DEMO")
    print("="*60)
    
    # Generate classification data
    X, y = make_classification(
        n_samples=200, 
        n_features=2, 
        n_informative=2,
        n_redundant=0,
        n_clusters_per_class=1,
        random_state=42
    )
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42
    )
    
    # Scale features (important for KNN!)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Train KNN classifier
    print("\nTraining KNN Classifier (k=5, uniform weights)...")
    knn_clf = KNNClassifier(n_neighbors=5, weights='uniform', metric='euclidean')
    knn_clf.fit(X_train_scaled, y_train)
    
    # Evaluate
    train_acc = knn_clf.score(X_train_scaled, y_train)
    test_acc = knn_clf.score(X_test_scaled, y_test)
    
    print(f"Training Accuracy: {train_acc:.4f}")
    print(f"Test Accuracy: {test_acc:.4f}")
    
    # Try distance weighting
    print("\nTraining KNN Classifier (k=5, distance weights)...")
    knn_clf_dist = KNNClassifier(n_neighbors=5, weights='distance', metric='euclidean')
    knn_clf_dist.fit(X_train_scaled, y_train)
    
    test_acc_dist = knn_clf_dist.score(X_test_scaled, y_test)
    print(f"Test Accuracy (distance weights): {test_acc_dist:.4f}")
    
    # Choose optimal k
    print("\nFinding optimal k...")
    k_values, scores = choose_k_elbow_method(
        X_train_scaled, y_train, 
        X_test_scaled, y_test,
        k_range=range(1, 21),
        task='classification'
    )
    
    best_k = k_values[np.argmax(scores)]
    print(f"Optimal k: {best_k} (Accuracy: {max(scores):.4f})")
    
    print("\n" + "="*60)
    print("KNN REGRESSOR DEMO")
    print("="*60)
    
    # Generate regression data
    X_reg, y_reg = make_regression(
        n_samples=200,
        n_features=1,
        noise=10,
        random_state=42
    )
    
    # Split data
    X_train_reg, X_test_reg, y_train_reg, y_test_reg = train_test_split(
        X_reg, y_reg, test_size=0.3, random_state=42
    )
    
    # Scale features
    X_train_reg_scaled = scaler.fit_transform(X_train_reg)
    X_test_reg_scaled = scaler.transform(X_test_reg)
    
    # Train KNN regressor
    print("\nTraining KNN Regressor (k=5, uniform weights)...")
    knn_reg = KNNRegressor(n_neighbors=5, weights='uniform', metric='euclidean')
    knn_reg.fit(X_train_reg_scaled, y_train_reg)
    
    # Evaluate
    train_r2 = knn_reg.score(X_train_reg_scaled, y_train_reg)
    test_r2 = knn_reg.score(X_test_reg_scaled, y_test_reg)
    
    print(f"Training R²: {train_r2:.4f}")
    print(f"Test R²: {test_r2:.4f}")
    
    # Try distance weighting
    print("\nTraining KNN Regressor (k=5, distance weights)...")
    knn_reg_dist = KNNRegressor(n_neighbors=5, weights='distance', metric='euclidean')
    knn_reg_dist.fit(X_train_reg_scaled, y_train_reg)
    
    test_r2_dist = knn_reg_dist.score(X_test_reg_scaled, y_test_reg)
    print(f"Test R² (distance weights): {test_r2_dist:.4f}")
    
    print("\n" + "="*60)
    print("DISTANCE METRICS COMPARISON")
    print("="*60)
    
    metrics = ['euclidean', 'manhattan', 'minkowski']
    print("\nClassification Accuracy by Metric:")
    for metric in metrics:
        knn = KNNClassifier(n_neighbors=5, metric=metric)
        knn.fit(X_train_scaled, y_train)
        acc = knn.score(X_test_scaled, y_test)
        print(f"  {metric.capitalize()}: {acc:.4f}")
    
    print("\n✓ KNN demonstration completed!")
