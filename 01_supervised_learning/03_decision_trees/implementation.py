"""
Decision Tree Implementation from Scratch

This module provides a complete implementation of Decision Trees
for both classification and regression without using scikit-learn.

Author: ML Learning Repository
Date: 2024
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from collections import Counter
from typing import List, Tuple, Optional, Union
import warnings
warnings.filterwarnings('ignore')


class Node:
    """Node class for the decision tree"""
    
    def __init__(self, feature=None, threshold=None, left=None, right=None, 
                 value=None, impurity=None):
        self.feature = feature          # Feature index to split on
        self.threshold = threshold      # Threshold value for split
        self.left = left               # Left child node
        self.right = right             # Right child node
        self.value = value             # Prediction value (for leaf nodes)
        self.impurity = impurity       # Impurity measure at this node
        
    def is_leaf_node(self):
        """Check if node is a leaf"""
        return self.value is not None


class DecisionTreeScratch:
    """
    Decision Tree implementation from scratch
    
    Parameters:
    -----------
    task_type : str, default='classification'
        Type of task: 'classification' or 'regression'
    criterion : str, default='gini'
        Split criterion: 'gini', 'entropy' for classification; 'mse' for regression
    max_depth : int, default=10
        Maximum depth of the tree
    min_samples_split : int, default=2
        Minimum number of samples required to split an internal node
    min_samples_leaf : int, default=1
        Minimum number of samples required in a leaf node
    min_impurity_decrease : float, default=0.0
        Minimum impurity decrease required for a split
    random_state : int, default=None
        Random seed for reproducibility
    """
    
    def __init__(self, task_type='classification', criterion='gini', 
                 max_depth=10, min_samples_split=2, min_samples_leaf=1,
                 min_impurity_decrease=0.0, random_state=None):
        self.task_type = task_type
        self.criterion = criterion
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.min_impurity_decrease = min_impurity_decrease
        self.random_state = random_state
        
        # Will be set during training
        self.root = None
        self.feature_importances_ = None
        self.n_features_ = None
        self.tree_depth_ = 0
        
        if random_state is not None:
            np.random.seed(random_state)
    
    def _calculate_impurity(self, y: np.ndarray) -> float:
        """Calculate impurity measure based on criterion"""
        if len(y) == 0:
            return 0
            
        if self.task_type == 'classification':
            if self.criterion == 'gini':
                return self._gini_impurity(y)
            elif self.criterion == 'entropy':
                return self._entropy(y)
        else:  # regression
            if self.criterion == 'mse':
                return self._mse(y)
        
        raise ValueError(f"Unknown criterion: {self.criterion}")
    
    def _gini_impurity(self, y: np.ndarray) -> float:
        """Calculate Gini impurity"""
        proportions = np.bincount(y) / len(y)
        return 1 - np.sum(proportions ** 2)
    
    def _entropy(self, y: np.ndarray) -> float:
        """Calculate entropy"""
        proportions = np.bincount(y) / len(y)
        proportions = proportions[proportions > 0]  # Remove zero proportions
        return -np.sum(proportions * np.log2(proportions))
    
    def _mse(self, y: np.ndarray) -> float:
        """Calculate mean squared error (variance)"""
        if len(y) == 0:
            return 0
        mean_y = np.mean(y)
        return np.mean((y - mean_y) ** 2)
    
    def _information_gain(self, y: np.ndarray, y_left: np.ndarray, y_right: np.ndarray) -> float:
        """Calculate information gain from a split"""
        n = len(y)
        n_left, n_right = len(y_left), len(y_right)
        
        if n_left == 0 or n_right == 0:
            return 0
        
        # Calculate weighted impurity after split
        impurity_parent = self._calculate_impurity(y)
        impurity_left = self._calculate_impurity(y_left)
        impurity_right = self._calculate_impurity(y_right)
        
        weighted_impurity = (n_left / n) * impurity_left + (n_right / n) * impurity_right
        
        return impurity_parent - weighted_impurity
    
    def _best_split(self, X: np.ndarray, y: np.ndarray) -> Tuple[int, float, float]:
        """Find the best split for the given data"""
        m, n = X.shape
        
        # If all samples have the same target value, no split needed
        if len(np.unique(y)) <= 1:
            return None, None, 0
        
        best_gain = 0
        best_feature = None
        best_threshold = None
        
        # Try each feature
        for feature in range(n):
            feature_values = X[:, feature]
            thresholds = np.unique(feature_values)
            
            # Try each unique value as threshold
            for threshold in thresholds:
                # Split data based on threshold
                left_mask = feature_values <= threshold
                right_mask = ~left_mask
                
                y_left, y_right = y[left_mask], y[right_mask]
                
                # Skip if split results in empty child
                if len(y_left) < self.min_samples_leaf or len(y_right) < self.min_samples_leaf:
                    continue
                
                # Calculate information gain
                gain = self._information_gain(y, y_left, y_right)
                
                # Update best split if this is better
                if gain > best_gain:
                    best_gain = gain
                    best_feature = feature
                    best_threshold = threshold
        
        return best_feature, best_threshold, best_gain
    
    def _leaf_value(self, y: np.ndarray):
        """Calculate the value for a leaf node"""
        if self.task_type == 'classification':
            # Return most common class
            return Counter(y).most_common(1)[0][0]
        else:  # regression
            # Return mean value
            return np.mean(y)
    
    def _build_tree(self, X: np.ndarray, y: np.ndarray, depth: int = 0) -> Node:
        """Recursively build the decision tree"""
        n_samples, n_features = X.shape
        n_classes = len(np.unique(y))
        
        # Calculate impurity at current node
        current_impurity = self._calculate_impurity(y)
        
        # Stopping criteria
        if (depth >= self.max_depth or 
            n_samples < self.min_samples_split or
            n_classes == 1 or
            n_samples < 2 * self.min_samples_leaf):
            
            leaf_value = self._leaf_value(y)
            return Node(value=leaf_value, impurity=current_impurity)
        
        # Find the best split
        best_feature, best_threshold, best_gain = self._best_split(X, y)
        
        # If no beneficial split found, create leaf
        if best_gain < self.min_impurity_decrease or best_feature is None:
            leaf_value = self._leaf_value(y)
            return Node(value=leaf_value, impurity=current_impurity)
        
        # Create split
        left_mask = X[:, best_feature] <= best_threshold
        right_mask = ~left_mask
        
        # Recursively build left and right subtrees
        left_child = self._build_tree(X[left_mask], y[left_mask], depth + 1)
        right_child = self._build_tree(X[right_mask], y[right_mask], depth + 1)
        
        # Update tree depth
        self.tree_depth_ = max(self.tree_depth_, depth + 1)
        
        return Node(feature=best_feature, threshold=best_threshold,
                   left=left_child, right=right_child, impurity=current_impurity)
    
    def fit(self, X: np.ndarray, y: np.ndarray):
        """Train the decision tree"""
        X = np.array(X)
        y = np.array(y)
        
        self.n_features_ = X.shape[1]
        
        # Build the tree
        self.root = self._build_tree(X, y)
        
        # Calculate feature importances
        self._calculate_feature_importances(X, y)
        
        return self
    
    def _calculate_feature_importances(self, X: np.ndarray, y: np.ndarray):
        """Calculate feature importances based on impurity decrease"""
        importances = np.zeros(self.n_features_)
        
        def _calculate_importances(node: Node, X_subset: np.ndarray, y_subset: np.ndarray):
            if node.is_leaf_node():
                return
            
            # Calculate weighted impurity decrease for this split
            feature = node.feature
            threshold = node.threshold
            
            left_mask = X_subset[:, feature] <= threshold
            right_mask = ~left_mask
            
            n = len(y_subset)
            n_left, n_right = np.sum(left_mask), np.sum(right_mask)
            
            if n_left > 0 and n_right > 0:
                impurity_parent = self._calculate_impurity(y_subset)
                impurity_left = self._calculate_impurity(y_subset[left_mask])
                impurity_right = self._calculate_impurity(y_subset[right_mask])
                
                weighted_impurity = (n_left / n) * impurity_left + (n_right / n) * impurity_right
                importance = n * (impurity_parent - weighted_impurity)
                
                importances[feature] += importance
                
                # Recursively calculate for children
                _calculate_importances(node.left, X_subset[left_mask], y_subset[left_mask])
                _calculate_importances(node.right, X_subset[right_mask], y_subset[right_mask])
        
        _calculate_importances(self.root, X, y)
        
        # Normalize importances
        if importances.sum() > 0:
            importances = importances / importances.sum()
        
        self.feature_importances_ = importances
    
    def _predict_sample(self, x: np.ndarray) -> Union[int, float]:
        """Predict a single sample"""
        node = self.root
        
        while not node.is_leaf_node():
            if x[node.feature] <= node.threshold:
                node = node.left
            else:
                node = node.right
        
        return node.value
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Make predictions for given samples"""
        X = np.array(X)
        return np.array([self._predict_sample(x) for x in X])
    
    def score(self, X: np.ndarray, y: np.ndarray) -> float:
        """Calculate accuracy (classification) or R² score (regression)"""
        predictions = self.predict(X)
        
        if self.task_type == 'classification':
            return np.mean(predictions == y)
        else:  # regression
            ss_res = np.sum((y - predictions) ** 2)
            ss_tot = np.sum((y - np.mean(y)) ** 2)
            return 1 - (ss_res / ss_tot) if ss_tot != 0 else 0
    
    def print_tree(self, node: Optional[Node] = None, depth: int = 0):
        """Print the tree structure"""
        if node is None:
            node = self.root
        
        if node.is_leaf_node():
            print("  " * depth + f"Predict: {node.value}")
        else:
            print("  " * depth + f"Feature {node.feature} <= {node.threshold:.3f}")
            print("  " * depth + "├─ True:")
            self.print_tree(node.left, depth + 1)
            print("  " * depth + "└─ False:")
            self.print_tree(node.right, depth + 1)
    
    def get_depth(self) -> int:
        """Get the depth of the tree"""
        return self.tree_depth_
    
    def get_n_leaves(self) -> int:
        """Get the number of leaves in the tree"""
        def _count_leaves(node: Node) -> int:
            if node.is_leaf_node():
                return 1
            return _count_leaves(node.left) + _count_leaves(node.right)
        
        return _count_leaves(self.root) if self.root else 0


def visualize_tree_decision_boundary(X: np.ndarray, y: np.ndarray, model: DecisionTreeScratch,
                                   title: str = "Decision Tree Decision Boundary"):
    """Visualize decision boundary for 2D data"""
    if X.shape[1] != 2:
        print("Can only plot decision boundary for 2D data")
        return
    
    # Create mesh
    h = 0.02
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                         np.arange(y_min, y_max, h))
    
    # Make predictions on mesh
    mesh_points = np.c_[xx.ravel(), yy.ravel()]
    Z = model.predict(mesh_points)
    Z = Z.reshape(xx.shape)
    
    # Plot
    plt.figure(figsize=(12, 8))
    plt.contourf(xx, yy, Z, alpha=0.6, cmap='viridis')
    
    # Plot data points
    scatter = plt.scatter(X[:, 0], X[:, 1], c=y, cmap='viridis', edgecolors='black')
    plt.colorbar(scatter, label='Class')
    
    plt.xlabel('Feature 1', fontsize=12)
    plt.ylabel('Feature 2', fontsize=12)
    plt.title(title, fontsize=14)
    plt.grid(True, alpha=0.3)
    plt.show()


def generate_classification_data(n_samples=1000, random_state=42):
    """Generate sample classification data"""
    np.random.seed(random_state)
    
    # Create two clusters with some overlap
    X1 = np.random.multivariate_normal([2, 2], [[1, 0.5], [0.5, 1]], n_samples//2)
    X2 = np.random.multivariate_normal([-1, -1], [[1, -0.3], [-0.3, 1]], n_samples//2)
    
    X = np.vstack((X1, X2))
    y = np.hstack((np.ones(n_samples//2), np.zeros(n_samples//2)))
    
    # Shuffle data
    indices = np.random.permutation(n_samples)
    return X[indices], y[indices].astype(int)


def generate_regression_data(n_samples=1000, noise=0.1, random_state=42):
    """Generate sample regression data"""
    np.random.seed(random_state)
    
    X = np.random.uniform(-3, 3, (n_samples, 2))
    
    # Create non-linear relationship
    y = (X[:, 0] ** 2 + X[:, 1] ** 2 + 
         np.sin(X[:, 0]) * np.cos(X[:, 1]) +
         np.random.normal(0, noise, n_samples))
    
    return X, y


def demonstrate_decision_tree():
    """Demonstrate the decision tree implementation"""
    print("=" * 70)
    print("DECISION TREE FROM SCRATCH DEMONSTRATION")
    print("=" * 70)
    
    # Classification Example
    print("\n1. CLASSIFICATION EXAMPLE")
    print("-" * 40)
    
    # Generate classification data
    X_class, y_class = generate_classification_data(n_samples=500)
    
    # Split data
    from sklearn.model_selection import train_test_split
    X_train_class, X_test_class, y_train_class, y_test_class = train_test_split(
        X_class, y_class, test_size=0.2, random_state=42
    )
    
    print(f"Classification dataset: {X_class.shape[0]} samples, {X_class.shape[1]} features")
    print(f"Class distribution: {Counter(y_class)}")
    
    # Train decision tree classifier
    dt_classifier = DecisionTreeScratch(
        task_type='classification',
        criterion='gini',
        max_depth=5,
        min_samples_split=10,
        min_samples_leaf=5,
        random_state=42
    )
    
    dt_classifier.fit(X_train_class, y_train_class)
    
    # Evaluate
    train_accuracy = dt_classifier.score(X_train_class, y_train_class)
    test_accuracy = dt_classifier.score(X_test_class, y_test_class)
    
    print(f"Training accuracy: {train_accuracy:.4f}")
    print(f"Test accuracy: {test_accuracy:.4f}")
    print(f"Tree depth: {dt_classifier.get_depth()}")
    print(f"Number of leaves: {dt_classifier.get_n_leaves()}")
    
    # Feature importance
    print(f"\nFeature importances:")
    for i, importance in enumerate(dt_classifier.feature_importances_):
        print(f"Feature {i}: {importance:.4f}")
    
    # Visualize decision boundary
    visualize_tree_decision_boundary(X_test_class, y_test_class, dt_classifier,
                                   "Decision Tree Classifier")
    
    # Print tree structure (first few levels)
    print("\nTree structure (first 3 levels):")
    dt_small = DecisionTreeScratch(task_type='classification', max_depth=3, random_state=42)
    dt_small.fit(X_train_class, y_train_class)
    dt_small.print_tree()
    
    # Regression Example
    print("\n2. REGRESSION EXAMPLE")
    print("-" * 40)
    
    # Generate regression data
    X_reg, y_reg = generate_regression_data(n_samples=500, noise=0.5)
    
    X_train_reg, X_test_reg, y_train_reg, y_test_reg = train_test_split(
        X_reg, y_reg, test_size=0.2, random_state=42
    )
    
    print(f"Regression dataset: {X_reg.shape[0]} samples, {X_reg.shape[1]} features")
    print(f"Target range: [{y_reg.min():.2f}, {y_reg.max():.2f}]")
    
    # Train decision tree regressor
    dt_regressor = DecisionTreeScratch(
        task_type='regression',
        criterion='mse',
        max_depth=5,
        min_samples_split=10,
        min_samples_leaf=5,
        random_state=42
    )
    
    dt_regressor.fit(X_train_reg, y_train_reg)
    
    # Evaluate
    train_r2 = dt_regressor.score(X_train_reg, y_train_reg)
    test_r2 = dt_regressor.score(X_test_reg, y_test_reg)
    
    # Calculate MSE
    predictions = dt_regressor.predict(X_test_reg)
    mse = np.mean((y_test_reg - predictions) ** 2)
    
    print(f"Training R²: {train_r2:.4f}")
    print(f"Test R²: {test_r2:.4f}")
    print(f"Test MSE: {mse:.4f}")
    print(f"Tree depth: {dt_regressor.get_depth()}")
    print(f"Number of leaves: {dt_regressor.get_n_leaves()}")
    
    # Compare with sklearn
    print("\n3. COMPARISON WITH SCIKIT-LEARN")
    print("-" * 40)
    
    from sklearn.tree import DecisionTreeClassifier
    from sklearn.metrics import accuracy_score
    
    sklearn_dt = DecisionTreeClassifier(
        criterion='gini',
        max_depth=5,
        min_samples_split=10,
        min_samples_leaf=5,
        random_state=42
    )
    
    sklearn_dt.fit(X_train_class, y_train_class)
    sklearn_pred = sklearn_dt.predict(X_test_class)
    sklearn_accuracy = accuracy_score(y_test_class, sklearn_pred)
    
    print(f"Our implementation: {test_accuracy:.4f}")
    print(f"Scikit-learn:      {sklearn_accuracy:.4f}")
    print(f"Difference:        {abs(test_accuracy - sklearn_accuracy):.4f}")
    
    # Feature importance comparison
    print(f"\nFeature importance comparison:")
    print(f"Our implementation: {dt_classifier.feature_importances_}")
    print(f"Scikit-learn:       {sklearn_dt.feature_importances_}")
    
    return dt_classifier, dt_regressor


if __name__ == "__main__":
    # Run demonstration
    classifier, regressor = demonstrate_decision_tree()
    
    print("\n" + "=" * 70)
    print("Try experimenting with:")
    print("- Different max_depth values")
    print("- Different splitting criteria")
    print("- Different min_samples_split values")
    print("- Your own datasets")
    print("=" * 70)