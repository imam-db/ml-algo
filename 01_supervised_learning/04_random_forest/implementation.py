"""
Random Forest Implementation from Scratch
=========================================

This module implements Random Forest algorithm from scratch, including:
- Bootstrap sampling
- Random feature selection
- Decision tree ensemble
- Out-of-Bag (OOB) error estimation
- Feature importance calculation

Author: ML Learning Project
Date: 2024
"""

from collections import Counter
from math import log2, sqrt

import numpy as np


class DecisionTreeNode:
    """Node class for decision tree used in Random Forest"""
    
    def __init__(self):
        self.feature_index = None      # Index of feature used for splitting
        self.threshold = None          # Threshold value for splitting
        self.left = None              # Left child node
        self.right = None             # Right child node
        self.value = None             # Prediction value for leaf nodes
        self.samples = None           # Number of samples in this node
        self.impurity = None          # Impurity measure of this node

class DecisionTreeRF:
    """Decision Tree implementation optimized for Random Forest"""
    
    def __init__(self, 
                 max_depth: int | None = None,
                 min_samples_split: int = 2,
                 min_samples_leaf: int = 1,
                 max_features: int | str | None = None,
                 criterion: str = 'gini',
                 random_state: int | None = None):
        
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.max_features = max_features
        self.criterion = criterion
        self.random_state = random_state
        self.root = None
        self.feature_importances_ = None
        self.n_features_ = None
        
        # Use is not None to avoid treating random_state=0 as unset
        if random_state is not None:
            self.rng = np.random.default_rng(random_state)
        else:
            self.rng = np.random.default_rng()
    
    def _calculate_impurity(self, y: np.ndarray) -> float:
        """Calculate impurity using specified criterion"""
        if len(y) == 0:
            return 0
        
        if self.criterion == 'gini':
            return self._gini_impurity(y)
        elif self.criterion == 'entropy':
            return self._entropy(y)
        else:
            raise ValueError(f"Unsupported criterion: {self.criterion}")
    
    def _gini_impurity(self, y: np.ndarray) -> float:
        """Calculate Gini impurity"""
        if len(y) == 0:
            return 0
        
        _, counts = np.unique(y, return_counts=True)
        probabilities = counts / len(y)
        gini = 1 - np.sum(probabilities ** 2)
        return gini
    
    def _entropy(self, y: np.ndarray) -> float:
        """Calculate entropy"""
        if len(y) == 0:
            return 0
        
        _, counts = np.unique(y, return_counts=True)
        probabilities = counts / len(y)
        entropy = -np.sum(probabilities * np.log2(probabilities + 1e-10))
        return entropy
    
    def _get_random_features(self, n_features: int) -> np.ndarray:
        """Get random subset of features for splitting"""
        if self.max_features is None:
            max_features = n_features
        elif self.max_features == 'sqrt':
            max_features = int(sqrt(n_features))
        elif self.max_features == 'log2':
            max_features = int(log2(n_features))
        elif isinstance(self.max_features, int):
            max_features = min(self.max_features, n_features)
        else:
            max_features = n_features
        
        # Ensure at least one feature
        max_features = max(1, max_features)
        
        # Return random subset of feature indices using instance RNG
        return self.rng.choice(n_features, size=max_features, replace=False)
    
    def _find_best_split(self, X: np.ndarray, y: np.ndarray, feature_indices: np.ndarray) -> tuple[int, float, float]:
        """Find the best split among random features"""
        best_feature = None
        best_threshold = None
        best_gain = -1
        
        current_impurity = self._calculate_impurity(y)
        
        for feature_idx in feature_indices:
            feature_values = X[:, feature_idx]
            unique_values = np.unique(feature_values)
            
            # Try different thresholds
            for i in range(len(unique_values) - 1):
                threshold = (unique_values[i] + unique_values[i + 1]) / 2
                
                # Split data
                left_mask = feature_values <= threshold
                right_mask = ~left_mask
                
                left_y = y[left_mask]
                right_y = y[right_mask]
                
                # Skip if split doesn't meet minimum samples requirement
                if len(left_y) < self.min_samples_leaf or len(right_y) < self.min_samples_leaf:
                    continue
                
                # Calculate weighted impurity after split
                left_impurity = self._calculate_impurity(left_y)
                right_impurity = self._calculate_impurity(right_y)
                
                weighted_impurity = (len(left_y) * left_impurity + len(right_y) * right_impurity) / len(y)
                
                # Calculate information gain
                gain = current_impurity - weighted_impurity
                
                if gain > best_gain:
                    best_gain = gain
                    best_feature = feature_idx
                    best_threshold = threshold
        
        return best_feature, best_threshold, best_gain
    
    def _build_tree(self, X: np.ndarray, y: np.ndarray, depth: int = 0) -> DecisionTreeNode:
        """Recursively build decision tree"""
        node = DecisionTreeNode()
        node.samples = len(y)
        node.impurity = self._calculate_impurity(y)
        
        # Check stopping criteria
        if (self.max_depth is not None and depth >= self.max_depth) or \
           (len(y) < self.min_samples_split) or \
           (len(np.unique(y)) == 1):
            # Create leaf node
            if len(y) > 0:
                node.value = Counter(y).most_common(1)[0][0]  # Most common class
            return node
        
        # Get random features for this split
        feature_indices = self._get_random_features(X.shape[1])
        
        # Find best split
        best_feature, best_threshold, best_gain = self._find_best_split(X, y, feature_indices)
        
        # If no good split found, create leaf node
        if best_feature is None or best_gain <= 0:
            node.value = Counter(y).most_common(1)[0][0]
            return node
        
        # Store split information
        node.feature_index = best_feature
        node.threshold = best_threshold
        
        # Split data
        left_mask = X[:, best_feature] <= best_threshold
        right_mask = ~left_mask
        
        # Recursively build child nodes
        node.left = self._build_tree(X[left_mask], y[left_mask], depth + 1)
        node.right = self._build_tree(X[right_mask], y[right_mask], depth + 1)
        
        return node
    
    def fit(self, X: np.ndarray, y: np.ndarray):
        """Train the decision tree"""
        X = np.array(X)
        y = np.array(y)
        
        self.n_features_ = X.shape[1]
        self.feature_importances_ = np.zeros(self.n_features_)
        self.root = self._build_tree(X, y)
        
        # Calculate feature importances based on weighted impurity decrease
        self._calculate_feature_importances(X, y)
        
        return self
    
    def _calculate_feature_importances(self, X: np.ndarray, y: np.ndarray):
        """Calculate feature importances based on impurity decrease"""
        importances = np.zeros(self.n_features_)
        
        def _traverse_and_accumulate(node, X_subset, y_subset):
            if node.value is not None:  # Leaf node
                return
            
            # Calculate weighted impurity decrease for this split
            feature = node.feature_index
            threshold = node.threshold
            
            left_mask = X_subset[:, feature] <= threshold
            right_mask = ~left_mask
            
            n_samples = len(y_subset)
            n_left = np.sum(left_mask)
            n_right = np.sum(right_mask)
            
            if n_left > 0 and n_right > 0:
                # Weighted impurity decrease
                impurity_parent = self._calculate_impurity(y_subset)
                impurity_left = self._calculate_impurity(y_subset[left_mask])
                impurity_right = self._calculate_impurity(y_subset[right_mask])
                
                weighted_impurity = (n_left / n_samples) * impurity_left + (n_right / n_samples) * impurity_right
                importance = n_samples * (impurity_parent - weighted_impurity)
                
                importances[feature] += importance
                
                # Recurse on children
                _traverse_and_accumulate(node.left, X_subset[left_mask], y_subset[left_mask])
                _traverse_and_accumulate(node.right, X_subset[right_mask], y_subset[right_mask])
        
        _traverse_and_accumulate(self.root, X, y)
        
        # Normalize importances
        if importances.sum() > 0:
            importances = importances / importances.sum()
        
        self.feature_importances_ = importances
    
    def _predict_sample(self, x: np.ndarray, node: DecisionTreeNode):
        """Predict single sample by traversing tree"""
        if node.value is not None:  # Leaf node
            return node.value
        
        if x[node.feature_index] <= node.threshold:
            return self._predict_sample(x, node.left)
        else:
            return self._predict_sample(x, node.right)
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Make predictions for multiple samples"""
        X = np.array(X)
        predictions = []
        
        for x in X:
            prediction = self._predict_sample(x, self.root)
            predictions.append(prediction)
        
        return np.array(predictions)


class DecisionTreeRegRF:
    """Decision Tree Regressor implementation optimized for Random Forest"""
    
    def __init__(self, 
                 max_depth: int | None = None,
                 min_samples_split: int = 2,
                 min_samples_leaf: int = 1,
                 max_features: str | float | None = None,
                 criterion: str = 'mse',
                 random_state: int | None = None):
        
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.max_features = max_features
        self.criterion = criterion
        self.random_state = random_state
        self.root = None
        self.feature_importances_ = None
        self.n_features_ = None
        self.is_classifier = False  # For regression trees
        
        # Use is not None to avoid treating random_state=0 as unset
        if random_state is not None:
            self.rng = np.random.default_rng(random_state)
        else:
            self.rng = np.random.default_rng()
    
    def _calculate_mse(self, y: np.ndarray) -> float:
        """Calculate mean squared error (variance)"""
        if len(y) == 0:
            return 0
        mean_y = np.mean(y)
        return np.mean((y - mean_y) ** 2)
    
    def _get_random_features(self, n_features: int) -> np.ndarray:
        """Get random subset of features for splitting"""
        if self.max_features is None or self.max_features == 1.0:
            max_features = n_features
        elif self.max_features == 'sqrt':
            max_features = int(sqrt(n_features))
        elif self.max_features == 'log2':
            max_features = int(log2(n_features))
        elif isinstance(self.max_features, float):
            max_features = max(1, int(self.max_features * n_features))
        elif isinstance(self.max_features, int):
            max_features = min(self.max_features, n_features)
        else:
            max_features = n_features
        
        # Ensure at least one feature
        max_features = max(1, max_features)
        
        # Return random subset of feature indices using instance RNG
        return self.rng.choice(n_features, size=max_features, replace=False)
    
    def _find_best_split(self, X: np.ndarray, y: np.ndarray, feature_indices: np.ndarray) -> tuple[int, float, float]:
        """Find the best split among random features for regression"""
        best_feature = None
        best_threshold = None
        best_gain = -1
        
        current_mse = self._calculate_mse(y)
        
        for feature_idx in feature_indices:
            feature_values = X[:, feature_idx]
            unique_values = np.unique(feature_values)
            
            # Try midpoint thresholds
            for i in range(len(unique_values) - 1):
                threshold = (unique_values[i] + unique_values[i + 1]) / 2
                
                # Split data
                left_mask = feature_values <= threshold
                right_mask = ~left_mask
                
                left_y = y[left_mask]
                right_y = y[right_mask]
                
                # Skip if split doesn't meet minimum samples requirement
                if len(left_y) < self.min_samples_leaf or len(right_y) < self.min_samples_leaf:
                    continue
                
                # Calculate weighted MSE after split
                left_mse = self._calculate_mse(left_y)
                right_mse = self._calculate_mse(right_y)
                
                weighted_mse = (len(left_y) * left_mse + len(right_y) * right_mse) / len(y)
                
                # Calculate MSE reduction
                gain = current_mse - weighted_mse
                
                if gain > best_gain:
                    best_gain = gain
                    best_feature = feature_idx
                    best_threshold = threshold
        
        return best_feature, best_threshold, best_gain
    
    def _build_tree(self, X: np.ndarray, y: np.ndarray, depth: int = 0) -> DecisionTreeNode:
        """Recursively build regression tree"""
        node = DecisionTreeNode()
        node.samples = len(y)
        node.impurity = self._calculate_mse(y)
        
        # Check stopping criteria
        if (self.max_depth is not None and depth >= self.max_depth) or \
           (len(y) < self.min_samples_split) or \
           (node.impurity == 0):  # All values are the same
            # Create leaf node with mean value
            node.value = np.mean(y)
            return node
        
        # Get random features for this split
        feature_indices = self._get_random_features(X.shape[1])
        
        # Find best split
        best_feature, best_threshold, best_gain = self._find_best_split(X, y, feature_indices)
        
        # If no good split found, create leaf node
        if best_feature is None or best_gain <= 0:
            node.value = np.mean(y)
            return node
        
        # Store split information
        node.feature_index = best_feature
        node.threshold = best_threshold
        
        # Split data
        left_mask = X[:, best_feature] <= best_threshold
        right_mask = ~left_mask
        
        # Recursively build child nodes
        node.left = self._build_tree(X[left_mask], y[left_mask], depth + 1)
        node.right = self._build_tree(X[right_mask], y[right_mask], depth + 1)
        
        return node
    
    def fit(self, X: np.ndarray, y: np.ndarray):
        """Train the regression tree"""
        X = np.array(X)
        y = np.array(y)
        
        self.n_features_ = X.shape[1]
        self.feature_importances_ = np.zeros(self.n_features_)
        self.root = self._build_tree(X, y)
        
        # Calculate feature importances based on weighted MSE decrease
        self._calculate_feature_importances(X, y)
        
        return self
    
    def _calculate_feature_importances(self, X: np.ndarray, y: np.ndarray):
        """Calculate feature importances based on MSE decrease"""
        importances = np.zeros(self.n_features_)
        
        def _traverse_and_accumulate(node, X_subset, y_subset):
            if node.value is not None:  # Leaf node
                return
            
            # Calculate weighted MSE decrease for this split
            feature = node.feature_index
            threshold = node.threshold
            
            left_mask = X_subset[:, feature] <= threshold
            right_mask = ~left_mask
            
            n_samples = len(y_subset)
            n_left = np.sum(left_mask)
            n_right = np.sum(right_mask)
            
            if n_left > 0 and n_right > 0:
                # Weighted MSE decrease
                mse_parent = self._calculate_mse(y_subset)
                mse_left = self._calculate_mse(y_subset[left_mask])
                mse_right = self._calculate_mse(y_subset[right_mask])
                
                weighted_mse = (n_left / n_samples) * mse_left + (n_right / n_samples) * mse_right
                importance = n_samples * (mse_parent - weighted_mse)
                
                importances[feature] += importance
                
                # Recurse on children
                _traverse_and_accumulate(node.left, X_subset[left_mask], y_subset[left_mask])
                _traverse_and_accumulate(node.right, X_subset[right_mask], y_subset[right_mask])
        
        _traverse_and_accumulate(self.root, X, y)
        
        # Normalize importances
        if importances.sum() > 0:
            importances = importances / importances.sum()
        
        self.feature_importances_ = importances
    
    def _predict_sample(self, x: np.ndarray, node: DecisionTreeNode):
        """Predict single sample by traversing tree"""
        if node.value is not None:  # Leaf node
            return node.value
        
        if x[node.feature_index] <= node.threshold:
            return self._predict_sample(x, node.left)
        else:
            return self._predict_sample(x, node.right)
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Make predictions for multiple samples"""
        X = np.array(X)
        predictions = []
        
        for x in X:
            prediction = self._predict_sample(x, self.root)
            predictions.append(prediction)
        
        return np.array(predictions)


class RandomForestClassifier:
    """Random Forest Classifier implementation from scratch"""
    
    def __init__(self,
                 n_estimators: int = 100,
                 criterion: str = 'gini',
                 max_depth: int | None = None,
                 min_samples_split: int = 2,
                 min_samples_leaf: int = 1,
                 max_features: str = 'sqrt',
                 bootstrap: bool = True,
                 oob_score: bool = False,
                 n_jobs: int | None = None,
                 random_state: int | None = None,
                 verbose: int = 0):
        
        self.n_estimators = n_estimators
        self.criterion = criterion
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.max_features = max_features
        self.bootstrap = bootstrap
        self.oob_score = oob_score
        self.n_jobs = n_jobs
        self.random_state = random_state
        self.verbose = verbose
        
        # Initialize containers
        self.trees = []
        self.feature_importances_ = None
        self.oob_score_ = None
        self.oob_decision_function_ = None
        self.classes_ = None
        self.n_classes_ = None
        self.n_features_ = None
        self.n_outputs_ = 1
        
        # Set random seed using is not None to avoid treating random_state=0 as unset
        if random_state is not None:
            self.rng = np.random.default_rng(random_state)
        else:
            self.rng = np.random.default_rng()
        
        # Store bootstrap indices for OOB calculation
        self.bootstrap_indices = []
        self.oob_indices = []
    
    def _bootstrap_sample(self, X: np.ndarray, y: np.ndarray, sample_idx: int) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Create bootstrap sample of the data"""
        n_samples = X.shape[0]
        
        if self.bootstrap:
            # Sample with replacement using per-tree RNG for reproducibility
            if self.random_state is not None:
                tree_rng = np.random.default_rng(self.random_state + sample_idx)
            else:
                tree_rng = self.rng
            
            bootstrap_idx = tree_rng.choice(n_samples, size=n_samples, replace=True)
            oob_idx = np.setdiff1d(np.arange(n_samples), bootstrap_idx)
        else:
            # Use entire dataset
            bootstrap_idx = np.arange(n_samples)
            oob_idx = np.array([])
        
        X_bootstrap = X[bootstrap_idx]
        y_bootstrap = y[bootstrap_idx]
        
        return X_bootstrap, y_bootstrap, bootstrap_idx, oob_idx
    
    def _calculate_oob_score(self, X: np.ndarray, y: np.ndarray):
        """Calculate Out-of-Bag score using zero-initialized vote counts"""
        if not self.bootstrap or not self.oob_score:
            return
        
        n_samples = X.shape[0]
        # Use zero-initialized vote counts instead of NaN to fix argmax issue
        oob_vote_counts = np.zeros((n_samples, self.n_classes_))
        oob_counts = np.zeros(n_samples)
        
        # Collect OOB predictions from all trees
        for tree_idx, (tree, oob_idx) in enumerate(zip(self.trees, self.oob_indices)):
            if len(oob_idx) > 0:
                # Get predictions for OOB samples
                oob_pred = tree.predict(X[oob_idx])
                
                # Accumulate votes per class
                for i, sample_idx in enumerate(oob_idx):
                    class_idx = np.where(self.classes_ == oob_pred[i])[0][0]
                    oob_vote_counts[sample_idx, class_idx] += 1
                    oob_counts[sample_idx] += 1
        
        # Calculate OOB score only for samples with at least one OOB vote
        valid_samples = oob_counts > 0
        if np.sum(valid_samples) > 0:
            # Get final predictions by majority vote
            final_predictions = np.argmax(oob_vote_counts[valid_samples], axis=1)
            final_predictions = self.classes_[final_predictions]
            
            # Calculate accuracy
            self.oob_score_ = np.mean(final_predictions == y[valid_samples])
            
            # Store normalized decision function
            self.oob_decision_function_ = oob_vote_counts.copy()
            self.oob_decision_function_[valid_samples] /= oob_counts[valid_samples].reshape(-1, 1)
        else:
            self.oob_score_ = 0.0
    
    def _calculate_feature_importance(self):
        """Calculate feature importance based on average impurity decrease across trees"""
        if not self.trees:
            return
        
        self.feature_importances_ = np.zeros(self.n_features_)
        
        # Accumulate feature importances from all trees
        for tree in self.trees:
            if hasattr(tree, 'feature_importances_') and tree.feature_importances_ is not None:
                self.feature_importances_ += tree.feature_importances_
        
        # Average across trees and normalize
        if len(self.trees) > 0:
            self.feature_importances_ /= len(self.trees)
        
        if np.sum(self.feature_importances_) > 0:
            self.feature_importances_ /= np.sum(self.feature_importances_)
    
    def fit(self, X: np.ndarray, y: np.ndarray):
        """Train the Random Forest"""
        # Convert to numpy arrays
        X = np.array(X)
        y = np.array(y)
        
        # Store basic information
        self.classes_ = np.unique(y)
        self.n_classes_ = len(self.classes_)
        self.n_features_ = X.shape[1]
        
        # Initialize containers
        self.trees = []
        self.bootstrap_indices = []
        self.oob_indices = []
        
        # Train trees
        for i in range(self.n_estimators):
            if self.verbose > 0 and (i + 1) % 10 == 0:
                print(f"Training tree {i + 1}/{self.n_estimators}")
            
            # Create bootstrap sample
            X_bootstrap, y_bootstrap, bootstrap_idx, oob_idx = self._bootstrap_sample(X, y, i)
            
            # Store indices for OOB calculation
            self.bootstrap_indices.append(bootstrap_idx)
            self.oob_indices.append(oob_idx)
            
            # Create and train tree
            tree = DecisionTreeRF(
                max_depth=self.max_depth,
                min_samples_split=self.min_samples_split,
                min_samples_leaf=self.min_samples_leaf,
                max_features=self.max_features,
                criterion=self.criterion,
                random_state=self.random_state + i if self.random_state else None
            )
            
            tree.fit(X_bootstrap, y_bootstrap)
            self.trees.append(tree)
        
        # Calculate OOB score if requested
        if self.oob_score:
            self._calculate_oob_score(X, y)
        
        # Calculate feature importance
        self._calculate_feature_importance()
        
        return self
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Make predictions using majority voting"""
        X = np.array(X)
        n_samples = X.shape[0]
        
        # Collect predictions from all trees
        all_predictions = np.zeros((n_samples, self.n_estimators), dtype=object)
        
        for i, tree in enumerate(self.trees):
            all_predictions[:, i] = tree.predict(X)
        
        # Majority voting
        final_predictions = []
        for i in range(n_samples):
            sample_predictions = all_predictions[i, :]
            most_common = Counter(sample_predictions).most_common(1)[0][0]
            final_predictions.append(most_common)
        
        return np.array(final_predictions)
    
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Predict class probabilities"""
        X = np.array(X)
        n_samples = X.shape[0]
        
        # Initialize probability matrix
        probabilities = np.zeros((n_samples, self.n_classes_))
        
        # Collect predictions from all trees
        for tree in self.trees:
            predictions = tree.predict(X)
            
            # Convert predictions to one-hot and accumulate
            for i, pred in enumerate(predictions):
                class_idx = np.where(self.classes_ == pred)[0][0]
                probabilities[i, class_idx] += 1
        
        # Normalize to get probabilities
        probabilities /= self.n_estimators
        
        return probabilities
    
    def score(self, X: np.ndarray, y: np.ndarray) -> float:
        """Calculate accuracy score"""
        predictions = self.predict(X)
        return np.mean(predictions == y)

class RandomForestRegressor:
    """Random Forest Regressor implementation from scratch"""
    
    def __init__(self,
                 n_estimators: int = 100,
                 criterion: str = 'mse',
                 max_depth: int | None = None,
                 min_samples_split: int = 2,
                 min_samples_leaf: int = 1,
                 max_features: str | float = 1.0,  # sklearn default for regression
                 bootstrap: bool = True,
                 oob_score: bool = False,
                 random_state: int | None = None,
                 verbose: int = 0):
        
        self.n_estimators = n_estimators
        self.criterion = criterion
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.max_features = max_features
        self.bootstrap = bootstrap
        self.oob_score = oob_score
        self.random_state = random_state
        self.verbose = verbose
        
        # Initialize containers
        self.trees = []
        self.feature_importances_ = None
        self.oob_score_ = None
        self.oob_prediction_ = None
        self.n_features_ = None
        
        # Use is not None to avoid treating random_state=0 as unset
        if random_state is not None:
            self.rng = np.random.default_rng(random_state)
        else:
            self.rng = np.random.default_rng()
        
        # Store bootstrap indices for OOB calculation
        self.bootstrap_indices = []
        self.oob_indices = []
    
    def _bootstrap_sample(self, X: np.ndarray, y: np.ndarray, sample_idx: int) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Create bootstrap sample of the data"""
        n_samples = X.shape[0]
        
        if self.bootstrap:
            # Sample with replacement using per-tree RNG for reproducibility
            if self.random_state is not None:
                tree_rng = np.random.default_rng(self.random_state + sample_idx)
            else:
                tree_rng = self.rng
            
            bootstrap_idx = tree_rng.choice(n_samples, size=n_samples, replace=True)
            oob_idx = np.setdiff1d(np.arange(n_samples), bootstrap_idx)
        else:
            # Use entire dataset
            bootstrap_idx = np.arange(n_samples)
            oob_idx = np.array([])
        
        X_bootstrap = X[bootstrap_idx]
        y_bootstrap = y[bootstrap_idx]
        
        return X_bootstrap, y_bootstrap, bootstrap_idx, oob_idx
    
    def _calculate_oob_score(self, X: np.ndarray, y: np.ndarray):
        """Calculate Out-of-Bag R² score"""
        if not self.bootstrap or not self.oob_score:
            return
        
        n_samples = X.shape[0]
        oob_predictions_sum = np.zeros(n_samples)
        oob_counts = np.zeros(n_samples)
        
        # Collect OOB predictions from all trees
        for tree_idx, (tree, oob_idx) in enumerate(zip(self.trees, self.oob_indices)):
            if len(oob_idx) > 0:
                # Get predictions for OOB samples
                oob_pred = tree.predict(X[oob_idx])
                
                # Accumulate predictions
                for i, sample_idx in enumerate(oob_idx):
                    oob_predictions_sum[sample_idx] += oob_pred[i]
                    oob_counts[sample_idx] += 1
        
        # Calculate OOB score only for samples with at least one OOB prediction
        valid_samples = oob_counts > 0
        if np.sum(valid_samples) > 0:
            # Average predictions
            oob_predictions = oob_predictions_sum.copy()
            oob_predictions[valid_samples] /= oob_counts[valid_samples]
            
            # Calculate R² score
            y_valid = y[valid_samples]
            y_pred_valid = oob_predictions[valid_samples]
            
            ss_res = np.sum((y_valid - y_pred_valid) ** 2)
            ss_tot = np.sum((y_valid - np.mean(y_valid)) ** 2)
            
            self.oob_score_ = 1 - (ss_res / ss_tot) if ss_tot != 0 else 0.0
            self.oob_prediction_ = oob_predictions
        else:
            self.oob_score_ = 0.0
    
    def _calculate_feature_importance(self):
        """Calculate feature importance based on average MSE decrease across trees"""
        if not self.trees:
            return
        
        self.feature_importances_ = np.zeros(self.n_features_)
        
        # Accumulate feature importances from all trees
        for tree in self.trees:
            if hasattr(tree, 'feature_importances_') and tree.feature_importances_ is not None:
                self.feature_importances_ += tree.feature_importances_
        
        # Average across trees and normalize
        if len(self.trees) > 0:
            self.feature_importances_ /= len(self.trees)
        
        if np.sum(self.feature_importances_) > 0:
            self.feature_importances_ /= np.sum(self.feature_importances_)
    
    def fit(self, X: np.ndarray, y: np.ndarray):
        """Train the Random Forest Regressor with bootstrap samples and regression trees"""
        # Convert to numpy arrays
        X = np.array(X)
        y = np.array(y)
        
        # Store basic information
        self.n_features_ = X.shape[1]
        
        # Initialize containers
        self.trees = []
        self.bootstrap_indices = []
        self.oob_indices = []
        
        # Train trees
        for i in range(self.n_estimators):
            if self.verbose > 0 and (i + 1) % 10 == 0:
                print(f"Training tree {i + 1}/{self.n_estimators}")
            
            # Create bootstrap sample
            X_bootstrap, y_bootstrap, bootstrap_idx, oob_idx = self._bootstrap_sample(X, y, i)
            
            # Store indices for OOB calculation
            self.bootstrap_indices.append(bootstrap_idx)
            self.oob_indices.append(oob_idx)
            
            # Create and train regression tree
            tree = DecisionTreeRegRF(
                max_depth=self.max_depth,
                min_samples_split=self.min_samples_split,
                min_samples_leaf=self.min_samples_leaf,
                max_features=self.max_features,
                criterion=self.criterion,
                random_state=self.random_state + i if self.random_state is not None else None
            )
            
            tree.fit(X_bootstrap, y_bootstrap)
            self.trees.append(tree)
        
        # Calculate OOB score if requested
        if self.oob_score:
            self._calculate_oob_score(X, y)
        
        # Calculate feature importance
        self._calculate_feature_importance()
        
        return self
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Make predictions by averaging predictions from all trees"""
        X = np.array(X)
        n_samples = X.shape[0]
        
        # Collect predictions from all trees
        all_predictions = np.zeros((n_samples, self.n_estimators))
        
        for i, tree in enumerate(self.trees):
            all_predictions[:, i] = tree.predict(X)
        
        # Average predictions
        final_predictions = np.mean(all_predictions, axis=1)
        
        return final_predictions
    
    def score(self, X: np.ndarray, y: np.ndarray) -> float:
        """Calculate R² score"""
        predictions = self.predict(X)
        ss_res = np.sum((y - predictions) ** 2)
        ss_tot = np.sum((y - np.mean(y)) ** 2)
        return 1 - (ss_res / ss_tot) if ss_tot != 0 else 0.0

def demonstrate_random_forest():
    """Demonstrate Random Forest implementation"""
    print("🌲 Random Forest Implementation Demo")
    print("=" * 50)
    
    # Generate sample data
    np.random.seed(42)
    n_samples = 1000
    n_features = 4
    
    # Create synthetic classification dataset
    X = np.random.randn(n_samples, n_features)
    # Create non-linear decision boundary
    y = ((X[:, 0] + X[:, 1] > 0) & (X[:, 2] - X[:, 3] > 0)).astype(int)
    
    # Add some noise
    noise_indices = np.random.choice(n_samples, size=int(0.1 * n_samples), replace=False)
    y[noise_indices] = 1 - y[noise_indices]
    
    print(f"Dataset shape: {X.shape}")
    print(f"Classes distribution: {Counter(y)}")
    
    # Split data
    split_idx = int(0.8 * n_samples)
    X_train, X_test = X[:split_idx], X[split_idx:]
    y_train, y_test = y[:split_idx], y[split_idx:]
    
    # Train Random Forest
    print("\nTraining Random Forest...")
    rf = RandomForestClassifier(
        n_estimators=50,
        max_depth=10,
        min_samples_split=5,
        max_features='sqrt',
        bootstrap=True,
        oob_score=True,
        random_state=42,
        verbose=1
    )
    
    rf.fit(X_train, y_train)
    
    # Make predictions
    print("\nMaking predictions...")
    y_pred = rf.predict(X_test)
    y_proba = rf.predict_proba(X_test)
    
    # Calculate metrics
    accuracy = rf.score(X_test, y_test)
    train_accuracy = rf.score(X_train, y_train)
    
    print("\nResults:")
    print(f"Training Accuracy: {train_accuracy:.4f}")
    print(f"Test Accuracy: {accuracy:.4f}")
    if rf.oob_score_:
        print(f"OOB Score: {rf.oob_score_:.4f}")
    
    # Show feature importance
    if rf.feature_importances_ is not None:
        print("\nFeature Importances:")
        for i, importance in enumerate(rf.feature_importances_):
            print(f"Feature {i}: {importance:.4f}")
    
    # Show some predictions
    print("\nSample Predictions (first 10):")
    print(f"True:      {y_test[:10]}")
    print(f"Predicted: {y_pred[:10]}")
    print(f"Probabilities (class 1): {y_proba[:10, 1]}")
    
    return rf, X_test, y_test, y_pred

def compare_with_single_tree():
    """Compare Random Forest with single Decision Tree"""
    print("\n🆚 Random Forest vs Single Decision Tree")
    print("=" * 50)
    
    # Generate sample data with more noise
    np.random.seed(42)
    n_samples = 500
    X = np.random.randn(n_samples, 4)
    y = ((X[:, 0] + X[:, 1] > 0) & (X[:, 2] - X[:, 3] > 0)).astype(int)
    
    # Add significant noise
    noise_indices = np.random.choice(n_samples, size=int(0.3 * n_samples), replace=False)
    y[noise_indices] = 1 - y[noise_indices]
    
    # Split data
    split_idx = int(0.8 * n_samples)
    X_train, X_test = X[:split_idx], X[split_idx:]
    y_train, y_test = y[:split_idx], y[split_idx:]
    
    # Train single decision tree
    single_tree = DecisionTreeRF(max_depth=10, random_state=42)
    single_tree.fit(X_train, y_train)
    tree_pred = single_tree.predict(X_test)
    tree_acc = np.mean(tree_pred == y_test)
    
    # Train Random Forest
    rf = RandomForestClassifier(n_estimators=50, max_depth=10, random_state=42)
    rf.fit(X_train, y_train)
  
    rf_acc = rf.score(X_test, y_test)
    
    print(f"Single Decision Tree Accuracy: {tree_acc:.4f}")
    print(f"Random Forest Accuracy: {rf_acc:.4f}")
    print(f"Improvement: {rf_acc - tree_acc:.4f}")
    
    return tree_acc, rf_acc

if __name__ == "__main__":
    # Run demonstrations
    rf_model, X_test, y_test, y_pred = demonstrate_random_forest()
    tree_acc, rf_acc = compare_with_single_tree()
    
    print("\n✅ Random Forest Implementation Complete!")
    print("\nKey Features Implemented:")
    print("- Bootstrap sampling")
    print("- Random feature selection")
    print("- Decision tree ensemble")
    print("- Majority voting for classification")
    print("- Out-of-Bag (OOB) error estimation")
    print("- Feature importance calculation")
    print("- Probability prediction")