"""
Gradient Boosting Implementation from Scratch

This module implements Gradient Boosting for both regression and classification
using decision trees as weak learners. The implementation includes:
- Regression with squared loss
- Binary classification with log-loss
- Learning rate (shrinkage)
- Early stopping with validation set
- Subsample (stochastic gradient boosting)

Author: ML-Algo Repository
"""

import numpy as np
from typing import Optional, Literal, Tuple
import warnings


class RegressionTreeStump:
    """
    Simple regression tree used as weak learner in gradient boosting.
    Supports controlled depth for boosting ensemble.
    """
    
    def __init__(self, max_depth: int = 3, min_samples_split: int = 2):
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.tree = None
        
    def fit(self, X: np.ndarray, y: np.ndarray):
        """Build regression tree"""
        self.n_features = X.shape[1]
        self.tree = self._grow_tree(X, y)
        return self
    
    def _grow_tree(self, X: np.ndarray, y: np.ndarray, depth: int = 0):
        """Recursively grow the tree"""
        n_samples = X.shape[0]
        
        # Stopping criteria
        if depth >= self.max_depth or n_samples < self.min_samples_split or len(np.unique(y)) == 1:
            return {"value": np.mean(y)}
        
        # Find best split
        best_feature, best_threshold, best_gain = self._best_split(X, y)
        
        if best_gain <= 0:
            return {"value": np.mean(y)}
        
        # Split dataset
        left_idx = X[:, best_feature] <= best_threshold
        right_idx = ~left_idx
        
        # Recursively build subtrees
        left_subtree = self._grow_tree(X[left_idx], y[left_idx], depth + 1)
        right_subtree = self._grow_tree(X[right_idx], y[right_idx], depth + 1)
        
        return {
            "feature": best_feature,
            "threshold": best_threshold,
            "left": left_subtree,
            "right": right_subtree
        }
    
    def _best_split(self, X: np.ndarray, y: np.ndarray) -> Tuple[int, float, float]:
        """Find the best split for a node"""
        best_gain = -np.inf
        best_feature = None
        best_threshold = None
        
        parent_variance = np.var(y)
        
        for feature_idx in range(self.n_features):
            thresholds = np.unique(X[:, feature_idx])
            
            for threshold in thresholds:
                left_idx = X[:, feature_idx] <= threshold
                right_idx = ~left_idx
                
                if np.sum(left_idx) == 0 or np.sum(right_idx) == 0:
                    continue
                
                # Calculate weighted variance reduction
                n = len(y)
                n_left, n_right = np.sum(left_idx), np.sum(right_idx)
                var_left = np.var(y[left_idx])
                var_right = np.var(y[right_idx])
                
                weighted_var = (n_left / n) * var_left + (n_right / n) * var_right
                gain = parent_variance - weighted_var
                
                if gain > best_gain:
                    best_gain = gain
                    best_feature = feature_idx
                    best_threshold = threshold
        
        return best_feature, best_threshold, best_gain
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict using the regression tree"""
        return np.array([self._traverse_tree(x, self.tree) for x in X])
    
    def _traverse_tree(self, x: np.ndarray, node: dict):
        """Traverse tree to make prediction for single sample"""
        if "value" in node:
            return node["value"]
        
        if x[node["feature"]] <= node["threshold"]:
            return self._traverse_tree(x, node["left"])
        else:
            return self._traverse_tree(x, node["right"])


class GradientBoostingRegressor:
    """
    Gradient Boosting for Regression
    
    Fits an additive model by sequentially training decision trees on the
    negative gradients (residuals) of the loss function. Uses squared loss.
    
    Parameters
    ----------
    n_estimators : int, default=100
        Number of boosting stages (trees) to fit
    learning_rate : float, default=0.1
        Shrinkage parameter to prevent overfitting
    max_depth : int, default=3
        Maximum depth of individual regression trees
    min_samples_split : int, default=2
        Minimum samples required to split an internal node
    subsample : float, default=1.0
        Fraction of samples used for fitting each tree (stochastic GB)
    early_stopping_rounds : int, default=None
        Stop training if validation loss doesn't improve for n rounds
    validation_fraction : float, default=0.1
        Fraction of training data to use for validation (early stopping)
    random_state : int, default=None
        Random seed for reproducibility
    verbose : int, default=0
        Verbosity level
        
    Attributes
    ----------
    trees_ : list
        Fitted regression trees
    init_prediction_ : float
        Initial prediction (mean of targets)
    train_losses_ : list
        Training loss at each boosting iteration
    val_losses_ : list
        Validation loss at each boosting iteration (if early stopping)
    """
    
    def __init__(
        self,
        n_estimators: int = 100,
        learning_rate: float = 0.1,
        max_depth: int = 3,
        min_samples_split: int = 2,
        subsample: float = 1.0,
        early_stopping_rounds: Optional[int] = None,
        validation_fraction: float = 0.1,
        random_state: Optional[int] = None,
        verbose: int = 0
    ):
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.subsample = subsample
        self.early_stopping_rounds = early_stopping_rounds
        self.validation_fraction = validation_fraction
        self.random_state = random_state
        self.verbose = verbose
        
        self.trees_ = []
        self.init_prediction_ = None
        self.train_losses_ = []
        self.val_losses_ = []
        self.n_estimators_ = n_estimators
        
    def fit(self, X: np.ndarray, y: np.ndarray) -> 'GradientBoostingRegressor':
        """
        Fit gradient boosting regressor
        
        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training data
        y : array-like of shape (n_samples,)
            Target values
            
        Returns
        -------
        self : GradientBoostingRegressor
        """
        np.random.seed(self.random_state)
        X = np.asarray(X)
        y = np.asarray(y)
        
        # Split for early stopping if needed
        X_train, y_train = X, y
        X_val, y_val = None, None
        
        if self.early_stopping_rounds is not None:
            n_val = int(len(X) * self.validation_fraction)
            indices = np.random.permutation(len(X))
            val_indices = indices[:n_val]
            train_indices = indices[n_val:]
            
            X_val, y_val = X[val_indices], y[val_indices]
            X_train, y_train = X[train_indices], y[train_indices]
        
        # Initialize with mean prediction
        self.init_prediction_ = np.mean(y_train)
        predictions = np.full(len(y_train), self.init_prediction_)
        
        if X_val is not None:
            val_predictions = np.full(len(y_val), self.init_prediction_)
        
        # Boosting iterations
        best_val_loss = np.inf
        rounds_without_improvement = 0
        
        for i in range(self.n_estimators):
            # Calculate negative gradients (residuals for squared loss)
            residuals = y_train - predictions
            
            # Subsample if needed
            if self.subsample < 1.0:
                n_samples = int(len(X_train) * self.subsample)
                sample_indices = np.random.choice(len(X_train), n_samples, replace=False)
                X_sample = X_train[sample_indices]
                residuals_sample = residuals[sample_indices]
            else:
                X_sample = X_train
                residuals_sample = residuals
            
            # Fit tree to negative gradients
            tree = RegressionTreeStump(
                max_depth=self.max_depth,
                min_samples_split=self.min_samples_split
            )
            tree.fit(X_sample, residuals_sample)
            self.trees_.append(tree)
            
            # Update predictions with learning rate
            tree_predictions = tree.predict(X_train)
            predictions += self.learning_rate * tree_predictions
            
            # Calculate training loss
            train_loss = np.mean((y_train - predictions) ** 2)
            self.train_losses_.append(train_loss)
            
            # Early stopping check
            if X_val is not None:
                val_tree_predictions = tree.predict(X_val)
                val_predictions += self.learning_rate * val_tree_predictions
                val_loss = np.mean((y_val - val_predictions) ** 2)
                self.val_losses_.append(val_loss)
                
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    rounds_without_improvement = 0
                else:
                    rounds_without_improvement += 1
                
                if rounds_without_improvement >= self.early_stopping_rounds:
                    if self.verbose > 0:
                        print(f"Early stopping at iteration {i+1}")
                    self.n_estimators_ = i + 1 - self.early_stopping_rounds
                    self.trees_ = self.trees_[:self.n_estimators_]
                    break
            
            if self.verbose > 0 and (i + 1) % 10 == 0:
                msg = f"Iteration {i+1}/{self.n_estimators}, Train Loss: {train_loss:.4f}"
                if X_val is not None:
                    msg += f", Val Loss: {val_loss:.4f}"
                print(msg)
        
        return self
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Make predictions
        
        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Samples to predict
            
        Returns
        -------
        predictions : array of shape (n_samples,)
            Predicted values
        """
        X = np.asarray(X)
        predictions = np.full(len(X), self.init_prediction_)
        
        for tree in self.trees_:
            predictions += self.learning_rate * tree.predict(X)
        
        return predictions
    
    def staged_predict(self, X: np.ndarray):
        """
        Return predictions at each boosting stage
        Useful for monitoring learning progress
        
        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Samples to predict
            
        Yields
        ------
        predictions : array of shape (n_samples,)
            Predictions at each stage
        """
        X = np.asarray(X)
        predictions = np.full(len(X), self.init_prediction_)
        yield predictions.copy()
        
        for tree in self.trees_:
            predictions += self.learning_rate * tree.predict(X)
            yield predictions.copy()


class GradientBoostingClassifier:
    """
    Gradient Boosting for Binary Classification
    
    Fits an additive model by sequentially training decision trees on the
    negative gradients of the log-loss function.
    
    Parameters
    ----------
    n_estimators : int, default=100
        Number of boosting stages (trees) to fit
    learning_rate : float, default=0.1
        Shrinkage parameter to prevent overfitting
    max_depth : int, default=3
        Maximum depth of individual regression trees
    min_samples_split : int, default=2
        Minimum samples required to split an internal node
    subsample : float, default=1.0
        Fraction of samples used for fitting each tree
    early_stopping_rounds : int, default=None
        Stop training if validation loss doesn't improve for n rounds
    validation_fraction : float, default=0.1
        Fraction of training data to use for validation
    random_state : int, default=None
        Random seed for reproducibility
    verbose : int, default=0
        Verbosity level
        
    Attributes
    ----------
    trees_ : list
        Fitted regression trees
    init_prediction_ : float
        Initial log-odds prediction
    train_losses_ : list
        Training loss at each boosting iteration
    val_losses_ : list
        Validation loss at each boosting iteration
    classes_ : array
        Class labels
    """
    
    def __init__(
        self,
        n_estimators: int = 100,
        learning_rate: float = 0.1,
        max_depth: int = 3,
        min_samples_split: int = 2,
        subsample: float = 1.0,
        early_stopping_rounds: Optional[int] = None,
        validation_fraction: float = 0.1,
        random_state: Optional[int] = None,
        verbose: int = 0
    ):
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.subsample = subsample
        self.early_stopping_rounds = early_stopping_rounds
        self.validation_fraction = validation_fraction
        self.random_state = random_state
        self.verbose = verbose
        
        self.trees_ = []
        self.init_prediction_ = None
        self.train_losses_ = []
        self.val_losses_ = []
        self.classes_ = None
        self.n_estimators_ = n_estimators
        
    @staticmethod
    def _sigmoid(x):
        """Numerically stable sigmoid function"""
        return np.where(
            x >= 0,
            1 / (1 + np.exp(-x)),
            np.exp(x) / (1 + np.exp(x))
        )
    
    @staticmethod
    def _log_loss(y_true, y_pred_proba):
        """Binary log loss"""
        eps = 1e-15
        y_pred_proba = np.clip(y_pred_proba, eps, 1 - eps)
        return -np.mean(y_true * np.log(y_pred_proba) + (1 - y_true) * np.log(1 - y_pred_proba))
    
    def fit(self, X: np.ndarray, y: np.ndarray) -> 'GradientBoostingClassifier':
        """
        Fit gradient boosting classifier
        
        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training data
        y : array-like of shape (n_samples,)
            Target values (0 or 1)
            
        Returns
        -------
        self : GradientBoostingClassifier
        """
        np.random.seed(self.random_state)
        X = np.asarray(X)
        y = np.asarray(y)
        
        self.classes_ = np.unique(y)
        if len(self.classes_) != 2:
            raise ValueError("Only binary classification is supported")
        
        # Ensure labels are 0 and 1
        y = (y == self.classes_[1]).astype(int)
        
        # Split for early stopping if needed
        X_train, y_train = X, y
        X_val, y_val = None, None
        
        if self.early_stopping_rounds is not None:
            n_val = int(len(X) * self.validation_fraction)
            indices = np.random.permutation(len(X))
            val_indices = indices[:n_val]
            train_indices = indices[n_val:]
            
            X_val, y_val = X[val_indices], y[val_indices]
            X_train, y_train = X[train_indices], y[train_indices]
        
        # Initialize with log-odds
        p = np.clip(np.mean(y_train), 1e-15, 1 - 1e-15)
        self.init_prediction_ = np.log(p / (1 - p))
        
        # Raw predictions (log-odds)
        raw_predictions = np.full(len(y_train), self.init_prediction_)
        
        if X_val is not None:
            val_raw_predictions = np.full(len(y_val), self.init_prediction_)
        
        # Boosting iterations
        best_val_loss = np.inf
        rounds_without_improvement = 0
        
        for i in range(self.n_estimators):
            # Calculate probabilities and negative gradients
            probabilities = self._sigmoid(raw_predictions)
            residuals = y_train - probabilities
            
            # Subsample if needed
            if self.subsample < 1.0:
                n_samples = int(len(X_train) * self.subsample)
                sample_indices = np.random.choice(len(X_train), n_samples, replace=False)
                X_sample = X_train[sample_indices]
                residuals_sample = residuals[sample_indices]
            else:
                X_sample = X_train
                residuals_sample = residuals
            
            # Fit tree to negative gradients
            tree = RegressionTreeStump(
                max_depth=self.max_depth,
                min_samples_split=self.min_samples_split
            )
            tree.fit(X_sample, residuals_sample)
            self.trees_.append(tree)
            
            # Update raw predictions
            tree_predictions = tree.predict(X_train)
            raw_predictions += self.learning_rate * tree_predictions
            
            # Calculate training loss
            probabilities = self._sigmoid(raw_predictions)
            train_loss = self._log_loss(y_train, probabilities)
            self.train_losses_.append(train_loss)
            
            # Early stopping check
            if X_val is not None:
                val_tree_predictions = tree.predict(X_val)
                val_raw_predictions += self.learning_rate * val_tree_predictions
                val_probabilities = self._sigmoid(val_raw_predictions)
                val_loss = self._log_loss(y_val, val_probabilities)
                self.val_losses_.append(val_loss)
                
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    rounds_without_improvement = 0
                else:
                    rounds_without_improvement += 1
                
                if rounds_without_improvement >= self.early_stopping_rounds:
                    if self.verbose > 0:
                        print(f"Early stopping at iteration {i+1}")
                    self.n_estimators_ = i + 1 - self.early_stopping_rounds
                    self.trees_ = self.trees_[:self.n_estimators_]
                    break
            
            if self.verbose > 0 and (i + 1) % 10 == 0:
                msg = f"Iteration {i+1}/{self.n_estimators}, Train Loss: {train_loss:.4f}"
                if X_val is not None:
                    msg += f", Val Loss: {val_loss:.4f}"
                print(msg)
        
        return self
    
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """
        Predict class probabilities
        
        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Samples to predict
            
        Returns
        -------
        probabilities : array of shape (n_samples, 2)
            Probability of each class
        """
        X = np.asarray(X)
        raw_predictions = np.full(len(X), self.init_prediction_)
        
        for tree in self.trees_:
            raw_predictions += self.learning_rate * tree.predict(X)
        
        proba_class1 = self._sigmoid(raw_predictions)
        proba_class0 = 1 - proba_class1
        
        return np.column_stack([proba_class0, proba_class1])
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict class labels
        
        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Samples to predict
            
        Returns
        -------
        predictions : array of shape (n_samples,)
            Predicted class labels
        """
        proba = self.predict_proba(X)
        return self.classes_[np.argmax(proba, axis=1)]
    
    def staged_predict_proba(self, X: np.ndarray):
        """
        Return probability predictions at each boosting stage
        
        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Samples to predict
            
        Yields
        ------
        probabilities : array of shape (n_samples, 2)
            Probabilities at each stage
        """
        X = np.asarray(X)
        raw_predictions = np.full(len(X), self.init_prediction_)
        
        proba_class1 = self._sigmoid(raw_predictions)
        proba_class0 = 1 - proba_class1
        yield np.column_stack([proba_class0, proba_class1])
        
        for tree in self.trees_:
            raw_predictions += self.learning_rate * tree.predict(X)
            proba_class1 = self._sigmoid(raw_predictions)
            proba_class0 = 1 - proba_class1
            yield np.column_stack([proba_class0, proba_class1])


def demo_regression():
    """Demonstration of Gradient Boosting Regressor"""
    print("=" * 60)
    print("Gradient Boosting Regressor Demo")
    print("=" * 60)
    
    # Generate synthetic data: Daily store sales
    np.random.seed(42)
    n_samples = 500
    
    # Features: day of week (0-6), holiday (0/1), promotion (0/1), temperature
    day_of_week = np.random.randint(0, 7, n_samples)
    is_holiday = np.random.binomial(1, 0.1, n_samples)
    has_promotion = np.random.binomial(1, 0.2, n_samples)
    temperature = np.random.normal(20, 8, n_samples)
    
    X = np.column_stack([day_of_week, is_holiday, has_promotion, temperature])
    
    # Target: Sales (with non-linear effects)
    sales = (
        1000 +  # Base sales
        150 * (day_of_week >= 5) +  # Weekend boost
        300 * is_holiday +  # Holiday boost
        200 * has_promotion +  # Promotion boost
        10 * temperature +  # Temperature effect
        -0.2 * temperature ** 2 +  # Non-linear temperature effect
        np.random.normal(0, 50, n_samples)  # Noise
    )
    
    # Train-test split
    n_train = 400
    X_train, X_test = X[:n_train], X[n_train:]
    y_train, y_test = sales[:n_train], sales[n_train:]
    
    # Fit model
    print("\nTraining Gradient Boosting Regressor...")
    gb = GradientBoostingRegressor(
        n_estimators=100,
        learning_rate=0.1,
        max_depth=3,
        subsample=0.8,
        early_stopping_rounds=10,
        random_state=42,
        verbose=0
    )
    gb.fit(X_train, y_train)
    
    # Predictions
    y_pred = gb.predict(X_test)
    
    # Evaluation
    mse = np.mean((y_test - y_pred) ** 2)
    rmse = np.sqrt(mse)
    r2 = 1 - np.sum((y_test - y_pred) ** 2) / np.sum((y_test - np.mean(y_test)) ** 2)
    
    print(f"\nResults:")
    print(f"  Number of trees used: {gb.n_estimators_}")
    print(f"  Test RMSE: {rmse:.2f}")
    print(f"  Test R²: {r2:.4f}")
    print(f"\nSample predictions:")
    for i in range(5):
        print(f"  Actual: {y_test[i]:.2f}, Predicted: {y_pred[i]:.2f}")


def demo_classification():
    """Demonstration of Gradient Boosting Classifier"""
    print("\n" + "=" * 60)
    print("Gradient Boosting Classifier Demo")
    print("=" * 60)
    
    # Generate synthetic data: Customer churn prediction
    np.random.seed(42)
    n_samples = 1000
    
    # Features: tenure, monthly_charges, total_charges, num_complaints
    tenure = np.random.exponential(24, n_samples)
    monthly_charges = np.random.uniform(30, 150, n_samples)
    total_charges = tenure * monthly_charges + np.random.normal(0, 100, n_samples)
    num_complaints = np.random.poisson(2, n_samples)
    
    X = np.column_stack([tenure, monthly_charges, total_charges, num_complaints])
    
    # Target: Churn (1) or not (0)
    # Probability increases with complaints, charges, and decreases with tenure
    logit = (
        -2 +
        -0.05 * tenure +
        0.02 * monthly_charges +
        0.5 * num_complaints +
        np.random.normal(0, 0.5, n_samples)
    )
    churn_prob = 1 / (1 + np.exp(-logit))
    y = (np.random.random(n_samples) < churn_prob).astype(int)
    
    # Train-test split
    n_train = 800
    X_train, X_test = X[:n_train], X[n_train:]
    y_train, y_test = y[:n_train], y[n_train:]
    
    # Fit model
    print("\nTraining Gradient Boosting Classifier...")
    gb = GradientBoostingClassifier(
        n_estimators=100,
        learning_rate=0.1,
        max_depth=3,
        subsample=0.8,
        early_stopping_rounds=10,
        random_state=42,
        verbose=0
    )
    gb.fit(X_train, y_train)
    
    # Predictions
    y_pred = gb.predict(X_test)
    y_proba = gb.predict_proba(X_test)
    
    # Evaluation
    accuracy = np.mean(y_pred == y_test)
    
    # Confusion matrix
    tp = np.sum((y_pred == 1) & (y_test == 1))
    tn = np.sum((y_pred == 0) & (y_test == 0))
    fp = np.sum((y_pred == 1) & (y_test == 0))
    fn = np.sum((y_pred == 0) & (y_test == 1))
    
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    
    print(f"\nResults:")
    print(f"  Number of trees used: {gb.n_estimators_}")
    print(f"  Test Accuracy: {accuracy:.4f}")
    print(f"  Precision: {precision:.4f}")
    print(f"  Recall: {recall:.4f}")
    print(f"  F1-Score: {f1:.4f}")
    print(f"\nSample predictions (class probabilities):")
    for i in range(5):
        print(f"  True: {y_test[i]}, Pred: {y_pred[i]}, Prob(Churn): {y_proba[i, 1]:.3f}")


if __name__ == "__main__":
    demo_regression()
    demo_classification()
