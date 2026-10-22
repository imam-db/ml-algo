"""
Naive Bayes - Implementation from Scratch

Gaussian and Multinomial Naive Bayes classifiers.
"""

import numpy as np
from typing import Optional
from collections import defaultdict


class GaussianNaiveBayes:
    """
    Gaussian Naive Bayes classifier.
    
    Assumes features follow Gaussian (normal) distribution.
    Good for continuous features.
    """
    
    def __init__(self, var_smoothing: float = 1e-9):
        """
        Initialize Gaussian Naive Bayes.
        
        Args:
            var_smoothing: Portion of the largest variance added to variances
                          for calculation stability
        """
        self.var_smoothing = var_smoothing
        
        self.classes = None
        self.class_priors = None
        self.means = None
        self.variances = None
    
    def fit(self, X: np.ndarray, y: np.ndarray) -> 'GaussianNaiveBayes':
        """
        Fit Gaussian Naive Bayes model.
        
        Args:
            X: Training features, shape (n_samples, n_features)
            y: Training labels, shape (n_samples,)
            
        Returns:
            self
        """
        n_samples, n_features = X.shape
        
        # Get unique classes
        self.classes = np.unique(y)
        n_classes = len(self.classes)
        
        # Initialize parameters
        self.means = np.zeros((n_classes, n_features))
        self.variances = np.zeros((n_classes, n_features))
        self.class_priors = np.zeros(n_classes)
        
        # Calculate mean, variance, and prior for each class
        for idx, c in enumerate(self.classes):
            X_c = X[y == c]
            
            self.means[idx, :] = X_c.mean(axis=0)
            self.variances[idx, :] = X_c.var(axis=0) + self.var_smoothing
            self.class_priors[idx] = X_c.shape[0] / n_samples
        
        return self
    
    def _calculate_likelihood(self, class_idx: int, x: np.ndarray) -> float:
        """
        Calculate Gaussian likelihood P(x|class).
        
        Using Gaussian PDF:
        P(x|class) = (1/√(2πσ²)) * exp(-(x-μ)²/(2σ²))
        """
        mean = self.means[class_idx]
        var = self.variances[class_idx]
        
        # Log likelihood for numerical stability
        log_likelihood = -0.5 * np.sum(np.log(2 * np.pi * var))
        log_likelihood -= 0.5 * np.sum(((x - mean) ** 2) / var)
        
        return log_likelihood
    
    def _calculate_posterior(self, x: np.ndarray) -> np.ndarray:
        """
        Calculate posterior probabilities for all classes.
        
        P(class|x) ∝ P(x|class) * P(class)
        Using log probabilities for numerical stability.
        """
        posteriors = []
        
        for idx, c in enumerate(self.classes):
            # Log prior
            log_prior = np.log(self.class_priors[idx])
            
            # Log likelihood
            log_likelihood = self._calculate_likelihood(idx, x)
            
            # Log posterior
            posterior = log_prior + log_likelihood
            posteriors.append(posterior)
        
        return np.array(posteriors)
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict class labels.
        
        Args:
            X: Features to predict, shape (n_samples, n_features)
            
        Returns:
            Predicted labels, shape (n_samples,)
        """
        predictions = [self._predict_single(x) for x in X]
        return np.array(predictions)
    
    def _predict_single(self, x: np.ndarray):
        """Predict class for single sample."""
        posteriors = self._calculate_posterior(x)
        return self.classes[np.argmax(posteriors)]
    
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """
        Predict class probabilities.
        
        Args:
            X: Features to predict, shape (n_samples, n_features)
            
        Returns:
            Class probabilities, shape (n_samples, n_classes)
        """
        probas = []
        
        for x in X:
            posteriors = self._calculate_posterior(x)
            
            # Convert log probabilities to probabilities
            # Use softmax for numerical stability
            max_posterior = np.max(posteriors)
            exp_posteriors = np.exp(posteriors - max_posterior)
            probas.append(exp_posteriors / np.sum(exp_posteriors))
        
        return np.array(probas)
    
    def predict_log_proba(self, X: np.ndarray) -> np.ndarray:
        """Predict log probabilities."""
        log_probas = []
        
        for x in X:
            posteriors = self._calculate_posterior(x)
            
            # Normalize log probabilities
            max_posterior = np.max(posteriors)
            log_sum_exp = max_posterior + np.log(
                np.sum(np.exp(posteriors - max_posterior))
            )
            normalized = posteriors - log_sum_exp
            log_probas.append(normalized)
        
        return np.array(log_probas)
    
    def score(self, X: np.ndarray, y: np.ndarray) -> float:
        """Calculate accuracy score."""
        predictions = self.predict(X)
        return np.mean(predictions == y)


class MultinomialNaiveBayes:
    """
    Multinomial Naive Bayes classifier.
    
    Suitable for discrete features (e.g., word counts, term frequencies).
    Good for text classification.
    """
    
    def __init__(self, alpha: float = 1.0):
        """
        Initialize Multinomial Naive Bayes.
        
        Args:
            alpha: Additive (Laplace/Lidstone) smoothing parameter
                   (0 = no smoothing, 1 = Laplace smoothing)
        """
        self.alpha = alpha
        
        self.classes = None
        self.class_priors = None
        self.feature_log_probs = None
    
    def fit(self, X: np.ndarray, y: np.ndarray) -> 'MultinomialNaiveBayes':
        """
        Fit Multinomial Naive Bayes model.
        
        Args:
            X: Training features (counts), shape (n_samples, n_features)
            y: Training labels, shape (n_samples,)
            
        Returns:
            self
        """
        n_samples, n_features = X.shape
        
        # Get unique classes
        self.classes = np.unique(y)
        n_classes = len(self.classes)
        
        # Initialize parameters
        self.class_priors = np.zeros(n_classes)
        self.feature_log_probs = np.zeros((n_classes, n_features))
        
        # Calculate parameters for each class
        for idx, c in enumerate(self.classes):
            X_c = X[y == c]
            
            # Class prior: P(class)
            self.class_priors[idx] = X_c.shape[0] / n_samples
            
            # Feature probabilities with Laplace smoothing
            # P(feature|class) = (count(feature, class) + alpha) / (count(class) + alpha * n_features)
            feature_counts = X_c.sum(axis=0)
            total_count = feature_counts.sum()
            
            # Log probabilities for numerical stability
            self.feature_log_probs[idx, :] = np.log(
                (feature_counts + self.alpha) / 
                (total_count + self.alpha * n_features)
            )
        
        return self
    
    def _calculate_posterior(self, x: np.ndarray) -> np.ndarray:
        """
        Calculate posterior log probabilities.
        
        P(class|x) ∝ P(class) * ∏P(feature_i|class)^(x_i)
        log P(class|x) = log P(class) + Σ x_i * log P(feature_i|class)
        """
        posteriors = []
        
        for idx, c in enumerate(self.classes):
            # Log prior
            log_prior = np.log(self.class_priors[idx])
            
            # Log likelihood (sum of x_i * log P(feature_i|class))
            log_likelihood = np.sum(x * self.feature_log_probs[idx, :])
            
            # Log posterior
            posterior = log_prior + log_likelihood
            posteriors.append(posterior)
        
        return np.array(posteriors)
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict class labels."""
        predictions = [self._predict_single(x) for x in X]
        return np.array(predictions)
    
    def _predict_single(self, x: np.ndarray):
        """Predict class for single sample."""
        posteriors = self._calculate_posterior(x)
        return self.classes[np.argmax(posteriors)]
    
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Predict class probabilities."""
        probas = []
        
        for x in X:
            posteriors = self._calculate_posterior(x)
            
            # Convert log probabilities to probabilities using softmax
            max_posterior = np.max(posteriors)
            exp_posteriors = np.exp(posteriors - max_posterior)
            probas.append(exp_posteriors / np.sum(exp_posteriors))
        
        return np.array(probas)
    
    def predict_log_proba(self, X: np.ndarray) -> np.ndarray:
        """Predict log probabilities."""
        log_probas = []
        
        for x in X:
            posteriors = self._calculate_posterior(x)
            
            # Normalize
            max_posterior = np.max(posteriors)
            log_sum_exp = max_posterior + np.log(
                np.sum(np.exp(posteriors - max_posterior))
            )
            normalized = posteriors - log_sum_exp
            log_probas.append(normalized)
        
        return np.array(log_probas)
    
    def score(self, X: np.ndarray, y: np.ndarray) -> float:
        """Calculate accuracy score."""
        predictions = self.predict(X)
        return np.mean(predictions == y)


if __name__ == "__main__":
    from sklearn.datasets import make_classification, load_iris
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import StandardScaler
    
    print("="*60)
    print("GAUSSIAN NAIVE BAYES DEMO")
    print("="*60)
    
    # Load iris dataset (continuous features)
    iris = load_iris()
    X, y = iris.data, iris.target
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42
    )
    
    # Train Gaussian Naive Bayes
    print("\nTraining Gaussian Naive Bayes...")
    gnb = GaussianNaiveBayes()
    gnb.fit(X_train, y_train)
    
    # Evaluate
    train_acc = gnb.score(X_train, y_train)
    test_acc = gnb.score(X_test, y_test)
    
    print(f"\nTraining Accuracy: {train_acc:.4f}")
    print(f"Test Accuracy: {test_acc:.4f}")
    
    # Show predictions with probabilities
    print("\nSample Predictions:")
    sample_X = X_test[:5]
    sample_y = y_test[:5]
    predictions = gnb.predict(sample_X)
    probabilities = gnb.predict_proba(sample_X)
    
    for i in range(5):
        print(f"  True: {sample_y[i]}, Predicted: {predictions[i]}, " +
              f"Probabilities: {probabilities[i]}")
    
    print("\n" + "="*60)
    print("MULTINOMIAL NAIVE BAYES DEMO")
    print("="*60)
    
    # Generate discrete count data (simulating word counts)
    np.random.seed(42)
    n_samples = 300
    n_features = 20
    
    # Generate count data for 3 classes
    X_counts = []
    y_labels = []
    
    for label in range(3):
        # Each class has different feature distributions
        for _ in range(n_samples // 3):
            # Simulate word counts (Poisson distribution)
            counts = np.random.poisson(lam=2 + label, size=n_features)
            # Add class-specific patterns
            if label == 0:
                counts[:5] *= 3  # Class 0 has more of features 0-4
            elif label == 1:
                counts[5:10] *= 3  # Class 1 has more of features 5-9
            else:
                counts[10:15] *= 3  # Class 2 has more of features 10-14
            
            X_counts.append(counts)
            y_labels.append(label)
    
    X_counts = np.array(X_counts)
    y_labels = np.array(y_labels)
    
    X_train_mnb, X_test_mnb, y_train_mnb, y_test_mnb = train_test_split(
        X_counts, y_labels, test_size=0.3, random_state=42
    )
    
    # Train Multinomial Naive Bayes
    print("\nTraining Multinomial Naive Bayes...")
    mnb = MultinomialNaiveBayes(alpha=1.0)
    mnb.fit(X_train_mnb, y_train_mnb)
    
    # Evaluate
    train_acc = mnb.score(X_train_mnb, y_train_mnb)
    test_acc = mnb.score(X_test_mnb, y_test_mnb)
    
    print(f"\nTraining Accuracy: {train_acc:.4f}")
    print(f"Test Accuracy: {test_acc:.4f}")
    
    # Show predictions
    print("\nSample Predictions:")
    sample_X = X_test_mnb[:5]
    sample_y = y_test_mnb[:5]
    predictions = mnb.predict(sample_X)
    probabilities = mnb.predict_proba(sample_X)
    
    for i in range(5):
        print(f"  True: {sample_y[i]}, Predicted: {predictions[i]}, " +
              f"Probabilities: {probabilities[i]}")
    
    print("\n" + "="*60)
    print("SMOOTHING PARAMETER COMPARISON")
    print("="*60)
    
    print("\nMultinomial NB with different alpha values:")
    for alpha in [0.1, 1.0, 5.0, 10.0]:
        mnb = MultinomialNaiveBayes(alpha=alpha)
        mnb.fit(X_train_mnb, y_train_mnb)
        test_acc = mnb.score(X_test_mnb, y_test_mnb)
        print(f"  Alpha={alpha:>5.1f}: Test Accuracy = {test_acc:.4f}")
    
    print("\n✓ Naive Bayes demonstration completed!")
