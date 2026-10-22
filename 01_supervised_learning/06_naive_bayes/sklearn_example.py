"""
Naive Bayes using scikit-learn
Comparison with from-scratch implementation
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.naive_bayes import GaussianNB, MultinomialNB
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import make_classification, load_iris
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# Import our implementation
import sys
import os
import importlib.util

nb_path = os.path.join(os.path.dirname(__file__), 'implementation.py')
spec = importlib.util.spec_from_file_location("nb_impl", nb_path)
nb_impl = importlib.util.module_from_spec(spec)
spec.loader.exec_module(nb_impl)


def compare_gaussian_nb():
    """Compare sklearn and custom Gaussian Naive Bayes."""
    print("="*70)
    print("COMPARING IMPLEMENTATIONS: GAUSSIAN NAIVE BAYES")
    print("="*70)
    
    # Load Iris dataset
    iris = load_iris()
    X, y = iris.data, iris.target
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42
    )
    
    # Custom implementation
    print("\nCustom Gaussian NB:")
    custom_gnb = nb_impl.GaussianNaiveBayes(var_smoothing=1e-9)
    custom_gnb.fit(X_train, y_train)
    custom_pred = custom_gnb.predict(X_test)
    custom_acc = custom_gnb.score(X_test, y_test)
    print(f"  Accuracy: {custom_acc:.4f}")
    
    # Sklearn implementation
    print("\nScikit-learn Gaussian NB:")
    sklearn_gnb = GaussianNB(var_smoothing=1e-9)
    sklearn_gnb.fit(X_train, y_train)
    sklearn_pred = sklearn_gnb.predict(X_test)
    sklearn_acc = sklearn_gnb.score(X_test, y_test)
    print(f"  Accuracy: {sklearn_acc:.4f}")
    
    # Compare predictions
    agreement = np.mean(custom_pred == sklearn_pred)
    print(f"\nPrediction Agreement: {agreement*100:.2f}%")
    
    print("\nClassification Report (sklearn):")
    print(classification_report(y_test, sklearn_pred, 
                               target_names=iris.target_names))


def compare_multinomial_nb():
    """Compare sklearn and custom Multinomial Naive Bayes."""
    print("\n" + "="*70)
    print("COMPARING IMPLEMENTATIONS: MULTINOMIAL NAIVE BAYES")
    print("="*70)
    
    # Generate count data
    np.random.seed(42)
    n_samples = 300
    n_features = 20
    
    X_counts = []
    y_labels = []
    
    for label in range(3):
        for _ in range(n_samples // 3):
            counts = np.random.poisson(lam=2 + label, size=n_features)
            if label == 0:
                counts[:5] *= 3
            elif label == 1:
                counts[5:10] *= 3
            else:
                counts[10:15] *= 3
            
            X_counts.append(counts)
            y_labels.append(label)
    
    X_counts = np.array(X_counts)
    y_labels = np.array(y_labels)
    
    X_train, X_test, y_train, y_test = train_test_split(
        X_counts, y_labels, test_size=0.3, random_state=42
    )
    
    # Custom implementation
    print("\nCustom Multinomial NB:")
    custom_mnb = nb_impl.MultinomialNaiveBayes(alpha=1.0)
    custom_mnb.fit(X_train, y_train)
    custom_pred = custom_mnb.predict(X_test)
    custom_acc = custom_mnb.score(X_test, y_test)
    print(f"  Accuracy: {custom_acc:.4f}")
    
    # Sklearn implementation
    print("\nScikit-learn Multinomial NB:")
    sklearn_mnb = MultinomialNB(alpha=1.0)
    sklearn_mnb.fit(X_train, y_train)
    sklearn_pred = sklearn_mnb.predict(X_test)
    sklearn_acc = sklearn_mnb.score(X_test, y_test)
    print(f"  Accuracy: {sklearn_acc:.4f}")
    
    # Compare predictions
    agreement = np.mean(custom_pred == sklearn_pred)
    print(f"\nPrediction Agreement: {agreement*100:.2f}%")


def hyperparameter_tuning_gaussian():
    """Demonstrate hyperparameter tuning for Gaussian NB."""
    print("\n" + "="*70)
    print("HYPERPARAMETER TUNING: GAUSSIAN NAIVE BAYES")
    print("="*70)
    
    # Generate data
    X, y = make_classification(
        n_samples=500,
        n_features=10,
        n_informative=8,
        n_redundant=2,
        n_classes=3,
        random_state=42
    )
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42
    )
    
    # Grid search
    param_grid = {
        'var_smoothing': [1e-10, 1e-9, 1e-8, 1e-7, 1e-6]
    }
    
    print("\nPerforming Grid Search...")
    gnb = GaussianNB()
    grid_search = GridSearchCV(
        gnb, 
        param_grid, 
        cv=5, 
        scoring='accuracy',
        verbose=0
    )
    
    grid_search.fit(X_train, y_train)
    
    print(f"\nBest Parameters: {grid_search.best_params_}")
    print(f"Best CV Score: {grid_search.best_score_:.4f}")
    
    # Test with best model
    best_model = grid_search.best_estimator_
    test_score = best_model.score(X_test, y_test)
    print(f"Test Score: {test_score:.4f}")


def hyperparameter_tuning_multinomial():
    """Demonstrate hyperparameter tuning for Multinomial NB."""
    print("\n" + "="*70)
    print("HYPERPARAMETER TUNING: MULTINOMIAL NAIVE BAYES")
    print("="*70)
    
    # Generate count data
    np.random.seed(42)
    X = np.random.poisson(lam=5, size=(400, 15))
    # Create patterns for classes
    X[:133, :5] *= 2  # Class 0 has more of first 5 features
    X[133:266, 5:10] *= 2  # Class 1 has more of middle 5
    X[266:, 10:] *= 2  # Class 2 has more of last 5
    y = np.array([0]*133 + [1]*133 + [2]*134)
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42
    )
    
    # Test different alpha values
    print("\nTesting Different Alpha Values:")
    print(f"{'Alpha':>10} | {'Train Acc':>10} | {'Test Acc':>10}")
    print("-" * 36)
    
    for alpha in [0.01, 0.1, 1.0, 5.0, 10.0]:
        mnb = MultinomialNB(alpha=alpha)
        mnb.fit(X_train, y_train)
        
        train_acc = mnb.score(X_train, y_train)
        test_acc = mnb.score(X_test, y_test)
        
        print(f"{alpha:>10.2f} | {train_acc:>10.4f} | {test_acc:>10.4f}")


def cross_validation_demo():
    """Demonstrate cross-validation."""
    print("\n" + "="*70)
    print("CROSS-VALIDATION")
    print("="*70)
    
    # Generate data
    X, y = make_classification(
        n_samples=400,
        n_features=8,
        n_informative=6,
        n_redundant=2,
        random_state=42
    )
    
    print("\nGaussian Naive Bayes - 5-Fold CV:")
    gnb = GaussianNB()
    cv_scores = cross_val_score(gnb, X, y, cv=5, scoring='accuracy')
    
    print(f"CV Scores: {cv_scores}")
    print(f"Mean: {cv_scores.mean():.4f} (+/- {cv_scores.std():.4f})")


def feature_importance_demo():
    """Demonstrate feature importance analysis."""
    print("\n" + "="*70)
    print("FEATURE IMPORTANCE ANALYSIS")
    print("="*70)
    
    # Load Iris
    iris = load_iris()
    X, y = iris.data, iris.target
    
    # Train model
    gnb = GaussianNB()
    gnb.fit(X, y)
    
    # Analyze feature importance through variance ratio
    print("\nFeature Statistics per Class:")
    print(f"{'Feature':<20} | {'Class 0':>10} | {'Class 1':>10} | {'Class 2':>10}")
    print("-" * 56)
    
    for i, feature_name in enumerate(iris.feature_names):
        means = gnb.theta_[:, i]  # Mean per class
        print(f"{feature_name:<20} | {means[0]:>10.2f} | {means[1]:>10.2f} | {means[2]:>10.2f}")


def probability_analysis():
    """Analyze prediction probabilities."""
    print("\n" + "="*70)
    print("PROBABILITY ANALYSIS")
    print("="*70)
    
    # Load Iris
    iris = load_iris()
    X, y = iris.data, iris.target
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42
    )
    
    # Train model
    gnb = GaussianNB()
    gnb.fit(X_train, y_train)
    
    # Get probabilities
    probas = gnb.predict_proba(X_test)
    
    # Analyze confidence
    max_probas = probas.max(axis=1)
    
    print(f"\nProbability Statistics:")
    print(f"  Mean confidence: {max_probas.mean():.4f}")
    print(f"  Median confidence: {np.median(max_probas):.4f}")
    print(f"  Min confidence: {max_probas.min():.4f}")
    print(f"  Max confidence: {max_probas.max():.4f}")
    
    # Show some examples
    print(f"\nSample Predictions with Probabilities:")
    print(f"{'True':>6} | {'Pred':>6} | {'Prob 0':>8} | {'Prob 1':>8} | {'Prob 2':>8} | {'Confidence':>10}")
    print("-" * 62)
    
    preds = gnb.predict(X_test[:10])
    for i in range(10):
        true_label = y_test[i]
        pred_label = preds[i]
        probs = probas[i]
        confidence = max_probas[i]
        
        print(f"{true_label:>6} | {pred_label:>6} | {probs[0]:>8.4f} | {probs[1]:>8.4f} | {probs[2]:>8.4f} | {confidence:>10.4f}")


def text_classification_simulation():
    """Simulate text classification with Multinomial NB."""
    print("\n" + "="*70)
    print("TEXT CLASSIFICATION SIMULATION")
    print("="*70)
    
    # Simulate word counts for document classification
    # Categories: Tech, Sports, Politics
    np.random.seed(42)
    
    # Tech documents (high counts for tech words)
    tech_docs = np.random.poisson(lam=2, size=(50, 30))
    tech_docs[:, :10] *= 5  # Tech vocabulary
    
    # Sports documents
    sports_docs = np.random.poisson(lam=2, size=(50, 30))
    sports_docs[:, 10:20] *= 5  # Sports vocabulary
    
    # Politics documents
    politics_docs = np.random.poisson(lam=2, size=(50, 30))
    politics_docs[:, 20:] *= 5  # Politics vocabulary
    
    X = np.vstack([tech_docs, sports_docs, politics_docs])
    y = np.array(['tech']*50 + ['sports']*50 + ['politics']*50)
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42
    )
    
    # Train Multinomial NB
    mnb = MultinomialNB(alpha=1.0)
    mnb.fit(X_train, y_train)
    
    # Evaluate
    train_acc = mnb.score(X_train, y_train)
    test_acc = mnb.score(X_test, y_test)
    
    print(f"\nDocument Classification Results:")
    print(f"  Training Accuracy: {train_acc:.4f}")
    print(f"  Test Accuracy: {test_acc:.4f}")
    
    # Show predictions
    y_pred = mnb.predict(X_test)
    
    print(f"\nConfusion Matrix:")
    cm = confusion_matrix(y_test, y_pred, labels=['tech', 'sports', 'politics'])
    print(cm)
    
    print(f"\nClassification Report:")
    print(classification_report(y_test, y_pred))


if __name__ == "__main__":
    # Run all demonstrations
    compare_gaussian_nb()
    compare_multinomial_nb()
    hyperparameter_tuning_gaussian()
    hyperparameter_tuning_multinomial()
    cross_validation_demo()
    feature_importance_demo()
    probability_analysis()
    text_classification_simulation()
    
    print("\n" + "="*70)
    print("✓ All sklearn Naive Bayes demonstrations completed!")
    print("="*70)
