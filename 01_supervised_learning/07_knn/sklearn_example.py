"""
K-Nearest Neighbors using scikit-learn
Comparison with from-scratch implementation
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import make_classification, make_regression
from sklearn.metrics import accuracy_score, classification_report, r2_score

# Import our implementation
from implementation import KNNClassifier, KNNRegressor


def compare_implementations_classification():
    """Compare sklearn and custom KNN classification."""
    print("="*70)
    print("COMPARING IMPLEMENTATIONS: CLASSIFICATION")
    print("="*70)
    
    # Generate data
    X, y = make_classification(
        n_samples=300,
        n_features=4,
        n_informative=3,
        n_redundant=1,
        n_clusters_per_class=2,
        random_state=42
    )
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42
    )
    
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Custom implementation
    print("\nCustom KNN Implementation:")
    custom_knn = KNNClassifier(n_neighbors=5, weights='uniform')
    custom_knn.fit(X_train_scaled, y_train)
    custom_pred = custom_knn.predict(X_test_scaled)
    custom_acc = custom_knn.score(X_test_scaled, y_test)
    print(f"  Accuracy: {custom_acc:.4f}")
    
    # Sklearn implementation
    print("\nScikit-learn KNN:")
    sklearn_knn = KNeighborsClassifier(n_neighbors=5, weights='uniform')
    sklearn_knn.fit(X_train_scaled, y_train)
    sklearn_pred = sklearn_knn.predict(X_test_scaled)
    sklearn_acc = sklearn_knn.score(X_test_scaled, y_test)
    print(f"  Accuracy: {sklearn_acc:.4f}")
    
    # Compare predictions
    agreement = np.mean(custom_pred == sklearn_pred)
    print(f"\nPrediction Agreement: {agreement*100:.2f}%")
    
    print("\nClassification Report (sklearn):")
    print(classification_report(y_test, sklearn_pred))


def compare_implementations_regression():
    """Compare sklearn and custom KNN regression."""
    print("\n" + "="*70)
    print("COMPARING IMPLEMENTATIONS: REGRESSION")
    print("="*70)
    
    # Generate data
    X, y = make_regression(
        n_samples=300,
        n_features=3,
        noise=10,
        random_state=42
    )
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42
    )
    
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Custom implementation
    print("\nCustom KNN Regressor:")
    custom_knn = KNNRegressor(n_neighbors=5, weights='uniform')
    custom_knn.fit(X_train_scaled, y_train)
    custom_pred = custom_knn.predict(X_test_scaled)
    custom_r2 = custom_knn.score(X_test_scaled, y_test)
    print(f"  R² Score: {custom_r2:.4f}")
    
    # Sklearn implementation
    print("\nScikit-learn KNN Regressor:")
    sklearn_knn = KNeighborsRegressor(n_neighbors=5, weights='uniform')
    sklearn_knn.fit(X_train_scaled, y_train)
    sklearn_pred = sklearn_knn.predict(X_test_scaled)
    sklearn_r2 = sklearn_knn.score(X_test_scaled, y_test)
    print(f"  R² Score: {sklearn_r2:.4f}")
    
    # Compare predictions
    mse_diff = np.mean((custom_pred - sklearn_pred) ** 2)
    print(f"\nMean Squared Difference: {mse_diff:.6f}")


def hyperparameter_tuning():
    """Demonstrate hyperparameter tuning with sklearn."""
    print("\n" + "="*70)
    print("HYPERPARAMETER TUNING WITH GRID SEARCH")
    print("="*70)
    
    # Generate data
    X, y = make_classification(
        n_samples=500,
        n_features=10,
        n_informative=8,
        n_redundant=2,
        random_state=42
    )
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42
    )
    
    # Scale
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Define parameter grid
    param_grid = {
        'n_neighbors': [3, 5, 7, 9, 11, 15],
        'weights': ['uniform', 'distance'],
        'metric': ['euclidean', 'manhattan', 'minkowski']
    }
    
    # Grid search
    print("\nPerforming Grid Search...")
    knn = KNeighborsClassifier()
    grid_search = GridSearchCV(
        knn, 
        param_grid, 
        cv=5, 
        scoring='accuracy',
        n_jobs=-1,
        verbose=0
    )
    
    grid_search.fit(X_train_scaled, y_train)
    
    print(f"\nBest Parameters: {grid_search.best_params_}")
    print(f"Best CV Score: {grid_search.best_score_:.4f}")
    
    # Test with best model
    best_model = grid_search.best_estimator_
    test_score = best_model.score(X_test_scaled, y_test)
    print(f"Test Score: {test_score:.4f}")


def cross_validation_demo():
    """Demonstrate cross-validation."""
    print("\n" + "="*70)
    print("CROSS-VALIDATION")
    print("="*70)
    
    # Generate data
    X, y = make_classification(
        n_samples=400,
        n_features=6,
        n_informative=5,
        n_redundant=1,
        random_state=42
    )
    
    # Scale
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Try different k values
    k_values = [3, 5, 7, 9, 11, 15, 21]
    
    print("\nK-Fold Cross-Validation (k=5 folds):")
    print(f"{'K':>5} | {'Mean Accuracy':>13} | {'Std Dev':>8}")
    print("-" * 35)
    
    for k in k_values:
        knn = KNeighborsClassifier(n_neighbors=k)
        scores = cross_val_score(knn, X_scaled, y, cv=5, scoring='accuracy')
        print(f"{k:>5} | {scores.mean():>13.4f} | {scores.std():>8.4f}")


def distance_metrics_comparison():
    """Compare different distance metrics."""
    print("\n" + "="*70)
    print("DISTANCE METRICS COMPARISON")
    print("="*70)
    
    # Generate data
    X, y = make_classification(
        n_samples=300,
        n_features=5,
        n_informative=4,
        n_redundant=1,
        random_state=42
    )
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42
    )
    
    # Scale
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Test different metrics
    metrics = ['euclidean', 'manhattan', 'minkowski', 'chebyshev']
    
    print(f"\n{'Metric':>12} | {'Accuracy':>10} | {'CV Score':>10}")
    print("-" * 38)
    
    for metric in metrics:
        knn = KNeighborsClassifier(n_neighbors=5, metric=metric)
        knn.fit(X_train_scaled, y_train)
        
        # Test accuracy
        test_acc = knn.score(X_test_scaled, y_test)
        
        # CV score
        cv_scores = cross_val_score(knn, X_train_scaled, y_train, cv=3)
        cv_mean = cv_scores.mean()
        
        print(f"{metric:>12} | {test_acc:>10.4f} | {cv_mean:>10.4f}")


def weighting_comparison():
    """Compare uniform vs distance weighting."""
    print("\n" + "="*70)
    print("WEIGHTING SCHEMES COMPARISON")
    print("="*70)
    
    # Generate data with some noise
    X, y = make_classification(
        n_samples=400,
        n_features=4,
        n_informative=3,
        n_redundant=1,
        n_clusters_per_class=3,
        flip_y=0.1,  # Add 10% label noise
        random_state=42
    )
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42
    )
    
    # Scale
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Compare uniform vs distance weights
    print(f"\n{'Weights':>12} | {'K':>3} | {'Accuracy':>10}")
    print("-" * 32)
    
    for weights in ['uniform', 'distance']:
        for k in [3, 5, 7, 9]:
            knn = KNeighborsClassifier(n_neighbors=k, weights=weights)
            knn.fit(X_train_scaled, y_train)
            acc = knn.score(X_test_scaled, y_test)
            print(f"{weights:>12} | {k:>3} | {acc:>10.4f}")


if __name__ == "__main__":
    # Run all demonstrations
    compare_implementations_classification()
    compare_implementations_regression()
    hyperparameter_tuning()
    cross_validation_demo()
    distance_metrics_comparison()
    weighting_comparison()
    
    print("\n" + "="*70)
    print("✓ All sklearn KNN demonstrations completed!")
    print("="*70)
