"""
Logistic Regression with Scikit-Learn

This module provides practical examples of using Logistic Regression
with scikit-learn for various classification tasks.

Author: ML Learning Repository
Date: 2024
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report, roc_curve, auc,
    roc_auc_score
)
from sklearn.datasets import make_classification, load_breast_cancer, load_wine
import warnings
warnings.filterwarnings('ignore')


def example_1_basic_binary_classification():
    """Example 1: Basic Binary Classification"""
    print("=" * 80)
    print("EXAMPLE 1: BASIC BINARY CLASSIFICATION")
    print("=" * 80)
    
    # Generate synthetic dataset
    X, y = make_classification(
        n_samples=1000,
        n_features=2,
        n_redundant=0,
        n_informative=2,
        n_clusters_per_class=1,
        random_state=42
    )
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    print(f"Dataset shape: {X.shape}")
    print(f"Training set: {X_train.shape[0]} samples")
    print(f"Test set: {X_test.shape[0]} samples")
    print(f"Class distribution: {np.bincount(y)}")
    
    # Create and train model
    model = LogisticRegression(random_state=42)
    model.fit(X_train, y_train)
    
    # Make predictions
    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)[:, 1]
    
    # Evaluate model
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    roc_auc = roc_auc_score(y_test, y_pred_proba)
    
    print(f"\nModel Performance:")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1-Score: {f1:.4f}")
    print(f"ROC-AUC: {roc_auc:.4f}")
    
    # Plot decision boundary
    plot_decision_boundary_sklearn(X_test, y_test, model, "Basic Binary Classification")
    
    # Plot ROC curve
    plot_roc_curve(y_test, y_pred_proba, "Basic Binary Classification")
    
    return model, X_test, y_test


def example_2_breast_cancer_diagnosis():
    """Example 2: Breast Cancer Diagnosis"""
    print("\n" + "=" * 80)
    print("EXAMPLE 2: BREAST CANCER DIAGNOSIS")
    print("=" * 80)
    
    # Load breast cancer dataset
    data = load_breast_cancer()
    X, y = data.data, data.target
    feature_names = data.feature_names
    
    print(f"Dataset shape: {X.shape}")
    print(f"Features: {X.shape[1]}")
    print(f"Classes: {data.target_names}")
    print(f"Class distribution: {np.bincount(y)}")
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # Feature scaling (recommended for logistic regression)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Train model with regularization
    model = LogisticRegression(
        C=1.0,  # Inverse of regularization strength
        penalty='l2',  # L2 regularization
        solver='liblinear',  # Good for small datasets
        random_state=42
    )
    model.fit(X_train_scaled, y_train)
    
    # Make predictions
    y_pred = model.predict(X_test_scaled)
    y_pred_proba = model.predict_proba(X_test_scaled)[:, 1]
    
    # Evaluate
    print(f"\nModel Performance (with feature scaling):")
    print(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}")
    print(f"Precision: {precision_score(y_test, y_pred):.4f}")
    print(f"Recall: {recall_score(y_test, y_pred):.4f}")
    print(f"F1-Score: {f1_score(y_test, y_pred):.4f}")
    print(f"ROC-AUC: {roc_auc_score(y_test, y_pred_proba):.4f}")
    
    # Feature importance (coefficients)
    feature_importance = np.abs(model.coef_[0])
    top_features_idx = np.argsort(feature_importance)[-10:]
    
    print(f"\nTop 10 Most Important Features:")
    for i, idx in enumerate(reversed(top_features_idx)):
        print(f"{i+1:2d}. {feature_names[idx][:30]:30s} {feature_importance[idx]:.4f}")
    
    # Plot feature importance
    plt.figure(figsize=(12, 8))
    top_10_features = feature_importance[top_features_idx]
    top_10_names = [feature_names[i] for i in top_features_idx]
    
    plt.barh(range(len(top_10_features)), top_10_features)
    plt.yticks(range(len(top_10_features)), top_10_names)
    plt.xlabel('Coefficient Magnitude')
    plt.title('Top 10 Feature Importance (Breast Cancer Diagnosis)')
    plt.tight_layout()
    plt.show()
    
    # Confusion matrix
    plot_confusion_matrix(y_test, y_pred, ['Malignant', 'Benign'], 
                         "Breast Cancer Diagnosis")
    
    return model, scaler, X_test_scaled, y_test


def example_3_multiclass_classification():
    """Example 3: Multi-class Classification"""
    print("\n" + "=" * 80)
    print("EXAMPLE 3: MULTI-CLASS CLASSIFICATION (Wine Dataset)")
    print("=" * 80)
    
    # Load wine dataset
    data = load_wine()
    X, y = data.data, data.target
    
    print(f"Dataset shape: {X.shape}")
    print(f"Classes: {data.target_names}")
    print(f"Class distribution: {np.bincount(y)}")
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # Feature scaling
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Train model for multiclass
    model = LogisticRegression(
        multi_class='ovr',  # One-vs-Rest
        solver='liblinear',
        random_state=42
    )
    model.fit(X_train_scaled, y_train)
    
    # Predictions
    y_pred = model.predict(X_test_scaled)
    
    # Evaluate
    accuracy = accuracy_score(y_test, y_pred)
    print(f"\nMulti-class Classification Performance:")
    print(f"Accuracy: {accuracy:.4f}")
    
    # Detailed classification report
    print(f"\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=data.target_names))
    
    # Confusion matrix
    plot_confusion_matrix(y_test, y_pred, data.target_names, 
                         "Wine Classification")
    
    return model, scaler, X_test_scaled, y_test


def example_4_hyperparameter_tuning():
    """Example 4: Hyperparameter Tuning"""
    print("\n" + "=" * 80)
    print("EXAMPLE 4: HYPERPARAMETER TUNING")
    print("=" * 80)
    
    # Generate dataset
    X, y = make_classification(
        n_samples=2000,
        n_features=20,
        n_informative=10,
        n_redundant=10,
        n_clusters_per_class=1,
        random_state=42
    )
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # Feature scaling
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Define parameter grid
    param_grid = {
        'C': [0.01, 0.1, 1, 10, 100],  # Regularization strength
        'penalty': ['l1', 'l2'],       # Regularization type
        'solver': ['liblinear']        # Solver that supports both L1 and L2
    }
    
    # Grid search with cross-validation
    grid_search = GridSearchCV(
        LogisticRegression(random_state=42),
        param_grid,
        cv=5,
        scoring='accuracy',
        n_jobs=-1,
        verbose=1
    )
    
    print("Performing grid search...")
    grid_search.fit(X_train_scaled, y_train)
    
    print(f"\nBest parameters: {grid_search.best_params_}")
    print(f"Best cross-validation score: {grid_search.best_score_:.4f}")
    
    # Test best model
    best_model = grid_search.best_estimator_
    y_pred = best_model.predict(X_test_scaled)
    test_accuracy = accuracy_score(y_test, y_pred)
    
    print(f"Test accuracy with best model: {test_accuracy:.4f}")
    
    # Compare different C values
    plot_regularization_path(X_train_scaled, y_train, X_test_scaled, y_test)
    
    return best_model, scaler


def example_5_cross_validation():
    """Example 5: Cross-Validation"""
    print("\n" + "=" * 80)
    print("EXAMPLE 5: CROSS-VALIDATION")
    print("=" * 80)
    
    # Load breast cancer dataset
    data = load_breast_cancer()
    X, y = data.data, data.target
    
    # Feature scaling
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Create model
    model = LogisticRegression(random_state=42)
    
    # Perform cross-validation
    cv_scores = cross_val_score(model, X_scaled, y, cv=10, scoring='accuracy')
    
    print(f"10-Fold Cross-Validation Results:")
    print(f"Individual scores: {cv_scores}")
    print(f"Mean accuracy: {cv_scores.mean():.4f} (+/- {cv_scores.std() * 2:.4f})")
    print(f"Min accuracy: {cv_scores.min():.4f}")
    print(f"Max accuracy: {cv_scores.max():.4f}")
    
    # Compare different scoring metrics
    scoring_metrics = ['accuracy', 'precision', 'recall', 'f1']
    
    print(f"\nCross-validation with different metrics:")
    for metric in scoring_metrics:
        scores = cross_val_score(model, X_scaled, y, cv=5, scoring=metric)
        print(f"{metric:10s}: {scores.mean():.4f} (+/- {scores.std() * 2:.4f})")
    
    return model, cv_scores


def plot_decision_boundary_sklearn(X, y, model, title):
    """Plot decision boundary for 2D data"""
    if X.shape[1] != 2:
        return
    
    h = 0.02
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                         np.arange(y_min, y_max, h))
    
    Z = model.predict_proba(np.c_[xx.ravel(), yy.ravel()])[:, 1]
    Z = Z.reshape(xx.shape)
    
    plt.figure(figsize=(12, 8))
    plt.contourf(xx, yy, Z, levels=50, alpha=0.6, cmap='RdYlBu')
    plt.colorbar(label='Probability')
    plt.contour(xx, yy, Z, levels=[0.5], colors='black', linestyles='--', linewidths=2)
    
    scatter = plt.scatter(X[:, 0], X[:, 1], c=y, cmap='RdYlBu', edgecolors='black')
    plt.colorbar(scatter, label='Class')
    
    plt.title(title)
    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')
    plt.show()


def plot_roc_curve(y_true, y_scores, title):
    """Plot ROC curve"""
    fpr, tpr, _ = roc_curve(y_true, y_scores)
    roc_auc = auc(fpr, tpr)
    
    plt.figure(figsize=(10, 8))
    plt.plot(fpr, tpr, color='darkorange', lw=2, 
             label=f'ROC curve (AUC = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(f'ROC Curve - {title}')
    plt.legend(loc="lower right")
    plt.grid(alpha=0.3)
    plt.show()


def plot_confusion_matrix(y_true, y_pred, class_names, title):
    """Plot confusion matrix"""
    cm = confusion_matrix(y_true, y_pred)
    
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=class_names, yticklabels=class_names)
    plt.title(f'Confusion Matrix - {title}')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.show()


def plot_regularization_path(X_train, y_train, X_test, y_test):
    """Plot how regularization affects performance"""
    C_values = np.logspace(-4, 4, 50)
    train_scores = []
    test_scores = []
    
    for C in C_values:
        model = LogisticRegression(C=C, random_state=42, max_iter=1000)
        model.fit(X_train, y_train)
        
        train_scores.append(model.score(X_train, y_train))
        test_scores.append(model.score(X_test, y_test))
    
    plt.figure(figsize=(12, 8))
    plt.semilogx(C_values, train_scores, 'b-', label='Training accuracy')
    plt.semilogx(C_values, test_scores, 'r-', label='Test accuracy')
    plt.xlabel('C (Inverse of regularization strength)')
    plt.ylabel('Accuracy')
    plt.title('Regularization Path')
    plt.legend()
    plt.grid(alpha=0.3)
    plt.show()


def comprehensive_evaluation(model, X_test, y_test, title="Model"):
    """Comprehensive model evaluation"""
    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)[:, 1] if len(np.unique(y_test)) == 2 else None
    
    print(f"\n{title} Evaluation:")
    print("-" * 40)
    print(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}")
    
    if len(np.unique(y_test)) == 2:  # Binary classification
        print(f"Precision: {precision_score(y_test, y_pred):.4f}")
        print(f"Recall: {recall_score(y_test, y_pred):.4f}")
        print(f"F1-Score: {f1_score(y_test, y_pred):.4f}")
        if y_pred_proba is not None:
            print(f"ROC-AUC: {roc_auc_score(y_test, y_pred_proba):.4f}")
    else:  # Multi-class classification
        print(f"Precision (weighted): {precision_score(y_test, y_pred, average='weighted'):.4f}")
        print(f"Recall (weighted): {recall_score(y_test, y_pred, average='weighted'):.4f}")
        print(f"F1-Score (weighted): {f1_score(y_test, y_pred, average='weighted'):.4f}")


def main():
    """Run all examples"""
    print("LOGISTIC REGRESSION WITH SCIKIT-LEARN")
    print("=" * 80)
    print("This script demonstrates various aspects of Logistic Regression:")
    print("1. Basic binary classification")
    print("2. Real-world application (breast cancer)")
    print("3. Multi-class classification") 
    print("4. Hyperparameter tuning")
    print("5. Cross-validation")
    
    # Run examples
    model1, X_test1, y_test1 = example_1_basic_binary_classification()
    model2, scaler2, X_test2, y_test2 = example_2_breast_cancer_diagnosis()
    model3, scaler3, X_test3, y_test3 = example_3_multiclass_classification()
    model4, scaler4 = example_4_hyperparameter_tuning()
    model5, cv_scores = example_5_cross_validation()
    
    print("\n" + "=" * 80)
    print("KEY TAKEAWAYS:")
    print("=" * 80)
    print("✅ Logistic Regression is great for:")
    print("   - Binary and multi-class classification")
    print("   - When you need probability estimates")
    print("   - Interpretable models")
    print("   - Baseline models")
    
    print("\n⚠️  Important considerations:")
    print("   - Feature scaling often improves performance")
    print("   - Regularization helps prevent overfitting")
    print("   - Cross-validation gives reliable performance estimates")
    print("   - Choose appropriate solvers for your data size")
    
    print("\n🔧 Hyperparameter tips:")
    print("   - C: Lower values = more regularization")
    print("   - penalty: 'l1' for feature selection, 'l2' for general use")
    print("   - solver: 'liblinear' for small data, 'lbfgs' for large data")
    print("   - max_iter: Increase if convergence warnings appear")


if __name__ == "__main__":
    main()