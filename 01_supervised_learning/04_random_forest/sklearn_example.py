"""
Random Forest Examples using Scikit-Learn
==========================================

This module demonstrates practical applications of Random Forest using scikit-learn:
- Classification examples with different datasets
- Regression examples
- Hyperparameter tuning with Grid Search and Random Search
- Feature importance analysis
- Cross-validation and model evaluation
- Handling imbalanced datasets
- Out-of-bag (OOB) scoring

Author: ML Learning Project
Date: 2024
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.model_selection import (train_test_split, cross_val_score, 
                                   GridSearchCV, RandomizedSearchCV)
from sklearn.metrics import (classification_report, confusion_matrix, 
                           mean_squared_error, r2_score, roc_auc_score, roc_curve)
from sklearn.datasets import (make_classification, make_regression, 
                            load_iris, load_wine, load_breast_cancer,
                            fetch_california_housing)
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.feature_selection import SelectFromModel
import warnings
warnings.filterwarnings('ignore')

def basic_classification_example():
    """Basic Random Forest Classification Example"""
    print("🎯 Random Forest Classification - Basic Example")
    print("=" * 60)
    
    # Load Iris dataset
    iris = load_iris()
    X, y = iris.data, iris.target
    
    print(f"Dataset: {iris.data.shape[0]} samples, {iris.data.shape[1]} features")
    print(f"Classes: {iris.target_names}")
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y
    )
    
    # Create and train Random Forest
    rf = RandomForestClassifier(
        n_estimators=100,
        random_state=42,
        oob_score=True  # Calculate out-of-bag score
    )
    
    rf.fit(X_train, y_train)
    
    # Make predictions
    y_pred = rf.predict(X_test)
    y_proba = rf.predict_proba(X_test)
    
    # Evaluate model
    accuracy = rf.score(X_test, y_test)
    train_accuracy = rf.score(X_train, y_train)
    
    print(f"\nResults:")
    print(f"Training Accuracy: {train_accuracy:.4f}")
    print(f"Test Accuracy: {accuracy:.4f}")
    print(f"OOB Score: {rf.oob_score_:.4f}")
    
    # Feature importance
    print(f"\nFeature Importance:")
    for i, importance in enumerate(rf.feature_importances_):
        print(f"{iris.feature_names[i]}: {importance:.4f}")
    
    # Classification report
    print(f"\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=iris.target_names))
    
    return rf, X_test, y_test, y_pred

def advanced_classification_example():
    """Advanced Random Forest Classification with Breast Cancer Dataset"""
    print("\n🏥 Advanced Classification - Breast Cancer Dataset")
    print("=" * 60)
    
    # Load breast cancer dataset
    cancer = load_breast_cancer()
    X, y = cancer.data, cancer.target
    
    print(f"Dataset: {X.shape[0]} samples, {X.shape[1]} features")
    print(f"Classes: {cancer.target_names}")
    print(f"Class distribution: {np.bincount(y)}")
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # Create Random Forest with optimized parameters
    rf = RandomForestClassifier(
        n_estimators=200,
        max_depth=10,
        min_samples_split=5,
        min_samples_leaf=2,
        max_features='sqrt',
        bootstrap=True,
        oob_score=True,
        random_state=42,
        n_jobs=-1  # Use all available cores
    )
    
    rf.fit(X_train, y_train)
    
    # Predictions
    y_pred = rf.predict(X_test)
    y_proba = rf.predict_proba(X_test)[:, 1]  # Probability of positive class
    
    # Metrics
    accuracy = rf.score(X_test, y_test)
    auc_score = roc_auc_score(y_test, y_proba)
    
    print(f"\nResults:")
    print(f"Test Accuracy: {accuracy:.4f}")
    print(f"OOB Score: {rf.oob_score_:.4f}")
    print(f"ROC AUC Score: {auc_score:.4f}")
    
    # Feature importance analysis
    feature_importance = pd.DataFrame({
        'feature': cancer.feature_names,
        'importance': rf.feature_importances_
    }).sort_values('importance', ascending=False)
    
    print(f"\nTop 10 Most Important Features:")
    print(feature_importance.head(10))
    
    # Plot feature importance
    plt.figure(figsize=(10, 8))
    plt.barh(range(10), feature_importance.head(10)['importance'])
    plt.yticks(range(10), feature_importance.head(10)['feature'])
    plt.xlabel('Feature Importance')
    plt.title('Top 10 Feature Importances - Breast Cancer Dataset')
    plt.tight_layout()
    plt.show()
    
    # ROC Curve
    fpr, tpr, _ = roc_curve(y_test, y_proba)
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, label=f'Random Forest (AUC = {auc_score:.3f})')
    plt.plot([0, 1], [0, 1], 'k--', label='Random Guess')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve - Breast Cancer Classification')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.show()
    
    return rf, feature_importance

def regression_example():
    """Random Forest Regression Example"""
    print("\n🏠 Random Forest Regression - Housing Prices")
    print("=" * 60)
    
    # Load California housing dataset
    housing = fetch_california_housing()
    X, y = housing.data, housing.target
    
    print(f"Dataset: {X.shape[0]} samples, {X.shape[1]} features")
    print(f"Target: Median house value (in hundreds of thousands of dollars)")
    print(f"Features: {housing.feature_names}")
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    # Create Random Forest Regressor
    rf_reg = RandomForestRegressor(
        n_estimators=100,
        max_depth=15,
        min_samples_split=10,
        min_samples_leaf=4,
        max_features='auto',  # For regression, auto = n_features
        bootstrap=True,
        oob_score=True,
        random_state=42,
        n_jobs=-1
    )
    
    rf_reg.fit(X_train, y_train)
    
    # Predictions
    y_pred = rf_reg.predict(X_test)
    
    # Metrics
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test, y_pred)
    
    print(f"\nResults:")
    print(f"Mean Squared Error: {mse:.4f}")
    print(f"Root Mean Squared Error: {rmse:.4f}")
    print(f"R² Score: {r2:.4f}")
    print(f"OOB Score: {rf_reg.oob_score_:.4f}")
    
    # Feature importance
    feature_importance = pd.DataFrame({
        'feature': housing.feature_names,
        'importance': rf_reg.feature_importances_
    }).sort_values('importance', ascending=False)
    
    print(f"\nFeature Importance:")
    print(feature_importance)
    
    # Plot predictions vs actual
    plt.figure(figsize=(10, 6))
    plt.subplot(1, 2, 1)
    plt.scatter(y_test, y_pred, alpha=0.5)
    plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
    plt.xlabel('Actual Values')
    plt.ylabel('Predicted Values')
    plt.title('Predictions vs Actual Values')
    
    # Plot residuals
    plt.subplot(1, 2, 2)
    residuals = y_test - y_pred
    plt.scatter(y_pred, residuals, alpha=0.5)
    plt.axhline(y=0, color='r', linestyle='--')
    plt.xlabel('Predicted Values')
    plt.ylabel('Residuals')
    plt.title('Residual Plot')
    
    plt.tight_layout()
    plt.show()
    
    return rf_reg, feature_importance

def hyperparameter_tuning_example():
    """Hyperparameter Tuning with Grid Search and Random Search"""
    print("\n🔧 Hyperparameter Tuning - Grid Search vs Random Search")
    print("=" * 60)
    
    # Create synthetic dataset
    X, y = make_classification(
        n_samples=1000,
        n_features=20,
        n_informative=15,
        n_redundant=5,
        n_classes=2,
        random_state=42
    )
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # Define parameter grid
    param_grid = {
        'n_estimators': [50, 100, 200],
        'max_depth': [5, 10, 15, None],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4],
        'max_features': ['sqrt', 'log2', None]
    }
    
    # Grid Search
    print("Running Grid Search...")
    rf = RandomForestClassifier(random_state=42)
    grid_search = GridSearchCV(
        rf, param_grid, cv=5, scoring='accuracy', n_jobs=-1, verbose=1
    )
    grid_search.fit(X_train, y_train)
    
    print(f"Grid Search Best Score: {grid_search.best_score_:.4f}")
    print(f"Grid Search Best Parameters: {grid_search.best_params_}")
    
    # Random Search
    print("\nRunning Random Search...")
    param_dist = {
        'n_estimators': [50, 100, 200, 300],
        'max_depth': [5, 10, 15, 20, None],
        'min_samples_split': [2, 5, 10, 15],
        'min_samples_leaf': [1, 2, 4, 6],
        'max_features': ['sqrt', 'log2', None]
    }
    
    random_search = RandomizedSearchCV(
        rf, param_dist, n_iter=50, cv=5, scoring='accuracy', 
        n_jobs=-1, random_state=42, verbose=1
    )
    random_search.fit(X_train, y_train)
    
    print(f"Random Search Best Score: {random_search.best_score_:.4f}")
    print(f"Random Search Best Parameters: {random_search.best_params_}")
    
    # Test both models
    grid_pred = grid_search.predict(X_test)
    random_pred = random_search.predict(X_test)
    
    grid_accuracy = np.mean(grid_pred == y_test)
    random_accuracy = np.mean(random_pred == y_test)
    
    print(f"\nTest Set Results:")
    print(f"Grid Search Accuracy: {grid_accuracy:.4f}")
    print(f"Random Search Accuracy: {random_accuracy:.4f}")
    
    return grid_search, random_search

def feature_selection_example():
    """Feature Selection with Random Forest"""
    print("\n🎯 Feature Selection with Random Forest")
    print("=" * 60)
    
    # Create dataset with many irrelevant features
    X, y = make_classification(
        n_samples=1000,
        n_features=50,
        n_informative=10,
        n_redundant=10,
        n_clusters_per_class=1,
        random_state=42
    )
    
    print(f"Original dataset: {X.shape[1]} features")
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # Train Random Forest for feature selection
    rf = RandomForestClassifier(n_estimators=100, random_state=42)
    rf.fit(X_train, y_train)
    
    # Feature selection based on importance
    selector = SelectFromModel(rf, threshold='mean')
    X_train_selected = selector.fit_transform(X_train, y_train)
    X_test_selected = selector.transform(X_test)
    
    print(f"Selected features: {X_train_selected.shape[1]}")
    print(f"Selected feature indices: {np.where(selector.get_support())[0]}")
    
    # Train models with original and selected features
    # Original features
    rf_original = RandomForestClassifier(n_estimators=100, random_state=42)
    rf_original.fit(X_train, y_train)
    original_score = rf_original.score(X_test, y_test)
    
    # Selected features
    rf_selected = RandomForestClassifier(n_estimators=100, random_state=42)
    rf_selected.fit(X_train_selected, y_train)
    selected_score = rf_selected.score(X_test_selected, y_test)
    
    print(f"\nResults:")
    print(f"Original features accuracy: {original_score:.4f}")
    print(f"Selected features accuracy: {selected_score:.4f}")
    print(f"Feature reduction: {(1 - X_train_selected.shape[1] / X.shape[1]) * 100:.1f}%")
    
    # Plot feature importance
    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    plt.bar(range(len(rf.feature_importances_)), rf.feature_importances_)
    plt.axhline(y=selector.threshold_, color='r', linestyle='--', 
                label=f'Threshold: {selector.threshold_:.3f}')
    plt.xlabel('Feature Index')
    plt.ylabel('Importance')
    plt.title('Feature Importance (All Features)')
    plt.legend()
    
    plt.subplot(1, 2, 2)
    selected_importances = rf.feature_importances_[selector.get_support()]
    selected_indices = np.where(selector.get_support())[0]
    plt.bar(range(len(selected_importances)), selected_importances)
    plt.xlabel('Selected Feature Index')
    plt.ylabel('Importance')
    plt.title('Feature Importance (Selected Features)')
    
    plt.tight_layout()
    plt.show()
    
    return rf, selector, selected_indices

def cross_validation_example():
    """Cross-Validation Analysis"""
    print("\n📊 Cross-Validation Analysis")
    print("=" * 60)
    
    # Load wine dataset
    wine = load_wine()
    X, y = wine.data, wine.target
    
    print(f"Dataset: Wine classification")
    print(f"Samples: {X.shape[0]}, Features: {X.shape[1]}, Classes: {len(wine.target_names)}")
    
    # Different Random Forest configurations
    configs = {
        'Default': RandomForestClassifier(random_state=42),
        'Deep Trees': RandomForestClassifier(max_depth=None, min_samples_split=2, 
                                           min_samples_leaf=1, random_state=42),
        'Shallow Trees': RandomForestClassifier(max_depth=5, min_samples_split=10, 
                                              min_samples_leaf=5, random_state=42),
        'More Trees': RandomForestClassifier(n_estimators=500, random_state=42),
        'Feature Limited': RandomForestClassifier(max_features='log2', random_state=42)
    }
    
    # Perform cross-validation for each configuration
    results = {}
    for name, model in configs.items():
        scores = cross_val_score(model, X, y, cv=10, scoring='accuracy')
        results[name] = {
            'mean': scores.mean(),
            'std': scores.std(),
            'scores': scores
        }
        print(f"{name}: {scores.mean():.4f} (+/- {scores.std() * 2:.4f})")
    
    # Plot results
    plt.figure(figsize=(12, 6))
    names = list(results.keys())
    means = [results[name]['mean'] for name in names]
    stds = [results[name]['std'] for name in names]
    
    plt.subplot(1, 2, 1)
    bars = plt.bar(names, means, yerr=stds, capsize=5, alpha=0.7)
    plt.ylabel('Cross-Validation Accuracy')
    plt.title('Model Comparison - Mean CV Accuracy')
    plt.xticks(rotation=45)
    
    # Box plot
    plt.subplot(1, 2, 2)
    scores_list = [results[name]['scores'] for name in names]
    plt.boxplot(scores_list, labels=names)
    plt.ylabel('Cross-Validation Accuracy')
    plt.title('Model Comparison - CV Score Distribution')
    plt.xticks(rotation=45)
    
    plt.tight_layout()
    plt.show()
    
    return results

def imbalanced_dataset_example():
    """Handling Imbalanced Datasets"""
    print("\n⚖️ Handling Imbalanced Datasets")
    print("=" * 60)
    
    # Create imbalanced dataset
    X, y = make_classification(
        n_samples=1000,
        n_features=10,
        n_informative=8,
        n_classes=2,
        weights=[0.9, 0.1],  # 90% class 0, 10% class 1
        random_state=42
    )
    
    print(f"Class distribution: {np.bincount(y)}")
    print(f"Imbalance ratio: {np.bincount(y)[0] / np.bincount(y)[1]:.1f}:1")
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # Different approaches to handle imbalance
    models = {
        'Default RF': RandomForestClassifier(random_state=42),
        'Balanced RF': RandomForestClassifier(class_weight='balanced', random_state=42),
        'Balanced Subsample RF': RandomForestClassifier(class_weight='balanced_subsample', 
                                                       random_state=42)
    }
    
    results = {}
    for name, model in models.items():
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        y_proba = model.predict_proba(X_test)[:, 1]
        
        accuracy = np.mean(y_pred == y_test)
        auc = roc_auc_score(y_test, y_proba)
        
        results[name] = {
            'accuracy': accuracy,
            'auc': auc,
            'y_pred': y_pred,
            'y_proba': y_proba
        }
        
        print(f"\n{name}:")
        print(f"  Accuracy: {accuracy:.4f}")
        print(f"  ROC AUC: {auc:.4f}")
        print("  Classification Report:")
        print(classification_report(y_test, y_pred, indent=4))
    
    # Plot ROC curves
    plt.figure(figsize=(10, 6))
    plt.subplot(1, 2, 1)
    for name, result in results.items():
        fpr, tpr, _ = roc_curve(y_test, result['y_proba'])
        plt.plot(fpr, tpr, label=f"{name} (AUC: {result['auc']:.3f})")
    
    plt.plot([0, 1], [0, 1], 'k--', alpha=0.5)
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curves - Imbalanced Dataset')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Plot confusion matrices
    plt.subplot(1, 2, 2)
    for i, (name, result) in enumerate(results.items()):
        cm = confusion_matrix(y_test, result['y_pred'])
        print(f"\n{name} Confusion Matrix:")
        print(cm)
    
    plt.text(0.1, 0.9, 'Confusion Matrices printed to console', 
             transform=plt.gca().transAxes, fontsize=12)
    plt.axis('off')
    
    plt.tight_layout()
    plt.show()
    
    return results

def oob_score_analysis():
    """Out-of-Bag (OOB) Score Analysis"""
    print("\n📈 Out-of-Bag (OOB) Score Analysis")
    print("=" * 60)
    
    # Generate dataset
    X, y = make_classification(n_samples=500, n_features=10, n_classes=2, random_state=42)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Analyze OOB score vs number of estimators
    n_estimators_range = range(10, 201, 10)
    oob_scores = []
    test_scores = []
    
    for n_est in n_estimators_range:
        rf = RandomForestClassifier(
            n_estimators=n_est,
            oob_score=True,
            random_state=42
        )
        rf.fit(X_train, y_train)
        
        oob_scores.append(rf.oob_score_)
        test_scores.append(rf.score(X_test, y_test))
    
    # Plot results
    plt.figure(figsize=(10, 6))
    plt.plot(n_estimators_range, oob_scores, label='OOB Score', marker='o')
    plt.plot(n_estimators_range, test_scores, label='Test Score', marker='s')
    plt.xlabel('Number of Estimators')
    plt.ylabel('Accuracy Score')
    plt.title('OOB Score vs Test Score')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.show()
    
    print(f"Best OOB Score: {max(oob_scores):.4f}")
    print(f"Best Test Score: {max(test_scores):.4f}")
    print(f"Correlation between OOB and Test scores: {np.corrcoef(oob_scores, test_scores)[0,1]:.4f}")
    
    return n_estimators_range, oob_scores, test_scores

def main():
    """Run all Random Forest examples"""
    print("🌲🌲🌲 Random Forest Examples with Scikit-Learn 🌲🌲🌲")
    print("=" * 80)
    
    # Run examples
    basic_classification_example()
    advanced_classification_example()
    regression_example()
    hyperparameter_tuning_example()
    feature_selection_example()
    cross_validation_example()
    imbalanced_dataset_example()
    oob_score_analysis()
    
    print("\n" + "=" * 80)
    print("🎉 All Random Forest Examples Completed!")
    print("\nKey Concepts Demonstrated:")
    print("✅ Basic and advanced classification")
    print("✅ Regression with feature importance")
    print("✅ Hyperparameter tuning (Grid Search & Random Search)")
    print("✅ Feature selection using Random Forest")
    print("✅ Cross-validation for model evaluation")
    print("✅ Handling imbalanced datasets")
    print("✅ Out-of-bag (OOB) score analysis")
    print("✅ ROC curves and performance metrics")

if __name__ == "__main__":
    main()