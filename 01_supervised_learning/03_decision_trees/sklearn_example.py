"""
Decision Trees with Scikit-Learn

This module provides practical examples of using Decision Trees
with scikit-learn for various classification and regression tasks.

Author: ML Learning Repository
Date: 2024
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor, plot_tree, export_text
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report, mean_squared_error,
    r2_score, mean_absolute_error
)
from sklearn.datasets import (
    make_classification, load_iris, load_wine, 
    load_boston, make_regression
)
import warnings
warnings.filterwarnings('ignore')


def example_1_basic_classification():
    """Example 1: Basic Classification with Iris Dataset"""
    print("=" * 80)
    print("EXAMPLE 1: BASIC CLASSIFICATION - IRIS DATASET")
    print("=" * 80)
    
    # Load iris dataset
    iris = load_iris()
    X, y = iris.data, iris.target
    feature_names = iris.feature_names
    target_names = iris.target_names
    
    print(f"Dataset shape: {X.shape}")
    print(f"Features: {feature_names}")
    print(f"Classes: {target_names}")
    print(f"Class distribution: {np.bincount(y)}")
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # Create and train decision tree
    dt = DecisionTreeClassifier(
        criterion='gini',
        max_depth=3,
        min_samples_split=2,
        min_samples_leaf=1,
        random_state=42
    )
    
    dt.fit(X_train, y_train)
    
    # Make predictions
    y_pred = dt.predict(X_test)
    y_pred_proba = dt.predict_proba(X_test)
    
    # Evaluate
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average='weighted')
    recall = recall_score(y_test, y_pred, average='weighted')
    f1 = f1_score(y_test, y_pred, average='weighted')
    
    print(f"\nModel Performance:")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1-Score: {f1:.4f}")
    
    print(f"\nTree Information:")
    print(f"Tree depth: {dt.get_depth()}")
    print(f"Number of leaves: {dt.get_n_leaves()}")
    print(f"Number of nodes: {dt.tree_.node_count}")
    
    # Feature importance
    print(f"\nFeature Importance:")
    feature_importance = dt.feature_importances_
    for i, importance in enumerate(feature_importance):
        print(f"{feature_names[i]:20s}: {importance:.4f}")
    
    # Plot feature importance
    plt.figure(figsize=(10, 6))
    plt.bar(feature_names, feature_importance)
    plt.title('Feature Importance - Iris Classification')
    plt.xlabel('Features')
    plt.ylabel('Importance')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()
    
    # Confusion matrix
    plot_confusion_matrix_custom(y_test, y_pred, target_names, 
                                "Iris Classification")
    
    # Visualize tree structure
    plt.figure(figsize=(20, 10))
    plot_tree(dt, feature_names=feature_names, class_names=target_names,
              filled=True, rounded=True, fontsize=10)
    plt.title('Decision Tree Structure - Iris Dataset')
    plt.show()
    
    # Print text representation
    print(f"\nTree Rules (Text Format):")
    tree_rules = export_text(dt, feature_names=feature_names)
    print(tree_rules[:500] + "..." if len(tree_rules) > 500 else tree_rules)
    
    return dt, X_test, y_test


def example_2_wine_quality_classification():
    """Example 2: Wine Quality Classification"""
    print("\n" + "=" * 80)
    print("EXAMPLE 2: WINE QUALITY CLASSIFICATION")
    print("=" * 80)
    
    # Load wine dataset
    wine = load_wine()
    X, y = wine.data, wine.target
    
    print(f"Dataset shape: {X.shape}")
    print(f"Classes: {wine.target_names}")
    print(f"Class distribution: {np.bincount(y)}")
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # Train decision tree with different criteria
    criteria = ['gini', 'entropy']
    results = {}
    
    for criterion in criteria:
        dt = DecisionTreeClassifier(
            criterion=criterion,
            max_depth=5,
            min_samples_split=5,
            min_samples_leaf=2,
            random_state=42
        )
        
        dt.fit(X_train, y_train)
        y_pred = dt.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        
        results[criterion] = {
            'model': dt,
            'accuracy': accuracy,
            'depth': dt.get_depth(),
            'leaves': dt.get_n_leaves()
        }
        
        print(f"\n{criterion.upper()} Criterion:")
        print(f"Accuracy: {accuracy:.4f}")
        print(f"Tree depth: {dt.get_depth()}")
        print(f"Number of leaves: {dt.get_n_leaves()}")
    
    # Compare criteria
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    accuracies = [results[c]['accuracy'] for c in criteria]
    plt.bar(criteria, accuracies, color=['skyblue', 'lightcoral'])
    plt.title('Accuracy Comparison')
    plt.ylabel('Accuracy')
    
    plt.subplot(1, 2, 2)
    depths = [results[c]['depth'] for c in criteria]
    leaves = [results[c]['leaves'] for c in criteria]
    
    x = np.arange(len(criteria))
    width = 0.35
    plt.bar(x - width/2, depths, width, label='Tree Depth', color='lightblue')
    plt.bar(x + width/2, leaves, width, label='Number of Leaves', color='lightgreen')
    plt.xlabel('Criterion')
    plt.ylabel('Count')
    plt.title('Tree Complexity Comparison')
    plt.xticks(x, criteria)
    plt.legend()
    
    plt.tight_layout()
    plt.show()
    
    # Detailed classification report for best model
    best_criterion = max(results, key=lambda x: results[x]['accuracy'])
    best_model = results[best_criterion]['model']
    best_pred = best_model.predict(X_test)
    
    print(f"\nDetailed Classification Report ({best_criterion} criterion):")
    print(classification_report(y_test, best_pred, target_names=wine.target_names))
    
    return best_model, X_test, y_test


def example_3_regression_with_boston_housing():
    """Example 3: Regression with Boston Housing Dataset"""
    print("\n" + "=" * 80)
    print("EXAMPLE 3: REGRESSION - BOSTON HOUSING DATASET")
    print("=" * 80)
    
    # Generate regression data (Boston housing is deprecated)
    X, y = make_regression(
        n_samples=506, n_features=13, noise=0.1, random_state=42
    )
    
    # Create feature names
    feature_names = [f'feature_{i}' for i in range(X.shape[1])]
    
    print(f"Dataset shape: {X.shape}")
    print(f"Target range: [{y.min():.2f}, {y.max():.2f}]")
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    # Train regression tree
    dt_reg = DecisionTreeRegressor(
        criterion='squared_error',
        max_depth=5,
        min_samples_split=10,
        min_samples_leaf=5,
        random_state=42
    )
    
    dt_reg.fit(X_train, y_train)
    
    # Make predictions
    y_pred = dt_reg.predict(X_test)
    
    # Evaluate
    mse = mean_squared_error(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    
    print(f"\nRegression Performance:")
    print(f"MSE: {mse:.4f}")
    print(f"MAE: {mae:.4f}")
    print(f"R² Score: {r2:.4f}")
    print(f"Tree depth: {dt_reg.get_depth()}")
    print(f"Number of leaves: {dt_reg.get_n_leaves()}")
    
    # Plot actual vs predicted
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    plt.scatter(y_test, y_pred, alpha=0.7)
    plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
    plt.xlabel('Actual Values')
    plt.ylabel('Predicted Values')
    plt.title('Actual vs Predicted Values')
    plt.grid(True, alpha=0.3)
    
    plt.subplot(1, 2, 2)
    residuals = y_test - y_pred
    plt.scatter(y_pred, residuals, alpha=0.7)
    plt.axhline(y=0, color='r', linestyle='--')
    plt.xlabel('Predicted Values')
    plt.ylabel('Residuals')
    plt.title('Residual Plot')
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    # Feature importance
    top_features = np.argsort(dt_reg.feature_importances_)[-10:]
    plt.figure(figsize=(10, 6))
    plt.barh(range(len(top_features)), dt_reg.feature_importances_[top_features])
    plt.yticks(range(len(top_features)), [feature_names[i] for i in top_features])
    plt.xlabel('Feature Importance')
    plt.title('Top 10 Feature Importance - Regression Tree')
    plt.tight_layout()
    plt.show()
    
    return dt_reg, X_test, y_test


def example_4_hyperparameter_tuning():
    """Example 4: Hyperparameter Tuning"""
    print("\n" + "=" * 80)
    print("EXAMPLE 4: HYPERPARAMETER TUNING")
    print("=" * 80)
    
    # Generate dataset
    X, y = make_classification(
        n_samples=1000, n_features=20, n_informative=10,
        n_redundant=5, n_clusters_per_class=1, random_state=42
    )
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    print(f"Dataset shape: {X.shape}")
    print(f"Class distribution: {np.bincount(y)}")
    
    # Define parameter grid
    param_grid = {
        'max_depth': [3, 5, 7, 10, None],
        'min_samples_split': [2, 5, 10, 20],
        'min_samples_leaf': [1, 2, 5, 10],
        'criterion': ['gini', 'entropy']
    }
    
    # Grid search
    dt = DecisionTreeClassifier(random_state=42)
    grid_search = GridSearchCV(
        dt, param_grid, cv=5, scoring='accuracy',
        n_jobs=-1, verbose=1
    )
    
    print("Performing grid search...")
    grid_search.fit(X_train, y_train)
    
    print(f"\nBest parameters: {grid_search.best_params_}")
    print(f"Best CV score: {grid_search.best_score_:.4f}")
    
    # Test best model
    best_model = grid_search.best_estimator_
    y_pred = best_model.predict(X_test)
    test_accuracy = accuracy_score(y_test, y_pred)
    
    print(f"Test accuracy: {test_accuracy:.4f}")
    
    # Analyze parameter effects
    results_df = pd.DataFrame(grid_search.cv_results_)
    
    # Plot max_depth effect
    plt.figure(figsize=(15, 10))
    
    plt.subplot(2, 3, 1)
    depth_scores = results_df.groupby('param_max_depth')['mean_test_score'].mean()
    depth_scores.plot(kind='bar')
    plt.title('Max Depth Effect')
    plt.xlabel('Max Depth')
    plt.ylabel('CV Score')
    plt.xticks(rotation=45)
    
    plt.subplot(2, 3, 2)
    split_scores = results_df.groupby('param_min_samples_split')['mean_test_score'].mean()
    split_scores.plot(kind='bar')
    plt.title('Min Samples Split Effect')
    plt.xlabel('Min Samples Split')
    plt.ylabel('CV Score')
    
    plt.subplot(2, 3, 3)
    leaf_scores = results_df.groupby('param_min_samples_leaf')['mean_test_score'].mean()
    leaf_scores.plot(kind='bar')
    plt.title('Min Samples Leaf Effect')
    plt.xlabel('Min Samples Leaf')
    plt.ylabel('CV Score')
    
    plt.subplot(2, 3, 4)
    criterion_scores = results_df.groupby('param_criterion')['mean_test_score'].mean()
    criterion_scores.plot(kind='bar')
    plt.title('Criterion Effect')
    plt.xlabel('Criterion')
    plt.ylabel('CV Score')
    
    # Feature importance of best model
    plt.subplot(2, 3, 5)
    top_features = np.argsort(best_model.feature_importances_)[-10:]
    plt.barh(range(len(top_features)), best_model.feature_importances_[top_features])
    plt.yticks(range(len(top_features)), [f'Feature {i}' for i in top_features])
    plt.xlabel('Importance')
    plt.title('Top 10 Feature Importance')
    
    plt.tight_layout()
    plt.show()
    
    return best_model, grid_search


def example_5_overfitting_analysis():
    """Example 5: Overfitting Analysis"""
    print("\n" + "=" * 80)
    print("EXAMPLE 5: OVERFITTING ANALYSIS")
    print("=" * 80)
    
    # Generate dataset
    X, y = make_classification(
        n_samples=500, n_features=10, n_informative=5,
        n_redundant=2, random_state=42
    )
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # Test different max_depth values
    max_depths = range(1, 21)
    train_scores = []
    test_scores = []
    
    for depth in max_depths:
        dt = DecisionTreeClassifier(max_depth=depth, random_state=42)
        dt.fit(X_train, y_train)
        
        train_score = dt.score(X_train, y_train)
        test_score = dt.score(X_test, y_test)
        
        train_scores.append(train_score)
        test_scores.append(test_score)
    
    # Plot learning curve
    plt.figure(figsize=(12, 8))
    
    plt.subplot(2, 2, 1)
    plt.plot(max_depths, train_scores, 'b-o', label='Training Accuracy')
    plt.plot(max_depths, test_scores, 'r-o', label='Test Accuracy')
    plt.xlabel('Max Depth')
    plt.ylabel('Accuracy')
    plt.title('Overfitting Analysis - Max Depth')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Find optimal depth
    test_scores = np.array(test_scores)
    optimal_depth = max_depths[np.argmax(test_scores)]
    print(f"Optimal max_depth: {optimal_depth}")
    print(f"Best test accuracy: {test_scores.max():.4f}")
    
    # Test different min_samples_split values
    min_splits = range(2, 51, 2)
    train_scores_split = []
    test_scores_split = []
    
    for split in min_splits:
        dt = DecisionTreeClassifier(min_samples_split=split, random_state=42)
        dt.fit(X_train, y_train)
        
        train_scores_split.append(dt.score(X_train, y_train))
        test_scores_split.append(dt.score(X_test, y_test))
    
    plt.subplot(2, 2, 2)
    plt.plot(min_splits, train_scores_split, 'b-o', label='Training Accuracy')
    plt.plot(min_splits, test_scores_split, 'r-o', label='Test Accuracy')
    plt.xlabel('Min Samples Split')
    plt.ylabel('Accuracy')
    plt.title('Regularization Effect - Min Samples Split')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Cross-validation with different parameters
    depths_cv = [3, 5, 7, 10, 15, None]
    cv_means = []
    cv_stds = []
    
    for depth in depths_cv:
        dt = DecisionTreeClassifier(max_depth=depth, random_state=42)
        scores = cross_val_score(dt, X_train, y_train, cv=5)
        cv_means.append(scores.mean())
        cv_stds.append(scores.std())
    
    plt.subplot(2, 2, 3)
    depths_str = [str(d) if d is not None else 'None' for d in depths_cv]
    plt.errorbar(range(len(depths_cv)), cv_means, yerr=cv_stds, 
                 fmt='o-', capsize=5)
    plt.xlabel('Max Depth')
    plt.ylabel('CV Accuracy')
    plt.title('Cross-Validation Scores')
    plt.xticks(range(len(depths_cv)), depths_str)
    plt.grid(True, alpha=0.3)
    
    # Tree complexity vs performance
    complexities = []
    performances = []
    
    for depth in [3, 5, 7, 10, 15]:
        dt = DecisionTreeClassifier(max_depth=depth, random_state=42)
        dt.fit(X_train, y_train)
        
        complexity = dt.get_n_leaves()  # Use number of leaves as complexity measure
        performance = dt.score(X_test, y_test)
        
        complexities.append(complexity)
        performances.append(performance)
    
    plt.subplot(2, 2, 4)
    plt.scatter(complexities, performances, s=100, alpha=0.7)
    for i, depth in enumerate([3, 5, 7, 10, 15]):
        plt.annotate(f'depth={depth}', 
                    (complexities[i], performances[i]),
                    xytext=(5, 5), textcoords='offset points')
    plt.xlabel('Tree Complexity (Number of Leaves)')
    plt.ylabel('Test Accuracy')
    plt.title('Complexity vs Performance')
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    return optimal_depth


def plot_confusion_matrix_custom(y_true, y_pred, class_names, title):
    """Plot confusion matrix"""
    cm = confusion_matrix(y_true, y_pred)
    
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=class_names, yticklabels=class_names)
    plt.title(f'Confusion Matrix - {title}')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.show()


def comprehensive_decision_tree_analysis():
    """Comprehensive analysis of decision tree behavior"""
    print("\n" + "=" * 80)
    print("COMPREHENSIVE DECISION TREE ANALYSIS")
    print("=" * 80)
    
    # Generate dataset with different characteristics
    datasets = {
        'Linear': make_classification(n_samples=500, n_features=2, n_redundant=0, 
                                     n_informative=2, n_clusters_per_class=1, random_state=42),
        'Non-linear': make_classification(n_samples=500, n_features=2, n_redundant=0,
                                         n_informative=2, n_clusters_per_class=2, random_state=42),
        'Noisy': make_classification(n_samples=500, n_features=2, n_redundant=0,
                                    n_informative=2, flip_y=0.1, random_state=42)
    }
    
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    axes = axes.ravel()
    
    for idx, (name, (X, y)) in enumerate(datasets.items()):
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.3, random_state=42
        )
        
        # Train decision tree
        dt = DecisionTreeClassifier(max_depth=5, random_state=42)
        dt.fit(X_train, y_train)
        
        # Plot decision boundary
        h = 0.02
        x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
        y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
        xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                             np.arange(y_min, y_max, h))
        
        Z = dt.predict(np.c_[xx.ravel(), yy.ravel()])
        Z = Z.reshape(xx.shape)
        
        # Plot
        axes[idx].contourf(xx, yy, Z, alpha=0.6, cmap='viridis')
        scatter = axes[idx].scatter(X[:, 0], X[:, 1], c=y, cmap='viridis', 
                                   edgecolors='black')
        axes[idx].set_title(f'{name} Data\nAccuracy: {dt.score(X_test, y_test):.3f}')
        axes[idx].set_xlabel('Feature 1')
        axes[idx].set_ylabel('Feature 2')
        
        # Plot tree structure (simplified)
        axes[idx + 3].text(0.1, 0.5, f'Tree Depth: {dt.get_depth()}\n'
                                     f'Leaves: {dt.get_n_leaves()}\n'
                                     f'Nodes: {dt.tree_.node_count}',
                          transform=axes[idx + 3].transAxes, fontsize=12,
                          verticalalignment='center')
        axes[idx + 3].set_title(f'{name} Tree Stats')
        axes[idx + 3].axis('off')
    
    plt.tight_layout()
    plt.show()


def main():
    """Run all examples"""
    print("DECISION TREES WITH SCIKIT-LEARN")
    print("=" * 80)
    print("This script demonstrates various aspects of Decision Trees:")
    print("1. Basic classification")
    print("2. Multi-class classification")
    print("3. Regression")
    print("4. Hyperparameter tuning")
    print("5. Overfitting analysis")
    
    # Run examples
    model1, X_test1, y_test1 = example_1_basic_classification()
    model2, X_test2, y_test2 = example_2_wine_quality_classification()
    model3, X_test3, y_test3 = example_3_regression_with_boston_housing()
    model4, grid_search = example_4_hyperparameter_tuning()
    optimal_depth = example_5_overfitting_analysis()
    
    # Comprehensive analysis
    comprehensive_decision_tree_analysis()
    
    print("\n" + "=" * 80)
    print("KEY TAKEAWAYS:")
    print("=" * 80)
    print("✅ Decision Trees are great for:")
    print("   - Interpretable models")
    print("   - Non-linear relationships")
    print("   - Mixed data types")
    print("   - Feature importance ranking")
    
    print("\n⚠️  Important considerations:")
    print("   - Prone to overfitting")
    print("   - Unstable (high variance)")
    print("   - Biased toward features with more levels")
    print("   - May not capture linear relationships efficiently")
    
    print("\n🔧 Best practices:")
    print("   - Use max_depth to control overfitting")
    print("   - Set min_samples_split and min_samples_leaf")
    print("   - Use cross-validation for parameter selection")
    print("   - Consider ensemble methods for better performance")
    print("   - Visualize trees to understand decisions")


if __name__ == "__main__":
    main()