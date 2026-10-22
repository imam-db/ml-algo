"""
Gradient Boosting: Comparison with Scikit-Learn

This script demonstrates:
1. Comparison between from-scratch and sklearn implementations
2. Hyperparameter tuning examples
3. Feature importance analysis
4. Learning curves and model diagnostics
5. Real-world-like applications

Author: ML-Algo Repository
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import GradientBoostingRegressor as SklearnGBR
from sklearn.ensemble import GradientBoostingClassifier as SklearnGBC
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import mean_squared_error, r2_score, accuracy_score, classification_report
from sklearn.metrics import roc_curve, auc, confusion_matrix
import sys
import os

# Import our implementation
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from implementation import GradientBoostingRegressor, GradientBoostingClassifier


def generate_store_sales_data(n_samples=1000, random_state=42):
    """
    Generate synthetic store sales data for regression
    
    Features:
    - day_of_week: 0-6 (Monday to Sunday)
    - is_weekend: 0 or 1
    - is_holiday: 0 or 1
    - has_promotion: 0 or 1
    - temperature: Celsius
    - competitor_nearby: 0 or 1
    - store_age_years: years since opening
    
    Target: daily_sales in currency units
    """
    np.random.seed(random_state)
    
    day_of_week = np.random.randint(0, 7, n_samples)
    is_weekend = (day_of_week >= 5).astype(int)
    is_holiday = np.random.binomial(1, 0.08, n_samples)
    has_promotion = np.random.binomial(1, 0.15, n_samples)
    temperature = np.random.normal(20, 8, n_samples)
    competitor_nearby = np.random.binomial(1, 0.3, n_samples)
    store_age_years = np.random.uniform(0, 10, n_samples)
    
    # Complex non-linear relationship
    sales = (
        1200 +  # Base sales
        200 * is_weekend +
        400 * is_holiday +
        300 * has_promotion +
        15 * temperature +
        -0.3 * temperature ** 2 +  # Optimal temperature effect
        -150 * competitor_nearby +
        50 * store_age_years +  # Established store bonus
        -2 * store_age_years ** 2 +  # But decline over time
        100 * is_weekend * has_promotion +  # Interaction
        np.random.normal(0, 100, n_samples)  # Noise
    )
    
    X = np.column_stack([
        day_of_week, is_weekend, is_holiday, has_promotion,
        temperature, competitor_nearby, store_age_years
    ])
    
    feature_names = [
        'day_of_week', 'is_weekend', 'is_holiday', 'has_promotion',
        'temperature', 'competitor_nearby', 'store_age_years'
    ]
    
    return X, sales, feature_names


def generate_churn_data(n_samples=2000, random_state=42):
    """
    Generate synthetic customer churn data for classification
    
    Features:
    - tenure_months: how long customer has been with company
    - monthly_charges: monthly bill amount
    - total_charges: total amount charged to customer
    - num_support_calls: number of calls to support
    - contract_type: 0=month-to-month, 1=one-year, 2=two-year
    - payment_auto: 1 if automatic payment, 0 otherwise
    - age: customer age
    
    Target: churned (1) or not (0)
    """
    np.random.seed(random_state)
    
    tenure_months = np.random.exponential(24, n_samples)
    monthly_charges = np.random.uniform(20, 120, n_samples)
    total_charges = tenure_months * monthly_charges + np.random.normal(0, 200, n_samples)
    num_support_calls = np.random.poisson(1.5, n_samples)
    contract_type = np.random.choice([0, 1, 2], n_samples, p=[0.5, 0.3, 0.2])
    payment_auto = np.random.binomial(1, 0.6, n_samples)
    age = np.random.normal(45, 15, n_samples)
    
    # Churn probability model
    logit = (
        -1 +
        -0.06 * tenure_months +  # Longer tenure = less churn
        0.025 * monthly_charges +  # Higher charges = more churn
        0.8 * num_support_calls +  # More issues = more churn
        -0.5 * contract_type +  # Longer contract = less churn
        -0.8 * payment_auto +  # Auto payment = less churn
        -0.02 * age +  # Older customers = less churn
        0.01 * monthly_charges * (contract_type == 0) +  # High cost month-to-month worst
        np.random.normal(0, 0.5, n_samples)
    )
    
    churn_prob = 1 / (1 + np.exp(-logit))
    y = (np.random.random(n_samples) < churn_prob).astype(int)
    
    X = np.column_stack([
        tenure_months, monthly_charges, total_charges,
        num_support_calls, contract_type, payment_auto, age
    ])
    
    feature_names = [
        'tenure_months', 'monthly_charges', 'total_charges',
        'num_support_calls', 'contract_type', 'payment_auto', 'age'
    ]
    
    return X, y, feature_names


def example_1_regression_comparison():
    """Compare our implementation with sklearn for regression"""
    print("=" * 80)
    print("Example 1: Regression - Store Sales Prediction")
    print("=" * 80)
    
    # Generate data
    X, y, feature_names = generate_store_sales_data(n_samples=1000)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    print(f"\nDataset: {len(X_train)} training samples, {len(X_test)} test samples")
    print(f"Features: {', '.join(feature_names)}")
    print(f"Target: daily_sales (mean: ${np.mean(y_train):.2f}, std: ${np.std(y_train):.2f})")
    
    # Our implementation
    print("\n" + "-" * 80)
    print("Training our Gradient Boosting Regressor...")
    our_gb = GradientBoostingRegressor(
        n_estimators=100,
        learning_rate=0.1,
        max_depth=3,
        subsample=0.8,
        random_state=42,
        verbose=0
    )
    our_gb.fit(X_train, y_train)
    our_pred = our_gb.predict(X_test)
    
    our_rmse = np.sqrt(mean_squared_error(y_test, our_pred))
    our_r2 = r2_score(y_test, our_pred)
    
    print(f"Our Implementation - RMSE: ${our_rmse:.2f}, R²: {our_r2:.4f}")
    
    # Sklearn implementation
    print("\nTraining sklearn Gradient Boosting Regressor...")
    sklearn_gb = SklearnGBR(
        n_estimators=100,
        learning_rate=0.1,
        max_depth=3,
        subsample=0.8,
        random_state=42,
        verbose=0
    )
    sklearn_gb.fit(X_train, y_train)
    sklearn_pred = sklearn_gb.predict(X_test)
    
    sklearn_rmse = np.sqrt(mean_squared_error(y_test, sklearn_pred))
    sklearn_r2 = r2_score(y_test, sklearn_pred)
    
    print(f"Sklearn Implementation - RMSE: ${sklearn_rmse:.2f}, R²: {sklearn_r2:.4f}")
    
    # Comparison
    print("\n" + "-" * 80)
    print("Comparison:")
    print(f"  RMSE Difference: ${abs(our_rmse - sklearn_rmse):.2f}")
    print(f"  R² Difference: {abs(our_r2 - sklearn_r2):.4f}")
    
    # Sample predictions
    print("\nSample Predictions (first 5 test samples):")
    print(f"{'Actual':>10} {'Our Pred':>12} {'Sklearn':>12} {'Diff':>10}")
    print("-" * 48)
    for i in range(5):
        diff = abs(our_pred[i] - sklearn_pred[i])
        print(f"${y_test[i]:>9.2f} ${our_pred[i]:>11.2f} ${sklearn_pred[i]:>11.2f} ${diff:>9.2f}")


def example_2_classification_comparison():
    """Compare our implementation with sklearn for classification"""
    print("\n\n" + "=" * 80)
    print("Example 2: Classification - Customer Churn Prediction")
    print("=" * 80)
    
    # Generate data
    X, y, feature_names = generate_churn_data(n_samples=2000)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    print(f"\nDataset: {len(X_train)} training samples, {len(X_test)} test samples")
    print(f"Features: {', '.join(feature_names)}")
    print(f"Target: churned (class distribution: {np.mean(y_train):.2%} churn)")
    
    # Our implementation
    print("\n" + "-" * 80)
    print("Training our Gradient Boosting Classifier...")
    our_gb = GradientBoostingClassifier(
        n_estimators=100,
        learning_rate=0.1,
        max_depth=3,
        subsample=0.8,
        random_state=42,
        verbose=0
    )
    our_gb.fit(X_train, y_train)
    our_pred = our_gb.predict(X_test)
    our_proba = our_gb.predict_proba(X_test)[:, 1]
    
    our_acc = accuracy_score(y_test, our_pred)
    
    print(f"Our Implementation - Accuracy: {our_acc:.4f}")
    print("\nClassification Report (Our Implementation):")
    print(classification_report(y_test, our_pred, target_names=['Not Churn', 'Churn']))
    
    # Sklearn implementation
    print("-" * 80)
    print("Training sklearn Gradient Boosting Classifier...")
    sklearn_gb = SklearnGBC(
        n_estimators=100,
        learning_rate=0.1,
        max_depth=3,
        subsample=0.8,
        random_state=42,
        verbose=0
    )
    sklearn_gb.fit(X_train, y_train)
    sklearn_pred = sklearn_gb.predict(X_test)
    sklearn_proba = sklearn_gb.predict_proba(X_test)[:, 1]
    
    sklearn_acc = accuracy_score(y_test, sklearn_pred)
    
    print(f"Sklearn Implementation - Accuracy: {sklearn_acc:.4f}")
    print("\nClassification Report (Sklearn):")
    print(classification_report(y_test, sklearn_pred, target_names=['Not Churn', 'Churn']))
    
    # Comparison
    print("-" * 80)
    print("Comparison:")
    print(f"  Accuracy Difference: {abs(our_acc - sklearn_acc):.4f}")
    print(f"  Average Probability Difference: {np.mean(np.abs(our_proba - sklearn_proba)):.4f}")


def example_3_feature_importance():
    """Demonstrate feature importance with sklearn"""
    print("\n\n" + "=" * 80)
    print("Example 3: Feature Importance Analysis")
    print("=" * 80)
    
    # Generate data
    X, y, feature_names = generate_store_sales_data(n_samples=1000)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Train model
    print("\nTraining Gradient Boosting Regressor for feature importance...")
    gb = SklearnGBR(
        n_estimators=100,
        learning_rate=0.1,
        max_depth=3,
        random_state=42
    )
    gb.fit(X_train, y_train)
    
    # Get feature importance
    importances = gb.feature_importances_
    indices = np.argsort(importances)[::-1]
    
    print("\nFeature Importance Ranking:")
    print("-" * 50)
    for i, idx in enumerate(indices, 1):
        print(f"{i}. {feature_names[idx]:20s}: {importances[idx]:.4f}")
    
    # Plot
    plt.figure(figsize=(10, 6))
    plt.bar(range(len(importances)), importances[indices])
    plt.xticks(range(len(importances)), [feature_names[i] for i in indices], rotation=45, ha='right')
    plt.xlabel('Features')
    plt.ylabel('Importance')
    plt.title('Feature Importance in Store Sales Prediction')
    plt.tight_layout()
    plt.savefig('gradient_boosting_feature_importance.png', dpi=100, bbox_inches='tight')
    print("\n📊 Feature importance plot saved as 'gradient_boosting_feature_importance.png'")


def example_4_learning_curves():
    """Visualize learning curves and staged predictions"""
    print("\n\n" + "=" * 80)
    print("Example 4: Learning Curves and Model Training Progress")
    print("=" * 80)
    
    # Generate data
    X, y, _ = generate_store_sales_data(n_samples=1000)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    print("\nTraining model and tracking performance...")
    
    # Train our model to get loss history
    our_gb = GradientBoostingRegressor(
        n_estimators=200,
        learning_rate=0.1,
        max_depth=3,
        early_stopping_rounds=20,
        random_state=42,
        verbose=0
    )
    our_gb.fit(X_train, y_train)
    
    # Train sklearn model
    sklearn_gb = SklearnGBR(
        n_estimators=200,
        learning_rate=0.1,
        max_depth=3,
        random_state=42
    )
    sklearn_gb.fit(X_train, y_train)
    
    # Get staged predictions for sklearn
    sklearn_train_scores = []
    sklearn_test_scores = []
    
    for y_pred_train, y_pred_test in zip(
        sklearn_gb.staged_predict(X_train),
        sklearn_gb.staged_predict(X_test)
    ):
        sklearn_train_scores.append(mean_squared_error(y_train, y_pred_train))
        sklearn_test_scores.append(mean_squared_error(y_test, y_pred_test))
    
    # Plot learning curves
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Our implementation
    axes[0].plot(our_gb.train_losses_, label='Training Loss', linewidth=2)
    if our_gb.val_losses_:
        axes[0].plot(our_gb.val_losses_, label='Validation Loss', linewidth=2)
    axes[0].set_xlabel('Boosting Iteration')
    axes[0].set_ylabel('Mean Squared Error')
    axes[0].set_title('Our Implementation: Training Progress')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # Sklearn
    axes[1].plot(sklearn_train_scores, label='Training Loss', linewidth=2)
    axes[1].plot(sklearn_test_scores, label='Test Loss', linewidth=2)
    axes[1].set_xlabel('Boosting Iteration')
    axes[1].set_ylabel('Mean Squared Error')
    axes[1].set_title('Sklearn: Training Progress')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('gradient_boosting_learning_curves.png', dpi=100, bbox_inches='tight')
    print("\n📊 Learning curves saved as 'gradient_boosting_learning_curves.png'")
    
    print(f"\nOur implementation stopped at {our_gb.n_estimators_} trees (early stopping)")
    print(f"Final training loss: {our_gb.train_losses_[-1]:.2f}")
    if our_gb.val_losses_:
        print(f"Final validation loss: {our_gb.val_losses_[-1]:.2f}")


def example_5_hyperparameter_tuning():
    """Demonstrate hyperparameter tuning with GridSearchCV"""
    print("\n\n" + "=" * 80)
    print("Example 5: Hyperparameter Tuning")
    print("=" * 80)
    
    # Generate data
    X, y, _ = generate_churn_data(n_samples=1500)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    print("\nPerforming grid search for optimal hyperparameters...")
    print("(This may take a minute...)")
    
    # Define parameter grid
    param_grid = {
        'n_estimators': [50, 100, 150],
        'learning_rate': [0.05, 0.1, 0.2],
        'max_depth': [2, 3, 4],
        'subsample': [0.8, 1.0]
    }
    
    # Grid search
    gb = SklearnGBC(random_state=42, verbose=0)
    grid_search = GridSearchCV(
        gb,
        param_grid,
        cv=3,
        scoring='accuracy',
        n_jobs=-1,
        verbose=0
    )
    grid_search.fit(X_train, y_train)
    
    print("\nBest Parameters:")
    for param, value in grid_search.best_params_.items():
        print(f"  {param}: {value}")
    
    print(f"\nBest Cross-Validation Score: {grid_search.best_score_:.4f}")
    
    # Test best model
    best_model = grid_search.best_estimator_
    test_score = best_model.score(X_test, y_test)
    print(f"Test Set Accuracy: {test_score:.4f}")
    
    # Compare with default parameters
    default_gb = SklearnGBC(random_state=42)
    default_gb.fit(X_train, y_train)
    default_score = default_gb.score(X_test, y_test)
    
    print(f"\nComparison with default parameters:")
    print(f"  Default Test Accuracy: {default_score:.4f}")
    print(f"  Tuned Test Accuracy: {test_score:.4f}")
    print(f"  Improvement: {(test_score - default_score) * 100:.2f} percentage points")


def example_6_learning_rate_effect():
    """Demonstrate effect of learning rate on bias-variance trade-off"""
    print("\n\n" + "=" * 80)
    print("Example 6: Learning Rate vs Number of Trees Trade-off")
    print("=" * 80)
    
    # Generate data
    X, y, _ = generate_store_sales_data(n_samples=800)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)
    
    print("\nAnalyzing different learning rate configurations...")
    
    learning_rates = [0.01, 0.05, 0.1, 0.2]
    colors = ['blue', 'green', 'orange', 'red']
    
    plt.figure(figsize=(12, 5))
    
    for lr, color in zip(learning_rates, colors):
        gb = SklearnGBR(
            n_estimators=200,
            learning_rate=lr,
            max_depth=3,
            random_state=42
        )
        gb.fit(X_train, y_train)
        
        # Get test scores at each stage
        test_scores = []
        for y_pred in gb.staged_predict(X_test):
            test_scores.append(mean_squared_error(y_test, y_pred))
        
        plt.plot(test_scores, label=f'LR={lr}', color=color, linewidth=2)
        
        # Print final performance
        final_rmse = np.sqrt(test_scores[-1])
        print(f"  Learning Rate {lr:4.2f}: Final RMSE = ${final_rmse:.2f}")
    
    plt.xlabel('Number of Trees')
    plt.ylabel('Test MSE')
    plt.title('Effect of Learning Rate on Model Performance')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('gradient_boosting_learning_rate_effect.png', dpi=100, bbox_inches='tight')
    print("\n📊 Learning rate effect plot saved as 'gradient_boosting_learning_rate_effect.png'")
    
    print("\nKey Insights:")
    print("  • Lower learning rates require more trees but often generalize better")
    print("  • Higher learning rates converge faster but may overfit")
    print("  • Optimal: balance between learning rate and number of trees")


def main():
    """Run all examples"""
    print("\n" + "=" * 80)
    print("GRADIENT BOOSTING: Comprehensive Examples")
    print("Comparing from-scratch implementation with scikit-learn")
    print("=" * 80)
    
    example_1_regression_comparison()
    example_2_classification_comparison()
    example_3_feature_importance()
    example_4_learning_curves()
    example_5_hyperparameter_tuning()
    example_6_learning_rate_effect()
    
    print("\n" + "=" * 80)
    print("All examples completed successfully!")
    print("=" * 80)


if __name__ == "__main__":
    main()
