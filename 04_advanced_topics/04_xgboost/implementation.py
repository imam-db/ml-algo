"""
XGBoost Implementation and Examples
===================================

Comprehensive examples demonstrating XGBoost for both regression and classification
with hyperparameter tuning, feature importance analysis, and model interpretation.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV, RandomizedSearchCV
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, roc_auc_score
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.datasets import make_classification, make_regression, load_breast_cancer, fetch_california_housing
import xgboost as xgb
from xgboost import XGBClassifier, XGBRegressor
import warnings
warnings.filterwarnings('ignore')

# Set plotting style
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

def create_synthetic_datasets():
    """
    Create synthetic datasets for demonstration
    """
    # Classification dataset
    X_clf, y_clf = make_classification(
        n_samples=1000,
        n_features=20,
        n_informative=15,
        n_redundant=5,
        n_clusters_per_class=1,
        random_state=42
    )
    
    # Regression dataset
    X_reg, y_reg = make_regression(
        n_samples=1000,
        n_features=10,
        noise=0.1,
        random_state=42
    )
    
    return (X_clf, y_clf), (X_reg, y_reg)

def xgboost_classification_demo():
    """
    Comprehensive XGBoost Classification Demo
    """
    print("\\n" + "="*60)
    print("🎯 XGBOOST CLASSIFICATION DEMO")
    print("="*60)
    
    # Load breast cancer dataset
    data = load_breast_cancer()
    X, y = data.data, data.target
    feature_names = data.feature_names
    
    print(f"Dataset: {data.DESCR.split('**')[1].split('**')[0].strip()}")
    print(f"Samples: {X.shape[0]}, Features: {X.shape[1]}")
    print(f"Classes: {len(np.unique(y))} ({list(data.target_names)})")
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # Create and train XGBoost classifier
    print("\\n🚀 Training XGBoost Classifier...")
    
    xgb_clf = XGBClassifier(
        n_estimators=100,
        max_depth=6,
        learning_rate=0.1,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        eval_metric='logloss'
    )
    
    # Fit with early stopping
    xgb_clf.fit(
        X_train, y_train,
        eval_set=[(X_train, y_train), (X_test, y_test)],
        early_stopping_rounds=10,
        verbose=False
    )
    
    # Predictions
    y_pred = xgb_clf.predict(X_test)
    y_pred_proba = xgb_clf.predict_proba(X_test)[:, 1]
    
    # Evaluate
    accuracy = accuracy_score(y_test, y_pred)
    precision, recall, f1, _ = precision_recall_fscore_support(y_test, y_pred, average='weighted')
    auc = roc_auc_score(y_test, y_pred_proba)
    
    print(f"\\n📊 Model Performance:")
    print(f"  Accuracy: {accuracy:.4f}")
    print(f"  Precision: {precision:.4f}")
    print(f"  Recall: {recall:.4f}")
    print(f"  F1-Score: {f1:.4f}")
    print(f"  AUC-ROC: {auc:.4f}")
    
    # Feature importance analysis
    feature_importance = pd.DataFrame({
        'feature': feature_names,
        'importance': xgb_clf.feature_importances_
    }).sort_values('importance', ascending=False)
    
    print(f"\\n🔍 Top 10 Most Important Features:")
    for i, row in feature_importance.head(10).iterrows():
        print(f"  {row['feature'][:30]:30}: {row['importance']:.4f}")
    
    # Visualizations
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    # 1. Feature Importance (Top 15)
    top_features = feature_importance.head(15)
    axes[0, 0].barh(range(len(top_features)), top_features['importance'])
    axes[0, 0].set_yticks(range(len(top_features)))
    axes[0, 0].set_yticklabels([feat[:20] for feat in top_features['feature']])
    axes[0, 0].set_xlabel('Feature Importance')
    axes[0, 0].set_title('Top 15 Feature Importances')
    axes[0, 0].invert_yaxis()
    
    # 2. Learning Curve (from evals_result)
    train_scores = xgb_clf.evals_result()['validation_0']['logloss']
    test_scores = xgb_clf.evals_result()['validation_1']['logloss']
    epochs = range(1, len(train_scores) + 1)
    
    axes[0, 1].plot(epochs, train_scores, label='Training')
    axes[0, 1].plot(epochs, test_scores, label='Validation')
    axes[0, 1].set_xlabel('Epochs')
    axes[0, 1].set_ylabel('Log Loss')
    axes[0, 1].set_title('Learning Curve')
    axes[0, 1].legend()
    axes[0, 1].grid(True)
    
    # 3. Confusion Matrix
    from sklearn.metrics import confusion_matrix
    cm = confusion_matrix(y_test, y_pred)
    sns.heatmap(cm, annot=True, fmt='d', ax=axes[1, 0], 
                xticklabels=data.target_names, yticklabels=data.target_names)
    axes[1, 0].set_title('Confusion Matrix')
    axes[1, 0].set_ylabel('Actual')
    axes[1, 0].set_xlabel('Predicted')
    
    # 4. ROC Curve
    from sklearn.metrics import roc_curve
    fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
    axes[1, 1].plot(fpr, tpr, label=f'ROC Curve (AUC = {auc:.3f})')
    axes[1, 1].plot([0, 1], [0, 1], 'k--', alpha=0.5)
    axes[1, 1].set_xlabel('False Positive Rate')
    axes[1, 1].set_ylabel('True Positive Rate')
    axes[1, 1].set_title('ROC Curve')
    axes[1, 1].legend()
    axes[1, 1].grid(True)
    
    plt.tight_layout()
    plt.show()
    
    return xgb_clf, feature_importance

def xgboost_regression_demo():
    """
    Comprehensive XGBoost Regression Demo
    """
    print("\\n" + "="*60)
    print("📈 XGBOOST REGRESSION DEMO")
    print("="*60)
    
    # Load California housing dataset
    housing = fetch_california_housing()
    X, y = housing.data, housing.target
    feature_names = housing.feature_names
    
    print(f"Dataset: California Housing")
    print(f"Samples: {X.shape[0]}, Features: {X.shape[1]}")
    print(f"Target: House prices (in hundreds of thousands)")
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    # Create and train XGBoost regressor
    print("\\n🚀 Training XGBoost Regressor...")
    
    xgb_reg = XGBRegressor(
        n_estimators=100,
        max_depth=6,
        learning_rate=0.1,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        eval_metric='rmse'
    )
    
    # Fit with early stopping
    xgb_reg.fit(
        X_train, y_train,
        eval_set=[(X_train, y_train), (X_test, y_test)],
        early_stopping_rounds=10,
        verbose=False
    )
    
    # Predictions
    y_pred_train = xgb_reg.predict(X_train)
    y_pred_test = xgb_reg.predict(X_test)
    
    # Evaluate
    train_r2 = r2_score(y_train, y_pred_train)
    test_r2 = r2_score(y_test, y_pred_test)
    test_rmse = np.sqrt(mean_squared_error(y_test, y_pred_test))
    test_mae = mean_absolute_error(y_test, y_pred_test)
    
    print(f"\\n📊 Model Performance:")
    print(f"  Training R²: {train_r2:.4f}")
    print(f"  Test R²: {test_r2:.4f}")
    print(f"  Test RMSE: {test_rmse:.4f}")
    print(f"  Test MAE: {test_mae:.4f}")
    
    # Feature importance
    feature_importance = pd.DataFrame({
        'feature': feature_names,
        'importance': xgb_reg.feature_importances_
    }).sort_values('importance', ascending=False)
    
    print(f"\\n🔍 Feature Importance:")
    for i, row in feature_importance.iterrows():
        print(f"  {row['feature']:20}: {row['importance']:.4f}")
    
    # Visualizations
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    # 1. Feature Importance
    axes[0, 0].barh(feature_importance['feature'], feature_importance['importance'])
    axes[0, 0].set_xlabel('Feature Importance')
    axes[0, 0].set_title('Feature Importances')
    
    # 2. Learning Curve
    train_scores = xgb_reg.evals_result()['validation_0']['rmse']
    test_scores = xgb_reg.evals_result()['validation_1']['rmse']
    epochs = range(1, len(train_scores) + 1)
    
    axes[0, 1].plot(epochs, train_scores, label='Training RMSE')
    axes[0, 1].plot(epochs, test_scores, label='Validation RMSE')
    axes[0, 1].set_xlabel('Epochs')
    axes[0, 1].set_ylabel('RMSE')
    axes[0, 1].set_title('Learning Curve')
    axes[0, 1].legend()
    axes[0, 1].grid(True)
    
    # 3. Predictions vs Actual
    axes[1, 0].scatter(y_test, y_pred_test, alpha=0.6)
    axes[1, 0].plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
    axes[1, 0].set_xlabel('Actual Values')
    axes[1, 0].set_ylabel('Predicted Values')
    axes[1, 0].set_title(f'Predictions vs Actual (R² = {test_r2:.3f})')
    axes[1, 0].grid(True)
    
    # 4. Residuals Plot
    residuals = y_test - y_pred_test
    axes[1, 1].scatter(y_pred_test, residuals, alpha=0.6)
    axes[1, 1].axhline(y=0, color='r', linestyle='--')
    axes[1, 1].set_xlabel('Predicted Values')
    axes[1, 1].set_ylabel('Residuals')
    axes[1, 1].set_title('Residuals Plot')
    axes[1, 1].grid(True)
    
    plt.tight_layout()
    plt.show()
    
    return xgb_reg, feature_importance

def hyperparameter_tuning_demo():
    """
    Demonstrate hyperparameter tuning for XGBoost
    """
    print("\\n" + "="*60)
    print("⚙️ HYPERPARAMETER TUNING DEMO")
    print("="*60)
    
    # Use synthetic classification data
    (X_clf, y_clf), _ = create_synthetic_datasets()
    X_train, X_test, y_train, y_test = train_test_split(
        X_clf, y_clf, test_size=0.2, random_state=42, stratify=y_clf
    )
    
    # Define parameter grid for RandomizedSearchCV
    param_grid = {
        'n_estimators': [50, 100, 200, 300],
        'max_depth': [3, 4, 5, 6, 7],
        'learning_rate': [0.01, 0.1, 0.2, 0.3],
        'subsample': [0.6, 0.8, 1.0],
        'colsample_bytree': [0.6, 0.8, 1.0],
        'reg_alpha': [0, 0.1, 0.5, 1],
        'reg_lambda': [1, 1.5, 2]
    }
    
    # Base model
    base_model = XGBClassifier(random_state=42, eval_metric='logloss')
    
    # Randomized search
    print("🔍 Performing Randomized Search...")
    random_search = RandomizedSearchCV(
        base_model, 
        param_distributions=param_grid,
        n_iter=50,  # Number of parameter settings sampled
        cv=5,
        scoring='accuracy',
        n_jobs=-1,
        random_state=42,
        verbose=0
    )
    
    random_search.fit(X_train, y_train)
    
    # Best parameters
    print("\\n✨ Best Parameters Found:")
    for param, value in random_search.best_params_.items():
        print(f"  {param}: {value}")
    
    print(f"\\n📊 Best Cross-Validation Score: {random_search.best_score_:.4f}")
    
    # Compare default vs tuned model
    default_model = XGBClassifier(random_state=42, eval_metric='logloss')
    default_model.fit(X_train, y_train)
    
    tuned_model = random_search.best_estimator_
    
    # Evaluate both models
    default_score = accuracy_score(y_test, default_model.predict(X_test))
    tuned_score = accuracy_score(y_test, tuned_model.predict(X_test))
    
    print(f"\\n📈 Model Comparison:")
    print(f"  Default Model Accuracy: {default_score:.4f}")
    print(f"  Tuned Model Accuracy:   {tuned_score:.4f}")
    print(f"  Improvement:            {tuned_score - default_score:.4f}")
    
    # Plot parameter importance (top parameters from random search)
    results_df = pd.DataFrame(random_search.cv_results_)
    
    plt.figure(figsize=(12, 8))
    
    # Plot CV scores distribution
    plt.subplot(2, 2, 1)
    plt.hist(results_df['mean_test_score'], bins=20, alpha=0.7)
    plt.xlabel('Cross-Validation Score')
    plt.ylabel('Frequency')
    plt.title('Distribution of CV Scores')
    plt.axvline(random_search.best_score_, color='red', linestyle='--', 
                label=f'Best: {random_search.best_score_:.4f}')
    plt.legend()
    
    # Plot learning rate vs max_depth
    plt.subplot(2, 2, 2)
    for lr in [0.01, 0.1, 0.2, 0.3]:
        mask = results_df['param_learning_rate'] == lr
        if mask.sum() > 0:
            plt.scatter(results_df[mask]['param_max_depth'], 
                       results_df[mask]['mean_test_score'], 
                       label=f'LR={lr}', alpha=0.7)
    plt.xlabel('Max Depth')
    plt.ylabel('CV Score')
    plt.title('Learning Rate vs Max Depth')
    plt.legend()
    
    # Plot n_estimators impact
    plt.subplot(2, 2, 3)
    for n_est in [50, 100, 200, 300]:
        mask = results_df['param_n_estimators'] == n_est
        if mask.sum() > 0:
            plt.scatter([n_est] * mask.sum(), 
                       results_df[mask]['mean_test_score'], 
                       alpha=0.7, s=30)
    plt.xlabel('N Estimators')
    plt.ylabel('CV Score')
    plt.title('N Estimators vs Performance')
    
    # Plot regularization impact
    plt.subplot(2, 2, 4)
    plt.scatter(results_df['param_reg_alpha'], results_df['mean_test_score'], alpha=0.6)
    plt.xlabel('Reg Alpha (L1)')
    plt.ylabel('CV Score')
    plt.title('L1 Regularization Impact')
    
    plt.tight_layout()
    plt.show()
    
    return random_search.best_estimator_, random_search.best_params_

def cross_validation_demo():
    """
    Demonstrate XGBoost cross-validation
    """
    print("\\n" + "="*60)
    print("🔄 CROSS-VALIDATION DEMO")
    print("="*60)
    
    # Use synthetic regression data
    _, (X_reg, y_reg) = create_synthetic_datasets()
    
    # XGBoost native cross-validation
    print("🚀 XGBoost Native Cross-Validation...")
    
    # Convert to DMatrix (XGBoost's internal data structure)
    dtrain = xgb.DMatrix(X_reg, label=y_reg)
    
    # Parameters for XGBoost
    params = {
        'objective': 'reg:squarederror',
        'eval_metric': 'rmse',
        'max_depth': 6,
        'learning_rate': 0.1,
        'subsample': 0.8,
        'colsample_bytree': 0.8,
        'seed': 42
    }
    
    # Perform cross-validation
    cv_results = xgb.cv(
        params,
        dtrain,
        num_boost_round=200,
        nfold=5,
        early_stopping_rounds=10,
        seed=42,
        verbose_eval=False,
        show_stdv=True
    )
    
    print(f"\\n📊 CV Results:")
    print(f"  Best iteration: {len(cv_results)}")
    print(f"  Best train RMSE: {cv_results['train-rmse-mean'].iloc[-1]:.4f} ± {cv_results['train-rmse-std'].iloc[-1]:.4f}")
    print(f"  Best test RMSE:  {cv_results['test-rmse-mean'].iloc[-1]:.4f} ± {cv_results['test-rmse-std'].iloc[-1]:.4f}")
    
    # Plot cross-validation results
    plt.figure(figsize=(12, 4))
    
    # Plot 1: CV curves
    plt.subplot(1, 2, 1)
    iterations = range(1, len(cv_results) + 1)
    plt.plot(iterations, cv_results['train-rmse-mean'], label='Train RMSE', alpha=0.8)
    plt.fill_between(iterations, 
                     cv_results['train-rmse-mean'] - cv_results['train-rmse-std'],
                     cv_results['train-rmse-mean'] + cv_results['train-rmse-std'], 
                     alpha=0.2)
    
    plt.plot(iterations, cv_results['test-rmse-mean'], label='Test RMSE', alpha=0.8)
    plt.fill_between(iterations,
                     cv_results['test-rmse-mean'] - cv_results['test-rmse-std'],
                     cv_results['test-rmse-mean'] + cv_results['test-rmse-std'], 
                     alpha=0.2)
    
    plt.xlabel('Iterations')
    plt.ylabel('RMSE')
    plt.title('Cross-Validation Learning Curve')
    plt.legend()
    plt.grid(True)
    
    # Plot 2: Overfitting analysis
    plt.subplot(1, 2, 2)
    gap = cv_results['train-rmse-mean'] - cv_results['test-rmse-mean']
    plt.plot(iterations, gap, label='Train-Test Gap')
    plt.axhline(y=0, color='red', linestyle='--', alpha=0.5)
    plt.xlabel('Iterations')
    plt.ylabel('RMSE Gap (Train - Test)')
    plt.title('Overfitting Analysis')
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    plt.show()
    
    return cv_results

def main():
    """
    Main function to run all XGBoost demonstrations
    """
    print("🤖 XGBoost Comprehensive Tutorial")
    print("=" * 80)
    print("""
    This tutorial covers:
    1. 🎯 Classification with XGBoost
    2. 📈 Regression with XGBoost  
    3. ⚙️ Hyperparameter Tuning
    4. 🔄 Cross-Validation
    """)
    
    # Run all demos
    print("\\n🎬 Starting XGBoost Tutorial...")
    
    # Classification demo
    clf_model, clf_importance = xgboost_classification_demo()
    
    # Regression demo
    reg_model, reg_importance = xgboost_regression_demo()
    
    # Hyperparameter tuning demo
    tuned_model, best_params = hyperparameter_tuning_demo()
    
    # Cross-validation demo
    cv_results = cross_validation_demo()
    
    # Summary and tips
    print("\\n" + "="*80)
    print("🎓 XGBOOST TUTORIAL SUMMARY")
    print("="*80)
    print("""
    Key Takeaways:
    ✅ XGBoost excels on tabular data for both classification and regression
    ✅ Feature importance helps understand model decisions
    ✅ Hyperparameter tuning significantly improves performance
    ✅ Early stopping prevents overfitting
    ✅ Cross-validation provides robust performance estimates
    ✅ Learning curves help diagnose overfitting/underfitting
    
    Next Steps:
    📚 Practice with your own datasets
    🔧 Experiment with different hyperparameters
    📊 Compare with other algorithms (Random Forest, LightGBM)
    🧠 Learn about SHAP for model interpretation
    🏆 Participate in Kaggle competitions
    
    Happy Learning! 🚀
    """)

if __name__ == "__main__":
    main()