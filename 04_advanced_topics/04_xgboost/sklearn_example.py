"""
XGBoost with Scikit-learn Interface - Practical Examples
=======================================================

Advanced XGBoost examples using scikit-learn interface with:
- Feature engineering and preprocessing
- Advanced hyperparameter tuning with Optuna
- Model interpretation with SHAP
- Ensemble methods and model comparison
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.preprocessing import LabelEncoder, StandardScaler, PolynomialFeatures
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.datasets import load_wine, load_diabetes
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
import xgboost as xgb
from xgboost import XGBClassifier, XGBRegressor
import warnings
warnings.filterwarnings('ignore')

# Set plotting style
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

def load_and_prepare_wine_dataset():
    """
    Load and prepare wine dataset for classification
    """
    # Load dataset
    wine = load_wine()
    X, y = wine.data, wine.target
    feature_names = wine.feature_names
    target_names = wine.target_names
    
    # Create DataFrame
    df = pd.DataFrame(X, columns=feature_names)
    df['target'] = y
    df['target_name'] = [target_names[i] for i in y]
    
    print("🍷 Wine Dataset Information:")
    print(f"  Samples: {len(df)}")
    print(f"  Features: {len(feature_names)}")
    print(f"  Classes: {len(target_names)} {list(target_names)}")
    print(f"  Class distribution: {dict(zip(target_names, np.bincount(y)))}")
    
    return df, X, y, feature_names, target_names

def advanced_feature_engineering(X, feature_names):
    """
    Advanced feature engineering for better XGBoost performance
    """
    print("\\n🔧 Performing Advanced Feature Engineering...")
    
    # Convert to DataFrame if numpy array
    if isinstance(X, np.ndarray):
        df = pd.DataFrame(X, columns=feature_names)
    else:
        df = X.copy()
    
    # Original features count
    original_features = len(df.columns)
    
    # 1. Polynomial features (degree 2) - select subset to avoid explosion
    important_features = ['alcohol', 'flavanoids', 'color_intensity', 'od280/od315_of_diluted_wines']
    if all(feat in df.columns for feat in important_features):
        poly = PolynomialFeatures(degree=2, include_bias=False)
        poly_features = poly.fit_transform(df[important_features])
        poly_feature_names = [f"poly_{name}" for name in poly.get_feature_names_out(important_features)]
        poly_df = pd.DataFrame(poly_features, columns=poly_feature_names, index=df.index)
        df = pd.concat([df, poly_df.iloc[:, len(important_features):]], axis=1)  # Skip original features
    
    # 2. Ratio features (domain knowledge)
    if 'alcohol' in df.columns and 'total_phenols' in df.columns:
        df['alcohol_phenols_ratio'] = df['alcohol'] / (df['total_phenols'] + 1e-8)
    
    if 'flavanoids' in df.columns and 'nonflavanoid_phenols' in df.columns:
        df['flavanoid_ratio'] = df['flavanoids'] / (df['nonflavanoid_phenols'] + 1e-8)
    
    # 3. Binning continuous features
    if 'alcohol' in df.columns:
        df['alcohol_binned'] = pd.cut(df['alcohol'], bins=5, labels=['very_low', 'low', 'medium', 'high', 'very_high'])
        df['alcohol_binned'] = LabelEncoder().fit_transform(df['alcohol_binned'])
    
    # 4. Statistical features
    numeric_cols = df.select_dtypes(include=[np.number]).columns[:len(feature_names)]  # Only original features
    if len(numeric_cols) > 1:
        df['feature_mean'] = df[numeric_cols].mean(axis=1)
        df['feature_std'] = df[numeric_cols].std(axis=1)
        df['feature_max'] = df[numeric_cols].max(axis=1)
        df['feature_min'] = df[numeric_cols].min(axis=1)
    
    new_features = len(df.columns) - original_features
    print(f"  ✅ Added {new_features} new features")
    print(f"  Total features: {len(df.columns)}")
    
    return df.values, df.columns.tolist()

def xgboost_with_feature_selection(X, y, feature_names):
    """
    XGBoost with automatic feature selection
    """
    print("\\n🎯 XGBoost with Feature Selection")
    print("="*50)
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y
    )
    
    # Feature selection using SelectKBest
    print("🔍 Performing feature selection...")
    selector = SelectKBest(score_func=f_classif, k=15)  # Select top 15 features
    X_train_selected = selector.fit_transform(X_train, y_train)
    X_test_selected = selector.transform(X_test)
    
    # Get selected feature names
    selected_features = [feature_names[i] for i in selector.get_support(indices=True)]
    print(f"Selected features: {len(selected_features)}")
    
    # Train XGBoost on original features
    xgb_original = XGBClassifier(
        n_estimators=100, max_depth=4, learning_rate=0.1,
        random_state=42, eval_metric='mlogloss'
    )
    xgb_original.fit(X_train, y_train)
    score_original = xgb_original.score(X_test, y_test)
    
    # Train XGBoost on selected features
    xgb_selected = XGBClassifier(
        n_estimators=100, max_depth=4, learning_rate=0.1,
        random_state=42, eval_metric='mlogloss'
    )
    xgb_selected.fit(X_train_selected, y_train)
    score_selected = xgb_selected.score(X_test_selected, y_test)
    
    print(f"\\n📊 Results:")
    print(f"  Original features ({X.shape[1]}): {score_original:.4f}")
    print(f"  Selected features ({len(selected_features)}): {score_selected:.4f}")
    print(f"  Improvement: {score_selected - score_original:.4f}")
    
    return xgb_selected, selected_features, (X_train_selected, X_test_selected, y_train, y_test)

def hyperparameter_tuning_with_optuna(X_train, X_test, y_train, y_test):
    """
    Advanced hyperparameter tuning using Optuna
    """
    try:
        import optuna
    except ImportError:
        print("⚠️ Optuna not installed. Using RandomizedSearchCV instead...")
        from sklearn.model_selection import RandomizedSearchCV
        
        param_grid = {
            'n_estimators': [50, 100, 200],
            'max_depth': [3, 4, 5, 6],
            'learning_rate': [0.01, 0.1, 0.2],
            'subsample': [0.8, 0.9, 1.0],
            'colsample_bytree': [0.8, 0.9, 1.0],
            'reg_alpha': [0, 0.1, 0.5],
            'reg_lambda': [1, 1.5, 2]
        }
        
        xgb_model = XGBClassifier(random_state=42, eval_metric='mlogloss')
        search = RandomizedSearchCV(
            xgb_model, param_grid, n_iter=30, cv=3,
            scoring='accuracy', n_jobs=-1, random_state=42
        )
        search.fit(X_train, y_train)
        return search.best_estimator_, search.best_params_
    
    print("\\n⚙️ Advanced Hyperparameter Tuning with Optuna")
    print("="*55)
    
    def objective(trial):
        params = {
            'n_estimators': trial.suggest_int('n_estimators', 50, 300),
            'max_depth': trial.suggest_int('max_depth', 3, 8),
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3),
            'subsample': trial.suggest_float('subsample', 0.6, 1.0),
            'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
            'reg_alpha': trial.suggest_float('reg_alpha', 0, 2),
            'reg_lambda': trial.suggest_float('reg_lambda', 1, 3),
            'random_state': 42,
            'eval_metric': 'mlogloss'
        }
        
        model = XGBClassifier(**params)
        
        # Cross-validation score
        cv_scores = cross_val_score(model, X_train, y_train, cv=3, scoring='accuracy')
        return cv_scores.mean()
    
    # Create and run study
    study = optuna.create_study(direction='maximize')
    study.optimize(objective, n_trials=100, show_progress_bar=False)
    
    print(f"🏆 Best score: {study.best_value:.4f}")
    print(f"✨ Best parameters:")
    for key, value in study.best_params.items():
        print(f"  {key}: {value}")
    
    # Train final model with best parameters
    best_model = XGBClassifier(**study.best_params, random_state=42, eval_metric='mlogloss')
    best_model.fit(X_train, y_train)
    
    # Plot optimization history
    plt.figure(figsize=(12, 4))
    
    plt.subplot(1, 2, 1)
    trials_df = study.trials_dataframe()
    plt.plot(trials_df['number'], trials_df['value'])
    plt.xlabel('Trial Number')
    plt.ylabel('Accuracy')
    plt.title('Optuna Optimization History')
    plt.grid(True)
    
    plt.subplot(1, 2, 2)
    importance = optuna.importance.get_param_importances(study)
    params = list(importance.keys())
    values = list(importance.values())
    plt.barh(params, values)
    plt.xlabel('Importance')
    plt.title('Parameter Importance')
    
    plt.tight_layout()
    plt.show()
    
    return best_model, study.best_params

def model_interpretation_with_shap(model, X_test, feature_names):
    """
    Model interpretation using SHAP
    """
    try:
        import shap
    except ImportError:
        print("⚠️ SHAP not installed. Skipping interpretation...")
        print("Install with: pip install shap")
        return
    
    print("\\n🧠 Model Interpretation with SHAP")
    print("="*40)
    
    # Create SHAP explainer
    explainer = shap.Explainer(model)
    shap_values = explainer(X_test[:100])  # Use subset for speed
    
    # Summary plot
    plt.figure(figsize=(10, 6))
    shap.summary_plot(shap_values, X_test[:100], feature_names=feature_names[:X_test.shape[1]], show=False)
    plt.tight_layout()
    plt.show()
    
    # Feature importance plot
    plt.figure(figsize=(10, 6))
    shap.plots.bar(shap_values, show=False)
    plt.tight_layout()
    plt.show()
    
    print("✅ SHAP analysis completed!")

def ensemble_comparison():
    """
    Compare XGBoost with other ensemble methods
    """
    print("\\n🏆 Ensemble Methods Comparison")
    print("="*45)
    
    # Load diabetes dataset for regression
    diabetes = load_diabetes()
    X, y = diabetes.data, diabetes.target
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    
    # Define models
    models = {
        'XGBoost': XGBRegressor(n_estimators=100, random_state=42, eval_metric='rmse'),
        'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42),
        'Gradient Boosting': GradientBoostingClassifier(n_estimators=100, random_state=42)
    }
    
    # Note: Using regression models but RandomForest/GradientBoosting classifiers - let's fix this
    from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
    
    models = {
        'XGBoost': XGBRegressor(n_estimators=100, random_state=42, eval_metric='rmse'),
        'Random Forest': RandomForestRegressor(n_estimators=100, random_state=42),
        'Gradient Boosting': GradientBoostingRegressor(n_estimators=100, random_state=42)
    }
    
    results = {}
    
    print("🚀 Training models...")
    for name, model in models.items():
        # Train model
        model.fit(X_train, y_train)
        
        # Predictions
        y_pred = model.predict(X_test)
        
        # Evaluate
        r2 = r2_score(y_test, y_pred)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        mae = mean_absolute_error(y_test, y_pred)
        
        results[name] = {'R²': r2, 'RMSE': rmse, 'MAE': mae}
        
        print(f"  {name:15}: R²={r2:.4f}, RMSE={rmse:.4f}, MAE={mae:.4f}")
    
    # Visualize comparison
    results_df = pd.DataFrame(results).T
    
    plt.figure(figsize=(15, 5))
    
    for i, metric in enumerate(['R²', 'RMSE', 'MAE']):
        plt.subplot(1, 3, i+1)
        bars = plt.bar(results_df.index, results_df[metric])
        plt.title(f'{metric} Comparison')
        plt.ylabel(metric)
        
        # Highlight best performer
        if metric == 'R²':
            best_idx = results_df[metric].idxmax()
            bars[list(results_df.index).index(best_idx)].set_color('gold')
        else:
            best_idx = results_df[metric].idxmin()
            bars[list(results_df.index).index(best_idx)].set_color('gold')
        
        plt.xticks(rotation=45)
    
    plt.tight_layout()
    plt.show()
    
    return results

def main():
    """
    Main function demonstrating advanced XGBoost usage
    """
    print("🚀 Advanced XGBoost with Scikit-learn Interface")
    print("=" * 70)
    
    # Load and prepare dataset
    df, X, y, feature_names, target_names = load_and_prepare_wine_dataset()
    
    # Advanced feature engineering
    X_engineered, engineered_features = advanced_feature_engineering(X, feature_names)
    
    # XGBoost with feature selection
    best_model, selected_features, (X_train_sel, X_test_sel, y_train, y_test) = xgboost_with_feature_selection(
        X_engineered, y, engineered_features
    )
    
    # Advanced hyperparameter tuning
    tuned_model, best_params = hyperparameter_tuning_with_optuna(
        X_train_sel, X_test_sel, y_train, y_test
    )
    
    # Model interpretation
    model_interpretation_with_shap(tuned_model, X_test_sel, selected_features)
    
    # Ensemble comparison
    ensemble_results = ensemble_comparison()
    
    # Final evaluation
    y_pred = tuned_model.predict(X_test_sel)
    final_accuracy = (y_pred == y_test).mean()
    
    print("\\n" + "="*70)
    print("🎓 ADVANCED XGBOOST TUTORIAL SUMMARY")
    print("="*70)
    print(f"""
    ✅ Feature Engineering: {len(engineered_features)} total features
    ✅ Feature Selection: {len(selected_features)} selected features  
    ✅ Hyperparameter Tuning: {final_accuracy:.4f} final accuracy
    ✅ Model Interpretation: SHAP analysis completed
    ✅ Ensemble Comparison: Performance benchmarking done
    
    🏆 Key Advanced Techniques:
    • Polynomial feature generation
    • Domain-specific feature engineering
    • Automated feature selection
    • Optuna-based hyperparameter optimization
    • SHAP model interpretation
    • Ensemble method comparison
    
    💡 Pro Tips:
    • Feature engineering often beats hyperparameter tuning
    • Use domain knowledge for feature creation
    • SHAP helps understand model decisions
    • Ensemble different algorithms for better results
    • Always validate on unseen data
    
    Ready for production! 🚀
    """)

if __name__ == "__main__":
    main()