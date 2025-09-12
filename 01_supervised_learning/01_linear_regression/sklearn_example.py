"""
Linear Regression with Scikit-learn
===================================

Practical examples using scikit-learn's Linear Regression with
multiple datasets and evaluation techniques.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.datasets import make_regression
import warnings
warnings.filterwarnings('ignore')

def create_synthetic_dataset():
    """
    Create synthetic datasets for demonstration
    """
    np.random.seed(42)
    
    # Dataset 1: Simple linear relationship
    X_simple = np.linspace(0, 10, 100).reshape(-1, 1)
    y_simple = 2 * X_simple.ravel() + 3 + np.random.normal(0, 1, 100)
    
    # Dataset 2: Multiple features
    X_multi, y_multi = make_regression(
        n_samples=200, 
        n_features=3, 
        noise=10, 
        random_state=42
    )
    
    return X_simple, y_simple, X_multi, y_multi

def simple_linear_regression_demo():
    """
    Simple Linear Regression demo
    """
    print("\\n" + "="*50)
    print("SIMPLE LINEAR REGRESSION DEMO")
    print("="*50)
    
    # Create data
    X_simple, y_simple, _, _ = create_synthetic_dataset()
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X_simple, y_simple, test_size=0.2, random_state=42
    )
    
    # Create and train model
    model = LinearRegression()
    model.fit(X_train, y_train)
    
    # Predictions
    y_pred_train = model.predict(X_train)
    y_pred_test = model.predict(X_test)
    
    # Evaluate
    train_r2 = r2_score(y_train, y_pred_train)
    test_r2 = r2_score(y_test, y_pred_test)
    test_mse = mean_squared_error(y_test, y_pred_test)
    test_mae = mean_absolute_error(y_test, y_pred_test)
    
    # Print results
    print(f"Model Coefficients:")
    print(f"  Slope (β₁): {model.coef_[0]:.4f}")
    print(f"  Intercept (β₀): {model.intercept_:.4f}")
    print(f"\\nModel Performance:")
    print(f"  Training R²: {train_r2:.4f}")
    print(f"  Test R²: {test_r2:.4f}")
    print(f"  Test MSE: {test_mse:.4f}")
    print(f"  Test MAE: {test_mae:.4f}")
    
    # Visualize
    plt.figure(figsize=(12, 4))
    
    # Plot 1: Training data with regression line
    plt.subplot(1, 2, 1)
    plt.scatter(X_train, y_train, alpha=0.6, label='Training data')
    plt.scatter(X_test, y_test, alpha=0.6, color='red', label='Test data')
    
    X_plot = np.linspace(X_simple.min(), X_simple.max(), 100).reshape(-1, 1)
    y_plot = model.predict(X_plot)
    plt.plot(X_plot, y_plot, 'g-', linewidth=2, label=f'Regression Line')
    
    plt.xlabel('X')
    plt.ylabel('y')
    plt.title('Simple Linear Regression')
    plt.legend()
    plt.grid(True)
    
    # Plot 2: Residuals
    plt.subplot(1, 2, 2)
    residuals = y_test - y_pred_test
    plt.scatter(y_pred_test, residuals, alpha=0.6)
    plt.axhline(y=0, color='r', linestyle='--')
    plt.xlabel('Predicted Values')
    plt.ylabel('Residuals')
    plt.title('Residuals Plot')
    plt.grid(True)
    
    plt.tight_layout()
    plt.show()

def multiple_linear_regression_demo():
    """
    Multiple Linear Regression demo
    """
    print("\\n" + "="*50)
    print("MULTIPLE LINEAR REGRESSION DEMO")
    print("="*50)
    
    # Create data
    _, _, X_multi, y_multi = create_synthetic_dataset()
    
    # Create DataFrame for easier handling
    feature_names = [f'Feature_{i+1}' for i in range(X_multi.shape[1])]
    df = pd.DataFrame(X_multi, columns=feature_names)
    df['target'] = y_multi
    
    print(f"Dataset shape: {df.shape}")
    print(f"\nDataset info:")
    print(df.describe())
    
    # Split data
    X = df[feature_names]
    y = df['target']
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    # Feature scaling
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Train models (with and without scaling)
    model_unscaled = LinearRegression()
    model_scaled = LinearRegression()
    
    model_unscaled.fit(X_train, y_train)
    model_scaled.fit(X_train_scaled, y_train)
    
    # Predictions
    y_pred_unscaled = model_unscaled.predict(X_test)
    y_pred_scaled = model_scaled.predict(X_test_scaled)
    
    # Evaluate
    r2_unscaled = r2_score(y_test, y_pred_unscaled)
    r2_scaled = r2_score(y_test, y_pred_scaled)
    
    print(f"\nModel Coefficients (Unscaled):")
    for i, coef in enumerate(model_unscaled.coef_):
        print(f"  {feature_names[i]}: {coef:.4f}")
    print(f"  Intercept: {model_unscaled.intercept_:.4f}")
    
    print(f"\nModel Performance:")
    print(f"  Unscaled R²: {r2_unscaled:.4f}")
    print(f"  Scaled R²: {r2_scaled:.4f}")
    
    # Feature importance visualization
    plt.figure(figsize=(12, 4))
    
    # Plot 1: Coefficients comparison
    plt.subplot(1, 2, 1)
    x_pos = range(len(feature_names))
    plt.bar(x_pos, model_unscaled.coef_, alpha=0.7)
    plt.xlabel('Features')
    plt.ylabel('Coefficients')
    plt.title('Feature Coefficients (Unscaled)')
    plt.xticks(x_pos, feature_names)
    plt.grid(True, alpha=0.3)
    
    # Plot 2: Predictions vs Actual
    plt.subplot(1, 2, 2)
    plt.scatter(y_test, y_pred_unscaled, alpha=0.6)
    plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
    plt.xlabel('Actual Values')
    plt.ylabel('Predicted Values')
    plt.title(f'Predictions vs Actual (R² = {r2_unscaled:.4f})')
    plt.grid(True)
    
    plt.tight_layout()
    plt.show()

def cross_validation_demo():
    """
    Cross-validation demo
    """
    print("\\n" + "="*50)
    print("CROSS VALIDATION DEMO")
    print("="*50)
    
    # Create data
    _, _, X_multi, y_multi = create_synthetic_dataset()
    
    # Create model
    model = LinearRegression()
    
    # Perform cross-validation
    cv_scores = cross_val_score(model, X_multi, y_multi, cv=5, scoring='r2')
    cv_mse_scores = -cross_val_score(model, X_multi, y_multi, cv=5, scoring='neg_mean_squared_error')
    
    print(f"Cross-Validation Results (5-fold):")
    print(f"  R² Scores: {cv_scores}")
    print(f"  Mean R²: {cv_scores.mean():.4f} (+/- {cv_scores.std() * 2:.4f})")
    print(f"  MSE Scores: {cv_mse_scores}")
    print(f"  Mean MSE: {cv_mse_scores.mean():.4f} (+/- {cv_mse_scores.std() * 2:.4f})")
    
    # Visualize CV results
    plt.figure(figsize=(10, 4))
    
    plt.subplot(1, 2, 1)
    plt.boxplot([cv_scores])
    plt.ylabel('R² Score')
    plt.title('Cross-Validation R² Scores')
    plt.xticks([1], ['Linear Regression'])
    plt.grid(True, alpha=0.3)
    
    plt.subplot(1, 2, 2)
    plt.boxplot([cv_mse_scores])
    plt.ylabel('MSE')
    plt.title('Cross-Validation MSE Scores')
    plt.xticks([1], ['Linear Regression'])
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()

def polynomial_features_demo():
    """
    Linear Regression with Polynomial Features demo
    """
    print("\\n" + "="*50)
    print("POLYNOMIAL FEATURES DEMO")
    print("="*50)
    
    # Create non-linear data
    np.random.seed(42)
    X = np.linspace(0, 4, 100).reshape(-1, 1)
    y = 0.5 * X.ravel() ** 3 - 2 * X.ravel() ** 2 + X.ravel() + np.random.normal(0, 2, 100)
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    # Try different polynomial degrees
    degrees = [1, 2, 3, 4]
    models = {}
    scores = {}
    
    from sklearn.preprocessing import PolynomialFeatures
    from sklearn.pipeline import Pipeline
    
    plt.figure(figsize=(15, 10))
    
    for i, degree in enumerate(degrees):
        # Create pipeline with polynomial features
        model = Pipeline([
            ('poly', PolynomialFeatures(degree=degree)),
            ('linear', LinearRegression())
        ])
        
        # Fit model
        model.fit(X_train, y_train)
        models[degree] = model
        
        # Predictions
        y_pred_train = model.predict(X_train)
        y_pred_test = model.predict(X_test)
        
        # Scores
        train_score = r2_score(y_train, y_pred_train)
        test_score = r2_score(y_test, y_pred_test)
        scores[degree] = {'train': train_score, 'test': test_score}
        
        # Plot
        plt.subplot(2, 2, i + 1)
        plt.scatter(X_train, y_train, alpha=0.6, label='Training data')
        plt.scatter(X_test, y_test, alpha=0.6, color='red', label='Test data')
        
        X_plot = np.linspace(0, 4, 100).reshape(-1, 1)
        y_plot = model.predict(X_plot)
        plt.plot(X_plot, y_plot, 'g-', linewidth=2, 
                label=f'Degree {degree}')
        
        plt.xlabel('X')
        plt.ylabel('y')
        plt.title(f'Polynomial Degree {degree}\nTrain R²: {train_score:.3f}, Test R²: {test_score:.3f}')
        plt.legend()
        plt.grid(True)
    
    plt.tight_layout()
    plt.show()
    
    # Print results
    print(f"\nPolynomial Regression Results:")
    for degree in degrees:
        print(f"  Degree {degree}: Train R² = {scores[degree]['train']:.4f}, Test R² = {scores[degree]['test']:.4f}")

def main():
    """
    Main function to run all demos
    """
    print("Linear Regression with Scikit-learn")
    print("="*60)
    
    # Run all demos
    simple_linear_regression_demo()
    multiple_linear_regression_demo()
    cross_validation_demo()
    polynomial_features_demo()
    
    print("\n" + "="*60)
    print("SUMMARY & TIPS")
    print("="*60)
    print("""
    Key Takeaways:
    1. Linear Regression suits linear relationships
    2. Feature scaling helps coefficient interpretation
    3. Cross-validation gives robust performance estimates
    4. Polynomial features can capture non-linear patterns
    5. Watch for overfitting at high polynomial degrees
    
    Best Practices:
    - Always visualize data before modeling
    - Check assumptions (linearity, homoscedasticity, normality)
    - Use cross-validation for evaluation
    - Consider regularization for high-dimensional data
    """)

if __name__ == "__main__":
    main()
