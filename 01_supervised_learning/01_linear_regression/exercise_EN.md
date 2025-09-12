# Linear Regression — Practical Exercises (English)

This companion guide mirrors the notebook exercises in English. Use it side‑by‑side with `exercise.ipynb`, or follow the steps here if you prefer Markdown.

## Learning Objectives
After completing these exercises, you will be able to:
- Implement Linear Regression from scratch
- Use Linear Regression with scikit‑learn
- Perform basic data exploration and visualization
- Evaluate model performance using appropriate metrics

## 📊 Exercise 1: Simple Linear Regression
Instructions:
1. Generate a simple linear dataset with noise (e.g., 100 points).
2. Implement Linear Regression from scratch using gradient descent (or use the provided class).
3. Visualize the regression line and data points.
4. Compute R², MSE, and MAE on a held‑out test set.

Hints:
- Split data into train/test with `train_test_split`.
- Standardize inputs if you notice slow convergence with gradient descent.

## 📊 Exercise 2: Multiple Linear Regression (Real Dataset)
Instructions:
1. Load a small real dataset (e.g., from scikit‑learn or a CSV).
2. Split data into training and test sets.
3. Train `LinearRegression` (scikit‑learn) on the training set.
4. Evaluate performance on the test set (R², RMSE, MAE) and interpret coefficients.

Hints:
- Check feature scales; consider `StandardScaler` for improved interpretability.
- Inspect correlations to spot multicollinearity.

## 📈 Exercise 3: Polynomial Features
Instructions:
1. Add polynomial features (degree 2 and 3) to a non‑linear synthetic dataset.
2. Compare performance against plain Linear Regression.
3. Visualize fitted curves for each degree.

Hints:
- Use `PolynomialFeatures` + `LinearRegression` in a `Pipeline`.
- Watch for overfitting at higher degrees; use a validation split or cross‑validation.

## 💡 Tips
- Always visualize the data before modeling.
- Check Linear Regression assumptions (linearity, homoscedasticity, normality, independence, no high multicollinearity).
- Use cross‑validation for more robust evaluation.

## 🧪 Stretch Task
Implement and compare Linear Regression with regularization (Ridge/Lasso):
- Tune the regularization strength.
- Compare bias/variance behavior and test performance.

Happy learning!
