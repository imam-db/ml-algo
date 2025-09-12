# Linear Regression

## 📖 Theory and Concepts

### What is Linear Regression?
Linear Regression is a supervised learning algorithm used to predict target values (continuous) based on one or more input features. This algorithm finds the best linear relationship between input and output.

### Mathematical Formula
**Simple Linear Regression (1 variable):**
```
y = β₀ + β₁x + ε
```

**Multiple Linear Regression (multiple variables):**
```
y = β₀ + β₁x₁ + β₂x₂ + ... + βₙxₙ + ε
```

Where:
- `y` = target variable (dependent)
- `x` = feature variable (independent) 
- `β₀` = intercept (bias)
- `β₁, β₂, ..., βₙ` = coefficients (slope)
- `ε` = error/noise

### Cost Function
Linear regression menggunakan **Mean Squared Error (MSE)** sebagai cost function:
```
MSE = (1/n) * Σ(yi - ŷi)²
```

### Optimization
To find the best parameters, you can use:
1. **Normal Equation** (Closed-form solution)
2. **Gradient Descent** (Iterative approach)

## 🎯 When to Use Linear Regression?

### ✅ Suitable for:
- Predicting continuous values (house prices, temperature, salary, etc.)
- Data with linear relationships
- Model interpretation that is easy to understand
- Baseline model for comparison
- Small to medium-sized datasets

### ❌ Not suitable for:
- Data with complex non-linear relationships
- Many outliers
- Categorical target variables
- High multicollinearity

## 📊 Linear Regression Assumptions

1. **Linearity**: Linear relationship between X and y
2. **Independence**: Observations are independent
3. **Homoscedasticity**: Constant error variance
4. **Normal Distribution**: Errors are normally distributed
5. **No Multicollinearity**: Features are not highly correlated

## 📈 Model Evaluation

### Metrics for Regression:
- **R² (Coefficient of Determination)**: Proportion of variance explained
- **MSE (Mean Squared Error)**: Average squared error
- **RMSE (Root Mean Squared Error)**: Square root of MSE
- **MAE (Mean Absolute Error)**: Average absolute error

## 🔍 Advantages and Disadvantages

### Advantages:
- ✅ Simple and fast
- ✅ Easy to interpret
- ✅ No complex parameter tuning required
- ✅ Good baseline
- ✅ Not prone to overfitting

### Disadvantages:
- ❌ Can only capture linear relationships
- ❌ Sensitive to outliers
- ❌ Strict assumptions
- ❌ Poor performance on non-linear data

## 📝 Implementation

In this folder, you will find:
- `implementation.py` - From-scratch implementation using NumPy
- `sklearn_example.py` - Examples using scikit-learn
- `exercise.ipynb` - Practical exercises with real datasets

## 🎓 Tips for Practice

1. **Always visualize data** before modeling
2. **Check assumptions** of linear regression
3. **Handle outliers** if necessary
4. **Feature scaling** for multiple regression
5. **Validate model** using cross-validation

## 📚 References

- [Scikit-learn Linear Regression](https://scikit-learn.org/stable/modules/linear_model.html#ordinary-least-squares)
- [Introduction to Statistical Learning - Chapter 3](https://www.statlearning.com/)

---
**Next Step**: After understanding Linear Regression, continue to **Logistic Regression** for classification problems!
