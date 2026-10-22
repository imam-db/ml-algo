# Linear Regression

**🇬🇧 English | [🇮🇩 Bahasa Indonesia](README.id.md)**

**🏠 [Back to Main](../../README.md)** | **📁 [Supervised Learning](../README.md)** | **➡️ [Next: Logistic Regression](../02_logistic_regression/README.md)**

## 🔑 Key Terms
**Quick definitions for this algorithm** - 📚 [Full Glossary](../../GLOSSARY.md)

- **[Linear Regression](../../GLOSSARY.md#linear-model)** - Algorithm assuming linear relationship between features and target
- **[Overfitting](../../GLOSSARY.md#overfitting)** - When model learns training data too well, performs poorly on new data  
- **[MSE](../../GLOSSARY.md#mse-mean-squared-error)** - Mean Squared Error, measures average squared prediction errors
- **[R²](../../GLOSSARY.md#r²-r-squared--coefficient-of-determination)** - Proportion of variance explained by model (0-1, higher better)
- **[Gradient Descent](../../GLOSSARY.md#gradient-descent)** - Optimization algorithm that minimizes cost function iteratively
- **[Normal Equation](../../GLOSSARY.md#normal-equation)** - Analytical solution: θ = (X^T × X)^(-1) × X^T × y

## 🗺️ Quick Access

| File | Description | Quick Run |
|------|-------------|----------|
| [`implementation.py`](./implementation.py) | From-scratch implementation | `uv run python implementation.py` |
| [`sklearn_example.py`](./sklearn_example.py) | Scikit-learn examples | `uv run python sklearn_example.py` |
| [`exercise.ipynb`](./exercise.ipynb) | Interactive exercises (Bahasa Indonesia) | `uv run jupyter lab exercise.ipynb` |
| [`exercise_en.ipynb`](./exercise_en.ipynb) | Interactive exercises (English) | `uv run jupyter lab exercise_en.ipynb` |
| [`exercise_EN.md`](./exercise_EN.md) | Exercises (English, Markdown) | Open and follow instructions |

## 📖 Theory and Concepts

### What is Linear Regression?
[**Linear Regression**](../../GLOSSARY.md#linear-model) is a [**supervised learning**](../../GLOSSARY.md#supervised-learning) [**algorithm**](../../GLOSSARY.md#algorithm) used to predict [**target values**](../../GLOSSARY.md#target-variable) (continuous) based on one or more input [**features**](../../GLOSSARY.md#feature). This algorithm finds the best linear relationship between input and output.

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
Linear regression uses **Mean Squared Error (MSE)** as the cost function:
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

## 🎯 Practical Scenarios

### 🏠 Scenario 1: House Price Prediction
**Dataset**: House features (size, bedrooms, location)
**Goal**: Predict house prices
**Run**: `uv run python sklearn_example.py --scenario house_prices`

**What you'll learn:**
- Feature scaling for multiple variables
- Handling categorical variables
- Model interpretation with coefficients

### 🌡️ Scenario 2: Temperature Prediction
**Dataset**: Weather features (humidity, pressure, wind speed)
**Goal**: Predict daily temperature
**Run**: `uv run python sklearn_example.py --scenario temperature`

**What you'll learn:**
- Time series data preparation
- Feature correlation analysis
- Seasonal trend modeling

### 💰 Scenario 3: Salary Estimation
**Dataset**: Employee data (experience, education, skills)
**Goal**: Estimate salary ranges
**Run**: `uv run python sklearn_example.py --scenario salary`

**What you'll learn:**
- Polynomial features
- Cross-validation techniques
- Confidence intervals

### 📈 Scenario 4: Stock Price Trend
**Dataset**: Historical stock data
**Goal**: Simple trend analysis
**Run**: `uv run python implementation.py --scenario stocks`

**What you'll learn:**
- Gradient descent vs Normal equation
- Learning rate optimization
- Cost function visualization

### 🔬 Compare Methods
**Compare Normal Equation vs Gradient Descent:**
```bash
uv run python implementation.py --compare-methods
```

## 📝 Implementation Files

### [`implementation.py`](./implementation.py)
- **LinearRegressionFromScratch** class
- **Two optimization methods**: Normal Equation & Gradient Descent
- **Built-in scenarios**: All 4 practical examples above
- **Visualization tools**: Cost history, prediction plots

### [`sklearn_example.py`](./sklearn_example.py)
- **Production-ready examples** using scikit-learn
- **Advanced techniques**: Polynomial features, regularization
- **Model evaluation**: R², RMSE, MAE metrics
- **Cross-validation** and **hyperparameter tuning**

### [`exercise.ipynb`](./exercise.ipynb)
- **Interactive Jupyter notebook**
- **Step-by-step tutorials** for each scenario
- **Visualization exercises**
- **Real dataset exploration**

## 🎓 Tips for Practice

1. **Always visualize data** before modeling
2. **Check assumptions** of linear regression
3. **Handle outliers** if necessary
4. **Feature scaling** for multiple regression
5. **Validate model** using cross-validation

## 📚 References

- [Scikit-learn Linear Regression](https://scikit-learn.org/stable/modules/linear_model.html#ordinary-least-squares)
- [Introduction to Statistical Learning - Chapter 3](https://www.statlearning.com/)

## 🎓 Learning Progression

### Prerequisites
- Basic Python knowledge
- NumPy fundamentals ([NumPy Tutorial](https://numpy.org/doc/stable/user/quickstart.html))
- Basic statistics (mean, variance)

### After Linear Regression, explore:

1. **[Logistic Regression](../02_logistic_regression/)** - Classification extension
2. **[Polynomial Features](../../06_utils/)** - Non-linear relationships  
3. **[Model Evaluation](../../04_advanced_topics/03_model_evaluation/)** - Advanced metrics
4. **[Regularization](../05_svm/)** - Ridge/Lasso regression

### Related Algorithms

| Algorithm | Similarity | When to Use |
|-----------|------------|-------------|
| [**Logistic Regression**](../02_logistic_regression/) | Same math, different output | Classification problems |
| [**SVM Regression**](../05_svm/) | Linear boundary | Robust to outliers |
| [**Neural Networks**](../08_neural_networks/) | Extension | Complex non-linear patterns |
| [**Random Forest**](../04_random_forest/) | Ensemble approach | Feature importance |

---
**➡️ Next Step**: Continue to **[Logistic Regression](../02_logistic_regression/)** for classification problems!
