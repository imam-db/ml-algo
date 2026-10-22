# Gradient Boosting

[🇮🇩 Baca dalam Bahasa Indonesia](README.id.md) | 🇬🇧 English

🏠 Back to Supervised Index: `../README.md`

Gradient Boosting is a powerful ensemble learning technique that builds models sequentially, where each new model corrects the errors made by previous models. It's one of the most effective machine learning algorithms for both regression and classification tasks.

## 📚 Table of Contents
- [Theory & Concepts](#-theory--concepts)
- [Mathematical Foundation](#-mathematical-foundation)
- [How Gradient Boosting Works](#-how-gradient-boosting-works)
- [Bias-Variance Trade-off](#-bias-variance-trade-off)
- [Learning Rate vs Number of Trees](#-learning-rate-vs-number-of-trees)
- [Advantages & Disadvantages](#-advantages--disadvantages)
- [When to Use](#-when-to-use)
- [Files in This Directory](#-files-in-this-directory)
- [Related Algorithms](#-related-algorithms)

## 🧠 Theory & Concepts

### What is Gradient Boosting?

Gradient Boosting is an ensemble method that combines multiple weak learners (typically shallow decision trees) into a strong predictive model. Unlike Random Forest which builds trees independently in parallel, Gradient Boosting builds trees sequentially, where each tree tries to correct the mistakes of the previous ensemble.

### Key Concepts

1. **Sequential Learning**: Models are built one after another, each focusing on the errors of the previous models
2. **Gradient Descent in Function Space**: The algorithm performs gradient descent to minimize a loss function
3. **Weak Learners**: Typically shallow decision trees (depth 3-8) that are only slightly better than random guessing
4. **Residual Fitting**: Each new tree fits the residuals (negative gradients) of the previous predictions
5. **Additive Model**: Final prediction is the sum of all tree predictions
6. **Shrinkage (Learning Rate)**: Each tree's contribution is scaled down to prevent overfitting

### Intuition: Fitting the Residuals Step by Step

Think of Gradient Boosting as a team of specialists working together:

```
Initial Prediction:
- Start with a simple baseline (e.g., mean of targets)
- Error = Actual - Prediction

Tree 1:
- Learns to predict the error
- New Prediction = Baseline + Learning_Rate × Tree1

Tree 2:
- Learns to predict the remaining error
- New Prediction = Previous + Learning_Rate × Tree2

Tree 3, 4, 5...
- Each tree corrects remaining errors
- Gradually improve the ensemble prediction
```

**Example**: Predicting house prices
- Tree 1: Learns that location affects price (+$50k for good neighborhood)
- Tree 2: Learns that size matters (+$30k per bedroom)
- Tree 3: Learns age effect (-$10k for older houses)
- Tree 4: Captures non-linear effects and interactions
- Final prediction: Sum of all corrections

## 📐 Mathematical Foundation

### General Framework

Gradient Boosting builds an additive model:

```
F(x) = F₀(x) + η·h₁(x) + η·h₂(x) + ... + η·hₘ(x)

Where:
- F(x) = final prediction
- F₀(x) = initial prediction (constant)
- hₜ(x) = tree t's prediction
- η = learning rate (shrinkage)
- M = number of trees
```

### Algorithm Steps

```
1. Initialize model with constant:
   F₀(x) = argmin_γ Σᵢ L(yᵢ, γ)
   (For regression: mean of y; for classification: log-odds)

2. For t = 1 to M:
   
   a) Calculate negative gradients (pseudo-residuals):
      rᵢₜ = -[∂L(yᵢ, F(xᵢ))/∂F(xᵢ)]_{F=Fₜ₋₁}
   
   b) Fit regression tree hₜ to pseudo-residuals:
      hₜ = argmin_h Σᵢ (rᵢₜ - h(xᵢ))²
   
   c) Update model:
      Fₜ(x) = Fₜ₋₁(x) + η·hₜ(x)

3. Output final model:
   F_M(x)
```

### Loss Functions and Gradients

#### Regression (Squared Loss)

```
Loss: L(y, F(x)) = ½(y - F(x))²

Negative Gradient: -∂L/∂F = y - F(x) = residual

Interpretation: Directly fit the residuals
```

**Example Calculation**:
```
Actual value: y = 100
Current prediction: F(x) = 85
Residual: 100 - 85 = 15
→ Next tree learns to predict +15
```

#### Binary Classification (Log Loss)

```
Loss: L(y, F(x)) = -[y·log(p) + (1-y)·log(1-p)]
where p = 1/(1 + e^(-F(x)))

Negative Gradient: -∂L/∂F = y - p

Interpretation: Fit the difference between true labels and probabilities
```

**Example Calculation**:
```
Actual label: y = 1 (churn)
Current log-odds: F(x) = -0.5
Current probability: p = 1/(1+e^0.5) = 0.378
Residual: 1 - 0.378 = 0.622
→ Next tree learns to increase log-odds by +0.622
```

### Gradient Descent Analogy

Traditional Gradient Descent (parameter space):
```
θ_new = θ_old - η·∇L(θ)
```

Gradient Boosting (function space):
```
F_new(x) = F_old(x) - η·[∂L/∂F]
         = F_old(x) + η·h(x)
where h(x) fits the negative gradient
```

The key insight: Instead of updating parameters, we add a new function that points in the negative gradient direction!

## 🎯 How Gradient Boosting Works

### Step-by-Step Example: Regression

Let's predict store sales with features [is_weekend, has_promotion]:

**Dataset**:
```
Sample 1: [1, 0] → Actual Sales: 1500
Sample 2: [0, 1] → Actual Sales: 1300
Sample 3: [1, 1] → Actual Sales: 1800
```

**Iteration 0** (Initialization):
```
F₀(x) = mean(sales) = (1500 + 1300 + 1800) / 3 = 1533

Predictions: [1533, 1533, 1533]
Residuals: [1500-1533, 1300-1533, 1800-1533]
         = [-33, -233, 267]
```

**Iteration 1** (First Tree):
```
Tree 1 learns to predict residuals:
- Splits on has_promotion
- Left (no promotion): predict -133
- Right (has promotion): predict 17

Update with learning_rate = 0.1:
F₁(x) = F₀(x) + 0.1 × Tree1(x)

New predictions: [1533-3.3, 1533+1.7, 1533+1.7]
                = [1529.7, 1534.7, 1534.7]

New residuals: [-29.7, -234.7, 265.3]
```

**Iteration 2** (Second Tree):
```
Tree 2 learns from remaining residuals:
- Further refines predictions
- F₂(x) = F₁(x) + 0.1 × Tree2(x)
```

Continue for M iterations...

### Stochastic Gradient Boosting (Subsampling)

Regular GB uses all data for each tree. Stochastic GB randomly samples:

```
For each tree:
1. Randomly sample (e.g., 80%) of training data without replacement
2. Train tree on this subsample
3. Use full tree for predictions

Benefits:
- Reduces overfitting
- Speeds up training
- Introduces randomness like Random Forest
```

## ⚖️ Bias-Variance Trade-off

Gradient Boosting explicitly controls the bias-variance trade-off through its hyperparameters:

### Individual Trees (High Bias, Low Variance)

```
Single shallow tree:
- High bias: Underfits, simple model
- Low variance: Stable predictions
- Poor performance alone
```

### Ensemble Effect (Reduces Bias)

```
Adding trees sequentially:
- Reduces bias: Corrects systematic errors
- Each tree captures patterns previous trees missed
- Ensemble becomes more complex and accurate
```

### Role of Hyperparameters

**Learning Rate (η)**:
```
Large learning rate (e.g., 1.0):
- Fast learning, few trees needed
- Risk of overshooting optimal solution
- Higher variance

Small learning rate (e.g., 0.01):
- Slow learning, many trees needed
- More stable, better generalization
- Lower variance
```

**Number of Trees (M)**:
```
Too few trees:
- High bias: Model underfits
- Training and test error both high

Optimal number:
- Low bias and variance
- Best test performance

Too many trees:
- Low bias but HIGH variance
- Overfits: low training error, high test error
```

**Tree Depth**:
```
Shallow trees (depth 2-3):
- High bias, low variance
- Less prone to overfitting
- Faster training

Deep trees (depth 8-10):
- Low bias, high variance
- Can overfit easily
- Capture complex interactions
```

## 📊 Learning Rate vs Number of Trees

The most important trade-off in Gradient Boosting:

```
Learning Rate × Number of Trees ≈ Constant (for similar performance)

Rule of Thumb:
- η = 0.01 → Need ~1000-5000 trees
- η = 0.1  → Need ~100-500 trees
- η = 0.3  → Need ~50-200 trees
- η = 1.0  → Need ~10-50 trees
```

### Why Lower Learning Rate is Better

```
Small learning rate + Many trees:
✅ Better generalization
✅ Smoother optimization path
✅ Less sensitive to individual tree errors
✅ Better chance of finding global optimum
❌ Slower training

Large learning rate + Few trees:
✅ Fast training
❌ May overshoot optimal solution
❌ More sensitive to noise
❌ Higher variance
```

### Visual Intuition

```
Learning Rate = 1.0:        Learning Rate = 0.1:
━━━━━━━━━━━━━━━━━          ━━━━━━━━━━━━━━━━━
   ▲                          ▲
   │  *                       │    ___---*
   │   ╲                      │  _/
   │    ╲ (overshoots)        │ / (smooth convergence)
   │     *                    │*
   │      ╲                   │
   └───────────►             └───────────►
   Iterations                 Iterations
```

## ✅ Advantages & Disadvantages

### ✅ Advantages

- **State-of-the-art Performance**: Often achieves best results on structured/tabular data
- **Handles Mixed Data Types**: Works with numerical and categorical features
- **Captures Non-linear Relationships**: Trees naturally handle non-linearities
- **Feature Interactions**: Automatically discovers feature interactions
- **Robust to Outliers**: Tree splits are based on ordering, not absolute values
- **Feature Importance**: Provides interpretable feature importance scores
- **Flexible Loss Functions**: Can optimize any differentiable loss function
- **No Feature Scaling**: Works with features on different scales
- **Missing Value Handling**: Can learn optimal missing value directions

### ❌ Disadvantages

- **Sequential Training**: Cannot parallelize tree building (unlike Random Forest)
- **Prone to Overfitting**: Especially with many trees and high learning rate
- **Slow Prediction**: Must evaluate all trees for each prediction
- **Hyperparameter Sensitive**: Requires careful tuning for optimal performance
- **Less Interpretable**: Ensemble of trees harder to understand than single tree
- **Memory Intensive**: Must store all trees
- **Training Time**: Can be slow with many trees or large datasets
- **Extrapolation Issues**: Cannot predict beyond training data range

## 🎯 When to Use

### ✅ Use Gradient Boosting When

- You need the best possible accuracy on structured/tabular data
- You have sufficient computational resources for training
- Dataset is small to medium sized (< 1M rows typically)
- Relationships between features and target are complex and non-linear
- You can invest time in hyperparameter tuning
- Prediction speed is not critical
- You need feature importance for interpretation
- Working with classification or regression problems

### ❌ Consider Alternatives When

- **Need fast training on large data**: Use LightGBM or XGBoost (optimized versions)
- **Need high interpretability**: Use single Decision Tree or Linear Models
- **Need parallel training**: Use Random Forest
- **Real-time predictions critical**: Consider simpler models
- **Very small dataset (< 100 samples)**: Use simpler models to avoid overfitting
- **High-dimensional sparse data (text, images)**: Consider neural networks or specialized algorithms

### Real-World Applications

**Finance**:
- Credit risk assessment
- Fraud detection
- Customer lifetime value prediction
- Algorithmic trading strategies

**E-commerce**:
- Product recommendation ranking
- Customer churn prediction
- Price optimization
- Conversion rate prediction

**Healthcare**:
- Disease diagnosis
- Readmission risk prediction
- Treatment response prediction
- Medical claim fraud detection

**Marketing**:
- Customer segmentation
- Campaign response prediction
- Lead scoring
- Click-through rate prediction

**Technology**:
- Search ranking (e.g., web search, app store)
- Anomaly detection
- Load forecasting
- Quality prediction in manufacturing

## 📁 Files in This Directory

- **`README.md`** (this file) - Comprehensive theory and concepts in English
- **`README.id.md`** - Dokumentasi lengkap dalam Bahasa Indonesia
- **`implementation.py`** - Gradient Boosting from scratch with NumPy
- **`sklearn_example.py`** - Practical examples using scikit-learn
- **`exercise.ipynb`** - Hands-on exercises and experiments

## 🚀 Quick Start

1. **Understand the Theory**: Read this README thoroughly
2. **See Implementation**: Check `implementation.py` for from-scratch code
3. **Try Examples**: Run `sklearn_example.py` for practical applications
4. **Practice**: Complete `exercise.ipynb` exercises
5. **Experiment**: Try on your own datasets

## 🎛️ Important Hyperparameters

### Scikit-learn Parameters

```python
from sklearn.ensemble import GradientBoostingRegressor, GradientBoostingClassifier

GradientBoostingRegressor(
    n_estimators=100,          # Number of boosting stages (trees)
    learning_rate=0.1,         # Shrinkage parameter
    max_depth=3,               # Maximum depth of trees
    min_samples_split=2,       # Min samples to split node
    min_samples_leaf=1,        # Min samples in leaf
    subsample=1.0,             # Fraction of samples for each tree
    max_features=None,         # Features to consider for splits
    loss='squared_error',      # Loss function
    criterion='friedman_mse',  # Split quality measure
    random_state=None,         # Random seed
    verbose=0,                 # Verbosity level
    warm_start=False,          # Add trees incrementally
    validation_fraction=0.1,   # Fraction for early stopping
    n_iter_no_change=None,     # Early stopping rounds
)
```

### Hyperparameter Tuning Priority

**1. Start with these (most important)**:
```python
n_estimators = 100-500       # More trees with lower learning rate
learning_rate = 0.01-0.1     # Smaller is better, but slower
max_depth = 3-8              # Shallow trees for GB
```

**2. Fine-tune these**:
```python
subsample = 0.5-1.0          # 0.8 often works well
min_samples_split = 2-20     # Prevent overfitting
min_samples_leaf = 1-10      # Smooth predictions
```

**3. Advanced options**:
```python
max_features = 'sqrt', 'log2', None  # Add randomness
loss = 'squared_error', 'huber', etc # Robust to outliers
validation_fraction = 0.1-0.2        # For early stopping
```

### Recommended Starting Point

**For most problems**:
```python
GradientBoostingRegressor(
    n_estimators=200,
    learning_rate=0.1,
    max_depth=3,
    subsample=0.8,
    min_samples_split=10,
    random_state=42
)
```

Then use GridSearchCV or RandomizedSearchCV to optimize!

## 🔗 Related Algorithms

### Bridge Between Algorithms

Gradient Boosting sits at the intersection of several concepts:

```
Decision Trees ─────► Random Forest ─────► Gradient Boosting ─────► XGBoost
                      (parallel)           (sequential)            (optimized)
```

**Decision Trees** ([03_decision_trees](../03_decision_trees/)): 
- Single tree that can overfit
- Used as weak learners in GB
- GB uses many shallow trees

**Random Forest** ([04_random_forest](../04_random_forest/)):
- Builds trees independently in parallel
- Uses bootstrap sampling + random features
- GB builds trees sequentially, each fixing errors
- Both are ensemble methods but different strategies

**XGBoost** ([../../04_advanced_topics/04_xgboost/](../../04_advanced_topics/04_xgboost/)):
- Highly optimized implementation of gradient boosting
- Adds regularization, parallel processing, handling of missing values
- Same core algorithm but with engineering improvements
- 10-100x faster training than sklearn GradientBoosting

### Comparison Table

| Feature | Random Forest | Gradient Boosting | XGBoost |
|---------|--------------|-------------------|---------|
| Tree Building | Parallel | Sequential | Sequential (but parallelized) |
| Tree Depth | Deep (unpruned) | Shallow (3-8) | Shallow (3-8) |
| Learning Strategy | Bagging | Boosting | Boosting + Regularization |
| Overfitting Risk | Low | Medium-High | Low-Medium |
| Training Speed | Fast | Medium | Fast |
| Prediction Speed | Fast | Medium | Fast |
| Accuracy | Very Good | Excellent | Excellent |
| Hyperparameter Sensitivity | Low | High | Medium |

### Other Related Algorithms

- **AdaBoost**: Earlier boosting algorithm, adjusts sample weights instead of fitting residuals
- **LightGBM**: Microsoft's optimized GB, even faster than XGBoost on large data
- **CatBoost**: Yandex's GB optimized for categorical features
- **Histogram-based GB**: Sklearn's fast GB for large datasets

---

**Next Steps**: 
- Dive into `implementation.py` to see how it works under the hood
- Run `sklearn_example.py` to see practical applications
- Complete `exercise.ipynb` to solidify your understanding
- Then explore **XGBoost** for production-grade gradient boosting!
