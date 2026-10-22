# Naive Bayes

**🇬🇧 English | [🇮🇩 Bahasa Indonesia](README.id.md)**

**🏠 [Back to Main](../../README.md)** | **📁 [Supervised Learning](../README.md)**

## 🗺️ Quick Access

| File | Description | Run |
|------|-------------|-----|
| [`implementation.py`](./implementation.py) | Implementation from scratch | `python3 implementation.py` |
| [`sklearn_example.py`](./sklearn_example.py) | Scikit-learn examples | `python3 sklearn_example.py` |
| [`exercise.ipynb`](./exercise.ipynb) | Interactive exercises | `jupyter lab exercise.ipynb` |

## 📖 Theory and Concepts

### What is Naive Bayes?

**Naive Bayes** is a classification algorithm based on **Bayes' Theorem** with a "naive" assumption that all features are independent of each other. Although this assumption is often unrealistic, Naive Bayes remains highly effective for many real-world applications.

**SPG (Sales Promoter Girl) Analogy:**
An SPG wants to predict whether a customer will buy a product based on customer characteristics (age, gender, visit time, etc.). Naive Bayes looks at how often these characteristic combinations appear in customers who buy vs don't buy, then makes predictions based on probabilities.

### Bayes' Theorem

Basic formula:

```
P(Class|Features) = P(Features|Class) × P(Class) / P(Features)
```

In practice:

```
P(y|x₁, x₂, ..., xₙ) = P(x₁, x₂, ..., xₙ|y) × P(y) / P(x₁, x₂, ..., xₙ)
```

### Naive Assumption

Naive assumption (independent features):

```
P(x₁, x₂, ..., xₙ|y) = P(x₁|y) × P(x₂|y) × ... × P(xₙ|y)
```

So Bayes' Theorem becomes:

```
P(y|x₁, ..., xₙ) ∝ P(y) × P(x₁|y) × P(x₂|y) × ... × P(xₙ|y)
```

## 🔢 Types of Naive Bayes

### 1. Gaussian Naive Bayes

**For:** Continuous features

**Assumption:** Features follow a Gaussian (normal) distribution

**Likelihood Formula:**
```
P(xᵢ|y) = (1/√(2πσᵧ²)) × exp(-(xᵢ-μᵧ)²/(2σᵧ²))
```

Where:
- `μᵧ` = mean of feature for class y
- `σᵧ²` = variance of feature for class y

**When to use:**
- Features are continuous numbers (height, weight, price, etc.)
- Iris classification, medical diagnosis
- Sensor data analysis

**SPG Use Case Example:**
```python
# Predict customer segment based on:
# - Age (continuous)
# - Income (continuous)
# - Spending score (continuous)
X = [[25, 50000, 65], [45, 80000, 80], ...]
y = ['budget', 'premium', ...]
```

### 2. Multinomial Naive Bayes

**For:** Discrete count data (frequency data)

**Assumption:** Features are counts or frequencies

**Likelihood Formula:**
```
P(xᵢ|y) = (countᵢᵧ + α) / (total_countᵧ + α × n_features)
```

Where:
- `α` = smoothing parameter (Laplace/Lidstone)
- `countᵢᵧ` = count of feature i for class y

**When to use:**
- Text classification (word counts, TF-IDF)
- Document categorization
- Spam filtering

**SPG Use Case Example:**
```python
# Predict customer category based on:
# - Number of visits per week
# - Number of products purchased per category
# - Purchase frequency
X = [[5, 2, 1, 3], [1, 0, 5, 2], ...]  # counts
y = ['frequent', 'occasional', ...]
```

### 3. Bernoulli Naive Bayes (not implemented)

**For:** Binary features (0/1, True/False)

**Example:** Email spam (word present or not)

## 💡 Advantages and Disadvantages

### ✅ Advantages

1. **Fast Training and Prediction**
   - Training: O(n × d) - very fast
   - Prediction: O(c × d) - real-time capable

2. **Works Well with Small Data**
   - Doesn't need much training data
   - Good with imbalanced classes

3. **Simple and Interpretable**
   - Easy to understand
   - Probability outputs can be interpreted

4. **Handles High Dimensions**
   - Works with many features
   - Common for text classification

5. **Not Sensitive to Irrelevant Features**
   - Independence assumption helps

### ❌ Disadvantages

1. **Independence Assumption**
   - Features are rarely truly independent
   - Can produce overconfident predictions

2. **Zero Frequency Problem**
   - If feature doesn't appear in training data
   - Solved with smoothing (Laplace)

3. **Bad Estimator**
   - Probability outputs are often too extreme
   - Don't rely on exact probability values

4. **Can't Learn Feature Interactions**
   - Cannot capture correlations between features

## 🎯 When to Use Naive Bayes?

### ✅ Good for:
- **Text classification**
  - Spam detection, sentiment analysis
  - Document categorization
- **Real-time prediction**
  - Very fast, suitable for production
- **Multi-class problems**
  - Natural extension
- **Baseline model**
  - Quick benchmark
- **High-dimensional data**
  - Many features
- **Small training data**
  - Works with limited data

### ❌ Less suitable for:
- **Highly correlated features**
  - Violates independence assumption
- **Regression problems**
  - Naive Bayes is for classification only
- **Feature interactions are important**
  - Cannot capture complex relationships
- **Need precise probability estimates**
  - Probabilities are often overconfident

## 🔧 Implementation

### Gaussian Naive Bayes

```python
from implementation import GaussianNaiveBayes

# Create and train
gnb = GaussianNaiveBayes(var_smoothing=1e-9)
gnb.fit(X_train, y_train)

# Predict
predictions = gnb.predict(X_test)
probabilities = gnb.predict_proba(X_test)
log_probs = gnb.predict_log_proba(X_test)

# Score
accuracy = gnb.score(X_test, y_test)
```

### Multinomial Naive Bayes

```python
from implementation import MultinomialNaiveBayes

# Create and train (with Laplace smoothing)
mnb = MultinomialNaiveBayes(alpha=1.0)
mnb.fit(X_train_counts, y_train)

# Predict
predictions = mnb.predict(X_test_counts)
probabilities = mnb.predict_proba(X_test_counts)

# Score
accuracy = mnb.score(X_test_counts, y_test)
```

## 📊 Laplace Smoothing

**Problem:** Zero frequency for unseen features

**Example:**
```
Training: "Buy now" → spam
          "Meeting tomorrow" → not spam

Test: "Buy tomorrow" → P("tomorrow"|spam) = 0!
```

**Solution: Laplace Smoothing**

```
P(word|class) = (count(word, class) + α) / (total_count(class) + α × vocab_size)
```

**Parameter α:**
- α = 0: No smoothing (risky!)
- α = 1: Laplace smoothing (standard)
- α > 1: More smoothing (for very sparse data)
- α < 1: Less smoothing

**Tips:**
- Start with α = 1.0
- Tune with cross-validation
- Larger α for smaller datasets

## 🔍 Hyperparameter Tuning

### Gaussian Naive Bayes

```python
# var_smoothing: portion of largest variance added
# Range: 1e-10 to 1e-7

for var_smooth in [1e-10, 1e-9, 1e-8, 1e-7]:
    gnb = GaussianNaiveBayes(var_smoothing=var_smooth)
    gnb.fit(X_train, y_train)
    score = gnb.score(X_val, y_val)
    print(f"var_smoothing={var_smooth}: {score:.4f}")
```

### Multinomial Naive Bayes

```python
# alpha: smoothing parameter
# Range: 0.01 to 10.0

for alpha in [0.01, 0.1, 1.0, 5.0, 10.0]:
    mnb = MultinomialNaiveBayes(alpha=alpha)
    mnb.fit(X_train, y_train)
    score = mnb.score(X_val, y_val)
    print(f"alpha={alpha}: {score:.4f}")
```

## 💻 Example Use Case: Customer Purchase Prediction

### Gaussian NB: Continuous Features

```python
# Features: age, income, spending_score
# Target: will_purchase (yes/no)

from implementation import GaussianNaiveBayes
import numpy as np

# SPG customer data
X = np.array([
    [25, 45000, 60],  # young, low income, medium spending
    [45, 95000, 85],  # older, high income, high spending
    [30, 55000, 40],  # young adult, medium income, low spending
    # ... more data
])

y = np.array([0, 1, 0, ...])  # 0=no purchase, 1=purchase

# Train
gnb = GaussianNaiveBayes()
gnb.fit(X, y)

# Predict new customer
new_customer = [[35, 70000, 75]]
prediction = gnb.predict(new_customer)
probability = gnb.predict_proba(new_customer)

print(f"Prediction: {'Purchase' if prediction[0] == 1 else 'No Purchase'}")
print(f"Probability: {probability[0]}")
```

### Multinomial NB: Count Features

```python
# Features: visit counts per category (electronics, fashion, food, home)
# Target: customer_type (browser, buyer, vip)

from implementation import MultinomialNaiveBayes

# Customer visit data
X_counts = np.array([
    [10, 2, 1, 0],   # lots of electronics browsing, rarely buy
    [2, 5, 8, 3],    # balanced, regular buyer
    [15, 20, 10, 8], # frequent visitor, VIP
    # ... more data
])

y = np.array(['browser', 'buyer', 'vip', ...])

# Train
mnb = MultinomialNaiveBayes(alpha=1.0)
mnb.fit(X_counts, y)

# Predict
new_customer_visits = [[8, 12, 6, 4]]
prediction = mnb.predict(new_customer_visits)
probability = mnb.predict_proba(new_customer_visits)

print(f"Customer Type: {prediction[0]}")
print(f"Probabilities: {probability[0]}")
```

## 📚 Comparison with Other Algorithms

| Aspect | Naive Bayes | Logistic Regression | Decision Tree | KNN |
|--------|-------------|-------------------|---------------|-----|
| Speed | Very Fast | Fast | Fast | Slow |
| Interpretability | High | High | Very High | Medium |
| Probabilistic | Yes | Yes | No (standard) | Yes |
| Feature Independence | Required | Not required | Not required | Not required |
| Small Data | Excellent | Good | Poor | Poor |
| High Dimensions | Excellent | Good | Poor | Poor |
| Text Classification | Excellent | Good | Poor | Poor |

## 🎓 Tips and Best Practices

### 1. Feature Preprocessing

**Gaussian NB:**
```python
# No scaling needed (uses mean/variance per class)
# But ensure no missing values
X = X.fillna(X.mean())
```

**Multinomial NB:**
```python
# Ensure non-negative values
X = np.abs(X)  # or X = X.clip(min=0)

# Normalize counts (optional, for TF-IDF)
from sklearn.preprocessing import Normalizer
normalizer = Normalizer()
X = normalizer.fit_transform(X)
```

### 2. Handling Imbalanced Classes

```python
# Naive Bayes naturally handles imbalance through priors
# Class priors automatically reflect class distribution

# Can also use class weights (manual)
# Oversample minority class in training data
```

### 3. Feature Selection

```python
# Naive Bayes works well even with many features
# But removing irrelevant features can improve:

from sklearn.feature_selection import SelectKBest, chi2

selector = SelectKBest(chi2, k=20)
X_selected = selector.fit_transform(X, y)
```

### 4. Probability Calibration

```python
# Naive Bayes probabilities often overconfident
# Use calibration for better probability estimates

from sklearn.calibration import CalibratedClassifierCV

# Wrap Naive Bayes with calibration
calibrated_nb = CalibratedClassifierCV(gnb, cv=5)
calibrated_nb.fit(X_train, y_train)
```

## 📚 Learning Resources

- [Naive Bayes - Scikit-learn](https://scikit-learn.org/stable/modules/naive_bayes.html)
- [StatQuest: Naive Bayes](https://www.youtube.com/watch?v=O2L2Uv9pdDA)
- [Andrew Ng ML Course - Naive Bayes](https://www.coursera.org/learn/machine-learning)

## 🎓 Exercises

See [`exercise.ipynb`](./exercise.ipynb) for hands-on practice:
1. Gaussian NB for customer segmentation
2. Multinomial NB for text classification
3. Comparing smoothing parameters
4. Feature importance analysis
5. Probability calibration

---

**Happy Learning Naive Bayes! 🎯**
