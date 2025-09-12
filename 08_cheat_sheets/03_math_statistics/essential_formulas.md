# 🧮 Essential Math & Statistics for ML

**🏠 [Cheat Sheets Home](../README.md)** | **📊 [Classification Metrics](../04_model_evaluation/)** | **🤖 [Algorithm Reference](../01_algorithms/)**

---

## 🎯 Quick Summary

Essential mathematical formulas, statistical concepts, and calculations commonly used in machine learning.

---

## 📊 Statistics Fundamentals

### **📈 Descriptive Statistics**

#### **Central Tendency**
```python
import numpy as np
import pandas as pd

# Mean (average)
mean = np.mean(data)
mean = data.mean()

# Median (middle value)
median = np.median(data)
median = data.median()

# Mode (most frequent)
from scipy import stats
mode = stats.mode(data)[0][0]
mode = data.mode()[0]  # pandas
```

**Formulas:**
- **Mean:** $\bar{x} = \frac{1}{n}\sum_{i=1}^{n} x_i$
- **Weighted Mean:** $\bar{x}_w = \frac{\sum_{i=1}^{n} w_i x_i}{\sum_{i=1}^{n} w_i}$

#### **Variability**
```python
# Variance
variance = np.var(data, ddof=1)  # Sample variance
variance = data.var()

# Standard Deviation
std_dev = np.std(data, ddof=1)  # Sample std
std_dev = data.std()

# Range
data_range = np.max(data) - np.min(data)
data_range = data.max() - data.min()

# Interquartile Range (IQR)
q75, q25 = np.percentile(data, [75, 25])
iqr = q75 - q25
```

**Formulas:**
- **Sample Variance:** $s^2 = \frac{1}{n-1}\sum_{i=1}^{n}(x_i - \bar{x})^2$
- **Population Variance:** $\sigma^2 = \frac{1}{n}\sum_{i=1}^{n}(x_i - \mu)^2$
- **Standard Deviation:** $s = \sqrt{s^2}$

#### **Shape of Distribution**
```python
from scipy import stats

# Skewness (asymmetry)
skewness = stats.skew(data)
skewness = data.skew()

# Kurtosis (tail heaviness)
kurtosis = stats.kurtosis(data)
kurtosis = data.kurtosis()
```

**Interpretation:**
- **Skewness:** 0 = symmetric, > 0 = right-skewed, < 0 = left-skewed
- **Kurtosis:** 0 = normal, > 0 = heavy tails, < 0 = light tails

---

### **🎯 Probability Distributions**

#### **Normal Distribution**
```python
from scipy import stats
import numpy as np

# Probability Density Function
x = 1.5
mean, std = 0, 1
pdf = stats.norm.pdf(x, mean, std)

# Cumulative Distribution Function
cdf = stats.norm.cdf(x, mean, std)

# Generate random samples
samples = np.random.normal(mean, std, 1000)

# Z-score (standardization)
z_score = (x - mean) / std
```

**Key Formulas:**
- **PDF:** $f(x) = \frac{1}{\sigma\sqrt{2\pi}} e^{-\frac{1}{2}\left(\frac{x-\mu}{\sigma}\right)^2}$
- **Z-score:** $z = \frac{x - \mu}{\sigma}$
- **68-95-99.7 Rule:** 68% within 1σ, 95% within 2σ, 99.7% within 3σ

#### **Central Limit Theorem**
```python
# Sample mean distribution
def sampling_distribution(population, sample_size, n_samples=1000):
    sample_means = []
    for _ in range(n_samples):
        sample = np.random.choice(population, sample_size)
        sample_means.append(np.mean(sample))
    return np.array(sample_means)

# Standard Error of the Mean
sem = std_dev / np.sqrt(sample_size)
```

**Key Points:**
- Sample means approach normal distribution as sample size increases
- **Standard Error:** $SE = \frac{\sigma}{\sqrt{n}}$

---

## 📊 Statistical Tests

### **🔍 Hypothesis Testing**

#### **t-Test**
```python
from scipy import stats

# One-sample t-test
t_stat, p_value = stats.ttest_1samp(data, population_mean)

# Two-sample t-test (independent)
t_stat, p_value = stats.ttest_ind(group1, group2)

# Paired t-test
t_stat, p_value = stats.ttest_rel(before, after)

# Interpretation
alpha = 0.05
if p_value < alpha:
    print("Reject null hypothesis")
else:
    print("Fail to reject null hypothesis")
```

**Formula (One-sample):** $t = \frac{\bar{x} - \mu}{s / \sqrt{n}}$

#### **Chi-Square Test**
```python
from scipy.stats import chi2_contingency

# Chi-square test of independence
chi2, p_value, dof, expected = chi2_contingency(contingency_table)

# Goodness of fit test
chi2, p_value = stats.chisquare(observed, expected)
```

#### **Correlation Tests**
```python
# Pearson correlation
corr_coef, p_value = stats.pearsonr(x, y)

# Spearman rank correlation (non-parametric)
spearman_corr, p_value = stats.spearmanr(x, y)
```

**Pearson Correlation:** $r = \frac{\sum(x_i - \bar{x})(y_i - \bar{y})}{\sqrt{\sum(x_i - \bar{x})^2 \sum(y_i - \bar{y})^2}}$

---

## 🔢 Linear Algebra Essentials

### **📊 Vectors and Matrices**

#### **Vector Operations**
```python
import numpy as np

# Vector creation
v1 = np.array([1, 2, 3])
v2 = np.array([4, 5, 6])

# Dot product
dot_product = np.dot(v1, v2)
dot_product = v1 @ v2  # Alternative syntax

# Cross product (3D vectors only)
cross_product = np.cross(v1, v2)

# Vector magnitude (L2 norm)
magnitude = np.linalg.norm(v1)
magnitude = np.sqrt(np.sum(v1**2))

# Unit vector (normalization)
unit_vector = v1 / np.linalg.norm(v1)
```

**Key Formulas:**
- **Dot Product:** $\mathbf{a} \cdot \mathbf{b} = \sum_{i=1}^{n} a_i b_i$
- **Magnitude:** $||\mathbf{v}|| = \sqrt{\sum_{i=1}^{n} v_i^2}$
- **Cosine Similarity:** $\cos(\theta) = \frac{\mathbf{a} \cdot \mathbf{b}}{||\mathbf{a}|| \cdot ||\mathbf{b}||}$

#### **Matrix Operations**
```python
# Matrix creation
A = np.array([[1, 2], [3, 4]])
B = np.array([[5, 6], [7, 8]])

# Matrix multiplication
C = A @ B  # Preferred
C = np.dot(A, B)  # Alternative

# Element-wise operations
element_wise = A * B  # Hadamard product
addition = A + B
subtraction = A - B

# Transpose
A_transpose = A.T
A_transpose = np.transpose(A)

# Inverse
A_inverse = np.linalg.inv(A)  # Only for square, invertible matrices

# Determinant
det = np.linalg.det(A)

# Eigenvalues and eigenvectors
eigenvalues, eigenvectors = np.linalg.eig(A)
```

---

### **📐 Distance Metrics**

```python
from scipy.spatial.distance import euclidean, manhattan, cosine
from sklearn.metrics.pairwise import euclidean_distances, cosine_similarity

# Euclidean distance
euclidean_dist = euclidean(point1, point2)
euclidean_dist = np.sqrt(np.sum((point1 - point2)**2))

# Manhattan distance (L1)
manhattan_dist = manhattan(point1, point2)
manhattan_dist = np.sum(np.abs(point1 - point2))

# Cosine distance
cosine_dist = cosine(point1, point2)

# For matrices
euclidean_matrix = euclidean_distances(X)
cosine_sim_matrix = cosine_similarity(X)
```

**Distance Formulas:**
- **Euclidean:** $d = \sqrt{\sum_{i=1}^{n}(x_i - y_i)^2}$
- **Manhattan:** $d = \sum_{i=1}^{n}|x_i - y_i|$
- **Cosine:** $d = 1 - \frac{\mathbf{x} \cdot \mathbf{y}}{||\mathbf{x}|| \cdot ||\mathbf{y}||}$

---

## 📈 Calculus for ML

### **🎯 Derivatives**

#### **Basic Derivatives**
Common derivatives used in ML:

**Power Rule:** $\frac{d}{dx}x^n = nx^{n-1}$

**Exponential:** $\frac{d}{dx}e^x = e^x$

**Logarithm:** $\frac{d}{dx}\ln(x) = \frac{1}{x}$

**Sigmoid:** $\frac{d}{dx}\sigma(x) = \sigma(x)(1-\sigma(x))$ where $\sigma(x) = \frac{1}{1+e^{-x}}$

#### **Chain Rule**
```python
# For composite functions f(g(x))
# df/dx = df/dg * dg/dx

# Example: derivative of (x² + 1)³
# Let f(u) = u³, g(x) = x² + 1
# df/dx = 3u² * 2x = 3(x² + 1)² * 2x = 6x(x² + 1)²
```

#### **Partial Derivatives**
```python
import sympy as sp

# Define symbols
x, y = sp.symbols('x y')
f = x**2 + 2*x*y + y**2

# Partial derivatives
df_dx = sp.diff(f, x)  # ∂f/∂x = 2x + 2y
df_dy = sp.diff(f, y)  # ∂f/∂y = 2x + 2y

print(f"∂f/∂x = {df_dx}")
print(f"∂f/∂y = {df_dy}")
```

#### **Gradient**
```python
# Gradient is vector of partial derivatives
# ∇f = [∂f/∂x₁, ∂f/∂x₂, ..., ∂f/∂xₙ]

# For f(x,y) = x² + y²
# ∇f = [2x, 2y]

# Numerical gradient approximation
def numerical_gradient(f, x, h=1e-5):
    grad = np.zeros_like(x)
    for i in range(len(x)):
        x_plus = x.copy()
        x_minus = x.copy()
        x_plus[i] += h
        x_minus[i] -= h
        grad[i] = (f(x_plus) - f(x_minus)) / (2 * h)
    return grad
```

---

## 🎯 Optimization

### **📉 Loss Functions**

#### **Regression Loss Functions**
```python
# Mean Squared Error
def mse(y_true, y_pred):
    return np.mean((y_true - y_pred)**2)

# Mean Absolute Error  
def mae(y_true, y_pred):
    return np.mean(np.abs(y_true - y_pred))

# Huber Loss (robust to outliers)
def huber_loss(y_true, y_pred, delta=1.0):
    error = y_true - y_pred
    condition = np.abs(error) <= delta
    squared_loss = 0.5 * error**2
    linear_loss = delta * np.abs(error) - 0.5 * delta**2
    return np.where(condition, squared_loss, linear_loss)
```

**Formulas:**
- **MSE:** $L = \frac{1}{n}\sum_{i=1}^{n}(y_i - \hat{y}_i)^2$
- **MAE:** $L = \frac{1}{n}\sum_{i=1}^{n}|y_i - \hat{y}_i|$

#### **Classification Loss Functions**
```python
# Binary Cross-Entropy
def binary_crossentropy(y_true, y_pred):
    epsilon = 1e-15  # Prevent log(0)
    y_pred = np.clip(y_pred, epsilon, 1 - epsilon)
    return -np.mean(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))

# Categorical Cross-Entropy
def categorical_crossentropy(y_true, y_pred):
    epsilon = 1e-15
    y_pred = np.clip(y_pred, epsilon, 1)
    return -np.sum(y_true * np.log(y_pred))

# Hinge Loss (SVM)
def hinge_loss(y_true, y_pred):
    return np.mean(np.maximum(0, 1 - y_true * y_pred))
```

**Formulas:**
- **Binary Cross-Entropy:** $L = -\frac{1}{n}\sum_{i=1}^{n}[y_i\log(\hat{y}_i) + (1-y_i)\log(1-\hat{y}_i)]$
- **Hinge Loss:** $L = \max(0, 1 - y \cdot \hat{y})$

---

### **⬇️ Gradient Descent**

#### **Basic Gradient Descent**
```python
def gradient_descent(X, y, learning_rate=0.01, epochs=1000):
    m, n = X.shape
    theta = np.random.randn(n)  # Initialize parameters
    
    for epoch in range(epochs):
        # Forward pass
        predictions = X @ theta
        
        # Compute loss
        loss = np.mean((predictions - y)**2)
        
        # Compute gradients
        gradients = (2/m) * X.T @ (predictions - y)
        
        # Update parameters
        theta -= learning_rate * gradients
        
        if epoch % 100 == 0:
            print(f"Epoch {epoch}, Loss: {loss:.4f}")
    
    return theta
```

**Update Rule:** $\theta_{new} = \theta_{old} - \alpha \nabla L(\theta)$

#### **Gradient Descent Variants**
```python
# Stochastic Gradient Descent (SGD)
def sgd(X, y, learning_rate=0.01, epochs=1000):
    m, n = X.shape
    theta = np.random.randn(n)
    
    for epoch in range(epochs):
        for i in range(m):
            # Single sample
            xi = X[i:i+1]
            yi = y[i:i+1]
            
            prediction = xi @ theta
            gradient = 2 * xi.T @ (prediction - yi)
            theta -= learning_rate * gradient
    
    return theta

# Mini-batch Gradient Descent
def mini_batch_gd(X, y, batch_size=32, learning_rate=0.01, epochs=1000):
    m, n = X.shape
    theta = np.random.randn(n)
    
    for epoch in range(epochs):
        # Shuffle data
        indices = np.random.permutation(m)
        X_shuffled = X[indices]
        y_shuffled = y[indices]
        
        # Process mini-batches
        for i in range(0, m, batch_size):
            xi = X_shuffled[i:i+batch_size]
            yi = y_shuffled[i:i+batch_size]
            
            predictions = xi @ theta
            gradients = (2/batch_size) * xi.T @ (predictions - yi)
            theta -= learning_rate * gradients
    
    return theta
```

---

## 🎯 Information Theory

### **📊 Entropy and Information**

#### **Shannon Entropy**
```python
def entropy(probabilities):
    """Calculate Shannon entropy"""
    # Remove zero probabilities
    p = probabilities[probabilities > 0]
    return -np.sum(p * np.log2(p))

# For binary classification
def binary_entropy(p):
    """Binary entropy for probability p"""
    if p == 0 or p == 1:
        return 0
    return -(p * np.log2(p) + (1-p) * np.log2(1-p))

# Example
probabilities = np.array([0.5, 0.3, 0.2])
h = entropy(probabilities)
print(f"Entropy: {h:.3f} bits")
```

**Formula:** $H(X) = -\sum_{i=1}^{n} P(x_i) \log_2 P(x_i)$

#### **Mutual Information**
```python
from sklearn.metrics import mutual_info_score

# Mutual information between two variables
mi = mutual_info_score(X, y)

# Normalized mutual information
from sklearn.metrics import normalized_mutual_info_score
nmi = normalized_mutual_info_score(X, y)
```

**Formula:** $I(X;Y) = \sum_{x,y} P(x,y) \log \frac{P(x,y)}{P(x)P(y)}$

---

## 📊 Bayesian Statistics

### **🎯 Bayes' Theorem**
```python
# P(A|B) = P(B|A) * P(A) / P(B)

def bayes_theorem(prior, likelihood, evidence):
    """
    Calculate posterior probability using Bayes' theorem
    
    prior: P(A)
    likelihood: P(B|A) 
    evidence: P(B)
    """
    posterior = (likelihood * prior) / evidence
    return posterior

# Example: Medical diagnosis
prior_disease = 0.01        # P(Disease) = 1%
test_sensitivity = 0.95     # P(Test+|Disease) = 95%
test_specificity = 0.90     # P(Test-|No Disease) = 90%

# P(Test+) = P(Test+|Disease)*P(Disease) + P(Test+|No Disease)*P(No Disease)
evidence = test_sensitivity * prior_disease + (1 - test_specificity) * (1 - prior_disease)

# P(Disease|Test+)
posterior = bayes_theorem(prior_disease, test_sensitivity, evidence)
print(f"Probability of disease given positive test: {posterior:.3f}")
```

---

## 🔗 Quick Reference Tables

### **📊 Common Distributions**

| Distribution | Parameters | Mean | Variance | Use Case |
|-------------|------------|------|----------|----------|
| **Normal** | μ, σ² | μ | σ² | Continuous data, errors |
| **Bernoulli** | p | p | p(1-p) | Binary outcomes |
| **Binomial** | n, p | np | np(1-p) | Count successes |
| **Poisson** | λ | λ | λ | Rare events |
| **Uniform** | a, b | (a+b)/2 | (b-a)²/12 | Random sampling |

### **📈 Activation Functions**

| Function | Formula | Derivative | Range |
|----------|---------|------------|-------|
| **Sigmoid** | $\frac{1}{1+e^{-x}}$ | $\sigma(x)(1-\sigma(x))$ | (0,1) |
| **Tanh** | $\frac{e^x-e^{-x}}{e^x+e^{-x}}$ | $1-\tanh^2(x)$ | (-1,1) |
| **ReLU** | $\max(0,x)$ | $1$ if $x>0$, $0$ else | [0,∞) |
| **Softmax** | $\frac{e^{x_i}}{\sum e^{x_j}}$ | $\sigma_i(1-\sigma_i)$ | (0,1) |

---

## 💡 Implementation Tips

### **🚀 Numerical Stability**
```python
# Avoid numerical issues
def log_sum_exp(x):
    """Numerically stable log-sum-exp"""
    max_x = np.max(x)
    return max_x + np.log(np.sum(np.exp(x - max_x)))

def softmax_stable(x):
    """Numerically stable softmax"""
    exp_x = np.exp(x - np.max(x))
    return exp_x / np.sum(exp_x)

# Avoid log(0) by adding small epsilon
epsilon = 1e-15
safe_log = np.log(np.maximum(x, epsilon))
```

### **⚡ Vectorization Examples**
```python
# Instead of loops, use vectorized operations

# ❌ Slow: Loop-based computation
def slow_distance(X1, X2):
    distances = []
    for i in range(len(X1)):
        for j in range(len(X2)):
            dist = np.sqrt(np.sum((X1[i] - X2[j])**2))
            distances.append(dist)
    return np.array(distances).reshape(len(X1), len(X2))

# ✅ Fast: Vectorized computation
def fast_distance(X1, X2):
    # Broadcasting magic
    return np.sqrt(np.sum((X1[:, np.newaxis] - X2)**2, axis=2))

# Even faster with scipy
from scipy.spatial.distance import cdist
distances = cdist(X1, X2, metric='euclidean')
```

---

## 🔗 Related Cheat Sheets

- **[Algorithm Selection](../01_algorithms/algorithm_selection_flowchart.md)** - Choose algorithms based on math requirements
- **[Classification Metrics](../04_model_evaluation/classification_metrics.md)** - Apply statistical concepts to evaluation
- **[Scikit-Learn Quick Reference](../02_python_sklearn/sklearn_quick_reference.md)** - Implement mathematical concepts
- **[Interactive Tools](../../06_playground/)** - Practice mathematical concepts hands-on

---

## 💡 Key Takeaways

### **🎯 Essential Concepts**
1. **Statistics** - Understand your data distribution
2. **Linear Algebra** - Foundation for most ML algorithms
3. **Calculus** - Required for optimization and understanding gradients
4. **Probability** - Uncertainty quantification and Bayesian methods
5. **Information Theory** - Feature selection and model complexity

### **🚨 Common Pitfalls**
- ❌ Ignoring numerical stability (overflow/underflow)
- ❌ Using inappropriate statistical tests
- ❌ Misunderstanding probability vs. likelihood
- ❌ Not vectorizing computations (slow code)
- ❌ Confusing population vs. sample statistics

---

**📋 Keep this reference handy** for mathematical foundations! **🔍 Use Ctrl+F** to find specific formulas quickly.

**🏠 [Back to Cheat Sheets](../README.md)** | **🎮 [Try Interactive Tools](../../06_playground/)** | **📊 [Metrics Guide](../04_model_evaluation/classification_metrics.md)**