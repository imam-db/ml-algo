# 🤖 ML Algorithms Quick Reference

**🏠 [Cheat Sheets Home](../README.md)** | **🧭 [Algorithm Selection](algorithm_selection_flowchart.md)** | **🐍 [Scikit-Learn Reference](../02_python_sklearn/)**

---

## 🎯 Quick Summary

Essential machine learning algorithms with key concepts, parameters, and code examples for rapid reference.

---

## 📊 Classification Algorithms

### **📈 Logistic Regression**

**🔍 When to Use:** Linear relationships, interpretable results, baseline model  
**⚡ Speed:** Very Fast | **🎯 Accuracy:** Medium | **🔍 Interpretability:** High

```python
from sklearn.linear_model import LogisticRegression

# Basic usage
clf = LogisticRegression(random_state=42)
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)
y_prob = clf.predict_proba(X_test)  # Probabilities

# Key parameters
clf = LogisticRegression(
    C=1.0,              # Regularization strength (lower = more regularization)
    penalty='l2',       # 'l1', 'l2', 'elasticnet'
    max_iter=100,       # Max iterations
    random_state=42
)

# Get feature importance
feature_importance = clf.coef_[0]  # Coefficients as importance
```

**💡 Key Points:**
- Assumes linear relationship between features and log-odds
- Outputs probabilities (good for ranking)
- Sensitive to feature scaling
- L1 penalty for feature selection, L2 for regularization

---

### **🌳 Decision Tree**

**🔍 When to Use:** Interpretable model, non-linear patterns, mixed data types  
**⚡ Speed:** Fast | **🎯 Accuracy:** Medium | **🔍 Interpretability:** Very High

```python
from sklearn.tree import DecisionTreeClassifier

# Basic usage
clf = DecisionTreeClassifier(random_state=42)
clf.fit(X_train, y_train)

# Key parameters
clf = DecisionTreeClassifier(
    max_depth=None,         # Max tree depth (None = unlimited)
    min_samples_split=2,    # Min samples to split node
    min_samples_leaf=1,     # Min samples in leaf
    max_features=None,      # Max features per split ('sqrt', 'log2')
    criterion='gini',       # Split quality ('gini', 'entropy')
    random_state=42
)

# Feature importance
importance = clf.feature_importances_
```

**💡 Key Points:**
- Prone to overfitting (limit depth)
- Handles missing values and mixed data types
- Creates interpretable rules
- Can capture non-linear relationships

---

### **🌲 Random Forest**

**🔍 When to Use:** High accuracy needed, robust to overfitting, feature importance  
**⚡ Speed:** Medium | **🎯 Accuracy:** High | **🔍 Interpretability:** Medium

```python
from sklearn.ensemble import RandomForestClassifier

# Basic usage
clf = RandomForestClassifier(random_state=42)
clf.fit(X_train, y_train)

# Key parameters
clf = RandomForestClassifier(
    n_estimators=100,       # Number of trees
    max_depth=None,         # Max depth per tree
    min_samples_split=2,    # Min samples to split
    min_samples_leaf=1,     # Min samples in leaf
    max_features='sqrt',    # Features per tree ('sqrt', 'log2', int)
    bootstrap=True,         # Bootstrap sampling
    random_state=42,
    n_jobs=-1              # Use all processors
)

# Feature importance
importance = clf.feature_importances_
```

**💡 Key Points:**
- Ensemble of decision trees
- Reduces overfitting vs single tree
- Good default choice for many problems
- Provides feature importance naturally

---

### **🔍 Support Vector Machine (SVM)**

**🔍 When to Use:** High-dimensional data, clear margin of separation, small datasets  
**⚡ Speed:** Slow (large data) | **🎯 Accuracy:** High | **🔍 Interpretability:** Low

```python
from sklearn.svm import SVC

# Basic usage
clf = SVC(random_state=42)
clf.fit(X_train, y_train)

# Key parameters
clf = SVC(
    C=1.0,                  # Regularization parameter
    kernel='rbf',           # 'linear', 'poly', 'rbf', 'sigmoid'
    gamma='scale',          # Kernel coefficient ('scale', 'auto', float)
    degree=3,               # Polynomial degree (if poly kernel)
    probability=False,      # Enable probability estimates (slower)
    random_state=42
)

# For probabilities
clf = SVC(probability=True, random_state=42)
y_prob = clf.predict_proba(X_test)
```

**💡 Key Points:**
- Effective in high dimensions
- Memory efficient (uses support vectors)
- Requires feature scaling
- Different kernels for different patterns

---

### **🎯 K-Nearest Neighbors (KNN)**

**🔍 When to Use:** Simple baseline, local patterns, small datasets  
**⚡ Speed:** Fast (train), Slow (predict) | **🎯 Accuracy:** Medium | **🔍 Interpretability:** High

```python
from sklearn.neighbors import KNeighborsClassifier

# Basic usage
clf = KNeighborsClassifier()
clf.fit(X_train, y_train)

# Key parameters
clf = KNeighborsClassifier(
    n_neighbors=5,          # Number of neighbors
    weights='uniform',      # 'uniform', 'distance'
    algorithm='auto',       # 'ball_tree', 'kd_tree', 'brute', 'auto'
    metric='minkowski',     # Distance metric
    p=2                     # Power parameter for Minkowski metric
)
```

**💡 Key Points:**
- Lazy learning (no training phase)
- Sensitive to irrelevant features
- Requires feature scaling
- Good for local patterns

---

### **🧠 Naive Bayes**

**🔍 When to Use:** Text classification, small datasets, fast baseline  
**⚡ Speed:** Very Fast | **🎯 Accuracy:** Medium | **🔍 Interpretability:** High

```python
from sklearn.naive_bayes import GaussianNB, MultinomialNB

# Gaussian (continuous features)
clf = GaussianNB()
clf.fit(X_train, y_train)

# Multinomial (count data, text)
clf = MultinomialNB(alpha=1.0)  # Laplace smoothing
clf.fit(X_train, y_train)

# Get class probabilities
log_proba = clf.predict_log_proba(X_test)
```

**💡 Key Points:**
- Assumes feature independence
- Very fast training and prediction
- Great for text classification
- Handles missing values well

---

## 📏 Regression Algorithms

### **📈 Linear Regression**

**🔍 When to Use:** Linear relationships, interpretable coefficients, baseline  
**⚡ Speed:** Very Fast | **🎯 Accuracy:** Medium | **🔍 Interpretability:** Very High

```python
from sklearn.linear_model import LinearRegression

# Basic usage
reg = LinearRegression()
reg.fit(X_train, y_train)
y_pred = reg.predict(X_test)

# Get coefficients and intercept
coefficients = reg.coef_
intercept = reg.intercept_
r2_score = reg.score(X_test, y_test)  # R² score
```

**💡 Key Points:**
- Assumes linear relationship
- No regularization (can overfit)
- Fast and interpretable
- Sensitive to outliers

---

### **🎯 Ridge Regression**

**🔍 When to Use:** Many features, multicollinearity, prevent overfitting  
**⚡ Speed:** Fast | **🎯 Accuracy:** Medium-High | **🔍 Interpretability:** High

```python
from sklearn.linear_model import Ridge

# Basic usage
reg = Ridge(alpha=1.0, random_state=42)
reg.fit(X_train, y_train)

# Key parameters
reg = Ridge(
    alpha=1.0,              # Regularization strength
    fit_intercept=True,     # Whether to fit intercept
    solver='auto'           # Solver algorithm
)
```

**💡 Key Points:**
- L2 regularization (shrinks coefficients)
- Handles multicollinearity
- Keeps all features
- Good default for linear regression

---

### **🎯 Lasso Regression**

**🔍 When to Use:** Feature selection, sparse solutions, interpretable model  
**⚡ Speed:** Fast | **🎯 Accuracy:** Medium-High | **🔍 Interpretability:** Very High

```python
from sklearn.linear_model import Lasso

# Basic usage
reg = Lasso(alpha=1.0, random_state=42)
reg.fit(X_train, y_train)

# Key parameters
reg = Lasso(
    alpha=1.0,              # Regularization strength
    max_iter=1000,          # Max iterations
    tol=1e-4                # Tolerance for optimization
)

# Selected features (non-zero coefficients)
selected_features = X.columns[reg.coef_ != 0]
```

**💡 Key Points:**
- L1 regularization (sets coefficients to 0)
- Automatic feature selection
- Produces sparse models
- Can be unstable with correlated features

---

### **🌲 Random Forest Regressor**

**🔍 When to Use:** Non-linear relationships, robust model, feature importance  
**⚡ Speed:** Medium | **🎯 Accuracy:** High | **🔍 Interpretability:** Medium

```python
from sklearn.ensemble import RandomForestRegressor

# Basic usage
reg = RandomForestRegressor(random_state=42)
reg.fit(X_train, y_train)

# Key parameters (same as classifier)
reg = RandomForestRegressor(
    n_estimators=100,
    max_depth=None,
    min_samples_split=2,
    min_samples_leaf=1,
    max_features='sqrt',
    random_state=42,
    n_jobs=-1
)
```

**💡 Key Points:**
- Handles non-linear relationships
- Robust to outliers
- Provides prediction intervals
- Good feature importance estimates

---

## 🎯 Clustering Algorithms

### **⭕ K-Means**

**🔍 When to Use:** Spherical clusters, known number of clusters  
**⚡ Speed:** Fast | **🎯 Quality:** Medium | **🔍 Interpretability:** High

```python
from sklearn.cluster import KMeans

# Basic usage
kmeans = KMeans(n_clusters=3, random_state=42)
clusters = kmeans.fit_predict(X)

# Key parameters
kmeans = KMeans(
    n_clusters=8,           # Number of clusters
    init='k-means++',       # Initialization method
    n_init=10,              # Number of initializations
    max_iter=300,           # Max iterations
    random_state=42
)

# Cluster centers and labels
centers = kmeans.cluster_centers_
labels = kmeans.labels_
inertia = kmeans.inertia_  # Within-cluster sum of squares
```

**💡 Key Points:**
- Assumes spherical clusters
- Sensitive to initialization
- Need to specify number of clusters
- Scale features before clustering

---

### **🔍 DBSCAN**

**🔍 When to Use:** Unknown cluster count, arbitrary shapes, noise handling  
**⚡ Speed:** Medium | **🎯 Quality:** High | **🔍 Interpretability:** Medium

```python
from sklearn.cluster import DBSCAN

# Basic usage
dbscan = DBSCAN(eps=0.5, min_samples=5)
clusters = dbscan.fit_predict(X)

# Key parameters
dbscan = DBSCAN(
    eps=0.5,                # Maximum distance between samples
    min_samples=5,          # Min samples in neighborhood
    metric='euclidean',     # Distance metric
    algorithm='auto'        # Algorithm ('auto', 'ball_tree', 'kd_tree', 'brute')
)

# Cluster labels (-1 indicates noise)
labels = dbscan.labels_
n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
n_noise = list(labels).count(-1)
```

**💡 Key Points:**
- Finds arbitrary shaped clusters
- Automatically determines cluster count
- Robust to noise (outliers labeled as -1)
- Sensitive to parameters

---

## 🚀 Quick Algorithm Comparison

| Algorithm | Speed | Accuracy | Interpretability | Best For |
|-----------|-------|----------|------------------|----------|
| **Logistic Regression** | ⚡⚡⚡⚡ | 🎯🎯 | 🔍🔍🔍🔍 | Baseline, interpretable |
| **Decision Tree** | ⚡⚡⚡ | 🎯🎯 | 🔍🔍🔍🔍🔍 | Rules, mixed data |
| **Random Forest** | ⚡⚡ | 🎯🎯🎯🎯 | 🔍🔍 | High accuracy |
| **SVM** | ⚡ | 🎯🎯🎯🎯 | 🔍 | High dimensions |
| **KNN** | ⚡⚡⚡ (train) | 🎯🎯 | 🔍🔍🔍 | Local patterns |
| **Naive Bayes** | ⚡⚡⚡⚡⚡ | 🎯🎯 | 🔍🔍🔍 | Text, small data |

---

## 🛠️ Quick Implementation Template

```python
# Standard ML pipeline template
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

# 1. Prepare data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 2. Scale features (if needed)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 3. Train model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train_scaled, y_train)

# 4. Make predictions
y_pred = model.predict(X_test_scaled)

# 5. Evaluate
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy:.3f}")
print("\nDetailed Report:")
print(classification_report(y_test, y_pred))

# 6. Cross-validation
cv_scores = cross_val_score(model, X_train_scaled, y_train, cv=5)
print(f"CV Score: {cv_scores.mean():.3f} (+/- {cv_scores.std() * 2:.3f})")
```

---

## 🎯 Algorithm Selection Quick Guide

### **🔍 By Problem Type**
```python
# Classification
binary_classification = ['LogisticRegression', 'SVM', 'RandomForest']
multiclass = ['RandomForest', 'SVM', 'KNN', 'NaiveBayes']
text_classification = ['NaiveBayes', 'LogisticRegression', 'SVM']

# Regression  
linear_regression = ['LinearRegression', 'Ridge', 'Lasso']
nonlinear_regression = ['RandomForest', 'SVR', 'DecisionTree']

# Clustering
spherical_clusters = ['KMeans']
arbitrary_shapes = ['DBSCAN', 'AgglomerativeClustering']
```

### **🎯 By Data Characteristics**
```python
# Small data (< 1000 samples)
small_data = ['NaiveBayes', 'KNN', 'DecisionTree']

# Large data (> 100K samples)  
large_data = ['LogisticRegression', 'LinearSVM', 'SGDClassifier']

# High dimensions (> 1000 features)
high_dim = ['SVM', 'Ridge', 'Lasso']

# Mixed data types
mixed_data = ['DecisionTree', 'RandomForest']
```

### **⚡ By Speed Requirements**
```python
# Fastest training
fastest = ['NaiveBayes', 'LinearRegression', 'LogisticRegression']

# Fastest prediction
fast_predict = ['LinearRegression', 'LogisticRegression', 'NaiveBayes']

# Balanced speed/accuracy
balanced = ['RandomForest', 'DecisionTree']
```

---

## 🔗 Related Cheat Sheets

- **[Algorithm Selection Flowchart](algorithm_selection_flowchart.md)** - Visual guide to choose algorithms
- **[Scikit-Learn Quick Reference](../02_python_sklearn/sklearn_quick_reference.md)** - Implementation details
- **[Model Evaluation Metrics](../04_model_evaluation/classification_metrics.md)** - Performance measurement
- **[Hyperparameter Tuning](../07_troubleshooting/hyperparameter_tuning.md)** - Optimize algorithm performance

---

## 💡 Key Takeaways

### **🎯 Golden Rules**
1. **Start simple** - Try linear models first
2. **Random Forest** is a great default choice
3. **Scale features** for distance-based algorithms (SVM, KNN)
4. **Cross-validate** to avoid overfitting
5. **Feature engineering** often matters more than algorithm choice

### **🚨 Common Mistakes**
- ❌ Not scaling features for SVM/KNN
- ❌ Using complex models on small datasets
- ❌ Ignoring class imbalance
- ❌ Not setting `random_state` for reproducibility
- ❌ Choosing algorithms based on hype rather than problem fit

---

**📋 Print this reference** for quick algorithm selection! **🔍 Use Ctrl+F** to find specific algorithms quickly.

**🏠 [Back to Cheat Sheets](../README.md)** | **🎮 [Try Interactive Tools](../../06_playground/)** | **🧭 [Algorithm Selection Guide](algorithm_selection_flowchart.md)**