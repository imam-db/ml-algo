# 🧭 Algorithm Selection Flowchart

**🏠 [Cheat Sheets Home](../README.md)** | **🐍 [Scikit-Learn Quick Reference](../02_python_sklearn/)** | **🎯 [Model Evaluation](../04_model_evaluation/)**

---

## 🎯 Quick Summary

Visual decision tree to choose the right machine learning algorithm based on your data size, problem type, and requirements.

---

## 📊 Quick Decision Tree

```
📊 What type of problem do you have?
├─ 🏷️ CLASSIFICATION (Predicting categories)
│  ├─ Size < 100K samples?
│  │  ├─ Linear separable data? → 📈 **Naive Bayes**, **Logistic Regression**
│  │  ├─ Need interpretability? → 🌳 **Decision Tree**
│  │  └─ Complex patterns? → 🔍 **SVM**, **KNN**
│  └─ Size > 100K samples?
│     ├─ Need speed? → 📈 **Logistic Regression**, **Linear SVM**
│     ├─ Best performance? → 🌲 **Random Forest**, **Gradient Boosting**
│     └─ Deep patterns? → 🧠 **Neural Networks**
│
├─ 📏 REGRESSION (Predicting numbers)  
│  ├─ Linear relationship? → 📈 **Linear Regression**
│  ├─ Non-linear patterns? → 🌲 **Random Forest**, **SVR**
│  ├─ Need interpretability? → 🌳 **Decision Tree**
│  └─ High dimensions? → 🔍 **Ridge**, **Lasso**
│
├─ 🎯 CLUSTERING (Finding groups)
│  ├─ Know # of clusters? → ⭕ **K-Means**
│  ├─ Unknown # clusters? → 🔍 **DBSCAN**, **Hierarchical**
│  └─ High dimensions? → 📉 **PCA** + **K-Means**
│
└─ 🔍 DIMENSIONALITY REDUCTION
   ├─ Linear patterns? → 📉 **PCA**
   ├─ Non-linear patterns? → 🗺️ **t-SNE**, **UMAP**
   └─ Feature selection? → 🎯 **SelectKBest**, **RFE**
```

---

## 🔍 Detailed Algorithm Guide

### 📊 **Classification Problems**

#### **🏃 For Fast Training & Prediction**
```python
# Best choices for speed
✅ Naive Bayes         # Fastest, works well with text
✅ Logistic Regression # Fast, interpretable
✅ Linear SVM          # Fast for large datasets
```

#### **🎯 For High Accuracy**
```python
# Best performance (may be slower)
✅ Random Forest       # Great all-rounder
✅ Gradient Boosting   # Often highest accuracy
✅ XGBoost/LightGBM   # State-of-the-art
✅ Neural Networks     # For complex patterns
```

#### **🔍 For Interpretability**
```python
# Easiest to explain
✅ Decision Tree       # Visual rules
✅ Logistic Regression # Coefficient importance
✅ Linear SVM          # Linear boundaries
✅ Naive Bayes         # Feature probabilities
```

#### **📏 By Data Size**
```python
# Small datasets (< 10K)
✅ SVM, KNN, Decision Tree, Naive Bayes

# Medium datasets (10K - 100K)  
✅ Random Forest, Logistic Regression, SVM

# Large datasets (> 100K)
✅ Logistic Regression, Linear SVM, SGD Classifier
```

---

### 📈 **Regression Problems**

#### **📊 By Relationship Type**
```python
# Linear relationships
✅ Linear Regression   # Simple linear
✅ Ridge Regression    # With regularization
✅ Lasso Regression    # With feature selection

# Non-linear relationships
✅ Random Forest       # Handles non-linearity well
✅ Support Vector Regression (SVR)
✅ Gradient Boosting   # High accuracy
```

#### **🔍 By Data Characteristics**
```python
# High-dimensional data (many features)
✅ Ridge/Lasso        # Regularization prevents overfitting
✅ Elastic Net        # Combines Ridge + Lasso

# Small datasets
✅ Linear Regression  # Simple, less prone to overfitting
✅ Decision Tree      # Can capture non-linearity

# Large datasets
✅ Linear Regression  # Scales well
✅ Random Forest      # Parallel processing
```

---

### 🎯 **Clustering Problems**

#### **🔢 By Number of Clusters**
```python
# Known number of clusters
✅ K-Means            # Fast, works well for spherical clusters
✅ K-Medoids          # Robust to outliers

# Unknown number of clusters  
✅ DBSCAN             # Finds arbitrary shapes, handles noise
✅ Hierarchical       # Creates cluster tree
✅ Gaussian Mixture   # Probabilistic clustering
```

#### **📊 By Cluster Shape**
```python
# Spherical clusters
✅ K-Means            # Assumes spherical clusters

# Arbitrary shapes
✅ DBSCAN             # Any shape, handles noise
✅ Spectral Clustering # Non-convex shapes

# Overlapping clusters
✅ Gaussian Mixture   # Soft clustering
```

---

## 🚀 Algorithm Performance Comparison

### **🏃 Speed Rankings (Training Time)**

| Rank | Algorithm | Speed | Use Case |
|------|-----------|-------|----------|
| 🥇 | Naive Bayes | ⚡⚡⚡⚡⚡ | Text classification, small data |
| 🥈 | Linear Regression | ⚡⚡⚡⚡ | Simple regression problems |
| 🥉 | Logistic Regression | ⚡⚡⚡⚡ | Binary classification |
| 4️⃣ | Decision Tree | ⚡⚡⚡ | Interpretable models |
| 5️⃣ | KNN | ⚡⚡ | Simple classification (slow prediction) |
| 6️⃣ | Random Forest | ⚡⚡ | Balanced performance |
| 7️⃣ | SVM | ⚡ | Small datasets |
| 8️⃣ | Neural Networks | 🐌 | Complex patterns |

### **🎯 Accuracy Rankings (Typical Performance)**

| Rank | Algorithm | Accuracy | Trade-offs |
|------|-----------|----------|------------|
| 🥇 | Gradient Boosting | 🎯🎯🎯🎯🎯 | Slow, prone to overfitting |
| 🥈 | Random Forest | 🎯🎯🎯🎯 | Good all-rounder |
| 🥉 | SVM | 🎯🎯🎯🎯 | Slow on large data |
| 4️⃣ | Neural Networks | 🎯🎯🎯 | Need lots of data |
| 5️⃣ | Logistic Regression | 🎯🎯🎯 | Only linear boundaries |
| 6️⃣ | KNN | 🎯🎯 | Sensitive to irrelevant features |
| 7️⃣ | Decision Tree | 🎯🎯 | Prone to overfitting |
| 8️⃣ | Naive Bayes | 🎯 | Strong independence assumption |

---

## 🤔 Decision Scenarios

### **🔍 "I need to explain my model to stakeholders"**
```python
# Choose interpretable algorithms
✅ Decision Tree      # Visual rules
✅ Linear Regression  # Clear coefficients  
✅ Logistic Regression # Probability interpretation
❌ Random Forest      # Black box
❌ Neural Networks    # Very black box
```

### **⚡ "I need results fast"**
```python
# Choose fast algorithms
✅ Naive Bayes        # Fastest
✅ Linear models      # Very fast
✅ Decision Tree      # Fast training
❌ SVM               # Slow on large data
❌ Neural Networks   # Slow training
```

### **🎯 "I need the highest accuracy possible"**
```python
# Try ensemble methods
✅ Random Forest      # Try first
✅ Gradient Boosting  # Often best
✅ XGBoost           # State-of-the-art
✅ Neural Networks   # For complex data
```

### **📊 "My dataset is small (< 1000 samples)"**
```python
# Avoid complex models
✅ Naive Bayes       # Works with little data
✅ Linear models     # Simple, less overfitting
✅ KNN              # Non-parametric
❌ Neural Networks  # Need lots of data
❌ Deep ensembles   # Will overfit
```

### **🚀 "My dataset is huge (> 1M samples)"**
```python
# Choose scalable algorithms
✅ Linear models     # Scale well
✅ SGD variants      # Online learning
✅ Neural Networks   # Parallel processing
❌ SVM              # Memory intensive
❌ KNN              # Slow predictions
```

### **🔍 "I have many irrelevant features"**
```python
# Use feature selection or robust algorithms
✅ Lasso Regression  # Built-in feature selection
✅ Random Forest     # Feature importance
✅ SVM              # Robust to irrelevant features
❌ KNN              # Sensitive to irrelevant features
❌ Naive Bayes      # Assumes all features relevant
```

---

## 🛠️ Quick Implementation Guide

### **🏃 1-Minute Algorithm Test**
```python
# Quick performance comparison template
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression  
from sklearn.svm import SVC
from sklearn.model_selection import cross_val_score

models = {
    'Random Forest': RandomForestClassifier(random_state=42),
    'Logistic Regression': LogisticRegression(random_state=42),
    'SVM': SVC(random_state=42)
}

for name, model in models.items():
    scores = cross_val_score(model, X, y, cv=5)
    print(f"{name}: {scores.mean():.3f} (+/- {scores.std() * 2:.3f})")
```

### **🎯 Algorithm Testing Pipeline**
```python
def quick_model_comparison(X, y, problem_type='classification'):
    """
    Quick comparison of multiple algorithms
    """
    if problem_type == 'classification':
        models = {
            'Logistic Regression': LogisticRegression(random_state=42),
            'Random Forest': RandomForestClassifier(random_state=42),
            'SVM': SVC(random_state=42),
            'Naive Bayes': GaussianNB(),
        }
        scoring = 'accuracy'
    else:  # regression
        models = {
            'Linear Regression': LinearRegression(),
            'Random Forest': RandomForestRegressor(random_state=42),
            'SVR': SVR(),
        }
        scoring = 'r2'
    
    results = {}
    for name, model in models.items():
        scores = cross_val_score(model, X, y, cv=5, scoring=scoring)
        results[name] = {
            'mean': scores.mean(),
            'std': scores.std()
        }
    
    # Sort by performance
    sorted_results = sorted(results.items(), key=lambda x: x[1]['mean'], reverse=True)
    
    print("🏆 Model Performance Ranking:")
    for i, (name, score) in enumerate(sorted_results, 1):
        print(f"{i}. {name}: {score['mean']:.3f} (+/- {score['std'] * 2:.3f})")
    
    return sorted_results
```

---

## 🎨 Visualization: When to Use Each Algorithm

```
                📊 DATA SIZE
                     ↑
         Large   ┌─────────────┐
        (>100K)  │ Logistic    │ Neural
                 │ Linear SVM  │ Networks
                 └─────────────┘
                          
         Medium  ┌─────────────┐
       (1K-100K) │ Random      │ SVM
                 │ Forest      │ KNN
                 └─────────────┘
                          
         Small   ┌─────────────┐
        (<1K)    │ Naive Bayes │ Decision
                 │ Linear      │ Tree
                 └─────────────┘
                          
              Simple ←─────────→ Complex
                    🔍 PROBLEM COMPLEXITY
```

---

## 🔗 Related Cheat Sheets

- **[Scikit-Learn Quick Reference](../02_python_sklearn/sklearn_quick_reference.md)** - Implementation syntax
- **[Classification Metrics](../04_model_evaluation/classification_metrics.md)** - How to evaluate results
- **Hyperparameter Tuning** (`07_troubleshooting/hyperparameter_tuning.md`, planned) - Optimize your chosen algorithm
- **[Data Preprocessing](../05_data_preprocessing/preprocessing_pipeline.md)** - Prepare data for algorithms

---

## ❗ Key Takeaways

### **🎯 Golden Rules**
1. **Start simple** - Try linear models first
2. **Consider your constraints** - Speed vs accuracy vs interpretability  
3. **Test multiple algorithms** - Use cross-validation
4. **Ensemble methods** often win competitions
5. **Data quality** matters more than algorithm choice
6. **Feature engineering** can be more important than algorithm selection

### **🚨 Common Mistakes**
- ❌ Using complex models on small datasets
- ❌ Not scaling features for distance-based algorithms
- ❌ Choosing algorithms based on hype rather than fit
- ❌ Not considering prediction speed in production
- ❌ Ignoring interpretability requirements

---

**🏠 [Back to Cheat Sheets](../README.md)** | **🎮 [Try Interactive Tools](../../06_playground/)** | **📚 Algorithm Details** (`02_algorithm_explanations/`, planned)