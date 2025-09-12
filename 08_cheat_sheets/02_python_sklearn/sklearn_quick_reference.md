# 🐍 Scikit-Learn Quick Reference Cheat Sheet

**🏠 [Cheat Sheets Home](../README.md)** | **📚 [Algorithm Cheat Sheets](../01_algorithms/)** | **🔧 [Data Preprocessing](../05_data_preprocessing/)**

---

## 🎯 Quick Summary

Essential scikit-learn syntax for machine learning workflows - from data loading to model evaluation.

---

## ⚡ Essential Imports

```python
# Core ML libraries
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# Common algorithms
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.svm import SVC, SVR
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier

# Plotting
import matplotlib.pyplot as plt
import seaborn as sns
```

---

## 🔄 Complete ML Workflow

### **1. Data Loading & Exploration**
```python
# Load data
df = pd.read_csv('data.csv')
X = df.drop('target', axis=1)
y = df['target']

# Quick exploration
print(f"Shape: {X.shape}")
print(f"Missing values: {X.isnull().sum().sum()}")
print(f"Target classes: {y.value_counts()}")
```

### **2. Data Splitting**
```python
# Basic split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Validation split
X_train, X_val, y_train, y_val = train_test_split(
    X_train, y_train, test_size=0.25, random_state=42
)
```

### **3. Data Preprocessing**
```python
# Numerical scaling
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Categorical encoding
encoder = LabelEncoder()
y_encoded = encoder.fit_transform(y)

# One-hot encoding
from sklearn.preprocessing import OneHotEncoder
oh_encoder = OneHotEncoder(sparse=False, drop='first')
X_categorical = oh_encoder.fit_transform(df[['category_col']])
```

### **4. Model Training**
```python
# Initialize model
model = RandomForestClassifier(n_estimators=100, random_state=42)

# Train model
model.fit(X_train, y_train)

# Make predictions
y_pred = model.predict(X_test)
y_prob = model.predict_proba(X_test)  # For classifiers with probability
```

### **5. Model Evaluation**
```python
# Classification metrics
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy:.3f}")
print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# Confusion matrix
cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.show()

# Cross-validation
cv_scores = cross_val_score(model, X_train, y_train, cv=5)
print(f"CV Score: {cv_scores.mean():.3f} (+/- {cv_scores.std() * 2:.3f})")
```

---

## 🤖 Algorithm Quick Reference

### **Classification**
```python
# Logistic Regression
from sklearn.linear_model import LogisticRegression
clf = LogisticRegression(random_state=42)
clf.fit(X_train, y_train)

# Random Forest
from sklearn.ensemble import RandomForestClassifier
clf = RandomForestClassifier(n_estimators=100, random_state=42)
clf.fit(X_train, y_train)

# SVM
from sklearn.svm import SVC
clf = SVC(kernel='rbf', random_state=42)
clf.fit(X_train, y_train)

# K-Nearest Neighbors
from sklearn.neighbors import KNeighborsClassifier
clf = KNeighborsClassifier(n_neighbors=5)
clf.fit(X_train, y_train)

# Naive Bayes
from sklearn.naive_bayes import GaussianNB
clf = GaussianNB()
clf.fit(X_train, y_train)
```

### **Regression**
```python
# Linear Regression
from sklearn.linear_model import LinearRegression
reg = LinearRegression()
reg.fit(X_train, y_train)

# Random Forest Regressor
from sklearn.ensemble import RandomForestRegressor
reg = RandomForestRegressor(n_estimators=100, random_state=42)
reg.fit(X_train, y_train)

# Support Vector Regression
from sklearn.svm import SVR
reg = SVR(kernel='rbf')
reg.fit(X_train, y_train)
```

### **Clustering**
```python
# K-Means
from sklearn.cluster import KMeans
kmeans = KMeans(n_clusters=3, random_state=42)
clusters = kmeans.fit_predict(X)

# Hierarchical Clustering
from sklearn.cluster import AgglomerativeClustering
clustering = AgglomerativeClustering(n_clusters=3)
clusters = clustering.fit_predict(X)

# DBSCAN
from sklearn.cluster import DBSCAN
dbscan = DBSCAN(eps=0.5, min_samples=5)
clusters = dbscan.fit_predict(X)
```

---

## 🔧 Preprocessing Transformers

### **Numerical Features**
```python
# Scaling
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler

scaler = StandardScaler()  # Mean=0, Std=1
scaler = MinMaxScaler()    # Range [0,1]
scaler = RobustScaler()    # Uses median/IQR

X_scaled = scaler.fit_transform(X_train)
```

### **Categorical Features**
```python
# Label Encoding (ordinal)
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
y_encoded = le.fit_transform(y)

# One-Hot Encoding (nominal)
from sklearn.preprocessing import OneHotEncoder
ohe = OneHotEncoder(drop='first', sparse=False)
X_encoded = ohe.fit_transform(X_categorical)

# Target Encoding
from sklearn.preprocessing import TargetEncoder  # sklearn 1.3+
te = TargetEncoder()
X_encoded = te.fit_transform(X_categorical, y)
```

### **Feature Selection**
```python
# Univariate Selection
from sklearn.feature_selection import SelectKBest, f_classif
selector = SelectKBest(f_classif, k=10)
X_selected = selector.fit_transform(X, y)

# Recursive Feature Elimination
from sklearn.feature_selection import RFE
from sklearn.ensemble import RandomForestClassifier
rfe = RFE(RandomForestClassifier(n_estimators=50), n_features_to_select=10)
X_selected = rfe.fit_transform(X, y)

# Feature Importance
model = RandomForestClassifier()
model.fit(X, y)
feature_importance = pd.DataFrame({
    'feature': X.columns,
    'importance': model.feature_importances_
}).sort_values('importance', ascending=False)
```

---

## 🎯 Model Selection & Tuning

### **Cross-Validation**
```python
# K-Fold Cross-Validation
from sklearn.model_selection import cross_val_score, KFold
kfold = KFold(n_splits=5, shuffle=True, random_state=42)
scores = cross_val_score(model, X, y, cv=kfold, scoring='accuracy')

# Stratified K-Fold (for classification)
from sklearn.model_selection import StratifiedKFold
skfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
scores = cross_val_score(model, X, y, cv=skfold)
```

### **Hyperparameter Tuning**
```python
# Grid Search
from sklearn.model_selection import GridSearchCV
param_grid = {
    'n_estimators': [50, 100, 200],
    'max_depth': [3, 5, 10, None],
    'min_samples_split': [2, 5, 10]
}
grid_search = GridSearchCV(
    RandomForestClassifier(random_state=42),
    param_grid, cv=5, scoring='accuracy', n_jobs=-1
)
grid_search.fit(X_train, y_train)
best_model = grid_search.best_estimator_

# Random Search
from sklearn.model_selection import RandomizedSearchCV
random_search = RandomizedSearchCV(
    RandomForestClassifier(random_state=42),
    param_grid, n_iter=20, cv=5, scoring='accuracy', n_jobs=-1
)
random_search.fit(X_train, y_train)
```

---

## 📊 Evaluation Metrics

### **Classification**
```python
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, classification_report, confusion_matrix
)

# Basic metrics
accuracy = accuracy_score(y_true, y_pred)
precision = precision_score(y_true, y_pred, average='weighted')
recall = recall_score(y_true, y_pred, average='weighted')
f1 = f1_score(y_true, y_pred, average='weighted')

# ROC AUC (binary classification)
auc = roc_auc_score(y_true, y_proba)

# Comprehensive report
print(classification_report(y_true, y_pred))
```

### **Regression**
```python
from sklearn.metrics import (
    mean_squared_error, mean_absolute_error, r2_score
)

mse = mean_squared_error(y_true, y_pred)
rmse = np.sqrt(mse)
mae = mean_absolute_error(y_true, y_pred)
r2 = r2_score(y_true, y_pred)

print(f"RMSE: {rmse:.3f}")
print(f"MAE: {mae:.3f}")
print(f"R²: {r2:.3f}")
```

---

## 🔀 Pipelines

### **Simple Pipeline**
```python
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

# Create pipeline
pipe = Pipeline([
    ('scaler', StandardScaler()),
    ('classifier', SVC(random_state=42))
])

# Train pipeline
pipe.fit(X_train, y_train)

# Predict
y_pred = pipe.predict(X_test)
```

### **Column Transformer**
```python
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder

# Define transformers for different column types
preprocessor = ColumnTransformer([
    ('num', StandardScaler(), ['age', 'income']),
    ('cat', OneHotEncoder(drop='first'), ['category', 'region'])
])

# Full pipeline
full_pipeline = Pipeline([
    ('preprocessor', preprocessor),
    ('classifier', RandomForestClassifier(random_state=42))
])

full_pipeline.fit(X_train, y_train)
```

---

## ❗ Common Pitfalls

### **🚨 Data Leakage**
```python
# ❌ WRONG: Scaling before split
X_scaled = StandardScaler().fit_transform(X)
X_train, X_test = train_test_split(X_scaled, y)

# ✅ CORRECT: Scale after split
X_train, X_test, y_train, y_test = train_test_split(X, y)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)  # Only transform, don't fit!
```

### **🚨 Target Encoding**
```python
# ❌ WRONG: Using all data for encoding
encoder = TargetEncoder()
X_encoded = encoder.fit_transform(X, y)

# ✅ CORRECT: Fit only on training data
encoder = TargetEncoder()
X_train_encoded = encoder.fit_transform(X_train, y_train)
X_test_encoded = encoder.transform(X_test)
```

### **🚨 Cross-Validation Scoring**
```python
# ❌ WRONG: Using default scoring for imbalanced data
scores = cross_val_score(model, X, y, cv=5)

# ✅ CORRECT: Specify appropriate metric
scores = cross_val_score(model, X, y, cv=5, scoring='f1_weighted')
```

---

## 🔗 Related Cheat Sheets

- **[Algorithm Selection Guide](../01_algorithms/algorithm_selection_flowchart.md)** - Choose the right algorithm
- **[Data Preprocessing](../05_data_preprocessing/preprocessing_pipeline.md)** - Complete preprocessing workflow  
- **[Model Evaluation](../04_model_evaluation/classification_metrics.md)** - All evaluation metrics
- **[Hyperparameter Tuning](../07_troubleshooting/optimization_strategies.md)** - Optimization strategies

---

## 💡 Quick Tips

- **Always use `random_state`** for reproducible results
- **Fit transformers only on training data**, then transform test data
- **Use pipelines** to prevent data leakage
- **Check for class imbalance** before choosing metrics
- **Cross-validate** to get robust performance estimates
- **Scale features** for distance-based algorithms (SVM, KNN)
- **Use `n_jobs=-1`** for parallel processing when available

---

**📋 Print this cheat sheet** for quick offline reference! **🔍 Use Ctrl+F** to find specific functions quickly.

**🏠 [Back to Cheat Sheets](../README.md)** | **🤖 [Algorithm Details](../01_algorithms/)** | **🎮 [Try Interactive Tools](../../06_playground/)**