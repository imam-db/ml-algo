# 🔧 ML Troubleshooting & Common Errors

**🏠 [Cheat Sheets Home](../README.md)** | **🐍 [Scikit-Learn Reference](../02_python_sklearn/)** | **📊 [Model Evaluation](../04_model_evaluation/)**

---

## 🎯 Quick Summary

Common machine learning errors, debugging strategies, and practical solutions for faster problem resolution.

---

## 🚨 Data-Related Issues

### **📊 Shape Mismatch Errors**

#### **Problem: Feature Mismatch**
```python
# ❌ Error: X has n features, but estimator expects m features
ValueError: X has 10 features, but RandomForestClassifier is expecting 15 features
```

**🔍 Diagnosis:**
```python
print(f"Training data shape: {X_train.shape}")
print(f"Test data shape: {X_test.shape}")
print(f"Feature names train: {list(X_train.columns)}")
print(f"Feature names test: {list(X_test.columns)}")
```

**✅ Solutions:**
```python
# Solution 1: Ensure same preprocessing for train/test
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)  # Fit on train
X_test_scaled = scaler.transform(X_test)        # Transform test (don't fit!)

# Solution 2: Align features explicitly  
common_features = X_train.columns.intersection(X_test.columns)
X_train_aligned = X_train[common_features]
X_test_aligned = X_test[common_features]

# Solution 3: Save feature names during training
import joblib

# During training
feature_names = list(X_train.columns)
joblib.dump(feature_names, 'feature_names.pkl')

# During prediction
saved_features = joblib.load('feature_names.pkl')
X_test_aligned = X_test.reindex(columns=saved_features, fill_value=0)
```

---

### **🚨 Missing Values Issues**

#### **Problem: NaN in Dataset**
```python
# ❌ Error: Input contains NaN
ValueError: Input contains NaN, infinity or a value too large for dtype('float64')
```

**🔍 Diagnosis:**
```python
# Check for missing values
print("Missing values per column:")
print(df.isnull().sum())

# Check for infinite values
print("Infinite values per column:")
print(np.isinf(df.select_dtypes(include=[np.number])).sum())

# Detailed missing analysis
def diagnose_missing_values(df):
    missing_info = pd.DataFrame({
        'Column': df.columns,
        'Missing_Count': df.isnull().sum(),
        'Missing_Percentage': (df.isnull().sum() / len(df)) * 100,
        'Data_Type': df.dtypes
    })
    return missing_info[missing_info['Missing_Count'] > 0].sort_values('Missing_Percentage', ascending=False)

print(diagnose_missing_values(df))
```

**✅ Solutions:**
```python
# Solution 1: Quick fix with SimpleImputer
from sklearn.impute import SimpleImputer

# For numerical data
num_imputer = SimpleImputer(strategy='median')
X_train_num = num_imputer.fit_transform(X_train.select_dtypes(include=[np.number]))

# For categorical data
cat_imputer = SimpleImputer(strategy='most_frequent')
X_train_cat = cat_imputer.fit_transform(X_train.select_dtypes(include=['object']))

# Solution 2: Advanced imputation
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer

iterative_imputer = IterativeImputer(random_state=42)
X_train_imputed = iterative_imputer.fit_transform(X_train)

# Solution 3: Handle infinite values
def clean_infinite_values(df):
    # Replace infinite with NaN, then handle
    df_clean = df.replace([np.inf, -np.inf], np.nan)
    return df_clean.fillna(df_clean.median())

X_train_clean = clean_infinite_values(X_train)
```

---

### **📝 Categorical Encoding Issues**

#### **Problem: Unknown Categories**
```python
# ❌ Error: Found unknown categories during transform
ValueError: Found unknown categories ['new_category'] in column 0 during transform
```

**🔍 Diagnosis:**
```python
# Check unique values in train vs test
def compare_categories(train_df, test_df, cat_columns):
    for col in cat_columns:
        train_unique = set(train_df[col].unique())
        test_unique = set(test_df[col].unique())
        
        only_in_test = test_unique - train_unique
        only_in_train = train_unique - test_unique
        
        print(f"\nColumn: {col}")
        print(f"Only in test: {only_in_test}")
        print(f"Only in train: {only_in_train}")

compare_categories(X_train, X_test, categorical_columns)
```

**✅ Solutions:**
```python
# Solution 1: Handle unknown categories in OneHotEncoder
from sklearn.preprocessing import OneHotEncoder

encoder = OneHotEncoder(handle_unknown='ignore', drop='first')
X_train_encoded = encoder.fit_transform(X_train_categorical)
X_test_encoded = encoder.transform(X_test_categorical)  # Won't fail on new categories

# Solution 2: Pre-process to handle unknowns
def handle_unknown_categories(train_df, test_df, cat_columns):
    test_df_processed = test_df.copy()
    
    for col in cat_columns:
        train_categories = set(train_df[col].unique())
        # Replace unknown categories with 'Other'
        test_df_processed[col] = test_df_processed[col].apply(
            lambda x: x if x in train_categories else 'Other'
        )
        
        # Add 'Other' to training set if not present
        if 'Other' not in train_categories:
            # Add a few 'Other' samples to training set
            train_df.loc[train_df.index[:5], col] = 'Other'
    
    return train_df, test_df_processed

X_train_fixed, X_test_fixed = handle_unknown_categories(X_train, X_test, cat_columns)

# Solution 3: Use target encoding that handles unknowns
class SafeTargetEncoder:
    def __init__(self, smoothing=1.0):
        self.smoothing = smoothing
        self.global_mean = None
        self.encodings = {}
    
    def fit(self, X, y):
        self.global_mean = y.mean()
        for col in X.columns:
            category_means = y.groupby(X[col]).mean()
            category_counts = X[col].value_counts()
            
            # Smoothed target encoding
            smoothed_means = (
                (category_counts * category_means + self.smoothing * self.global_mean) /
                (category_counts + self.smoothing)
            )
            self.encodings[col] = smoothed_means.to_dict()
        return self
    
    def transform(self, X):
        X_encoded = X.copy()
        for col in X.columns:
            X_encoded[col] = X_encoded[col].map(self.encodings[col]).fillna(self.global_mean)
        return X_encoded
```

---

## 🎯 Model Training Issues

### **🔄 Convergence Problems**

#### **Problem: Model Not Converging**
```python
# ❌ Warning: lbfgs failed to converge
ConvergenceWarning: lbfgs failed to converge (status=1): STOP: TOTAL NO. of ITERATIONS REACHED LIMIT
```

**🔍 Diagnosis:**
```python
# Check data scaling
print("Feature statistics:")
print(X_train.describe())

# Check for extreme values
print("\nFeature ranges:")
for col in X_train.columns:
    print(f"{col}: [{X_train[col].min():.2f}, {X_train[col].max():.2f}]")
```

**✅ Solutions:**
```python
# Solution 1: Scale the data
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Retrain model
model = LogisticRegression(max_iter=1000)  # Increase iterations
model.fit(X_train_scaled, y_train)

# Solution 2: Change solver
model = LogisticRegression(
    solver='liblinear',  # Better for small datasets
    max_iter=1000,
    random_state=42
)

# Solution 3: Regularization adjustment
model = LogisticRegression(
    C=1.0,              # Try different values: 0.1, 1.0, 10.0
    max_iter=1000,
    random_state=42
)

# Solution 4: Use SGD for large datasets
from sklearn.linear_model import SGDClassifier

model = SGDClassifier(
    loss='log',         # For logistic regression
    max_iter=1000,
    tol=1e-3,
    random_state=42
)
```

---

### **📉 Poor Model Performance**

#### **Problem: Low Accuracy/High Loss**
```python
# Poor performance indicators
accuracy = 0.55  # Low accuracy
loss = 2.34      # High loss
cv_std = 0.25    # High variance in cross-validation
```

**🔍 Diagnosis Framework:**
```python
def diagnose_poor_performance(X, y, model, cv=5):
    from sklearn.model_selection import cross_val_score, validation_curve
    from sklearn.metrics import classification_report
    
    print("=== PERFORMANCE DIAGNOSIS ===")
    
    # 1. Basic cross-validation
    cv_scores = cross_val_score(model, X, y, cv=cv)
    print(f"CV Mean: {cv_scores.mean():.3f} (+/- {cv_scores.std() * 2:.3f})")
    
    # 2. Train vs validation performance
    model.fit(X, y)
    train_score = model.score(X, y)
    print(f"Training Score: {train_score:.3f}")
    
    if train_score - cv_scores.mean() > 0.1:
        print("⚠️ Possible overfitting (high train score, low CV score)")
    elif cv_scores.mean() < 0.6:
        print("⚠️ Possible underfitting (both scores low)")
    
    # 3. Learning curves
    train_sizes = np.linspace(0.1, 1.0, 10)
    train_scores, val_scores = validation_curve(
        model, X, y, param_name='random_state', param_range=[42],
        cv=cv, scoring='accuracy'
    )
    
    return {
        'cv_mean': cv_scores.mean(),
        'cv_std': cv_scores.std(),
        'train_score': train_score,
        'likely_issue': 'overfitting' if train_score - cv_scores.mean() > 0.1 else 'underfitting'
    }

# Run diagnosis
diagnosis = diagnose_poor_performance(X_train, y_train, RandomForestClassifier())
```

**✅ Solutions by Problem Type:**

**For Overfitting:**
```python
# Solution 1: Regularization
from sklearn.linear_model import LogisticRegression

model = LogisticRegression(C=0.1)  # Stronger regularization

# Solution 2: Reduce model complexity
from sklearn.ensemble import RandomForestClassifier

model = RandomForestClassifier(
    n_estimators=50,      # Fewer trees
    max_depth=5,          # Limit depth
    min_samples_split=20, # More samples required to split
    min_samples_leaf=10   # More samples required in leaf
)

# Solution 3: Early stopping (for iterative models)
from sklearn.ensemble import GradientBoostingClassifier

model = GradientBoostingClassifier(
    n_estimators=100,
    learning_rate=0.1,
    validation_fraction=0.2,
    n_iter_no_change=10,  # Early stopping
    random_state=42
)

# Solution 4: Cross-validation with more folds
from sklearn.model_selection import StratifiedKFold

cv = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)
scores = cross_val_score(model, X, y, cv=cv)
```

**For Underfitting:**
```python
# Solution 1: Increase model complexity
model = RandomForestClassifier(
    n_estimators=200,     # More trees
    max_depth=None,       # No depth limit
    min_samples_split=2,  # Allow more splits
    random_state=42
)

# Solution 2: Feature engineering
from sklearn.preprocessing import PolynomialFeatures

poly = PolynomialFeatures(degree=2, include_bias=False)
X_poly = poly.fit_transform(X_train)

# Solution 3: Try different algorithms
from sklearn.svm import SVC
from sklearn.ensemble import GradientBoostingClassifier

models = {
    'SVM': SVC(kernel='rbf', C=1.0),
    'Gradient Boosting': GradientBoostingClassifier(n_estimators=200),
    'Random Forest': RandomForestClassifier(n_estimators=200)
}

for name, model in models.items():
    scores = cross_val_score(model, X_train, y_train, cv=5)
    print(f"{name}: {scores.mean():.3f} (+/- {scores.std() * 2:.3f})")

# Solution 4: Ensemble methods
from sklearn.ensemble import VotingClassifier

ensemble = VotingClassifier([
    ('rf', RandomForestClassifier(n_estimators=100)),
    ('svm', SVC(probability=True)),
    ('gb', GradientBoostingClassifier(n_estimators=100))
], voting='soft')
```

---

## 💾 Memory & Performance Issues

### **🐌 Slow Training/Prediction**

#### **Problem: Training Takes Too Long**

**🔍 Diagnosis:**
```python
import time
from sklearn.datasets import make_classification

# Profile training time
def profile_training_time(model, X, y):
    start_time = time.time()
    model.fit(X, y)
    training_time = time.time() - start_time
    
    start_time = time.time()
    predictions = model.predict(X[:1000])  # Predict on subset
    prediction_time = (time.time() - start_time) * (len(X) / 1000)
    
    return {
        'training_time': training_time,
        'prediction_time': prediction_time,
        'data_size': X.shape,
        'model_type': type(model).__name__
    }

# Test different models
models = [
    LogisticRegression(),
    RandomForestClassifier(),
    SVC(),
    GradientBoostingClassifier()
]

for model in models:
    stats = profile_training_time(model, X_train, y_train)
    print(f"{stats['model_type']}: Train={stats['training_time']:.2f}s, "
          f"Predict={stats['prediction_time']:.2f}s")
```

**✅ Solutions:**
```python
# Solution 1: Use faster algorithms for large data
from sklearn.linear_model import SGDClassifier
from sklearn.naive_bayes import MultinomialNB

# SGD for linear models (very fast)
fast_model = SGDClassifier(max_iter=1000, random_state=42)

# Naive Bayes (extremely fast)
nb_model = MultinomialNB()

# Solution 2: Reduce data size intelligently
from sklearn.model_selection import StratifiedShuffleSplit

# Sample subset for initial experiments
splitter = StratifiedShuffleSplit(n_splits=1, train_size=0.1, random_state=42)
subset_idx, _ = next(splitter.split(X_train, y_train))
X_subset = X_train.iloc[subset_idx]
y_subset = y_train.iloc[subset_idx]

# Solution 3: Parallel processing
model = RandomForestClassifier(n_jobs=-1)  # Use all CPU cores

# Solution 4: Feature selection to reduce dimensions
from sklearn.feature_selection import SelectKBest, f_classif

selector = SelectKBest(f_classif, k=20)  # Keep top 20 features
X_train_selected = selector.fit_transform(X_train, y_train)
X_test_selected = selector.transform(X_test)

# Solution 5: Early stopping for iterative algorithms
from sklearn.ensemble import GradientBoostingClassifier

model = GradientBoostingClassifier(
    n_estimators=1000,
    validation_fraction=0.2,
    n_iter_no_change=10,  # Stop if no improvement
    random_state=42
)
```

### **💾 Memory Issues**

#### **Problem: Out of Memory**
```python
# ❌ Error: Unable to allocate array
MemoryError: Unable to allocate 2.23 GiB for an array
```

**✅ Solutions:**
```python
# Solution 1: Use sparse matrices
from scipy.sparse import csr_matrix
from sklearn.preprocessing import OneHotEncoder

# One-hot encoding with sparse output
encoder = OneHotEncoder(sparse=True)  # Default is sparse=True
X_encoded_sparse = encoder.fit_transform(X_categorical)

# Solution 2: Process data in chunks
def process_in_chunks(data, chunk_size=10000):
    results = []
    for i in range(0, len(data), chunk_size):
        chunk = data[i:i+chunk_size]
        # Process chunk
        processed_chunk = some_transformation(chunk)
        results.append(processed_chunk)
    return np.concatenate(results)

# Solution 3: Use memory-efficient data types
def optimize_memory_usage(df):
    for col in df.columns:
        if df[col].dtype == 'int64':
            if df[col].min() >= 0:
                if df[col].max() < 255:
                    df[col] = df[col].astype('uint8')
                elif df[col].max() < 65535:
                    df[col] = df[col].astype('uint16')
        elif df[col].dtype == 'float64':
            df[col] = df[col].astype('float32')
    return df

# Solution 4: Use incremental learning
from sklearn.linear_model import SGDClassifier

model = SGDClassifier()
chunk_size = 1000

for i in range(0, len(X_train), chunk_size):
    X_chunk = X_train[i:i+chunk_size]
    y_chunk = y_train[i:i+chunk_size]
    
    if i == 0:
        model.partial_fit(X_chunk, y_chunk, classes=np.unique(y_train))
    else:
        model.partial_fit(X_chunk, y_chunk)
```

---

## 🎯 Evaluation & Interpretation Issues

### **📊 Misleading Metrics**

#### **Problem: High Accuracy but Poor Performance**
```python
# Example: 99% accuracy but model is useless
accuracy = 0.99  # Looks great!
# But data is 99% class 0, 1% class 1
# Model just predicts class 0 always
```

**🔍 Diagnosis:**
```python
def evaluate_classification_properly(y_true, y_pred, y_prob=None):
    from sklearn.metrics import (
        accuracy_score, precision_score, recall_score, f1_score,
        roc_auc_score, classification_report, confusion_matrix
    )
    
    print("=== COMPREHENSIVE EVALUATION ===")
    
    # Check class distribution
    print("True class distribution:")
    print(pd.Series(y_true).value_counts(normalize=True))
    
    print("\nPredicted class distribution:")
    print(pd.Series(y_pred).value_counts(normalize=True))
    
    # Multiple metrics
    print(f"\nAccuracy: {accuracy_score(y_true, y_pred):.3f}")
    print(f"Precision: {precision_score(y_true, y_pred, average='weighted'):.3f}")
    print(f"Recall: {recall_score(y_true, y_pred, average='weighted'):.3f}")
    print(f"F1-Score: {f1_score(y_true, y_pred, average='weighted'):.3f}")
    
    if y_prob is not None and len(np.unique(y_true)) == 2:
        print(f"ROC-AUC: {roc_auc_score(y_true, y_prob):.3f}")
    
    print("\nConfusion Matrix:")
    print(confusion_matrix(y_true, y_pred))
    
    print("\nDetailed Classification Report:")
    print(classification_report(y_true, y_pred))

# Use it
evaluate_classification_properly(y_test, y_pred, y_prob)
```

**✅ Solutions:**
```python
# Solution 1: Use appropriate metrics for imbalanced data
from sklearn.metrics import balanced_accuracy_score, matthews_corrcoef

balanced_acc = balanced_accuracy_score(y_true, y_pred)
mcc = matthews_corrcoef(y_true, y_pred)

print(f"Balanced Accuracy: {balanced_acc:.3f}")
print(f"Matthews Correlation Coefficient: {mcc:.3f}")

# Solution 2: Stratified sampling
from sklearn.model_selection import StratifiedKFold

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
scores = cross_val_score(model, X, y, cv=cv, scoring='f1_weighted')

# Solution 3: Handle class imbalance
from sklearn.utils.class_weight import compute_class_weight

# Compute class weights
class_weights = compute_class_weight(
    'balanced',
    classes=np.unique(y_train),
    y=y_train
)
weight_dict = dict(zip(np.unique(y_train), class_weights))

# Use in model
model = RandomForestClassifier(class_weight=weight_dict)

# Or use SMOTE for oversampling
from imblearn.over_sampling import SMOTE

smote = SMOTE(random_state=42)
X_resampled, y_resampled = smote.fit_resample(X_train, y_train)
```

---

## 🔧 Quick Debugging Checklist

### **🎯 Systematic Debugging Approach**

```python
def debug_ml_pipeline(X_train, X_test, y_train, y_test, model):
    """
    Systematic debugging checklist for ML pipelines
    """
    print("=== ML PIPELINE DEBUG CHECKLIST ===\n")
    
    # 1. Data shape and types
    print("1. DATA VALIDATION:")
    print(f"   X_train shape: {X_train.shape}")
    print(f"   X_test shape: {X_test.shape}")
    print(f"   y_train shape: {y_train.shape}")
    print(f"   y_test shape: {y_test.shape}")
    
    if X_train.shape[1] != X_test.shape[1]:
        print("   ❌ Feature mismatch between train and test")
    else:
        print("   ✅ Feature counts match")
    
    # 2. Missing values
    print("\n2. MISSING VALUES:")
    train_missing = X_train.isnull().sum().sum() if hasattr(X_train, 'isnull') else 0
    test_missing = X_test.isnull().sum().sum() if hasattr(X_test, 'isnull') else 0
    
    if train_missing > 0 or test_missing > 0:
        print(f"   ❌ Missing values: train={train_missing}, test={test_missing}")
    else:
        print("   ✅ No missing values detected")
    
    # 3. Data scaling
    print("\n3. DATA SCALING:")
    if hasattr(X_train, 'std'):
        train_std = X_train.std().mean()
        if train_std > 10 or train_std < 0.1:
            print(f"   ⚠️ Consider scaling (mean std: {train_std:.2f})")
        else:
            print("   ✅ Data appears reasonably scaled")
    
    # 4. Class distribution
    print("\n4. TARGET DISTRIBUTION:")
    y_dist = pd.Series(y_train).value_counts(normalize=True)
    min_class_ratio = y_dist.min()
    
    if min_class_ratio < 0.1:
        print(f"   ⚠️ Imbalanced classes (min class: {min_class_ratio:.2%})")
    else:
        print("   ✅ Reasonable class balance")
    
    # 5. Model training
    print("\n5. MODEL TRAINING:")
    try:
        model.fit(X_train, y_train)
        print("   ✅ Model trained successfully")
        
        # Check for overfitting
        train_score = model.score(X_train, y_train)
        test_score = model.score(X_test, y_test)
        
        if train_score - test_score > 0.1:
            print(f"   ⚠️ Possible overfitting (train: {train_score:.3f}, test: {test_score:.3f})")
        else:
            print(f"   ✅ Reasonable train/test performance (train: {train_score:.3f}, test: {test_score:.3f})")
            
    except Exception as e:
        print(f"   ❌ Model training failed: {str(e)}")
    
    print("\n=== DEBUG COMPLETE ===")

# Example usage
debug_ml_pipeline(X_train, X_test, y_train, y_test, RandomForestClassifier())
```

---

## 🔗 Related Cheat Sheets

- **[Scikit-Learn Quick Reference](../02_python_sklearn/sklearn_quick_reference.md)** - Implementation syntax
- **[Data Preprocessing Pipeline](../05_data_preprocessing/preprocessing_pipeline.md)** - Data issues prevention
- **[Classification Metrics](../04_model_evaluation/classification_metrics.md)** - Proper evaluation methods
- **[Algorithm Selection](../01_algorithms/algorithm_selection_flowchart.md)** - Choose right algorithm

---

## 💡 Prevention Best Practices

### **🎯 Error Prevention Strategies**

1. **Always validate data shapes** before training
2. **Use pipelines** to prevent preprocessing errors
3. **Set random_state** for reproducible debugging
4. **Start with simple models** before complex ones
5. **Use cross-validation** to catch overfitting early
6. **Monitor multiple metrics** not just accuracy
7. **Save intermediate results** during long experiments

### **🚨 Common Debugging Commands**

```python
# Essential debugging snippets
print(f"Data shape: {X.shape}")
print(f"Missing values: {X.isnull().sum().sum()}")
print(f"Data types: {X.dtypes.value_counts()}")
print(f"Target distribution: {pd.Series(y).value_counts()}")
print(f"Memory usage: {X.memory_usage(deep=True).sum() / 1024**2:.1f} MB")

# Model debugging
print(f"Model parameters: {model.get_params()}")
print(f"Feature importance: {model.feature_importances_ if hasattr(model, 'feature_importances_') else 'N/A'}")
print(f"Classes: {model.classes_ if hasattr(model, 'classes_') else 'N/A'}")
```

---

**🔧 Keep this guide handy** for faster debugging! **🔍 Use Ctrl+F** to find specific error messages quickly.

**🏠 [Back to Cheat Sheets](../README.md)** | **🎮 [Try Interactive Tools](../../06_playground/)** | **📊 [Evaluation Guide](../04_model_evaluation/classification_metrics.md)**