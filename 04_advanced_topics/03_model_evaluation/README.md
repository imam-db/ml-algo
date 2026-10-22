# Model Evaluation

**🇬🇧 English | [🇮🇩 Bahasa Indonesia](README.id.md)**

**🏠 [Back to Main](../../README.md)** | **📁 [Advanced Topics](../README.md)**

## 🗺️ Quick Access

| File | Description | Run |
|------|-------------|-----|
| [`implementation.py`](./implementation.py) | Implementation from scratch | `python3 implementation.py` |
| [`sklearn_example.py`](./sklearn_example.py) | Scikit-learn examples | `python3 sklearn_example.py` |
| [`exercise.ipynb`](./exercise.ipynb) | Interactive exercises | `jupyter lab exercise.ipynb` |

## 📖 Theory and Concepts

### Why is Model Evaluation Important?

**Model evaluation** is the process of measuring how well a machine learning model works. Without proper evaluation, we don't know if our model:
- Actually learns patterns or just memorizes data
- Will perform well on new data
- Is better than other models

**SPG Analogy:** Like an SPG evaluating their sales strategy performance by looking at conversion rate, revenue, and customer satisfaction - not just the number of customers approached.

## 🔄 Train/Test Split

### Basic Concept

**Problem:** If we evaluate a model on the same data used for training, results are too optimistic (overfitting goes undetected)

**Solution:** Split data into training set and test set

```python
from implementation import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    X, y, 
    test_size=0.2,      # 20% for test
    random_state=42,    # for reproducibility
    shuffle=True        # shuffle before split
)
```

### Best Practices

**Split Ratios:**
- **80/20 split**: Standard for large datasets (>1000 samples)
- **70/30 split**: For medium datasets
- **60/40 split**: For small datasets (<500 samples)

**Important:**
- Shuffle data before split (except for time series)
- Use stratified split for imbalanced classes
- Set random_state for reproducibility

## 📊 Classification Metrics

### 1. Confusion Matrix

**Definition:** A table showing predictions vs actuals

```
                Predicted
                Neg   Pos
Actual  Neg     TN    FP
        Pos     FN    TP
```

Where:
- **TP (True Positive)**: Predicted positive, actually positive ✅
- **TN (True Negative)**: Predicted negative, actually negative ✅
- **FP (False Positive)**: Predicted positive, actually negative ❌ (Type I Error)
- **FN (False Negative)**: Predicted negative, actually positive ❌ (Type II Error)

**SPG Example:**
```
Customer will buy (actual) vs SPG predicts will buy:
- TP: Predicted buy, actually bought (success! 🎉)
- TN: Predicted won't buy, actually didn't (saved time ✅)
- FP: Predicted buy, but didn't (wasted time ⚠️)
- FN: Predicted won't buy, but did (opportunity lost 💔)
```

### 2. Accuracy

**Formula:**
```
Accuracy = (TP + TN) / (TP + TN + FP + FN)
```

**When to use:**
- Classes are balanced
- All errors are equally important

**When NOT to use:**
- Imbalanced classes (misleading!)
- Different costs for FP vs FN

**Example:**
```python
from implementation import accuracy_score

accuracy = accuracy_score(y_true, y_pred)
# 0.95 = 95% predictions correct
```

### 3. Precision

**Formula:**
```
Precision = TP / (TP + FP)
```

**Meaning:** Of all positive predictions, how many are actually positive?

**When important:**
- Cost of FP is high
- We want confidence in positive predictions
- Medical diagnosis (avoid false alarms)
- Spam detection (avoid blocking real emails)

**SPG Example:**
High precision = When an SPG says "this customer will buy", they're usually right. The SPG doesn't make many wrong predictions.

### 4. Recall (Sensitivity)

**Formula:**
```
Recall = TP / (TP + FN)
```

**Meaning:** Of all actually positive cases, how many did we catch?

**When important:**
- Cost of FN is high
- We want to catch all positive cases
- Medical diagnosis (don't miss diseases)
- Fraud detection (catch all frauds)

**SPG Example:**
High recall = The SPG successfully identifies almost all customers who will actually buy. Not many opportunities are missed.

### 5. F1 Score

**Formula:**
```
F1 = 2 × (Precision × Recall) / (Precision + Recall)
```

**Meaning:** Harmonic mean of Precision and Recall

**When to use:**
- Need balance between Precision and Recall
- Imbalanced classes
- Single metric to compare models

**Interpretation:**
- F1 = 1.0: Perfect precision and recall
- F1 = 0.0: Worst case
- F1 > 0.8: Generally good

### Precision vs Recall Trade-off

```
High Precision, Low Recall:
  Model is conservative - only predicts positive when very confident
  Example: Spam filter that rarely makes mistakes, but misses some spam
  
Low Precision, High Recall:
  Model is aggressive - predicts positive more often
  Example: Medical screening that catches all diseases, but has many false alarms
```

### 6. Macro vs Micro Averaging (Multiclass)

**Macro Average:**
```
Macro = (Metric_class1 + Metric_class2 + ... + Metric_classN) / N
```
- Treats all classes equally
- Good for balanced classes
- Each class contributes equally

**Micro Average:**
```
Micro = Metric(all TP, all FP, all FN combined)
```
- Weighted by class frequency
- Good for imbalanced classes
- Large classes dominate

**Weighted Average:**
```
Weighted = Σ(Metric_class × support_class) / total_support
```
- Weighted by number of samples per class
- Best for imbalanced datasets

## 📈 ROC Curve and AUC

### ROC Curve (Receiver Operating Characteristic)

**Definition:** Plot of TPR vs FPR at various classification thresholds

```
TPR (True Positive Rate) = Recall = TP / (TP + FN)
FPR (False Positive Rate) = FP / (FP + TN)
```

**Interpretation:**
- **Perfect classifier**: TPR=1, FPR=0 (top-left corner)
- **Random classifier**: Diagonal line (TPR = FPR)
- **Bad classifier**: Below diagonal

### AUC (Area Under Curve)

**Interpretation:**
- **AUC = 1.0**: Perfect classifier
- **AUC = 0.5**: Random classifier (no discriminative power)
- **AUC < 0.5**: Worse than random (inverted predictions!)
- **AUC > 0.9**: Excellent
- **AUC 0.8-0.9**: Good
- **AUC 0.7-0.8**: Fair
- **AUC 0.6-0.7**: Poor
- **AUC < 0.6**: Very poor

**Advantages of AUC:**
- Threshold-independent
- Works with imbalanced classes
- Single number to compare models

**Example:**
```python
from implementation import roc_curve, roc_auc_score

# Compute ROC curve
fpr, tpr, thresholds = roc_curve(y_true, y_scores)

# Compute AUC
auc = roc_auc_score(y_true, y_scores)
print(f"AUC: {auc:.3f}")
```

## 🔄 Cross-Validation

### K-Fold Cross-Validation

**Concept:** Split data into K folds, train on K-1 folds, test on 1 fold, repeat K times

```
Fold 1: [TEST][TRAIN][TRAIN][TRAIN][TRAIN]
Fold 2: [TRAIN][TEST][TRAIN][TRAIN][TRAIN]
Fold 3: [TRAIN][TRAIN][TEST][TRAIN][TRAIN]
Fold 4: [TRAIN][TRAIN][TRAIN][TEST][TRAIN]
Fold 5: [TRAIN][TRAIN][TRAIN][TRAIN][TEST]
```

**Advantages:**
- Every data point is used for both training and testing
- More reliable estimate than single train/test split
- Reduces variance in performance estimate

**Best Practices:**
- **K=5**: Standard choice (good bias-variance trade-off)
- **K=10**: More computationally expensive, slightly better estimate
- **Leave-One-Out (K=n)**: Maximum data usage, very expensive

**Example:**
```python
from implementation import cross_val_score, k_fold_split

# K-fold splits
folds = k_fold_split(X, n_splits=5, shuffle=True, random_state=42)

# Cross-validation score
scores = cross_val_score(
    model, X, y, 
    cv=5, 
    scoring='accuracy',
    random_state=42
)

print(f"CV Scores: {scores}")
print(f"Mean: {scores.mean():.3f} (+/- {scores.std():.3f})")
```

### Stratified K-Fold

**For:** Imbalanced classes

**Concept:** Each fold has the same class distribution as the overall dataset

**When to use:**
- Classes are imbalanced
- Classification tasks
- Small datasets

## 📝 Classification Report

**Comprehensive summary** of precision, recall, F1 for each class

```python
from implementation import classification_report

report = classification_report(y_true, y_pred)
print(report)
```

**Example output:**
```
              precision    recall  f1-score   support

           0       0.88      0.85      0.86        48
           1       0.92      0.95      0.93        52

    macro avg       0.90      0.90      0.90       100
 weighted avg       0.90      0.90      0.90       100
 
     accuracy                           0.90       100
```

**Interpretation:**
- **support**: Number of samples per class
- **macro avg**: Unweighted mean (all classes equal)
- **weighted avg**: Weighted by support (preferred for imbalanced)
- **accuracy**: Overall accuracy

## 🎯 Choosing the Right Metric

### Decision Framework

**Question 1: Regression or Classification?**
- Regression → R², MSE, MAE
- Classification → Continue below

**Question 2: Binary or Multiclass?**
- Binary → Accuracy, Precision, Recall, F1, ROC-AUC
- Multiclass → Accuracy, Macro/Micro/Weighted F1

**Question 3: Balanced or Imbalanced classes?**
- Balanced → Accuracy is fine
- Imbalanced → F1, Precision, Recall, ROC-AUC

**Question 4: What's more important?**
- False Positives costly → Optimize **Precision**
- False Negatives costly → Optimize **Recall**
- Both equally important → Optimize **F1**
- Need ranking/probability → Optimize **ROC-AUC**

### Use Case Examples

**Medical Diagnosis (Cancer Detection):**
- **Metric:** Recall (Sensitivity)
- **Reason:** Missing cancer (FN) is very bad
- **Target:** Recall > 0.95

**Spam Detection:**
- **Metric:** Precision
- **Reason:** Blocking real email (FP) frustrates users
- **Target:** Precision > 0.95

**Customer Churn Prediction:**
- **Metric:** F1 or ROC-AUC
- **Reason:** Balance - want to catch churners without too many false alarms
- **Target:** F1 > 0.7

**Fraud Detection:**
- **Metric:** Recall, then Precision
- **Reason:** Must catch frauds, then filter false positives
- **Target:** Recall > 0.9, Precision > 0.5

## 💻 Best Practices

### 1. Always Use Train/Test Split

```python
# DON'T: Evaluate on training data
model.fit(X, y)
score = model.score(X, y)  # ❌ Optimistic!

# DO: Evaluate on unseen test data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
model.fit(X_train, y_train)
score = model.score(X_test, y_test)  # ✅ Realistic!
```

### 2. Use Cross-Validation for Model Selection

```python
# Compare models with CV
models = [LogisticRegression(), RandomForest(), KNN()]

for model in models:
    scores = cross_val_score(model, X, y, cv=5)
    print(f"{model.__class__.__name__}: {scores.mean():.3f} (+/- {scores.std():.3f})")
```

### 3. Report Multiple Metrics

```python
# Don't rely on single metric
from implementation import accuracy_score, precision_score, recall_score, f1_score

print(f"Accuracy:  {accuracy_score(y_test, y_pred):.3f}")
print(f"Precision: {precision_score(y_test, y_pred):.3f}")
print(f"Recall:    {recall_score(y_test, y_pred):.3f}")
print(f"F1 Score:  {f1_score(y_test, y_pred):.3f}")
```

### 4. Visualize Results

```python
from implementation import plot_confusion_matrix, plot_roc_curve_viz

# Confusion matrix
cm = confusion_matrix(y_test, y_pred)
plot_confusion_matrix(cm, class_names=['No Buy', 'Buy'])

# ROC curve
fpr, tpr, _ = roc_curve(y_test, y_scores)
auc = roc_auc_score(y_test, y_scores)
plot_roc_curve_viz(fpr, tpr, auc)
```

### 5. Consider Class Imbalance

```python
# For imbalanced data, use:
# - Stratified splitting
# - Weighted metrics (macro/micro/weighted)
# - ROC-AUC instead of accuracy
# - F1 score

from sklearn.model_selection import StratifiedKFold
skf = StratifiedKFold(n_splits=5)

# Use weighted F1
f1_weighted = f1_score(y_test, y_pred, average='weighted')
```

## 📊 Common Pitfalls

### ❌ Pitfall 1: Data Leakage

```python
# WRONG: Scale before split (leaks test data info into training!)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
X_train, X_test = train_test_split(X_scaled, ...)  # ❌

# RIGHT: Split first, then scale
X_train, X_test = train_test_split(X, ...)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)  # Fit on train
X_test_scaled = scaler.transform(X_test)        # Transform test  # ✅
```

### ❌ Pitfall 2: Using Accuracy for Imbalanced Data

```python
# Example: 95% class A, 5% class B
# Model that predicts all A → 95% accuracy! But useless.

# Use F1, Precision/Recall, or ROC-AUC instead
```

### ❌ Pitfall 3: Not Setting Random State

```python
# Different splits every run → results not reproducible
X_train, X_test = train_test_split(X, y)  # ❌

# Set random_state for reproducibility
X_train, X_test = train_test_split(X, y, random_state=42)  # ✅
```

### ❌ Pitfall 4: Overfitting to Validation Set

```python
# Don't tune hyperparameters against test set!
# Use train/validation/test split or nested CV

# Option 1: Train/Val/Test split
X_temp, X_test, y_temp, y_test = train_test_split(X, y, test_size=0.2)
X_train, X_val, y_train, y_val = train_test_split(X_temp, y_temp, test_size=0.25)

# Option 2: Nested cross-validation
# Outer loop: performance estimation
# Inner loop: hyperparameter tuning
```

## 📚 Learning Resources

- [Scikit-learn Model Evaluation](https://scikit-learn.org/stable/modules/model_evaluation.html)
- [Precision vs Recall - Google ML Crash Course](https://developers.google.com/machine-learning/crash-course/classification/precision-and-recall)
- [ROC Curves and AUC Explained](https://www.youtube.com/watch?v=4jRBRDbJemM)

## 🎓 Exercises

See [`exercise.ipynb`](./exercise.ipynb) for hands-on practice:
1. Train/test split and evaluation
2. Confusion matrix interpretation
3. Precision vs Recall trade-off
4. ROC curve analysis
5. Cross-validation comparison
6. Handling imbalanced data

---

**Happy Learning Model Evaluation! 📊**
