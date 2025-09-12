# 📊 Classification Metrics Cheat Sheet

**🏠 [Cheat Sheets Home](../README.md)** | **🤖 [Algorithms Reference](../01_algorithms/)** | **🐍 [Scikit-Learn Quick Reference](../02_python_sklearn/)**

---

## 🎯 Quick Summary

Comprehensive guide to classification performance metrics - when to use each metric and how to interpret results.

---

## ⚡ Essential Imports

```python
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, roc_curve, precision_recall_curve,
    classification_report, confusion_matrix,
    average_precision_score, cohen_kappa_score,
    matthews_corrcoef, balanced_accuracy_score
)
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
```

---

## 🎯 Core Metrics Quick Reference

### **📊 Confusion Matrix - The Foundation**
```python
from sklearn.metrics import confusion_matrix
import seaborn as sns

# Calculate confusion matrix
cm = confusion_matrix(y_true, y_pred)

# Visualize
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
            xticklabels=['Predicted 0', 'Predicted 1'],
            yticklabels=['Actual 0', 'Actual 1'])
plt.title('Confusion Matrix')
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.show()

# Extract values
tn, fp, fn, tp = cm.ravel()  # For binary classification
print(f"True Negatives: {tn}")
print(f"False Positives: {fp}")
print(f"False Negatives: {fn}")
print(f"True Positives: {tp}")
```

---

## 🎯 Primary Metrics

### **✅ Accuracy**
**When to use:** Balanced datasets, equal cost for all errors  
**Formula:** (TP + TN) / (TP + TN + FP + FN)

```python
from sklearn.metrics import accuracy_score

accuracy = accuracy_score(y_true, y_pred)
print(f"Accuracy: {accuracy:.3f}")

# Manual calculation
accuracy_manual = (tp + tn) / (tp + tn + fp + fn)
```

**💡 Key Points:**
- ✅ Easy to understand and interpret
- ❌ Misleading with imbalanced datasets
- ❌ Doesn't show which classes are confused

---

### **🎯 Precision (Positive Predictive Value)**
**When to use:** When false positives are costly (spam detection, medical diagnosis)  
**Formula:** TP / (TP + FP)

```python
from sklearn.metrics import precision_score

# Binary classification
precision = precision_score(y_true, y_pred)

# Multiclass
precision_macro = precision_score(y_true, y_pred, average='macro')
precision_weighted = precision_score(y_true, y_pred, average='weighted')
precision_per_class = precision_score(y_true, y_pred, average=None)

print(f"Precision: {precision:.3f}")
```

**💡 Key Points:**
- "Of all positive predictions, how many were correct?"
- ✅ Good when false positives are expensive
- ❌ Ignores false negatives

---

### **🔍 Recall (Sensitivity, True Positive Rate)**
**When to use:** When false negatives are costly (cancer detection, fraud detection)  
**Formula:** TP / (TP + FN)

```python
from sklearn.metrics import recall_score

recall = recall_score(y_true, y_pred)
recall_per_class = recall_score(y_true, y_pred, average=None)

print(f"Recall: {recall:.3f}")
```

**💡 Key Points:**
- "Of all actual positives, how many did we catch?"
- ✅ Good when missing positives is expensive
- ❌ Ignores false positives

---

### **⚖️ F1-Score**
**When to use:** Balance between precision and recall, imbalanced datasets  
**Formula:** 2 × (Precision × Recall) / (Precision + Recall)

```python
from sklearn.metrics import f1_score

f1 = f1_score(y_true, y_pred)
f1_macro = f1_score(y_true, y_pred, average='macro')
f1_weighted = f1_score(y_true, y_pred, average='weighted')

print(f"F1-Score: {f1:.3f}")
```

**💡 Key Points:**
- Harmonic mean of precision and recall
- ✅ Good single metric for imbalanced data
- ❌ May not reflect business costs

---

## 🎨 Advanced Metrics

### **📈 ROC-AUC (Receiver Operating Characteristic)**
**When to use:** Binary classification, ranking quality, balanced datasets

```python
from sklearn.metrics import roc_auc_score, roc_curve

# Calculate AUC
auc = roc_auc_score(y_true, y_proba)

# Plot ROC curve
fpr, tpr, thresholds = roc_curve(y_true, y_proba)

plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {auc:.2f})')
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='Random classifier')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic')
plt.legend(loc="lower right")
plt.show()

print(f"ROC-AUC: {auc:.3f}")
```

**💡 Key Points:**
- Measures ranking quality across all thresholds
- ✅ Threshold-independent
- ❌ Overly optimistic on imbalanced datasets

---

### **📊 Precision-Recall AUC**
**When to use:** Imbalanced datasets, when positive class is rare

```python
from sklearn.metrics import average_precision_score, precision_recall_curve

# Calculate PR-AUC
pr_auc = average_precision_score(y_true, y_proba)

# Plot Precision-Recall curve
precision, recall, thresholds = precision_recall_curve(y_true, y_proba)

plt.figure(figsize=(8, 6))
plt.plot(recall, precision, color='blue', lw=2, label=f'PR curve (AUC = {pr_auc:.2f})')
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title('Precision-Recall Curve')
plt.legend()
plt.show()

print(f"PR-AUC: {pr_auc:.3f}")
```

**💡 Key Points:**
- Better than ROC-AUC for imbalanced datasets
- ✅ Focuses on positive class performance
- ❌ More difficult to interpret

---

### **⚖️ Balanced Accuracy**
**When to use:** Imbalanced datasets, equal importance of all classes

```python
from sklearn.metrics import balanced_accuracy_score

balanced_acc = balanced_accuracy_score(y_true, y_pred)
print(f"Balanced Accuracy: {balanced_acc:.3f}")

# Manual calculation
sensitivity = tp / (tp + fn)  # Recall for positive class
specificity = tn / (tn + fp)  # Recall for negative class
balanced_acc_manual = (sensitivity + specificity) / 2
```

**💡 Key Points:**
- Average of recall for each class
- ✅ Not biased by class imbalance
- ✅ Easy to interpret

---

### **🎯 Cohen's Kappa**
**When to use:** Account for chance agreement, multi-rater scenarios

```python
from sklearn.metrics import cohen_kappa_score

kappa = cohen_kappa_score(y_true, y_pred)
print(f"Cohen's Kappa: {kappa:.3f}")

# Interpretation
if kappa < 0:
    interpretation = "Poor agreement"
elif kappa < 0.20:
    interpretation = "Slight agreement"
elif kappa < 0.40:
    interpretation = "Fair agreement"
elif kappa < 0.60:
    interpretation = "Moderate agreement"
elif kappa < 0.80:
    interpretation = "Substantial agreement"
else:
    interpretation = "Almost perfect agreement"

print(f"Interpretation: {interpretation}")
```

**💡 Key Points:**
- Adjusts for chance agreement
- ✅ Good for imbalanced datasets
- Range: -1 to 1 (1 = perfect agreement)

---

### **🔢 Matthews Correlation Coefficient (MCC)**
**When to use:** Binary classification, imbalanced datasets, single balanced metric

```python
from sklearn.metrics import matthews_corrcoef

mcc = matthews_corrcoef(y_true, y_pred)
print(f"Matthews Correlation Coefficient: {mcc:.3f}")

# Interpretation
if mcc > 0.8:
    interpretation = "Very strong"
elif mcc > 0.6:
    interpretation = "Strong"
elif mcc > 0.4:
    interpretation = "Moderate"
elif mcc > 0.2:
    interpretation = "Weak"
else:
    interpretation = "Very weak or no correlation"

print(f"Interpretation: {interpretation}")
```

**💡 Key Points:**
- Balanced metric even for very imbalanced datasets
- Range: -1 to 1 (1 = perfect prediction)
- ✅ Single metric that considers all confusion matrix categories

---

## 📋 Multiclass Metrics

### **🎯 Averaging Methods**
```python
# Different averaging strategies for multiclass
precision_micro = precision_score(y_true, y_pred, average='micro')    # Global
precision_macro = precision_score(y_true, y_pred, average='macro')    # Unweighted mean
precision_weighted = precision_score(y_true, y_pred, average='weighted')  # Weighted by support
precision_per_class = precision_score(y_true, y_pred, average=None)   # Per class

print(f"Micro-averaged Precision: {precision_micro:.3f}")
print(f"Macro-averaged Precision: {precision_macro:.3f}")
print(f"Weighted-averaged Precision: {precision_weighted:.3f}")
print(f"Per-class Precision: {precision_per_class}")
```

**💡 Averaging Methods:**
- **Micro:** Calculate globally (good for imbalanced datasets)
- **Macro:** Simple average (treats all classes equally)
- **Weighted:** Average weighted by class frequency
- **None:** Return score for each class individually

---

### **📊 Classification Report**
```python
from sklearn.metrics import classification_report

report = classification_report(y_true, y_pred)
print("Classification Report:")
print(report)

# As dictionary for programmatic access
report_dict = classification_report(y_true, y_pred, output_dict=True)
print(f"\nOverall Accuracy: {report_dict['accuracy']:.3f}")
print(f"Macro F1-Score: {report_dict['macro avg']['f1-score']:.3f}")
```

---

## 🎯 Metric Selection Guide

### **📊 By Dataset Characteristics**

| Dataset Type | Primary Metrics | Secondary Metrics |
|-------------|-----------------|-------------------|
| **Balanced** | Accuracy, F1-Score | ROC-AUC, Precision, Recall |
| **Imbalanced** | F1-Score, PR-AUC, Balanced Accuracy | MCC, Cohen's Kappa |
| **Multi-class Balanced** | Accuracy, Macro F1 | Per-class metrics |
| **Multi-class Imbalanced** | Weighted F1, Balanced Accuracy | Micro F1, MCC |

### **🎯 By Business Context**

| Use Case | Primary Metric | Reasoning |
|----------|---------------|-----------|
| **Spam Detection** | Precision | False positives annoy users |
| **Medical Diagnosis** | Recall | Missing disease is costly |
| **Fraud Detection** | F1-Score, PR-AUC | Balance precision/recall, rare events |
| **A/B Testing** | Statistical significance tests | Need confidence intervals |
| **Ranking Systems** | ROC-AUC, NDCG | Order matters more than classification |

---

## 🛠️ Complete Evaluation Pipeline

### **📊 Comprehensive Evaluation Function**
```python
def comprehensive_classification_report(y_true, y_pred, y_proba=None, class_names=None):
    """
    Generate comprehensive classification evaluation report
    """
    print("=== CLASSIFICATION EVALUATION REPORT ===\n")
    
    # Basic metrics
    accuracy = accuracy_score(y_true, y_pred)
    balanced_acc = balanced_accuracy_score(y_true, y_pred)
    
    print(f"📊 BASIC METRICS")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Balanced Accuracy: {balanced_acc:.4f}")
    
    # Classification report
    print(f"\n📋 DETAILED CLASSIFICATION REPORT")
    print(classification_report(y_true, y_pred, target_names=class_names))
    
    # Advanced metrics
    if len(np.unique(y_true)) == 2:  # Binary classification
        precision = precision_score(y_true, y_pred)
        recall = recall_score(y_true, y_pred)
        f1 = f1_score(y_true, y_pred)
        mcc = matthews_corrcoef(y_true, y_pred)
        kappa = cohen_kappa_score(y_true, y_pred)
        
        print(f"\n🎯 ADVANCED BINARY METRICS")
        print(f"Precision: {precision:.4f}")
        print(f"Recall: {recall:.4f}")
        print(f"F1-Score: {f1:.4f}")
        print(f"Matthews Correlation Coefficient: {mcc:.4f}")
        print(f"Cohen's Kappa: {kappa:.4f}")
        
        if y_proba is not None:
            auc = roc_auc_score(y_true, y_proba)
            pr_auc = average_precision_score(y_true, y_proba)
            print(f"ROC-AUC: {auc:.4f}")
            print(f"PR-AUC: {pr_auc:.4f}")
    
    # Confusion matrix
    print(f"\n📊 CONFUSION MATRIX")
    cm = confusion_matrix(y_true, y_pred)
    print(cm)
    
    # Visual confusion matrix
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=class_names or [f'Class {i}' for i in range(len(np.unique(y_true)))],
                yticklabels=class_names or [f'Class {i}' for i in range(len(np.unique(y_true)))])
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.show()

# Example usage
comprehensive_classification_report(y_test, y_pred, y_proba, class_names=['Negative', 'Positive'])
```

---

### **📈 ROC and PR Curves Side by Side**
```python
def plot_roc_pr_curves(y_true, y_proba, title="Model Performance"):
    """
    Plot ROC and Precision-Recall curves side by side
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # ROC Curve
    fpr, tpr, _ = roc_curve(y_true, y_proba)
    auc_roc = roc_auc_score(y_true, y_proba)
    
    ax1.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {auc_roc:.2f})')
    ax1.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='Random')
    ax1.set_xlim([0.0, 1.0])
    ax1.set_ylim([0.0, 1.05])
    ax1.set_xlabel('False Positive Rate')
    ax1.set_ylabel('True Positive Rate')
    ax1.set_title('ROC Curve')
    ax1.legend(loc="lower right")
    ax1.grid(True)
    
    # Precision-Recall Curve
    precision, recall, _ = precision_recall_curve(y_true, y_proba)
    auc_pr = average_precision_score(y_true, y_proba)
    
    ax2.plot(recall, precision, color='blue', lw=2, label=f'PR curve (AUC = {auc_pr:.2f})')
    ax2.axhline(y=np.mean(y_true), color='red', linestyle='--', label='Baseline')
    ax2.set_xlim([0.0, 1.0])
    ax2.set_ylim([0.0, 1.05])
    ax2.set_xlabel('Recall')
    ax2.set_ylabel('Precision')
    ax2.set_title('Precision-Recall Curve')
    ax2.legend()
    ax2.grid(True)
    
    plt.tight_layout()
    plt.suptitle(title, y=1.02)
    plt.show()
    
    return auc_roc, auc_pr

# Example usage
auc_roc, auc_pr = plot_roc_pr_curves(y_test, y_proba_positive, "Random Forest Performance")
```

---

## ⚠️ Common Pitfalls & Solutions

### **🚨 Imbalanced Dataset Issues**
```python
# ❌ WRONG: Using accuracy on imbalanced data
print(f"Accuracy: {accuracy_score(y_true, y_pred):.3f}")  # Misleading!

# ✅ CORRECT: Use appropriate metrics
print(f"Balanced Accuracy: {balanced_accuracy_score(y_true, y_pred):.3f}")
print(f"F1-Score: {f1_score(y_true, y_pred):.3f}")
print(f"MCC: {matthews_corrcoef(y_true, y_pred):.3f}")
```

### **🚨 Threshold Selection**
```python
# Find optimal threshold for F1-score
from sklearn.metrics import f1_score

thresholds = np.linspace(0, 1, 101)
f1_scores = []

for threshold in thresholds:
    y_pred_threshold = (y_proba >= threshold).astype(int)
    f1_scores.append(f1_score(y_true, y_pred_threshold))

optimal_threshold = thresholds[np.argmax(f1_scores)]
print(f"Optimal threshold for F1: {optimal_threshold:.3f}")

# Apply optimal threshold
y_pred_optimal = (y_proba >= optimal_threshold).astype(int)
```

### **🚨 Multiclass Averaging Confusion**
```python
# Understanding different averaging methods
print("=== MULTICLASS AVERAGING COMPARISON ===")
print(f"Micro F1 (global): {f1_score(y_true, y_pred, average='micro'):.3f}")
print(f"Macro F1 (unweighted): {f1_score(y_true, y_pred, average='macro'):.3f}")  
print(f"Weighted F1 (by support): {f1_score(y_true, y_pred, average='weighted'):.3f}")

# Per-class breakdown
f1_per_class = f1_score(y_true, y_pred, average=None)
for i, score in enumerate(f1_per_class):
    print(f"Class {i} F1: {score:.3f}")
```

---

## 🔗 Related Cheat Sheets

- **[Algorithms Quick Reference](../01_algorithms/algorithms_quick_reference.md)** - Choose right algorithm
- **[Scikit-Learn Quick Reference](../02_python_sklearn/sklearn_quick_reference.md)** - Implementation syntax
- **[Data Preprocessing](../05_data_preprocessing/preprocessing_pipeline.md)** - Prepare data properly
- **[Hyperparameter Tuning](../07_troubleshooting/hyperparameter_tuning.md)** - Optimize performance

---

## 💡 Key Takeaways

### **🎯 Golden Rules**
1. **Always check data balance** before choosing metrics
2. **Use multiple metrics** - no single metric tells the whole story
3. **Consider business context** when selecting primary metric
4. **Visualize confusion matrix** to understand model behavior
5. **Cross-validate metrics** for robust evaluation

### **🚨 Common Mistakes**
- ❌ Using accuracy on imbalanced datasets
- ❌ Ignoring class distribution when interpreting metrics
- ❌ Not using probability scores when available
- ❌ Choosing metrics without considering business costs
- ❌ Not validating metrics across different data splits

---

**📋 Bookmark this reference** for evaluation guidance! **🔍 Use Ctrl+F** to find specific metrics quickly.

**🏠 [Back to Cheat Sheets](../README.md)** | **🎮 [Try Interactive Tools](../../06_playground/)** | **🤖 [Algorithm Selection](../01_algorithms/algorithm_selection_flowchart.md)**