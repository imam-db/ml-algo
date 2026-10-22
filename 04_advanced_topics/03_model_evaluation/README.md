# Model Evaluation

**🇬🇧 English | [🇮🇩 Bahasa Indonesia](README.id.md)**

## Overview

Comprehensive guide to evaluating machine learning models using various metrics and validation techniques.

## Metrics

### Classification Metrics

- **Accuracy**: Correct predictions / Total predictions
- **Precision**: True positives / (True positives + False positives)
- **Recall (Sensitivity)**: True positives / (True positives + False negatives)
- **F1-Score**: Harmonic mean of precision and recall
- **ROC-AUC**: Area under ROC curve

### Confusion Matrix

```
              Predicted
           Pos    Neg
Actual Pos  TP    FN
       Neg  FP    TN
```

## Cross-Validation

### K-Fold Cross-Validation

1. Split data into K folds
2. Train on K-1 folds
3. Validate on remaining fold
4. Repeat K times
5. Average results

## Implementation

```python
from implementation import (
    train_test_split, confusion_matrix,
    accuracy_score, precision_score, recall_score, f1_score,
    roc_curve, roc_auc_score,
    cross_val_score
)

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Evaluate
accuracy = accuracy_score(y_test, predictions)
precision = precision_score(y_test, predictions)
recall = recall_score(y_test, predictions)
f1 = f1_score(y_test, predictions)

# ROC-AUC
fpr, tpr, thresholds = roc_curve(y_test, probabilities)
auc = roc_auc_score(y_test, probabilities)

# Cross-validation
scores = cross_val_score(model, X, y, cv=5)
```

## Averaging Methods (Multiclass)

- **Macro**: Unweighted mean (treats all classes equally)
- **Micro**: Global average (good for imbalanced data)
- **Weighted**: Weighted by class support

## Related Files

- `implementation.py` - Metrics and CV from scratch
- `sklearn_example.py` - Comparison with sklearn
- `exercise.ipynb` - Interactive exercises

**Happy Learning! 🎓** | *[🇮🇩 Bahasa Indonesia](README.id.md)*
