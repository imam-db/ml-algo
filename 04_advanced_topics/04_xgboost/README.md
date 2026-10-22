# XGBoost

**🇬🇧 English | [🇮🇩 Bahasa Indonesia](README.id.md)**

## Overview

XGBoost (eXtreme Gradient Boosting) is an optimized gradient boosting library known for speed and performance in machine learning competitions.

## Key Features

- Regularization (L1 & L2)
- Parallel processing
- Tree pruning
- Handles missing values
- Built-in cross-validation

## When to Use

### ✅ Excellent for:

- **Structured/tabular data** (CSV, databases)
- **Data with missing values**
- **Feature importance analysis**
- **Need high performance**
- **Competition/benchmark tasks**

### ❌ Not ideal for:

- **Unstructured data** (images, text, audio - use deep learning)
- **Very small datasets** (< 1000 samples)
- **Data with complex temporal structure**
- **Real-time predictions** (inference can be slow)

## Advantages and Disadvantages

### ✅ Advantages

- ✅ **State-of-the-art performance** on tabular data
- ✅ **Accurate feature importance**
- ✅ **Handles missing values** automatically
- ✅ **Early stopping** to prevent overfitting
- ✅ **Regularization** built-in

### ❌ Disadvantages

- ❌ **Complex hyperparameter tuning**
- ❌ **Memory intensive** for large datasets
- ❌ **Prone to overfitting** if not tuned properly
- ❌ **Black box** model (harder to interpret than decision trees)

## Implementation

```python
import xgboost as xgb

model = xgb.XGBClassifier(
    n_estimators=100,
    max_depth=5,
    learning_rate=0.1,
    random_state=42
)
model.fit(X_train, y_train)
predictions = model.predict(X_test)
```

## Key Hyperparameters

- `n_estimators`: Number of boosting rounds
- `max_depth`: Maximum tree depth
- `learning_rate`: Step size shrinkage
- `subsample`: Row sampling ratio
- `colsample_bytree`: Column sampling ratio

## Related Files

- Tutorial notebooks and examples
- Hyperparameter tuning scripts

**Happy Learning! 🎓** | *[🇮🇩 Bahasa Indonesia](README.id.md)*
