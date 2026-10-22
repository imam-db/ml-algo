# K-Nearest Neighbors (KNN)

**🇬🇧 English | [🇮🇩 Bahasa Indonesia](README.id.md)**

## What is KNN?

K-Nearest Neighbors is a simple, intuitive supervised learning algorithm that makes predictions based on the K closest training examples.

### Key Idea

**"Show me your friends, and I'll tell you who you are"**

- Find K nearest neighbors to a new point
- Classification: Majority vote of neighbors
- Regression: Average of neighbors

---

## How it Works

### Classification

```
1. Calculate distance from new point to all training points
2. Find K nearest neighbors
3. Take majority vote of their labels
4. Assign that label to new point
```

### Regression

```
1. Calculate distance from new point to all training points
2. Find K nearest neighbors
3. Average their target values
4. Assign that average to new point
```

---

## Distance Metrics

### Euclidean Distance (default)

$$d(p, q) = \sqrt{\sum_{i=1}^{n} (p_i - q_i)^2}$$

### Manhattan Distance

$$d(p, q) = \sum_{i=1}^{n} |p_i - q_i|$$

### Minkowski Distance

$$d(p, q) = \left(\sum_{i=1}^{n} |p_i - q_i|^p\right)^{1/p}$$

---

## Choosing K

- **Small K** (1-3): More complex boundaries, sensitive to noise
- **Large K** (20+): Smoother boundaries, may underfit
- **Rule of thumb**: K = √n (n = number of samples)
- **Best practice**: Try multiple K values, use cross-validation

---

## Feature Scaling

**Critical for KNN!** Distance-based algorithms require scaled features.

```python
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
```

---

## Implementation

```python
from implementation import KNNClassifier, KNNRegressor

# Classification
knn = KNNClassifier(n_neighbors=5, metric='euclidean')
knn.fit(X_train, y_train)
predictions = knn.predict(X_test)

# Regression
knn_reg = KNNRegressor(n_neighbors=5, weights='distance')
knn_reg.fit(X_train, y_train)
predictions = knn_reg.predict(X_test)
```

---

## Advantages and Disadvantages

### ✅ Advantages

1. **Simple and intuitive**
2. **No training phase** (lazy learning)
3. **Works for classification and regression**
4. **Naturally handles multi-class**

### ❌ Disadvantages

1. **Slow prediction** (must search all training points)
2. **Memory intensive** (stores all training data)
3. **Sensitive to irrelevant features**
4. **Curse of dimensionality** (poor in high dimensions)

---

## References

- **Sklearn**: [`sklearn.neighbors.KNeighborsClassifier`](https://scikit-learn.org/stable/modules/generated/sklearn.neighbors.KNeighborsClassifier.html)

---

## Related Files

- `implementation.py` - KNN from scratch
- `sklearn_example.py` - Comparison with sklearn
- `exercise.ipynb` - Interactive exercises

**Happy Learning! 🎓**

*[🇮🇩 Indonesian version](README.id.md)*
