# K-Nearest Neighbors (KNN)

**🇬🇧 English | [🇮🇩 Bahasa Indonesia](README.id.md)**

**🏠 [Back to Main](../../README.md)** | **📁 [Supervised Learning](../README.md)**

## 🗺️ Quick Access

| File | Description | Run |
|------|-------------|-----|
| [`implementation.py`](./implementation.py) | Implementation from scratch | `python3 implementation.py` |
| [`sklearn_example.py`](./sklearn_example.py) | Scikit-learn examples | `python3 sklearn_example.py` |
| [`exercise.ipynb`](./exercise.ipynb) | Interactive exercises | `jupyter lab exercise.ipynb` |

## 📖 Theory and Concepts

### What is K-Nearest Neighbors?

**K-Nearest Neighbors (KNN)** is a simple yet powerful supervised learning algorithm. KNN makes predictions based on K nearest samples in the training data. This algorithm is **instance-based** or **lazy learning** because it doesn't build an explicit model during training.

**SPG (Sales Promoter Girl) Analogy:**
Imagine a new SPG who wants to know which product is suitable for a customer. She will look at the K nearest customers (most similar) and see what products they bought, then recommend based on the majority.

### How KNN Works

**Classification:**
```
1. Calculate distance from test sample to all training samples
2. Select K nearest samples
3. Voting: choose the class that appears most often in K neighbors
4. Return that class
```

**Regression:**
```
1. Calculate distance from test sample to all training samples
2. Select K nearest samples
3. Average: calculate mean of K neighbors' values
4. Return the average value
```

### Distance Metrics

#### 1. Euclidean Distance (L2)
```
d(x, y) = √(Σ(xi - yi)²)
```
- Most commonly used
- Sensitive to outliers
- Good for continuous data

#### 2. Manhattan Distance (L1)
```
d(x, y) = Σ|xi - yi|
```
- More robust to outliers
- Suitable for grid-like data
- Faster to compute

#### 3. Minkowski Distance (generalization)
```
d(x, y) = (Σ|xi - yi|^p)^(1/p)
```
- p=1: Manhattan
- p=2: Euclidean
- p→∞: Chebyshev

### Weighting Schemes

**Uniform Weights:**
- All neighbors have equal weight
- Simple majority vote / average
- Suitable if neighbors are relatively uniform

**Distance Weights:**
- Closer neighbors have greater weight
- Weight = 1 / distance
- More accurate if distance varies
- Prevents tie votes

### Choosing K

**Small K (k=1, 3, 5):**
- ✅ Sensitive to local patterns
- ✅ Can capture complex decision boundaries
- ❌ Sensitive to noise
- ❌ High overfitting risk

**Large K (k=20, 50, 100):**
- ✅ More stable, less sensitive to noise
- ✅ Smoother decision boundaries
- ❌ Can be too smooth (underfitting)
- ❌ Slower computation

**Optimal K:**
- Use cross-validation
- Plot accuracy vs k (elbow method)
- Usually k = √n (rule of thumb)
- Odd K for classification (avoid ties)

## 🎯 When to Use KNN?

### ✅ Good for:
- **Small to medium datasets**
  - Training O(1), prediction O(n)
- **Non-linear decision boundaries**
  - No assumptions about data distribution
- **Multi-class classification**
  - Natural extension, no need for one-vs-rest
- **Prototype for baseline**
  - Easy to implement
- **Anomaly detection**
  - K-distance to detect outliers

### ❌ Less suitable for:
- **Very large datasets**
  - Slow prediction (must scan all data)
- **High-dimensional data (curse of dimensionality)**
  - Distance becomes meaningless
  - Needs dimensionality reduction
- **Imbalanced classes**
  - Majority class will dominate
- **Features with different scales**
  - Feature scaling MANDATORY
- **Real-time prediction**
  - Too slow for production

## 💡 Implementation

### KNN Classifier

```python
from implementation import KNNClassifier
from sklearn.preprocessing import StandardScaler

# IMPORTANT: Scale features!
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Create and train
knn = KNNClassifier(
    n_neighbors=5,
    weights='uniform',  # or 'distance'
    metric='euclidean'  # or 'manhattan', 'minkowski'
)
knn.fit(X_train_scaled, y_train)

# Predict
predictions = knn.predict(X_test_scaled)
probabilities = knn.predict_proba(X_test_scaled)
accuracy = knn.score(X_test_scaled, y_test)
```

### KNN Regressor

```python
from implementation import KNNRegressor

knn_reg = KNNRegressor(n_neighbors=5, weights='distance')
knn_reg.fit(X_train_scaled, y_train)

predictions = knn_reg.predict(X_test_scaled)
r2_score = knn_reg.score(X_test_scaled, y_test)
```

### Choosing K

```python
from implementation import choose_k_elbow_method, plot_k_selection

k_values, scores = choose_k_elbow_method(
    X_train, y_train, X_val, y_val,
    k_range=range(1, 31),
    task='classification'
)

# Plot
fig = plot_k_selection(k_values, scores)
plt.show()
```

## 🔧 Best Practices

### 1. Feature Scaling (CRITICAL!)

KNN depends on distance, so feature scaling is essential:

```python
from sklearn.preprocessing import StandardScaler, MinMaxScaler

# StandardScaler (mean=0, std=1) - recommended
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_train)

# MinMaxScaler (0-1 range)
scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(X_train)
```

**Without scaling:**
- Features with large ranges will dominate
- Example: salary (0-100000) vs age (0-100)
- Distance will be dominated by salary

### 2. Handling High Dimensions

**Curse of Dimensionality:**
- More dimensions = sparser data
- Distance becomes less meaningful
- All points appear "far"

**Solutions:**
```python
# PCA for dimensionality reduction
from sklearn.decomposition import PCA
pca = PCA(n_components=10)
X_reduced = pca.fit_transform(X)

# Feature selection
from sklearn.feature_selection import SelectKBest
selector = SelectKBest(k=10)
X_selected = selector.fit_transform(X, y)
```

### 3. Handling Imbalanced Data

```python
# Option 1: Distance weighting
knn = KNNClassifier(n_neighbors=5, weights='distance')

# Option 2: SMOTE for resampling
from imblearn.over_sampling import SMOTE
smote = SMOTE()
X_resampled, y_resampled = smote.fit_resample(X, y)

# Option 3: Larger K
knn = KNNClassifier(n_neighbors=15)  # Reduces majority bias
```

### 4. Efficient Implementation

```python
# Use KD-Tree or Ball-Tree for large datasets
from sklearn.neighbors import KNeighborsClassifier

knn = KNeighborsClassifier(
    n_neighbors=5,
    algorithm='auto'  # chooses best: kd_tree, ball_tree, brute
)
```

## 📊 Hyperparameter Tuning

### Grid Search

```python
param_grid = {
    'n_neighbors': [3, 5, 7, 9, 11],
    'weights': ['uniform', 'distance'],
    'metric': ['euclidean', 'manhattan']
}

# Manual grid search or use sklearn
best_params = {'n_neighbors': 5, 'weights': 'distance', 'metric': 'euclidean'}
```

### Cross-Validation

```python
from sklearn.model_selection import cross_val_score

scores = []
for k in range(1, 31):
    knn = KNNClassifier(n_neighbors=k)
    # Assume we have cross-validation implementation
    score = np.mean([...])  # CV scores
    scores.append(score)

best_k = np.argmax(scores) + 1
```

## 🔍 Interpretation and Debugging

### Visualizing Decision Boundaries

```python
import matplotlib.pyplot as plt

# For 2D data
def plot_decision_boundary(model, X, y):
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    
    xx, yy = np.meshgrid(
        np.linspace(x_min, x_max, 200),
        np.linspace(y_min, y_max, 200)
    )
    
    Z = model.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    
    plt.contourf(xx, yy, Z, alpha=0.3)
    plt.scatter(X[:, 0], X[:, 1], c=y, edgecolors='black')
    plt.show()
```

### Analyzing Nearest Neighbors

```python
# Inspect what neighbors are used for prediction
def analyze_prediction(model, x_sample):
    k_indices, k_distances = model._get_neighbors(x_sample)
    k_labels = model.y_train[k_indices]
    
    print(f"K nearest neighbors:")
    for i, (idx, dist, label) in enumerate(zip(k_indices, k_distances, k_labels)):
        print(f"  Neighbor {i+1}: distance={dist:.3f}, label={label}")
```

## 📚 Comparison with Other Algorithms

| Aspect | KNN | Decision Tree | SVM | Neural Network |
|--------|-----|---------------|-----|----------------|
| Training | Instant (lazy) | Fast | Moderate | Slow |
| Prediction | Slow | Fast | Fast | Fast |
| Interpretability | High | High | Low | Very Low |
| Handling Non-linear | Good | Excellent | Good | Excellent |
| Memory Usage | High | Low | Moderate | Moderate |
| Feature Scaling | Required | Not needed | Required | Required |

## 💻 Example Use Case: Customer Segmentation

```python
# SPG wants to classify new customer based on purchase history
# Features: age, income, frequency, recency
# Target: segment (low, medium, high value)

from implementation import KNNClassifier
from sklearn.preprocessing import StandardScaler

# Prepare data
X = customer_data[['age', 'income', 'frequency', 'recency']].values
y = customer_data['segment'].values

# Scale
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Train
knn = KNNClassifier(n_neighbors=7, weights='distance')
knn.fit(X_train_scaled, y_train)

# Predict new customer
new_customer = [[25, 50000, 10, 5]]  # age, income, freq, recency
new_customer_scaled = scaler.transform(new_customer)
segment = knn.predict(new_customer_scaled)
proba = knn.predict_proba(new_customer_scaled)

print(f"Predicted segment: {segment[0]}")
print(f"Confidence: {proba[0]}")
```

## 📚 Learning Resources

- [Scikit-learn KNN Documentation](https://scikit-learn.org/stable/modules/neighbors.html)
- [StatQuest: K-Nearest Neighbors](https://www.youtube.com/watch?v=HVXime0nQeI)
- [KNN Visualization](https://www.youtube.com/watch?v=UqYde-LULfs)

## 🎓 Exercises

See [`exercise.ipynb`](./exercise.ipynb) for hands-on practice:
1. KNN Classification with real data
2. KNN Regression
3. Choosing optimal K
4. Comparing distance metrics
5. Feature scaling impact
6. Handling imbalanced data

---

**Happy Learning K-Nearest Neighbors! 🎯**
