# K-Nearest Neighbors (KNN)

**🏠 [Kembali ke Menu Utama](../../README.md)** | **📁 [Supervised Learning](../README.md)**

## 🗺️ Akses Cepat

| File | Deskripsi | Jalankan |
|------|-----------|----------|
| [`implementation.py`](./implementation.py) | Implementasi dari nol | `python3 implementation.py` |
| [`sklearn_example.py`](./sklearn_example.py) | Contoh scikit-learn | `python3 sklearn_example.py` |
| [`exercise.ipynb`](./exercise.ipynb) | Latihan interaktif | `jupyter lab exercise.ipynb` |

## 📖 Teori dan Konsep

### Apa itu K-Nearest Neighbors?

**K-Nearest Neighbors (KNN)** adalah algoritma supervised learning yang sederhana namun powerful. KNN melakukan prediksi berdasarkan K sampel terdekat dalam training data. Algoritma ini bersifat **instance-based** atau **lazy learning** karena tidak membangun model eksplisit saat training.

**Analogi SPG (Sales Promoter Girl):**
Bayangkan seorang SPG baru yang ingin tahu produk mana yang cocok untuk customer. Dia akan melihat K customer terdekat (yang paling mirip) dan melihat produk apa yang mereka beli, kemudian merekomendasikan berdasarkan mayoritas.

### Cara Kerja KNN

**Classification:**
```
1. Hitung jarak dari sampel test ke semua training samples
2. Pilih K sampel terdekat
3. Voting: pilih class yang paling sering muncul di K neighbors
4. Return class tersebut
```

**Regression:**
```
1. Hitung jarak dari sampel test ke semua training samples
2. Pilih K sampel terdekat
3. Rata-rata: hitung mean dari nilai K neighbors
4. Return nilai rata-rata tersebut
```

### Distance Metrics

#### 1. Euclidean Distance (L2)
```
d(x, y) = √(Σ(xi - yi)²)
```
- Paling umum digunakan
- Sensitif terhadap outliers
- Baik untuk data continuous

#### 2. Manhattan Distance (L1)
```
d(x, y) = Σ|xi - yi|
```
- Lebih robust terhadap outliers
- Cocok untuk data grid-like
- Lebih cepat dihitung

#### 3. Minkowski Distance (generalisasi)
```
d(x, y) = (Σ|xi - yi|^p)^(1/p)
```
- p=1: Manhattan
- p=2: Euclidean
- p→∞: Chebyshev

### Weighting Schemes

**Uniform Weights:**
- Semua neighbors punya bobot sama
- Simple majority vote / average
- Cocok jika neighbors relatif merata

**Distance Weights:**
- Neighbors lebih dekat punya bobot lebih besar
- Weight = 1 / distance
- Lebih akurat jika ada variasi jarak
- Mencegah tie votes

### Choosing K

**K Kecil (k=1, 3, 5):**
- ✅ Sensitif terhadap pola lokal
- ✅ Dapat menangkap decision boundaries kompleks
- ❌ Sensitif terhadap noise
- ❌ Overfitting risk tinggi

**K Besar (k=20, 50, 100):**
- ✅ Lebih stabil, less sensitive to noise
- ✅ Smoother decision boundaries
- ❌ Bisa terlalu smooth (underfitting)
- ❌ Komputasi lebih lambat

**K Optimal:**
- Gunakan cross-validation
- Plot accuracy vs k (elbow method)
- Biasanya k = √n (rule of thumb)
- K ganjil untuk classification (avoid ties)

## 🎯 Kapan Menggunakan KNN?

### ✅ Cocok untuk:
- **Dataset kecil hingga menengah**
  - Training O(1), prediction O(n)
- **Decision boundaries non-linear**
  - Tidak ada asumsi tentang data distribution
- **Multi-class classification**
  - Natural extension, tidak perlu one-vs-rest
- **Prototype untuk baseline**
  - Mudah diimplementasikan
- **Anomaly detection**
  - K-distance untuk detect outliers

### ❌ Kurang cocok untuk:
- **Dataset sangat besar**
  - Prediction lambat (harus scan semua data)
- **High-dimensional data (curse of dimensionality)**
  - Distance menjadi meaningless
  - Perlu dimensionality reduction
- **Imbalanced classes**
  - Majority class akan dominate
- **Features dengan skala berbeda**
  - Feature scaling WAJIB
- **Real-time prediction**
  - Terlalu lambat untuk production

## 💡 Implementasi

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

KNN bergantung pada distance, sehingga feature scaling sangat penting:

```python
from sklearn.preprocessing import StandardScaler, MinMaxScaler

# StandardScaler (mean=0, std=1) - recommended
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_train)

# MinMaxScaler (0-1 range)
scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(X_train)
```

**Tanpa scaling:**
- Feature dengan range besar akan dominate
- Contoh: salary (0-100000) vs age (0-100)
- Distance akan dikuasai oleh salary

### 2. Handling High Dimensions

**Curse of Dimensionality:**
- Semakin banyak dimensions, semakin sparse data
- Distance menjadi kurang meaningful
- Semua points terlihat "jauh"

**Solutions:**
```python
# PCA untuk dimensionality reduction
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

# Option 2: SMOTE untuk resampling
from imblearn.over_sampling import SMOTE
smote = SMOTE()
X_resampled, y_resampled = smote.fit_resample(X, y)

# Option 3: Larger K
knn = KNNClassifier(n_neighbors=15)  # Reduces majority bias
```

### 4. Efficient Implementation

```python
# Use KD-Tree or Ball-Tree untuk large datasets
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

## 🔍 Interpretasi dan Debugging

### Visualisasi Decision Boundaries

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

## 📚 Perbandingan dengan Algoritma Lain

| Aspek | KNN | Decision Tree | SVM | Neural Network |
|-------|-----|---------------|-----|----------------|
| Training | Instant (lazy) | Fast | Moderate | Slow |
| Prediction | Slow | Fast | Fast | Fast |
| Interpretability | High | High | Low | Very Low |
| Handling Non-linear | Good | Excellent | Good | Excellent |
| Memory Usage | High | Low | Moderate | Moderate |
| Feature Scaling | Required | Not needed | Required | Required |

## 💻 Contoh Use Case: Customer Segmentation

```python
# SPG ingin classify customer baru berdasarkan purchase history
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

## 📚 Sumber Belajar

- [Scikit-learn KNN Documentation](https://scikit-learn.org/stable/modules/neighbors.html)
- [StatQuest: K-Nearest Neighbors](https://www.youtube.com/watch?v=HVXime0nQeI)
- [KNN Visualization](https://www.youtube.com/watch?v=UqYde-LULfs)

## 🎓 Latihan

Lihat [`exercise.ipynb`](./exercise.ipynb) untuk latihan hands-on:
1. KNN Classification dengan real data
2. KNN Regression
3. Choosing optimal K
4. Comparing distance metrics
5. Feature scaling impact
6. Handling imbalanced data

---

**Selamat Belajar K-Nearest Neighbors! 🎯**
