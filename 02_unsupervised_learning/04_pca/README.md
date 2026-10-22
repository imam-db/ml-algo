# Principal Component Analysis (PCA)

**🇬🇧 English | [🇮🇩 Bahasa Indonesia](README.id.md)**

**🏠 [Back to Main](../../README.md)** | **📁 [Unsupervised Learning](../README.md)** | **⬅️ [Previous: K-Means](../01_kmeans/README.md)**

## 🗺️ Quick Access

| File | Description | Quick Run |
|------|-------------|----------|
| [`implementation.py`](./implementation.py) | From-scratch implementation | `python3 implementation.py` |
| [`sklearn_example.py`](./sklearn_example.py) | Scikit-learn examples | `python3 sklearn_example.py` |
| [`exercise.ipynb`](./exercise.ipynb) | Interactive exercises | `jupyter lab exercise.ipynb` |
| [`datasets/`](./datasets/) | SPG retail outlets dataset | - |

## 📖 Theory and Concepts

### What is PCA?

**Principal Component Analysis (PCA)** is an unsupervised learning technique for dimensionality reduction. It transforms data into a new coordinate system where the axes (principal components) are ordered by the amount of variance they explain. This allows you to:

1. **Visualize** high-dimensional data in 2D or 3D
2. **Reduce noise** by keeping only components with high variance
3. **Speed up** machine learning algorithms by reducing features
4. **Understand** which features contribute most to variance

### Mathematical Formulation

Given data matrix **X** with shape (n_samples, n_features):

1. **Center the data:**
   ```
   X_centered = X - mean(X)
   ```

2. **Compute covariance matrix:**
   ```
   Cov = (1/(n-1)) × X_centered^T × X_centered
   ```

3. **Find eigenvalues λ and eigenvectors v:**
   ```
   Cov × v = λ × v
   ```

4. **Principal components** are eigenvectors ordered by eigenvalues (largest first)

### PCA via SVD (Our Implementation)

More numerically stable approach using **Singular Value Decomposition**:

```
X_centered = U × Σ × V^T
```

Where:
- **U**: (n_samples, n_samples) left singular vectors
- **Σ**: (min(n,p), min(n,p)) diagonal matrix of singular values σ
- **V^T**: (n_features, n_features) right singular vectors (transposed)

**Key insight:** The rows of V^T are the principal components!

**Explained variance:**
```
explained_variance_i = σ_i² / (n_samples - 1)
```

**Explained variance ratio:**
```
explained_variance_ratio_i = explained_variance_i / Σ(explained_variance)
```

### Transformation and Reconstruction

**Transform (dimensionality reduction):**
```
X_transformed = X_centered × components^T
```
Shape: (n_samples, n_components)

**Inverse transform (reconstruction):**
```
X_reconstructed = X_transformed × components + mean
```
Shape: (n_samples, n_features)

**Note:** Reconstruction with fewer than all components loses information (some variance).

## 🎯 When to Use PCA?

### ✅ Suitable for:
- Visualizing high-dimensional data (reduce to 2D/3D)
- Removing correlated/redundant features
- Noise reduction (keep high-variance components)
- Speeding up algorithms (fewer features)
- Pre-processing before clustering or classification
- Exploratory data analysis

### ❌ Not suitable for:
- When you need to interpret individual features (PCs are combinations)
- When variance doesn't align with importance
- Sparse data (PCA makes it dense)
- Categorical data (needs numeric features)
- When non-linear relationships dominate

## 📊 Key Concepts

### Explained Variance Ratio

Tells you how much information (variance) each component captures:

```python
explained_variance_ratio = [0.45, 0.25, 0.15, 0.10, 0.05]
cumulative = [0.45, 0.70, 0.85, 0.95, 1.00]
```

- PC1 captures 45% of variance
- First 2 PCs capture 70% of variance
- Need 4 PCs for 95% of variance

**Rule of thumb:** Keep components that explain 90-95% of cumulative variance.

### Sign Ambiguity

PCA components are only defined **up to sign** (direction can be flipped):

```
components = [v1, v2, v3, ...]  or  [-v1, -v2, -v3, ...]
```

Both are mathematically equivalent! When comparing PCA results:
- Components may have opposite signs
- Use `np.allclose(abs(comp1), abs(comp2))` or correlation
- Our implementation includes `compare_pca_sign_invariance()` helper

### Feature Scaling is CRITICAL

PCA is sensitive to feature scales because it maximizes variance:

```python
# Without scaling: features with larger ranges dominate
X = [[1000000, 5],    # Sales (millions), Rating (1-5)
     [2000000, 4]]
# PCA will be dominated by Sales!

# Always scale first:
from sklearn.preprocessing import StandardScaler
X_scaled = StandardScaler().fit_transform(X)
```

## 🔍 Advantages and Disadvantages

### Advantages:
- ✅ Reduces dimensionality while preserving variance
- ✅ Removes correlated features automatically
- ✅ Speeds up downstream algorithms
- ✅ Great for visualization
- ✅ Deterministic (same result each time)
- ✅ Fast computation via SVD

### Disadvantages:
- ❌ Principal components are harder to interpret
- ❌ Assumes linear relationships
- ❌ Sensitive to outliers
- ❌ Requires feature scaling
- ❌ May lose information with too few components
- ❌ Sign ambiguity when comparing results

## 🎯 Practical Scenario: Visualizing Outlet Segments

### 🏪 Indonesian FMCG Retail Outlet Analysis

**Dataset**: Same 500 retail outlets from K-Means example

**Original features (5D):**
- `monthly_sales_idr` - Monthly sales in IDR
- `sku_count` - Number of SKUs
- `visit_freq_per_month` - Visit frequency
- `oos_rate` - Out-of-stock rate
- `avg_basket_idr` - Average basket size

**Problem:** Can't visualize 5D data or understand which features vary together.

**Solution:** Use PCA to:
1. Reduce 5D data to 2D for visualization
2. Understand how much variance 2D captures
3. See which original features contribute to each PC
4. Visualize K-Means clusters in 2D PCA space

**Run the example:**
```bash
python3 sklearn_example.py
```

**What you'll learn:**
- How to interpret explained variance ratio
- When 2D projection is sufficient (check cumulative variance)
- How to visualize clusters in reduced dimensions
- Which original features contribute to each principal component

## 💡 Key Implementation Details

### Important Parameters

```python
PCA(
    n_components=None,    # Number of components (or None for all)
    whiten=False,        # Divide by sqrt(explained_variance)
    svd_solver='auto'    # SVD algorithm: 'auto', 'full', 'arpack', 'randomized'
)
```

### Choosing n_components

**By variance:**
```python
pca = PCA(n_components=0.95)  # Keep 95% of variance
```

**By number:**
```python
pca = PCA(n_components=2)  # Keep exactly 2 components
```

**Inspect after fitting:**
```python
pca = PCA()
pca.fit(X)
print(pca.explained_variance_ratio_.cumsum())
# Choose n where cumulative ≥ 0.90
```

### Feature Scaling

```python
from sklearn.preprocessing import StandardScaler

# Always scale before PCA!
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)
```

### Reconstruction Error

Measure information loss from dimensionality reduction:

```python
# Transform and reconstruct
X_pca = pca.transform(X_scaled)
X_reconstructed = pca.inverse_transform(X_pca)

# Compute MSE
mse = np.mean((X_scaled - X_reconstructed) ** 2)
```

With all components: MSE ≈ 0 (perfect reconstruction)

## 📚 References

- [Scikit-learn PCA](https://scikit-learn.org/stable/modules/decomposition.html#pca)
- [Principal Component Analysis (Wikipedia)](https://en.wikipedia.org/wiki/Principal_component_analysis)
- [Singular Value Decomposition](https://en.wikipedia.org/wiki/Singular_value_decomposition)
- [Understanding PCA Visually](http://setosa.io/ev/principal-component-analysis/)

## 🎓 Learning Progression

### Prerequisites
- Linear algebra basics (vectors, matrices, matrix multiplication)
- Understanding of variance and covariance
- NumPy fundamentals

### After PCA, explore:

1. **[t-SNE](../05_tsne/)** - Non-linear dimensionality reduction for visualization
2. **[Autoencoders](../../04_advanced_topics/02_deep_learning/)** - Non-linear PCA with neural networks
3. **[LDA](../07_lda/)** - Linear Discriminant Analysis (supervised version)
4. **[Factor Analysis](../08_factor_analysis/)** - Similar to PCA but models noise

### Combining with other algorithms

```python
# PCA before clustering
pca = PCA(n_components=0.95)
X_reduced = pca.fit_transform(X_scaled)
kmeans = KMeans(n_clusters=4)
labels = kmeans.fit_predict(X_reduced)

# PCA before classification (faster training)
pca = PCA(n_components=50)
X_train_pca = pca.fit_transform(X_train)
X_test_pca = pca.transform(X_test)
clf.fit(X_train_pca, y_train)
```

---

## Ringkasan (ID)

**Principal Component Analysis (PCA)** adalah teknik unsupervised learning untuk reduksi dimensi. PCA mentransformasi data ke sistem koordinat baru di mana sumbu (principal components) diurutkan berdasarkan variance yang dijelaskan.

**Cara Kerja:**
1. Center data (kurangi mean)
2. Hitung SVD: X_centered = U × Σ × V^T
3. Principal components = baris dari V^T
4. Explained variance = σ²/(n-1)

**Transformasi:**
- **Transform:** X_transformed = X_centered × components^T (reduksi dimensi)
- **Inverse Transform:** X_reconstructed = X_transformed × components + mean

**Explained Variance Ratio:** Persentase informasi yang dipertahankan oleh setiap component. Contoh: [0.45, 0.25, 0.15, 0.10, 0.05] berarti PC1 menjelaskan 45% variance, PC1+PC2 menjelaskan 70%.

**Penting:** 
- SELALU lakukan feature scaling sebelum PCA!
- Sign ambiguity: components bisa bernilai positif atau negatif (keduanya valid)
- Gunakan cumulative variance ≥ 90-95% untuk memilih n_components

**Studi Kasus:** Dataset 500 outlet retail dengan 5 fitur (sales, SKU, visit freq, OOS rate, basket). Gunakan PCA untuk:
1. Reduksi dari 5D ke 2D untuk visualisasi
2. Cek berapa % variance yang dipertahankan 2 PC
3. Visualisasi hasil clustering K-Means di ruang 2D
4. Pahami fitur mana yang berkontribusi ke setiap PC

**Latihan:** Tentukan jumlah components yang cukup untuk capture 90% variance, lalu interpretasikan loading (kontribusi fitur asli) di setiap PC.

---

**➡️ Next Step**: Explore **[t-SNE](../05_tsne/)** for non-linear dimensionality reduction!
