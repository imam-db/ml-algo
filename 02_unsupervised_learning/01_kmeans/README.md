# K-Means Clustering

**🇬🇧 English | [🇮🇩 Bahasa Indonesia](README.id.md)**

**🏠 [Back to Main](../../README.md)** | **📁 [Unsupervised Learning](../README.md)** | **➡️ [Next: PCA](../04_pca/README.md)**

## 🗺️ Quick Access

| File | Description | Quick Run |
|------|-------------|----------|
| [`implementation.py`](./implementation.py) | From-scratch implementation | `python3 implementation.py` |
| [`sklearn_example.py`](./sklearn_example.py) | Scikit-learn examples | `python3 sklearn_example.py` |
| [`exercise.ipynb`](./exercise.ipynb) | Interactive exercises | `jupyter lab exercise.ipynb` |
| [`datasets/`](./datasets/) | SPG retail outlets dataset | - |

## 📖 Theory and Concepts

### What is K-Means Clustering?

**K-Means** is an unsupervised learning algorithm that partitions data into K distinct, non-overlapping clusters. Each data point belongs to the cluster with the nearest mean (centroid). It's one of the simplest and most widely used clustering algorithms.

### Mathematical Formulation

**Objective:** Minimize the within-cluster sum of squared distances (inertia):

```
J = Σ(i=1 to K) Σ(x ∈ Ci) ||x - μi||²
```

Where:
- `K` = number of clusters
- `Ci` = set of points in cluster i
- `μi` = centroid (mean) of cluster i
- `||x - μi||²` = squared Euclidean distance

### Algorithm Steps

1. **Initialize** K centroids (randomly or using k-means++)
2. **Assignment step:** Assign each point to nearest centroid
   ```
   ci = argmin(k) ||xi - μk||²
   ```
3. **Update step:** Recompute centroids as mean of assigned points
   ```
   μk = (1/|Ck|) Σ(x ∈ Ck) x
   ```
4. **Repeat** steps 2-3 until convergence (centroids don't change or change is below tolerance)

### K-Means++ Initialization

K-means++ improves initial centroid selection:

1. Choose first centroid uniformly at random from data points
2. For each subsequent centroid:
   - Compute distance D(x) from each point x to nearest existing centroid
   - Choose next centroid with probability proportional to D(x)²
3. Proceed with standard k-means algorithm

This initialization makes convergence faster and often produces better results.

### Convergence

K-Means is guaranteed to converge, but:
- May converge to local optimum (not global)
- Solution depends on initial centroids
- **Solution:** Run multiple times with different initializations (`n_init`)

## 🎯 When to Use K-Means?

### ✅ Suitable for:
- Customer segmentation
- Image compression
- Document clustering
- Anomaly detection (outliers far from centroids)
- Pre-processing for other algorithms
- When number of clusters is known or can be estimated

### ❌ Not suitable for:
- Non-spherical cluster shapes
- Clusters with very different sizes or densities
- Many outliers (sensitive to outliers)
- Unknown or variable number of clusters
- Overlapping clusters

## 📊 Model Evaluation

### Elbow Method
Plot inertia vs. number of clusters K. Look for the "elbow" where inertia decrease slows down.

```
Inertia(K) = Σ(i=1 to K) Σ(x ∈ Ci) ||x - μi||²
```

### Silhouette Score
Measures how similar a point is to its own cluster compared to other clusters:

```
s(i) = (b(i) - a(i)) / max{a(i), b(i)}
```

Where:
- `a(i)` = mean distance from point i to other points in same cluster
- `b(i)` = mean distance from point i to nearest other cluster

**Score range:** -1 to 1 (higher is better)
- `1.0`: Well-matched to own cluster, far from neighbors
- `0.0`: On or very close to decision boundary
- `-1.0`: Assigned to wrong cluster

## 🔍 Advantages and Disadvantages

### Advantages:
- ✅ Simple and easy to understand
- ✅ Fast and scalable (O(n·K·i·d) where n=samples, i=iterations, d=dimensions)
- ✅ Works well when clusters are spherical and similar sized
- ✅ Guaranteed to converge

### Disadvantages:
- ❌ Must specify K in advance
- ❌ Sensitive to initial centroids
- ❌ Sensitive to outliers
- ❌ Assumes spherical clusters of similar size
- ❌ May converge to local optimum

## 🎯 Practical Scenario: SPG Visit Planning

### 🏪 Indonesian FMCG Retail Outlet Segmentation

**Dataset**: 500 retail outlets in Denpasar/Bali visited by SPG (Sales Promotion Girls/Guys)

**Features:**
- `monthly_sales_idr` - Monthly sales in Indonesian Rupiah
- `sku_count` - Number of SKUs (Stock Keeping Units) carried
- `visit_freq_per_month` - How often outlet needs visits
- `oos_rate` - Out-of-stock rate (0-1)
- `avg_basket_idr` - Average basket size in IDR

**Business Goal:** A supervisor wants to group outlets into segments to optimize SPG visit routes and prioritization.

**Outlet types:**
- **Warung** - Traditional small shops
- **Minimarket** - Small self-service stores
- **Grosir** - Wholesalers

**Run the example:**
```bash
python3 sklearn_example.py
```

**What you'll learn:**
- Why feature scaling is critical for K-Means
- How to choose K using elbow method and silhouette score
- How to interpret clusters for business decisions
- Converting clustering results into actionable visit plans

### Expected Segments

The dataset contains 4 hidden segments:
1. **Small warungs** - Low sales, few SKUs, high OOS rate
2. **Active minimarkets** - Medium-high sales, good variety
3. **High-volume grosir** - Very high sales and SKU count
4. **Struggling outlets** - Low sales, medium SKUs, high OOS

Your task: Use K-Means to discover these segments and recommend visit strategies!

## 💡 Key Implementation Details

### Important Parameters

```python
KMeans(
    n_clusters=8,        # Number of clusters
    init='k-means++',    # Initialization: 'k-means++' or 'random'
    n_init=10,          # Number of different initializations
    max_iter=300,       # Maximum iterations
    tol=1e-4,           # Tolerance for convergence
    random_state=42     # For reproducibility
)
```

### Feature Scaling is CRITICAL

K-Means uses Euclidean distance, so features with larger ranges dominate:

```python
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)  # Zero mean, unit variance
```

**Example:** Without scaling, `monthly_sales_idr` (millions) would dominate over `oos_rate` (0-1).

### Random State and Determinism

For reproducibility:
```python
kmeans = KMeans(n_clusters=3, random_state=42)
```

In our implementation:
```python
self.rng_ = np.random.default_rng(random_state)
```

## 📚 References

- [Scikit-learn K-Means](https://scikit-learn.org/stable/modules/clustering.html#k-means)
- [k-means++: The Advantages of Careful Seeding](http://ilpubs.stanford.edu:8090/778/1/2006-13.pdf)
- [Silhouette Coefficient](https://en.wikipedia.org/wiki/Silhouette_(clustering))

## 🎓 Learning Progression

### Prerequisites
- Basic NumPy knowledge
- Understanding of Euclidean distance
- Basic statistics (mean, variance)

### After K-Means, explore:

1. **[Hierarchical Clustering](../02_hierarchical_clustering/)** - No need to specify K
2. **[DBSCAN](../03_dbscan/)** - Finds arbitrarily-shaped clusters
3. **[PCA](../04_pca/)** - Visualize high-dimensional clusters in 2D
4. **[Gaussian Mixture Models](../../04_advanced_topics/)** - Soft clustering

---

## Ringkasan (ID)

**K-Means Clustering** is algoritma unsupervised learning that divides data menjadi K groups (cluster) that not overlapping. Each data point is placed in cluster with centroid (average) terdekat.

**How it Works:**
1. Pilih K centroid awal (random or k-means++)
2. Assign each points to centroid terdekat
3. Update centroid as average from points that assigned
4. Repeat until konvergen

**Metrik Evaluasi:**
- **Inertia:** Jumlah jarak kuadrat from each points to centroid cluster-nya (semakin kecil semakin good)
- **Elbow Method:** Plot inertia vs K, cari "siku" di mana penurunan melambat
- **Silhouette Score:** Ukur seberapa good points cocok with cluster-nya (-1 until 1, semakin tinggi semakin good)

**Penting:** Always do feature scaling before K-Means because this algorithm uses jarak Euclidean!

**Studi Kasus:** Dataset berisi 500 outlet retail (warung, minimarket, grosir) di Denpasar/Bali that dikunjungi SPG. Supervisor ingin mengelompokkan outlet for mengoptimalkan jadwal kunjungan based on sales, jumlah SKU, frekuensi kunjungan, OOS rate, and average transaksi.

**Exercises:** Gunakan elbow method and silhouette score for memilih K optimal, lalu interpretasikan cluster menjadi strategi kunjungan (prioritas tinggi/sedang/rendah).

---

**➡️ Next Step**: Continue to **[PCA](../04_pca/)** to visualize clusters in 2D!