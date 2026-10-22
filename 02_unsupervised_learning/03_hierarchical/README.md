# Hierarchical Clustering (Agglomerative)

## What is Hierarchical Clustering?

Hierarchical clustering builds a **cluster hierarchy** in the form of a tree (dendrogram), rather than directly dividing the data into a fixed number of clusters. There are two approaches:

1. **Agglomerative (bottom-up)** ⬆️ - Start with each point as its own cluster, then merge iteratively
2. **Divisive (top-down)** ⬇️ - Start with one large cluster, then split iteratively

This implementation focuses on **agglomerative** clustering, which is more commonly used.

### SPG Analogy (Sales Promotion Girl/Guy)

Imagine grouping SPG outlets in a city using a hierarchical approach:

1. **Bottom level**: Each outlet is its own "cluster"
2. **Gradual merging**: The closest outlets are combined → into a sub-district
3. **Merge again**: Sub-districts are combined → into a district
4. **Finally**: All outlets in one city

**Advantage**: You can **cut the hierarchy tree at any level** to get the desired number of clusters (2 districts? 5 sub-districts? 10 zones?).

---

## Key Concepts

### 1. Agglomerative Algorithm

```
1. Start: each sample = one cluster (n clusters)

2. Repeat until n_clusters dicapai:
   a. Compute distances between all cluster pairs
   b. Merge dua cluster terdekat
   c. Update distance matrix

3. Stop: when number of clusters = n_clusters
```

### 2. Linkage Criteria

**How do you compute the distance between two clusters?**

#### Single Linkage (Nearest Point)

$$d_{\text{single}}(C_i, C_j) = \min_{x \in C_i, y \in C_j} \text{dist}(x, y)$$

- Distance = the **shortest** distance between any pair of points
- **Advantages**: Finds elongated/chain-like clusters
- **Disadvantages**: Sensitive to noise (chaining effect)

```
Cluster A: ●---●
                   ← min distance
Cluster B:     ●---●
```

#### Complete Linkage (Farthest Point)

$$d_{\text{complete}}(C_i, C_j) = \max_{x \in C_i, y \in C_j} \text{dist}(x, y)$$

- Distance = the **farthest** distance between any pair of points
- **Advantages**: Produces compact/spherical clusters
- **Disadvantages**: Sensitive to outliers

```
Cluster A: ●---●
           ↑       ↑
           └───────┘ max distance
Cluster B: ●---●
```

#### Average Linkage

$$d_{\text{average}}(C_i, C_j) = \frac{1}{|C_i| \cdot |C_j|} \sum_{x \in C_i} \sum_{y \in C_j} \text{dist}(x, y)$$

- Distance = the **average** of all distances between points
- **Balanced**: Not too sensitive to noise or outliers
- **Commonly used**: A good trade-off

### 3. Dendrogram

**Dendrogram** = a visualization of the cluster hierarchy.

```
      ┌─────────┴────────┐
      │                  │
   ┌──┴──┐            ┌──┴──┐
   │     │            │     │
 ┌─┴─┐ ┌─┴─┐        ┌─┴─┐ ┌─┴─┐
 1   2 3   4        5   6 7   8
```

- **Y-axis**: Distance/dissimilarity at the time of merging
- **X-axis**: Individual samples
- **Horizontal cut**: Determines the number of clusters

**How to read it**:
- Two branches that merge at a low height = similar
- A high merge = different
- Cut the dendrogram at a certain height = determine n_clusters

---

## Advantages and Disadvantages

### ✅ Advantages

1. **Don't need to determine k upfront** - The dendrogram shows the complete hierarchical structure
2. **Interpretable** - Hierarchical visualization helps in understanding the data structure
3. **Flexible** - Can cut at any level to get a different number of clusters
4. **Deterministic** - No randomness, results are always the same

### ❌ Disadvantages

1. **High complexity** - O(n³) for a naive implementation, O(n² log n) for an optimized one
2. **Not scalable** - Slow for large datasets (> 10,000 samples)
3. **Sensitive to noise** (especially single linkage)
4. **Cannot undo merges** - Once merged, clusters cannot be separated again (greedy)

---

## Linkage Method Comparison

| Linkage | Cluster Distance | Cluster Shape | Noise Sensitivity | Outlier Sensitivity |
|---------|---------------|----------------|----------------|------------------|
| **Single** | MIN of pairs | Elongated/chain | ⚠️ High | ✅ Low |
| **Complete** | MAX of pairs | Compact/spherical | ✅ Low | ⚠️ High |
| **Average** | MEAN of pairs | Balanced | ✅ Medium | ✅ Medium |

**Recommendations**:
- **Single**: Data with arbitrary/elongated cluster shapes
- **Complete**: Data with compact clusters, avoid chaining
- **Average**: Default choice, a balanced trade-off

---

## Hierarchical vs K-Means

| Aspect | Hierarchical | K-Means |
|-------|--------------|---------|
| **K upfront?** | Not needed (can cut later) | Must be specified |
| **Dendrogram** | Yes, a hierarchy visualization | No |
| **Complexity** | O(n² log n) - O(n³) | O(nkdi) |
| **Scalability** | Poor (< 10k samples) | Good (millions) |
| **Deterministic** | Yes | No (random init) |
| **Cluster shape** | Depends on linkage | Spherical |
| **Good for** | Exploratory analysis, small datasets | Production, large datasets |

---

## How to Choose the Number of Clusters

### 1. Visual Inspection of the Dendrogram

Look for the **largest gap** on the Y-axis (merge height).

```python
# Plot dendrogram
from scipy.cluster.hierarchy import dendrogram, linkage

Z = linkage(X, method='average')
plt.figure(figsize=(12, 6))
dendrogram(Z)
plt.axhline(y=threshold, color='r', linestyle='--')
plt.show()
```

Cut at the threshold that maximizes the gap → the optimal number of clusters.

### 2. Silhouette Score

Try various values of n_clusters and choose the one that maximizes the silhouette score.

```python
from sklearn.metrics import silhouette_score

scores = []
for n in range(2, 11):
    hc = AgglomerativeClustering(n_clusters=n)
    labels = hc.fit_predict(X)
    scores.append(silhouette_score(X, labels))

best_n = np.argmax(scores) + 2
```

### 3. Domain Knowledge

Use business knowledge:
- For SPG: How many teams/zones can you manage?
- For customer segmentation: How many categories make sense?

---

## From-Scratch Implementation

See `implementation.py` for the full code. Main functions:

```python
from implementation import AgglomerativeClustering, generate_hierarchical_data

# Generate SPG outlet data
X = generate_hierarchical_data(n_samples=150, random_state=42)

# Fit hierarchical clustering
hc = AgglomerativeClustering(n_clusters=4, linkage='average')
labels = hc.fit_predict(X)

print(f"Clusters: {hc.n_clusters_}")
for i in range(hc.n_clusters_):
    print(f"Cluster {i}: {np.sum(labels == i)} outlets")
```

---

## Case Study: SPG Outlet Segmentation

```python
# Data: koordinat 150 outlet
X = generate_hierarchical_data(n_samples=150, random_state=42)

# Hierarchical clustering with 4 zona
hc = AgglomerativeClustering(n_clusters=4, linkage='average')
hc.fit(X)

# Visualisasi
from implementation import plot_clustering_results, plot_dendrogram

# Clustering result
fig = plot_clustering_results(X, hc.labels_, "SPG Outlet Zones")
plt.show()

# Dendrogram (hirarki merge)
fig = plot_dendrogram(hc, "Outlet Hierarchy")
plt.show()
```

**Interpretation**:
- The dendrogram shows how outlets are grouped gradually
- Each cluster = an SPG work zone
- You can cut at different levels to get a different number of zones (2, 3, 4, ...)

---

## Comparison of Linkage Methods

```python
import matplotlib.pyplot as plt
from implementation import AgglomerativeClustering

linkages = ['single', 'complete', 'average']
fig, axes = plt.subplots(1, 3, figsize=(18, 5))

for ax, linkage in zip(axes, linkages):
    hc = AgglomerativeClustering(n_clusters=4, linkage=linkage)
    labels = hc.fit_predict(X)
    
    for i in range(4):
        mask = labels == i
        ax.scatter(X[mask, 0], X[mask, 1], s=50, alpha=0.7)
    
    ax.set_title(f'{linkage.capitalize()} Linkage', fontweight='bold')
    ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()
```

**Observations**:
- **Single**: Clusters may be elongated/chain-like
- **Complete**: Clusters are more compact
- **Average**: Balanced

---

## Advanced: Cutting the Dendrogram at Various Levels

```python
from scipy.cluster.hierarchy import fcluster, linkage

# Build the linkage matrix
Z = linkage(X, method='average')

# Cut at different thresholds
for threshold in [2.0, 3.0, 4.0]:
    labels = fcluster(Z, threshold, criterion='distance')
    n_clusters = len(np.unique(labels))
    print(f"Threshold {threshold}: {n_clusters} clusters")
```

This provides flexibility: **one model, many choices of the number of clusters**.

---

## When to Use Hierarchical Clustering?

### ✅ Use it when:

- The dataset is small to medium (< 10,000 samples)
- You need to **explore the hierarchical structure** of the data
- You are unsure how many clusters are optimal
- Interpretability matters (dendrograms help with communication)
- The data has a **natural hierarchical structure** (e.g., product → category → sub-category)

### ❌ Avoid it when:

- The dataset is very large (> 100,000 samples) - use K-Means or Mini-Batch K-Means
- You need fast predictions in production - hierarchical clustering is not efficient for inference
- Clusters must be updated online/incrementally

---

## References

- **Sklearn**: [`sklearn.cluster.AgglomerativeClustering`](https://scikit-learn.org/stable/modules/generated/sklearn.cluster.AgglomerativeClustering.html)
- **Scipy**: [`scipy.cluster.hierarchy`](https://docs.scipy.org/doc/scipy/reference/cluster.hierarchy.html)

---

## Related Files

- `implementation.py` - From-scratch hierarchical clustering implementation
- `sklearn_example.py` - Comparison with sklearn
- `exercise.ipynb` - Interactive exercises with dendrograms
- `test_hierarchical.py` - Unit tests

**Happy learning! 🎓**