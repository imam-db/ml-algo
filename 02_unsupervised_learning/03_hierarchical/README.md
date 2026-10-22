# Hierarchical Clustering (Agglomerative)

**🇬🇧 English | [🇮🇩 Bahasa Indonesia](README.id.md)**

## What is Hierarchical Clustering?

Hierarchical clustering builds a **hierarchy of clusters** in tree form (dendrogram), rather than directly dividing data into a fixed number of clusters. There are two approaches:

1. **Agglomerative (bottom-up)** ⬆️ - Start with each point as cluster, merge iteratively
2. **Divisive (top-down)** ⬇️ - Start with one big cluster, split iteratively

This implementation focuses on **agglomerative**, which is more commonly used.

### SPG Analogy

Imagine grouping SPG outlets in a city with a hierarchical approach:

1. **Lowest level**: Each outlet is its own "cluster"
2. **Gradual merging**: Nearest outlets merged → become sub-districts
3. **Merge again**: Sub-districts merged → become districts
4. **Until**: All outlets in one city

**Advantage**: You can **cut the hierarchy tree at any level** to get desired number of clusters (2 districts? 5 sub-districts? 10 zones?).

---

## Key Concepts

### 1. Agglomerative Algorithm

```
1. Start: Each sample = one cluster (n clusters)

2. Repeat until n_clusters reached:
   a. Calculate distance between all cluster pairs
   b. Merge two closest clusters
   c. Update distance matrix

3. Stop: When number of clusters = n_clusters
```

### 2. Linkage Criteria

**How to calculate distance between two clusters?**

#### Single Linkage (Nearest Point)

- Distance = **shortest** distance between all point pairs
- **Advantage**: Finds elongated/chain-like clusters
- **Disadvantage**: Sensitive to noise (chaining effect)

#### Complete Linkage (Farthest Point)

- Distance = **longest** distance between all point pairs
- **Advantage**: Produces compact/spherical clusters
- **Disadvantage**: Sensitive to outliers

#### Average Linkage

- Distance = **average** of all inter-point distances
- **Balanced**: Not too sensitive to noise or outliers
- **Commonly used**: Good trade-off

### 3. Dendrogram

**Dendrogram** = visualization of hierarchy tree.

- **Y-axis**: Distance/dissimilarity at merge
- **X-axis**: Individual samples
- **Horizontal cut**: Determines number of clusters

**How to read**:
- Two branches merge low = similar
- Merge high = different
- Cut dendrogram at certain height = determine n_clusters

---

## Advantages and Disadvantages

### ✅ Advantages

1. **No need to specify k upfront** - Dendrogram shows complete hierarchical structure
2. **Interpretable** - Hierarchy visualization aids understanding
3. **Flexible** - Can cut at any level for different cluster counts
4. **Deterministic** - No randomness, consistent results

### ❌ Disadvantages

1. **High complexity** - O(n³) for naive implementation, O(n² log n) for optimized
2. **Not scalable** - Slow for large datasets (> 10,000 samples)
3. **Sensitive to noise** (especially single linkage)
4. **Cannot undo merges** - Once merged, cannot separate (greedy)

---

## Linkage Method Comparison

| Linkage | Cluster Distance | Cluster Shape | Noise Sensitive | Outlier Sensitive |
|---------|-----------------|---------------|-----------------|-------------------|
| **Single** | MIN of pairs | Elongated/chain | ⚠️ High | ✅ Low |
| **Complete** | MAX of pairs | Compact/spherical | ✅ Low | ⚠️ High |
| **Average** | MEAN of pairs | Balanced | ✅ Medium | ✅ Medium |

**Recommendation**:
- **Single**: Data with arbitrary/elongated cluster shapes
- **Complete**: Data with compact clusters, avoid chaining
- **Average**: Default choice, balanced trade-off

---

## Hierarchical vs K-Means

| Aspect | Hierarchical | K-Means |
|--------|--------------|---------|
| **K upfront?** | Not needed (cut later) | Must specify |
| **Dendrogram** | Yes, hierarchy viz | No |
| **Complexity** | O(n² log n) - O(n³) | O(nkdi) |
| **Scalability** | Poor (< 10k samples) | Good (millions) |
| **Deterministic** | Yes | No (random init) |
| **Cluster shape** | Depends on linkage | Spherical |
| **Best for** | Exploratory analysis, small datasets | Production, large datasets |

---

## Choosing Number of Clusters

### 1. Visual Inspection from Dendrogram

Find **largest gap** on Y-axis (merge height).

Cut at threshold that maximizes gap → optimal cluster count.

### 2. Silhouette Score

Try various n_clusters, choose one that maximizes silhouette score.

### 3. Domain Knowledge

Use business knowledge:
- For SPG: How many teams/zones can you manage?
- For customer segmentation: How many categories make sense?

---

## Implementation from Scratch

See `implementation.py` for complete code. Main functions:

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

## Example: SPG Outlet Segmentation

```python
# Data: coordinates of 150 outlets
X = generate_hierarchical_data(n_samples=150, random_state=42)

# Hierarchical clustering with 4 zones
hc = AgglomerativeClustering(n_clusters=4, linkage='average')
hc.fit(X)

# Visualize
from implementation import plot_clustering_results, plot_dendrogram

# Clustering result
fig = plot_clustering_results(X, hc.labels_, "SPG Outlet Zones")
plt.show()

# Dendrogram (hierarchy of merges)
fig = plot_dendrogram(hc, "Outlet Hierarchy")
plt.show()
```

**Interpretation**:
- Dendrogram shows how outlets grouped gradually
- Each cluster = SPG work zone
- You can cut at different levels for different zone counts (2, 3, 4, ...)

---

## When to Use Hierarchical Clustering?

### ✅ Use when:

- Small to medium dataset (< 10,000 samples)
- Need to **explore hierarchical structure** of data
- Uncertain about optimal cluster count
- Interpretability important (dendrogram helps communication)
- Data has **natural hierarchical structure** (e.g., products → categories → sub-categories)

### ❌ Avoid when:

- Very large dataset (> 100,000 samples) - use K-Means or Mini-Batch K-Means
- Need fast predictions in production - hierarchical clustering inefficient for inference
- Clusters need online/incremental updates

---

## References

- **Sklearn**: [`sklearn.cluster.AgglomerativeClustering`](https://scikit-learn.org/stable/modules/generated/sklearn.cluster.AgglomerativeClustering.html)
- **Scipy**: [`scipy.cluster.hierarchy`](https://docs.scipy.org/doc/scipy/reference/cluster.hierarchy.html)

---

## Related Files

- `implementation.py` - Hierarchical clustering implementation from scratch
- `sklearn_example.py` - Comparison with sklearn
- `exercise.ipynb` - Interactive exercises with dendrograms

**Happy Learning! 🎓**

*[🇮🇩 Indonesian version](README.id.md)*
