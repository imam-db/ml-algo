# DBSCAN (Density-Based Spatial Clustering of Applications with Noise)

**🇬🇧 English | [🇮🇩 Bahasa Indonesia](README.id.md)**

## What is DBSCAN?

DBSCAN is a clustering algorithm that differs from K-Means. Instead of forming spherical clusters of uniform size, **DBSCAN finds clusters based on density** and can:

1. ✅ Find clusters with **arbitrary shapes** (not necessarily spherical)
2. ✅ **Detect outliers** (noise points)
3. ✅ **No need to specify number of clusters** upfront
4. ❌ Sensitive to `eps` and `min_samples` parameters

### SPG (Sales Promotion Girl/Guy) Analogy

Imagine you're an SPG coordinator mapping store locations in a city:

- **Cluster**: Areas with many nearby stores (malls, residential areas, business districts)
- **Outlier**: Isolated stores on city outskirts or quiet areas
- **Core Point**: Store surrounded by many others (≥ min_samples within radius eps)
- **Border Point**: Store at cluster edge (near core point but doesn't have many neighbors)

DBSCAN helps you identify:
- SPG work zones based on store density
- Which stores are isolated and potentially inefficient to visit

---

## Key Concepts

### 1. Key Parameters

#### `eps` (epsilon)
- **Neighborhood radius**: Maximum distance between two points to be considered neighbors
- In SPG context: Distance (km) considered "close"
- Too small → Many noise points
- Too large → Everything becomes one cluster

#### `min_samples`
- **Minimum neighbors**: How many points within radius `eps` for a point to be considered core
- Includes the point itself
- In SPG context: How many stores minimum within radius `eps` for area to be "dense"

### 2. Point Types

```
Core Point (●): Has ≥ min_samples neighbors within radius eps
├─ Center of cluster
└─ Used to expand cluster

Border Point (○): Not core point, but within eps of core point
├─ At cluster edge
└─ Not used to expand

Noise Point (×): Neither core nor border
├─ Outlier / anomaly
└─ Label = -1
```

### 3. Algorithm

```
1. For each unvisited point:
   
   a. Find neighbors within radius eps
   
   b. If neighbors < min_samples:
      → Mark as noise (temporarily)
   
   c. If neighbors ≥ min_samples:
      → Core point! Create new cluster
      → Expand cluster using BFS:
         - Add all neighbors to cluster
         - If neighbor also core, expand from there
         - Border points that were noise → join cluster

2. Noise points never claimed → remain noise
```

---

## Choosing Parameters

### Method 1: K-Distance Graph

1. For each point, calculate distance to k-th nearest neighbor (k = min_samples)
2. Sort these distances descending
3. Plot the graph
4. Find "elbow" (point where curve starts rising sharply)
5. Distance at elbow → good `eps` value

### Method 2: Domain Knowledge

- **`eps`**: Based on domain knowledge of "close" distance
  - For store locations: 1-2 km may make sense
  - For sensor data: depends on measurement units

- **`min_samples`**: Rule of thumb
  - `min_samples = 2 * dimensions` for most cases
  - Minimum 3-5 for 2D data
  - Larger if lots of noise

---

## Advantages and Disadvantages

### ✅ Advantages

1. **No need to specify number of clusters** upfront (unlike K-Means)
2. **Finds arbitrary-shaped clusters** (not just spherical/convex)
3. **Robust to outliers** - detects and ignores noise
4. **Only 2 parameters** that are relatively intuitive

### ❌ Disadvantages

1. **Sensitive to parameters** - `eps` and `min_samples` must be chosen carefully
2. **Difficult for varying density** - clusters with different densities hard to detect with single `eps`
3. **Curse of dimensionality** - Euclidean distance less meaningful in high dimensions
4. **High complexity** - O(n²) for naive implementation

---

## DBSCAN vs K-Means

| Aspect | DBSCAN | K-Means |
|--------|--------|---------|
| **Number of clusters** | Automatically found | Must be specified (k) |
| **Cluster shape** | Arbitrary | Spherical |
| **Outlier handling** | Detects as noise | All points in cluster |
| **Parameters** | eps, min_samples | k, max_iter |
| **Complexity** | O(n²) or O(n log n) | O(nkdi) |
| **Best for** | Data with noise, complex shapes | Normally distributed, spherical data |

---

## Implementation from Scratch

See `implementation.py` for complete code. Main functions:

```python
from implementation import DBSCAN, generate_clustering_data

# Generate SPG store data
X = generate_clustering_data(n_samples=300, random_state=42)

# Fit DBSCAN
dbscan = DBSCAN(eps=1.5, min_samples=5)
labels = dbscan.fit_predict(X)

print(f"Clusters found: {dbscan.n_clusters_}")
print(f"Noise points: {np.sum(labels == -1)}")
```

---

## Example: SPG Store Segmentation

```python
# Data: coordinates of 300 stores in city
X = generate_clustering_data(n_samples=300, random_state=42)

# DBSCAN with eps=1.5 km, min_samples=5
dbscan = DBSCAN(eps=1.5, min_samples=5)
dbscan.fit(X)

# Analyze results
print(f"Work zones found: {dbscan.n_clusters_}")
print(f"Isolated stores: {np.sum(dbscan.labels_ == -1)}")

for cluster_id in range(dbscan.n_clusters_):
    n_stores = np.sum(dbscan.labels_ == cluster_id)
    print(f"Zone {cluster_id}: {n_stores} stores")
```

**Interpretation**:
- Each cluster = efficient SPG work zone
- Noise points = isolated stores needing special strategy
- Core points = stores at zone centers (high priority)

---

## References

- **Original paper**: Ester, M., Kriegel, H. P., Sander, J., & Xu, X. (1996). "A density-based algorithm for discovering clusters in large spatial databases with noise."
- **Sklearn**: [`sklearn.cluster.DBSCAN`](https://scikit-learn.org/stable/modules/generated/sklearn.cluster.DBSCAN.html)

---

## Related Files

- `implementation.py` - DBSCAN implementation from scratch
- `sklearn_example.py` - Comparison with sklearn
- `exercise.ipynb` - Interactive exercises

**Happy Learning! 🎓**

*[🇮🇩 Indonesian version](README.id.md)*
