# DBSCAN (Density-Based Spatial Clustering of Applications with Noise)

## What is DBSCAN?

DBSCAN is a clustering algorithm that differs from K-Means. Instead of forming round clusters of uniform size, **DBSCAN finds clusters based on density** and can:

1. ✅ Find clusters with **arbitrary shapes** (they don't have to be round)
2. ✅ **Detect outliers** (noise points)
3. ✅ **Doesn't need to determine the number of clusters** in advance
4. ❌ Is sensitive to the `eps` and `min_samples` parameters

### SPG Analogy (Sales Promotion Girl/Guy)

Imagine you are an SPG coordinator mapping store locations across a city:

- **Cluster**: An area with many nearby stores (malls, residential areas, business districts)
- **Outlier**: An isolated store on the edge of the city or in a quiet area
- **Core Point**: A store surrounded by many other stores (≥ min_samples within radius eps)
- **Border Point**: A store on the edge of a cluster (near a core point but without many neighbors)

DBSCAN helps you identify:
- SPG work zones based on store density
- Which stores are isolated and may not be efficient to visit

---

## Key Concepts

### 1. Key Parameters

#### `eps` (epsilon)
- **Neighborhood radius**: The maximum distance between two points for them to be considered neighbors
- In the SPG context: The distance (km) considered "close"
- Too small → Lots of noise
- Too large → All points become a single cluster

#### `min_samples`
- **Minimum number of neighbors**: How many points must lie within radius `eps` for a point to be considered a core point
- Includes the point itself
- In the SPG context: The minimum number of stores within radius `eps` for an area to be considered "dense"

### 2. Point Types

```
Core Point (●): Has ≥ min_samples neighbors within radius eps
├─ Center of a cluster
└─ Used to expand the cluster

Border Point (○): Not a core point, but within radius eps of a core point
├─ On the edge of a cluster
└─ Not used for expansion

Noise Point (×): Neither core nor border
├─ Outlier / anomaly
└─ Label = -1
```

### 3. Algorithm

```
1. For each point not yet visited:
   
   a. Count neighbors within radius eps
   
   b. If number of neighbors < min_samples:
      → Mark as noise (temporarily)
   
   c. If number of neighbors ≥ min_samples:
      → Core point! Create a new cluster
      → Expand the cluster using BFS:
         - Add all neighbors to the cluster
         - If a neighbor is also a core point, expand from there
         - Border points previously marked as noise → join the cluster

2. Noise points never claimed → remain noise
```

---

## Mathematical Formula

### Epsilon-Neighborhood

$$N_\varepsilon(p) = \{q \in D \mid \text{dist}(p, q) \leq \varepsilon\}$$

All points $q$ in dataset $D$ whose distance from $p$ is ≤ $\varepsilon$.

### Core Point Condition

$$|N_\varepsilon(p)| \geq \text{minPts}$$

A point $p$ is a core point if it has at least `minPts` neighbors within radius $\varepsilon$.

### Directly Density-Reachable

A point $q$ is directly density-reachable from $p$ if:
1. $q \in N_\varepsilon(p)$, and
2. $p$ is a core point

### Density-Reachable (Transitive)

$q$ is density-reachable from $p$ if there is a chain: $p = p_1, p_2, \ldots, p_n = q$ where $p_{i+1}$ is directly density-reachable from $p_i$.

### Density-Connected

$p$ and $q$ are density-connected if there is a point $o$ such that both $p$ and $q$ are density-reachable from $o$.

**Cluster** = all points that are density-connected to each other + their border points.

---

## How to Choose Parameters

### Method 1: K-Distance Graph

1. For each point, calculate the distance to the k-th nearest neighbor (k = min_samples)
2. Sort these distances in descending order
3. Plot the graph
4. Find the "elbow" (the point where the curve starts rising sharply)
5. The distance at the elbow → a good `eps` value

```python
from sklearn.neighbors import NearestNeighbors

k = 5  # min_samples that desired
nbrs = NearestNeighbors(n_neighbors=k).fit(X)
distances, indices = nbrs.kneighbors(X)

# Sort k-th nearest neighbor distances
k_dist = np.sort(distances[:, k-1], axis=0)[::-1]

plt.plot(k_dist)
plt.ylabel(f'{k}-th Nearest Neighbor Distance')
plt.xlabel('Points sorted by distance')
plt.title('K-Distance Graph (find the elbow)')
plt.show()
```

### Method 2: Domain Knowledge

- **`eps`**: Based on domain knowledge about what distance counts as "close"
  - For store locations: 1-2 km may make sense
  - For sensor data: depends on the measurement unit

- **`min_samples`**: Rule of thumb
  - `min_samples = 2 * dimensions` for most cases
  - At least 3-5 for 2D data
  - Larger if there is a lot of noise

---

## Advantages and Disadvantages

### ✅ Advantages

1. **Don't need to determine the number of clusters** in advance (unlike K-Means)
2. **Finds clusters of arbitrary shape** (they don't have to be round/convex)
3. **Robust to outliers** - detects and ignores noise
4. **Only needs 2 parameters** that are relatively intuitive

### ❌ Disadvantages

1. **Sensitive to parameters** - `eps` and `min_samples` must be chosen carefully
2. **Difficult for data with varying density** - clusters with different densities can be hard to detect with a single `eps` value
3. **Curse of dimensionality** - Euclidean distance becomes less meaningful in high dimensions
4. **High complexity** - O(n²) for a naive implementation (can be sped up with index structures)

---

## DBSCAN vs K-Means

| Aspect | DBSCAN | K-Means |
|-------|--------|---------|
| **Number of clusters** | Found automatically | Must be specified (k) |
| **Cluster shape** | Arbitrary | Round (spherical) |
| **Outlier handling** | Detected as noise | All points are assigned to a cluster |
| **Parameters** | eps, min_samples | k, max_iter |
| **Complexity** | O(n²) or O(n log n) | O(nkdi) |
| **Good for** | Data with noise, complex shapes | Normally distributed, round data |

---

## From-Scratch Implementation

See `implementation.py` for the full code. Main functions:

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

## Case Study: SPG Store Segmentation

```python
# Data: coordinates of 300 stores in a city
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
- Each cluster = an efficient SPG work zone
- Noise points = isolated stores that may need a special strategy
- Core points = stores at the center of a zone (high priority)

---

## Parameter Tuning

### Experiments with Grid Search

```python
import numpy as np

eps_values = [0.5, 1.0, 1.5, 2.0, 2.5]
min_samples_values = [3, 5, 10, 15]

best_params = None
best_score = -1

for eps in eps_values:
    for min_samples in min_samples_values:
        dbscan = DBSCAN(eps=eps, min_samples=min_samples)
        labels = dbscan.fit_predict(X)
        
        # Evaluate with silhouette score (if any clusters exist)
        if dbscan.n_clusters_ > 0 and np.sum(labels != -1) > 1:
            from sklearn.metrics import silhouette_score
            mask = labels != -1  # Exclude noise
            if len(set(labels[mask])) > 1:
                score = silhouette_score(X[mask], labels[mask])
                print(f"eps={eps}, min_samples={min_samples}: "
                      f"{dbscan.n_clusters_} clusters, score={score:.3f}")
                
                if score > best_score:
                    best_score = score
                    best_params = (eps, min_samples)

print(f"\nBest params: eps={best_params[0]}, min_samples={best_params[1]}")
```

---

## References

- **Original paper**: Ester, M., Kriegel, H. P., Sander, J., & Xu, X. (1996). "A density-based algorithm for discovering clusters in large spatial databases with noise." In KDD (Vol. 96, No. 34, pp. 226-231).
- **Sklearn**: [`sklearn.cluster.DBSCAN`](https://scikit-learn.org/stable/modules/generated/sklearn.cluster.DBSCAN.html)

---

## Related Files

- `implementation.py` - From-scratch DBSCAN implementation
- `sklearn_example.py` - Comparison with sklearn
- `exercise.ipynb` - Interactive exercises
- `test_dbscan.py` - Unit tests

**Happy learning! 🎓**