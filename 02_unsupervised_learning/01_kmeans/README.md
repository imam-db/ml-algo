# K-Means Clustering

**🇬🇧 English | [🇮🇩 Bahasa Indonesia](README.id.md)**

## Overview

K-Means is an unsupervised learning algorithm that partitions data into K non-overlapping clusters. Each data point is assigned to the cluster with the nearest centroid (mean).

### Algorithm Steps

1. Initialize K cluster centroids randomly
2. Assign each point to the nearest centroid
3. Update centroids as the mean of assigned points
4. Repeat steps 2-3 until convergence

### Evaluation Metrics

- **Inertia:** Sum of squared distances from each point to its cluster centroid (lower is better)
- **Silhouette Score:** Measures how well points fit their clusters (-1 to 1, higher is better)

**Important:** Always perform feature scaling before K-Means since it uses Euclidean distance!

## Implementation

```python
from implementation import KMeans

kmeans = KMeans(n_clusters=3, random_state=42)
kmeans.fit(X)
labels = kmeans.labels_
centroids = kmeans.cluster_centers_
```

## Choosing K

- **Elbow method**: Plot inertia vs K, find the elbow point
- **Silhouette score**: Measure cluster quality
- **Domain knowledge**: Business requirements guide K selection

## Case Study

Dataset contains 500 retail outlets (warung, minimarket, grosir) in Denpasar/Bali visited by SPG. Supervisor wants to group outlets to optimize visit schedules based on sales, number of SKUs, visit frequency, OOS rate, and average transaction value.

**Exercise:** Use elbow method and silhouette score to choose optimal K, then interpret clusters into visit strategies (high/medium/low priority).

## Related Files

- `implementation.py` - K-Means from scratch
- `sklearn_example.py` - Comparison with sklearn
- `exercise.ipynb` - Interactive exercises

**Happy Learning! 🎓** | *[🇮🇩 Bahasa Indonesia](README.id.md)*
