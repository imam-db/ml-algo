"""
K-Means Clustering Implementation from Scratch
==============================================

K-Means clustering with k-means++ initialization, multiple random initializations,
and convergence detection.
"""

import numpy as np
from typing import Optional, Tuple

class KMeansFromScratch:
    """
    K-Means clustering implementation from scratch using NumPy.
    
    Parameters:
    -----------
    n_clusters : int, default=8
        The number of clusters to form
    init : str, default='k-means++'
        Initialization method: 'k-means++' or 'random'
    n_init : int, default=10
        Number of times the k-means algorithm is run with different centroid seeds
    max_iter : int, default=300
        Maximum number of iterations for a single run
    tol : float, default=1e-4
        Relative tolerance to declare convergence (change in inertia)
    random_state : int or None, default=None
        Random seed for reproducibility
        
    Attributes:
    -----------
    cluster_centers_ : ndarray of shape (n_clusters, n_features)
        Coordinates of cluster centers
    labels_ : ndarray of shape (n_samples,)
        Labels of each point
    inertia_ : float
        Sum of squared distances of samples to their closest cluster center
    n_iter_ : int
        Number of iterations run in the best trial
    """
    
    def __init__(self, n_clusters: int = 8, init: str = 'k-means++', 
                 n_init: int = 10, max_iter: int = 300, tol: float = 1e-4,
                 random_state: Optional[int] = None):
        self.n_clusters = n_clusters
        self.init = init
        self.n_init = n_init
        self.max_iter = max_iter
        self.tol = tol
        self.random_state = random_state
        
        # Attributes set during fit
        self.cluster_centers_ = None
        self.labels_ = None
        self.inertia_ = None
        self.n_iter_ = None
        
        # Random number generator
        self.rng_ = None
    
    def _init_random_state(self):
        """Initialize random number generator."""
        if self.random_state is not None:
            self.rng_ = np.random.default_rng(self.random_state)
        else:
            self.rng_ = np.random.default_rng()
    
    def _kmeans_plusplus_init(self, X: np.ndarray) -> np.ndarray:
        """
        Initialize cluster centers using k-means++ algorithm.
        
        This method selects initial centers by:
        1. Choose first center uniformly at random
        2. For each subsequent center, choose from remaining points with
           probability proportional to squared distance from nearest existing center
        """
        n_samples, n_features = X.shape
        centers = np.zeros((self.n_clusters, n_features))
        
        # Choose first center uniformly at random
        centers[0] = X[self.rng_.integers(0, n_samples)]
        
        # Choose remaining centers
        for i in range(1, self.n_clusters):
            # Compute squared distances to nearest existing center
            distances = np.min(
                np.sum((X[:, np.newaxis, :] - centers[:i]) ** 2, axis=2),
                axis=1
            )
            
            # Avoid division by zero
            distances = np.maximum(distances, 1e-10)
            
            # Choose next center with probability proportional to squared distance
            probabilities = distances / distances.sum()
            cumulative_prob = probabilities.cumsum()
            r = self.rng_.random()
            
            # Find index where cumulative probability exceeds random value
            idx = np.searchsorted(cumulative_prob, r)
            centers[i] = X[idx]
        
        return centers
    
    def _random_init(self, X: np.ndarray) -> np.ndarray:
        """Initialize cluster centers by randomly selecting from data points."""
        n_samples = X.shape[0]
        indices = self.rng_.choice(n_samples, self.n_clusters, replace=False)
        return X[indices].copy()
    
    def _assign_labels(self, X: np.ndarray, centers: np.ndarray) -> np.ndarray:
        """Assign each sample to the nearest cluster center."""
        # Compute squared Euclidean distances: (n_samples, n_clusters)
        distances = np.sum((X[:, np.newaxis, :] - centers) ** 2, axis=2)
        return np.argmin(distances, axis=1)
    
    def _compute_inertia(self, X: np.ndarray, labels: np.ndarray, 
                         centers: np.ndarray) -> float:
        """Compute sum of squared distances to assigned cluster centers."""
        inertia = 0.0
        for i in range(self.n_clusters):
            cluster_mask = labels == i
            if cluster_mask.any():
                cluster_points = X[cluster_mask]
                inertia += np.sum((cluster_points - centers[i]) ** 2)
        return inertia
    
    def _kmeans_single_run(self, X: np.ndarray, centers_init: np.ndarray) -> Tuple:
        """
        Run k-means algorithm once from given initial centers.
        
        Returns:
        --------
        centers : ndarray
            Final cluster centers
        labels : ndarray
            Final cluster assignments
        inertia : float
            Final inertia
        n_iter : int
            Number of iterations performed
        """
        centers = centers_init.copy()
        prev_inertia = np.inf
        
        for iteration in range(self.max_iter):
            # Assignment step: assign each point to nearest center
            labels = self._assign_labels(X, centers)
            
            # Update step: recompute centers as mean of assigned points
            new_centers = np.zeros_like(centers)
            for i in range(self.n_clusters):
                cluster_mask = labels == i
                if cluster_mask.any():
                    new_centers[i] = X[cluster_mask].mean(axis=0)
                else:
                    # If cluster is empty, reinitialize it randomly
                    new_centers[i] = X[self.rng_.integers(0, X.shape[0])]
            
            centers = new_centers
            
            # Check convergence
            inertia = self._compute_inertia(X, labels, centers)
            
            if prev_inertia - inertia < self.tol:
                break
            
            prev_inertia = inertia
        
        return centers, labels, inertia, iteration + 1
    
    def fit(self, X: np.ndarray) -> 'KMeansFromScratch':
        """
        Compute k-means clustering.
        
        Parameters:
        -----------
        X : ndarray of shape (n_samples, n_features)
            Training instances to cluster
            
        Returns:
        --------
        self : object
            Fitted estimator
        """
        X = np.asarray(X, dtype=np.float64)
        if X.ndim == 1:
            X = X.reshape(-1, 1)
        
        n_samples, n_features = X.shape
        
        if self.n_clusters > n_samples:
            raise ValueError(
                f"n_clusters={self.n_clusters} must be <= n_samples={n_samples}"
            )
        
        self._init_random_state()
        
        # Run k-means multiple times and keep best result
        best_inertia = np.inf
        best_centers = None
        best_labels = None
        best_n_iter = 0
        
        for trial in range(self.n_init):
            # Initialize centers
            if self.init == 'k-means++':
                centers_init = self._kmeans_plusplus_init(X)
            elif self.init == 'random':
                centers_init = self._random_init(X)
            else:
                raise ValueError(
                    f"init should be 'k-means++' or 'random', got '{self.init}'"
                )
            
            # Run k-means
            centers, labels, inertia, n_iter = self._kmeans_single_run(X, centers_init)
            
            # Keep best result
            if inertia < best_inertia:
                best_inertia = inertia
                best_centers = centers
                best_labels = labels
                best_n_iter = n_iter
        
        self.cluster_centers_ = best_centers
        self.labels_ = best_labels
        self.inertia_ = best_inertia
        self.n_iter_ = best_n_iter
        
        return self
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict the closest cluster each sample in X belongs to.
        
        Parameters:
        -----------
        X : ndarray of shape (n_samples, n_features)
            New data to predict
            
        Returns:
        --------
        labels : ndarray of shape (n_samples,)
            Index of the cluster each sample belongs to
        """
        if self.cluster_centers_ is None:
            raise ValueError("Model is not fitted. Call fit() first.")
        
        X = np.asarray(X, dtype=np.float64)
        if X.ndim == 1:
            X = X.reshape(-1, 1)
        
        return self._assign_labels(X, self.cluster_centers_)
    
    def fit_predict(self, X: np.ndarray) -> np.ndarray:
        """Compute cluster centers and predict cluster index for each sample."""
        return self.fit(X).labels_


def compute_elbow_scores(X: np.ndarray, k_range: range, 
                         random_state: Optional[int] = None) -> np.ndarray:
    """
    Compute inertia values for different numbers of clusters (elbow method).
    
    Parameters:
    -----------
    X : ndarray
        Data to cluster
    k_range : range
        Range of k values to try
    random_state : int or None
        Random seed
        
    Returns:
    --------
    inertias : ndarray
        Inertia value for each k
    """
    inertias = []
    for k in k_range:
        kmeans = KMeansFromScratch(n_clusters=k, random_state=random_state)
        kmeans.fit(X)
        inertias.append(kmeans.inertia_)
    return np.array(inertias)


def silhouette_score(X: np.ndarray, labels: np.ndarray) -> float:
    """
    Compute the mean Silhouette Coefficient of all samples.
    
    The Silhouette Coefficient is calculated using the mean intra-cluster 
    distance (a) and the mean nearest-cluster distance (b) for each sample.
    The score for a sample is (b - a) / max(a, b).
    
    Parameters:
    -----------
    X : ndarray of shape (n_samples, n_features)
        Feature array
    labels : ndarray of shape (n_samples,)
        Cluster labels for each point
        
    Returns:
    --------
    score : float
        Mean Silhouette Coefficient for all samples (-1 to 1, higher is better)
    """
    X = np.asarray(X, dtype=np.float64)
    labels = np.asarray(labels)
    
    n_samples = X.shape[0]
    unique_labels = np.unique(labels)
    n_clusters = len(unique_labels)
    
    if n_clusters == 1 or n_clusters >= n_samples:
        return 0.0
    
    # Compute pairwise distances
    distances = np.sqrt(np.sum((X[:, np.newaxis, :] - X) ** 2, axis=2))
    
    silhouette_vals = np.zeros(n_samples)
    
    for i in range(n_samples):
        own_cluster = labels[i]
        
        # a(i): mean distance to other points in same cluster
        same_cluster_mask = labels == own_cluster
        same_cluster_distances = distances[i, same_cluster_mask]
        
        if same_cluster_mask.sum() > 1:
            a_i = same_cluster_distances.sum() / (same_cluster_mask.sum() - 1)
        else:
            a_i = 0.0
        
        # b(i): mean distance to points in nearest other cluster
        b_i = np.inf
        for cluster in unique_labels:
            if cluster == own_cluster:
                continue
            
            other_cluster_mask = labels == cluster
            if other_cluster_mask.any():
                mean_dist = distances[i, other_cluster_mask].mean()
                b_i = min(b_i, mean_dist)
        
        # Silhouette coefficient for this sample
        if max(a_i, b_i) > 0:
            silhouette_vals[i] = (b_i - a_i) / max(a_i, b_i)
        else:
            silhouette_vals[i] = 0.0
    
    return silhouette_vals.mean()


if __name__ == "__main__":
    # Simple demonstration
    print("K-Means from Scratch - Demo")
    print("=" * 40)
    
    # Generate sample data (3 clusters)
    rng = np.random.default_rng(42)
    X1 = rng.normal(0, 1, (100, 2))
    X2 = rng.normal(5, 1, (100, 2))
    X3 = rng.normal(2.5, 0.8, (100, 2))
    X = np.vstack([X1, X2, X3])
    
    print(f"Generated data: {X.shape[0]} samples, {X.shape[1]} features")
    
    # Fit K-Means
    kmeans = KMeansFromScratch(n_clusters=3, random_state=42)
    kmeans.fit(X)
    
    print(f"\nK-Means Results:")
    print(f"  Inertia: {kmeans.inertia_:.2f}")
    print(f"  Iterations: {kmeans.n_iter_}")
    print(f"  Cluster centers shape: {kmeans.cluster_centers_.shape}")
    
    # Compute silhouette score
    sil_score = silhouette_score(X, kmeans.labels_)
    print(f"  Silhouette score: {sil_score:.3f}")
    
    # Elbow method
    print(f"\nElbow method (k=2 to 8):")
    inertias = compute_elbow_scores(X, range(2, 9), random_state=42)
    for k, inertia in enumerate(inertias, start=2):
        print(f"  k={k}: inertia={inertia:.2f}")
