"""
Hierarchical (Agglomerative) Clustering
Implementasi dari nol untuk tujuan pembelajaran.

Hierarchical clustering membangun hirarki cluster dari bawah ke atas (agglomerative)
atau dari atas ke bawah (divisive). Implementasi ini fokus pada agglomerative.
"""

import numpy as np
from typing import List, Tuple, Optional, Dict


class AgglomerativeClustering:
    """
    Agglomerative (bottom-up) Hierarchical Clustering dari nol.
    
    Mulai dengan setiap sampel sebagai cluster sendiri, lalu iteratif
    merge dua cluster terdekat hingga mencapai jumlah cluster yang diinginkan.
    
    Parameters:
    -----------
    n_clusters : int, default=2
        Jumlah cluster yang diinginkan.
    
    linkage : str, default='single'
        Linkage criterion untuk menghitung jarak antar cluster:
        - 'single': minimum distance (nearest point)
        - 'complete': maximum distance (farthest point)
        - 'average': average distance antara semua pasangan
    
    Attributes:
    -----------
    labels_ : ndarray of shape (n_samples,)
        Label cluster untuk setiap sampel.
    
    n_clusters_ : int
        Jumlah cluster yang ditemukan.
    
    dendrogram_data_ : List[Dict]
        Data untuk membuat dendrogram (merge history).
    
    Example:
    --------
    >>> X = np.array([[1, 2], [2, 2], [2, 3], [8, 7], [8, 8], [25, 80]])
    >>> hc = AgglomerativeClustering(n_clusters=2, linkage='average')
    >>> hc.fit(X)
    >>> print(hc.labels_)
    [0 0 0 1 1 1]
    """
    
    def __init__(self, n_clusters: int = 2, linkage: str = 'single'):
        if linkage not in ['single', 'complete', 'average']:
            raise ValueError("linkage must be 'single', 'complete', or 'average'")
        
        self.n_clusters = n_clusters
        self.linkage = linkage
        
        # Akan di-set setelah fit
        self.labels_ = None
        self.n_clusters_ = None
        self.dendrogram_data_ = []
        self._X = None
    
    def fit(self, X: np.ndarray) -> 'AgglomerativeClustering':
        """
        Fit hierarchical clustering pada data X.
        
        Parameters:
        -----------
        X : ndarray of shape (n_samples, n_features)
            Data training.
        
        Returns:
        --------
        self : object
            Returns self.
        """
        X = np.asarray(X)
        n_samples = X.shape[0]
        
        if self.n_clusters > n_samples:
            raise ValueError(f"n_clusters ({self.n_clusters}) cannot be larger than n_samples ({n_samples})")
        
        self._X = X.copy()
        
        # Initialize: setiap sampel adalah cluster sendiri
        # Gunakan dict untuk menyimpan cluster dan anggotanya
        clusters = {i: [i] for i in range(n_samples)}
        
        # Compute initial distance matrix
        dist_matrix = self._compute_distance_matrix(X)
        
        # Merge history untuk dendrogram
        self.dendrogram_data_ = []
        merge_count = 0
        
        # Iteratively merge closest clusters
        while len(clusters) > self.n_clusters:
            # Find closest pair of clusters
            min_dist = np.inf
            merge_i, merge_j = None, None
            
            cluster_ids = list(clusters.keys())
            for i in range(len(cluster_ids)):
                for j in range(i + 1, len(cluster_ids)):
                    cluster_i = cluster_ids[i]
                    cluster_j = cluster_ids[j]
                    
                    # Compute distance between clusters
                    dist = self._cluster_distance(
                        clusters[cluster_i],
                        clusters[cluster_j],
                        dist_matrix
                    )
                    
                    if dist < min_dist:
                        min_dist = dist
                        merge_i, merge_j = cluster_i, cluster_j
            
            # Merge the two closest clusters
            # Create new cluster dengan ID unik
            new_cluster_id = n_samples + merge_count
            clusters[new_cluster_id] = clusters[merge_i] + clusters[merge_j]
            
            # Record merge untuk dendrogram
            self.dendrogram_data_.append({
                'cluster1': merge_i,
                'cluster2': merge_j,
                'distance': min_dist,
                'size': len(clusters[new_cluster_id])
            })
            
            # Remove merged clusters
            del clusters[merge_i]
            del clusters[merge_j]
            
            merge_count += 1
        
        # Assign final labels
        self.labels_ = np.zeros(n_samples, dtype=int)
        for cluster_label, (cluster_id, members) in enumerate(clusters.items()):
            for sample_idx in members:
                self.labels_[sample_idx] = cluster_label
        
        self.n_clusters_ = len(clusters)
        
        return self
    
    def _compute_distance_matrix(self, X: np.ndarray) -> np.ndarray:
        """Compute pairwise Euclidean distance matrix."""
        n_samples = X.shape[0]
        dist_matrix = np.zeros((n_samples, n_samples))
        
        for i in range(n_samples):
            for j in range(i + 1, n_samples):
                dist = np.linalg.norm(X[i] - X[j])
                dist_matrix[i, j] = dist
                dist_matrix[j, i] = dist
        
        return dist_matrix
    
    def _cluster_distance(self, cluster_i: List[int], cluster_j: List[int],
                         dist_matrix: np.ndarray) -> float:
        """
        Compute distance between two clusters based on linkage criterion.
        """
        distances = []
        for i in cluster_i:
            for j in cluster_j:
                distances.append(dist_matrix[i, j])
        
        if self.linkage == 'single':
            # Single linkage: minimum distance
            return min(distances)
        elif self.linkage == 'complete':
            # Complete linkage: maximum distance
            return max(distances)
        elif self.linkage == 'average':
            # Average linkage: mean distance
            return np.mean(distances)
    
    def fit_predict(self, X: np.ndarray) -> np.ndarray:
        """
        Fit dan return cluster labels.
        
        Parameters:
        -----------
        X : ndarray of shape (n_samples, n_features)
            Data untuk di-cluster.
        
        Returns:
        --------
        labels : ndarray of shape (n_samples,)
            Cluster labels.
        """
        self.fit(X)
        return self.labels_


def generate_hierarchical_data(n_samples: int = 150, random_state: Optional[int] = None) -> np.ndarray:
    """
    Generate synthetic data untuk hierarchical clustering.
    
    Tema SPG: Lokasi outlet dalam region dengan struktur hirarki
    (pusat kota → sub-districts → toko individual).
    
    Parameters:
    -----------
    n_samples : int
        Total jumlah sampel.
    
    random_state : int, optional
        Random seed untuk reproducibility.
    
    Returns:
    --------
    X : ndarray of shape (n_samples, 2)
        Data koordinat outlet.
    """
    if random_state is not None:
        np.random.seed(random_state)
    
    # Level 1: Region A (Downtown)
    n_region_a = n_samples // 2
    
    # Sub-region A1
    n_a1 = n_region_a // 2
    cluster_a1 = np.random.randn(n_a1, 2) * 0.8 + np.array([2, 2])
    
    # Sub-region A2
    n_a2 = n_region_a - n_a1
    cluster_a2 = np.random.randn(n_a2, 2) * 0.6 + np.array([5, 3])
    
    # Level 2: Region B (Suburbs)
    n_region_b = n_samples - n_region_a
    
    # Sub-region B1
    n_b1 = n_region_b // 2
    cluster_b1 = np.random.randn(n_b1, 2) * 0.7 + np.array([2, 8])
    
    # Sub-region B2
    n_b2 = n_region_b - n_b1
    cluster_b2 = np.random.randn(n_b2, 2) * 0.5 + np.array([8, 8])
    
    X = np.vstack([cluster_a1, cluster_a2, cluster_b1, cluster_b2])
    
    return X


def plot_dendrogram(hc: AgglomerativeClustering, title: str = "Dendrogram"):
    """
    Plot dendrogram dari hierarchical clustering.
    
    Parameters:
    -----------
    hc : AgglomerativeClustering
        Fitted hierarchical clustering model.
    
    title : str
        Judul plot.
    
    Returns:
    --------
    fig : matplotlib.figure.Figure
    """
    try:
        from scipy.cluster.hierarchy import dendrogram as scipy_dendrogram
        import matplotlib.pyplot as plt
    except ImportError:
        raise ImportError("scipy and matplotlib required for dendrogram plotting")
    
    if not hc.dendrogram_data_:
        raise ValueError("Model must be fitted before plotting dendrogram")
    
    # Convert merge history to scipy linkage format
    # linkage matrix: [cluster1, cluster2, distance, size]
    n_samples = len(hc._X)
    linkage_matrix = np.zeros((len(hc.dendrogram_data_), 4))
    
    for i, merge in enumerate(hc.dendrogram_data_):
        linkage_matrix[i, 0] = merge['cluster1']
        linkage_matrix[i, 1] = merge['cluster2']
        linkage_matrix[i, 2] = merge['distance']
        linkage_matrix[i, 3] = merge['size']
    
    # Plot
    fig, ax = plt.subplots(figsize=(12, 6))
    scipy_dendrogram(linkage_matrix, ax=ax)
    ax.set_xlabel('Sample Index', fontsize=12)
    ax.set_ylabel('Distance', fontsize=12)
    ax.set_title(f'{title}\n(Linkage: {hc.linkage})', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    return fig


def plot_clustering_results(X: np.ndarray, labels: np.ndarray, 
                           title: str = "Hierarchical Clustering"):
    """
    Visualisasi hasil hierarchical clustering.
    
    Parameters:
    -----------
    X : ndarray of shape (n_samples, 2)
        Data (harus 2D untuk plotting).
    
    labels : ndarray of shape (n_samples,)
        Cluster labels.
    
    title : str
        Judul plot.
    
    Returns:
    --------
    fig : matplotlib.figure.Figure
    """
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        raise ImportError("Matplotlib required for plotting")
    
    fig, ax = plt.subplots(figsize=(10, 7))
    
    unique_labels = np.unique(labels)
    n_clusters = len(unique_labels)
    colors = plt.cm.Spectral(np.linspace(0, 1, n_clusters))
    
    for label, color in zip(unique_labels, colors):
        mask = labels == label
        ax.scatter(X[mask, 0], X[mask, 1], c=[color], 
                  s=100, alpha=0.7, edgecolors='k', 
                  linewidth=0.5, label=f'Cluster {label}')
    
    ax.set_xlabel('X Coordinate (km)', fontsize=12)
    ax.set_ylabel('Y Coordinate (km)', fontsize=12)
    ax.set_title(f'{title}\n({n_clusters} clusters)', fontsize=14, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    return fig


if __name__ == "__main__":
    # Demo
    print("Hierarchical (Agglomerative) Clustering Demo")
    print("=" * 50)
    
    # Generate data
    X = generate_hierarchical_data(n_samples=150, random_state=42)
    print(f"\nGenerated {X.shape[0]} outlet locations")
    
    # Fit hierarchical clustering
    hc = AgglomerativeClustering(n_clusters=4, linkage='average')
    labels = hc.fit_predict(X)
    
    print(f"\nHierarchical Clustering Results:")
    print(f"  Number of clusters: {hc.n_clusters_}")
    
    # Count per cluster
    for cluster_id in range(hc.n_clusters_):
        count = np.sum(labels == cluster_id)
        print(f"  Cluster {cluster_id}: {count} outlets")
    
    # Plot results
    try:
        fig = plot_clustering_results(X, labels, f"Hierarchical Clustering ({hc.linkage} linkage)")
        fig.savefig('02_unsupervised_learning/03_hierarchical/clustering_result.png', 
                   dpi=150, bbox_inches='tight')
        print("\n✓ Clustering plot saved")
        
        # Plot dendrogram
        fig = plot_dendrogram(hc, "Outlet Hierarchy Dendrogram")
        fig.savefig('02_unsupervised_learning/03_hierarchical/dendrogram.png',
                   dpi=150, bbox_inches='tight')
        print("✓ Dendrogram saved")
    except ImportError as e:
        print(f"\n! {e}")
