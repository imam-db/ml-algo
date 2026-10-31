"""
DBSCAN (Density-Based Spatial Clustering of Applications with Noise)
Implementasi dari nol untuk tujuan pembelajaran.

DBSCAN mengelompokkan data berdasarkan kepadatan (density) wilayah,
dapat menemukan cluster dengan bentuk arbitrary dan mengidentifikasi outlier.
"""

from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from matplotlib.figure import Figure


class DBSCAN:
    """
    DBSCAN Clustering dari nol.

    DBSCAN mengelompokkan titik-titik yang saling berdekatan (dense regions)
    dan menandai titik-titik yang terisolasi sebagai noise/outlier.

    Konsep utama:
    - Core point: titik dengan minimal min_samples tetangga dalam radius eps
    - Border point: bukan core point tapi dalam radius eps dari core point
    - Noise point: bukan core maupun border point

    Parameters:
    -----------
    eps : float, default=0.5
        Maximum distance antara dua sampel agar dianggap sebagai tetangga.
        Radius lingkungan (epsilon-neighborhood).

    min_samples : int, default=5
        Minimum jumlah sampel dalam eps-neighborhood agar titik dianggap
        sebagai core point (termasuk titik itu sendiri).

    metric : str, default='euclidean'
        Metrik jarak. Untuk saat ini hanya mendukung 'euclidean'.

    Attributes:
    -----------
    labels_ : ndarray of shape (n_samples,)
        Label cluster untuk setiap sampel. Label -1 menandakan noise.

    core_sample_indices_ : ndarray
        Indeks dari core samples.

    n_clusters_ : int
        Jumlah cluster yang ditemukan (tidak termasuk noise).

    Example:
    --------
    >>> X = np.array([[1, 2], [2, 2], [2, 3], [8, 7], [8, 8], [25, 80]])
    >>> dbscan = DBSCAN(eps=3, min_samples=2)
    >>> dbscan.fit(X)
    >>> print(dbscan.labels_)
    [0 0 0 1 1 -1]
    >>> print(dbscan.n_clusters_)
    2
    """

    def __init__(self, eps: float = 0.5, min_samples: int = 5, metric: str = "euclidean") -> None:
        self.eps = eps
        self.min_samples = min_samples
        self.metric = metric

        # Akan di-set setelah fit
        self.labels_ = None
        self.core_sample_indices_ = None
        self.n_clusters_ = None
        self._X = None

    def fit(self, X: np.ndarray) -> "DBSCAN":
        """
        Fit DBSCAN pada data X.

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

        self._X = X.copy()
        self.labels_ = np.full(n_samples, -1, dtype=int)  # -1 = noise

        # Hitung distance matrix
        distances = self._compute_distances(X)

        # Identifikasi core samples
        # Untuk setiap titik, hitung jumlah tetangga dalam radius eps
        neighbors_count = np.sum(distances <= self.eps, axis=1)
        core_mask = neighbors_count >= self.min_samples
        self.core_sample_indices_ = np.where(core_mask)[0]

        # Clustering: expand dari setiap core point
        cluster_id = 0

        for idx in range(n_samples):
            # Skip jika sudah di-assign ke cluster
            if self.labels_[idx] != -1:
                continue

            # Skip jika bukan core point
            if not core_mask[idx]:
                continue

            # Expand cluster dari core point ini
            self._expand_cluster(idx, cluster_id, distances, core_mask)
            cluster_id += 1

        self.n_clusters_ = cluster_id

        return self

    def _compute_distances(self, X: np.ndarray) -> np.ndarray:
        """Compute pairwise Euclidean distances."""
        n_samples = X.shape[0]
        distances = np.zeros((n_samples, n_samples))

        for i in range(n_samples):
            for j in range(i, n_samples):
                dist = np.linalg.norm(X[i] - X[j])
                distances[i, j] = dist
                distances[j, i] = dist

        return distances

    def _expand_cluster(
        self, start_idx: int, cluster_id: int, distances: np.ndarray, core_mask: np.ndarray
    ) -> None:
        """
        Expand cluster dari core point start_idx menggunakan BFS.
        """
        # Queue untuk BFS
        queue = [start_idx]
        self.labels_[start_idx] = cluster_id

        while queue:
            current_idx = queue.pop(0)

            # Dapatkan semua tetangga dalam radius eps
            neighbors = np.where(distances[current_idx] <= self.eps)[0]

            for neighbor_idx in neighbors:
                # Jika neighbor adalah noise, assign ke cluster ini
                if self.labels_[neighbor_idx] == -1:
                    self.labels_[neighbor_idx] = cluster_id

                    # Jika neighbor adalah core point, expand dari situ juga
                    if core_mask[neighbor_idx]:
                        queue.append(neighbor_idx)

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
            Cluster labels. -1 = noise.
        """
        self.fit(X)
        return self.labels_


def generate_clustering_data(n_samples: int = 300, random_state: int | None = None) -> np.ndarray:
    """
    Generate synthetic data dengan multiple clusters untuk DBSCAN.

    Tema SPG: Lokasi toko dalam kota (koordinat x, y).

    Parameters:
    -----------
    n_samples : int
        Total jumlah sampel.

    random_state : int, optional
        Random seed untuk reproducibility.

    Returns:
    --------
    X : ndarray of shape (n_samples, 2)
        Data koordinat toko.
    """
    if random_state is not None:
        np.random.seed(random_state)

    # Cluster 1: Toko di area mall pusat kota
    n_cluster1 = n_samples // 3
    cluster1 = np.random.randn(n_cluster1, 2) * 1.5 + np.array([0, 0])

    # Cluster 2: Toko di area perumahan
    n_cluster2 = n_samples // 3
    cluster2 = np.random.randn(n_cluster2, 2) * 1.2 + np.array([8, 3])

    # Cluster 3: Toko di area bisnis
    n_cluster3 = n_samples // 3
    cluster3 = np.random.randn(n_cluster3, 2) * 1.0 + np.array([4, 8])

    # Outliers: Toko terisolasi
    n_outliers = n_samples - n_cluster1 - n_cluster2 - n_cluster3
    outliers = np.random.uniform(low=-5, high=15, size=(n_outliers, 2))

    X = np.vstack([cluster1, cluster2, cluster3, outliers])

    return X


def plot_dbscan_results(
    X: np.ndarray,
    labels: np.ndarray,
    core_indices: np.ndarray | None = None,
    title: str = "DBSCAN Clustering",
) -> "Figure":
    """
    Visualisasi hasil DBSCAN clustering.

    Parameters:
    -----------
    X : ndarray of shape (n_samples, 2)
        Data (harus 2D untuk plotting).

    labels : ndarray of shape (n_samples,)
        Cluster labels dari DBSCAN.

    core_indices : ndarray, optional
        Indeks dari core samples.

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

    # Pisahkan noise dari cluster
    unique_labels = set(labels)
    n_clusters = len(unique_labels) - (1 if -1 in unique_labels else 0)

    colors = plt.cm.Spectral(np.linspace(0, 1, len(unique_labels)))

    for label, color in zip(unique_labels, colors):
        if label == -1:
            # Noise: warna hitam
            color = [0, 0, 0, 1]
            marker = "x"
            label_name = "Noise (Outlier)"
        else:
            marker = "o"
            label_name = f"Cluster {label}"

        mask = labels == label
        ax.scatter(
            X[mask, 0],
            X[mask, 1],
            c=[color],
            marker=marker,
            s=100,
            alpha=0.6,
            edgecolors="k",
            linewidth=0.5,
            label=label_name,
        )

    # Highlight core samples
    if core_indices is not None:
        ax.scatter(
            X[core_indices, 0],
            X[core_indices, 1],
            s=200,
            facecolors="none",
            edgecolors="red",
            linewidth=2,
            label="Core Samples",
        )

    ax.set_xlabel("X Coordinate (km)", fontsize=12)
    ax.set_ylabel("Y Coordinate (km)", fontsize=12)
    ax.set_title(f"{title}\n({n_clusters} clusters found)", fontsize=14, fontweight="bold")
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    return fig


if __name__ == "__main__":
    # Demo
    print("DBSCAN Clustering Demo")
    print("=" * 50)

    # Generate data
    X = generate_clustering_data(n_samples=300, random_state=42)
    print(f"\nGenerated {X.shape[0]} store locations")

    # Fit DBSCAN
    dbscan = DBSCAN(eps=1.5, min_samples=5)
    labels = dbscan.fit_predict(X)

    print("\nDBSCAN Results:")
    print(f"  Number of clusters: {dbscan.n_clusters_}")
    print(f"  Number of core samples: {len(dbscan.core_sample_indices_)}")
    print(f"  Number of noise points: {np.sum(labels == -1)}")

    # Count per cluster
    for cluster_id in range(dbscan.n_clusters_):
        count = np.sum(labels == cluster_id)
        print(f"  Cluster {cluster_id}: {count} stores")

    # Plot
    try:
        fig = plot_dbscan_results(X, labels, dbscan.core_sample_indices_)
        fig.savefig(
            "02_unsupervised_learning/02_dbscan/dbscan_result.png", dpi=150, bbox_inches="tight"
        )
        print("\n✓ Plot saved to dbscan_result.png")
    except ImportError:
        print("\n! Matplotlib not available, skipping plot")
