"""
Perbandingan implementasi DBSCAN dari nol dengan sklearn.
"""

import numpy as np
from sklearn.cluster import DBSCAN as SklearnDBSCAN
from sklearn.metrics import silhouette_score, adjusted_rand_score
import matplotlib.pyplot as plt

from implementation import DBSCAN, generate_clustering_data, plot_dbscan_results


def compare_dbscan():
    """Bandingkan implementasi custom dengan sklearn."""
    print("DBSCAN: Custom vs Sklearn")
    print("=" * 60)
    
    # Generate data
    X = generate_clustering_data(n_samples=300, random_state=42)
    print(f"Data: {X.shape[0]} samples, {X.shape[1]} features\n")
    
    # Parameters
    eps = 1.5
    min_samples = 5
    
    # Custom DBSCAN
    print("Custom Implementation:")
    custom_dbscan = DBSCAN(eps=eps, min_samples=min_samples)
    custom_labels = custom_dbscan.fit_predict(X)
    custom_n_clusters = custom_dbscan.n_clusters_
    custom_n_noise = np.sum(custom_labels == -1)
    print(f"  Clusters: {custom_n_clusters}")
    print(f"  Noise points: {custom_n_noise}")
    print(f"  Core samples: {len(custom_dbscan.core_sample_indices_)}")
    
    # Sklearn DBSCAN
    print("\nSklearn Implementation:")
    sklearn_dbscan = SklearnDBSCAN(eps=eps, min_samples=min_samples)
    sklearn_labels = sklearn_dbscan.fit_predict(X)
    sklearn_n_clusters = len(set(sklearn_labels)) - (1 if -1 in sklearn_labels else 0)
    sklearn_n_noise = np.sum(sklearn_labels == -1)
    print(f"  Clusters: {sklearn_n_clusters}")
    print(f"  Noise points: {sklearn_n_noise}")
    print(f"  Core samples: {len(sklearn_dbscan.core_sample_indices_)}")
    
    # Comparison
    print("\nComparison:")
    print(f"  Clusters match: {custom_n_clusters == sklearn_n_clusters}")
    print(f"  Noise count match: {custom_n_noise == sklearn_n_noise}")
    
    # Adjusted Rand Index (mengukur agreement antara dua clustering)
    ari = adjusted_rand_score(custom_labels, sklearn_labels)
    print(f"  Adjusted Rand Index: {ari:.4f}")
    print(f"  {'✓ Perfect agreement!' if ari == 1.0 else '✗ Some differences exist'}")
    
    # Silhouette scores (exclude noise)
    if custom_n_clusters > 1:
        custom_mask = custom_labels != -1
        sklearn_mask = sklearn_labels != -1
        
        if np.sum(custom_mask) > 0 and len(set(custom_labels[custom_mask])) > 1:
            custom_silhouette = silhouette_score(X[custom_mask], custom_labels[custom_mask])
            print(f"\n  Custom Silhouette Score: {custom_silhouette:.4f}")
        
        if np.sum(sklearn_mask) > 0 and len(set(sklearn_labels[sklearn_mask])) > 1:
            sklearn_silhouette = silhouette_score(X[sklearn_mask], sklearn_labels[sklearn_mask])
            print(f"  Sklearn Silhouette Score: {sklearn_silhouette:.4f}")
    
    # Visualize comparison
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    
    # Custom
    ax = axes[0]
    unique_labels = set(custom_labels)
    colors = plt.cm.Spectral(np.linspace(0, 1, len(unique_labels)))
    
    for label, color in zip(unique_labels, colors):
        if label == -1:
            color = [0, 0, 0, 1]
            marker = 'x'
        else:
            marker = 'o'
        mask = custom_labels == label
        ax.scatter(X[mask, 0], X[mask, 1], c=[color], marker=marker, 
                  s=100, alpha=0.6, edgecolors='k', linewidth=0.5)
    
    ax.scatter(X[custom_dbscan.core_sample_indices_, 0], 
              X[custom_dbscan.core_sample_indices_, 1],
              s=200, facecolors='none', edgecolors='red', linewidth=2)
    ax.set_title(f'Custom DBSCAN\n{custom_n_clusters} clusters', fontsize=13, fontweight='bold')
    ax.set_xlabel('X Coordinate')
    ax.set_ylabel('Y Coordinate')
    ax.grid(True, alpha=0.3)
    
    # Sklearn
    ax = axes[1]
    colors = plt.cm.Spectral(np.linspace(0, 1, len(set(sklearn_labels))))
    
    for label, color in zip(set(sklearn_labels), colors):
        if label == -1:
            color = [0, 0, 0, 1]
            marker = 'x'
        else:
            marker = 'o'
        mask = sklearn_labels == label
        ax.scatter(X[mask, 0], X[mask, 1], c=[color], marker=marker,
                  s=100, alpha=0.6, edgecolors='k', linewidth=0.5)
    
    ax.scatter(X[sklearn_dbscan.core_sample_indices_, 0],
              X[sklearn_dbscan.core_sample_indices_, 1],
              s=200, facecolors='none', edgecolors='red', linewidth=2)
    ax.set_title(f'Sklearn DBSCAN\n{sklearn_n_clusters} clusters', fontsize=13, fontweight='bold')
    ax.set_xlabel('X Coordinate')
    ax.set_ylabel('Y Coordinate')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('02_unsupervised_learning/02_dbscan/comparison.png', dpi=150, bbox_inches='tight')
    print("\n✓ Comparison plot saved to comparison.png")
    
    return custom_labels, sklearn_labels


def test_parameter_sensitivity():
    """Test sensitivitas parameter eps dan min_samples."""
    print("\n" + "=" * 60)
    print("Parameter Sensitivity Analysis")
    print("=" * 60)
    
    X = generate_clustering_data(n_samples=300, random_state=42)
    
    # Test berbagai eps values
    print("\nVarying eps (min_samples=5):")
    for eps in [0.5, 1.0, 1.5, 2.0, 3.0]:
        dbscan = DBSCAN(eps=eps, min_samples=5)
        labels = dbscan.fit_predict(X)
        n_clusters = dbscan.n_clusters_
        n_noise = np.sum(labels == -1)
        print(f"  eps={eps:.1f}: {n_clusters} clusters, {n_noise} noise points")
    
    # Test berbagai min_samples values
    print("\nVarying min_samples (eps=1.5):")
    for min_samples in [2, 5, 10, 15, 20]:
        dbscan = DBSCAN(eps=1.5, min_samples=min_samples)
        labels = dbscan.fit_predict(X)
        n_clusters = dbscan.n_clusters_
        n_noise = np.sum(labels == -1)
        print(f"  min_samples={min_samples:2d}: {n_clusters} clusters, {n_noise} noise points")


def demonstrate_advantage_over_kmeans():
    """Tunjukkan keunggulan DBSCAN vs K-Means pada data non-spherical."""
    print("\n" + "=" * 60)
    print("DBSCAN vs K-Means on Non-Spherical Data")
    print("=" * 60)
    
    # Create moon-shaped data
    from sklearn.datasets import make_moons
    X, _ = make_moons(n_samples=300, noise=0.05, random_state=42)
    
    # DBSCAN
    dbscan = SklearnDBSCAN(eps=0.2, min_samples=5)
    dbscan_labels = dbscan.fit_predict(X)
    dbscan_n_clusters = len(set(dbscan_labels)) - (1 if -1 in dbscan_labels else 0)
    
    # K-Means
    from sklearn.cluster import KMeans
    kmeans = KMeans(n_clusters=2, random_state=42)
    kmeans_labels = kmeans.fit_predict(X)
    
    # Plot
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    # DBSCAN
    ax = axes[0]
    unique_labels = set(dbscan_labels)
    colors = plt.cm.Spectral(np.linspace(0, 1, len(unique_labels)))
    for label, color in zip(unique_labels, colors):
        if label == -1:
            color = [0, 0, 0, 1]
            marker = 'x'
        else:
            marker = 'o'
        mask = dbscan_labels == label
        ax.scatter(X[mask, 0], X[mask, 1], c=[color], marker=marker, s=50, alpha=0.7)
    ax.set_title(f'DBSCAN (eps=0.2, min_samples=5)\n{dbscan_n_clusters} clusters', 
                fontweight='bold')
    ax.set_xlabel('Feature 1')
    ax.set_ylabel('Feature 2')
    
    # K-Means
    ax = axes[1]
    scatter = ax.scatter(X[:, 0], X[:, 1], c=kmeans_labels, cmap='Spectral', s=50, alpha=0.7)
    ax.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1],
              c='red', marker='X', s=200, edgecolors='black', linewidth=2, label='Centroids')
    ax.set_title('K-Means (k=2)\nStruggles with non-spherical shapes', fontweight='bold')
    ax.set_xlabel('Feature 1')
    ax.set_ylabel('Feature 2')
    ax.legend()
    
    plt.tight_layout()
    plt.savefig('02_unsupervised_learning/02_dbscan/dbscan_vs_kmeans.png', dpi=150, bbox_inches='tight')
    print("\n✓ DBSCAN advantage plot saved to dbscan_vs_kmeans.png")
    print("\nObservation:")
    print("  - DBSCAN correctly identifies the two moon-shaped clusters")
    print("  - K-Means struggles because it assumes spherical clusters")


if __name__ == "__main__":
    # Main comparison
    compare_dbscan()
    
    # Parameter sensitivity
    test_parameter_sensitivity()
    
    # Advantage over K-Means
    demonstrate_advantage_over_kmeans()
    
    print("\n" + "=" * 60)
    print("✓ All comparisons complete!")
    print("=" * 60)
