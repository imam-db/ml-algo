"""
Perbandingan implementasi Hierarchical Clustering dari nol dengan sklearn.
"""

import numpy as np
from sklearn.cluster import AgglomerativeClustering as SklearnAgglomerative
from sklearn.metrics import silhouette_score, adjusted_rand_score
import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import dendrogram, linkage as scipy_linkage

from implementation import AgglomerativeClustering, generate_hierarchical_data, plot_dendrogram


def compare_hierarchical():
    """Bandingkan implementasi custom dengan sklearn."""
    print("Hierarchical Clustering: Custom vs Sklearn")
    print("=" * 60)
    
    # Generate data
    X = generate_hierarchical_data(n_samples=150, random_state=42)
    print(f"Data: {X.shape[0]} samples, {X.shape[1]} features\n")
    
    # Parameters
    n_clusters = 4
    linkage_method = 'average'
    
    # Custom implementation
    print(f"Custom Implementation (linkage={linkage_method}):")
    custom_hc = AgglomerativeClustering(n_clusters=n_clusters, linkage=linkage_method)
    custom_labels = custom_hc.fit_predict(X)
    print(f"  Clusters: {custom_hc.n_clusters_}")
    for i in range(custom_hc.n_clusters_):
        count = np.sum(custom_labels == i)
        print(f"  Cluster {i}: {count} samples")
    
    # Sklearn implementation
    print(f"\nSklearn Implementation (linkage={linkage_method}):")
    sklearn_hc = SklearnAgglomerative(n_clusters=n_clusters, linkage=linkage_method)
    sklearn_labels = sklearn_hc.fit_predict(X)
    print(f"  Clusters: {n_clusters}")
    for i in range(n_clusters):
        count = np.sum(sklearn_labels == i)
        print(f"  Cluster {i}: {count} samples")
    
    # Comparison
    print("\nComparison:")
    ari = adjusted_rand_score(custom_labels, sklearn_labels)
    print(f"  Adjusted Rand Index: {ari:.4f}")
    print(f"  {'✓ High agreement!' if ari > 0.9 else '~ Some label differences (normal)'}")
    
    # Silhouette scores
    custom_silhouette = silhouette_score(X, custom_labels)
    sklearn_silhouette = silhouette_score(X, sklearn_labels)
    print(f"\n  Custom Silhouette Score: {custom_silhouette:.4f}")
    print(f"  Sklearn Silhouette Score: {sklearn_silhouette:.4f}")
    
    # Visualize comparison
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    
    # Custom
    ax = axes[0]
    colors = plt.cm.Spectral(np.linspace(0, 1, n_clusters))
    for i, color in enumerate(colors):
        mask = custom_labels == i
        ax.scatter(X[mask, 0], X[mask, 1], c=[color], s=80, alpha=0.7,
                  edgecolors='k', linewidth=0.5, label=f'Cluster {i}')
    ax.set_title(f'Custom Hierarchical\n{linkage_method.capitalize()} Linkage', 
                fontsize=13, fontweight='bold')
    ax.set_xlabel('X Coordinate')
    ax.set_ylabel('Y Coordinate')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Sklearn
    ax = axes[1]
    for i, color in enumerate(colors):
        mask = sklearn_labels == i
        ax.scatter(X[mask, 0], X[mask, 1], c=[color], s=80, alpha=0.7,
                  edgecolors='k', linewidth=0.5, label=f'Cluster {i}')
    ax.set_title(f'Sklearn Hierarchical\n{linkage_method.capitalize()} Linkage',
                fontsize=13, fontweight='bold')
    ax.set_xlabel('X Coordinate')
    ax.set_ylabel('Y Coordinate')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('02_unsupervised_learning/03_hierarchical/comparison.png', 
               dpi=150, bbox_inches='tight')
    print("\n✓ Comparison plot saved to comparison.png")
    
    return custom_labels, sklearn_labels


def compare_linkage_methods():
    """Bandingkan berbagai linkage methods."""
    print("\n" + "=" * 60)
    print("Linkage Method Comparison")
    print("=" * 60)
    
    X = generate_hierarchical_data(n_samples=150, random_state=42)
    linkages = ['single', 'complete', 'average']
    
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    
    for ax, linkage_method in zip(axes, linkages):
        print(f"\n{linkage_method.capitalize()} Linkage:")
        
        hc = AgglomerativeClustering(n_clusters=4, linkage=linkage_method)
        labels = hc.fit_predict(X)
        
        # Compute silhouette
        silhouette = silhouette_score(X, labels)
        print(f"  Silhouette Score: {silhouette:.4f}")
        
        # Count per cluster
        for i in range(4):
            count = np.sum(labels == i)
            print(f"  Cluster {i}: {count} samples")
        
        # Plot
        colors = plt.cm.Spectral(np.linspace(0, 1, 4))
        for i, color in enumerate(colors):
            mask = labels == i
            ax.scatter(X[mask, 0], X[mask, 1], c=[color], s=60, alpha=0.7,
                      edgecolors='k', linewidth=0.5)
        
        ax.set_title(f'{linkage_method.capitalize()} Linkage\nSilhouette: {silhouette:.3f}',
                    fontweight='bold')
        ax.set_xlabel('X Coordinate')
        ax.set_ylabel('Y Coordinate')
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('02_unsupervised_learning/03_hierarchical/linkage_comparison.png',
               dpi=150, bbox_inches='tight')
    print("\n✓ Linkage comparison plot saved")


def demonstrate_dendrogram():
    """Demonstrasi dendrogram dan cutting di berbagai level."""
    print("\n" + "=" * 60)
    print("Dendrogram Analysis")
    print("=" * 60)
    
    X = generate_hierarchical_data(n_samples=50, random_state=42)  # Smaller for clarity
    
    # Create linkage matrix using scipy
    Z = scipy_linkage(X, method='average')
    
    # Plot dendrogram
    fig, ax = plt.subplots(figsize=(14, 6))
    
    dendrogram(Z, ax=ax, color_threshold=3.0)
    
    # Add horizontal lines at different cutting thresholds
    thresholds = [2.0, 3.0, 4.0]
    colors = ['red', 'green', 'blue']
    
    for threshold, color in zip(thresholds, colors):
        ax.axhline(y=threshold, color=color, linestyle='--', linewidth=2,
                  label=f'Cut at {threshold}')
    
    ax.set_xlabel('Sample Index', fontsize=12)
    ax.set_ylabel('Distance', fontsize=12)
    ax.set_title('Dendrogram with Different Cut Levels', fontsize=14, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig('02_unsupervised_learning/03_hierarchical/dendrogram_cuts.png',
               dpi=150, bbox_inches='tight')
    print("\n✓ Dendrogram with cuts saved")
    
    # Show resulting cluster counts for different cuts
    from scipy.cluster.hierarchy import fcluster
    
    print("\nCutting at different thresholds:")
    for threshold in thresholds:
        labels = fcluster(Z, threshold, criterion='distance')
        n_clusters = len(np.unique(labels))
        print(f"  Threshold {threshold:.1f}: {n_clusters} clusters")


def choosing_optimal_clusters():
    """Gunakan silhouette score untuk memilih jumlah cluster optimal."""
    print("\n" + "=" * 60)
    print("Choosing Optimal Number of Clusters")
    print("=" * 60)
    
    X = generate_hierarchical_data(n_samples=150, random_state=42)
    
    # Test berbagai n_clusters
    n_clusters_range = range(2, 11)
    silhouette_scores = []
    
    print("\nSilhouette scores for different n_clusters:")
    for n in n_clusters_range:
        hc = SklearnAgglomerative(n_clusters=n, linkage='average')
        labels = hc.fit_predict(X)
        score = silhouette_score(X, labels)
        silhouette_scores.append(score)
        print(f"  n_clusters={n:2d}: {score:.4f}")
    
    # Plot
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(n_clusters_range, silhouette_scores, marker='o', linewidth=2, markersize=8)
    
    best_n = list(n_clusters_range)[np.argmax(silhouette_scores)]
    best_score = max(silhouette_scores)
    ax.axvline(x=best_n, color='red', linestyle='--', linewidth=2,
              label=f'Optimal: {best_n} clusters')
    ax.scatter([best_n], [best_score], color='red', s=200, zorder=5,
              edgecolors='black', linewidth=2)
    
    ax.set_xlabel('Number of Clusters', fontsize=12)
    ax.set_ylabel('Silhouette Score', fontsize=12)
    ax.set_title('Optimal Cluster Selection (Silhouette Method)', 
                fontsize=14, fontweight='bold')
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    ax.set_xticks(n_clusters_range)
    
    plt.tight_layout()
    plt.savefig('02_unsupervised_learning/03_hierarchical/optimal_clusters.png',
               dpi=150, bbox_inches='tight')
    print(f"\n✓ Optimal clusters plot saved")
    print(f"\nRecommendation: Use {best_n} clusters (highest silhouette score)")


if __name__ == "__main__":
    # Main comparison
    compare_hierarchical()
    
    # Linkage comparison
    compare_linkage_methods()
    
    # Dendrogram demo
    demonstrate_dendrogram()
    
    # Optimal clusters
    choosing_optimal_clusters()
    
    print("\n" + "=" * 60)
    print("✓ All comparisons complete!")
    print("=" * 60)
