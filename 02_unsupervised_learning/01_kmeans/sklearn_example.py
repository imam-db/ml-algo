"""
K-Means Clustering with scikit-learn
====================================

Practical examples using scikit-learn's KMeans for the Indonesian FMCG
retail outlet dataset. Demonstrates feature scaling, optimal k selection,
and cluster interpretation.
"""

import numpy as np
try:
    from sklearn.cluster import KMeans
    from sklearn.preprocessing import StandardScaler
    from sklearn.metrics import silhouette_score
    import matplotlib.pyplot as plt
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False
    print("Warning: scikit-learn not available")


def load_spg_outlets_data(filepath='datasets/spg_outlets.csv'):
    """Load the SPG outlets dataset."""
    data = []
    with open(filepath, 'r') as f:
        headers = f.readline().strip().split(',')
        for line in f:
            data.append(line.strip().split(','))
    
    # Extract feature columns (skip outlet_id, lat, long, outlet_type)
    # Features: monthly_sales_idr, sku_count, visit_freq_per_month, oos_rate, avg_basket_idr
    features = []
    outlet_ids = []
    outlet_types = []
    
    for row in data:
        outlet_ids.append(row[0])
        outlet_types.append(row[3])
        features.append([
            float(row[4]),  # monthly_sales_idr
            float(row[5]),  # sku_count
            float(row[6]),  # visit_freq_per_month
            float(row[7]),  # oos_rate
            float(row[8])   # avg_basket_idr
        ])
    
    return np.array(features), outlet_ids, outlet_types, headers


def example_basic_clustering():
    """Example 1: Basic K-Means clustering on SPG outlets."""
    print("\n" + "=" * 60)
    print("Example 1: Basic K-Means Clustering")
    print("=" * 60)
    
    # Load data
    X, outlet_ids, outlet_types, headers = load_spg_outlets_data()
    print(f"Loaded {len(X)} outlets with {X.shape[1]} features")
    
    # IMPORTANT: Always scale features before K-Means!
    # K-Means uses Euclidean distance, so features with larger ranges
    # will dominate the clustering
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    print("✓ Features scaled to zero mean and unit variance")
    
    # Fit K-Means with k=4 (we know there are 4 hidden segments)
    kmeans = KMeans(n_clusters=4, random_state=42, n_init=10)
    labels = kmeans.fit_predict(X_scaled)
    
    print(f"\nClustering results:")
    print(f"  Inertia: {kmeans.inertia_:.2f}")
    print(f"  Iterations: {kmeans.n_iter_}")
    
    # Analyze clusters
    print(f"\nCluster distribution:")
    unique, counts = np.unique(labels, return_counts=True)
    for cluster_id, count in zip(unique, counts):
        print(f"  Cluster {cluster_id}: {count} outlets")
    
    # Show cluster centers (in original scale)
    feature_names = ['monthly_sales_idr', 'sku_count', 'visit_freq_per_month', 
                     'oos_rate', 'avg_basket_idr']
    print(f"\nCluster centers (original scale):")
    centers_original = scaler.inverse_transform(kmeans.cluster_centers_)
    for i, center in enumerate(centers_original):
        print(f"\n  Cluster {i}:")
        for fname, val in zip(feature_names, center):
            if 'idr' in fname:
                print(f"    {fname}: Rp {val:,.0f}")
            elif 'rate' in fname:
                print(f"    {fname}: {val:.3f}")
            else:
                print(f"    {fname}: {val:.2f}")
    
    return X_scaled, labels, kmeans


def example_elbow_method():
    """Example 2: Use elbow method to find optimal k."""
    print("\n" + "=" * 60)
    print("Example 2: Elbow Method for Optimal k")
    print("=" * 60)
    
    X, _, _, _ = load_spg_outlets_data()
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Try different values of k
    k_range = range(2, 11)
    inertias = []
    
    print("Testing different values of k...")
    for k in k_range:
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
        kmeans.fit(X_scaled)
        inertias.append(kmeans.inertia_)
        print(f"  k={k}: inertia={kmeans.inertia_:.2f}")
    
    print("\nLook for the 'elbow' where inertia starts decreasing more slowly.")
    print("For this dataset, k=4 or k=5 might be good choices.")
    
    return k_range, inertias


def example_silhouette_analysis():
    """Example 3: Use silhouette score to evaluate clustering quality."""
    print("\n" + "=" * 60)
    print("Example 3: Silhouette Analysis")
    print("=" * 60)
    
    X, _, _, _ = load_spg_outlets_data()
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Try different values of k
    k_range = range(2, 11)
    silhouette_scores = []
    
    print("Computing silhouette scores...")
    for k in k_range:
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
        labels = kmeans.fit_predict(X_scaled)
        score = silhouette_score(X_scaled, labels)
        silhouette_scores.append(score)
        print(f"  k={k}: silhouette={score:.3f}")
    
    best_k = k_range[np.argmax(silhouette_scores)]
    print(f"\nBest k by silhouette score: {best_k}")
    print("Silhouette score ranges from -1 to 1:")
    print("  1.0: Perfect, well-separated clusters")
    print("  0.0: Overlapping clusters")
    print(" -1.0: Wrong assignments")
    
    return k_range, silhouette_scores


def example_cluster_interpretation():
    """Example 4: Interpret clusters for SPG visit planning."""
    print("\n" + "=" * 60)
    print("Example 4: Cluster Interpretation for Visit Planning")
    print("=" * 60)
    
    X, outlet_ids, outlet_types, _ = load_spg_outlets_data()
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Fit K-Means with k=4
    kmeans = KMeans(n_clusters=4, random_state=42, n_init=10)
    labels = kmeans.fit_predict(X_scaled)
    
    feature_names = ['monthly_sales_idr', 'sku_count', 'visit_freq_per_month', 
                     'oos_rate', 'avg_basket_idr']
    centers_original = scaler.inverse_transform(kmeans.cluster_centers_)
    
    print("Cluster characteristics and recommended visit strategy:\n")
    
    for i in range(4):
        cluster_mask = labels == i
        cluster_size = cluster_mask.sum()
        center = centers_original[i]
        
        # Interpret based on center values
        sales = center[0]
        sku = center[1]
        oos = center[3]
        
        print(f"Cluster {i} ({cluster_size} outlets):")
        print(f"  Avg monthly sales: Rp {sales:,.0f}")
        print(f"  Avg SKU count: {sku:.0f}")
        print(f"  Avg OOS rate: {oos:.2%}")
        
        # Suggest visit strategy
        if sales > 20_000_000:
            strategy = "HIGH PRIORITY - Weekly visits, full merchandising support"
        elif sales > 8_000_000:
            strategy = "MEDIUM PRIORITY - Bi-weekly visits, regular stock checks"
        elif oos > 0.30:
            strategy = "NEEDS ATTENTION - Focus on reducing out-of-stock issues"
        else:
            strategy = "LOW PRIORITY - Monthly visits, basic support"
        
        print(f"  → Strategy: {strategy}")
        print()


def visualize_results():
    """Create visualizations if matplotlib is available."""
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("Matplotlib not available, skipping visualizations")
        return
    
    X, _, _, _ = load_spg_outlets_data()
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Elbow plot
    k_range, inertias = example_elbow_method()
    
    # Silhouette plot
    k_range_sil, sil_scores = example_silhouette_analysis()
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Plot 1: Elbow curve
    axes[0].plot(list(k_range), inertias, 'bo-', linewidth=2, markersize=8)
    axes[0].set_xlabel('Number of clusters (k)', fontsize=12)
    axes[0].set_ylabel('Inertia', fontsize=12)
    axes[0].set_title('Elbow Method', fontsize=14, fontweight='bold')
    axes[0].grid(True, alpha=0.3)
    
    # Plot 2: Silhouette scores
    axes[1].plot(list(k_range_sil), sil_scores, 'ro-', linewidth=2, markersize=8)
    axes[1].set_xlabel('Number of clusters (k)', fontsize=12)
    axes[1].set_ylabel('Silhouette Score', fontsize=12)
    axes[1].set_title('Silhouette Analysis', fontsize=14, fontweight='bold')
    axes[1].grid(True, alpha=0.3)
    axes[1].axhline(y=0, color='gray', linestyle='--', linewidth=1)
    
    plt.tight_layout()
    plt.savefig('kmeans_evaluation.png', dpi=150, bbox_inches='tight')
    print("\n✓ Saved plots to kmeans_evaluation.png")
    plt.show()


if __name__ == "__main__":
    if not SKLEARN_AVAILABLE:
        print("Please install scikit-learn to run this example:")
        print("  pip install scikit-learn")
        exit(1)
    
    print("K-Means Clustering - Indonesian FMCG Retail Outlets")
    print("Story: A supervisor wants to segment outlets to optimize SPG visits")
    
    # Run examples
    example_basic_clustering()
    example_elbow_method()
    example_silhouette_analysis()
    example_cluster_interpretation()
    
    # Visualizations
    try:
        visualize_results()
    except Exception as e:
        print(f"Could not create visualizations: {e}")
    
    print("\n" + "=" * 60)
    print("Key Takeaways:")
    print("=" * 60)
    print("1. ALWAYS scale features before K-Means")
    print("2. Use elbow method and silhouette score together to choose k")
    print("3. Interpret clusters in business terms for actionable insights")
    print("4. K-Means works best when clusters are roughly spherical and similar sized")
