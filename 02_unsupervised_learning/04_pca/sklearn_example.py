"""
Principal Component Analysis (PCA) with scikit-learn
===================================================

Practical examples using scikit-learn's PCA for the Indonesian FMCG
retail outlet dataset. Demonstrates dimensionality reduction, visualization,
and variance analysis.
"""

import numpy as np
from pathlib import Path

try:
    from sklearn.decomposition import PCA
    from sklearn.preprocessing import StandardScaler
    from sklearn.cluster import KMeans
    import matplotlib.pyplot as plt
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False
    print("Warning: scikit-learn not available")


def load_spg_outlets_data(filepath=None):
    """Load the SPG outlets dataset (path is resolved relative to this script)."""
    if filepath is None:
        filepath = Path(__file__).resolve().parent.parent / '01_kmeans' / 'datasets' / 'spg_outlets.csv'
    else:
        filepath = Path(filepath)

    data = []
    with open(filepath, 'r') as f:
        headers = f.readline().strip().split(',')
        for line in f:
            data.append(line.strip().split(','))
    
    # Extract feature columns
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
    
    return np.array(features), outlet_ids, outlet_types


def example_basic_pca():
    """Example 1: Basic PCA for dimensionality reduction."""
    print("\n" + "=" * 60)
    print("Example 1: Basic PCA - Reduce 5D to 2D")
    print("=" * 60)
    
    # Load data
    X, outlet_ids, outlet_types = load_spg_outlets_data()
    print(f"Loaded {len(X)} outlets with {X.shape[1]} features")
    
    # Scale features (ALWAYS scale before PCA!)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    print("✓ Features scaled")
    
    # Fit PCA to reduce to 2 components
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X_scaled)
    
    print(f"\nPCA Results:")
    print(f"  Original shape: {X_scaled.shape}")
    print(f"  Reduced shape: {X_pca.shape}")
    print(f"  Explained variance ratio: {pca.explained_variance_ratio_}")
    print(f"  Cumulative variance: {pca.explained_variance_ratio_.cumsum()}")
    print(f"  → First 2 components capture {pca.explained_variance_ratio_.sum()*100:.1f}% of variance")
    
    return X_scaled, X_pca, pca


def example_variance_analysis():
    """Example 2: Analyze variance explained by each component."""
    print("\n" + "=" * 60)
    print("Example 2: Variance Analysis")
    print("=" * 60)
    
    X, _, _ = load_spg_outlets_data()
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Fit PCA with all components
    pca = PCA()
    pca.fit(X_scaled)
    
    print(f"Number of features: {X_scaled.shape[1]}")
    print(f"Number of components: {pca.n_components_}")
    print(f"\nVariance explained by each component:")
    
    for i, (var, ratio) in enumerate(zip(pca.explained_variance_, 
                                         pca.explained_variance_ratio_)):
        cumsum = pca.explained_variance_ratio_[:i+1].sum()
        print(f"  PC{i+1}: variance={var:.2f}, ratio={ratio:.3f} ({ratio*100:.1f}%), cumulative={cumsum:.3f}")
    
    # How many components for 90% variance?
    cumsum = pca.explained_variance_ratio_.cumsum()
    n_90 = np.argmax(cumsum >= 0.90) + 1
    print(f"\nTo capture 90% of variance, need {n_90} components")
    
    return pca


def example_visualization_with_clusters():
    """Example 3: Visualize K-Means clusters in 2D PCA space."""
    print("\n" + "=" * 60)
    print("Example 3: Visualize Clusters in PCA Space")
    print("=" * 60)
    
    X, outlet_ids, outlet_types = load_spg_outlets_data()
    
    # Scale and cluster
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    kmeans = KMeans(n_clusters=4, random_state=42, n_init=10)
    labels = kmeans.fit_predict(X_scaled)
    print(f"✓ Clustered {len(X)} outlets into 4 segments")
    
    # Reduce to 2D with PCA
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X_scaled)
    print(f"✓ Reduced to 2D (captures {pca.explained_variance_ratio_.sum()*100:.1f}% variance)")
    
    print(f"\nCluster distribution:")
    unique, counts = np.unique(labels, return_counts=True)
    for cluster_id, count in zip(unique, counts):
        print(f"  Cluster {cluster_id}: {count} outlets")
    
    return X_pca, labels, pca


def example_feature_importance():
    """Example 4: Understand which original features contribute to each PC."""
    print("\n" + "=" * 60)
    print("Example 4: Feature Importance in Principal Components")
    print("=" * 60)
    
    X, _, _ = load_spg_outlets_data()
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    pca = PCA(n_components=2)
    pca.fit(X_scaled)
    
    feature_names = ['Monthly Sales', 'SKU Count', 'Visit Freq', 'OOS Rate', 'Avg Basket']
    
    print("Principal Component Loadings:\n")
    print("(How much each original feature contributes to each PC)")
    print()
    
    for i in range(2):
        print(f"PC{i+1} (explains {pca.explained_variance_ratio_[i]*100:.1f}% variance):")
        loadings = pca.components_[i]
        
        # Sort by absolute magnitude
        sorted_indices = np.argsort(np.abs(loadings))[::-1]
        
        for idx in sorted_indices:
            contribution = loadings[idx]
            print(f"  {feature_names[idx]:15s}: {contribution:+.3f}")
        print()


def example_reconstruction():
    """Example 5: Reconstruct data from reduced dimensions."""
    print("\n" + "=" * 60)
    print("Example 5: Data Reconstruction")
    print("=" * 60)
    
    X, _, _ = load_spg_outlets_data()
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    print("Reconstruction error vs. number of components:\n")
    
    for n_comp in [1, 2, 3, 4, 5]:
        pca = PCA(n_components=n_comp)
        X_pca = pca.fit_transform(X_scaled)
        X_reconstructed = pca.inverse_transform(X_pca)
        
        mse = np.mean((X_scaled - X_reconstructed) ** 2)
        var_explained = pca.explained_variance_ratio_.sum()
        
        print(f"  {n_comp} components: MSE={mse:.4f}, variance={var_explained:.3f} ({var_explained*100:.1f}%)")
    
    print("\nNote: With all 5 components, reconstruction is perfect (MSE ≈ 0)")


def example_why_scaling_matters():
    """Example 6: Demonstrate why scaling is critical for PCA (Kenapa harus scaling?)."""
    print("\n" + "=" * 60)
    print("Example 6: Kenapa Harus Scaling? (Why Scaling Matters)")
    print("=" * 60)
    
    X, _, _ = load_spg_outlets_data()
    
    print("\nOriginal feature ranges (before scaling):")
    feature_names = ['Monthly Sales (IDR)', 'SKU Count', 'Visit Freq', 'OOS Rate', 'Avg Basket (IDR)']
    for i, name in enumerate(feature_names):
        print(f"  {name:25s}: {X[:, i].min():12,.0f} to {X[:, i].max():12,.0f} (range: {X[:, i].max()-X[:, i].min():12,.0f})")
    
    # PCA WITHOUT scaling
    print("\n" + "-" * 60)
    print("WITHOUT Scaling (BAD!):")
    print("-" * 60)
    pca_no_scale = PCA(n_components=2)
    pca_no_scale.fit(X)
    
    print(f"  PC1 explains: {pca_no_scale.explained_variance_ratio_[0]*100:.1f}%")
    print(f"  PC2 explains: {pca_no_scale.explained_variance_ratio_[1]*100:.1f}%")
    print(f"  Total (2 PCs): {pca_no_scale.explained_variance_ratio_[:2].sum()*100:.1f}%")
    
    print(f"\n  Feature contributions to PC1:")
    loadings = pca_no_scale.components_[0]
    for i, name in enumerate(feature_names):
        print(f"    {name:25s}: {loadings[i]:+.6f}")
    
    print(f"\n  ⚠️  PC1 is dominated by features with large ranges (IDR columns)")
    print(f"  ⚠️  Features with small ranges (OOS rate) are ignored")
    
    # PCA WITH scaling
    print("\n" + "-" * 60)
    print("WITH Scaling (GOOD!):")
    print("-" * 60)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    pca_scaled = PCA(n_components=2)
    pca_scaled.fit(X_scaled)
    
    print(f"  PC1 explains: {pca_scaled.explained_variance_ratio_[0]*100:.1f}%")
    print(f"  PC2 explains: {pca_scaled.explained_variance_ratio_[1]*100:.1f}%")
    print(f"  Total (2 PCs): {pca_scaled.explained_variance_ratio_[:2].sum()*100:.1f}%")
    
    print(f"\n  Feature contributions to PC1:")
    loadings = pca_scaled.components_[0]
    for i, name in enumerate(feature_names):
        print(f"    {name:25s}: {loadings[i]:+.6f}")
    
    print(f"\n  ✓ All features contribute meaningfully to PC1")
    print(f"  ✓ PC1 captures overall outlet 'size/performance'")
    
    # Key insights
    print("\n" + "=" * 60)
    print("Key Insights:")
    print("=" * 60)
    print("1. WITHOUT scaling: Large-range features dominate")
    print("   - Monthly Sales (IDR) and Avg Basket (IDR) have huge ranges")
    print("   - PCA finds variance only in those features")
    print("   - Small-range features (OOS rate, visit freq) are ignored")
    print("")
    print("2. WITH scaling: All features contribute equally")
    print("   - Each feature has mean=0, std=1")
    print("   - PCA finds meaningful patterns across ALL features")
    print("   - Better representation of outlet characteristics")
    print("")
    print("3. Rule: ALWAYS scale before PCA unless features are already")
    print("   on the same scale (e.g., all percentages, all counts)")
    
    return X, X_scaled, pca_no_scale, pca_scaled


def visualize_results():
    """Create visualizations if matplotlib is available."""
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("Matplotlib not available, skipping visualizations")
        return
    
    X, _, _ = load_spg_outlets_data()
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Variance analysis
    pca_all = PCA()
    pca_all.fit(X_scaled)
    
    # Clustering in PCA space
    kmeans = KMeans(n_clusters=4, random_state=42, n_init=10)
    labels = kmeans.fit_predict(X_scaled)
    
    pca_2d = PCA(n_components=2)
    X_pca = pca_2d.fit_transform(X_scaled)
    
    # Create plots
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    
    # Plot 1: Scree plot (variance explained)
    axes[0].bar(range(1, 6), pca_all.explained_variance_ratio_, alpha=0.7, color='steelblue')
    axes[0].plot(range(1, 6), pca_all.explained_variance_ratio_.cumsum(), 
                'ro-', linewidth=2, markersize=8)
    axes[0].set_xlabel('Principal Component', fontsize=12)
    axes[0].set_ylabel('Variance Explained', fontsize=12)
    axes[0].set_title('Scree Plot', fontsize=14, fontweight='bold')
    axes[0].set_xticks(range(1, 6))
    axes[0].legend(['Cumulative', 'Individual'], loc='center right')
    axes[0].grid(True, alpha=0.3, axis='y')
    axes[0].axhline(y=0.9, color='red', linestyle='--', alpha=0.5, label='90%')
    
    # Plot 2: Clusters in 2D PCA space
    colors = ['red', 'blue', 'green', 'orange']
    for i in range(4):
        cluster_mask = labels == i
        axes[1].scatter(X_pca[cluster_mask, 0], X_pca[cluster_mask, 1],
                       c=colors[i], label=f'Cluster {i}', alpha=0.6, s=50)
    
    axes[1].set_xlabel(f'PC1 ({pca_2d.explained_variance_ratio_[0]*100:.1f}%)', fontsize=12)
    axes[1].set_ylabel(f'PC2 ({pca_2d.explained_variance_ratio_[1]*100:.1f}%)', fontsize=12)
    axes[1].set_title('Clusters in PCA Space', fontsize=14, fontweight='bold')
    axes[1].legend(loc='best')
    axes[1].grid(True, alpha=0.3)
    
    # Plot 3: Feature contributions (biplot-style)
    feature_names = ['Sales', 'SKU', 'Freq', 'OOS', 'Basket']
    for i, feature in enumerate(feature_names):
        axes[2].arrow(0, 0, 
                     pca_2d.components_[0, i] * 3,
                     pca_2d.components_[1, i] * 3,
                     head_width=0.15, head_length=0.15, fc='darkred', ec='darkred', 
                     linewidth=2, alpha=0.7)
        axes[2].text(pca_2d.components_[0, i] * 3.3,
                    pca_2d.components_[1, i] * 3.3,
                    feature, fontsize=11, fontweight='bold',
                    ha='center', va='center')
    
    axes[2].set_xlabel(f'PC1 ({pca_2d.explained_variance_ratio_[0]*100:.1f}%)', fontsize=12)
    axes[2].set_ylabel(f'PC2 ({pca_2d.explained_variance_ratio_[1]*100:.1f}%)', fontsize=12)
    axes[2].set_title('Feature Contributions (Loadings)', fontsize=14, fontweight='bold')
    axes[2].grid(True, alpha=0.3)
    axes[2].axhline(y=0, color='black', linewidth=0.5)
    axes[2].axvline(x=0, color='black', linewidth=0.5)
    axes[2].set_xlim(-3, 3)
    axes[2].set_ylim(-3, 3)
    
    plt.tight_layout()
    plt.savefig('pca_analysis.png', dpi=150, bbox_inches='tight')
    print("\n✓ Saved plots to pca_analysis.png")
    plt.show()


if __name__ == "__main__":
    if not SKLEARN_AVAILABLE:
        print("Please install scikit-learn to run this example:")
        print("  pip install scikit-learn")
        exit(1)
    
    print("PCA - Indonesian FMCG Retail Outlets")
    print("Story: Visualize 5D outlet data in 2D to understand segments")
    
    # Run examples
    example_basic_pca()
    example_variance_analysis()
    example_why_scaling_matters()
    example_visualization_with_clusters()
    example_feature_importance()
    example_reconstruction()
    
    # Visualizations
    try:
        visualize_results()
    except Exception as e:
        print(f"Could not create visualizations: {e}")
    
    print("\n" + "=" * 60)
    print("Key Takeaways:")
    print("=" * 60)
    print("1. PCA reduces dimensionality while preserving variance")
    print("2. ALWAYS scale features before PCA")
    print("3. Check explained variance ratio to choose n_components")
    print("4. PCA is great for visualization (reduce to 2D or 3D)")
    print("5. Use PCA before clustering when you have many features")
