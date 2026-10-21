"""
Generate synthetic Indonesian FMCG retail outlet data for SPG (Sales Promoter) visit planning.

Story: A supervisor wants to group retail outlets (warung, minimarket, grosir) in Denpasar/Bali
to optimize SPG visit planning. Outlets differ in sales volume, SKU count, visit frequency needs,
and out-of-stock rates. K-Means can identify natural segments, PCA can visualize them in 2D.
"""

import numpy as np
from pathlib import Path

def generate_spg_outlets_data(n_samples=500, random_state=42):
    """
    Generate synthetic retail outlet data with realistic hidden segments.
    
    Parameters:
    -----------
    n_samples : int
        Number of outlets to generate
    random_state : int
        Random seed for reproducibility
        
    Returns:
    --------
    data : dict
        Dictionary with column arrays
    """
    rng = np.random.default_rng(random_state)
    
    # Define 4 realistic segments for Indonesian retail
    # Segment 0: Small warungs - low sales, few SKUs, high OOS rate
    # Segment 1: Active minimarkets - medium-high sales, good variety
    # Segment 2: High-volume grosir (wholesalers) - very high sales and SKU count
    # Segment 3: Struggling outlets - low sales, medium SKUs, high OOS
    
    segment_sizes = [200, 150, 75, 75]  # Different segment sizes
    outlets = []
    
    # Denpasar/Bali coordinates (approximately)
    base_lat, base_long = -8.65, 115.22
    
    for seg_idx, seg_size in enumerate(segment_sizes):
        if seg_idx == 0:  # Small warungs
            monthly_sales = rng.normal(3_500_000, 800_000, seg_size)  # ~3.5M IDR
            sku_count = rng.normal(25, 8, seg_size)
            visit_freq = rng.normal(2, 0.5, seg_size)
            oos_rate = rng.normal(0.25, 0.08, seg_size)
            avg_basket = rng.normal(45_000, 12_000, seg_size)
            outlet_type = rng.choice(['warung'], size=seg_size)
            
        elif seg_idx == 1:  # Active minimarkets
            monthly_sales = rng.normal(12_000_000, 2_500_000, seg_size)  # ~12M IDR
            sku_count = rng.normal(80, 15, seg_size)
            visit_freq = rng.normal(4, 0.8, seg_size)
            oos_rate = rng.normal(0.12, 0.04, seg_size)
            avg_basket = rng.normal(85_000, 20_000, seg_size)
            outlet_type = rng.choice(['minimarket'], size=seg_size)
            
        elif seg_idx == 2:  # High-volume grosir
            monthly_sales = rng.normal(35_000_000, 8_000_000, seg_size)  # ~35M IDR
            sku_count = rng.normal(150, 30, seg_size)
            visit_freq = rng.normal(6, 1.2, seg_size)
            oos_rate = rng.normal(0.08, 0.03, seg_size)
            avg_basket = rng.normal(250_000, 60_000, seg_size)
            outlet_type = rng.choice(['grosir'], size=seg_size)
            
        else:  # Struggling outlets
            monthly_sales = rng.normal(2_000_000, 600_000, seg_size)  # ~2M IDR
            sku_count = rng.normal(45, 12, seg_size)
            visit_freq = rng.normal(1.5, 0.4, seg_size)
            oos_rate = rng.normal(0.35, 0.10, seg_size)
            avg_basket = rng.normal(35_000, 10_000, seg_size)
            outlet_type = rng.choice(['warung', 'minimarket'], size=seg_size)
        
        # Generate geographic clusters (outlets in same segment tend to be near each other)
        cluster_lat = base_lat + rng.normal(0, 0.05)
        cluster_long = base_long + rng.normal(0, 0.08)
        lat = rng.normal(cluster_lat, 0.02, seg_size)
        long = rng.normal(cluster_long, 0.03, seg_size)
        
        # Clip to realistic ranges
        monthly_sales = np.clip(monthly_sales, 500_000, 60_000_000)
        sku_count = np.clip(sku_count, 10, 250).astype(int)
        visit_freq = np.clip(visit_freq, 0.5, 10)
        oos_rate = np.clip(oos_rate, 0.01, 0.60)
        avg_basket = np.clip(avg_basket, 10_000, 400_000)
        
        for i in range(seg_size):
            outlets.append([
                f'OUT_{len(outlets)+1:04d}',  # outlet_id
                lat[i],  # lat
                long[i],  # long
                outlet_type[i],  # outlet_type
                monthly_sales[i],  # monthly_sales_idr
                sku_count[i],  # sku_count
                visit_freq[i],  # visit_freq_per_month
                oos_rate[i],  # oos_rate
                avg_basket[i],  # avg_basket_idr
                seg_idx  # true_segment (hidden label for validation)
            ])
    
    # Convert to numpy array and shuffle
    outlets_array = np.array(outlets, dtype=object)
    shuffle_idx = rng.permutation(len(outlets_array))
    outlets_array = outlets_array[shuffle_idx]
    
    return outlets_array

def save_data(output_path='spg_outlets.csv'):
    """Generate and save the dataset."""
    data = generate_spg_outlets_data()
    
    headers = ['outlet_id', 'lat', 'long', 'outlet_type', 'monthly_sales_idr', 
               'sku_count', 'visit_freq_per_month', 'oos_rate', 'avg_basket_idr']
    headers_with_label = headers + ['true_segment']
    
    # Save without true_segment column (that's for validation only)
    with open(output_path, 'w') as f:
        f.write(','.join(headers) + '\n')
        for row in data:
            f.write(','.join(str(x) for x in row[:-1]) + '\n')
    
    # Also save version with labels for testing
    output_path_labels = output_path.replace('.csv', '_with_labels.csv')
    with open(output_path_labels, 'w') as f:
        f.write(','.join(headers_with_label) + '\n')
        for row in data:
            f.write(','.join(str(x) for x in row) + '\n')
    
    print(f"Generated {len(data)} outlets")
    print(f"Saved to: {output_path}")
    print(f"Saved to: {output_path_labels}")
    print(f"\nDataset preview (first 5 rows):")
    for i in range(min(5, len(data))):
        print(f"  {data[i][:5]}...")  # Show first few columns

if __name__ == "__main__":
    # Generate data in current directory
    output_path = Path(__file__).parent / 'spg_outlets.csv'
    save_data(str(output_path))
