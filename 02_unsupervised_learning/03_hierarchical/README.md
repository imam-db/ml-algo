# Hierarchical Clustering (Agglomerative)

## What is Hierarchical Clustering?

Hierarchical clustering membangun **hirarki cluster** in the form tree (dendrogram), bukan langsung divides data to a number of clusters tetap. Ada dua pendekatan:

1. **Agglomerative (bottom-up)** ⬆️ - Mulai with each points as cluster, merge secara iteratif
2. **Divisive (top-down)** ⬇️ - Mulai with satu cluster besar, split secara iteratif

Implementation this focuses at **agglomerative**, that more common used.

### Analogi SPG (Sales Promotion Girl/Guy)

Bayangkan Anda mengelompokkan outlet SPG di kota with pendekatan hirarki:

1. **Bottom level**: Each outlet is "cluster" sendiri
2. **Merge bertahap**: Outlet terdekat digabung → menjadi sub-district
3. **Merge lagi**: Sub-districts digabung → menjadi district
4. **Hingga**: Semua outlet in one city

**Keunggulan**: Anda bisa **memotong pohon hirarki di level any** for get jumlah cluster that desired (2 district? 5 sub-district? 10 zones?).

---

## Konsep Utama

### 1. Algoritma Agglomerative

```
1. Start: Setiap sampel = satu cluster (n clusters)

2. Repeat until n_clusters dicapai:
   a. Hitung jarak antara semua pasangan cluster
   b. Merge dua cluster terdekat
   c. Update distance matrix

3. Stop: Ketika jumlah cluster = n_clusters
```

### 2. Linkage Criteria

**Bagaimana menghitung jarak antara dua cluster?**

#### Single Linkage (Nearest Point)

$$d_{\text{single}}(C_i, C_j) = \min_{x \in C_i, y \in C_j} \text{dist}(x, y)$$

- Jarak = jarak **terpendek** antara semua pasangan points
- **Advantages**: Menemukan cluster elongated/chain-like
- **Disadvantages**: Sensitif terhadap noise (chaining effect)

```
Cluster A: ●---●
                   ← min distance
Cluster B:     ●---●
```

#### Complete Linkage (Farthest Point)

$$d_{\text{complete}}(C_i, C_j) = \max_{x \in C_i, y \in C_j} \text{dist}(x, y)$$

- Jarak = jarak **terjauh** antara semua pasangan points
- **Advantages**: Menghasilkan cluster compact/spherical
- **Disadvantages**: Sensitif terhadap outlier

```
Cluster A: ●---●
           ↑       ↑
           └───────┘ max distance
Cluster B: ●---●
```

#### Average Linkage

$$d_{\text{average}}(C_i, C_j) = \frac{1}{|C_i| \cdot |C_j|} \sum_{x \in C_i} \sum_{y \in C_j} \text{dist}(x, y)$$

- Jarak = **average** semua jarak antar-points
- **Balanced**: Not too sensitive to noise or outlier
- **Umum used**: Trade-off that good

### 3. Dendrogram

**Dendrogram** = visualisasi pohon hirarki.

```
      ┌─────────┴────────┐
      │                  │
   ┌──┴──┐            ┌──┴──┐
   │     │            │     │
 ┌─┴─┐ ┌─┴─┐        ┌─┴─┐ ┌─┴─┐
 1   2 3   4        5   6 7   8
```

- **Sumbu Y**: Jarak/dissimilarity saat merge
- **Sumbu X**: Sampel individual
- **Memotong horizontal**: Tentukan jumlah cluster

**Cara membaca**:
- Dua branch that merge rendah = mirip
- Merge tinggi = different
- Cut dendrogram di ketinggian tertentu = tentukan n_clusters

---

## Advantages and Disadvantages

### ✅ Advantages

1. **Don't need to determine k upfront** - Dendrogram menunjukkan struktur hirarki lengkap
2. **Interpretable** - Visualisasi hirarki membantu pemahaman struktur data
3. **Fleksibel** - Bisa cut di level any for get jumlah cluster different
4. **Deterministic** - No randomness, hasil selalu sama

### ❌ Disadvantages

1. **Kompleksitas tinggi** - O(n³) for naive implementation, O(n² log n) for optimized
2. **Not scalable** - Lambat for dataset besar (> 10,000 sampel)
3. **Sensitif terhadap noise** (especially single linkage)
4. **Cannot undo merge** - Once merge, not bisa dipisah lagi (greedy)

---

## Linkage Method Comparison

| Linkage | Jarak Cluster | Bentuk Cluster | Sensitif Noise | Sensitif Outlier |
|---------|---------------|----------------|----------------|------------------|
| **Single** | MIN pasangan | Elongated/chain | ⚠️ Tinggi | ✅ Rendah |
| **Complete** | MAX pasangan | Compact/spherical | ✅ Rendah | ⚠️ Tinggi |
| **Average** | MEAN pasangan | Balanced | ✅ Medium | ✅ Medium |

**Rekomendasi**:
- **Single**: Data with cluster bentuk arbitrary/elongated
- **Complete**: Data with cluster compact, hindari chaining
- **Average**: Default choice, balanced trade-off

---

## Hierarchical vs K-Means

| Aspek | Hierarchical | K-Means |
|-------|--------------|---------|
| **K di awal?** | Don't need (can cut later) | Harus tentukan |
| **Dendrogram** | Ya, visualisasi hirarki | No |
| **Kompleksitas** | O(n² log n) - O(n³) | O(nkdi) |
| **Scalability** | Buruk (< 10k samples) | Baik (millions) |
| **Deterministic** | Ya | Tidak (random init) |
| **Bentuk cluster** | Tergantung linkage | Spherical |
| **Good for** | Analisis eksploratori, dataset kecil | Produksi, dataset besar |

---

## Cara Memilih Jumlah Cluster

### 1. Visual Inspection from Dendrogram

Cari **gap terbesar** at sumbu Y (tinggi merge).

```python
# Plot dendrogram
from scipy.cluster.hierarchy import dendrogram, linkage

Z = linkage(X, method='average')
plt.figure(figsize=(12, 6))
dendrogram(Z)
plt.axhline(y=threshold, color='r', linestyle='--')
plt.show()
```

Cut di threshold that memaksimalkan gap → jumlah cluster optimal.

### 2. Silhouette Score

Coba berbagai n_clusters, pilih that maksimalkan silhouette score.

```python
from sklearn.metrics import silhouette_score

scores = []
for n in range(2, 11):
    hc = AgglomerativeClustering(n_clusters=n)
    labels = hc.fit_predict(X)
    scores.append(silhouette_score(X, labels))

best_n = np.argmax(scores) + 2
```

### 3. Domain Knowledge

Gunakan pengetahuan bisnis:
- Untuk SPG: Berapa many tim/zona that bisa Anda manage?
- Untuk segmentasi pelanggan: Berapa kategori that masuk akal?

---

## Implementation from Nol

See `implementation.py` for kode lengkap. Fungsi utama:

```python
from implementation import AgglomerativeClustering, generate_hierarchical_data

# Generate data outlet SPG
X = generate_hierarchical_data(n_samples=150, random_state=42)

# Fit hierarchical clustering
hc = AgglomerativeClustering(n_clusters=4, linkage='average')
labels = hc.fit_predict(X)

print(f"Clusters: {hc.n_clusters_}")
for i in range(hc.n_clusters_):
    print(f"Cluster {i}: {np.sum(labels == i)} outlets")
```

---

## Contoh Kasus: Segmentasi Outlet SPG

```python
# Data: koordinat 150 outlet
X = generate_hierarchical_data(n_samples=150, random_state=42)

# Hierarchical clustering with 4 zona
hc = AgglomerativeClustering(n_clusters=4, linkage='average')
hc.fit(X)

# Visualisasi
from implementation import plot_clustering_results, plot_dendrogram

# Clustering result
fig = plot_clustering_results(X, hc.labels_, "SPG Outlet Zones")
plt.show()

# Dendrogram (hirarki merge)
fig = plot_dendrogram(hc, "Outlet Hierarchy")
plt.show()
```

**Interpretasi**:
- Dendrogram menunjukkan bagaimana outlet dikelompokkan secara bertahap
- Setiap cluster = zona kerja SPG
- Anda can cut di level different for jumlah zona different (2, 3, 4, ...)

---

## Perbandingan Linkage Methods

```python
import matplotlib.pyplot as plt
from implementation import AgglomerativeClustering

linkages = ['single', 'complete', 'average']
fig, axes = plt.subplots(1, 3, figsize=(18, 5))

for ax, linkage in zip(axes, linkages):
    hc = AgglomerativeClustering(n_clusters=4, linkage=linkage)
    labels = hc.fit_predict(X)
    
    for i in range(4):
        mask = labels == i
        ax.scatter(X[mask, 0], X[mask, 1], s=50, alpha=0.7)
    
    ax.set_title(f'{linkage.capitalize()} Linkage', fontweight='bold')
    ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()
```

**Observasi**:
- **Single**: Cluster bisa elongated/chain
- **Complete**: Cluster more compact
- **Average**: Balanced

---

## Advanced: Cutting Dendrogram di Berbagai Level

```python
from scipy.cluster.hierarchy import fcluster, linkage

# Buat linkage matrix
Z = linkage(X, method='average')

# Cut di berbagai threshold
for threshold in [2.0, 3.0, 4.0]:
    labels = fcluster(Z, threshold, criterion='distance')
    n_clusters = len(np.unique(labels))
    print(f"Threshold {threshold}: {n_clusters} clusters")
```

Ini memberi fleksibilitas: **satu model, many pilihan jumlah cluster**.

---

## Kapan Gunakan Hierarchical Clustering?

### ✅ Gunakan ketika:

- Dataset kecil hingga medium (< 10,000 sampel)
- Perlu **eksplorasi struktur hirarki** data
- Tidak yakin berapa jumlah cluster optimal
- Interpretability penting (dendrogram membantu komunikasi)
- Data memiliki **struktur hirarki natural** (misalnya: produk → kategori → sub-kategori)

### ❌ Hindari ketika:

- Dataset sangat besar (> 100,000 sampel) - gunakan K-Means or Mini-Batch K-Means
- Butuh prediksi cepat di production - hierarchical clustering not efisien for inference
- Cluster must diupdate secara online/incremental

---

## Referensi

- **Sklearn**: [`sklearn.cluster.AgglomerativeClustering`](https://scikit-learn.org/stable/modules/generated/sklearn.cluster.AgglomerativeClustering.html)
- **Scipy**: [`scipy.cluster.hierarchy`](https://docs.scipy.org/doc/scipy/reference/cluster.hierarchy.html)

---

## File Terkait

- `implementation.py` - Implementation hierarchical clustering from nol
- `sklearn_example.py` - Perbandingan with sklearn
- `exercise.ipynb` - Interactive exercises with dendrogram
- `test_hierarchical.py` - Unit tests

**Selamat learns! 🎓**