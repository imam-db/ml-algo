# DBSCAN (Density-Based Spatial Clustering of Applications with Noise)

## What is DBSCAN?

DBSCAN is algoritma clustering that different from K-Means. Alih-alih membentuk cluster bulat with ukuran seragam, **DBSCAN menemukan cluster based on kepadatan (density)** and can:

1. ✅ Menemukan cluster with **bentuk arbitrary** (not must bulat)
2. ✅ **Mendeteksi outlier** (noise points)
3. ✅ **Doesn't need to determine jumlah cluster** di awal
4. ❌ Sensitif terhadap parameter `eps` and `min_samples`

### Analogi SPG (Sales Promotion Girl/Guy)

Bayangkan Anda is koordinator SPG that memetakan lokasi toko di kota:

- **Cluster**: Area with many toko berdekatan (mall, kawasan perumahan, pusat bisnis)
- **Outlier**: Toko terisolasi di pinggir kota or area sepi
- **Core Point**: Toko that surrounded by many toko lain (≥ min_samples within radius eps)
- **Border Point**: Toko di pinggir cluster (dekat core point tapi not punya many tetangga)

DBSCAN membantu Anda mengidentifikasi:
- Zona kerja SPG based on kepadatan toko
- Toko mana that terisolasi and mungkin not efisien dikunjungi

---

## Konsep Utama

### 1. Parameter Kunci

#### `eps` (epsilon)
- **Radius lingkungan**: Jarak maksimal antara dua points agar dianggap tetangga
- In SPG context: Jarak (km) that dianggap "dekat"
- Terlalu kecil → Banyak noise
- Terlalu besar → Semua points jadi satu cluster

#### `min_samples`
- **Minimum jumlah tetangga**: Berapa many points within radius `eps` agar sebuah points dianggap core point
- Termasuk points itu sendiri
- In SPG context: Berapa toko minimal within radius `eps` agar area dianggap "padat"

### 2. Tipe Titik

```
Core Point (●): Punya ≥ min_samples tetangga within radius eps
├─ Pusat from cluster
└─ Digunakan for expand cluster

Border Point (○): Bukan core point, tapi ada within radius eps from core point
├─ Di pinggir cluster
└─ Tidak used for expand

Noise Point (×): Bukan core maupun border
├─ Outlier / anomaly
└─ Label = -1
```

### 3. Algoritma

```
1. For each points that belum di-visit:
   
   a. Hitung tetangga within radius eps
   
   b. Jika jumlah tetangga < min_samples:
      → Tandai as noise (sementara)
   
   c. Jika jumlah tetangga ≥ min_samples:
      → Core point! Buat cluster baru
      → Expand cluster menggunakan BFS:
         - Tambahkan semua tetangga ke cluster
         - Jika tetangga juga core point, expand from situ
         - Border points that tadinya noise → masuk cluster

2. Noise points that not pernah di-claim → tetap noise
```

---

## Formula Matematis

### Epsilon-Neighborhood

$$N_\varepsilon(p) = \{q \in D \mid \text{dist}(p, q) \leq \varepsilon\}$$

Semua points $q$ in dataset $D$ that jaraknya from $p$ ≤ $\varepsilon$.

### Core Point Condition

$$|N_\varepsilon(p)| \geq \text{minPts}$$

Titik $p$ is core point if has minimum `minPts` tetangga within radius $\varepsilon$.

### Directly Density-Reachable

Titik $q$ directly density-reachable from $p$ if:
1. $q \in N_\varepsilon(p)$, and
2. $p$ is core point

### Density-Reachable (Transitive)

$q$ density-reachable from $p$ if there is a chain: $p = p_1, p_2, \ldots, p_n = q$ di mana $p_{i+1}$ directly density-reachable from $p_i$.

### Density-Connected

$p$ and $q$ density-connected if there is a point $o$ such that $p$ and $q$ both density-reachable from $o$.

**Cluster** = semua points that density-connected satu sama lain + border points mereka.

---

## Cara Memilih Parameter

### Metode 1: K-Distance Graph

1. For each points, calculate distance to k-th neighbor terdekat (k = min_samples)
2. Sort jarak this secara descending
3. Plot grafik
4. Cari "elbow" (points di mana kurva mulai naik tajam)
5. Jarak at elbow → nilai `eps` that good

```python
from sklearn.neighbors import NearestNeighbors

k = 5  # min_samples that desired
nbrs = NearestNeighbors(n_neighbors=k).fit(X)
distances, indices = nbrs.kneighbors(X)

# Sort k-th nearest neighbor distances
k_dist = np.sort(distances[:, k-1], axis=0)[::-1]

plt.plot(k_dist)
plt.ylabel(f'{k}-th Nearest Neighbor Distance')
plt.xlabel('Points sorted by distance')
plt.title('K-Distance Graph (find the elbow)')
plt.show()
```

### Metode 2: Domain Knowledge

- **`eps`**: Berdasarkan pengetahuan domain about distance "dekat"
  - For store locations: 1-2 km mungkin masuk akal
  - For sensor data: tergantung unit pengukuran

- **`min_samples`**: Rule of thumb
  - `min_samples = 2 * dimensi` for kebanyakan kasus
  - Minimal 3-5 for data 2D
  - Larger if much noise

---

## Advantages and Disadvantages

### ✅ Advantages

1. **Don't need to determine jumlah cluster** di awal (different from K-Means)
2. **Menemukan cluster bentuk arbitrary** (not must bulat/convex)
3. **Robust terhadap outlier** - mendeteksi and mengabaikan noise
4. **Only needs 2 parameter** that relatif intuitif

### ❌ Disadvantages

1. **Sensitif terhadap parameter** - `eps` and `min_samples` must chosen with hati-hati
2. **Sulit for data with density bervariasi** - cluster with density different bisa sulit dideteksi with satu nilai `eps`
3. **Curse of dimensionality** - jarak Euclidean kurang bermakna di dimensi tinggi
4. **Kompleksitas tinggi** - O(n²) for naive implementation (bisa dipercepat with index structures)

---

## DBSCAN vs K-Means

| Aspek | DBSCAN | K-Means |
|-------|--------|---------|
| **Jumlah cluster** | Otomatis ditemukan | Harus ditentukan (k) |
| **Bentuk cluster** | Arbitrary | Bulat (spherical) |
| **Outlier handling** | Deteksi as noise | Semua points masuk cluster |
| **Parameter** | eps, min_samples | k, max_iter |
| **Kompleksitas** | O(n²) or O(n log n) | O(nkdi) |
| **Good for** | Data with noise, bentuk kompleks | Data terdistribusi normal, bulat |

---

## Implementation from Nol

See `implementation.py` for kode lengkap. Fungsi utama:

```python
from implementation import DBSCAN, generate_clustering_data

# Generate data toko SPG
X = generate_clustering_data(n_samples=300, random_state=42)

# Fit DBSCAN
dbscan = DBSCAN(eps=1.5, min_samples=5)
labels = dbscan.fit_predict(X)

print(f"Clusters found: {dbscan.n_clusters_}")
print(f"Noise points: {np.sum(labels == -1)}")
```

---

## Contoh Kasus: Segmentasi Toko SPG

```python
# Data: koordinat 300 toko di kota
X = generate_clustering_data(n_samples=300, random_state=42)

# DBSCAN with eps=1.5 km, min_samples=5
dbscan = DBSCAN(eps=1.5, min_samples=5)
dbscan.fit(X)

# Analisis hasil
print(f"Zona kerja ditemukan: {dbscan.n_clusters_}")
print(f"Toko terisolasi: {np.sum(dbscan.labels_ == -1)}")

for cluster_id in range(dbscan.n_clusters_):
    n_stores = np.sum(dbscan.labels_ == cluster_id)
    print(f"Zona {cluster_id}: {n_stores} toko")
```

**Interpretasi**:
- Each cluster = zona kerja SPG that efisien
- Noise points = toko terisolasi that mungkin perlu strategi khusus
- Core points = toko di pusat zona (prioritas tinggi)

---

## Tuning Parameter

### Eksperimen with Grid Search

```python
import numpy as np

eps_values = [0.5, 1.0, 1.5, 2.0, 2.5]
min_samples_values = [3, 5, 10, 15]

best_params = None
best_score = -1

for eps in eps_values:
    for min_samples in min_samples_values:
        dbscan = DBSCAN(eps=eps, min_samples=min_samples)
        labels = dbscan.fit_predict(X)
        
        # Evaluasi with silhouette score (jika ada cluster)
        if dbscan.n_clusters_ > 0 and np.sum(labels != -1) > 1:
            from sklearn.metrics import silhouette_score
            mask = labels != -1  # Exclude noise
            if len(set(labels[mask])) > 1:
                score = silhouette_score(X[mask], labels[mask])
                print(f"eps={eps}, min_samples={min_samples}: "
                      f"{dbscan.n_clusters_} clusters, score={score:.3f}")
                
                if score > best_score:
                    best_score = score
                    best_params = (eps, min_samples)

print(f"\nBest params: eps={best_params[0]}, min_samples={best_params[1]}")
```

---

## Referensi

- **Paper asli**: Ester, M., Kriegel, H. P., Sander, J., & Xu, X. (1996). "A density-based algorithm for discovering clusters in large spatial databases with noise." In KDD (Vol. 96, No. 34, pp. 226-231).
- **Sklearn**: [`sklearn.cluster.DBSCAN`](https://scikit-learn.org/stable/modules/generated/sklearn.cluster.DBSCAN.html)

---

## File Terkait

- `implementation.py` - Implementation DBSCAN from nol
- `sklearn_example.py` - Perbandingan with sklearn
- `exercise.ipynb` - Interactive exercises
- `test_dbscan.py` - Unit tests

**Selamat learns! 🎓**