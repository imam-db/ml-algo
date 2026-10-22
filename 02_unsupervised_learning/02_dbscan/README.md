# DBSCAN (Density-Based Spatial Clustering of Applications with Noise)

**🇬🇧 English | [🇮🇩 Bahasa Indonesia](README.id.md)**

## Apa itu DBSCAN?

DBSCAN adalah algoritma clustering yang berbeda dari K-Means. Alih-alih membentuk cluster bulat dengan ukuran seragam, **DBSCAN menemukan cluster berdasarkan kepadatan (density)** dan dapat:

1. ✅ Menemukan cluster dengan **bentuk arbitrary** (tidak harus bulat)
2. ✅ **Mendeteksi outlier** (noise points)
3. ✅ **Tidak perlu menentukan jumlah cluster** di awal
4. ❌ Sensitif terhadap parameter `eps` dan `min_samples`

### Analogi SPG (Sales Promotion Girl/Guy)

Bayangkan Anda adalah koordinator SPG yang memetakan lokasi toko di kota:

- **Cluster**: Area dengan banyak toko berdekatan (mall, kawasan perumahan, pusat bisnis)
- **Outlier**: Toko terisolasi di pinggir kota atau area sepi
- **Core Point**: Toko yang dikelilingi banyak toko lain (≥ min_samples dalam radius eps)
- **Border Point**: Toko di pinggir cluster (dekat core point tapi tidak punya banyak tetangga)

DBSCAN membantu Anda mengidentifikasi:
- Zona kerja SPG berdasarkan kepadatan toko
- Toko mana yang terisolasi dan mungkin tidak efisien dikunjungi

---

## Konsep Utama

### 1. Parameter Kunci

#### `eps` (epsilon)
- **Radius lingkungan**: Jarak maksimal antara dua titik agar dianggap tetangga
- Dalam konteks SPG: Jarak (km) yang dianggap "dekat"
- Terlalu kecil → Banyak noise
- Terlalu besar → Semua titik jadi satu cluster

#### `min_samples`
- **Minimum jumlah tetangga**: Berapa banyak titik dalam radius `eps` agar sebuah titik dianggap core point
- Termasuk titik itu sendiri
- Dalam konteks SPG: Berapa toko minimal dalam radius `eps` agar area dianggap "padat"

### 2. Tipe Titik

```
Core Point (●): Punya ≥ min_samples tetangga dalam radius eps
├─ Pusat dari cluster
└─ Digunakan untuk expand cluster

Border Point (○): Bukan core point, tapi ada dalam radius eps dari core point
├─ Di pinggir cluster
└─ Tidak digunakan untuk expand

Noise Point (×): Bukan core maupun border
├─ Outlier / anomaly
└─ Label = -1
```

### 3. Algoritma

```
1. Untuk setiap titik yang belum di-visit:
   
   a. Hitung tetangga dalam radius eps
   
   b. Jika jumlah tetangga < min_samples:
      → Tandai sebagai noise (sementara)
   
   c. Jika jumlah tetangga ≥ min_samples:
      → Core point! Buat cluster baru
      → Expand cluster menggunakan BFS:
         - Tambahkan semua tetangga ke cluster
         - Jika tetangga juga core point, expand dari situ
         - Border points yang tadinya noise → masuk cluster

2. Noise points yang tidak pernah di-claim → tetap noise
```

---

## Formula Matematis

### Epsilon-Neighborhood

$$N_\varepsilon(p) = \{q \in D \mid \text{dist}(p, q) \leq \varepsilon\}$$

Semua titik $q$ dalam dataset $D$ yang jaraknya dari $p$ ≤ $\varepsilon$.

### Core Point Condition

$$|N_\varepsilon(p)| \geq \text{minPts}$$

Titik $p$ adalah core point jika punya minimal `minPts` tetangga dalam radius $\varepsilon$.

### Directly Density-Reachable

Titik $q$ directly density-reachable dari $p$ jika:
1. $q \in N_\varepsilon(p)$, dan
2. $p$ adalah core point

### Density-Reachable (Transitive)

$q$ density-reachable dari $p$ jika ada chain: $p = p_1, p_2, \ldots, p_n = q$ di mana $p_{i+1}$ directly density-reachable dari $p_i$.

### Density-Connected

$p$ dan $q$ density-connected jika ada titik $o$ sehingga $p$ dan $q$ keduanya density-reachable dari $o$.

**Cluster** = semua titik yang density-connected satu sama lain + border points mereka.

---

## Cara Memilih Parameter

### Metode 1: K-Distance Graph

1. Untuk setiap titik, hitung jarak ke tetangga ke-k terdekat (k = min_samples)
2. Sort jarak ini secara descending
3. Plot grafik
4. Cari "elbow" (titik di mana kurva mulai naik tajam)
5. Jarak pada elbow → nilai `eps` yang baik

```python
from sklearn.neighbors import NearestNeighbors

k = 5  # min_samples yang diinginkan
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

- **`eps`**: Berdasarkan pengetahuan domain tentang jarak "dekat"
  - Untuk lokasi toko: 1-2 km mungkin masuk akal
  - Untuk sensor data: tergantung unit pengukuran

- **`min_samples`**: Rule of thumb
  - `min_samples = 2 * dimensi` untuk kebanyakan kasus
  - Minimal 3-5 untuk data 2D
  - Lebih besar jika banyak noise

---

## Kelebihan dan Kekurangan

### ✅ Kelebihan

1. **Tidak perlu tentukan jumlah cluster** di awal (berbeda dari K-Means)
2. **Menemukan cluster bentuk arbitrary** (tidak harus bulat/convex)
3. **Robust terhadap outlier** - mendeteksi dan mengabaikan noise
4. **Hanya butuh 2 parameter** yang relatif intuitif

### ❌ Kekurangan

1. **Sensitif terhadap parameter** - `eps` dan `min_samples` harus dipilih dengan hati-hati
2. **Sulit untuk data dengan density bervariasi** - cluster dengan density berbeda bisa sulit dideteksi dengan satu nilai `eps`
3. **Curse of dimensionality** - jarak Euclidean kurang bermakna di dimensi tinggi
4. **Kompleksitas tinggi** - O(n²) untuk naive implementation (bisa dipercepat dengan index structures)

---

## DBSCAN vs K-Means

| Aspek | DBSCAN | K-Means |
|-------|--------|---------|
| **Jumlah cluster** | Otomatis ditemukan | Harus ditentukan (k) |
| **Bentuk cluster** | Arbitrary | Bulat (spherical) |
| **Outlier handling** | Deteksi sebagai noise | Semua titik masuk cluster |
| **Parameter** | eps, min_samples | k, max_iter |
| **Kompleksitas** | O(n²) atau O(n log n) | O(nkdi) |
| **Cocok untuk** | Data dengan noise, bentuk kompleks | Data terdistribusi normal, bulat |

---

## Implementasi dari Nol

Lihat `implementation.py` untuk kode lengkap. Fungsi utama:

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

# DBSCAN dengan eps=1.5 km, min_samples=5
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
- Setiap cluster = zona kerja SPG yang efisien
- Noise points = toko terisolasi yang mungkin perlu strategi khusus
- Core points = toko di pusat zona (prioritas tinggi)

---

## Tuning Parameter

### Eksperimen dengan Grid Search

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
        
        # Evaluasi dengan silhouette score (jika ada cluster)
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

- `implementation.py` - Implementasi DBSCAN dari nol
- `sklearn_example.py` - Perbandingan dengan sklearn
- `exercise.ipynb` - Latihan interaktif
- `test_dbscan.py` - Unit tests

**Selamat belajar! 🎓**
