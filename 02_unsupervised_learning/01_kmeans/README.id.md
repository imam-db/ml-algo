# K-Means Clustering

**[🇬🇧 English](README.md) | 🇮🇩 Bahasa Indonesia**

**🏠 [Kembali ke Utama](../../README.md)** | **📁 [Unsupervised Learning](../README.md)** | **➡️ [Berikutnya: PCA](../04_pca/README.md)**

## 🗺️ Akses Cepat

| File | Deskripsi | Cara Menjalankan |
|------|-------------|----------|
| [`implementation.py`](./implementation.py) | Implementasi from-scratch | `python3 implementation.py` |
| [`sklearn_example.py`](./sklearn_example.py) | Contoh scikit-learn | `python3 sklearn_example.py` |
| [`exercise.ipynb`](./exercise.ipynb) | Latihan interaktif | `jupyter lab exercise.ipynb` |
| [`datasets/`](./datasets/) | Dataset outlet retail SPG | - |

## 📖 Teori dan Konsep

### Apa itu K-Means Clustering?

**K-Means** adalah algoritma unsupervised learning yang mempartisi data menjadi K cluster yang berbeda dan tidak saling tumpang tindih. Setiap titik data termasuk dalam cluster dengan mean (centroid) terdekat. Algoritma ini merupakan salah satu algoritma clustering yang paling sederhana dan paling banyak digunakan.

### Formulasi Matematika

**Tujuan:** Meminimalkan jumlah kuadrat jarak di dalam cluster (inertia):

```
J = Σ(i=1 to K) Σ(x ∈ Ci) ||x - μi||²
```

Keterangan:
- `K` = jumlah cluster
- `Ci` = himpunan titik di cluster i
- `μi` = centroid (mean) dari cluster i
- `||x - μi||²` = kuadrat jarak Euclidean

### Langkah-langkah Algoritma

1. **Inisialisasi** K centroid (secara acak atau menggunakan k-means++)
2. **Langkah assignment:** Tetapkan setiap titik ke centroid terdekat
   ```
   ci = argmin(k) ||xi - μk||²
   ```
3. **Langkah update:** Hitung ulang centroid sebagai mean dari titik-titik yang ditetapkan
   ```
   μk = (1/|Ck|) Σ(x ∈ Ck) x
   ```
4. **Ulangi** langkah 2-3 hingga konvergen (centroid tidak berubah atau perubahannya di bawah toleransi)

### Inisialisasi K-Means++

K-means++ memperbaiki pemilihan centroid awal:

1. Pilih centroid pertama secara acak seragam dari titik-titik data
2. Untuk setiap centroid berikutnya:
   - Hitung jarak D(x) dari setiap titik x ke centroid terdekat yang sudah ada
   - Pilih centroid berikutnya dengan probabilitas yang proporsional terhadap D(x)²
3. Lanjutkan dengan algoritma k-means standar

Inisialisasi ini membuat konvergensi lebih cepat dan sering menghasilkan hasil yang lebih baik.

### Konvergensi

K-Means dijamin akan konvergen, tetapi:
- Dapat konvergen ke optimum lokal (bukan global)
- Solusinya bergantung pada centroid awal
- **Solusi:** Jalankan beberapa kali dengan inisialisasi berbeda (`n_init`)

## 🎯 Kapan Menggunakan K-Means?

### ✅ Cocok untuk:
- Segmentasi pelanggan
- Kompresi gambar
- Clustering dokumen
- Deteksi anomali (outlier yang jauh dari centroid)
- Pre-processing untuk algoritma lain
- Ketika jumlah cluster diketahui atau dapat diperkirakan

### ❌ Tidak cocok untuk:
- Bentuk cluster yang tidak bulat (non-spherical)
- Cluster dengan ukuran atau kepadatan yang sangat berbeda
- Banyak outlier (sensitif terhadap outlier)
- Jumlah cluster yang tidak diketahui atau bervariasi
- Cluster yang saling tumpang tindih

## 📊 Evaluasi Model

### Elbow Method
Plot inertia terhadap jumlah cluster K. Cari "siku" tempat penurunan inertia mulai melambat.

```
Inertia(K) = Σ(i=1 to K) Σ(x ∈ Ci) ||x - μi||²
```

### Silhouette Score
Mengukur seberapa mirip suatu titik dengan cluster-nya sendiri dibandingkan dengan cluster lain:

```
s(i) = (b(i) - a(i)) / max{a(i), b(i)}
```

Keterangan:
- `a(i)` = jarak rata-rata dari titik i ke titik lain di cluster yang sama
- `b(i)` = jarak rata-rata dari titik i ke cluster lain terdekat

**Rentang skor:** -1 sampai 1 (semakin tinggi semakin baik)
- `1.0`: Cocok dengan cluster sendiri, jauh dari tetangga
- `0.0`: Berada di atau sangat dekat dengan decision boundary
- `-1.0`: Ditetapkan ke cluster yang salah

## 🔍 Kelebihan dan Kekurangan

### Kelebihan:
- ✅ Sederhana dan mudah dipahami
- ✅ Cepat dan skalabel (O(n·K·i·d) dengan n=sampel, i=iterasi, d=dimensi)
- ✅ Bekerja baik ketika cluster berbentuk bulat dan berukuran serupa
- ✅ Dijamin konvergen

### Kekurangan:
- ❌ Harus menentukan K terlebih dahulu
- ❌ Sensitif terhadap centroid awal
- ❌ Sensitif terhadap outlier
- ❌ Mengasumsikan cluster berbentuk bulat dengan ukuran serupa
- ❌ Dapat konvergen ke optimum lokal

## 🎯 Skenario Praktis: Perencanaan Kunjungan SPG

### 🏪 Segmentasi Outlet Retail FMCG Indonesia

**Dataset**: 500 outlet retail di Denpasar/Bali yang dikunjungi SPG (Sales Promotion Girls/Guys)

**Fitur:**
- `monthly_sales_idr` - Penjualan bulanan dalam Rupiah
- `sku_count` - Jumlah SKU (Stock Keeping Unit) yang dijual
- `visit_freq_per_month` - Seberapa sering outlet perlu dikunjungi
- `oos_rate` - Tingkat out-of-stock (0-1)
- `avg_basket_idr` - Rata-rata ukuran basket dalam IDR

**Tujuan Bisnis:** Seorang supervisor ingin mengelompokkan outlet ke dalam segmen untuk mengoptimalkan rute dan prioritas kunjungan SPG.

**Tipe outlet:**
- **Warung** - Toko kecil tradisional
- **Minimarket** - Toko swalayan kecil
- **Grosir** - Pedagang grosir

**Jalankan contohnya:**
```bash
python3 sklearn_example.py
```

**Yang akan Anda pelajari:**
- Mengapa feature scaling sangat penting untuk K-Means
- Cara memilih K menggunakan elbow method dan silhouette score
- Cara menginterpretasikan cluster untuk keputusan bisnis
- Mengubah hasil clustering menjadi rencana kunjungan yang dapat ditindaklanjuti

### Segmen yang Diharapkan

Dataset berisi 4 segmen tersembunyi:
1. **Warung kecil** - Penjualan rendah, sedikit SKU, OOS rate tinggi
2. **Minimarket aktif** - Penjualan sedang-tinggi, variasi bagus
3. **Grosir bervolume tinggi** - Penjualan dan jumlah SKU sangat tinggi
4. **Outlet yang kesulitan** - Penjualan rendah, SKU sedang, OOS tinggi

Tugas Anda: Gunakan K-Means untuk menemukan segmen-segmen ini dan merekomendasikan strategi kunjungan!

## 💡 Detail Implementasi Penting

### Parameter Penting

```python
KMeans(
    n_clusters=8,        # Jumlah cluster
    init='k-means++',    # Inisialisasi: 'k-means++' atau 'random'
    n_init=10,          # Jumlah inisialisasi berbeda
    max_iter=300,       # Jumlah iterasi maksimum
    tol=1e-4,           # Toleransi untuk konvergensi
    random_state=42     # Untuk reproduksibilitas
)
```

### Feature Scaling SANGAT PENTING

K-Means menggunakan jarak Euclidean, sehingga fitur dengan rentang lebih besar akan mendominasi:

```python
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)  # Mean nol, varians satu
```

**Contoh:** Tanpa scaling, `monthly_sales_idr` (dalam jutaan) akan mendominasi `oos_rate` (0-1).

### Random State dan Determinisme

Untuk reproduksibilitas:
```python
kmeans = KMeans(n_clusters=3, random_state=42)
```

Dalam implementasi kami:
```python
self.rng_ = np.random.default_rng(random_state)
```

## 📚 Referensi

- [Scikit-learn K-Means](https://scikit-learn.org/stable/modules/clustering.html#k-means)
- [k-means++: The Advantages of Careful Seeding](http://ilpubs.stanford.edu:8090/778/1/2006-13.pdf)
- [Silhouette Coefficient](https://en.wikipedia.org/wiki/Silhouette_(clustering))

## 🎓 Progres Pembelajaran

### Prasyarat
- Pengetahuan dasar NumPy
- Pemahaman tentang jarak Euclidean
- Statistik dasar (mean, variance)

### Setelah K-Means, jelajahi:

1. **[Hierarchical Clustering](../03_hierarchical/)** - Tidak perlu menentukan K
2. **[DBSCAN](../02_dbscan/)** - Menemukan cluster dengan bentuk arbitrer
3. **[PCA](../04_pca/)** - Visualisasikan cluster berdimensi tinggi dalam 2D
4. **[Gaussian Mixture Models](../../04_advanced_topics/)** - Soft clustering

---

## Ringkasan

**K-Means Clustering** adalah algoritma unsupervised learning yang membagi data menjadi K kelompok (cluster) yang tidak tumpang tindih. Setiap titik data ditempatkan ke cluster dengan centroid (rata-rata) terdekat.

**Cara Kerja:**
1. Pilih K centroid awal (random atau k-means++)
2. Assign setiap titik ke centroid terdekat
3. Update centroid sebagai rata-rata dari titik yang di-assign
4. Ulangi sampai konvergen

**Metrik Evaluasi:**
- **Inertia:** Jumlah jarak kuadrat dari setiap titik ke centroid cluster-nya (semakin kecil semakin baik)
- **Elbow Method:** Plot inertia vs K, cari "siku" di mana penurunan melambat
- **Silhouette Score:** Ukur seberapa baik titik cocok dengan cluster-nya (-1 sampai 1, semakin tinggi semakin baik)

**Penting:** Selalu lakukan feature scaling sebelum K-Means karena algoritma ini menggunakan jarak Euclidean!

**Studi Kasus:** Dataset berisi 500 outlet retail (warung, minimarket, grosir) di Denpasar/Bali yang dikunjungi SPG. Supervisor ingin mengelompokkan outlet untuk mengoptimalkan jadwal kunjungan berdasarkan sales, jumlah SKU, frekuensi kunjungan, OOS rate, dan rata-rata transaksi.

**Latihan:** Gunakan elbow method dan silhouette score untuk memilih K optimal, lalu interpretasikan cluster menjadi strategi kunjungan (prioritas tinggi/sedang/rendah).

---

**➡️ Langkah Berikutnya**: Lanjutkan ke **[PCA](../04_pca/)** untuk memvisualisasikan cluster dalam 2D!
