# Principal Component Analysis (PCA)

**[🇬🇧 English](README.md) | 🇮🇩 Bahasa Indonesia**

**🏠 [Kembali ke Utama](../../README.md)** | **📁 [Unsupervised Learning](../README.md)** | **⬅️ [Sebelumnya: K-Means](../01_kmeans/README.md)**

## 🗺️ Akses Cepat

| File | Deskripsi | Cara Menjalankan |
|------|-------------|----------|
| [`implementation.py`](./implementation.py) | Implementasi from-scratch | `python3 implementation.py` |
| [`sklearn_example.py`](./sklearn_example.py) | Contoh scikit-learn | `python3 sklearn_example.py` |
| [`exercise.ipynb`](./exercise.ipynb) | Latihan interaktif | `jupyter lab exercise.ipynb` |
| [`datasets/`](./datasets/) | Dataset outlet retail SPG | - |

## 📖 Teori dan Konsep

### Apa itu PCA?

**Principal Component Analysis (PCA)** adalah teknik unsupervised learning untuk reduksi dimensi. PCA mentransformasi data ke sistem koordinat baru di mana sumbu (principal component) diurutkan berdasarkan besarnya variance yang dijelaskan. Hal ini memungkinkan Anda untuk:

1. **Memvisualisasikan** data berdimensi tinggi dalam 2D atau 3D
2. **Mengurangi noise** dengan hanya mempertahankan komponen bervariance tinggi
3. **Mempercepat** algoritma machine learning dengan mengurangi fitur
4. **Memahami** fitur mana yang paling berkontribusi terhadap variance

### Formulasi Matematika

Diberikan matriks data **X** dengan bentuk (n_samples, n_features):

1. **Pusatkan data:**
   ```
   X_centered = X - mean(X)
   ```

2. **Hitung matriks kovarians:**
   ```
   Cov = (1/(n-1)) × X_centered^T × X_centered
   ```

3. **Temukan eigenvalue λ dan eigenvector v:**
   ```
   Cov × v = λ × v
   ```

4. **Principal component** adalah eigenvector yang diurutkan berdasarkan eigenvalue (terbesar lebih dulu)

### PCA melalui SVD (Implementasi Kami)

Pendekatan yang lebih stabil secara numerik menggunakan **Singular Value Decomposition**:

```
X_centered = U × Σ × V^T
```

Keterangan:
- **U**: (n_samples, n_samples) left singular vector
- **Σ**: (min(n,p), min(n,p)) matriks diagonal dari singular value σ
- **V^T**: (n_features, n_features) right singular vector (ditranspos)

**Wawasan kunci:** Baris-baris dari V^T adalah principal component!

**Explained variance:**
```
explained_variance_i = σ_i² / (n_samples - 1)
```

**Explained variance ratio:**
```
explained_variance_ratio_i = explained_variance_i / Σ(explained_variance)
```

### Transformasi dan Rekonstruksi

**Transform (reduksi dimensi):**
```
X_transformed = X_centered × components^T
```
Bentuk: (n_samples, n_components)

**Inverse transform (rekonstruksi):**
```
X_reconstructed = X_transformed × components + mean
```
Bentuk: (n_samples, n_features)

**Catatan:** Rekonstruksi dengan komponen kurang dari seluruh komponen akan kehilangan informasi (sebagian variance).

## 🎯 Kapan Menggunakan PCA?

### ✅ Cocok untuk:
- Memvisualisasikan data berdimensi tinggi (reduksi ke 2D/3D)
- Menghapus fitur yang berkorelasi/redundan
- Pengurangan noise (pertahankan komponen bervariance tinggi)
- Mempercepat algoritma (lebih sedikit fitur)
- Pre-processing sebelum clustering atau klasifikasi
- Exploratory data analysis

### ❌ Tidak cocok untuk:
- Ketika Anda perlu menginterpretasikan fitur individual (PC adalah kombinasi)
- Ketika variance tidak sejalan dengan kepentingan
- Data sparse (PCA membuatnya menjadi dense)
- Data kategorikal (membutuhkan fitur numerik)
- Ketika hubungan non-linear mendominasi

## 📊 Konsep Kunci

### Explained Variance Ratio

Memberi tahu Anda seberapa banyak informasi (variance) yang ditangkap setiap komponen:

```python
explained_variance_ratio = [0.45, 0.25, 0.15, 0.10, 0.05]
cumulative = [0.45, 0.70, 0.85, 0.95, 1.00]
```

- PC1 menangkap 45% variance
- Dua PC pertama menangkap 70% variance
- Butuh 4 PC untuk 95% variance

**Aturan praktis:** Pertahankan komponen yang menjelaskan 90-95% cumulative variance.

### Ambiguitas Tanda

Komponen PCA hanya terdefinisi **hingga tanda** (arahnya dapat dibalik):

```
components = [v1, v2, v3, ...]  or  [-v1, -v2, -v3, ...]
```

Keduanya setara secara matematis! Saat membandingkan hasil PCA:
- Komponen dapat memiliki tanda yang berlawanan
- Gunakan `np.allclose(abs(comp1), abs(comp2))` atau korelasi
- Implementasi kami menyertakan helper `compare_pca_sign_invariance()`

### Feature Scaling SANGAT PENTING

PCA sensitif terhadap skala fitur karena memaksimalkan variance:

```python
# Tanpa scaling: fitur dengan rentang lebih besar akan mendominasi
X = [[1000000, 5],    # Sales (jutaan), Rating (1-5)
     [2000000, 4]]
# PCA akan didominasi oleh Sales!

# Selalu scale terlebih dahulu:
from sklearn.preprocessing import StandardScaler
X_scaled = StandardScaler().fit_transform(X)
```

## 🔍 Kelebihan dan Kekurangan

### Kelebihan:
- ✅ Mengurangi dimensi sambil mempertahankan variance
- ✅ Menghapus fitur yang berkorelasi secara otomatis
- ✅ Mempercepat algoritma hilir
- ✅ Sangat baik untuk visualisasi
- ✅ Deterministik (hasil sama setiap kali)
- ✅ Komputasi cepat melalui SVD

### Kekurangan:
- ❌ Principal component lebih sulit diinterpretasikan
- ❌ Mengasumsikan hubungan linear
- ❌ Sensitif terhadap outlier
- ❌ Memerlukan feature scaling
- ❌ Dapat kehilangan informasi jika komponen terlalu sedikit
- ❌ Ambiguitas tanda saat membandingkan hasil

## 🎯 Skenario Praktis: Memvisualisasikan Segmen Outlet

### 🏪 Analisis Outlet Retail FMCG Indonesia

**Dataset**: 500 outlet retail yang sama dari contoh K-Means

**Fitur asli (5D):**
- `monthly_sales_idr` - Penjualan bulanan dalam IDR
- `sku_count` - Jumlah SKU
- `visit_freq_per_month` - Frekuensi kunjungan
- `oos_rate` - Tingkat out-of-stock
- `avg_basket_idr` - Rata-rata ukuran basket

**Masalah:** Tidak dapat memvisualisasikan data 5D atau memahami fitur mana yang bervariasi bersama.

**Solusi:** Gunakan PCA untuk:
1. Mengurangi data 5D menjadi 2D untuk visualisasi
2. Memahami berapa banyak variance yang ditangkap 2D
3. Melihat fitur asli mana yang berkontribusi pada setiap PC
4. Memvisualisasikan cluster K-Means dalam ruang PCA 2D

**Jalankan contohnya:**
```bash
python3 sklearn_example.py
```

**Yang akan Anda pelajari:**
- Cara menginterpretasikan explained variance ratio
- Kapan proyeksi 2D sudah cukup (periksa cumulative variance)
- Cara memvisualisasikan cluster dalam dimensi yang direduksi
- Fitur asli mana yang berkontribusi pada setiap principal component

## 💡 Detail Implementasi Penting

### Parameter Penting

```python
PCA(
    n_components=None,    # Jumlah komponen (atau None untuk semua)
    whiten=False,        # Bagi dengan sqrt(explained_variance)
    svd_solver='auto'    # Algoritma SVD: 'auto', 'full', 'arpack', 'randomized'
)
```

### Memilih n_components

**Berdasarkan variance:**
```python
pca = PCA(n_components=0.95)  # Pertahankan 95% variance
```

**Berdasarkan jumlah:**
```python
pca = PCA(n_components=2)  # Pertahankan tepat 2 komponen
```

**Periksa setelah fitting:**
```python
pca = PCA()
pca.fit(X)
print(pca.explained_variance_ratio_.cumsum())
# Pilih n di mana cumulative ≥ 0.90
```

### Feature Scaling

```python
from sklearn.preprocessing import StandardScaler

# Selalu scale sebelum PCA!
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)
```

### Reconstruction Error

Mengukur kehilangan informasi akibat reduksi dimensi:

```python
# Transform dan rekonstruksi
X_pca = pca.transform(X_scaled)
X_reconstructed = pca.inverse_transform(X_pca)

# Hitung MSE
mse = np.mean((X_scaled - X_reconstructed) ** 2)
```

Dengan semua komponen: MSE ≈ 0 (rekonstruksi sempurna)

## 📚 Referensi

- [Scikit-learn PCA](https://scikit-learn.org/stable/modules/decomposition.html#pca)
- [Principal Component Analysis (Wikipedia)](https://en.wikipedia.org/wiki/Principal_component_analysis)
- [Singular Value Decomposition](https://en.wikipedia.org/wiki/Singular_value_decomposition)
- [Understanding PCA Visually](http://setosa.io/ev/principal-component-analysis/)

## 🎓 Progres Pembelajaran

### Prasyarat
- Dasar aljabar linear (vektor, matriks, perkalian matriks)
- Pemahaman tentang variance dan covariance
- Dasar-dasar NumPy

### Setelah PCA, jelajahi:

1. **[t-SNE](../05_tsne/)** - Reduksi dimensi non-linear untuk visualisasi
2. **Autoencoders** (`../../04_advanced_topics/02_deep_learning/`, direncanakan) - PCA non-linear dengan neural network
3. **LDA** (`../07_lda/`, direncanakan) - Linear Discriminant Analysis (versi supervised)
4. **Factor Analysis** (`../08_factor_analysis/`, direncanakan) - Mirip PCA tetapi memodelkan noise

### Menggabungkan dengan algoritma lain

```python
# PCA sebelum clustering
pca = PCA(n_components=0.95)
X_reduced = pca.fit_transform(X_scaled)
kmeans = KMeans(n_clusters=4)
labels = kmeans.fit_predict(X_reduced)

# PCA sebelum klasifikasi (training lebih cepat)
pca = PCA(n_components=50)
X_train_pca = pca.fit_transform(X_train)
X_test_pca = pca.transform(X_test)
clf.fit(X_train_pca, y_train)
```

---

## Ringkasan

**Principal Component Analysis (PCA)** adalah teknik unsupervised learning untuk reduksi dimensi. PCA mentransformasi data ke sistem koordinat baru di mana sumbu (principal components) diurutkan berdasarkan variance yang dijelaskan.

**Cara Kerja:**
1. Center data (kurangi mean)
2. Hitung SVD: X_centered = U × Σ × V^T
3. Principal components = baris dari V^T
4. Explained variance = σ²/(n-1)

**Transformasi:**
- **Transform:** X_transformed = X_centered × components^T (reduksi dimensi)
- **Inverse Transform:** X_reconstructed = X_transformed × components + mean

**Explained Variance Ratio:** Persentase informasi yang dipertahankan oleh setiap component. Contoh: [0.45, 0.25, 0.15, 0.10, 0.05] berarti PC1 menjelaskan 45% variance, PC1+PC2 menjelaskan 70%.

**Penting:** 
- SELALU lakukan feature scaling sebelum PCA!
- Sign ambiguity: components bisa bernilai positif atau negatif (keduanya valid)
- Gunakan cumulative variance ≥ 90-95% untuk memilih n_components

**Studi Kasus:** Dataset 500 outlet retail dengan 5 fitur (sales, SKU, visit freq, OOS rate, basket). Gunakan PCA untuk:
1. Reduksi dari 5D ke 2D untuk visualisasi
2. Cek berapa % variance yang dipertahankan 2 PC
3. Visualisasi hasil clustering K-Means di ruang 2D
4. Pahami fitur mana yang berkontribusi ke setiap PC

**Latihan:** Tentukan jumlah components yang cukup untuk capture 90% variance, lalu interpretasikan loading (kontribusi fitur asli) di setiap PC.

---

**➡️ Langkah Berikutnya**: Jelajahi **[t-SNE](../05_tsne/)** untuk reduksi dimensi non-linear!
