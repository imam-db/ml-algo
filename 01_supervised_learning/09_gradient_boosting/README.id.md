# Gradient Boosting

🇮🇩 Bahasa Indonesia | [🇬🇧 Read in English](README.md)

🏠 Kembali ke Indeks Supervised: `../README.md`

Gradient Boosting adalah teknik ensemble learning yang powerful yang membangun model secara berurutan, di mana setiap model baru memperbaiki kesalahan yang dibuat oleh model sebelumnya. Ini adalah salah satu algoritma machine learning paling efektif untuk tugas regresi dan klasifikasi.

## 📚 Daftar Isi
- [Teori & Konsep](#-teori--konsep)
- [Fondasi Matematis](#-fondasi-matematis)
- [Cara Kerja Gradient Boosting](#-cara-kerja-gradient-boosting)
- [Trade-off Bias-Variance](#-trade-off-bias-variance)
- [Learning Rate vs Jumlah Tree](#-learning-rate-vs-jumlah-tree)
- [Keuntungan & Kerugian](#-keuntungan--kerugian)
- [Kapan Menggunakan](#-kapan-menggunakan)
- [File dalam Direktori Ini](#-file-dalam-direktori-ini)
- [Algoritma Terkait](#-algoritma-terkait)

## 🧠 Teori & Konsep

### Apa itu Gradient Boosting?

Gradient Boosting adalah metode ensemble yang menggabungkan beberapa weak learner (biasanya decision tree yang dangkal) menjadi model prediktif yang kuat. Berbeda dengan Random Forest yang membangun tree secara independen dan paralel, Gradient Boosting membangun tree secara berurutan, di mana setiap tree mencoba memperbaiki kesalahan ensemble sebelumnya.

### Konsep Kunci

1. **Sequential Learning**: Model dibangun satu demi satu, masing-masing fokus pada kesalahan model sebelumnya
2. **Gradient Descent dalam Ruang Fungsi**: Algoritma melakukan gradient descent untuk meminimalkan fungsi loss
3. **Weak Learner**: Biasanya decision tree dangkal (kedalaman 3-8) yang hanya sedikit lebih baik dari tebakan acak
4. **Residual Fitting**: Setiap tree baru memprediksi residual (gradien negatif) dari prediksi sebelumnya
5. **Model Aditif**: Prediksi akhir adalah jumlah dari semua prediksi tree
6. **Shrinkage (Learning Rate)**: Kontribusi setiap tree diperkecil untuk mencegah overfitting

### Intuisi: Memprediksi Residual Langkah demi Langkah

Bayangkan Gradient Boosting seperti tim spesialis yang bekerja bersama:

```
Prediksi Awal:
- Mulai dengan baseline sederhana (misalnya, rata-rata target)
- Error = Aktual - Prediksi

Tree 1:
- Belajar memprediksi error
- Prediksi Baru = Baseline + Learning_Rate × Tree1

Tree 2:
- Belajar memprediksi error yang tersisa
- Prediksi Baru = Sebelumnya + Learning_Rate × Tree2

Tree 3, 4, 5...
- Setiap tree memperbaiki error yang tersisa
- Secara bertahap meningkatkan prediksi ensemble
```

**Contoh**: Memprediksi harga rumah
- Tree 1: Mempelajari bahwa lokasi mempengaruhi harga (+Rp 500 juta untuk lingkungan bagus)
- Tree 2: Mempelajari bahwa ukuran penting (+Rp 300 juta per kamar tidur)
- Tree 3: Mempelajari efek umur (-Rp 100 juta untuk rumah tua)
- Tree 4: Menangkap efek non-linear dan interaksi
- Prediksi akhir: Jumlah semua koreksi

## 📐 Fondasi Matematis

### Kerangka Umum

Gradient Boosting membangun model aditif:

```
F(x) = F₀(x) + η·h₁(x) + η·h₂(x) + ... + η·hₘ(x)

Di mana:
- F(x) = prediksi akhir
- F₀(x) = prediksi awal (konstanta)
- hₜ(x) = prediksi tree t
- η = learning rate (shrinkage)
- M = jumlah tree
```

### Langkah-langkah Algoritma

```
1. Inisialisasi model dengan konstanta:
   F₀(x) = argmin_γ Σᵢ L(yᵢ, γ)
   (Untuk regresi: rata-rata y; untuk klasifikasi: log-odds)

2. Untuk t = 1 sampai M:
   
   a) Hitung gradien negatif (pseudo-residual):
      rᵢₜ = -[∂L(yᵢ, F(xᵢ))/∂F(xᵢ)]_{F=Fₜ₋₁}
   
   b) Fit regression tree hₜ ke pseudo-residual:
      hₜ = argmin_h Σᵢ (rᵢₜ - h(xᵢ))²
   
   c) Update model:
      Fₜ(x) = Fₜ₋₁(x) + η·hₜ(x)

3. Output model akhir:
   F_M(x)
```

### Fungsi Loss dan Gradien

#### Regresi (Squared Loss)

```
Loss: L(y, F(x)) = ½(y - F(x))²

Gradien Negatif: -∂L/∂F = y - F(x) = residual

Interpretasi: Langsung memprediksi residual
```

**Contoh Perhitungan**:
```
Nilai aktual: y = 100
Prediksi saat ini: F(x) = 85
Residual: 100 - 85 = 15
→ Tree berikutnya belajar memprediksi +15
```

#### Klasifikasi Biner (Log Loss)

```
Loss: L(y, F(x)) = -[y·log(p) + (1-y)·log(1-p)]
di mana p = 1/(1 + e^(-F(x)))

Gradien Negatif: -∂L/∂F = y - p

Interpretasi: Memprediksi selisih antara label asli dan probabilitas
```

**Contoh Perhitungan**:
```
Label aktual: y = 1 (churn)
Log-odds saat ini: F(x) = -0.5
Probabilitas saat ini: p = 1/(1+e^0.5) = 0.378
Residual: 1 - 0.378 = 0.622
→ Tree berikutnya belajar meningkatkan log-odds sebesar +0.622
```

### Analogi dengan Gradient Descent

Gradient Descent Tradisional (ruang parameter):
```
θ_baru = θ_lama - η·∇L(θ)
```

Gradient Boosting (ruang fungsi):
```
F_baru(x) = F_lama(x) - η·[∂L/∂F]
          = F_lama(x) + η·h(x)
di mana h(x) memprediksi gradien negatif
```

Insight kunci: Alih-alih memperbarui parameter, kita menambahkan fungsi baru yang menunjuk ke arah gradien negatif!

## 🎯 Cara Kerja Gradient Boosting

### Contoh Step-by-Step: Regresi

Mari prediksi penjualan toko dengan fitur [is_weekend, has_promotion]:

**Dataset**:
```
Sampel 1: [1, 0] → Penjualan Aktual: 1500
Sampel 2: [0, 1] → Penjualan Aktual: 1300
Sampel 3: [1, 1] → Penjualan Aktual: 1800
```

**Iterasi 0** (Inisialisasi):
```
F₀(x) = mean(penjualan) = (1500 + 1300 + 1800) / 3 = 1533

Prediksi: [1533, 1533, 1533]
Residual: [1500-1533, 1300-1533, 1800-1533]
        = [-33, -233, 267]
```

**Iterasi 1** (Tree Pertama):
```
Tree 1 belajar memprediksi residual:
- Split pada has_promotion
- Kiri (tidak ada promosi): prediksi -133
- Kanan (ada promosi): prediksi 17

Update dengan learning_rate = 0.1:
F₁(x) = F₀(x) + 0.1 × Tree1(x)

Prediksi baru: [1533-3.3, 1533+1.7, 1533+1.7]
              = [1529.7, 1534.7, 1534.7]

Residual baru: [-29.7, -234.7, 265.3]
```

**Iterasi 2** (Tree Kedua):
```
Tree 2 belajar dari residual yang tersisa:
- Lebih memperbaiki prediksi
- F₂(x) = F₁(x) + 0.1 × Tree2(x)
```

Lanjutkan untuk M iterasi...

### Stochastic Gradient Boosting (Subsampling)

GB reguler menggunakan semua data untuk setiap tree. Stochastic GB melakukan random sampling:

```
Untuk setiap tree:
1. Random sample (misalnya 80%) dari data training tanpa replacement
2. Latih tree pada subsample ini
3. Gunakan tree penuh untuk prediksi

Manfaat:
- Mengurangi overfitting
- Mempercepat training
- Menambah keacakan seperti Random Forest
```

## ⚖️ Trade-off Bias-Variance

Gradient Boosting secara eksplisit mengontrol trade-off bias-variance melalui hyperparameter:

### Tree Individual (Bias Tinggi, Variance Rendah)

```
Single shallow tree:
- Bias tinggi: Underfit, model sederhana
- Variance rendah: Prediksi stabil
- Performa buruk sendirian
```

### Efek Ensemble (Mengurangi Bias)

```
Menambah tree secara berurutan:
- Mengurangi bias: Memperbaiki kesalahan sistematis
- Setiap tree menangkap pola yang terlewat tree sebelumnya
- Ensemble menjadi lebih kompleks dan akurat
```

### Peran Hyperparameter

**Learning Rate (η)**:
```
Learning rate besar (misalnya 1.0):
- Pembelajaran cepat, tree sedikit
- Risiko overshoot solusi optimal
- Variance lebih tinggi

Learning rate kecil (misalnya 0.01):
- Pembelajaran lambat, tree banyak
- Lebih stabil, generalisasi lebih baik
- Variance lebih rendah
```

**Jumlah Tree (M)**:
```
Tree terlalu sedikit:
- Bias tinggi: Model underfit
- Training dan test error sama-sama tinggi

Jumlah optimal:
- Bias dan variance rendah
- Performa test terbaik

Tree terlalu banyak:
- Bias rendah tapi variance TINGGI
- Overfit: training error rendah, test error tinggi
```

**Kedalaman Tree**:
```
Tree dangkal (depth 2-3):
- Bias tinggi, variance rendah
- Kurang rentan overfitting
- Training lebih cepat

Tree dalam (depth 8-10):
- Bias rendah, variance tinggi
- Mudah overfit
- Menangkap interaksi kompleks
```

## 📊 Learning Rate vs Jumlah Tree

Trade-off paling penting dalam Gradient Boosting:

```
Learning Rate × Jumlah Tree ≈ Konstan (untuk performa serupa)

Aturan Praktis:
- η = 0.01 → Perlu ~1000-5000 tree
- η = 0.1  → Perlu ~100-500 tree
- η = 0.3  → Perlu ~50-200 tree
- η = 1.0  → Perlu ~10-50 tree
```

### Mengapa Learning Rate Rendah Lebih Baik

```
Learning rate kecil + Tree banyak:
✅ Generalisasi lebih baik
✅ Jalur optimasi lebih halus
✅ Kurang sensitif terhadap error tree individual
✅ Peluang lebih baik menemukan optimum global
❌ Training lebih lambat

Learning rate besar + Tree sedikit:
✅ Training cepat
❌ Bisa overshoot solusi optimal
❌ Lebih sensitif terhadap noise
❌ Variance lebih tinggi
```

### Intuisi Visual

```
Learning Rate = 1.0:        Learning Rate = 0.1:
━━━━━━━━━━━━━━━━━          ━━━━━━━━━━━━━━━━━
   ▲                          ▲
   │  *                       │    ___---*
   │   ╲                      │  _/
   │    ╲ (overshoot)         │ / (konvergensi halus)
   │     *                    │*
   │      ╲                   │
   └───────────►             └───────────►
   Iterasi                    Iterasi
```

## ✅ Keuntungan & Kerugian

### ✅ Keuntungan

- **Performa State-of-the-art**: Sering mencapai hasil terbaik pada data terstruktur/tabular
- **Menangani Tipe Data Campuran**: Bekerja dengan fitur numerik dan kategorikal
- **Menangkap Relasi Non-linear**: Tree secara natural menangani non-linearitas
- **Interaksi Fitur**: Otomatis menemukan interaksi antar fitur
- **Robust terhadap Outlier**: Split tree berdasarkan urutan, bukan nilai absolut
- **Feature Importance**: Menyediakan skor kepentingan fitur yang interpretable
- **Fungsi Loss Fleksibel**: Bisa mengoptimasi fungsi loss yang differentiable
- **Tidak Perlu Scaling Fitur**: Bekerja dengan fitur dalam skala berbeda
- **Penanganan Missing Value**: Bisa belajar arah optimal untuk nilai missing

### ❌ Kerugian

- **Training Sequential**: Tidak bisa memparalelkan pembangunan tree (tidak seperti Random Forest)
- **Rentan Overfitting**: Terutama dengan banyak tree dan learning rate tinggi
- **Prediksi Lambat**: Harus evaluasi semua tree untuk setiap prediksi
- **Sensitif Hyperparameter**: Membutuhkan tuning hati-hati untuk performa optimal
- **Kurang Interpretable**: Ensemble tree lebih sulit dipahami daripada single tree
- **Memory Intensive**: Harus menyimpan semua tree
- **Waktu Training**: Bisa lambat dengan banyak tree atau dataset besar
- **Masalah Ekstrapolasi**: Tidak bisa prediksi di luar range data training

## 🎯 Kapan Menggunakan

### ✅ Gunakan Gradient Boosting Ketika

- Anda perlu akurasi terbaik pada data terstruktur/tabular
- Anda memiliki resource komputasi cukup untuk training
- Dataset berukuran kecil hingga menengah (< 1M baris biasanya)
- Hubungan antara fitur dan target kompleks dan non-linear
- Anda bisa investasi waktu untuk hyperparameter tuning
- Kecepatan prediksi tidak kritis
- Anda perlu feature importance untuk interpretasi
- Bekerja dengan masalah klasifikasi atau regresi

### ❌ Pertimbangkan Alternatif Ketika

- **Perlu training cepat pada data besar**: Gunakan LightGBM atau XGBoost (versi optimasi)
- **Perlu interpretabilitas tinggi**: Gunakan single Decision Tree atau Model Linear
- **Perlu training paralel**: Gunakan Random Forest
- **Prediksi real-time kritis**: Pertimbangkan model lebih sederhana
- **Dataset sangat kecil (< 100 sampel)**: Gunakan model sederhana untuk hindari overfitting
- **Data sparse berdimensi tinggi (teks, gambar)**: Pertimbangkan neural network atau algoritma khusus

### Aplikasi Dunia Nyata

**Finance**:
- Penilaian risiko kredit
- Deteksi fraud
- Prediksi customer lifetime value
- Strategi algorithmic trading

**E-commerce**:
- Ranking rekomendasi produk
- Prediksi customer churn
- Optimasi harga
- Prediksi conversion rate

**Healthcare**:
- Diagnosis penyakit
- Prediksi risiko readmission
- Prediksi respon treatment
- Deteksi fraud klaim medis

**Marketing**:
- Segmentasi customer
- Prediksi respon campaign
- Lead scoring
- Prediksi click-through rate

**Teknologi**:
- Search ranking (misalnya web search, app store)
- Anomaly detection
- Load forecasting
- Prediksi kualitas dalam manufacturing

## 📁 File dalam Direktori Ini

- **`README.md`** - Dokumentasi lengkap dalam Bahasa Inggris
- **`README.id.md`** (file ini) - Teori dan konsep komprehensif dalam Bahasa Indonesia
- **`implementation.py`** - Gradient Boosting from scratch dengan NumPy
- **`sklearn_example.py`** - Contoh praktis menggunakan scikit-learn
- **`exercise.ipynb`** - Latihan hands-on dan eksperimen

## 🚀 Quick Start

1. **Pahami Teori**: Baca README ini secara menyeluruh
2. **Lihat Implementasi**: Cek `implementation.py` untuk kode from-scratch
3. **Coba Contoh**: Jalankan `sklearn_example.py` untuk aplikasi praktis
4. **Praktik**: Selesaikan latihan `exercise.ipynb`
5. **Eksperimen**: Coba dengan dataset Anda sendiri

## 🎛️ Hyperparameter Penting

### Parameter Scikit-learn

```python
from sklearn.ensemble import GradientBoostingRegressor, GradientBoostingClassifier

GradientBoostingRegressor(
    n_estimators=100,          # Jumlah boosting stage (tree)
    learning_rate=0.1,         # Parameter shrinkage
    max_depth=3,               # Kedalaman maksimum tree
    min_samples_split=2,       # Min sampel untuk split node
    min_samples_leaf=1,        # Min sampel di leaf
    subsample=1.0,             # Fraksi sampel untuk setiap tree
    max_features=None,         # Fitur untuk dipertimbangkan saat split
    loss='squared_error',      # Fungsi loss
    criterion='friedman_mse',  # Ukuran kualitas split
    random_state=None,         # Random seed
    verbose=0,                 # Level verbosity
    warm_start=False,          # Tambah tree secara incremental
    validation_fraction=0.1,   # Fraksi untuk early stopping
    n_iter_no_change=None,     # Round early stopping
)
```

### Prioritas Tuning Hyperparameter

**1. Mulai dengan ini (paling penting)**:
```python
n_estimators = 100-500       # Tree lebih banyak dengan learning rate lebih rendah
learning_rate = 0.01-0.1     # Lebih kecil lebih baik, tapi lebih lambat
max_depth = 3-8              # Tree dangkal untuk GB
```

**2. Fine-tune ini**:
```python
subsample = 0.5-1.0          # 0.8 sering bekerja baik
min_samples_split = 2-20     # Mencegah overfitting
min_samples_leaf = 1-10      # Prediksi lebih halus
```

**3. Opsi lanjutan**:
```python
max_features = 'sqrt', 'log2', None  # Tambah keacakan
loss = 'squared_error', 'huber', dll # Robust terhadap outlier
validation_fraction = 0.1-0.2        # Untuk early stopping
```

### Starting Point yang Disarankan

**Untuk kebanyakan masalah**:
```python
GradientBoostingRegressor(
    n_estimators=200,
    learning_rate=0.1,
    max_depth=3,
    subsample=0.8,
    min_samples_split=10,
    random_state=42
)
```

Lalu gunakan GridSearchCV atau RandomizedSearchCV untuk optimasi!

## 🔗 Algoritma Terkait

### Jembatan Antar Algoritma

Gradient Boosting berada di persimpangan beberapa konsep:

```
Decision Trees ─────► Random Forest ─────► Gradient Boosting ─────► XGBoost
                      (paralel)            (sequential)             (optimasi)
```

**Decision Trees** ([03_decision_trees](../03_decision_trees/)): 
- Single tree yang bisa overfit
- Digunakan sebagai weak learner di GB
- GB menggunakan banyak tree dangkal

**Random Forest** ([04_random_forest](../04_random_forest/)):
- Membangun tree secara independen dan paralel
- Menggunakan bootstrap sampling + random features
- GB membangun tree secara sequential, masing-masing memperbaiki error
- Keduanya metode ensemble tapi strategi berbeda

**XGBoost** ([../../04_advanced_topics/04_xgboost/](../../04_advanced_topics/04_xgboost/)):
- Implementasi gradient boosting yang sangat optimasi
- Menambahkan regularisasi, parallel processing, penanganan missing value
- Algoritma inti sama tapi dengan perbaikan engineering
- 10-100x lebih cepat training daripada sklearn GradientBoosting

### Tabel Perbandingan

| Fitur | Random Forest | Gradient Boosting | XGBoost |
|-------|--------------|-------------------|---------|
| Pembangunan Tree | Paralel | Sequential | Sequential (tapi diparalelkan) |
| Kedalaman Tree | Dalam (unpruned) | Dangkal (3-8) | Dangkal (3-8) |
| Strategi Learning | Bagging | Boosting | Boosting + Regularisasi |
| Risiko Overfitting | Rendah | Sedang-Tinggi | Rendah-Sedang |
| Kecepatan Training | Cepat | Sedang | Cepat |
| Kecepatan Prediksi | Cepat | Sedang | Cepat |
| Akurasi | Sangat Baik | Excellent | Excellent |
| Sensitivitas Hyperparameter | Rendah | Tinggi | Sedang |

### Algoritma Terkait Lainnya

- **AdaBoost**: Algoritma boosting lebih awal, menyesuaikan bobot sampel alih-alih memprediksi residual
- **LightGBM**: GB optimasi dari Microsoft, lebih cepat daripada XGBoost pada data besar
- **CatBoost**: GB dari Yandex yang dioptimasi untuk fitur kategorikal
- **Histogram-based GB**: GB cepat sklearn untuk dataset besar

---

**Langkah Selanjutnya**: 
- Selami `implementation.py` untuk melihat cara kerjanya di balik layar
- Jalankan `sklearn_example.py` untuk melihat aplikasi praktis
- Selesaikan `exercise.ipynb` untuk memperkuat pemahaman Anda
- Lalu eksplorasi **XGBoost** untuk gradient boosting tingkat production!
