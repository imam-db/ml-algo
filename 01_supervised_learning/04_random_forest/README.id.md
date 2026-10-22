# Random Forest

**[🇬🇧 English](README.md) | 🇮🇩 Bahasa Indonesia**

🏠 Kembali ke Indeks Supervised: `../README.md`

Random Forest adalah metode ensemble learning yang kuat yang menggabungkan beberapa decision tree untuk membentuk model yang lebih kokoh dan akurat. Random Forest merupakan salah satu algoritma machine learning paling populer dan efektif.

## 📚 Daftar Isi
- [Teori & Konsep](#-teori--konsep)
- [Dasar Matematika](#-dasar-matematika)
- [Metode Ensemble](#-metode-ensemble)
- [Bootstrap Aggregating (Bagging)](#-bootstrap-aggregating-bagging)
- [Kelebihan & Kekurangan](#-kelebihan--kekurangan)
- [Kapan Digunakan](#-kapan-digunakan)
- [File di Direktori Ini](#-file-di-direktori-ini)

## 🧠 Teori & Konsep

### Apa itu Random Forest?
Random Forest adalah metode ensemble yang membangun beberapa decision tree selama training dan mengeluarkan rata-rata prediksi (regresi) atau suara mayoritas (klasifikasi) dari masing-masing tree. Metode ini menggabungkan konsep **bagging** dan **pemilihan fitur secara acak**.

### Konsep Kunci:
1. **Ensemble Learning**: Menggabungkan beberapa weak learner untuk membentuk strong learner
2. **Bootstrap Sampling**: Membuat dataset berbeda dengan sampling dengan pengembalian
3. **Seleksi Fitur Acak**: Setiap tree menggunakan subset fitur secara acak
4. **Voting/Averaging**: Menggabungkan prediksi dari semua tree
5. **Out-of-Bag (OOB) Error**: Estimasi error menggunakan sampel yang tidak dipakai saat training setiap tree
6. **Feature Importance**: Mengukur kepentingan berdasarkan penurunan impurity di semua tree

### Cara Kerja Random Forest:
```
1. Buat B bootstrap sample dari data training
2. Untuk setiap bootstrap sample:
   - Latih sebuah decision tree
   - Pada setiap split, pertimbangkan hanya m fitur acak (m < total fitur)
   - Tumbuhkan tree tanpa pruning
3. Untuk prediksi:
   - Klasifikasi: Suara mayoritas dari semua tree
   - Regresi: Rata-rata prediksi dari semua tree
```

## 📐 Dasar Matematika

### Bootstrap Sampling
```
Diberikan dataset D dengan n sampel:
- Buat bootstrap sample D_i dengan mengambil n sampel DENGAN pengembalian
- Setiap D_i berisi ~63,2% sampel unik dari D
- Sisanya ~36,8% adalah sampel "out-of-bag" (OOB)
```

### Random Feature Selection
```
Pada setiap split node:
- Total fitur: p
- Fitur acak yang dipertimbangkan: m = √p (klasifikasi) atau m = p/3 (regresi)
- Split terbaik dipilih hanya dari m fitur tersebut
```

### Prediksi Ensemble

#### Untuk Klasifikasi:
```
Final_Prediction = mode(Tree_1, Tree_2, ..., Tree_B)

Atau dengan probabilitas:
P(class_k) = (1/B) * Σ P_i(class_k)
dengan P_i(class_k) adalah probabilitas dari tree i
```

#### Untuk Regresi:
```
Final_Prediction = (1/B) * Σ Tree_i_Prediction
```

### Out-of-Bag (OOB) Error
```
Untuk setiap sampel x_i:
- Temukan semua tree yang tidak memasukkan x_i dalam training set
- Lakukan prediksi hanya menggunakan tree tersebut
- Bandingkan dengan label sebenarnya untuk menghitung OOB error

OOB_Error = (1/n) * Σ L(y_i, ŷ_i^OOB)
dengan L adalah loss function dan ŷ_i^OOB adalah prediksi OOB
```

### Feature Importance
```
Untuk setiap fitur j:
1. Hitung penurunan impurity ketika fitur j digunakan untuk splitting
2. Rata-ratakan di semua tree dan semua node yang menggunakan fitur j
3. Normalisasi dengan total penurunan impurity

Importance_j = Σ(trees) Σ(nodes using feature j) Impurity_Decrease / Total_Impurity_Decrease
```

## 🌳 Metode Ensemble

### Jenis-jenis Metode Ensemble:

1. **Bagging (Bootstrap Aggregating)**
   - Melatih model pada subset data yang berbeda
   - Menggabungkan prediksi dengan voting/averaging
   - Mengurangi varians (overfitting)
   - Random Forest menggunakan metode ini

2. **Boosting**
   - Melatih model secara sekuensial
   - Setiap model belajar dari error model sebelumnya
   - Mengurangi bias (underfitting)
   - Contoh: AdaBoost, Gradient Boosting

3. **Stacking**
   - Melatih beberapa algoritma yang berbeda
   - Menggunakan meta-learner untuk menggabungkan prediksi
   - Dapat mengurangi bias maupun varians

### Mengapa Ensemble Bekerja:

#### Bias-Variance Decomposition:
```
Total Error = Bias² + Variance + Irreducible Error

Random Forest:
- Tree individual: Varians tinggi, bias rendah
- Ensemble: Mengurangi varians sambil mempertahankan bias rendah
```

#### Wisdom of Crowds:
- Tree individual membuat error yang berbeda-beda
- Averaging mengurangi error keseluruhan
- Bekerja paling baik ketika tree beragam dan tidak berkorelasi

## 🎒 Bootstrap Aggregating (Bagging)

### Proses Bootstrap:
1. **Dataset Asli**: D = {(x₁,y₁), (x₂,y₂), ..., (xₙ,yₙ)}
2. **Bootstrap Sample**: Ambil n instance DENGAN pengembalian
3. **Beberapa Sampel**: Buat B bootstrap sample
4. **Latih Model**: Satu model per bootstrap sample
5. **Agregasi**: Gabungkan prediksi

### Manfaat Bagging:
- **Variance Reduction**: Averaging mengurangi varians prediksi
- **Stability**: Kurang sensitif terhadap perubahan kecil pada data  
- **Parallel Training**: Setiap model dapat dilatih secara independen
- **OOB Evaluation**: Validasi bawaan tanpa test set terpisah

### Intuisi Matematis:
```
Jika setiap tree memiliki varians error σ²:
- Varians satu tree: σ²
- Rata-rata B tree independen: σ²/B
- Dengan korelasi ρ antar tree: ρσ² + (1-ρ)σ²/B

Random feature selection mengurangi korelasi ρ
```

## ✅ Kelebihan & Kekurangan

### ✅ Kelebihan:
- **Akurasi Tinggi**: Sering memberikan performa yang sangat baik
- **Tahan terhadap Overfitting**: Averaging mengurangi risiko overfitting
- **Menangani Nilai yang Hilang**: Dapat mempertahankan akurasi dengan data yang hilang
- **Feature Importance**: Peringkat feature importance bawaan
- **Tidak Perlu Scaling**: Bekerja dengan fitur pada skala berbeda
- **Parallel Training**: Tree dapat dilatih secara bersamaan
- **Validasi OOB**: Validasi model bawaan
- **Versatile**: Bekerja untuk klasifikasi maupun regresi
- **Menangani Dataset Besar**: Skalanya baik terhadap ukuran data
- **Tahan terhadap Outlier**: Tree individual mungkin terpengaruh, tetapi ensemble tetap kokoh

### ❌ Kekurangan:
- **Kurang Mudah Diinterpretasikan**: Lebih sulit dipahami daripada satu decision tree
- **Intensif Memori**: Menyimpan banyak tree
- **Prediksi Lebih Lambat**: Harus menanyakan semua tree untuk prediksi
- **Dapat Mengalami Overfitting**: Pada data yang sangat bising atau tree yang terlalu banyak
- **Bias terhadap Fitur Kategorikal**: Dengan banyak kategori
- **Kurang Cocok untuk Hubungan Linear**: Seperti tree individual
- **Tuning Hyperparameter**: Lebih banyak parameter untuk dioptimalkan

## 🎯 Kapan Digunakan

### ✅ Gunakan Random Forest Ketika:
- Anda membutuhkan akurasi tinggi tanpa banyak tuning
- Dataset memiliki tipe fitur campuran (numerik + kategorikal)
- Anda menginginkan peringkat feature importance
- Interpretabilitas tidak sepenting performa
- Anda memiliki data training yang memadai
- Anda menginginkan algoritma yang kokoh dan serba guna
- Anda perlu menangani nilai yang hilang dengan baik
- Anda menginginkan validasi bawaan (OOB error)

### ❌ Jangan Gunakan Ketika:
- Anda membutuhkan model yang sangat mudah diinterpretasikan
- Anda memiliki memori/sumber daya komputasi yang sangat terbatas
- Dataset sangat kecil (< 1000 sampel)
- Kecepatan prediksi sangat kritis
- Hubungannya terutama linear
- Anda membutuhkan kalibrasi probabilitas (meskipun ini dapat diperbaiki)

### Aplikasi di Dunia Nyata:
- **Finance**: Credit scoring, algorithmic trading, penilaian risiko
- **Healthcare**: Prediksi penyakit, drug discovery, medical imaging
- **Technology**: Sistem rekomendasi, klasifikasi gambar, NLP
- **Marketing**: Segmentasi pelanggan, prediksi churn, A/B testing
- **Manufacturing**: Kontrol kualitas, predictive maintenance
- **Bioinformatics**: Analisis ekspresi gen, prediksi struktur protein

## 🔧 Varian Random Forest

### 1. **Random Forest Standar**
- Random bootstrap sampling
- Random feature selection pada setiap split
- Tanpa pruning pada tree individual

### 2. **Extremely Randomized Trees (Extra Trees)**
- Random feature selection
- Threshold acak (bukan split optimal)
- Menggunakan seluruh dataset (tanpa bootstrap)
- Training lebih cepat, generalisasi berpotensi lebih baik

### 3. **Balanced Random Forest**
- Menangani dataset yang tidak seimbang
- Menyeimbangkan bootstrap sample
- Baik untuk distribusi kelas yang miring

### 4. **Rotation Forest**
- Menerapkan PCA pada subset fitur
- Merotasi ruang fitur untuk setiap tree
- Dapat meningkatkan performa pada beberapa dataset

## 📁 File di Direktori Ini

- **`README.md`** (file ini) - Teori dan konsep
- **`implementation.py`** - Random Forest from scratch
- **`sklearn_example.py`** - Contoh praktis menggunakan scikit-learn
- **`exercise.ipynb`** - Latihan dan eksperimen langsung
- **`datasets/`** - Dataset contoh untuk latihan

## 🚀 Mulai Cepat

1. **Pahami Teori**: Baca README ini
2. **Lihat Implementasi**: Periksa `implementation.py`
3. **Coba Contoh**: Jalankan `sklearn_example.py`
4. **Latihan**: Selesaikan `exercise.ipynb`
5. **Eksperimen**: Gunakan dataset Anda sendiri

## 🎛️ Hyperparameter Penting

### Parameter Scikit-learn:
```python
RandomForestClassifier(
    n_estimators=100,          # Jumlah tree
    criterion='gini',          # Kriteria split
    max_depth=None,           # Kedalaman maksimum per tree
    min_samples_split=2,      # Sampel minimum untuk split node
    min_samples_leaf=1,       # Sampel minimum di leaf
    max_features='sqrt',      # Fitur per split
    bootstrap=True,           # Gunakan bootstrap sampling
    oob_score=False,          # Hitung OOB score
    n_jobs=None,              # Pekerjaan paralel (-1 untuk semua core)
    random_state=None,        # Random seed
    warm_start=False,         # Tambahkan tree secara bertahap
    class_weight=None,        # Tangani ketidakseimbangan kelas
)
```

### Penjelasan Parameter Kunci:
- **n_estimators**: Semakin banyak tree = performa lebih baik, dengan diminishing returns setelah ~100-500
- **max_features**: Mengontrol keacakan dan overfitting (sqrt untuk klasifikasi, 1/3 untuk regresi)
- **max_depth**: Mengontrol kompleksitas tree individual
- **min_samples_split/leaf**: Mencegah overfitting dengan mensyaratkan sampel minimum
- **bootstrap**: Apakah menggunakan bootstrap sampling (True untuk RF, False untuk Extra Trees)
- **oob_score**: Apakah menghitung skor validasi out-of-bag

## 📊 Tuning Performa

### Strategi Hyperparameter Tuning:
1. **Mulai dengan nilai default**: Sering kali sudah bekerja baik
2. **Tuning n_estimators**: Temukan titik ketika performa mendatar
3. **Tuning max_features**: Coba sqrt(p), log2(p), p/3
4. **Tuning kedalaman tree**: max_depth, min_samples_split, min_samples_leaf
5. **Tangani ketidakseimbangan kelas**: class_weight, balanced sampling

### Grid Cross-Validation:
```python
param_grid = {
    'n_estimators': [100, 200, 500],
    'max_features': ['sqrt', 'log2', None],
    'max_depth': [3, 5, 7, None],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4]
}
```

## 🔗 Algoritma Terkait

- **Decision Trees**: Blok penyusun Random Forest
- **Extra Trees**: Varian extremely randomized trees
- **Gradient Boosting**: Metode ensemble sekuensial
- **XGBoost/LightGBM**: Gradient boosting yang dioptimalkan
- **Bagging**: Metode bootstrap aggregating umum

---

**Langkah Berikutnya**: 
- Coba implementasinya di `sklearn_example.py`
- Selesaikan latihan di `exercise.ipynb`
- Bandingkan dengan metode ensemble lainnya
