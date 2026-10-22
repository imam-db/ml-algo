# XGBoost (eXtreme Gradient Boosting)

**[🇬🇧 English](README.md) | 🇮🇩 Bahasa Indonesia**

**🏠 [Kembali ke Utama](../../README.md)** | **📁 [Topik Lanjutan](../README.md)** | **⬅️ [Sebelumnya: Evaluasi Model](../03_model_evaluation/README.md)**

## 🔑 Istilah Kunci
**Definisi singkat untuk algoritma ini** - 📚 [Glosarium Lengkap](../../GLOSSARY.md)

- **[XGBoost](../../GLOSSARY.md#boosting)** - eXtreme Gradient Boosting, metode ensemble tingkat lanjut
- **[Ensemble Method](../../GLOSSARY.md#ensemble-method)** - Menggabungkan beberapa model untuk performa yang lebih baik
- **[Boosting](../../GLOSSARY.md#boosting)** - Pelatihan berurutan di mana setiap model memperbaiki kesalahan model sebelumnya
- **[Hyperparameter Tuning](../../GLOSSARY.md#hyperparameter-tuning)** - Menemukan konfigurasi pengaturan algoritma yang optimal
- **[Cross-Validation](../../GLOSSARY.md#cross-validation-cv)** - Teknik untuk menilai performa model secara robust
- **[Feature Importance](../../GLOSSARY.md#feature-importance)** - Ukuran kontribusi setiap fitur terhadap prediksi
- **[SHAP](../../GLOSSARY.md#shap-shapley-additive-explanations)** - Metode untuk menjelaskan prediksi individual

## 🗺️ Akses Cepat

| File | Deskripsi | Jalankan Cepat |
|------|-------------|----------|
| [`implementation.py`](./implementation.py) | Penggunaan dasar XGBoost | `uv run python implementation.py` |
| [`sklearn_example.py`](./sklearn_example.py) | Contoh komprehensif | `uv run python sklearn_example.py` |
| `hyperparameter_tuning.py` (direncanakan) | Tuning lanjutan | — |
| [`exercise.ipynb`](./exercise.ipynb) | Latihan interaktif | `uv run jupyter lab exercise.ipynb` |

## 📖 Teori dan Konsep

### Apa itu XGBoost?
[**XGBoost**](../../GLOSSARY.md#boosting) (eXtreme Gradient Boosting) adalah implementasi lanjutan dari [**gradient boosting**](../../GLOSSARY.md#boosting) yang dioptimalkan untuk kecepatan dan performa. XGBoost adalah salah satu [**algoritma**](../../GLOSSARY.md#algorithm) paling populer dan sukses dalam kompetisi machine learning, terutama untuk data tabular.

### Konsep Gradient Boosting
[**Gradient Boosting**](../../GLOSSARY.md#boosting) adalah [**ensemble method**](../../GLOSSARY.md#ensemble-method) yang:
1. Membangun [**models**](../../GLOSSARY.md#model) secara berurutan
2. Setiap model baru berusaha memperbaiki kesalahan model sebelumnya
3. Menggunakan [**gradient descent**](../../GLOSSARY.md#gradient-descent) untuk meminimalkan [**loss function**](../../GLOSSARY.md#cost-function-loss-function)

### Rumus Matematis
**Fungsi Objektif:**
```
Obj(θ) = Σ L(yi, ŷi) + Σ Ω(fk)
```

Di mana:
- `L(yi, ŷi)` = Fungsi loss (loss pelatihan)
- `Ω(fk)` = Suku regularisasi
- `ŷi = Σ fk(xi)` = Prediksi (jumlah pohon)

**Inovasi XGBoost:**
- **Aproksimasi orde dua** dari fungsi loss
- Suku **regularisasi** dalam fungsi objektif
- Algoritma **konstruksi pohon yang dioptimalkan**

## 🎯 Kapan Menggunakan XGBoost?

### ✅ Sangat Cocok untuk:
- **Data tabular/terstruktur**
- **Kompetisi machine learning** (Kaggle, dll.)
- **Data dengan missing values**
- **Tipe data campuran** (numerik + kategorikal)
- **Tugas klasifikasi dan regresi**
- Analisis **feature importance**
- **Dataset berukuran menengah-besar**

### ❌ Tidak ideal untuk:
- **Visi komputer** (CNN lebih baik)
- **Pemrosesan bahasa alami (NLP)** (Transformer lebih baik)
- **Dataset yang sangat kecil** (< 1000 sampel)
- **Prediksi real-time** (latensi tinggi)
- **Data dengan struktur temporal kompleks**

## 🔍 Kelebihan dan Kekurangan

### Kelebihan:
- ✅ **Performa sangat baik** pada data tabular
- ✅ **Menangani missing values** secara native
- ✅ **Regularisasi bawaan** (L1, L2)
- ✅ **Feature importance** yang akurat
- ✅ **Pemrosesan paralel** dan dukungan GPU
- ✅ **Cross-validation** bawaan
- ✅ **Early stopping** untuk mencegah overfitting
- ✅ **Robust terhadap outlier**

### Kekurangan:
- ❌ **Hyperparameter tuning** yang kompleks
- ❌ **Intensif memori** untuk dataset besar
- ❌ **Rentan overfitting** jika tidak di-tune dengan baik
- ❌ **Model black box** (sulit diinterpretasikan)
- ❌ **Waktu pelatihan** relatif lama
- ❌ **Sensitif terhadap data yang bising**

## ⚙️ Hyperparameter Utama

### Parameter Khusus Pohon (Tree):
- **n_estimators**: Jumlah pohon (default: 100)
- **max_depth**: Kedalaman maksimum pohon (default: 6)
- **min_child_weight**: Jumlah bobot minimum di leaf (default: 1)
- **subsample**: Fraksi sampel untuk setiap pohon (default: 1)
- **colsample_bytree**: Fraksi fitur untuk setiap pohon (default: 1)

### Parameter Pembelajaran:
- **learning_rate/eta**: Penyusutan ukuran langkah (default: 0.3)
- **objective**: Fungsi loss (reg:squarederror, binary:logistic, dll.)
- **eval_metric**: Metrik evaluasi (rmse, logloss, auc, dll.)

### Parameter Regularisasi:
- **reg_alpha**: Regularisasi L1 (default: 0)
- **reg_lambda**: Regularisasi L2 (default: 1)
- **gamma**: Pengurangan loss minimum untuk split (default: 0)

## 📊 XGBoost vs Algoritma Lain

| Aspek | XGBoost | Random Forest | Gradient Boosting | Model Linear |
|-------|---------|---------------|-------------------|---------------|
| **Akurasi** | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐ | ⭐⭐⭐⭐ | ⭐⭐⭐ |
| **Kecepatan** | ⭐⭐⭐ | ⭐⭐⭐⭐ | ⭐⭐ | ⭐⭐⭐⭐⭐ |
| **Interpretabilitas** | ⭐⭐ | ⭐⭐⭐ | ⭐⭐ | ⭐⭐⭐⭐⭐ |
| **Risiko Overfitting** | ⭐⭐ | ⭐⭐⭐⭐ | ⭐⭐ | ⭐⭐⭐⭐ |
| **Tuning Hyperparameter** | ⭐ | ⭐⭐⭐ | ⭐⭐ | ⭐⭐⭐⭐⭐ |

## 🛠️ Instalasi dan Penyiapan

### Instalasi XGBoost:
```bash
# Via pip
pip install xgboost

# Via conda
conda install -c conda-forge xgboost

# Via UV (recommended untuk project ini)
uv add xgboost
```

### Dukungan GPU (Opsional):
```bash
# For CUDA support
pip install xgboost[gpu]
```

## 📈 Evaluasi dan Pemilihan Model

### Metrik Klasifikasi:
- **Accuracy**: Prediksi benar secara keseluruhan
- **Precision/Recall**: Performa per kelas
- **F1-Score**: Rata-rata harmonik precision/recall
- **AUC-ROC**: Area di bawah kurva ROC
- **Log Loss**: Loss probabilistik

### Metrik Regresi:
- **RMSE**: Root Mean Squared Error
- **MAE**: Mean Absolute Error
- **R²**: Koefisien determinasi
- **MAPE**: Mean Absolute Percentage Error

### Cross-Validation:
XGBoost memiliki CV bawaan:
```python
cv_results = xgb.cv(
    params, dtrain, num_boost_round=100,
    nfold=5, early_stopping_rounds=10
)
```

## 🎓 Praktik Terbaik

### 1. **Persiapan Data**:
- Tangani missing values (XGBoost dapat menanganinya, tetapi prapemrosesan mungkin membantu)
- Enkode variabel kategorikal (LabelEncoder atau One-hot)
- Feature scaling tidak wajib, tetapi bisa membantu

### 2. **Tuning Hyperparameter**:
- Mulai dengan parameter default
- Lakukan tuning **n_estimators** dengan early stopping
- Sesuaikan **learning_rate** dan **max_depth**
- Gunakan **RandomizedSearchCV** atau **Optuna**

### 3. **Pencegahan Overfitting**:
- Gunakan **early_stopping_rounds**
- Terapkan **regularisasi** (reg_alpha, reg_lambda)
- Kurangi **learning_rate** dan tingkatkan **n_estimators**
- Gunakan **cross-validation** untuk pemantauan

### 4. **Rekayasa Fitur**:
- Buat fitur interaksi
- Fitur polinomial untuk hubungan non-linear
- Rekayasa fitur spesifik domain

### 5. **Interpretasi Model**:
- Buat plot **feature importance**
- Gunakan nilai **SHAP** untuk interpretasi terperinci
- Analisis **learning curves**

## 📊 Alur Kerja Tipikal

1. **Persiapan Data** → Bersihkan, enkode, bagi
2. **Model Baseline** → Parameter default
3. **Tuning Hyperparameter** → GridSearch/RandomSearch
4. **Cross-Validation** → Estimasi performa yang robust
5. **Feature Importance** → Memahami model
6. **Interpretasi Model** → SHAP, partial dependence
7. **Evaluasi Akhir** → Performa pada test set

## 🎯 Skenario Dunia Nyata

### 🏆 Skenario 1: Gaya Kompetisi Kaggle
**Dataset**: Data kompetisi tabular (Titanic, House Prices)
**Tujuan**: Memaksimalkan skor papan peringkat
**Jalankan**: `uv run python sklearn_example.py --scenario kaggle`

**Yang akan Anda kuasai:**
- Teknik rekayasa fitur
- Strategi cross-validation
- Metode ensemble dengan beberapa model
- Optimasi hyperparameter

### 💳 Skenario 2: Penilaian Risiko Kredit
**Dataset**: Data keuangan nasabah
**Tujuan**: Memprediksi probabilitas gagal bayar pinjaman
**Jalankan**: `uv run python sklearn_example.py --scenario credit_risk`

**Yang akan Anda kuasai:**
- Menangani dataset tidak seimbang
- Analisis feature importance
- Metrik berorientasi bisnis (precision/recall)
- Interpretabilitas model dengan SHAP

### 🏭 Skenario 3: Peramalan Penjualan E-commerce
**Dataset**: Data penjualan historis, data musiman
**Tujuan**: Memprediksi volume penjualan bulanan
**Jalankan**: `uv run python sklearn_example.py --scenario sales_forecast`

**Yang akan Anda kuasai:**
- Rekayasa fitur deret waktu
- Dekomposisi musiman
- Peramalan multi-langkah
- Evaluasi dampak bisnis

### 🎭 Skenario 4: Prediksi Churn Pelanggan
**Dataset**: Perilaku pelanggan, pola penggunaan
**Tujuan**: Mengidentifikasi pelanggan yang kemungkinan akan churn
**Jalankan**: `uv run python sklearn_example.py --scenario churn`

**Yang akan Anda kuasai:**
- Rekayasa fitur perilaku
- Pembelajaran cost-sensitive
- Menghasilkan wawasan yang dapat ditindaklanjuti
- Strategi pemantauan model

### 🏏 Skenario 5: Dukungan Diagnosis Medis
**Dataset**: Gejala pasien, riwayat medis
**Tujuan**: Membantu diagnosis penyakit
**Jalankan**: `uv run python sklearn_example.py --scenario medical`

**Yang akan Anda kuasai:**
- Skenario prediksi berisiko tinggi
- Estimasi interval kepercayaan
- Trade-off false positive/negative
- Pertimbangan kepatuhan regulasi

### 🔍 Skenario Tuning Hyperparameter

> 🚧 Skrip `hyperparameter_tuning.py` khusus masih direncanakan. Alur tuning di bawah belum dapat dijalankan; gunakan [`sklearn_example.py`](./sklearn_example.py) untuk contoh yang tersedia saat ini.

#### Tuning Cepat (5-10 menit)
Random Search dengan anggaran kecil.

#### Tuning Produksi (30-60 menit)
Pencarian berbasis Optuna dengan anggaran sedang.

#### Tuning Kompetisi (2-4 jam)
Optimasi Bayesian dengan anggaran besar.

### [`implementation.py`](./implementation.py)
- **Alur kerja dasar XGBoost** untuk pemula
- **Contoh mulai cepat** untuk setiap skenario
- **Penyiapan model baseline**
- **Perbandingan performa** dengan algoritma lain

### [`sklearn_example.py`](./sklearn_example.py)
- **Implementasi siap produksi** untuk kelima skenario
- Pipeline **prapemrosesan lanjutan**
- Otomasi **rekayasa fitur**
- **Evaluasi model** dan **metrik bisnis**
- Integrasi **interpretabilitas SHAP**

### `hyperparameter_tuning.py` (direncanakan)
- **Strategi tuning yang direncanakan**: Random Search, Bayesian, Optuna
- **Optimasi yang sadar anggaran**
- **Pemrosesan paralel** untuk tuning yang lebih cepat
- Ruang parameter **sekualitas kompetisi**

### [`exercise.ipynb`](./exercise.ipynb)
- Tutorial **interaktif langkah demi langkah**
- **Eksplorasi dataset nyata** dan **EDA**
- Analisis **sensitivitas hyperparameter**
- Latihan **interpretasi model**

## 🎓 Progres Pembelajaran

### Prasyarat
- **Dasar-dasar Machine Learning**: [Linear Regression](../../01_supervised_learning/01_linear_regression/), [Decision Trees](../../01_supervised_learning/03_decision_trees/)
- **Metode Ensemble**: [Random Forest](../../01_supervised_learning/04_random_forest/)
- **Evaluasi Model**: [Teknik Evaluasi](../03_model_evaluation/)

### Jalur Pembelajaran yang Direkomendasikan
1. **Mulai** dengan [`implementation.py`](./implementation.py) - Konsep dasar
2. **Berlatih** dengan [`exercise.ipynb`](./exercise.ipynb) - Pembelajaran interaktif
3. **Jelajahi** [`sklearn_example.py`](./sklearn_example.py) - Skenario nyata
4. **Kuasai** `hyperparameter_tuning.py` (direncanakan) - Teknik lanjutan

## 🏆 Tips Kompetisi

1. **Rekayasa Fitur** adalah raja
2. **Ensemble** dengan seed yang berbeda
3. **Blend** XGBoost dengan LightGBM/CatBoost
4. **Pseudo-labeling** untuk pembelajaran semi-supervised
5. **Target encoding** untuk kategorikal berkardinalitas tinggi
6. **Stratified K-Fold** untuk CV yang robust

### Algoritma Lanjutan Terkait

| Algoritma | Perbandingan | Kasus Penggunaan |
|-----------|------------|----------|
| **LightGBM** (direncanakan) | Pelatihan lebih cepat | Dataset besar |
| **CatBoost** (direncanakan) | Penanganan kategorikal lebih baik | Tipe data campuran |
| [**Random Forest**](../../01_supervised_learning/04_random_forest/) | Lebih mudah diinterpretasikan | Feature importance |
| [**Neural Networks**](../../01_supervised_learning/08_neural_networks/) | Lebih fleksibel | Pola kompleks |

## 📚 Referensi

- [Dokumentasi XGBoost](https://xgboost.readthedocs.io/)
- [Makalah XGBoost](https://arxiv.org/abs/1603.02754)
- [Tutorial XGBoost](https://xgboost.readthedocs.io/en/stable/tutorials/index.html)
- [ML yang Dapat Diinterpretasikan - XGBoost](https://christophm.github.io/interpretable-ml-book/)

---
**➡️ Langkah Berikutnya**: Setelah menguasai XGBoost, jelajahi **LightGBM** dan **CatBoost** untuk metode ensemble yang bahkan lebih kuat!
