# Linear Regression

**[🇬🇧 English](README.md) | 🇮🇩 Bahasa Indonesia**

**🏠 [Kembali ke Utama](../../README.md)** | **📁 [Supervised Learning](../README.md)** | **➡️ [Berikutnya: Logistic Regression](../02_logistic_regression/README.md)**

## 🔑 Istilah Kunci
**Definisi singkat untuk algoritma ini** - 📚 [Glosarium Lengkap](../../GLOSSARY.md)

- **[Linear Regression](../../GLOSSARY.md#linear-model)** - Algoritma yang mengasumsikan hubungan linear antara fitur dan target
- **[Overfitting](../../GLOSSARY.md#overfitting)** - Ketika model terlalu menghafal data training sehingga performanya buruk pada data baru
- **[MSE](../../GLOSSARY.md#mse-mean-squared-error)** - Mean Squared Error, mengukur rata-rata kuadrat error prediksi
- **[R²](../../GLOSSARY.md#r²-r-squared--coefficient-of-determination)** - Proporsi varians yang dijelaskan oleh model (0-1, semakin tinggi semakin baik)
- **[Gradient Descent](../../GLOSSARY.md#gradient-descent)** - Algoritma optimasi yang meminimalkan cost function secara iteratif
- **[Normal Equation](../../GLOSSARY.md#normal-equation)** - Solusi analitis: θ = (X^T × X)^(-1) × X^T × y

## 🗺️ Akses Cepat

| File | Deskripsi | Cara Menjalankan |
|------|-------------|----------|
| [`implementation.py`](./implementation.py) | Implementasi from-scratch | `uv run python implementation.py` |
| [`sklearn_example.py`](./sklearn_example.py) | Contoh scikit-learn | `uv run python sklearn_example.py` |
| [`exercise.ipynb`](./exercise.ipynb) | Latihan interaktif (Bahasa Indonesia) | `uv run jupyter lab exercise.ipynb` |
| [`exercise_en.ipynb`](./exercise_en.ipynb) | Latihan interaktif (English) | `uv run jupyter lab exercise_en.ipynb` |
| [`exercise_EN.md`](./exercise_EN.md) | Latihan (English, Markdown) | Buka dan ikuti instruksinya |

## 📖 Teori dan Konsep

### Apa itu Linear Regression?
[**Linear Regression**](../../GLOSSARY.md#linear-model) adalah [**algoritma**](../../GLOSSARY.md#algorithm) [**supervised learning**](../../GLOSSARY.md#supervised-learning) yang digunakan untuk memprediksi [**nilai target**](../../GLOSSARY.md#target-variable) (kontinu) berdasarkan satu atau lebih [**fitur**](../../GLOSSARY.md#feature) input. Algoritma ini mencari hubungan linear terbaik antara input dan output.

### Rumus Matematika
**Simple Linear Regression (1 variabel):**
```
y = β₀ + β₁x + ε
```

**Multiple Linear Regression (banyak variabel):**
```
y = β₀ + β₁x₁ + β₂x₂ + ... + βₙxₙ + ε
```

Keterangan:
- `y` = variabel target (dependen)
- `x` = variabel fitur (independen) 
- `β₀` = intercept (bias)
- `β₁, β₂, ..., βₙ` = koefisien (slope)
- `ε` = error/noise

### Cost Function
Linear regression menggunakan **Mean Squared Error (MSE)** sebagai cost function:
```
MSE = (1/n) * Σ(yi - ŷi)²
```

### Optimasi
Untuk menemukan parameter terbaik, Anda dapat menggunakan:
1. **Normal Equation** (Solusi closed-form)
2. **Gradient Descent** (Pendekatan iteratif)

## 🎯 Kapan Menggunakan Linear Regression?

### ✅ Cocok untuk:
- Memprediksi nilai kontinu (harga rumah, suhu, gaji, dan sebagainya)
- Data dengan hubungan linear
- Interpretasi model yang mudah dipahami
- Model baseline untuk perbandingan
- Dataset berukuran kecil hingga sedang

### ❌ Tidak cocok untuk:
- Data dengan hubungan non-linear yang kompleks
- Banyak outlier
- Variabel target kategorikal
- Multikolinearitas yang tinggi

## 📊 Asumsi Linear Regression

1. **Linearity**: Hubungan linear antara X dan y
2. **Independence**: Observasi bersifat independen
3. **Homoscedasticity**: Varians error konstan
4. **Distribusi Normal**: Error berdistribusi normal
5. **Tanpa Multikolinearitas**: Fitur tidak saling berkorelasi tinggi

## 📈 Evaluasi Model

### Metrik untuk Regresi:
- **R² (Coefficient of Determination)**: Proporsi varians yang dijelaskan
- **MSE (Mean Squared Error)**: Rata-rata kuadrat error
- **RMSE (Root Mean Squared Error)**: Akar kuadrat dari MSE
- **MAE (Mean Absolute Error)**: Rata-rata error absolut

## 🔍 Kelebihan dan Kekurangan

### Kelebihan:
- ✅ Sederhana dan cepat
- ✅ Mudah diinterpretasikan
- ✅ Tidak memerlukan tuning parameter yang rumit
- ✅ Baseline yang baik
- ✅ Tidak rentan terhadap overfitting

### Kekurangan:
- ❌ Hanya dapat menangkap hubungan linear
- ❌ Sensitif terhadap outlier
- ❌ Asumsi yang ketat
- ❌ Performa buruk pada data non-linear

## 🎯 Skenario Praktis

### 🏠 Skenario 1: Prediksi Harga Rumah
**Dataset**: Fitur rumah (luas, jumlah kamar tidur, lokasi)
**Tujuan**: Memprediksi harga rumah
**Jalankan**: `uv run python sklearn_example.py --scenario house_prices`

**Yang akan Anda pelajari:**
- Feature scaling untuk banyak variabel
- Menangani variabel kategorikal
- Interpretasi model dengan koefisien

### 🌡️ Skenario 2: Prediksi Suhu
**Dataset**: Fitur cuaca (kelembapan, tekanan, kecepatan angin)
**Tujuan**: Memprediksi suhu harian
**Jalankan**: `uv run python sklearn_example.py --scenario temperature`

**Yang akan Anda pelajari:**
- Persiapan data time series
- Analisis korelasi fitur
- Pemodelan tren musiman

### 💰 Skenario 3: Estimasi Gaji
**Dataset**: Data karyawan (pengalaman, pendidikan, keterampilan)
**Tujuan**: Mengestimasi rentang gaji
**Jalankan**: `uv run python sklearn_example.py --scenario salary`

**Yang akan Anda pelajari:**
- Polynomial features
- Teknik cross-validation
- Confidence intervals

### 📈 Skenario 4: Tren Harga Saham
**Dataset**: Data historis saham
**Tujuan**: Analisis tren sederhana
**Jalankan**: `uv run python implementation.py --scenario stocks`

**Yang akan Anda pelajari:**
- Gradient descent vs Normal equation
- Optimasi learning rate
- Visualisasi cost function

### 🔬 Bandingkan Metode
**Bandingkan Normal Equation vs Gradient Descent:**
```bash
uv run python implementation.py --compare-methods
```

## 📝 File Implementasi

### [`implementation.py`](./implementation.py)
- Kelas **LinearRegressionFromScratch**
- **Dua metode optimasi**: Normal Equation & Gradient Descent
- **Skenario bawaan**: Keempat contoh praktis di atas
- **Alat visualisasi**: Riwayat cost, plot prediksi

### [`sklearn_example.py`](./sklearn_example.py)
- **Contoh siap produksi** menggunakan scikit-learn
- **Teknik lanjutan**: Polynomial features, regularization
- **Evaluasi model**: Metrik R², RMSE, MAE
- **Cross-validation** dan **hyperparameter tuning**

### [`exercise.ipynb`](./exercise.ipynb)
- **Jupyter notebook interaktif**
- **Tutorial langkah demi langkah** untuk setiap skenario
- **Latihan visualisasi**
- **Eksplorasi dataset nyata**

## 🎓 Tips untuk Latihan

1. **Selalu visualisasikan data** sebelum melakukan pemodelan
2. **Periksa asumsi** linear regression
3. **Tangani outlier** jika diperlukan
4. **Feature scaling** untuk multiple regression
5. **Validasi model** menggunakan cross-validation

## 📚 Referensi

- [Scikit-learn Linear Regression](https://scikit-learn.org/stable/modules/linear_model.html#ordinary-least-squares)
- [Introduction to Statistical Learning - Chapter 3](https://www.statlearning.com/)

## 🎓 Progres Pembelajaran

### Prasyarat
- Pengetahuan dasar Python
- Dasar-dasar NumPy ([Tutorial NumPy](https://numpy.org/doc/stable/user/quickstart.html))
- Statistik dasar (mean, variance)

### Setelah Linear Regression, jelajahi:

1. **[Logistic Regression](../02_logistic_regression/)** - Perluasan untuk klasifikasi
2. **Polynomial Features** (`06_utils/`, direncanakan) - Hubungan non-linear  
3. **[Model Evaluation](../../04_advanced_topics/03_model_evaluation/)** - Metrik lanjutan
4. **[Regularization](../10_svm/)** - Ridge/Lasso regression

### Algoritma Terkait

| Algoritma | Kesamaan | Kapan Digunakan |
|-----------|------------|-------------|
| [**Logistic Regression**](../02_logistic_regression/) | Matematika sama, output berbeda | Masalah klasifikasi |
| [**SVM Regression**](../10_svm/) | Batas linear | Tahan terhadap outlier |
| [**Neural Networks**](../08_neural_networks/) | Perluasan | Pola non-linear yang kompleks |
| [**Random Forest**](../04_random_forest/) | Pendekatan ensemble | Feature importance |

---
**➡️ Langkah Berikutnya**: Lanjutkan ke **[Logistic Regression](../02_logistic_regression/)** untuk masalah klasifikasi!
