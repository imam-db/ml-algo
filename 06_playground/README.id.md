# 🎮 Playground Algoritma

**[🇬🇧 English](README.md) | 🇮🇩 Bahasa Indonesia**

**🏠 [Kembali ke Utama](../README.md)** | **🎯 [Progress Tracker](../PROGRESS_TRACKER.md)** | **🔬 [Perbandingan Algoritma](../ALGORITHM_COMPARISON.md)**

Alat interaktif untuk bereksperimen dengan algoritma machine learning, memvisualisasikan perilakunya, dan memahami cara kerjanya di balik layar.

---

## 🎨 Alat Interaktif

### **🎛️ Playground Parameter**
**[`parameter_playground.py`](./parameter_playground.py)**

Visualisasikan bagaimana hyperparameter memengaruhi perilaku algoritma secara real-time.

```bash
# Tuning parameter interaktif dengan visualisasi langsung
uv run python parameter_playground.py

# Coba berbagai algoritma
uv run python parameter_playground.py --algorithm linear_regression
uv run python parameter_playground.py --algorithm xgboost
```

**Fitur:**
- Penyesuaian parameter real-time dengan slider
- Visualisasi langsung dari perubahan model
- Pembaruan metrik performa
- Mode perbandingan berdampingan

---

### **🎲 Lab Generator Data**
**[`data_generator.py`](./data_generator.py)**

Buat dataset sintetis untuk pengujian dan eksperimen algoritma.

```bash
# Hasilkan berbagai tipe dataset
uv run python data_generator.py --type linear
uv run python data_generator.py --type clusters --n_clusters 4
uv run python data_generator.py --type moons --noise 0.2
```

**Tipe Dataset:**
- **Linear**: Hubungan linear sederhana
- **Polynomial**: Kurva polinomial non-linear
- **Clusters**: Cluster Gaussian untuk klasifikasi
- **Moons**: Data berbentuk bulan sabit
- **Circles**: Lingkaran konsentris
- **Blobs**: Cluster blob acak
- **Spiral**: Pola spiral
- **Custom**: Tentukan pola Anda sendiri

---

### **🎯 Visualizer Model**
**[`model_visualizer.py`](./model_visualizer.py)**

Lihat decision boundary dan wilayah klasifikasi untuk berbagai algoritma.

```bash
# Visualisasikan decision boundary
uv run python model_visualizer.py --algorithm svm
uv run python model_visualizer.py --algorithm random_forest --data_type moons
```

**Visualisasi:**
- Decision boundary untuk klasifikasi
- Garis regresi dan confidence interval
- Plot feature importance
- Confusion matrix
- Kurva ROC dan kurva precision-recall

---

### **🏁 Balapan Algoritma**
**[`algorithm_race.py`](./algorithm_race.py)**

Bandingkan beberapa algoritma secara berdampingan pada dataset yang sama dengan analisis performa yang komprehensif.

```bash
# Uji cepat dengan data yang dihasilkan
uv run python algorithm_race.py --quick_test

# Balapan dengan dataset Anda sendiri
uv run python algorithm_race.py --data_path your_dataset.csv

# Balapan algoritma tertentu
uv run python algorithm_race.py --data_path data.csv --algorithms "random_forest,svm,logistic_regression"

# Sertakan pengujian signifikansi statistik
uv run python algorithm_race.py --data_path data.csv --stats_test
```

**Fitur:**
- **Deteksi Masalah Otomatis**: Klasifikasi vs regresi
- **Metrik Komprehensif**: Accuracy, precision, recall, F1, R², RMSE, dan sebagainya
- **Visualisasi Performa**: Grafik perbandingan 4 panel
- **Pengujian Statistik**: Pairwise t-test antar algoritma
- **Analisis Kecepatan**: Benchmark waktu training dan prediksi
- **Cross-validation**: Evaluasi performa yang kokoh
- **Penentuan Pemenang**: Algoritma terbaik dengan analisis detail

**Algoritma yang Didukung:**
- Klasifikasi: Logistic Regression, Random Forest, SVM, Naive Bayes, KNN, Decision Tree, XGBoost
- Regresi: Linear Regression, Random Forest, SVM, KNN, Decision Tree, XGBoost

---

### **📈 Penjelajah Learning Curve**
**[`learning_curves.py`](./learning_curves.py)**

Amati bagaimana algoritma belajar seiring waktu dan iterasi.

```bash
# Visualisasikan proses pembelajaran
uv run python learning_curves.py --algorithm gradient_descent
uv run python learning_curves.py --algorithm xgboost --show_validation
```

**Visualisasi:**
- Kurva training vs validation
- Jalur optimasi cost function
- Analisis konvergensi
- Deteksi overfitting
- Visualisasi early stopping

---

### **⚗️ Lab Feature Engineering**
**[`feature_lab.py`](./feature_lab.py)**

Eksperimen dengan berbagai transformasi fitur dan lihat efeknya.

```bash
# Feature engineering interaktif
uv run python feature_lab.py
uv run python feature_lab.py --transformations polynomial,scaling,selection
```

**Transformasi:**
- Polynomial features
- Feature scaling (StandardScaler, MinMaxScaler)
- Feature selection (SelectKBest, RFE)
- Reduksi dimensi PCA
- Transformasi kustom
- Interaction features

---

### **🎬 Animasi Algoritma**
**[`algorithm_animator.py`](./algorithm_animator.py)**

Animasi visual langkah demi langkah tentang cara kerja internal algoritma.

```bash
# Animasi langkah-langkah algoritma
uv run python algorithm_animator.py --algorithm kmeans
uv run python algorithm_animator.py --algorithm gradient_descent --steps 50
```

**Animasi yang Tersedia:**
- Iterasi clustering K-Means
- Optimasi gradient descent
- Pemisahan decision tree
- Pencocokan linear regression
- Training neural network

---

## 🚀 Panduan Mulai Cepat

### **1. 🎯 Persiapan Pertama**
```bash
# Instal dependensi visualisasi tambahan
uv add matplotlib seaborn plotly ipywidgets

# Uji playground
uv run python parameter_playground.py --test
```

### **2. 🎮 Mode Interaktif**
```bash
# Jalankan playground interaktif
uv run python 06_playground/playground_launcher.py
```

### **3. 🔬 Eksperimen dengan Algoritma**
```bash
# Perbandingan algoritma cepat
uv run python algorithm_race.py --quick_test

# Hasilkan data uji
uv run python data_generator.py --interactive

# Visualisasikan algoritma apa pun
uv run python model_visualizer.py --interactive
```

---

## 🎓 Kasus Penggunaan Edukatif

### **👶 Untuk Pemula**
- **Memahami Dasar**: Gunakan `parameter_playground.py` untuk melihat bagaimana perubahan learning rate memengaruhi linear regression
- **Eksplorasi Data**: Hasilkan berbagai dataset dengan `data_generator.py` untuk memahami pola data
- **Belajar Visual**: Amati cara kerja algoritma dengan `algorithm_animator.py`

### **📚 Untuk Pelajar Tingkat Menengah**
- **Parameter Tuning**: Kuasai efek hyperparameter dengan slider interaktif
- **Perbandingan Algoritma**: Balapan algoritma untuk memahami trade-off
- **Feature Engineering**: Eksperimen dengan transformasi di `feature_lab.py`

### **🚀 Untuk Pengguna Lanjutan**
- **Analisis Algoritma**: Selami learning curve dan konvergensi
- **Eksperimen Kustom**: Modifikasi alat playground untuk penelitian tertentu
- **Alat Pengajaran**: Gunakan visualisasi untuk presentasi dan tutorial

---

## 🛠️ Arsitektur Playground

### **Komponen Inti**
```
06_playground/
├── README.md                    # File ini
├── playground_launcher.py       # Launcher interaktif utama
├── parameter_playground.py      # Alat visualisasi parameter
├── data_generator.py           # Pembuatan data sintetis
├── model_visualizer.py         # Plot decision boundary  
├── algorithm_race.py           # Perbandingan multi-algoritma
├── learning_curves.py          # Analisis learning curve
├── feature_lab.py              # Eksperimen feature engineering
├── algorithm_animator.py       # Animasi langkah demi langkah
├── utils/                      # Utilitas bersama
│   ├── __init__.py
│   ├── visualization.py       # Fungsi plotting umum
│   ├── data_utils.py          # Utilitas pembuatan data
│   └── algorithm_utils.py     # Utilitas wrapper algoritma
└── examples/                   # Notebook contoh
    ├── playground_tour.ipynb   # Tur interaktif
    ├── parameter_tuning.ipynb  # Eksplorasi parameter
    └── visual_learning.ipynb   # Panduan algoritma visual
```

### **Titik Integrasi**
- **Implementasi Algoritma**: Menggunakan algoritma dari `01_supervised_learning/`, `02_unsupervised_learning/`
- **Dataset**: Dapat memuat data CSV; folder bersama `05_datasets/` direncanakan
- **Pelacakan Progres**: Integrasi dengan `progress_tracker.py`
- **Terminologi**: Tautan ke `GLOSSARY.md` untuk penjelasan

---

## 🎯 Tujuan Pembelajaran

Setelah menggunakan Playground Algoritma, Anda akan:

### **🎓 Pemahaman Konseptual**
- [ ] **Memvisualisasikan** bagaimana berbagai algoritma membuat decision boundary
- [ ] **Memahami** efek hyperparameter terhadap perilaku model
- [ ] **Membandingkan** performa algoritma pada berbagai tipe data
- [ ] **Mengenali** pola overfitting dan underfitting

### **🔬 Keterampilan Praktis**  
- [ ] **Menghasilkan** dataset sintetis untuk pengujian algoritma
- [ ] **Bereksperimen** dengan transformasi feature engineering
- [ ] **Menganalisis** learning curve dan pola konvergensi
- [ ] **Mengoptimalkan** hyperparameter melalui umpan balik visual

### **🧠 Berpikir Analitis**
- [ ] **Memprediksi** algoritma mana yang paling cocok untuk pola data tertentu
- [ ] **Mendiagnosis** masalah model melalui visualisasi
- [ ] **Merancang** eksperimen untuk menguji hipotesis algoritma
- [ ] **Mengomunikasikan** perilaku algoritma melalui bukti visual

---

## 🎮 Contoh Interaktif

### **Contoh 1: Memahami Overfitting**
```bash
# Hasilkan data polinomial
uv run python data_generator.py --type polynomial --noise 0.1 --save overfitting_data.csv

# Bandingkan berbagai kompleksitas model
uv run python parameter_playground.py --data overfitting_data.csv --algorithm polynomial_regression

# Sesuaikan derajat polinomial dan amati performa validasi
# Derajat 1: Underfitting (bias tinggi)
# Derajat 3: Fit yang baik  
# Derajat 15: Overfitting (variance tinggi)
```

### **Contoh 2: Pemilihan Algoritma**
```bash
# Buat dataset klasifikasi  
uv run python data_generator.py --type moons --noise 0.3 --save moons_data.csv

# Balapan berbagai classifier
uv run python algorithm_race.py --data moons_data.csv --algorithms logistic_regression,svm,random_forest

# Visualisasikan decision boundary
uv run python model_visualizer.py --data moons_data.csv --algorithm svm --kernel rbf
```

### **Contoh 3: Dampak Feature Engineering**
```bash
# Hasilkan data non-linear
uv run python data_generator.py --type circles --save circles_data.csv

# Uji model linear (akan gagal)
uv run python model_visualizer.py --data circles_data.csv --algorithm logistic_regression

# Terapkan polynomial features dan lihat peningkatannya  
uv run python feature_lab.py --data circles_data.csv --transform polynomial --degree 2
```

---

## 💡 Tips Pro

### **🎯 Eksperimen yang Efektif**
1. **Mulai Sederhana**: Awali dengan data linear dan algoritma sederhana
2. **Satu Variabel**: Ubah satu parameter pada satu waktu untuk memahami efeknya
3. **Visualisasi Dulu**: Selalu visualisasikan sebelum menyelami metrik
4. **Bandingkan**: Gunakan alat balapan untuk memahami trade-off

### **🔬 Pendekatan Ilmiah**
1. **Hipotesis**: Buat prediksi sebelum menjalankan eksperimen
2. **Kontrol**: Jaga variabel lain tetap konstan saat menguji
3. **Ulangi**: Jalankan eksperimen beberapa kali dengan random seed berbeda
4. **Dokumentasikan**: Simpan temuan menarik dan kombinasi parameter

### **🎨 Praktik Terbaik Visualisasi**
1. **Skala yang Tepat**: Gunakan skala log untuk metrik dengan rentang lebar
2. **Kode Warna**: Warna yang konsisten untuk algoritma di semua plot
3. **Label Jelas**: Selalu beri label sumbu dan sertakan legenda
4. **Interaktif**: Gunakan slider dan widget untuk eksplorasi real-time

---

## 🔗 Sumber Daya Terkait

### **Materi Pembelajaran**
- **[Perbandingan Algoritma](../ALGORITHM_COMPARISON.md)** - Pilih algoritma yang tepat
- **[Glosarium](../GLOSSARY.md)** - Pahami terminologi
- **[Progress Tracker](../PROGRESS_TRACKER.md)** - Lacak pembelajaran Anda

### **Referensi Implementasi**
- **[Linear Regression](../01_supervised_learning/01_linear_regression/)** - Teori dan implementasi
- **[XGBoost](../04_advanced_topics/04_xgboost/)** - Metode ensemble lanjutan
- **[Model Evaluation](../04_advanced_topics/03_model_evaluation/)** - Metrik performa

### **Alat Pengembangan**
- **Alur kerja pengembangan** (`WARP.md`, direncanakan)
- **Jupyter Notebooks** - Lingkungan pengembangan interaktif

---

**🎮 Siap Bermain?** Mulai dengan playground launcher dan jelajahi dunia machine learning yang interaktif!

```bash
uv run python 06_playground/playground_launcher.py
```
