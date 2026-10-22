# Decision Trees

**[🇬🇧 English](README.md) | 🇮🇩 Bahasa Indonesia**

🏠 Kembali ke Indeks Supervised: `../README.md`

Decision Trees adalah salah satu algoritma machine learning yang paling intuitif dan mudah diinterpretasikan. Algoritma ini membuat model yang memprediksi nilai target dengan mempelajari aturan keputusan sederhana yang diturunkan dari fitur-fitur data.

## 📚 Daftar Isi
- [Teori & Konsep](#-teori--konsep)
- [Dasar Matematika](#-dasar-matematika)
- [Proses Pembentukan Tree](#-proses-pembentukan-tree)
- [Kriteria Pemisahan](#-kriteria-pemisahan)
- [Kelebihan & Kekurangan](#-kelebihan--kekurangan)
- [Kapan Digunakan](#-kapan-digunakan)
- [File di Direktori Ini](#-file-di-direktori-ini)

## 🧠 Teori & Konsep

### Apa itu Decision Trees?
Decision Trees adalah metode supervised learning non-parametrik yang digunakan untuk klasifikasi maupun regresi. Tujuannya adalah membuat model yang memprediksi nilai variabel target dengan mempelajari aturan keputusan sederhana yang diturunkan dari fitur-fitur data.

### Konsep Kunci:
1. **Root Node (Node Akar)**: Node paling atas yang merepresentasikan seluruh dataset
2. **Internal Node**: Node yang memiliki child node, merepresentasikan pengujian fitur
3. **Leaf Node**: Node terminal tanpa child, berisi prediksi
4. **Branch (Cabang)**: Koneksi antar node yang merepresentasikan jalur keputusan
5. **Depth**: Jalur terpanjang dari root ke leaf
6. **Pruning**: Menghapus node untuk mencegah overfitting

### Cara Kerja Decision Trees:
```
Apakah Feature_A > threshold?
├─ Ya → Apakah Feature_B > threshold?
│          ├─ Ya → Prediksi Kelas A
│          └─ Tidak → Prediksi Kelas B
└─ Tidak → Prediksi Kelas C
```

## 📐 Dasar Matematika

### Konsep Teori Informasi

#### Entropy (untuk Klasifikasi)
```
Entropy(S) = -Σ(p_i * log2(p_i))

Keterangan:
- S adalah dataset
- p_i adalah proporsi contoh di kelas i
- Entropy lebih rendah = data lebih murni/homogen
```

#### Gini Impurity
```
Gini(S) = 1 - Σ(p_i²)

Keterangan:
- p_i adalah proporsi contoh di kelas i
- Mengukur probabilitas salah klasifikasi elemen yang dipilih secara acak
```

#### Information Gain
```
Information_Gain(S, A) = Entropy(S) - Σ(|Sv|/|S| * Entropy(Sv))

Keterangan:
- S adalah dataset asli
- A adalah atribut/fitur
- Sv adalah subset dari S di mana atribut A bernilai v
```

#### Gain Ratio (Algoritma C4.5)
```
Gain_Ratio(S, A) = Information_Gain(S, A) / Split_Information(S, A)

Split_Information(S, A) = -Σ(|Sv|/|S| * log2(|Sv|/|S|))
```

### Untuk Regression Trees

#### Mean Squared Error (MSE)
```
MSE = (1/n) * Σ(yi - ŷ)²

Keterangan:
- yi adalah nilai aktual
- ŷ adalah nilai prediksi (mean dari leaf)
```

#### Variance Reduction
```
Variance_Reduction = Var(parent) - Σ(|child_i|/|parent| * Var(child_i))
```

## 🌳 Proses Pembentukan Tree

### 1. **Recursive Binary Splitting**
- Mulai dari root node yang berisi seluruh data training
- Temukan fitur dan titik split terbaik yang memaksimalkan information gain
- Buat dua child node berdasarkan split tersebut
- Terapkan proses yang sama secara rekursif pada child node

### 2. **Kriteria Berhenti**
- Kedalaman maksimum tercapai
- Jumlah minimum sampel per leaf
- Jumlah minimum sampel untuk melakukan split
- Tidak ada information gain lebih lanjut yang mungkin
- Semua sampel memiliki nilai target yang sama

### 3. **Proses Prediksi**
- Mulai dari root node
- Ikuti aturan keputusan menuruni tree
- Capai leaf node dan kembalikan prediksinya

## ⚖️ Kriteria Pemisahan

### Untuk Klasifikasi:
1. **Information Gain (ID3)**
   - Menggunakan entropy untuk mengukur impurity
   - Lebih memilih atribut dengan lebih banyak level
   - Dapat menyebabkan overfitting

2. **Gain Ratio (C4.5)**
   - Menormalisasi information gain dengan split information
   - Mengurangi bias terhadap atribut bernilai banyak
   - Pemisahan yang lebih seimbang

3. **Gini Index (CART)**
   - Mengukur impurity suatu dataset
   - Efisien secara komputasi
   - Kurang sensitif terhadap outlier

### Untuk Regresi:
1. **Mean Squared Error (MSE)**
   - Meminimalkan varians pada leaf node
   - Pendekatan standar untuk regression trees

2. **Mean Absolute Error (MAE)**
   - Lebih tahan terhadap outlier
   - Alternatif untuk MSE

## ✅ Kelebihan & Kekurangan

### ✅ Kelebihan:
- **Mudah Diinterpretasikan**: Mudah dipahami dan divisualisasikan
- **Tanpa Asumsi**: Bekerja dengan hubungan non-linear
- **Menangani Data Campuran**: Fitur numerik maupun kategorikal
- **Seleksi Fitur**: Feature importance implisit
- **Non-parametric**: Tidak ada asumsi tentang distribusi data
- **Nilai yang Hilang**: Dapat menangani data yang hilang secara alami
- **Prediksi Cepat**: Waktu prediksi O(log n)

### ❌ Kekurangan:
- **Overfitting**: Rentan overfitting, terutama pada tree yang dalam
- **Tidak Stabil**: Perubahan kecil pada data dapat menghasilkan tree yang berbeda
- **Bias**: Cenderung bias terhadap fitur dengan lebih banyak level
- **Ekstrapolasi Buruk**: Tidak dapat memprediksi di luar rentang data training
- **Sulit pada Hubungan Linear**: Tidak efisien untuk pola linear sederhana
- **Varians Tinggi**: Perubahan kecil pada data dapat mengubah tree secara drastis

## 🎯 Kapan Digunakan

### ✅ Gunakan Decision Trees Ketika:
- Interpretabilitas sangat penting (diagnosis medis, persetujuan pinjaman)
- Anda memiliki tipe data campuran (numerik dan kategorikal)
- Hubungan antara fitur dan target bersifat non-linear
- Anda membutuhkan peringkat feature importance
- Anda menginginkan model baseline yang mudah dipahami
- Data memiliki nilai yang hilang yang sulit diimputasi

### ❌ Jangan Gunakan Ketika:
- Anda memiliki data berdimensi tinggi dengan banyak fitur yang tidak relevan
- Hubungannya terutama linear (gunakan model linear sebagai gantinya)
- Data Anda sangat bising
- Dataset sangat kecil
- Anda membutuhkan akurasi setinggi mungkin (gunakan metode ensemble)

### Aplikasi di Dunia Nyata:
- **Diagnosis Medis**: Tree diagnosis berbasis gejala
- **Keuangan**: Credit scoring dan persetujuan pinjaman
- **Pemasaran**: Segmentasi dan penargetan pelanggan
- **Manufaktur**: Kontrol kualitas dan deteksi kerusakan
- **HR**: Penyaringan resume dan evaluasi karyawan
- **Olahraga**: Strategi permainan dan evaluasi pemain

## 🔧 Varian Tree

### 1. **ID3 (Iterative Dichotomiser 3)**
- Menggunakan information gain untuk pemisahan
- Hanya menangani fitur kategorikal
- Tidak ada mekanisme pruning
- Mudah mengalami overfitting

### 2. **C4.5**
- Peningkatan dari ID3
- Menggunakan gain ratio untuk pemisahan
- Menangani nilai kontinu dan nilai yang hilang
- Menyertakan mekanisme pruning

### 3. **CART (Classification and Regression Trees)**
- Digunakan oleh scikit-learn
- Hanya pemisahan biner
- Menggunakan Gini index untuk klasifikasi, MSE untuk regresi
- Pruning bawaan dengan cost-complexity

### 4. **CHAID (Chi-squared Automatic Interaction Detector)**
- Menggunakan uji chi-squared untuk pemisahan
- Memungkinkan pemisahan multi-arah
- Baik untuk target kategorikal

## 📁 File di Direktori Ini

- **`README.md`** (file ini) - Teori dan konsep
- **`implementation.py`** - Decision tree from scratch
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
DecisionTreeClassifier(
    criterion='gini',           # 'gini', 'entropy', 'log_loss'
    max_depth=None,            # Kedalaman maksimum tree
    min_samples_split=2,       # Jumlah sampel minimum untuk split internal node
    min_samples_leaf=1,        # Jumlah sampel minimum di leaf node
    max_features=None,         # Jumlah fitur untuk split terbaik
    max_leaf_nodes=None,       # Jumlah maksimum leaf node
    min_impurity_decrease=0.0, # Penurunan impurity minimum untuk split
    random_state=None          # Random seed untuk reproduksibilitas
)
```

### Penjelasan Parameter Kunci:
- **max_depth**: Mengontrol overfitting, semakin dalam = semakin kompleks
- **min_samples_split**: Nilai lebih tinggi mencegah overfitting
- **min_samples_leaf**: Memastikan leaf memiliki sampel minimum
- **max_features**: Mengontrol keacakan dan kecepatan
- **criterion**: Ukuran kualitas split yang berbeda-beda

## 📊 Metrik Evaluasi

### Klasifikasi:
- **Accuracy**: Tingkat kebenaran keseluruhan
- **Precision/Recall**: Performa spesifik per kelas
- **F1-Score**: Precision dan recall yang seimbang
- **ROC-AUC**: True positive vs false positive rate
- **Feature Importance**: Fitur mana yang paling berguna

### Regresi:
- **Mean Squared Error (MSE)**: Rata-rata selisih kuadrat
- **Mean Absolute Error (MAE)**: Rata-rata selisih absolut
- **R² Score**: Coefficient of determination
- **Feature Importance**: Dampak setiap fitur

## 🌲 Visualisasi Tree

Decision trees dapat divisualisasikan menggunakan:
```python
from sklearn.tree import plot_tree, export_text
import matplotlib.pyplot as plt

# Plot tree
plt.figure(figsize=(20,10))
plot_tree(model, feature_names=feature_names, 
          class_names=class_names, filled=True)

# Representasi teks
tree_rules = export_text(model, feature_names=feature_names)
print(tree_rules)
```

## 🔗 Algoritma Terkait

- **Random Forest**: Ensemble dari decision trees
- **Gradient Boosting**: Pembelajaran tree secara sekuensial
- **Extra Trees**: Tree yang sangat teracak
- **XGBoost/LightGBM**: Gradient boosting yang dioptimalkan

---

**Langkah Berikutnya**: 
- Coba implementasinya di `sklearn_example.py`
- Selesaikan latihan di `exercise.ipynb`
