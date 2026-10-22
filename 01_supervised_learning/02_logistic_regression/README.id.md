# Logistic Regression

**[🇬🇧 English](README.md) | 🇮🇩 Bahasa Indonesia**

🏠 Kembali ke Indeks Supervised: `../README.md`

Logistic Regression adalah algoritma klasifikasi fundamental yang menggunakan fungsi logistik untuk memodelkan probabilitas keanggotaan suatu kelas. Meskipun namanya mengandung "regression", algoritma ini digunakan untuk klasifikasi, bukan regresi.

## 📚 Daftar Isi
- [Teori & Konsep](#-teori--konsep)
- [Dasar Matematika](#-dasar-matematika)
- [Asumsi](#-asumsi)
- [Implementasi](#-implementasi)
- [Kelebihan & Kekurangan](#-kelebihan--kekurangan)
- [Kapan Digunakan](#-kapan-digunakan)
- [File di Direktori Ini](#-file-di-direktori-ini)

## 🧠 Teori & Konsep

### Apa itu Logistic Regression?
Logistic Regression memprediksi probabilitas bahwa suatu instance termasuk dalam kategori tertentu. Algoritma ini menggunakan **fungsi sigmoid (logistik)** untuk memetakan bilangan real apa pun ke nilai antara 0 dan 1, sehingga sangat cocok untuk klasifikasi biner.

### Konsep Kunci:
1. **Fungsi Sigmoid**: Memetakan input menjadi probabilitas (0-1)
2. **Batas Keputusan (Decision Boundary)**: Batas linear yang memisahkan kelas
3. **Maximum Likelihood Estimation**: Metode untuk menemukan parameter optimal
4. **Log-Odds**: Logaritma natural dari rasio odds

### Jenis Logistic Regression:
- **Binary**: Dua kelas (ya/tidak, spam/bukan spam)
- **Multinomial**: Banyak kelas (kucing/anjing/burung)
- **Ordinal**: Kelas berurutan (rendah/sedang/tinggi)

## 📐 Dasar Matematika

### Fungsi Sigmoid
```
σ(z) = 1 / (1 + e^(-z))
dengan z = β₀ + β₁x₁ + β₂x₂ + ... + βₙxₙ
```

### Probabilitas dan Odds
```
# Probabilitas
P(y=1|x) = σ(β₀ + β₁x₁ + β₂x₂ + ... + βₙxₙ)

# Odds
Odds = P(y=1|x) / (1 - P(y=1|x))

# Log-Odds (Logit)
logit(p) = ln(odds) = β₀ + β₁x₁ + β₂x₂ + ... + βₙxₙ
```

### Cost Function (Log-Likelihood)
```
J(θ) = -(1/m) * Σ[y*log(h(x)) + (1-y)*log(1-h(x))]
dengan h(x) = fungsi sigmoid
```

### Update Gradient Descent
```
θⱼ := θⱼ - α * (1/m) * Σ[(h(x⁽ⁱ⁾) - y⁽ⁱ⁾) * xⱼ⁽ⁱ⁾]
```

## ⚖️ Asumsi

1. **Hubungan Linear**: Antara logit outcome dan fitur
2. **Independence**: Observasi bersifat independen
3. **Tanpa Multikolinearitas Sempurna**: Fitur tidak berkorelasi sempurna
4. **Ukuran Sampel Besar**: Semakin banyak data, semakin baik akurasinya
5. **Tanpa Outlier Ekstrem**: Outlier ekstrem dapat memengaruhi model secara signifikan

## 🔧 Implementasi

### Implementasi From Scratch
Lihat `implementation.py` untuk implementasi langkah demi langkah yang mencakup:
- Fungsi sigmoid
- Perhitungan cost function
- Optimasi gradient descent
- Metode prediksi

### Menggunakan scikit-learn
Lihat `sklearn_example.py` untuk contoh praktis:
- Persiapan data
- Training model
- Metrik evaluasi
- Feature importance

## ✅ Kelebihan & Kekurangan

### ✅ Kelebihan:
- **Output Probabilistik**: Memberikan estimasi probabilitas
- **Minim Tuning**: Hyperparameter yang perlu di-tuning lebih sedikit
- **Overfitting Lebih Rendah**: Pada data berdimensi rendah
- **Interpretable**: Koefisien menunjukkan feature importance
- **Fast**: Training dan prediksi yang cepat
- **Tidak Perlu Scaling**: Tahan terhadap skala fitur

### ❌ Kekurangan:
- **Decision Boundary Linear**: Tidak dapat menangkap hubungan yang kompleks
- **Sensitif terhadap Outlier**: Nilai ekstrem dapat menggeser hasil
- **Ukuran Sampel Besar**: Membutuhkan data yang cukup besar untuk hasil stabil
- **Separasi Sempurna**: Bermasalah ketika kelas dapat dipisahkan sempurna
- **Multicollinearity**: Kesulitan dengan fitur yang saling berkorelasi

## 🎯 Kapan Digunakan

### ✅ Gunakan Logistic Regression Ketika:
- Dibutuhkan klasifikasi biner atau multi-kelas
- Anda membutuhkan estimasi probabilitas, bukan sekadar prediksi
- Dataset dapat dipisahkan secara linear atau hampir demikian
- Interpretabilitas penting
- Waktu training Anda terbatas
- Fitur memiliki hubungan linear dengan log-odds

### ❌ Jangan Gunakan Ketika:
- Anda perlu menangkap pola non-linear yang kompleks
- Dataset memiliki multikolinearitas sempurna
- Kelas dapat dipisahkan sempurna (menyebabkan masalah numerik)
- Anda memiliki dataset yang sangat kecil

### Aplikasi di Dunia Nyata:
- **Diagnosis Medis**: Prediksi penyakit berdasarkan gejala
- **Marketing**: Prediksi konversi pelanggan
- **Finance**: Persetujuan kredit, deteksi fraud
- **Email**: Klasifikasi spam
- **Media Sosial**: Analisis sentimen

## 📁 File di Direktori Ini

- **`README.md`** (file ini) - Teori dan konsep
- **`implementation.py`** - Logistic regression from scratch
- **`sklearn_example.py`** - Contoh praktis menggunakan scikit-learn
- **`exercise.ipynb`** - Latihan dan eksperimen langsung
- **`datasets/`** - Dataset contoh untuk latihan

## 🚀 Mulai Cepat

1. **Pahami Teori**: Baca README ini
2. **Lihat Implementasi**: Periksa `implementation.py`
3. **Coba Contoh**: Jalankan `sklearn_example.py`
4. **Latihan**: Selesaikan `exercise.ipynb`
5. **Eksperimen**: Gunakan dataset Anda sendiri

## 📊 Metrik Performa

Untuk masalah klasifikasi, evaluasi menggunakan:
- **Accuracy**: Tingkat kebenaran keseluruhan
- **Precision**: True positives / (True positives + False positives)
- **Recall**: True positives / (True positives + False negatives)
- **F1-Score**: Rata-rata harmonik dari precision dan recall
- **ROC-AUC**: Area di bawah kurva ROC
- **Confusion Matrix**: Rincian prediksi secara detail

## 🔗 Algoritma Terkait

- **Linear Regression**: Versi linear untuk output kontinu
- **Decision Trees**: Alternatif non-linear untuk klasifikasi
- **SVM**: Classifier linear lain dengan pendekatan berbeda
- **Naive Bayes**: Classifier probabilistik berdasarkan teorema Bayes

---

**Langkah Berikutnya**: 
- Coba implementasinya di `sklearn_example.py`
- Selesaikan latihan di `exercise.ipynb`
- Bandingkan dengan algoritma lain di `../`
