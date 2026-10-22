# Dataset Latihan Random Forest 📊

**[🇬🇧 English](README.md) | 🇮🇩 Bahasa Indonesia**

Folder ini berisi dataset contoh untuk berlatih algoritma Random Forest. Setiap dataset dirancang untuk menunjukkan aspek berbeda dari aplikasi Random Forest.

## 📁 Dataset yang Tersedia

### 1. Customer Churn (customer_churn.csv)
- **Tipe**: Klasifikasi Biner
- **Target**: `churn` (0/1)
- **Ukuran**: ~2.000 sampel, 13 fitur
- **Kasus Penggunaan**: Memprediksi churn pelanggan untuk layanan berlangganan
- **Fitur Utama**: Usia, tenure, biaya, tipe kontrak, layanan
- **Tantangan**: Ketidakseimbangan kelas moderat (~30% tingkat churn)

### 2. House Prices (house_prices.csv)
- **Tipe**: Regresi
- **Target**: `price` (kontinu)
- **Ukuran**: ~1.500 sampel, 13 fitur
- **Kasus Penggunaan**: Prediksi harga properti
- **Fitur Utama**: Luas bangunan, kamar tidur, kamar mandi, lokasi, usia
- **Tantangan**: Fitur numerik dan kategorikal bercampur

### 3. Credit Approval (credit_approval.csv)
- **Tipe**: Klasifikasi Biner (Tidak Seimbang)
- **Target**: `approved` (0/1)
- **Ukuran**: ~1.000 sampel, 10 fitur
- **Kasus Penggunaan**: Prediksi persetujuan aplikasi kredit
- **Fitur Utama**: Credit score, pendapatan, rasio utang terhadap pendapatan
- **Tantangan**: Sangat tidak seimbang (~25% tingkat persetujuan)

### 4. Marketing Campaign (marketing_campaign.csv)
- **Tipe**: Klasifikasi Multi-kelas
- **Target**: `response` (0: Tidak ada respons, 1: Bertanya, 2: Membeli)
- **Ukuran**: ~1.200 sampel, 11 fitur
- **Kasus Penggunaan**: Prediksi respons kampanye marketing
- **Fitur Utama**: Demografi pelanggan, riwayat pembelian, engagement
- **Tantangan**: Multi-kelas dengan kelas yang tidak seimbang

### 5. Employee Satisfaction (employee_satisfaction.csv)
- **Tipe**: Regresi
- **Target**: `satisfaction` (skala 1-10)
- **Ukuran**: ~800 sampel, 13 fitur
- **Kasus Penggunaan**: Memprediksi skor kepuasan karyawan
- **Fitur Utama**: Gaji, jam kerja, tunjangan, departemen
- **Tantangan**: Tipe fitur campuran, target ordinal

## 🚀 Contoh Penggunaan

### Memuat Data
```python
import pandas as pd

# Muat dataset apa pun
df = pd.read_csv('customer_churn.csv')
print(df.head())
print(df.info())
```

### Contoh Dasar Random Forest
```python
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

# Muat data
df = pd.read_csv('customer_churn.csv')

# Siapkan fitur (tangani variabel kategorikal)
# ... langkah preprocessing ...

# Latih model
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(X_train, y_train)
```

## 🎯 Tujuan Pembelajaran

Gunakan dataset ini untuk berlatih:

1. **Praproses Data**: Menangani variabel kategorikal dan nilai yang hilang
2. **Rekayasa Fitur**: Membuat fitur baru, menangani berbagai tipe data
3. **Pelatihan Model**: Berbagai konfigurasi Random Forest
4. **Tuning Hyperparameter**: Grid search, random search
5. **Evaluasi Model**: Berbagai metrik untuk berbagai tipe masalah
6. **Feature Importance**: Memahami fitur mana yang paling penting
7. **Data Tidak Seimbang**: Menangani dataset dengan distribusi kelas yang tidak merata
8. **Masalah Multi-kelas**: Melampaui klasifikasi biner
9. **Regression**: Menerapkan Random Forest pada target kontinu
10. **Aplikasi di Dunia Nyata**: Berlatih pada masalah bisnis yang realistis

## 📚 Latihan yang Disarankan

1. **Bandingkan Algoritma**: Random Forest vs. Decision Tree vs. algoritma lain
2. **Seleksi Fitur**: Gunakan feature importance Random Forest untuk pemilihan fitur
3. **Cross-Validation**: Evaluasi model yang kokoh
4. **Analisis Ensemble**: Pelajari bagaimana jumlah tree memengaruhi performa
5. **Interpretasi**: Gunakan feature importance dan partial dependence plot
6. **Pipeline Produksi**: Bangun pipeline ML lengkap dari data hingga prediksi

## 🔧 Pembuatan Data

Semua dataset dibuat secara sintetis menggunakan model statistik yang realistis. Skrip pembuatan (`generate_datasets.py`) dapat dimodifikasi untuk membuat dataset dengan karakteristik berbeda:

- Ukuran sampel
- Jumlah fitur
- Distribusi kelas
- Tingkat noise
- Hubungan antar fitur

Selamat belajar! 🌲🎓
