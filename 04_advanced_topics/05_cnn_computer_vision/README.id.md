# CNN & Dasar Computer Vision

**[🇬🇧 English](README.md) | 🇮🇩 Bahasa Indonesia**

## Gambaran Umum

Convolutional Neural Networks (CNN) adalah jaringan saraf khusus untuk memproses data berbentuk grid seperti gambar. Mereka menggunakan operasi konvolusi untuk otomatis belajar hierarki spasial dari fitur.

## Operasi Utama

### 1. Konvolusi (conv2d)

Menerapkan filter untuk mendeteksi pola:

```python
from implementation import conv2d

output = conv2d(image, kernel, stride=1, padding=0)
```

**Parameter:**
- `stride`: Ukuran langkah untuk sliding window
- `padding`: Zero-padding di sekitar tepi gambar

**Kegunaan**: Deteksi tepi, pengenalan pola, ekstraksi fitur

### 2. Pooling (max_pool2d)

Downsampling dimensi spasial:

```python
from implementation import max_pool2d

pooled = max_pool2d(image, pool_size=2, stride=2)
```

**Manfaat**: Mengurangi parameter, memberikan translation invariance

### 3. Arsitektur CNN

```
Input Gambar
    ↓
Conv2D + ReLU (deteksi fitur)
    ↓
MaxPool (downsampling)
    ↓
Flatten (convert ke vektor)
    ↓
Dense Layer (klasifikasi)
    ↓
Output (prediksi)
```

## Implementasi

```python
from implementation import SimpleCNN, generate_product_images

# Generate gambar synthetic rak produk
X_train, y_train = generate_product_images(n_samples=100, random_state=42)

# Train CNN
cnn = SimpleCNN(
    input_shape=(28, 28),
    n_filters=8,
    kernel_size=3,
    learning_rate=0.01,
    random_state=42
)
cnn.fit(X_train, y_train, epochs=200)

# Prediksi status stockout
prediction = cnn.predict(new_image)  # 0 atau 1
```

## Use Case SPG Retail

**Problem**: Deteksi stockout rak dari gambar  
**Input**: Gambar grayscale 28x28 dari rak produk  
**Output**: Klasifikasi biner (0=stockout, 1=terisi baik)

**Data Synthetic**: Gambar yang di-generate secara prosedural dengan persegi panjang berwarna mewakili produk di rak.

**Aplikasi Real-World**:
- Pengecekan kepatuhan planogram
- Audit rak otomatis
- Alert stockout real-time
- Optimasi ruang rak

## Koneksi ke YOLO & Object Detection

Module ini mencakup dasar CNN, aplikasi retail modern menggunakan:

**YOLO (You Only Look Once)**:
- Object detection real-time
- Deteksi multiple produk simultan
- Memberikan bounding box dan prediksi class
- Digunakan untuk: Pengenalan produk, counting, tracking posisi

**Progresi**:
1. **Module ini**: CNN dasar untuk klasifikasi biner
2. **Level berikutnya**: Pengenalan produk multi-class
3. **Advanced**: YOLO untuk deteksi + lokalisasi simultan
4. **Production**: Integrasi dengan sistem planogram

## Kapan Menggunakan CNN

### ✅ Gunakan Jika:
- Struktur spasial penting (gambar, video)
- Butuh automatic feature learning
- Translation invariance diinginkan
- Data training cukup tersedia

### ❌ Hindari Jika:
- Data tabular/terstruktur (gunakan MLP)
- Dataset sangat kecil
- Tidak ada struktur spasial
- Interpretabilitas kritis

## File Terkait

- `implementation.py` - Conv2d, pooling, SimpleCNN from-scratch
- `exercise.ipynb` - Latihan CNN interaktif

**Selamat Belajar! 🎓** | *[🇬🇧 English](README.md)*
