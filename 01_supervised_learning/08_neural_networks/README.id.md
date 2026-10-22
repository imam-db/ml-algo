# Neural Networks (Multi-Layer Perceptron)

**[🇬🇧 English](README.md) | 🇮🇩 Bahasa Indonesia**

## Gambaran Umum

Neural Networks (khususnya Multi-Layer Perceptron) adalah algoritma supervised learning yang terinspirasi dari jaringan saraf biologis. Mereka belajar pola non-linear yang kompleks melalui beberapa layer neuron yang saling terhubung.

## Arsitektur

```
Input Layer → Hidden Layer(s) → Output Layer
    ↓              ↓                 ↓
  Fitur       ReLU/Sigmoid        Softmax
```

### Komponen Utama

**1. Layer**
- Input: Fitur dari data
- Hidden: Belajar representasi perantara
- Output: Prediksi akhir

**2. Fungsi Aktivasi**
- **ReLU**: f(x) = max(0, x) - Paling umum untuk hidden layer
- **Sigmoid**: f(x) = 1/(1+e^-x) - Klasifikasi biner
- **Softmax**: Probabilitas multi-class

**3. Training**
- Forward pass: Hitung prediksi
- Loss: Cross-entropy untuk klasifikasi
- Backpropagation: Hitung gradien
- Update: Mini-batch SGD

## Implementasi

```python
from implementation import MLPClassifier, generate_spg_outlet_data

# Generate data prediksi stockout outlet SPG
X, y = generate_spg_outlet_data(n_samples=500, random_state=42)

# Train MLP
mlp = MLPClassifier(
    hidden_layers=[32, 16],
    activation='relu',
    learning_rate=0.01,
    max_iter=500,
    random_state=42
)
mlp.fit(X_train, y_train)

# Prediksi
predictions = mlp.predict(X_test)
probabilities = mlp.predict_proba(X_test)
```

## Gradient Checking

Verifikasi implementasi backpropagation:

```python
from implementation import gradient_check

max_diff, rel_diff = gradient_check(mlp, X_sample, y_sample)
print(f"Relative difference: {rel_diff:.2e}")  # Harus < 1e-4
```

## Use Case SPG

**Problem**: Prediksi risiko stockout outlet  
**Fitur**: Sales, jumlah SKU, frekuensi visit, OOS rate, basket size  
**Kelas**: 0=Risiko rendah, 1=Risiko sedang, 2=Risiko tinggi

Neural networks menangkap pola kompleks seperti:
- Hubungan non-linear antar fitur
- Efek interaksi (misal: sales tinggi + frekuensi rendah = risiko stockout)
- Pola tersembunyi dalam data historis

## Kapan Menggunakan

### ✅ Gunakan Jika:
- Pola non-linear dalam data
- Data training cukup (ratusan-ribuan sampel)
- Interaksi fitur penting
- Model black-box dapat diterima

### ❌ Hindari Jika:
- Dataset kecil (< 100 sampel)
- Butuh interpretabilitas model
- Pola linear sudah cukup
- Resource komputasi terbatas

## File Terkait

- `implementation.py` - MLP from-scratch dengan backpropagation
- `sklearn_example.py` - Perbandingan dengan sklearn
- `exercise.ipynb` - Latihan interaktif

**Selamat Belajar! 🎓** | *[🇬🇧 English](README.md)*
