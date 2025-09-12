# Linear Regression

## 📖 Teori dan Konsep

### Apa itu Linear Regression?
Linear Regression adalah algoritma supervised learning yang digunakan untuk memprediksi nilai target (continuous) berdasarkan satu atau lebih fitur input. Algoritma ini mencari hubungan linear terbaik antara input dan output.

### Rumus Matematika
**Simple Linear Regression (1 variable):**
```
y = β₀ + β₁x + ε
```

**Multiple Linear Regression (multiple variables):**
```
y = β₀ + β₁x₁ + β₂x₂ + ... + βₙxₙ + ε
```

Dimana:
- `y` = variabel target (dependent)
- `x` = variabel fitur (independent) 
- `β₀` = intercept (bias)
- `β₁, β₂, ..., βₙ` = koefisien (slope)
- `ε` = error/noise

### Cost Function
Linear regression menggunakan **Mean Squared Error (MSE)** sebagai cost function:
```
MSE = (1/n) * Σ(yi - ŷi)²
```

### Optimasi
Untuk mencari parameter terbaik, dapat menggunakan:
1. **Normal Equation** (Closed-form solution)
2. **Gradient Descent** (Iterative approach)

## 🎯 Kapan Menggunakan Linear Regression?

### ✅ Cocok untuk:
- Prediksi nilai kontinu (harga rumah, suhu, gaji, dll)
- Data dengan hubungan linear
- Interpretasi model yang mudah dipahami
- Baseline model untuk perbandingan
- Dataset berukuran kecil hingga menengah

### ❌ Tidak cocok untuk:
- Data dengan hubungan non-linear kompleks
- Outliers yang banyak
- Target variable yang categorical
- Multicollinearity yang tinggi

## 📊 Asumsi Linear Regression

1. **Linearity**: Hubungan linear antara X dan y
2. **Independence**: Observasi saling independen
3. **Homoscedasticity**: Variance error konstan
4. **Normal Distribution**: Error terdistribusi normal
5. **No Multicollinearity**: Fitur tidak berkorelasi tinggi

## 📈 Evaluasi Model

### Metrics untuk Regression:
- **R² (Coefficient of Determination)**: Proporsi variance yang dijelaskan
- **MSE (Mean Squared Error)**: Rata-rata kuadrat error
- **RMSE (Root Mean Squared Error)**: Akar kuadrat MSE
- **MAE (Mean Absolute Error)**: Rata-rata absolute error

## 🔍 Kelebihan dan Kekurangan

### Kelebihan:
- ✅ Sederhana dan cepat
- ✅ Mudah diinterpretasi
- ✅ Tidak memerlukan tuning parameter yang kompleks
- ✅ Baseline yang baik
- ✅ Tidak prone terhadap overfitting

### Kekurangan:
- ❌ Hanya bisa menangkap hubungan linear
- ❌ Sensitif terhadap outliers
- ❌ Asumsi yang ketat
- ❌ Performa buruk untuk data non-linear

## 📝 Implementasi

Dalam folder ini, Anda akan menemukan:
- `implementation.py` - Implementasi dari scratch menggunakan NumPy
- `sklearn_example.py` - Contoh menggunakan scikit-learn
- `exercise.ipynb` - Latihan praktis dengan dataset real

## 🎓 Tips untuk Praktik

1. **Selalu visualisasikan data** sebelum modeling
2. **Check asumsi** linear regression
3. **Handle outliers** jika diperlukan
4. **Feature scaling** untuk multiple regression
5. **Validate model** menggunakan cross-validation

## 📚 Referensi

- [Scikit-learn Linear Regression](https://scikit-learn.org/stable/modules/linear_model.html#ordinary-least-squares)
- [Introduction to Statistical Learning - Chapter 3](https://www.statlearning.com/)

---
**Next Step**: Setelah memahami Linear Regression, lanjutkan ke **Logistic Regression** untuk classification problems!