# Naive Bayes

**🏠 [Kembali ke Menu Utama](../../README.md)** | **📁 [Supervised Learning](../README.md)**

## 🗺️ Akses Cepat

| File | Deskripsi | Jalankan |
|------|-----------|----------|
| [`implementation.py`](./implementation.py) | Implementasi dari nol | `python3 implementation.py` |
| [`sklearn_example.py`](./sklearn_example.py) | Contoh scikit-learn | `python3 sklearn_example.py` |
| [`exercise.ipynb`](./exercise.ipynb) | Latihan interaktif | `jupyter lab exercise.ipynb` |

## 📖 Teori dan Konsep

### Apa itu Naive Bayes?

**Naive Bayes** adalah algoritma klasifikasi yang berdasarkan pada **Bayes' Theorem** dengan asumsi "naive" (naif) bahwa semua features adalah independent satu sama lain. Meskipun asumsi ini sering tidak realistis, Naive Bayes tetap sangat efektif untuk banyak aplikasi real-world.

**Analogi SPG (Sales Promoter Girl):**
Seorang SPG ingin memprediksi apakah customer akan membeli produk berdasarkan karakteristik customer (usia, gender, waktu kunjungan, dll). Naive Bayes melihat seberapa sering kombinasi karakteristik tersebut muncul pada customer yang membeli vs tidak membeli, kemudian membuat prediksi berdasarkan probabilitas.

### Bayes' Theorem

Formula dasar:

```
P(Class|Features) = P(Features|Class) × P(Class) / P(Features)
```

Dalam praktik:

```
P(y|x₁, x₂, ..., xₙ) = P(x₁, x₂, ..., xₙ|y) × P(y) / P(x₁, x₂, ..., xₙ)
```

### Naive Assumption

Asumsi naive (independent features):

```
P(x₁, x₂, ..., xₙ|y) = P(x₁|y) × P(x₂|y) × ... × P(xₙ|y)
```

Sehingga Bayes' Theorem menjadi:

```
P(y|x₁, ..., xₙ) ∝ P(y) × P(x₁|y) × P(x₂|y) × ... × P(xₙ|y)
```

## 🔢 Jenis-jenis Naive Bayes

### 1. Gaussian Naive Bayes

**Untuk:** Continuous features (data kontinu)

**Asumsi:** Features mengikuti distribusi Gaussian (normal)

**Formula Likelihood:**
```
P(xᵢ|y) = (1/√(2πσᵧ²)) × exp(-(xᵢ-μᵧ)²/(2σᵧ²))
```

Di mana:
- `μᵧ` = mean feature untuk class y
- `σᵧ²` = variance feature untuk class y

**Kapan digunakan:**
- Features berupa angka continuous (tinggi, berat, harga, dll)
- Iris classification, medical diagnosis
- Sensor data analysis

**Contoh Use Case SPG:**
```python
# Prediksi customer segment berdasarkan:
# - Usia (continuous)
# - Penghasilan (continuous)
# - Spending score (continuous)
X = [[25, 50000, 65], [45, 80000, 80], ...]
y = ['budget', 'premium', ...]
```

### 2. Multinomial Naive Bayes

**Untuk:** Discrete count data (data count/frekuensi)

**Asumsi:** Features adalah count atau frequency

**Formula Likelihood:**
```
P(xᵢ|y) = (countᵢᵧ + α) / (total_countᵧ + α × n_features)
```

Di mana:
- `α` = smoothing parameter (Laplace/Lidstone)
- `countᵢᵧ` = count feature i untuk class y

**Kapan digunakan:**
- Text classification (word counts, TF-IDF)
- Document categorization
- Spam filtering

**Contoh Use Case SPG:**
```python
# Prediksi kategori customer berdasarkan:
# - Jumlah kunjungan per minggu
# - Jumlah produk dibeli per kategori
# - Frekuensi pembelian
X = [[5, 2, 1, 3], [1, 0, 5, 2], ...]  # counts
y = ['frequent', 'occasional', ...]
```

### 3. Bernoulli Naive Bayes (tidak diimplementasikan)

**Untuk:** Binary features (0/1, True/False)

**Contoh:** Email spam (word present or not)

## 💡 Kelebihan dan Kekurangan

### ✅ Kelebihan

1. **Fast Training dan Prediction**
   - Training: O(n × d) - sangat cepat
   - Prediction: O(c × d) - real-time capable

2. **Works Well dengan Small Data**
   - Tidak perlu banyak training data
   - Good dengan imbalanced classes

3. **Simple dan Interpretable**
   - Easy to understand
   - Probabilitas output dapat diinterpretasi

4. **Handles High Dimensions**
   - Works dengan banyak features
   - Common untuk text classification

5. **Tidak Sensitive ke Irrelevant Features**
   - Independent assumption helps

### ❌ Kekurangan

1. **Independence Assumption**
   - Features jarang truly independent
   - Dapat menghasilkan overconfident predictions

2. **Zero Frequency Problem**
   - Jika feature tidak muncul di training data
   - Solved dengan smoothing (Laplace)

3. **Bad Estimator**
   - Probabilitas output sering terlalu ekstrim
   - Don't rely on exact probability values

4. **Can't Learn Feature Interactions**
   - Tidak bisa capture correlations antar features

## 🎯 Kapan Menggunakan Naive Bayes?

### ✅ Cocok untuk:
- **Text classification**
  - Spam detection, sentiment analysis
  - Document categorization
- **Real-time prediction**
  - Sangat cepat, suitable untuk production
- **Multi-class problems**
  - Natural extension
- **Baseline model**
  - Quick benchmark
- **High-dimensional data**
  - Banyak features
- **Small training data**
  - Works dengan data terbatas

### ❌ Kurang cocok untuk:
- **Features yang highly correlated**
  - Melanggar independence assumption
- **Regression problems**
  - Naive Bayes untuk classification only
- **Feature interactions penting**
  - Tidak bisa capture complex relationships
- **Need precise probability estimates**
  - Probabilities sering overconfident

## 🔧 Implementasi

### Gaussian Naive Bayes

```python
from implementation import GaussianNaiveBayes

# Create and train
gnb = GaussianNaiveBayes(var_smoothing=1e-9)
gnb.fit(X_train, y_train)

# Predict
predictions = gnb.predict(X_test)
probabilities = gnb.predict_proba(X_test)
log_probs = gnb.predict_log_proba(X_test)

# Score
accuracy = gnb.score(X_test, y_test)
```

### Multinomial Naive Bayes

```python
from implementation import MultinomialNaiveBayes

# Create and train (dengan Laplace smoothing)
mnb = MultinomialNaiveBayes(alpha=1.0)
mnb.fit(X_train_counts, y_train)

# Predict
predictions = mnb.predict(X_test_counts)
probabilities = mnb.predict_proba(X_test_counts)

# Score
accuracy = mnb.score(X_test_counts, y_test)
```

## 📊 Laplace Smoothing

**Problem:** Zero frequency untuk unseen features

**Example:**
```
Training: "Buy now" → spam
          "Meeting tomorrow" → not spam

Test: "Buy tomorrow" → P("tomorrow"|spam) = 0!
```

**Solution: Laplace Smoothing**

```
P(word|class) = (count(word, class) + α) / (total_count(class) + α × vocab_size)
```

**Parameter α:**
- α = 0: No smoothing (risky!)
- α = 1: Laplace smoothing (standard)
- α > 1: More smoothing (untuk very sparse data)
- α < 1: Less smoothing

**Tips:**
- Start dengan α = 1.0
- Tune dengan cross-validation
- Larger α untuk smaller datasets

## 🔍 Hyperparameter Tuning

### Gaussian Naive Bayes

```python
# var_smoothing: portion of largest variance added
# Range: 1e-10 to 1e-7

for var_smooth in [1e-10, 1e-9, 1e-8, 1e-7]:
    gnb = GaussianNaiveBayes(var_smoothing=var_smooth)
    gnb.fit(X_train, y_train)
    score = gnb.score(X_val, y_val)
    print(f"var_smoothing={var_smooth}: {score:.4f}")
```

### Multinomial Naive Bayes

```python
# alpha: smoothing parameter
# Range: 0.01 to 10.0

for alpha in [0.01, 0.1, 1.0, 5.0, 10.0]:
    mnb = MultinomialNaiveBayes(alpha=alpha)
    mnb.fit(X_train, y_train)
    score = mnb.score(X_val, y_val)
    print(f"alpha={alpha}: {score:.4f}")
```

## 💻 Contoh Use Case: Customer Purchase Prediction

### Gaussian NB: Continuous Features

```python
# Features: age, income, spending_score
# Target: will_purchase (yes/no)

from implementation import GaussianNaiveBayes
import numpy as np

# Data customer SPG
X = np.array([
    [25, 45000, 60],  # young, low income, medium spending
    [45, 95000, 85],  # older, high income, high spending
    [30, 55000, 40],  # young adult, medium income, low spending
    # ... more data
])

y = np.array([0, 1, 0, ...])  # 0=no purchase, 1=purchase

# Train
gnb = GaussianNaiveBayes()
gnb.fit(X, y)

# Predict new customer
new_customer = [[35, 70000, 75]]
prediction = gnb.predict(new_customer)
probability = gnb.predict_proba(new_customer)

print(f"Prediction: {'Purchase' if prediction[0] == 1 else 'No Purchase'}")
print(f"Probability: {probability[0]}")
```

### Multinomial NB: Count Features

```python
# Features: visit counts per category (electronics, fashion, food, home)
# Target: customer_type (browser, buyer, vip)

from implementation import MultinomialNaiveBayes

# Data kunjungan customer
X_counts = np.array([
    [10, 2, 1, 0],   # banyak lihat electronics, jarang beli
    [2, 5, 8, 3],    # balance, regular buyer
    [15, 20, 10, 8], # frequent visitor, VIP
    # ... more data
])

y = np.array(['browser', 'buyer', 'vip', ...])

# Train
mnb = MultinomialNaiveBayes(alpha=1.0)
mnb.fit(X_counts, y)

# Predict
new_customer_visits = [[8, 12, 6, 4]]
prediction = mnb.predict(new_customer_visits)
probability = mnb.predict_proba(new_customer_visits)

print(f"Customer Type: {prediction[0]}")
print(f"Probabilities: {probability[0]}")
```

## 📚 Perbandingan dengan Algoritma Lain

| Aspek | Naive Bayes | Logistic Regression | Decision Tree | KNN |
|-------|-------------|-------------------|---------------|-----|
| Speed | Very Fast | Fast | Fast | Slow |
| Interpretability | High | High | Very High | Medium |
| Probabilistic | Yes | Yes | No (standard) | Yes |
| Feature Independence | Required | Not required | Not required | Not required |
| Small Data | Excellent | Good | Poor | Poor |
| High Dimensions | Excellent | Good | Poor | Poor |
| Text Classification | Excellent | Good | Poor | Poor |

## 🎓 Tips dan Best Practices

### 1. Feature Preprocessing

**Gaussian NB:**
```python
# Tidak perlu scaling (uses mean/variance per class)
# Tapi ensure no missing values
X = X.fillna(X.mean())
```

**Multinomial NB:**
```python
# Ensure non-negative values
X = np.abs(X)  # or X = X.clip(min=0)

# Normalize counts (optional, for TF-IDF)
from sklearn.preprocessing import Normalizer
normalizer = Normalizer()
X = normalizer.fit_transform(X)
```

### 2. Handling Imbalanced Classes

```python
# Naive Bayes naturally handles imbalance through priors
# Class priors automatically reflect class distribution

# Can also use class weights (manual)
# Oversample minority class in training data
```

### 3. Feature Selection

```python
# Naive Bayes works well even with many features
# But removing irrelevant features can improve:

from sklearn.feature_selection import SelectKBest, chi2

selector = SelectKBest(chi2, k=20)
X_selected = selector.fit_transform(X, y)
```

### 4. Probability Calibration

```python
# Naive Bayes probabilities often overconfident
# Use calibration for better probability estimates

from sklearn.calibration import CalibratedClassifierCV

# Wrap Naive Bayes with calibration
calibrated_nb = CalibratedClassifierCV(gnb, cv=5)
calibrated_nb.fit(X_train, y_train)
```

## 📚 Sumber Belajar

- [Naive Bayes - Scikit-learn](https://scikit-learn.org/stable/modules/naive_bayes.html)
- [StatQuest: Naive Bayes](https://www.youtube.com/watch?v=O2L2Uv9pdDA)
- [Andrew Ng ML Course - Naive Bayes](https://www.coursera.org/learn/machine-learning)

## 🎓 Latihan

Lihat [`exercise.ipynb`](./exercise.ipynb) untuk latihan hands-on:
1. Gaussian NB untuk customer segmentation
2. Multinomial NB untuk text classification
3. Comparing smoothing parameters
4. Feature importance analysis
5. Probability calibration

---

**Selamat Belajar Naive Bayes! 🎯**
