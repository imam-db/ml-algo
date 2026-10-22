# Model Evaluation

**[🇬🇧 English](README.md) | 🇮🇩 Bahasa Indonesia**

**🏠 [Kembali ke Menu Utama](../../README.md)** | **📁 [Advanced Topics](../README.md)**

## 🗺️ Akses Cepat

| File | Deskripsi | Jalankan |
|------|-----------|----------|
| [`implementation.py`](./implementation.py) | Implementasi dari nol | `python3 implementation.py` |
| [`sklearn_example.py`](./sklearn_example.py) | Contoh scikit-learn | `python3 sklearn_example.py` |
| [`exercise.ipynb`](./exercise.ipynb) | Latihan interaktif | `jupyter lab exercise.ipynb` |

## 📖 Teori dan Konsep

### Mengapa Model Evaluation Penting?

**Model evaluation** adalah proses mengukur seberapa baik model machine learning bekerja. Tanpa evaluasi yang proper, kita tidak tahu apakah model kita:
- Benar-benar belajar pola atau hanya menghafalkan data
- Akan bekerja baik pada data baru
- Lebih baik dibanding model lain

**Analogi SPG:** Seperti SPG yang mengevaluasi performa sales strategy-nya dengan melihat conversion rate, revenue, dan customer satisfaction - bukan hanya jumlah customer yang didekati.

## 🔄 Train/Test Split

### Konsep Dasar

**Problem:** Jika evaluate model pada data yang sama dengan training, hasilnya terlalu optimis (overfitting tidak terdeteksi)

**Solution:** Split data menjadi training set dan test set

```python
from implementation import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    X, y, 
    test_size=0.2,      # 20% untuk test
    random_state=42,    # untuk reproducibility
    shuffle=True        # shuffle sebelum split
)
```

### Best Practices

**Split Ratios:**
- **80/20 split**: Standard untuk dataset besar (>1000 samples)
- **70/30 split**: Untuk dataset menengah
- **60/40 split**: Untuk dataset kecil (<500 samples)

**Important:**
- Shuffle data sebelum split (kecuali time series)
- Gunakan stratified split untuk imbalanced classes
- Set random_state untuk reproducibility

## 📊 Classification Metrics

### 1. Confusion Matrix

**Definisi:** Tabel yang menunjukkan prediksi vs aktual

```
                Predicted
                Neg   Pos
Actual  Neg     TN    FP
        Pos     FN    TP
```

Di mana:
- **TP (True Positive)**: Prediksi positif, aktual positif ✅
- **TN (True Negative)**: Prediksi negatif, aktual negatif ✅
- **FP (False Positive)**: Prediksi positif, aktual negatif ❌ (Type I Error)
- **FN (False Negative)**: Prediksi negatif, aktual positif ❌ (Type II Error)

**Contoh SPG:**
```
Customer akan beli (aktual) vs SPG prediksi akan beli:
- TP: Prediksi beli, benar beli (sukses! 🎉)
- TN: Prediksi tidak beli, benar tidak beli (hemat waktu ✅)
- FP: Prediksi beli, ternyata tidak (waktu terbuang ⚠️)
- FN: Prediksi tidak beli, ternyata beli (opportunity lost 💔)
```

### 2. Accuracy

**Formula:**
```
Accuracy = (TP + TN) / (TP + TN + FP + FN)
```

**Kapan digunakan:**
- Classes balanced
- Semua errors equally important

**Kapan JANGAN digunakan:**
- Imbalanced classes (misleading!)
- Different costs untuk FP vs FN

**Contoh:**
```python
from implementation import accuracy_score

accuracy = accuracy_score(y_true, y_pred)
# 0.95 = 95% predictions correct
```

### 3. Precision

**Formula:**
```
Precision = TP / (TP + FP)
```

**Arti:** Dari semua yang kita prediksi positif, berapa yang benar positif?

**Kapan penting:**
- Cost of FP tinggi
- Kita ingin confident dalam positive predictions
- Medical diagnosis (avoid false alarms)
- Spam detection (avoid blocking real emails)

**Contoh SPG:**
Precision tinggi = Ketika SPG bilang "customer ini akan beli", biasanya benar beli. SPG tidak banyak salah prediksi.

### 4. Recall (Sensitivity)

**Formula:**
```
Recall = TP / (TP + FN)
```

**Arti:** Dari semua yang sebenarnya positif, berapa yang kita berhasil tangkap?

**Kapan penting:**
- Cost of FN tinggi
- Kita ingin catch semua positive cases
- Medical diagnosis (don't miss diseases)
- Fraud detection (catch all frauds)

**Contoh SPG:**
Recall tinggi = SPG berhasil mengidentifikasi hampir semua customer yang benar-benar akan beli. Tidak banyak opportunity yang terlewat.

### 5. F1 Score

**Formula:**
```
F1 = 2 × (Precision × Recall) / (Precision + Recall)
```

**Arti:** Harmonic mean dari Precision dan Recall

**Kapan digunakan:**
- Need balance antara Precision dan Recall
- Imbalanced classes
- Single metric untuk compare models

**Interpretation:**
- F1 = 1.0: Perfect precision dan recall
- F1 = 0.0: Worst case
- F1 > 0.8: Generally good

### Precision vs Recall Trade-off

```
High Precision, Low Recall:
  Model conservative - hanya predict positive jika sangat yakin
  Example: Spam filter yang jarang salah, tapi miss beberapa spam
  
Low Precision, High Recall:
  Model aggressive - predict positive lebih sering
  Example: Medical screening yang catch semua penyakit, tapi banyak false alarms
```

### 6. Macro vs Micro Averaging (Multiclass)

**Macro Average:**
```
Macro = (Metric_class1 + Metric_class2 + ... + Metric_classN) / N
```
- Treats all classes equally
- Good untuk balanced classes
- Each class contributes equally

**Micro Average:**
```
Micro = Metric(all TP, all FP, all FN combined)
```
- Weighted by class frequency
- Good untuk imbalanced classes
- Large classes dominate

**Weighted Average:**
```
Weighted = Σ(Metric_class × support_class) / total_support
```
- Weighted by number of samples per class
- Best for imbalanced datasets

## 📈 ROC Curve dan AUC

### ROC Curve (Receiver Operating Characteristic)

**Definisi:** Plot TPR vs FPR pada berbagai classification thresholds

```
TPR (True Positive Rate) = Recall = TP / (TP + FN)
FPR (False Positive Rate) = FP / (FP + TN)
```

**Interpretation:**
- **Perfect classifier**: TPR=1, FPR=0 (top-left corner)
- **Random classifier**: Diagonal line (TPR = FPR)
- **Bad classifier**: Below diagonal

### AUC (Area Under Curve)

**Interpretation:**
- **AUC = 1.0**: Perfect classifier
- **AUC = 0.5**: Random classifier (no discriminative power)
- **AUC < 0.5**: Worse than random (inverted predictions!)
- **AUC > 0.9**: Excellent
- **AUC 0.8-0.9**: Good
- **AUC 0.7-0.8**: Fair
- **AUC 0.6-0.7**: Poor
- **AUC < 0.6**: Very poor

**Keuntungan AUC:**
- Threshold-independent
- Works dengan imbalanced classes
- Single number untuk compare models

**Contoh:**
```python
from implementation import roc_curve, roc_auc_score

# Compute ROC curve
fpr, tpr, thresholds = roc_curve(y_true, y_scores)

# Compute AUC
auc = roc_auc_score(y_true, y_scores)
print(f"AUC: {auc:.3f}")
```

## 🔄 Cross-Validation

### K-Fold Cross-Validation

**Konsep:** Split data menjadi K folds, train pada K-1 folds, test pada 1 fold, repeat K kali

```
Fold 1: [TEST][TRAIN][TRAIN][TRAIN][TRAIN]
Fold 2: [TRAIN][TEST][TRAIN][TRAIN][TRAIN]
Fold 3: [TRAIN][TRAIN][TEST][TRAIN][TRAIN]
Fold 4: [TRAIN][TRAIN][TRAIN][TEST][TRAIN]
Fold 5: [TRAIN][TRAIN][TRAIN][TRAIN][TEST]
```

**Keuntungan:**
- Setiap data point digunakan untuk training dan testing
- More reliable estimate daripada single train/test split
- Reduces variance dalam performance estimate

**Best Practices:**
- **K=5**: Standard choice (good bias-variance trade-off)
- **K=10**: More computationally expensive, slightly better estimate
- **Leave-One-Out (K=n)**: Maximum data usage, very expensive

**Contoh:**
```python
from implementation import cross_val_score, k_fold_split

# K-fold splits
folds = k_fold_split(X, n_splits=5, shuffle=True, random_state=42)

# Cross-validation score
scores = cross_val_score(
    model, X, y, 
    cv=5, 
    scoring='accuracy',
    random_state=42
)

print(f"CV Scores: {scores}")
print(f"Mean: {scores.mean():.3f} (+/- {scores.std():.3f})")
```

### Stratified K-Fold

**Untuk:** Imbalanced classes

**Konsep:** Each fold memiliki class distribution yang sama dengan overall dataset

**Kapan digunakan:**
- Classes imbalanced
- Classification tasks
- Small datasets

## 📝 Classification Report

**Comprehensive summary** dari precision, recall, F1 untuk setiap class

```python
from implementation import classification_report

report = classification_report(y_true, y_pred)
print(report)
```

**Output contoh:**
```
              precision    recall  f1-score   support

           0       0.88      0.85      0.86        48
           1       0.92      0.95      0.93        52

    macro avg       0.90      0.90      0.90       100
 weighted avg       0.90      0.90      0.90       100
 
     accuracy                           0.90       100
```

**Interpretasi:**
- **support**: Number of samples per class
- **macro avg**: Unweighted mean (all classes equal)
- **weighted avg**: Weighted by support (preferred untuk imbalanced)
- **accuracy**: Overall accuracy

## 🎯 Choosing the Right Metric

### Decision Framework

**Question 1: Regression atau Classification?**
- Regression → R², MSE, MAE
- Classification → Continue below

**Question 2: Binary atau Multiclass?**
- Binary → Accuracy, Precision, Recall, F1, ROC-AUC
- Multiclass → Accuracy, Macro/Micro/Weighted F1

**Question 3: Balanced atau Imbalanced classes?**
- Balanced → Accuracy is fine
- Imbalanced → F1, Precision, Recall, ROC-AUC

**Question 4: Apa yang lebih penting?**
- False Positives costly → Optimize **Precision**
- False Negatives costly → Optimize **Recall**
- Both equally important → Optimize **F1**
- Need ranking/probability → Optimize **ROC-AUC**

### Use Case Examples

**Medical Diagnosis (Cancer Detection):**
- **Metric:** Recall (Sensitivity)
- **Reason:** Missing cancer (FN) is very bad
- **Target:** Recall > 0.95

**Spam Detection:**
- **Metric:** Precision
- **Reason:** Blocking real email (FP) frustrates users
- **Target:** Precision > 0.95

**Customer Churn Prediction:**
- **Metric:** F1 or ROC-AUC
- **Reason:** Balance - want to catch churners without too many false alarms
- **Target:** F1 > 0.7

**Fraud Detection:**
- **Metric:** Recall, then Precision
- **Reason:** Must catch frauds, then filter false positives
- **Target:** Recall > 0.9, Precision > 0.5

## 💻 Best Practices

### 1. Always Use Train/Test Split

```python
# DON'T: Evaluate on training data
model.fit(X, y)
score = model.score(X, y)  # ❌ Optimistic!

# DO: Evaluate on unseen test data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
model.fit(X_train, y_train)
score = model.score(X_test, y_test)  # ✅ Realistic!
```

### 2. Use Cross-Validation untuk Model Selection

```python
# Compare models dengan CV
models = [LogisticRegression(), RandomForest(), KNN()]

for model in models:
    scores = cross_val_score(model, X, y, cv=5)
    print(f"{model.__class__.__name__}: {scores.mean():.3f} (+/- {scores.std():.3f})")
```

### 3. Report Multiple Metrics

```python
# Don't rely on single metric
from implementation import accuracy_score, precision_score, recall_score, f1_score

print(f"Accuracy:  {accuracy_score(y_test, y_pred):.3f}")
print(f"Precision: {precision_score(y_test, y_pred):.3f}")
print(f"Recall:    {recall_score(y_test, y_pred):.3f}")
print(f"F1 Score:  {f1_score(y_test, y_pred):.3f}")
```

### 4. Visualize Results

```python
from implementation import plot_confusion_matrix, plot_roc_curve_viz

# Confusion matrix
cm = confusion_matrix(y_test, y_pred)
plot_confusion_matrix(cm, class_names=['No Buy', 'Buy'])

# ROC curve
fpr, tpr, _ = roc_curve(y_test, y_scores)
auc = roc_auc_score(y_test, y_scores)
plot_roc_curve_viz(fpr, tpr, auc)
```

### 5. Consider Class Imbalance

```python
# For imbalanced data, use:
# - Stratified splitting
# - Weighted metrics (macro/micro/weighted)
# - ROC-AUC instead of accuracy
# - F1 score

from sklearn.model_selection import StratifiedKFold
skf = StratifiedKFold(n_splits=5)

# Use weighted F1
f1_weighted = f1_score(y_test, y_pred, average='weighted')
```

## 📊 Common Pitfalls

### ❌ Pitfall 1: Data Leakage

```python
# WRONG: Scale before split (leaks test data info into training!)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
X_train, X_test = train_test_split(X_scaled, ...)  # ❌

# RIGHT: Split first, then scale
X_train, X_test = train_test_split(X, ...)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)  # Fit on train
X_test_scaled = scaler.transform(X_test)        # Transform test  # ✅
```

### ❌ Pitfall 2: Using Accuracy untuk Imbalanced Data

```python
# Example: 95% class A, 5% class B
# Model yang prediksi semua A → 95% accuracy! But useless.

# Use F1, Precision/Recall, or ROC-AUC instead
```

### ❌ Pitfall 3: Not Setting Random State

```python
# Different splits setiap run → results not reproducible
X_train, X_test = train_test_split(X, y)  # ❌

# Set random_state untuk reproducibility
X_train, X_test = train_test_split(X, y, random_state=42)  # ✅
```

### ❌ Pitfall 4: Overfitting ke Validation Set

```python
# Don't tune hyperparameters terhadap test set!
# Use train/validation/test split atau nested CV

# Option 1: Train/Val/Test split
X_temp, X_test, y_temp, y_test = train_test_split(X, y, test_size=0.2)
X_train, X_val, y_train, y_val = train_test_split(X_temp, y_temp, test_size=0.25)

# Option 2: Nested cross-validation
# Outer loop: performance estimation
# Inner loop: hyperparameter tuning
```

## 📚 Sumber Belajar

- [Scikit-learn Model Evaluation](https://scikit-learn.org/stable/modules/model_evaluation.html)
- [Precision vs Recall - Google ML Crash Course](https://developers.google.com/machine-learning/crash-course/classification/precision-and-recall)
- [ROC Curves and AUC Explained](https://www.youtube.com/watch?v=4jRBRDbJemM)

## 🎓 Latihan

Lihat [`exercise.ipynb`](./exercise.ipynb) untuk latihan hands-on:
1. Train/test split dan evaluasi
2. Confusion matrix interpretation
3. Precision vs Recall trade-off
4. ROC curve analysis
5. Cross-validation comparison
6. Handling imbalanced data

---

**Selamat Belajar Model Evaluation! 📊**
