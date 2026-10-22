# Support Vector Machine (SVM)

🇮🇩 Bahasa Indonesia | [🇬🇧 Read in English](README.md)

🏠 Kembali ke Indeks Supervised: `../README.md`

Support Vector Machine adalah classifier berbasis margin. SVM tidak hanya
mencari batas keputusan yang memisahkan kelas, tetapi mencari batas dengan
margin terbesar di sekitarnya. Dengan trik kernel, ide yang sama dapat
menghasilkan batas yang sangat non-linear sambil tetap dilatih melalui masalah
optimasi konveks.

Modul ini mengimplementasikan dua SVM biner dari nol dengan NumPy:

- **SVM linear** yang dilatih dengan **Pegasos** (hinge loss + regularisasi L2),
- **SVM kernel** yang dilatih dengan **SMO sederhana** untuk kernel linear, RBF
  dan polinomial.

Semua contoh menggunakan data outlet sintetis: tujuannya adalah mengklasifikasi
outlet sebagai **berkinerja tinggi** atau **berkinerja rendah** berdasarkan
`daily_sales`, `foot_traffic` dan `staff_count`.

## 📚 Daftar Isi
- [Teori & Konsep](#-teori--konsep)
- [Fondasi Matematis](#-fondasi-matematis)
- [Pegasos untuk SVM Linear](#-pegasos-untuk-svm-linear)
- [SVM Kernel dan SMO Sederhana](#-svm-kernel-dan-smo-sederhana)
- [Dataset Outlet](#-dataset-outlet)
- [Memilih C dan Gamma](#-memilih-c-dan-gamma)
- [Keuntungan & Kerugian](#-keuntungan--kerugian)
- [Kapan Menggunakan](#-kapan-menggunakan)
- [File dalam Direktori Ini](#-file-dalam-direktori-ini)
- [Quick Start](#-quick-start)
- [Algoritma Terkait](#-algoritma-terkait)

## 🧠 Teori & Konsep

### Apa itu Support Vector Machine?

Classifier linear dapat ditulis sebagai:

```
f(x) = w · x + b
```

Kelas yang diprediksi adalah `+1` ketika `f(x) >= 0` dan `-1` jika sebaliknya.
Banyak hyperplane berbeda yang dapat memisahkan dua kelas. SVM memilih
hyperplane yang jaraknya ke titik training terdekat — **support vector** —
sebesar mungkin.

Jarak sebuah titik ke hyperplane adalah `|w · x + b| / ||w||`. Untuk titik yang
terklasifikasi benar dengan margin `y_i (w · x_i + b) >= 1`, margin geometrisnya
adalah `1 / ||w||`. Memaksimalkan margin karena itu setara dengan
meminimalkan `||w||²`.

### Hard Margin vs Soft Margin

SVM **hard margin** mewajibkan setiap titik training berada di sisi margin yang
benar. Data nyata biasanya memiliki noise atau kelas yang tumpang tindih,
sehingga digunakan formulasi **soft margin** dengan variabel slack `ξ_i >= 0`:

```
minimize    (1/2) ||w||² + C Σ_i ξ_i
subject to  y_i (w · x_i + b) >= 1 - ξ_i
            ξ_i >= 0
```

`C` mengatur trade-off:

- **C kecil**: regularisasi kuat, margin lebar, lebih banyak pelanggaran margin
  (bias tinggi, variance rendah).
- **C besar**: regularisasi lemah, margin sempit, sedikit pelanggaran margin
  (bias rendah, variance tinggi, risiko overfitting).

### Hinge Loss + Regularisasi L2

Masalah soft margin setara dengan meminimalkan objective tanpa kendala:

```
J(w, b) = (lambda / 2) ||w||² + (1/n) Σ_i max(0, 1 - y_i (w · x_i + b))
```

Suku kedua adalah **hinge loss**. Nilainya nol untuk titik di luar margin dan
naik secara linear untuk titik di dalam margin atau di sisi yang salah. Suku
pertama adalah **regularisasi L2**, yang dikendalikan oleh
`lambda = 1 / (C n)` pada parameterisasi standar.

### Support Vector

Hanya titik dengan `y_i (w · x_i + b) <= 1` yang memberi kontribusi gradien
hinge loss tidak nol. Titik-titik inilah support vector. Menggeser titik
training lain tidak mengubah batas keputusan, yang menjadi salah satu alasan SVM
dapat melakukan generalisasi dengan baik.

### Trik Kernel

Jika data tidak terpisah secara linear, petakan ke ruang fitur berdimensi lebih
tinggi `φ(x)` lalu terapkan SVM linear di sana. Optimasi hanya membutuhkan inner
product `φ(x_i) · φ(x_j)`, sehingga dapat diganti dengan fungsi kernel:

```
K(x_i, x_j) = φ(x_i) · φ(x_j)
```

Fungsi keputusannya menjadi:

```
f(x) = Σ_i α_i y_i K(x_i, x) + b
```

dengan `α_i` sebagai variabel dual. Training tidak perlu menghitung `φ(x)`
secara eksplisit.

## 📐 Fondasi Matematis

### Masalah Primal

```
minimize    (1/2) ||w||² + C Σ_i ξ_i
subject to  y_i (w · x_i + b) >= 1 - ξ_i
            ξ_i >= 0
```

### Masalah Dual

Untuk kernel `K`, masalah dualnya adalah:

```
maximize    Σ_i α_i - (1/2) Σ_i Σ_j α_i α_j y_i y_j K(x_i, x_j)
subject to  0 <= α_i <= C
            Σ_i α_i y_i = 0
```

Bobot primal dapat diperoleh kembali sebagai
`w = Σ_i α_i y_i φ(x_i)`. Untuk kernel linear, ini menjadi
`w = Σ_i α_i y_i x_i`.

### Kondisi KKT

Pada titik optimum, setiap titik training memenuhi salah satu kondisi:

```
α_i = 0      =>  y_i f(x_i) >= 1       (di luar margin, bukan support vector)
0 < α_i < C  =>  y_i f(x_i) = 1        (tepat di margin, support vector bebas)
α_i = C      =>  y_i f(x_i) <= 1       (di dalam margin atau salah klasifikasi)
```

SMO menggunakan pelanggaran kondisi KKT untuk menentukan variabel yang akan
diperbarui.

## ⚙️ Pegasos untuk SVM Linear

Pegasos (**Primal Estimated sub-GrAdient SOlver**) meminimalkan hinge loss
terregularisasi secara langsung dengan stochastic subgradient descent.

Pada iterasi `t`, learning rate-nya adalah:

```
eta_t = 1 / (lambda * t)
```

Untuk titik training `(x_i, y_i)` yang dipilih acak:

```
if y_i (w · x_i + b) < 1:
    w <- (1 - eta_t * lambda) * w + eta_t * y_i * x_i
    b <- b + eta_t * y_i
else:
    w <- (1 - eta_t * lambda) * w
```

Faktor `(1 - eta_t * lambda)` berasal dari regularisasi L2. Bias tidak
diregularisasi pada implementasi ini. Dengan `lambda` kecil, langkah awal
menjadi besar; ini normal dan decay akan menurunkan bobot seiring waktu.

`LinearSVM` juga menyimpan nilai objective penuh setelah setiap epoch di
`train_losses_`.

## 🌐 SVM Kernel dan SMO Sederhana

Kelas `KernelSVM` menyelesaikan masalah dual dengan
**Sequential Minimal Optimization**. SMO memperbarui dua variabel dual sekaligus
karena kendala kesamaan `Σ_i α_i y_i = 0` mengaitkan keduanya.

Implementasi ini menggunakan loop yang sengaja disederhanakan:

1. Kunjungi setiap `α_i` dalam urutan acak.
2. Periksa kondisi KKT dengan toleransi `tol`.
3. Pilih pasangan `α_j` dengan selisih error `|E_i - E_j|` terbesar; jika
   kendala box tidak menyisakan langkah yang feasible, coba pasangan berikutnya.
4. Perbarui pasangan secara analitik lalu clip ke `[0, C]`.
5. Hitung ulang intercept `b` dari pasangan yang diperbarui.
6. Berhenti jika satu pass penuh tidak menghasilkan update, atau setelah
   `max_iter` pass.

Tidak ada working-set cache, shrinking heuristic atau informasi orde kedua.
Kode tetap ringkas namun cukup akurat untuk dataset kecil dan menengah.

### Kernel yang Didukung

| Kernel | Rumus | Parameter penting |
|--------|-------|-------------------|
| Linear | `K(x, z) = x · z` | tidak ada |
| RBF | `K(x, z) = exp(-gamma * ||x - z||²)` | `gamma` |
| Polinomial | `K(x, z) = (gamma * x · z + coef0)^degree` | `gamma`, `coef0`, `degree` |

Untuk kernel RBF dan polinomial, `gamma` dapat berupa:

- `'scale'`: `1 / (n_features * X.var())` (default, menyesuaikan data),
- `'auto'`: `1 / n_features`,
- float positif.

## 🏪 Dataset Outlet

`generate_outlet_data(n_samples=300, random_state=42)` membuat dataset sintetis
yang reproducible:

| Fitur | Arti | Rentang umum |
|-------|------|--------------|
| `daily_sales` | Pendapatan harian | ratusan hingga ribuan |
| `foot_traffic` | Pengunjung harian | 250–1800 |
| `staff_count` | Jumlah staf hari itu | 6–35 |

Label didasarkan pada **penjualan per staf** dibandingkan dengan benchmark yang
bergantung pada foot traffic:

```
sales_per_staff = 55 + 0.035 * foot_traffic
                  + 10 * sin(foot_traffic / 350) + noise
benchmark       = 88 + 0.00004 * (foot_traffic - 900)²
high_performing = sales_per_staff > benchmark
```

Karena benchmark berbentuk kuadratik terhadap foot traffic, batas kelasnya
non-linear pada fitur mentah. SVM linear tetap menjadi baseline yang kuat;
kernel RBF dapat melengkung mengikuti benchmark dengan lebih baik. Skala antar
fitur berbeda jauh, sehingga kedua implementasi menstandardisasi fitur secara
internal secara default.

## 🎛️ Memilih C dan Gamma

### C (SVM linear maupun kernel)

```
Terlalu kecil: underfitting, hampir semua titik melanggar margin
Seimbang:     akurasi validasi baik, jumlah support vector wajar
Terlalu besar: overfitting, hampir tidak ada pelanggaran margin
```

### Gamma (kernel RBF)

```
Terlalu kecil: kernel hampir linear / sangat halus -> underfitting
Seimbang:     wilayah lokal yang saling tumpang tindih -> generalisasi baik
Terlalu besar: setiap titik hanya memengaruhi dirinya sendiri -> overfitting
```

Alur kerja praktis:

1. Standardisasi fitur.
2. Uji `C` pada skala logaritmik (`0.01, 0.1, 1, 10, 100`).
3. Uji `gamma` pada skala logaritmik (`0.01, 0.03, 0.1, 0.3, 1, 3`).
4. Pilih konfigurasi dengan skor cross-validation terbaik, bukan skor training.
5. Latih ulang pada seluruh training set dan evaluasi sekali pada test set.

## ✅ Keuntungan & Kerugian

### ✅ Keuntungan

- Performa kuat pada dataset kecil dan menengah.
- Efektif di ruang berdimensi tinggi, termasuk klasifikasi teks.
- Objective konveks untuk formulasi SVM standar: tidak ada local minimum.
- Trik kernel menangani batas non-linear tanpa feature map eksplisit.
- Support vector memberi representasi batas keputusan yang kompak.
- Hanya sebagian titik training yang memengaruhi model akhir.

### ❌ Kerugian

- Waktu training tumbuh cepat seiring ukuran dataset (terutama SVM kernel).
- Sensitif terhadap skala fitur dan hyperparameter (`C`, `gamma`).
- Tidak langsung menghasilkan probabilitas yang terkalibrasi.
- SVM standar bersifat biner; multi-kelas memerlukan one-vs-one atau
  one-vs-rest.
- SVM kernel harus menyimpan support vector dan menghitung kernel saat
  prediksi.
- Lebih sulit diinterpretasikan dibanding model linear/logistik saat memakai
  kernel.

## 🎯 Kapan Menggunakan

### ✅ Gunakan SVM ketika

- Dataset berukuran kecil atau menengah.
- Jumlah fitur besar relatif terhadap jumlah sampel.
- Anda membutuhkan batas non-linear yang kuat dan dapat menyetel `C` serta
  `gamma`.
- Anda menginginkan masalah optimasi konveks yang mapan.
- Representasi support vector yang kompak bermanfaat.

### ❌ Pertimbangkan alternatif ketika

- Dataset sangat besar — gunakan model linear, SGD, atau tree ensemble.
- Anda membutuhkan probabilitas terkalibrasi — gunakan logistic regression atau
  probability calibration.
- Anda membutuhkan koefisien yang sangat interpretable — gunakan model linear
  atau decision tree.
- Kecepatan training kritis — pertimbangkan LinearSVC, LightGBM atau XGBoost.
- Data sebagian besar kategorikal dengan banyak level — model berbasis tree
  sering menanganinya lebih alami.

## 📁 File dalam Direktori Ini

- **`README.md`** — teori dan konsep dalam bahasa Inggris
- **`README.id.md`** (file ini) — dokumentasi lengkap dalam Bahasa Indonesia
- **`implementation.py`** — `LinearSVM` (Pegasos), `KernelSVM` (SMO sederhana),
  generator data outlet sintetis dan demo `__main__`
- **`sklearn_example.py`** — eksperimen `SVC` scikit-learn untuk menunjukkan
  efek `C` dan `gamma` pada data outlet (headless, mencetak akurasi)
- **`exercise.ipynb`** — latihan dan eksperimen

## 🚀 Quick Start

Jalankan demo dari nol:

```bash
python 01_supervised_learning/10_svm/implementation.py
```

Jalankan perbandingan scikit-learn:

```bash
python 01_supervised_learning/10_svm/sklearn_example.py
```

Jalankan test:

```bash
pytest tests/test_svm.py -v
```

Penggunaan minimal:

```python
from implementation import LinearSVM, KernelSVM, generate_outlet_data

X, y = generate_outlet_data(n_samples=300, random_state=42)

linear = LinearSVM(lambda_=1e-3, n_epochs=60, random_state=42)
linear.fit(X, y)
print(linear.score(X, y))

rbf = KernelSVM(C=10.0, kernel="rbf", gamma="scale", random_state=42)
rbf.fit(X, y)
print(rbf.score(X, y), rbf.n_support_)
```

## 🔗 Algoritma Terkait

- **Logistic Regression** ([02_logistic_regression](../02_logistic_regression/)):
  juga classifier linear, tetapi mengoptimalkan log-loss dan menghasilkan
  probabilitas.
- **K-Nearest Neighbors** ([07_knn](../07_knn/)): metode berbasis jarak lain;
  tidak ada fase training, tetapi prediksi membutuhkan seluruh dataset.
- **Decision Trees** ([03_decision_trees](../03_decision_trees/)) dan
  **Random Forests** ([04_random_forest](../04_random_forest/)): model
  non-linear yang tidak memerlukan feature scaling.
- **Gradient Boosting** ([09_gradient_boosting](../09_gradient_boosting/)):
  sering menjadi pilihan terkuat pada data tabular, tetapi kurang cocok untuk
  data sparse berdimensi sangat tinggi dibanding SVM linear.
- **Neural Networks** ([08_neural_networks](../08_neural_networks/)): model
  non-linear fleksibel yang dapat mempelajari representasi fitur, dengan
  kebutuhan data dan tuning yang lebih besar.

---

**Langkah Selanjutnya**:
- Baca `implementation.py` untuk melihat Pegasos dan SMO bekerja.
- Jalankan `sklearn_example.py` untuk mengeksplorasi trade-off `C` dan `gamma`.
- Selesaikan `exercise.ipynb` untuk berlatih tuning dan membandingkan kernel.
- Coba alur kerja yang sama pada masalah klasifikasi tabular Anda sendiri.
