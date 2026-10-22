# Peramalan Time Series

**[🇬🇧 English](README.md) | 🇮🇩 Bahasa Indonesia**

🏠 Kembali ke Advanced Topics: `../README.md`

Penjualan harian outlet bukanlah kumpulan baris yang saling independen. Omzet
hari ini bergantung pada kemarin, hari yang sama minggu lalu, siklus gajian,
hari libur, dan tren bisnis yang bergerak perlahan. Oleh karena itu, pembagian
data train/test secara acak dan cross-validation K-fold acak akan membocorkan
informasi dari masa depan dan menghasilkan skor yang terlalu optimistis.

Modul ini membangun toolkit peramalan kecil yang transparan dengan **NumPy
saja**, lalu menunjukkan bagaimana ide yang sama dipetakan ke scikit-learn dan
statsmodels. Contoh yang digunakan adalah data sintetis penjualan outlet
Indonesia dengan tren linear, musiman mingguan, lonjakan *gajian* di sekitar
tanggal 25 sampai tanggal 1, dan efek hari libur bertanggal tetap.

## 📚 Daftar Isi

- [Mengapa Time Series Berbeda](#-mengapa-time-series-berbeda)
- [Dataset Outlet Sintetis](#-dataset-outlet-sintetis)
- [Persiapan Peramalan dan Kebocoran Data](#-persiapan-peramalan-dan-kebocoran-data)
- [Peramalan Baseline](#-peramalan-baseline)
- [Simple Exponential Smoothing](#-simple-exponential-smoothing)
- [Holt-Winters Aditif](#-holt-winters-aditif)
- [Model Autoregresif](#-model-autoregresif)
- [Fitur Lag dan LagRegression](#-fitur-lag-dan-lagregression)
- [Metrik Peramalan](#-metrik-peramalan)
- [Pembagian Berurutan dan Backtesting Rolling-Origin](#-pembagian-berurutan-dan-backtesting-rolling-origin)
- [Quick Start](#-quick-start)
- [Contoh Scikit-Learn dan statsmodels](#-contoh-scikit-learn-dan-statsmodels)
- [Latihan Notebook](#-latihan-notebook)
- [File di Direktori Ini](#-file-di-direktori-ini)
- [Kapan Menggunakan Setiap Metode](#-kapan-menggunakan-setiap-metode)
- [Kelebihan dan Keterbatasan](#-kelebihan-dan-keterbatasan)
- [Aplikasi di Retail](#-aplikasi-di-retail)
- [Modul Terkait](#-modul-terkait)

## 🧠 Mengapa Time Series Berbeda

Baris pada supervised learning biasa diasumsikan dapat dipertukarkan:
mengacak urutan baris tidak mengubah masalahnya. Time series memiliki urutan.
Pada waktu `t`, kita hanya boleh menggunakan informasi dari `t-1`, `t-2`, ...;
masa depan belum diketahui.

Tiga konsekuensi mendorong setiap keputusan desain dalam modul ini:

1. **Bagi data secara kronologis.** Data training harus lebih lama daripada data
   test.
2. **Bangun fitur hanya dari masa lalu.** Fitur untuk target `y[t]` boleh
   menggunakan `y[t-1]`, `y[t-2]`, ..., tetapi tidak boleh `y[t]` atau `y[t+1]`.
3. **Backtest dengan origin yang bergerak.** Evaluasi banyak jendela masa depan,
   bukan hanya satu pembagian, dan selalu latih ulang hanya dengan data yang
   tersedia pada origin tersebut.

Tujuannya bukan sekadar error rendah pada satu jendela. Tujuannya adalah
prosedur yang akan menghasilkan peramalan jujur pada saat peramalan itu
dibutuhkan.

## 🏪 Dataset Outlet Sintetis

`generate_outlet_sales(n_days, seed)` mengembalikan `(dates, values)`:

- `dates` adalah array `numpy.datetime64[D]` berisi hari kalender berurutan,
- `values` adalah vektor penjualan harian positif dengan dua angka desimal.

Seed yang sama selalu menghasilkan deret yang persis sama. Generator
menggabungkan lima komponen:

```
sales[t] = (base[t] * weekly[weekday(t)] + payday[t]) * (1 + holiday[t]) + noise[t]
```

| Komponen | Bentuk | Tujuan |
|----------|--------|--------|
| `base[t] = 900 + 0.55 * t` | Tren linear | Pertumbuhan bisnis yang perlahan |
| `weekly` | 7 multiplier | Kenaikan akhir pekan, Sabtu paling ramai |
| `payday` | Lonjakan aditif | Jendela gajian dari tanggal 25 sampai 2 |
| `holiday` | Kenaikan multiplikatif | Tahun Baru, Hari Buruh, Kemerdekaan, Natal |
| `noise` | Gaussian | Variasi acak harian |

### Musiman mingguan

Multiplier hari kerja adalah:

| Hari | Sen | Sel | Rab | Kam | Jum | Sab | Min |
|------|-----|-----|-----|-----|-----|-----|-----|
| Faktor | 0.92 | 0.94 | 0.97 | 1.00 | 1.08 | 1.16 | 1.05 |

Pola ini menciptakan periode berulang tujuh hari, sehingga `season=7` menjadi
nilai default di seluruh modul.

### Lonjakan gajian

Gaji di Indonesia umumnya dibayarkan pada tanggal 25. Generator menambahkan
boost aditif berdasarkan tanggal dalam bulan:

```
25: +300   26: +250   27: +175   28: +120
29: +80    30: +50    31: +30
 1: +260    2: +150    3: +80     4: +40
```

Hasilnya adalah jendela belanja besar di akhir bulan, periode tenang yang
singkat, dan lonjakan kedua saat bulan baru dimulai. Pola ini *tidak*
tertangkap oleh model musiman mingguan biasa, sehingga fitur kalender tambahan
sangat bernilai di produksi.

### Efek hari libur

Hari libur bertanggal tetap mendapat kenaikan multiplikatif, dengan kenaikan
lebih kecil pada satu hari sebelum dan sesudahnya:

- Tahun Baru (1 Januari),
- Hari Buruh (1 Mei),
- Hari Kemerdekaan (17 Agustus),
- Natal (25 Desember),
- Malam Tahun Baru (31 Desember).

Hari libur lunar seperti Idul Fitri berpindah setiap tahun. Sistem nyata
sebaiknya menggabungkan kalender hari libur resmi, tetapi versi bertanggal tetap
sudah cukup untuk menunjukkan ide pemodelannya.

## ⚙️ Persiapan Peramalan dan Kebocoran Data

Modul ini menggunakan satu pembagian berurutan untuk jendela test final:

```python
from implementation import generate_outlet_sales, time_train_test_split

dates, sales = generate_outlet_sales(n_days=365, seed=42)
train, test = time_train_test_split(sales, test_size=30)

# train berisi 335 hari pertama, test berisi 30 hari terakhir.
# Kedua irisan berurutan dan tidak pernah diacak.
```

`time_train_test_split` menerima ukuran test berupa integer atau fraksi float:

```python
train, test = time_train_test_split(sales, test_size=0.2)  # test = ceil(20%)
```

Pembagian acak akan memasukkan hari masa depan ke data training, sehingga model
belajar dari periode yang seharusnya diramalkan. Itulah kebocoran data. Logika
yang sama berlaku untuk fitur:

```python
# SALAH: fitur untuk y[t] memuat y[t] itu sendiri
X_bad = sales[t]

# BENAR: fitur untuk y[t] hanya memuat observasi sebelumnya
X_good = sales[t - 7]
```

Helper `make_lag_features` menegakkan aturan ini dan dijelaskan di bawah.

## 📈 Peramalan Baseline

Selalu bangun baseline sebelum melatih model kompleks. Jika Holt-Winters atau
gradient boosting tidak bisa mengalahkan seasonal naive, kompleksitas tambahan
itu tidak terjustifikasi.

### 1. Naive forecast

```
y_hat[T+h] = y[T]
```

Nilai terakhir yang diamati diulang untuk setiap langkah masa depan. Ini adalah
baseline random walk dan ternyata sulit dikalahkan pada deret permintaan atau
keuangan yang berisik.

```python
from implementation import naive_forecast

prediction = naive_forecast(train, horizon=30)
```

### 2. Seasonal naive forecast

```
y_hat[T+h] = y[T+h-m]
```

Nilai dari musim yang sama satu periode sebelumnya diulang. Dengan data harian
dan `m = 7`, Senin diramalkan dari Senin lalu, Selasa dari Selasa lalu, dan
seterusnya. Ini baseline yang kuat ketika musiman mingguan atau tahunan
mendominasi.

```python
from implementation import seasonal_naive_forecast

prediction = seasonal_naive_forecast(train, horizon=30, season=7)
```

### 3. Moving-average forecast

```
y_hat[T+h] = mean(y[T-w+1], ..., y[T])
```

Peramalan datar sebesar rata-rata `w` observasi terakhir. Metode ini
menghaluskan noise dan menangkap level saat ini, tetapi mengabaikan tren dan
musiman. Paling berguna untuk deret yang hanya memiliki level atau sebagai
pemeriksaan cepat.

```python
from implementation import moving_average_forecast

prediction = moving_average_forecast(train, horizon=30, window=7)
```

## 🌊 Simple Exponential Smoothing

Simple exponential smoothing (SES) menyimpan satu state, yaitu **level** `L_t`:

```
L_0 = y_0
L_t = alpha * y_t + (1 - alpha) * L_{t-1}
y_hat[T+h] = L_T
```

`alpha` dalam `(0, 1]` mengontrol memori:

- **Alpha kecil** (misalnya 0.1): reaksi lambat, level sangat halus, baik untuk
  deret stabil yang berisik.
- **Alpha besar** (misalnya 0.9): reaksi cepat, mendekati naive forecast, baik
  ketika level benar-benar berubah.

SES menangkap pergeseran level tetapi tidak dapat merepresentasikan tren atau
pola mingguan berulang. Peramalannya selalu datar.

```python
from implementation import SimpleExpSmoothing

model = SimpleExpSmoothing(alpha=0.3).fit(train)
prediction = model.forecast(horizon=30)

print(model.level_)          # level final
print(model.fitted_values_)  # fitted in-sample one-step-ahead
print(model.residuals_)
```

Class ini menyediakan `forecast(h)` dan alias `predict(h)`. Memanggil keduanya
sebelum `fit` akan memunculkan error yang jelas.

## ❄️ Holt-Winters Aditif

Holt-Winters memperluas SES dengan **tren** `T_t` dan komponen **musiman** `S_t`
yang memiliki satu nilai per fase. Untuk periode `m` dan musiman aditif:

```
y_hat_t = L_{t-1} + T_{t-1} + S_{t-m}

L_t = alpha * (y_t - S_{t-m}) + (1 - alpha) * (L_{t-1} + T_{t-1})
T_t = beta  * (L_t - L_{t-1})       + (1 - beta)  * T_{t-1}
S_t = gamma * (y_t - L_t)           + (1 - gamma) * S_{t-m}
```

Komponen musiman bersifat aditif: berupa jumlah penjualan yang ditambahkan ke
level saat ini, bukan multiplier persentase. Ini cocok untuk deret di mana
selisih akhir pekan relatif konstan seiring pertumbuhan bisnis.

### Inisialisasi

Implementasi menggunakan dua musim lengkap pertama:

```
L_0 = mean(musim pertama)
T_0 = (mean(musim kedua) - mean(musim pertama)) / m
S_i = musim_pertama[i] - L_0       (dipusatkan agar rata-rata nol)
```

Diperlukan minimal dua musim lengkap, sehingga data harian dengan `m = 7`
membutuhkan setidaknya 14 observasi. Riwayat yang lebih panjang menghasilkan
estimasi yang jauh lebih stabil.

### Peramalan

Setelah fitting, peramalan h langkah adalah:

```
y_hat[T+h] = L_T + h * T_T + S[(T+h-1) mod m]
```

Level diekstrapolasi dengan tren final, lalu state musiman untuk fase yang sesuai
ditambahkan kembali.

```python
from implementation import HoltWinters

model = HoltWinters(
    alpha=0.3,
    beta=0.05,
    gamma=0.3,
    season_length=7,
).fit(train)

prediction = model.forecast(30)
print(model.level_, model.trend_, model.seasonal_)
```

Panduan praktis untuk tiga faktor smoothing:

| Parameter | Terlalu kecil | Terlalu besar |
|-----------|---------------|---------------|
| `alpha` | Level tertinggal dari perubahan | Level mengejar noise |
| `beta` | Tren terlalu kaku | Tren berosilasi |
| `gamma` | Pola musiman membeku | Pola musiman menyerap noise |

Class ini sengaja menggunakan faktor tetap, bukan optimiser, agar persamaan
update tetap terlihat. Di produksi, tuning faktor pada jendela validasi atau
gunakan implementasi statsmodels yang sudah dioptimasi.

## 🔁 Model Autoregresif

Model `AR(p)` memprediksi nilai berikutnya dari `p` nilai sebelumnya:

```
y_t = c + phi_1 y_{t-1} + ... + phi_p y_{t-p} + e_t
```

`ARModel(p)` mengestimasi intercept dan koefisien dengan ordinary least squares.
Untuk observasi `t = p, ..., n-1`, design matrix-nya adalah:

```
[ 1  y[t-1]  y[t-2]  ...  y[t-p] ]
```

Kolom target adalah `y[p:]`. Menggunakan `numpy.linalg.lstsq` menghindari
pembentukan normal equations secara eksplisit dan lebih stabil secara numerik.

```python
from implementation import ARModel

model = ARModel(p=7).fit(train)
print(model.intercept_)
print(model.coef_)          # phi_1 ... phi_7
print(model.forecast(30))   # recursive multi-step forecast
```

### Peramalan multi-step rekursif

Prediksi satu langkah hanya membutuhkan riwayat yang diketahui. Untuk langkah
kedua, model menggunakan prediksi langkah pertamanya sebagai lag terbaru; untuk
langkah ketiga, model menggunakan prediksi langkah pertama dan kedua, dan
seterusnya:

```
y_hat[T+1] = c + phi_1 y[T]       + phi_2 y[T-1] + ...
y_hat[T+2] = c + phi_1 y_hat[T+1] + phi_2 y[T]   + ...
```

Ini praktis, tetapi error dapat terakumulasi. Model AR yang stabil akan menuju
rata-rata jangka panjangnya; model yang tidak stabil dapat meledak. Selalu
periksa peramalan multi-step, jangan langsung mempercayainya.

`ARModel` tidak menegakkan stasioneritas. Jika akar yang diestimasi berada di
luar unit circle, peramalan rekursif dapat tumbuh tanpa batas. Untuk produksi,
pertimbangkan differencing, regularisasi, atau parameterisasi yang dibatasi.

## 🧱 Fitur Lag dan LagRegression

Model supervised berbasis tree dan linear membutuhkan matriks fitur datar.
Jembatan dari time series ke matriks tersebut adalah fitur lag.

### `make_lag_features(y, lags, rolling_windows)`

Untuk target pada waktu `t`, helper membangun:

- satu kolom per lag `k`: `y[t-k]`,
- untuk setiap rolling window `w`: `mean(y[t-w:t])` dan `std(y[t-w:t])`.

Target pertama yang dapat digunakan adalah `max(max(lags), max(rolling_windows))`,
karena seluruh riwayat yang diminta harus tersedia. Lag 0 ditolak karena akan
sama dengan target dan membocorkan jawaban.

```python
from implementation import make_lag_features

X, target, index = make_lag_features(
    train,
    lags=(1, 2, 3, 7),
    rolling_windows=(7, 14),
    return_index=True,
)

# Baris i memprediksi train[index[i]] hanya menggunakan train[:index[i]].
print(X.shape, target.shape, index[:5])
```

Fungsi mengembalikan `(X, target)` secara default. Dengan `return_index=True`,
fungsi juga mengembalikan indeks target integer, sehingga baris fitur tetap
selaras dengan tanggal dan pembagian berurutan.

Mengapa rolling window aman: untuk target `t`, irisannya adalah `y[t-w:t]`,
yang berakhir tepat sebelum `t`. Irisan itu tidak memuat `y[t]`. Kesalahan umum
adalah `y[t-w+1:t+1]`, yang memuat target dan membuat skor validasi terlalu
tinggi.

### `LagRegression`

`LagRegression` menempatkan ordinary least squares di atas fitur-fitur tersebut:

```
y_t = c + sum_k phi_k y[t-k]
        + sum_w (a_w * rolling_mean_w[t] + b_w * rolling_std_w[t])
```

```python
from implementation import LagRegression

model = LagRegression(
    lags=(1, 2, 3, 4, 5, 6, 7, 14),
    rolling_windows=(7, 14),
).fit(train)

prediction = model.forecast(30)
print(model.feature_names_)
print(model.coef_)
```

Peramalan bersifat rekursif: setiap prediksi ditambahkan ke riwayat sebelum baris
fitur berikutnya dihitung. `feature_names_` mendokumentasikan urutan kolom:
`lag_1`, `lag_2`, ..., `rolling_mean_7`, `rolling_std_7`, dan seterusnya.

Untuk perbandingan yang adil dengan gradient boosting, `sklearn_example.py`
mengevaluasi prediksi one-step-ahead pada target yang ditahan. Ini mengukur
kualitas fitur tanpa mencampurnya dengan akumulasi error rekursif. Method
`LagRegression.forecast` dari modul ini menunjukkan tugas multi-step yang lebih
sulit.

## 📏 Metrik Peramalan

Misalkan `y_i` adalah nilai aktual dan `y_hat_i` adalah peramalan untuk
`i = 1 ... n`.

| Metrik | Formula | Satuan | Catatan |
|--------|---------|--------|---------|
| MAE | `mean(|y_i - y_hat_i|)` | sama dengan penjualan | Mudah dijelaskan |
| RMSE | `sqrt(mean((y_i - y_hat_i)^2))` | sama dengan penjualan | Menghukum error besar |
| MAPE | `100 * mean(|y_i - y_hat_i| / |y_i|)` | persen | Tidak terdefinisi saat aktual nol |
| SMAPE | `100 * mean(2|y_i - y_hat_i| / (|y_i| + |y_hat_i|))` | persen (0-200) | Simetris dan terbatas |

```python
from implementation import mae, rmse, mape, smape

print(mae(test, prediction))
print(rmse(test, prediction))
print(mape(test, prediction))
print(smape(test, prediction))
```

### Memilih metrik

- **MAE** paling mudah ditafsirkan ketika semua hari sama pentingnya.
- **RMSE** berguna ketika kesalahan besar jauh lebih mahal (misalnya stockout
  atau kelebihan stok).
- **MAPE** umum dalam pelaporan retail, tetapi meledak ketika penjualan aktual
  mendekati nol. Implementasi melewati nilai aktual nol dan hanya mengembalikan
  `inf` jika semua aktual nol sedangkan prediksi tidak.
- **SMAPE** tetap terbatas saat nilainya kecil, sehingga berguna untuk hari
  penjualan rendah, tetapi asimetrinya berbeda dan tidak boleh dibandingkan
  secara buta antar-dataset.

Laporkan lebih dari satu metrik. Sebuah model bisa menang pada MAE tetapi kalah
pada RMSE karena membuat beberapa error yang sangat besar.

## 🔄 Pembagian Berurutan dan Backtesting Rolling-Origin

Satu pembagian train/test bisa beruntung atau tidak beruntung. Validasi
walk-forward menjawab pertanyaan yang lebih realistis: *jika peramalan dibuat
pada setiap origin, seberapa baik model akan bekerja?*

### `time_train_test_split`

```python
train, test = time_train_test_split(sales, test_size=30)
```

Tanpa shuffle dan tanpa overlap. Observasi training terakhir tepat sebelum
observasi test pertama. Untuk `test_size` float, panjang test dibulatkan ke atas.

### `rolling_origin_backtest`

```python
from implementation import rolling_origin_backtest, seasonal_naive_forecast

folds = rolling_origin_backtest(
    sales,
    lambda history, h: seasonal_naive_forecast(history, h, season=7),
    initial=250,
    horizon=30,
    step=30,
)

print(len(folds))          # 3 fold lengkap untuk 365 observasi
print(folds[0]["mae"])
```

Origin-nya adalah `initial, initial + step, initial + 2*step, ...` selama
`origin + horizon <= n`. Dengan `n = 365`, `initial = 250`, `horizon = 30`, dan
`step = 30`, origin-nya adalah 250, 280, dan 310, sehingga ada tiga fold. Rumus
umum jumlah fold adalah:

```
floor((n - initial - horizon) / step) + 1
```

Setiap dictionary fold berisi:

| Key | Arti |
|-----|------|
| `fold` | Nomor fold berbasis nol |
| `train_start`, `train_end` | Jendela training `[0, train_end)` |
| `test_start`, `test_end` | Jendela test `[test_start, test_end)` |
| `actual`, `predicted` | Array sepanjang `horizon` |
| `mae`, `rmse`, `mape`, `smape` | Metrik per fold |

`model_fn` fleksibel. Bisa berupa:

```python
# 1. Fungsi yang mengembalikan prediksi
lambda history, h: naive_forecast(history, h)

# 2. Factory yang mengembalikan model terlatih dengan forecast/predict
lambda history: HoltWinters(season_length=7).fit(history)

# 3. Callable apa pun yang mengembalikan array prediksi
lambda history: seasonal_naive_forecast(history, 30, season=7)
```

Backtester tidak pernah melatih model pada observasi pada atau setelah awal test,
sehingga aman untuk pemilihan model. Tetap sisakan jendela test final yang tidak
disentuh atau periode yang lebih baru untuk laporan akhir.

### TimeSeriesSplit

Untuk model supervised berbasis lag, `TimeSeriesSplit` dari scikit-learn
menyediakan ide expanding-window yang sama:

```python
from sklearn.model_selection import TimeSeriesSplit

splitter = TimeSeriesSplit(n_splits=5)
for train_index, validation_index in splitter.split(X_train):
    ...
```

Baris validasi selalu berada setelah baris training. Jangan gunakan `KFold`,
`ShuffleSplit`, atau `train_test_split(shuffle=True)` pada data berurutan waktu.

## 🚀 Quick Start

Dari direktori ini:

```bash
python implementation.py
python sklearn_example.py
```

Perbandingan minimal:

```python
import numpy as np

from implementation import (
    ARModel,
    HoltWinters,
    LagRegression,
    SimpleExpSmoothing,
    generate_outlet_sales,
    mae,
    naive_forecast,
    seasonal_naive_forecast,
    time_train_test_split,
)

dates, sales = generate_outlet_sales(n_days=365, seed=42)
train, test = time_train_test_split(sales, test_size=30)
horizon = len(test)

models = {
    "Naive": naive_forecast(train, horizon),
    "Seasonal naive": seasonal_naive_forecast(train, horizon, season=7),
    "SES": SimpleExpSmoothing(alpha=0.3).fit(train).forecast(horizon),
    "Holt-Winters": HoltWinters(season_length=7).fit(train).forecast(horizon),
    "AR(7)": ARModel(7).fit(train).forecast(horizon),
    "LagRegression": LagRegression(
        lags=(1, 2, 3, 7), rolling_windows=(7,)
    ).fit(train).forecast(horizon),
}

for name, prediction in models.items():
    print(name, round(mae(test, prediction), 2))
```

## 🤖 Contoh Scikit-Learn dan statsmodels

`sklearn_example.py` mengubah deret menjadi masalah supervised dan menjalankan:

- `GradientBoostingRegressor`,
- `RandomForestRegressor`,
- keduanya dievaluasi dengan `TimeSeriesSplit` pada jendela training,
- jendela test one-step-ahead final,
- baseline `LagRegression` dan seasonal naive dari modul ini,
- statsmodels `ExponentialSmoothing` **hanya jika statsmodels dapat diimpor**.

Blok opsional dilindungi `try/except ImportError` dan mencetak pesan skip yang
jelas alih-alih gagal:

```text
statsmodels is not installed - skipping this optional comparison.
Install it with: pip install statsmodels
```

Matriks fitur mencakup lag 1--7 dan 14 plus rata-rata dan standar deviasi
rolling selama 7 dan 14 hari. Menyertakan lag 14 membuat model melihat dua
minggu riwayat penuh.

## 📓 Latihan Notebook

`exercise.ipynb` berisi sepuluh cell kode dan sembilan latihan terpandu dengan
solusi yang dikerjakan:

1. Menjelajahi tren, hari kerja, gajian, dan struktur hari libur.
2. Membangun pembagian berurutan dan memeriksa urutan kronologisnya.
3. Membandingkan peramalan naive, seasonal-naive, dan moving-average.
4. Melatih SES dan Holt-Winters lalu memeriksa state yang dipelajari.
5. Memulihkan koefisien AR(2) yang diketahui dari data simulasi.
6. Membuktikan bahwa fitur lag tidak membocorkan nilai masa depan.
7. Melatih dan meramalkan secara rekursif dengan `LagRegression`.
8. Menghitung dan memeriksa fold rolling-origin.
9. Memeringkat model pada jendela test dan mendiskusikan pemilihan model.

Notebook berjalan dari atas ke bawah baik dari direktori ini maupun dari root
repository dan hanya menggunakan NumPy plus modul ini.

## 📁 File di Direktori Ini

| File | Tujuan |
|------|--------|
| `implementation.py` | Generator data, model, fitur, backtest, dan metrik berbasis NumPy |
| `sklearn_example.py` | Gradient boosting, random forest, `TimeSeriesSplit`, statsmodels opsional |
| `exercise.ipynb` | Latihan terpandu dengan solusi |
| `README.md` | Dokumentasi bahasa Inggris |
| `README.id.md` | Dokumentasi bahasa Indonesia (file ini) |
| `../../tests/test_time_series.py` | Test otomatis untuk kebocoran, akurasi, dan metrik |

## 🎯 Kapan Menggunakan Setiap Metode

| Metode | Gunakan ketika | Hindari ketika |
|--------|----------------|----------------|
| Naive | Horizon sangat pendek, random walk berisik | Ada tren atau musiman jelas |
| Seasonal naive | Pola mingguan/tahunan kuat dan stabil | Pola berubah cepat |
| Moving average | Deret hanya level dan berisik | Ada tren atau musiman |
| SES | Pergeseran level, tanpa tren/musiman | Siklus mingguan penting |
| Holt-Winters | Tren + musiman aditif stabil | Musiman multiplikatif, banyak kovariat |
| AR(p) | Autokorelasi memori pendek, deret stasioner | Tren/musiman kuat tanpa differencing |
| LagRegression | Banyak lag, ringkasan rolling, baseline interpretable | Non-linearitas kuat, memori panjang |
| Gradient boosting | Interaksi fitur non-linear, riwayat cukup | Data sangat kecil, butuh ekstrapolasi |
| Random forest | Baseline non-linear yang robust | Ekstrapolasi tren di luar rentang training |

Model tree tidak dapat mengekstrapolasi tren linear di luar rentang yang dilihat
saat training. Untuk bisnis yang tumbuh, hilangkan tren dari deret, tambahkan
fitur kalender, atau gabungkan model tree dengan model tren.

## ⚖️ Kelebihan dan Keterbatasan

### Kelebihan

- **Transparan:** setiap update state dan koefisien terlihat.
- **Cepat:** semua model fit dalam milidetik pada data harian satu tahun.
- **Baseline kuat:** seasonal naive dan Holt-Winters sulit dikalahkan.
- **Evaluasi jujur:** pembagian berurutan dan rolling origin mencegah kebocoran.
- **Dapat dikomposisi:** fitur lag menghubungkan peramalan klasik dengan ML modern.

### Keterbatasan

- Hanya deret univariat yang dimodelkan; promosi, harga, dan foot traffic tidak
  disertakan.
- Holt-Winters menggunakan faktor smoothing tetap, bukan maximum likelihood.
- `ARModel` dan `LagRegression` dapat menghasilkan peramalan rekursif yang tidak
  stabil ketika koefisien mengimplikasikan non-stasioneritas.
- Hari libur bertanggal tetap mengabaikan hari libur lunar yang berpindah seperti
  Idul Fitri.
- Tidak ada prediction interval, peramalan probabilistik, atau rekonsiliasi
  hierarkis.
- Tidak ada deteksi musiman otomatis, differencing, atau uji stasioneritas.

Semua ini adalah pilihan cakupan yang disengaja. Modul ini mengajarkan mekanika
dan disiplin validasi; sistem produksi kemudian dapat menggunakan statsmodels,
Prophet, ARIMA/SARIMAX, gradient boosting, atau model sekuensial deep learning.

## 🏬 Aplikasi di Retail

- **Perencanaan penjualan harian:** meramalkan satu sampai empat minggu ke depan
  per outlet.
- **Penjadwalan staf:** menyelaraskan shift dengan permintaan hari kerja dan
  gajian.
- **Replenishment inventori:** memesan lebih banyak sebelum jendela gajian dan
  hari libur.
- **Evaluasi promosi:** membandingkan penjualan aktual dengan peramalan
  counterfactual.
- **Anggaran:** menggabungkan peramalan outlet menjadi rencana regional.
- **Deteksi anomali:** menandai hari yang menyimpang kuat dari peramalan.

Dalam skenario retail SPG, pola gajian sangat penting: model yang hanya mengetahui
musiman mingguan akan secara sistematis meramalkan terlalu rendah pada akhir
bulan. Menambahkan fitur tanggal dalam bulan, indikator gajian, atau komponen
gajian terpisah biasanya meningkatkan akurasi secara substansial.

## 🔗 Modul Terkait

- [Model Evaluation](../03_model_evaluation/) untuk metrik dan cross-validation.
- [XGBoost](../04_xgboost/) dan [CNN & Computer Vision](../05_cnn_computer_vision/)
  untuk jalur advanced topics yang lebih luas.
- [Linear Regression](../../01_supervised_learning/01_linear_regression/) untuk
  fondasi least squares yang digunakan `ARModel` dan `LagRegression`.
- [Random Forest](../../01_supervised_learning/04_random_forest/) dan
  [Gradient Boosting](../../01_supervised_learning/09_gradient_boosting/) untuk
  model supervised di `sklearn_example.py`.

**Selamat Belajar! 🎓** | *[🇬🇧 English version](README.md)*
