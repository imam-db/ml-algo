# t-SNE (t-distributed Stochastic Neighbor Embedding)

**[🇬🇧 English](README.md) | 🇮🇩 Bahasa Indonesia**

**🏠 [Kembali ke Beranda](../../README.id.md)** | **📁 [Pembelajaran Unsupervised](../README.id.md)** | **⬅️ [Sebelumnya: PCA](../04_pca/README.id.md)** | **➡️ [Berikutnya: Aturan Asosiasi](../06_association_rules/README.id.md)**

## 🗺️ Akses Cepat

| Berkas | Deskripsi | Cara Menjalankan |
|--------|-----------|------------------|
| [`implementation.py`](./implementation.py) | t-SNE NumPy dari nol, kalibrasi afinitas, gradien KL, metrik ketetanggaan, dan pembuat data segmen pelanggan | `python3 implementation.py` |
| [`sklearn_example.py`](./sklearn_example.py) | t-SNE scikit-learn vs PCA serta sapuan perplexity; menyimpan gambar PNG dengan backend Agg | `python3 sklearn_example.py` |
| [`exercise.ipynb`](./exercise.ipynb) | Latihan terpandu dengan TODO dan petunjuk | `jupyter lab exercise.ipynb` |

## 📚 Daftar Isi

- [Teori & Konsep](#-teori--konsep)
- [Dasar Matematika](#-dasar-matematika)
- [Perplexity dan Pencarian Biner Sigma](#-perplexity-dan-pencarian-biner-sigma)
- [P Simetris dan Q Student-t](#-p-simetris-dan-q-student-t)
- [Divergensi KL dan Gradiennya](#-divergensi-kl-dan-gradiennya)
- [Optimasi: Early Exaggeration, Momentum, Gains](#-optimasi-early-exaggeration-momentum-gains)
- [Algoritma Langkah demi Langkah](#-algoritma-langkah-demi-langkah)
- [Dataset Segmen Pelanggan](#-dataset-segmen-pelanggan)
- [Detail Implementasi](#-detail-implementasi)
- [Contoh scikit-learn](#-contoh-scikit-learn)
- [Kelebihan & Kekurangan](#-kelebihan--kekurangan)
- [Kapan Menggunakan](#-kapan-menggunakan)
- [Kesalahan Umum & Praktik Terbaik](#-kesalahan-umum--praktik-terbaik)
- [Berkas di Direktori Ini](#-berkas-di-direktori-ini)
- [Mulai Cepat](#-mulai-cepat)
- [Latihan](#-latihan)
- [Algoritma Terkait](#-algoritma-terkait)
- [Referensi](#-referensi)

## 🧠 Teori & Konsep

### Apa itu t-SNE?

t-distributed Stochastic Neighbor Embedding (t-SNE) adalah metode **reduksi
dimensi nonlinier dan visualisasi**. Metode ini mengambil titik-titik di ruang
fitur berdimensi tinggi lalu menempatkannya pada peta berdimensi rendah —
hampir selalu dua dimensi — sehingga titik yang berdekatan di ruang asli tetap
berdekatan di peta.

PCA adalah proyeksi linier: ia mempertahankan arah varians maksimum. t-SNE
mengoptimalkan tujuan yang berbeda. Ia berfokus pada **ketetanggaan lokal**
sehingga jauh lebih baik dalam menampilkan kelompok, manifold melengkung, dan
awan titik yang saling tumpang tindih, yang akan menyatu pada proyeksi linier.

Metode ini tidak memerlukan label: cukup matriks fitur `X`. t-SNE digunakan
untuk:

- analisis data eksploratif,
- memeriksa apakah ada klaster atau subpopulasi,
- memeriksa embedding dari jaringan saraf,
- menemukan kesalahan pelabelan atau outlier,
- mengomunikasikan struktur data berdimensi tinggi.

t-SNE adalah **alat visualisasi**, bukan algoritma klasterisasi. Ia menghasilkan
koordinat, bukan penetapan klaster. Menjalankan K-Means pada embedding t-SNE
biasanya keliru: embedding sengaja mendistorsi kepadatan dan jarak, sehingga
klaster yang ditemukan di sana tidak berkorespondensi dengan kelompok yang
andal pada data asli.

### Kemiripan Berdimensi Tinggi: Gaussian

Untuk setiap pasangan titik input `x_i` dan `x_j`, t-SNE mendefinisikan
probabilitas bersyarat bahwa `i` akan memilih `j` sebagai tetangga:

```
p_{j|i} = exp(-||x_i - x_j||^2 / (2 * sigma_i^2))
          / sum_{k != i} exp(-||x_i - x_k||^2 / (2 * sigma_i^2))
```

Setiap titik memiliki bandwidth `sigma_i` sendiri. Daerah padat membutuhkan
`sigma` kecil; daerah renggang membutuhkan `sigma` besar. Jika satu bandwidth
global dipakai, klaster padat akan runtuh menjadi satu tetangga dan klaster
renggang menjadi awan yang tidak terbedakan.

Bandwidth tidak dipilih manual untuk setiap titik. Sebagai gantinya, setiap
titik dikalibrasi ke **perplexity** yang sama, yaitu ukuran halus dari jumlah
tetangga efektif.

### Kemiripan Berdimensi Rendah: Student-t

Pada peta dua dimensi, t-SNE memakai kernel Student-t berekor berat dengan satu
derajat kebebasan:

```
q_{ij} = (1 + ||y_i - y_j||^2)^-1
         / sum_{k != l} (1 + ||y_k - y_l||^2)^-1
```

Ekor berat ini menyelesaikan **masalah crowding**. Di dua dimensi tidak cukup
ruang untuk mempertahankan semua jarak berpasangan dari ruang berdimensi tinggi.
Dengan Gaussian di peta, titik yang cukup berjauhan akan terdorong sangat jauh
dan optimasi menjadi tidak stabil. Ekor Student-t memungkinkan titik pada jarak
sedang tetap berada di peta tanpa mendominasi tujuan.

### Tujuan: Divergensi KL

t-SNE mencocokkan dua distribusi kemiripan dengan meminimalkan divergensi
Kullback-Leibler:

```
KL(P || Q) = sum_{i != j} p_{ij} * log(p_{ij} / q_{ij})
```

Sifat asimetrisnya penting. `p_{ij}` besar dengan `q_{ij}` kecil mendapat
penalti berat: **titik yang berdekatan di input harus tetap berdekatan di peta**.
`p_{ij}` kecil dengan `q_{ij}` besar hanya mendapat penalti ringan, sehingga
t-SNE boleh menempatkan titik yang tidak berhubungan di mana saja asalkan
ketetanggaan lokalnya terjaga. Inilah alasan jarak global pada plot t-SNE tidak
dapat dipercaya.

### Apa yang Dipertahankan t-SNE — dan Apa yang Tidak

| Properti | Dipertahankan? | Komentar |
|----------|----------------|----------|
| Ketetanggaan lokal | Ya, secara perkiraan | Dikendalikan perplexity dan tujuan KL |
| Jarak global | Tidak | Jarak antar-klaster tidak terkalibrasi |
| Ukuran klaster | Tidak | Klaster padat bisa tampak kecil, yang renggang bisa tampak besar |
| Kepadatan | Tidak | Kernel Student-t memperluas daerah renggang |
| Sumbu dan orientasi | Tidak | Rotasi, refleksi, dan skala bersifat arbitrer |
| Identitas klaster | Tidak | t-SNE tidak mengeluarkan label atau penetapan |

Jika Anda membutuhkan embedding kuantitatif untuk model hilir, pilih PCA atau
metode dengan transformasi out-of-sample yang eksplisit. Jika Anda membutuhkan
visualisasi nonlinier yang cepat, UMAP adalah alternatif populer; ia hanya
disebut sebagai teks di sini dan tidak diimplementasikan pada modul ini.

## 📐 Dasar Matematika

### Jarak Berpasangan

Semua kemiripan didasarkan pada jarak Euclidean kuadrat:

```
d_ij^2 = ||x_i - x_j||^2
```

Dengan menjabarkan norma, diperoleh komputasi tervektorisasi:

```
d_ij^2 = ||x_i||^2 + ||x_j||^2 - 2 * x_i . x_j
```

Ini jauh lebih cepat daripada perulangan Python ganda. Pembatalan floating-point
dapat menghasilkan nilai negatif sangat kecil, sehingga implementasi memotong
jarak pada nol.

### Probabilitas Bersyarat

Misalkan `beta_i = 1 / (2 * sigma_i^2)`. Maka:

```
p_{j|i} = exp(-beta_i * d_ij^2) / sum_{k != i} exp(-beta_i * d_ik^2)
```

Diagonal diatur nol: sebuah titik bukan tetangga dirinya sendiri. Setiap baris
adalah distribusi probabilitas atas `n - 1` titik lainnya dan berjumlah satu.

### Entropi

Entropi Shannon dari satu distribusi bersyarat adalah:

```
H(P_i) = -sum_{j != i} p_{j|i} * log(p_{j|i})
```

Entropi mengukur seberapa menyebar distribusi tetangga. `sigma` sangat kecil
menempatkan hampir seluruh massa pada titik terdekat dan memberi entropi
mendekati nol. `sigma` sangat besar menyebarkan massa ke semua titik dan memberi
entropi mendekati `log(n - 1)`.

### Perplexity

Perplexity didefinisikan sebagai:

```
perplexity(P_i) = exp(H(P_i))
```

Interpretasinya intuitif: **jumlah tetangga efektif**. `perplexity = 5` berarti
setiap titik berperilaku seolah memiliki sekitar lima tetangga bermakna;
`perplexity = 30` berarti sekitar tiga puluh.

Nilai tipikal adalah `5` sampai `50`, dengan `30` sebagai default umum. Nilai
kecil menekankan struktur sangat lokal; nilai besar memasukkan lebih banyak
konteks global. Perplexity harus lebih kecil daripada jumlah sampel, dan
sebaiknya lebih kecil dari sekitar `n / 3` agar estimasi entropi stabil.

### Gradien KL terhadap Y

Gradien eksaknya adalah:

```
dKL/dy_i = 4 * sum_j (p_ij - q_ij) * (y_i - y_j)
           / (1 + ||y_i - y_j||^2)
```

Setiap suku memiliki makna yang jelas:

- `p_ij - q_ij` adalah selisih kemiripan. Jika kemiripan input lebih besar
  daripada kemiripan peta, pasangan ditarik mendekat; jika lebih kecil, pasangan
  didorong menjauh.
- `(y_i - y_j)` memberi arah gaya.
- `1 / (1 + ||y_i - y_j||^2)` adalah turunan Student-t. Ia melemahkan gaya
  antara titik yang sudah berjauhan sehingga mencegah ledakan.

Dalam bentuk matriks, dengan `W_ij = (p_ij - q_ij) / (1 + ||y_i - y_j||^2)`:

```
grad_i = 4 * (sum_j W_ij) * y_i - 4 * sum_j W_ij * y_j
```

Implementasi memakai bentuk tervektorisasi ini. Karena `P` dan `Q` simetris dan
kedua diagonalnya nol, gradien berperilaku baik dan dapat diperiksa dengan
selisih hingga.

## 🎯 Perplexity dan Pencarian Biner Sigma

### Mengapa Perlu Pencarian

Untuk titik `i`, probabilitas bersyarat bergantung pada `beta_i`. Tidak ada
solusi bentuk tertutup untuk `beta_i` sedemikian sehingga
`H(P_i) = log(perplexity)`, karena jaraknya arbitrer. Namun, entropi **monoton
menurun** terhadap `beta`:

- menaikkan `beta` -> Gaussian lebih sempit -> lebih sedikit tetangga efektif ->
  entropi lebih rendah,
- menurunkan `beta` -> Gaussian lebih lebar -> lebih banyak tetangga efektif ->
  entropi lebih tinggi.

Sifat monoton ini membuat pencarian biner dapat diandalkan.

### Prosedur Bisection

Untuk setiap titik secara independen:

1. Hitung jarak kuadrat ke setiap titik lain.
2. Mulai dengan bracket `beta_min = 1e-20`, `beta_max = 1e20`, dan tebakan awal
   `beta = 1`.
3. Hitung baris bersyarat dan entropinya.
4. Jika entropi di atas target, Gaussian terlalu lebar: naikkan `beta_min` dan
   perbesar `beta`.
5. Jika entropi di bawah target, Gaussian terlalu sempit: turunkan `beta_max`
   dan perkecil `beta`.
6. Berhenti ketika `|H - log(perplexity)| <= tol`, atau setelah `max_iter`
   iterasi.
7. Simpan `p_{j|i}` dan `sigma_i = 1 / sqrt(beta_i)`.

Implementasi menggabungkan bracketing multiplikatif dengan bisection, yang
konvergen cepat bahkan ketika presisi berguna mencakup banyak orde besaran.

### Stabilitas Numerik

Dua detail menjaga pencarian tetap stabil:

- **Geser sebelum eksponensial.** Kurangi jarak terkecil pada baris sebelum
  menerapkan `exp(-beta * d)`. Konstanta itu saling menghapus dalam normalisasi,
  tetapi mencegah overflow untuk `beta` besar.
- **Nolkan diagonal.** Jarak titik ke dirinya sendiri adalah nol, sehingga tanpa
  penopengan ia akan menerima bobot terbesar. Diagonal dihapus sebelum
  normalisasi.

### Membaca Hasil

`compute_pairwise_affinities` mengembalikan:

- `conditional_P` berukuran `(n, n)`, dengan setiap baris berjumlah satu dan
  diagonal nol,
- `sigmas` berukuran `(n,)`, bandwidth yang ditemukan untuk setiap titik.

Diagnostik yang berguna adalah `exp(H(P_i))`: nilainya harus sama dengan
perplexity yang diminta untuk setiap baris, sampai toleransi entropi.

## 🔁 P Simetris dan Q Student-t

### Menyetarakan Probabilitas Bersyarat

Matriks bersyarat tidak simetris: `p_{j|i}` dan `p_{i|j}` dapat berbeda karena
setiap titik punya bandwidth sendiri. t-SNE memakai distribusi bersama:

```
p_ij = (p_{j|i} + p_{i|j}) / (2 * n)
```

Sifat-sifatnya:

- `p_ij = p_ji` (simetris),
- `sum_{i != j} p_ij = 1`,
- diagonal nol.

Faktor `1 / n` membuat total massa satu, dan faktor `1 / 2` merata-ratakan kedua
arah. Sebelum normalisasi, probabilitas sangat kecil dinaikkan ke `1e-12` agar
`log(p_ij / q_ij)` tidak pernah melihat nol eksak.

### Kemiripan Student-t di Peta

Untuk embedding `Y`, hitung jarak kuadrat `||y_i - y_j||^2` lalu definisikan:

```
w_ij = 1 / (1 + ||y_i - y_j||^2)
q_ij = w_ij / sum_{k != l} w_kl
```

Diagonal nol dan `Q` berjumlah satu. Kernel ini memiliki ekor lebih berat
daripada Gaussian: bahkan pasangan yang cukup berjauhan tetap memiliki
probabilitas yang tidak dapat diabaikan. Hal ini memberi ruang bagi optimizer
untuk menata kelompok yang berjauhan tanpa gaya tolak yang sangat besar.

## 📉 Divergensi KL dan Gradiennya

### Divergensi

```
KL(P || Q) = sum_{i != j} p_ij * log(p_ij / q_ij)
```

Fakta penting:

- `KL(P || P) = 0`,
- `KL(P || Q) >= 0`,
- tidak simetris: `KL(P || Q) != KL(Q || P)` pada umumnya,
- suku dengan `p_ij = 0` tidak memberi kontribusi,
- `p_ij` positif dengan `q_ij = 0` menghasilkan divergensi tak hingga
  (implementasi menghindarinya dengan menaikkan probabilitas minimum).

### Gradien

```
dKL/dy_i = 4 * sum_j (p_ij - q_ij) * (y_i - y_j)
           / (1 + ||y_i - y_j||^2)
```

Faktor `4` berasal dari diferensiasi jarak kuadrat di dalam kernel Student-t.
Uji selisih hingga pada rangkaian tes memeriksa rumus ini secara langsung:
menggeser satu koordinat `Y` lalu membandingkan
`(KL(Y + eps) - KL(Y - eps)) / (2 * eps)` dengan gradien analitik memberi
kesesuaian sekitar `1e-9` kesalahan relatif pada masalah kecil.

### Mengapa Gradien Ini Bekerja

Misalkan `p_ij` besar tetapi `q_ij` kecil: pasangan dekat di input tetapi jauh di
peta. Maka `p_ij - q_ij > 0`, dan gaya menarik `y_i` ke arah `y_j`. Misalkan
`p_ij` kecil tetapi `q_ij` besar: pasangan jauh di input tetapi dekat di peta.
Maka `p_ij - q_ij < 0`, dan gaya mendorong keduanya menjauh. Penyebut memperkecil
kedua gaya saat pasangan terpisah.

## 🚀 Optimasi: Early Exaggeration, Momentum, Gains

t-SNE meminimalkan tujuan non-konveks dengan gradient descent. Tiga trik praktis
membuat optimasi andal.

### Early Exaggeration

Untuk `early_exaggeration_iter` iterasi pertama, kalikan probabilitas bersama:

```
P_exaggerated = early_exaggeration * P
```

Pengali default adalah `12`. Ini memperkuat tarikan dan mendorong peta membentuk
klaster yang rapat dan terpisah sejak awal. Setelah fase exaggeration, optimizer
memakai `P` sebenarnya dan memperhalus tata letak. Tanpa early exaggeration,
klaster bisa tetap menyatu atau terpisah terlalu lambat.

### Momentum

Gradient descent biasa berzig-zag di lembah sempit. Momentum mengakumulasi
kecepatan:

```
update_t = momentum * update_{t-1} - learning_rate * gains * gradient_t
Y_t = Y_{t-1} + update_t
```

Nilai tipikal adalah `0.5` selama early exaggeration dan `0.8` sesudahnya.
Momentum awal yang lebih rendah menghindari overshoot saat gaya yang
diperbesar masih besar; momentum akhir yang lebih tinggi mempercepat konvergensi.

### Gains Adaptif

Setiap koordinat memiliki faktor gain sendiri, dimulai dari `1`:

- jika gradien mempertahankan tanda yang sama dengan update sebelumnya, gain
  bertambah `0.2`,
- jika gradien berubah tanda, gain dikalikan `0.8`,
- gain dibatasi di bawah oleh `min_gain` (default `0.01`).

Ini adalah adaptasi ukuran langkah diagonal per koordinat. Koordinat yang
konsisten bergerak ke satu arah dapat mengambil langkah lebih besar, sementara
koordinat yang berosilasi diredam.

### Recentering

Setelah setiap update, embedding digeser agar rata-ratanya nol. Hanya posisi
relatif yang penting, sehingga menghapus drift translasi tidak mengubah tujuan
tetapi menjaga angka tetap terkondisi baik.

### Learning Rate

Default adalah `200`. Dengan `learning_rate="auto"`, implementasi memakai:

```
learning_rate = max(n_samples / early_exaggeration / 4, 50)
```

Learning rate terlalu besar menghasilkan peta yang runtuh atau meledak; terlalu
kecil membuat klaster tetap menyatu dan butuh banyak iterasi untuk konvergen.

### Berhenti Lebih Awal?

Implementasi referensi menjalankan jumlah iterasi tetap dan mencatat
divergensi KL pada setiap langkah. Dalam praktik, Anda dapat memeriksa
`kl_history_` dan berhenti saat nilainya mendatar. Riwayat **tidak** dijamin
menurun pada setiap iterasi: momentum dan early exaggeration dapat menyebabkan
kenaikan kecil sementara. Yang penting adalah tren penurunan secara keseluruhan
dan kualitas ketetanggaan akhir.

## ⚙️ Algoritma Langkah demi Langkah

1. **Validasi dan standardisasi input.** Fitur dengan skala sangat berbeda
   mendominasi jarak Euclidean, jadi standardisasi terlebih dahulu.
2. **Hitung jarak Euclidean kuadrat** di ruang input.
3. **Kalibrasi satu Gaussian per titik** dengan pencarian biner pada `beta`
   sampai entropinya sama dengan `log(perplexity)`. Simpan probabilitas
   bersyarat dan `sigma_i`.
4. **Setarakan dan normalisasi** probabilitas bersyarat menjadi `P`.
5. **Inisialisasi `Y`.** Inisialisasi PCA (default) menempatkan peta di sepanjang
   arah varians utama dan biasanya lebih stabil daripada inisialisasi acak.
   Embedding diperkecil agar kemiripan awal hampir seragam.
6. **Jalankan gradient descent.** Untuk setiap iterasi:
   - terapkan early exaggeration bila sesuai,
   - hitung `Q` dengan kernel Student-t,
   - hitung divergensi KL dan gradiennya,
   - perbarui gains, kecepatan, dan koordinat,
   - lakukan recentering `Y`,
   - catat divergensi KL mentah.
7. **Kembalikan embedding.** `P`, `Q`, `sigmas`, dan `kl_history_` akhir tetap
   tersedia pada objek yang sudah dilatih.

### Kompleksitas

Dengan afinitas berpasangan eksak, satu iterasi membutuhkan waktu
`O(n^2 * d)` dan memori `O(n^2)`. Fase kalibrasi juga membutuhkan `O(n^2)`
komputasi jarak ditambah sejumlah kecil iterasi bisection per titik. Untuk `n`
besar, gunakan metode aproksimasi seperti Barnes-Hut t-SNE atau implementasi
pustaka; modul ini sengaja mengimplementasikan algoritma eksak untuk kejelasan.

## 🛍️ Dataset Segmen Pelanggan

`generate_customer_segments(n_samples=300, n_features=12, n_clusters=5,
cluster_std=0.7, separation=3.0, random_state=42)` membuat dataset
outlet/pelanggan sintetis yang dapat direproduksi dengan segmen yang diketahui:

- **12 fitur perilaku** dalam rentang 10-20 dimensi yang dimaksud:

  | Fitur | Arti |
  |-------|------|
  | `monthly_spend` | Rata-rata belanja bulanan |
  | `visit_frequency` | Kunjungan per bulan |
  | `avg_basket_size` | Item atau nilai per keranjang |
  | `days_since_last_visit` | Recency |
  | `product_diversity` | Jumlah kategori produk berbeda |
  | `promo_sensitivity` | Respons terhadap promosi |
  | `weekend_share` | Porsi kunjungan akhir pekan |
  | `online_share` | Porsi pesanan online |
  | `loyalty_score` | Keterlibatan program loyalitas |
  | `basket_value_std` | Variabilitas nilai keranjang |
  | `fresh_product_share` | Porsi produk segar |
  | `snack_share` | Porsi camilan dan minuman |

- **Lima segmen laten**: keluarga hemat, pembeli praktis, pelanggan premium
  loyal, pemburu promo, dan pembeli borongan akhir pekan.
- **Label seimbang dan diacak** agar urutan segmen tidak bocor ke matriks fitur.
- **Pusat yang terpisah jelas** dengan noise Gaussian, sehingga t-SNE, PCA, dan
  K-Means mudah dibandingkan.

Fungsi mengembalikan `(X, labels)`. `labels` hanya diketahui karena datanya
sintetis; model t-SNE tidak pernah melihatnya. Gunakan label untuk mewarnai plot
dan memeriksa ketetanggaan, bukan sebagai input model.

Jalankan tipikal pada 240 sampel dengan perplexity `30` menghasilkan
trustworthiness sekitar `0.97-0.99` dan pulau-pulau yang terpisah jelas. Angka
pastinya bergantung pada seed dan jumlah iterasi.

## 🛠️ Detail Implementasi

### API Publik

```python
from implementation import (
    compute_pairwise_affinities,
    joint_probabilities,
    q_distribution,
    kl_divergence,
    kl_gradient,
    TSNE,
    trustworthiness,
    knn_preservation,
    generate_customer_segments,
    FEATURE_NAMES,
    SEGMENT_NAMES,
)
```

### Contoh Minimal

```python
from implementation import TSNE, generate_customer_segments, trustworthiness

X, labels = generate_customer_segments(
    n_samples=240, n_features=12, n_clusters=5, random_state=42
)

model = TSNE(
    perplexity=30.0,
    learning_rate=200.0,
    n_iter=600,
    early_exaggeration=12.0,
    early_exaggeration_iter=200,
    random_state=42,
)
embedding = model.fit_transform(X)

print(embedding.shape)        # (240, 2)
print(model.kl_history_[0])   # KL awal
print(model.kl_history_[-1])  # KL akhir
print(trustworthiness(X, embedding, n_neighbors=10))
```

### Kalibrasi Afinitas

```python
conditional_P, sigmas = compute_pairwise_affinities(
    X, perplexity=30.0, tol=1e-5
)
P = joint_probabilities(conditional_P)

print(conditional_P.shape)  # (n, n)
print(P.sum())              # 1.0
print(sigmas.min(), sigmas.max())
```

### Tujuan Tingkat Rendah

```python
import numpy as np

rng = np.random.default_rng(0)
Y = rng.normal(size=(X.shape[0], 2))
Q = q_distribution(Y)
loss = kl_divergence(P, Q)
gradient = kl_gradient(P, Q, Y)
```

### Parameter

| Parameter | Arti | Nilai tipikal |
|-----------|------|---------------|
| `perplexity` | Tetangga efektif per titik | `5`-`50`, default `30` |
| `learning_rate` | Ukuran langkah gradien, atau `"auto"` | `50`-`1000`, default `200` |
| `n_iter` | Jumlah iterasi optimasi | `500`-`2000` |
| `max_iter` | Alias opsional yang menimpa `n_iter` | integer atau `None` |
| `early_exaggeration` | Pengali tarikan di awal | `4`-`20`, default `12` |
| `early_exaggeration_iter` | Panjang fase exaggeration | `100`-`300` |
| `momentum` | Momentum sebelum exaggeration berakhir | `0.5` |
| `final_momentum` | Momentum sesudahnya | `0.8` |
| `min_gain` | Batas bawah gain | `0.01` |
| `init` | Strategi inisialisasi | `"pca"` atau `"random"` |
| `random_state` | Seed untuk reproduksibilitas | integer atau `None` |
| `tol` | Toleransi entropi | `1e-5` |

### Atribut Hasil Latih

| Atribut | Ukuran | Arti |
|---------|--------|------|
| `embedding_` | `(n, 2)` | Koordinat dimensi rendah yang dipelajari |
| `P_` | `(n, n)` | Afinitas input bersama yang simetris |
| `Q_` | `(n, n)` | Kemiripan Student-t akhir |
| `sigmas_` | `(n,)` | Bandwidth Gaussian per titik |
| `kl_history_` | list | Divergensi KL mentah sekali per iterasi; entri terakhir milik embedding yang dikembalikan |
| `n_iter_` | int | Iterasi yang benar-benar dijalankan |

### Pembantu Ketetanggaan

```python
trust = trustworthiness(X, embedding, n_neighbors=10)
preserved = knn_preservation(X, embedding, n_neighbors=10)
```

- **Trustworthiness** memberi penalti pada titik yang bertetangga di peta tetapi
  jauh di ruang input. Nilainya `1` untuk embedding sempurna dan menurun ketika
  muncul ketetanggaan palsu.
- **kNN preservation** adalah rata-rata tumpang tindih antara `k` tetangga
  terdekat di input dan di embedding. Mudah ditafsirkan tetapi lebih ketat
  daripada trustworthiness.

Kedua pembantu berguna sebagai pemeriksaan kewajaran, bukan pengganti melihat
plot dan data asli.

### Kasus Khusus dan Validasi

- `perplexity` harus lebih besar dari `1` dan lebih kecil dari `n_samples`.
- `n_iter` harus positif; jika fase early exaggeration tidak lebih pendek dari
  `n_iter`, fase itu diperkecil menjadi seperempat durasi agar masih ada fase
  penghalusan.
- `init` harus `"pca"` atau `"random"`.
- Fitur tidak berhingga memunculkan `ValueError` yang jelas.
- `joint_probabilities` menolak matriks negatif atau tidak persegi.
- Uji gradien paling mudah pada `n` kecil karena algoritma eksak berbiaya
  `O(n^2)`.

## 📊 Contoh scikit-learn

`sklearn_example.py` memakai implementasi scikit-learn yang terpasang untuk
perbandingan praktis:

1. membuat 240 baris outlet/pelanggan dengan lima segmen yang diketahui,
2. memproyeksikannya ke dua dimensi dengan PCA,
3. menyematkannya dengan t-SNE scikit-learn pada perplexity `30`,
4. menyimpan `tsne_sklearn_vs_pca.png`,
5. menyapu perplexity `5`, `30`, dan `80`, mencetak trustworthiness dan kNN
   preservation, lalu menyimpan `tsne_perplexity_effect.png`.

Skrip memanggil `matplotlib.use("Agg")` sebelum mengimpor `pyplot`, sehingga
berjalan di server headless atau CI. Ia memakai `max_iter=500` dan sampel kecil
agar tetap cepat.

Keluaran tipikal:

```
PCA   trustworthiness: 0.955
t-SNE trustworthiness: 0.978

perplexity     5: trustworthiness=0.967, kNN preservation=0.498
perplexity    30: trustworthiness=0.978, kNN preservation=0.568
perplexity    80: trustworthiness=0.967, kNN preservation=0.445
```

Angka pastinya sedikit bervariasi menurut versi pustaka. Pelajaran kualitatifnya
stabil: t-SNE mempertahankan ketetanggaan lokal lebih baik daripada PCA, dan
perplexity mengendalikan trade-off lokal/global.

## ✅ Kelebihan & Kekurangan

### ✅ Kelebihan

- Menampilkan struktur lokal nonlinier yang mungkin terlewat oleh PCA.
- Menghasilkan klaster yang rapat dan mudah ditafsirkan secara visual.
- Menangani manifold melengkung dan awan titik yang tumpang tindih.
- Tidak memerlukan label; bekerja langsung pada matriks fitur.
- Tujuan dan gradiennya bersih secara matematis dan mudah diimplementasikan.
- Perplexity adalah tombol yang lebih intuitif daripada bandwidth mentah.
- Kernel berekor berat mengurangi masalah crowding di dimensi rendah.
- Sangat baik untuk eksplorasi, debugging, dan mengomunikasikan struktur.

### ❌ Kekurangan

- **Ukuran klaster dan jarak antar-klaster tidak bermakna.**
- Geometri global tidak dipertahankan; peta dapat berotasi, berefleksi, dan
  terdistorsi.
- Tidak menghasilkan label klaster, centroid, atau transformasi yang dapat
  dipakai ulang untuk titik baru.
- t-SNE eksak berbiaya `O(n^2)` per iterasi, mahal untuk dataset besar.
- Tujuannya non-konveks; seed berbeda dapat menghasilkan tata letak berbeda.
- Hasil bergantung pada perplexity, learning rate, inisialisasi, dan jumlah
  iterasi.
- Tidak ada ekstensi out-of-sample alami: menambahkan titik biasanya memerlukan
  menjalankan ulang seluruh embedding.
- Konsentrasi jarak di dimensi tinggi dapat membuat ketetanggaan berisik.
- Plot dua dimensi dapat menyembunyikan struktur yang membutuhkan tiga dimensi.

## 🎯 Kapan Menggunakan

### ✅ Gunakan t-SNE ketika

- Anda ingin **mengeksplorasi** data berdimensi tinggi secara visual.
- Anda menduga ada klaster lokal atau subpopulasi dan ingin melihatnya.
- Anda membutuhkan alternatif nonlinier PCA untuk plot dua dimensi.
- Anda sedang memeriksa representasi yang dipelajari seperti embedding gambar
  atau teks.
- Anda sanggup membayar biaya algoritma eksak atau memiliki implementasi
  aproksimasi.
- Anda akan memvalidasi temuan pada fitur asli dan dengan metrik ketetanggaan
  kuantitatif.

### ❌ Pertimbangkan alternatif ketika

- Anda membutuhkan proyeksi linier yang dapat ditafsirkan dengan explained
  variance: gunakan PCA.
- Anda membutuhkan visualisasi nonlinier cepat untuk data sangat besar: gunakan
  implementasi t-SNE aproksimasi atau UMAP.
- Anda membutuhkan penetapan klaster: gunakan K-Means, DBSCAN, klasterisasi
  hierarkis, atau model campuran pada fitur asli.
- Anda perlu menyematkan titik baru tanpa melatih ulang: gunakan PCA,
  autoencoder parametrik, atau metode dengan transformasi eksplisit.
- Anda perlu mempertahankan jarak global: gunakan PCA, MDS klasik, atau metode
  yang dirancang untuk tujuan itu.
- Anda membutuhkan kesimpulan kausal atau prediktif: gunakan pembelajaran
  terbimbing dan validasi, bukan visualisasi.

## 💡 Kesalahan Umum & Praktik Terbaik

1. **Jangan menafsirkan ukuran klaster.** t-SNE memperluas daerah renggang dan
   memampatkan yang padat. Pulau yang tampak besar belum tentu segmen besar.
2. **Jangan menafsirkan jarak antar-klaster.** Jarak antara dua pulau tidak
   mengukur seberapa berbeda kelompoknya. Hanya ketetanggaan lokal yang bermakna.
3. **Jangan memperlakukan t-SNE sebagai klasterisasi.** Ia mengembalikan
   koordinat, bukan label. Jalankan algoritma klasterisasi pada fitur asli dan
   bandingkan hasilnya.
4. **Standardisasi fitur.** Fitur dalam satuan juta akan mendominasi fitur dalam
   pecahan. Gunakan `StandardScaler` atau transformasi setara.
5. **Tuning perplexity dengan bijak.** Coba beberapa nilai antara `5` dan `50`.
   Perplexity rendah menonjolkan kelompok lokal kecil dan noise; perplexity
   tinggi dapat menyatukan klaster kecil.
6. **Gunakan beberapa seed.** Tujuannya non-konveks. Struktur yang stabil di
   berbagai seed lebih kredibel daripada satu plot yang indah.
7. **Jangan membandingkan koordinat absolut antar-jalankan.** Peta dapat
   berotasi, berefleksi, dan berubah skala. Bandingkan ketetanggaan, bukan nilai
   `x`/`y`.
8. **Beri cukup iterasi untuk early exaggeration.** Terlalu singkat membuat
   klaster tetap menyatu; terlalu lama membuat peta terlalu rapat secara
   artifisial.
9. **Periksa riwayat KL.** Nilainya harus cenderung menurun, tetapi kenaikan
   kecil sementara normal karena momentum dan exaggeration.
10. **Perhatikan waktu jalan dan memori.** t-SNE eksak bersifat kuadratik. Untuk
    lebih dari beberapa ribu titik, gunakan metode aproksimasi.
11. **Validasi pada fitur asli.** Plot yang indah adalah hipotesis. Konfirmasi
    perbedaan kelompok dengan statistik atau model terbimbing.
12. **Ingat konsentrasi jarak di dimensi tinggi.** Di dimensi sangat tinggi,
    tetangga terdekat dan terjauh dapat memiliki jarak serupa, sehingga struktur
    lokal kurang andal.
13. **Hati-hati dengan fitur kategorikal.** Encode dengan bijak; jarak Euclidean
    one-hot mentah mungkin tidak mencerminkan kemiripan yang bermakna.
14. **Jangan membaca kausalitas dari peta.** Klaster adalah pola kemiripan, bukan
    mekanisme kausal.

## 📁 Berkas di Direktori Ini

- **`README.md`** — teori dan konsep dalam bahasa Inggris
- **`README.id.md`** (berkas ini) — dokumentasi lengkap dalam Bahasa Indonesia
- **`implementation.py`** — kalibrasi afinitas, `P` simetris, `Q` Student-t,
  divergensi dan gradien KL, `TSNE`, trustworthiness, kNN preservation, segmen
  pelanggan sintetis, dan demo `__main__`
- **`sklearn_example.py`** — t-SNE scikit-learn vs PCA dan sapuan perplexity,
  menyimpan gambar PNG dengan backend Agg
- **`exercise.ipynb`** — latihan langsung dengan TODO dan petunjuk

## 🚀 Mulai Cepat

Jalankan demo dari nol:

```bash
python3 02_unsupervised_learning/05_tsne/implementation.py
```

Jalankan perbandingan scikit-learn dan simpan gambar:

```bash
python3 02_unsupervised_learning/05_tsne/sklearn_example.py
```

Jalankan tes:

```bash
pytest tests/test_tsne.py -v
```

Penggunaan minimal:

```python
from implementation import TSNE, generate_customer_segments, trustworthiness

X, labels = generate_customer_segments(
    n_samples=240, n_features=12, n_clusters=5, random_state=42
)
model = TSNE(perplexity=30.0, n_iter=600, random_state=42)
embedding = model.fit_transform(X)

print(embedding.shape)
print(model.kl_history_[-1])
print(trustworthiness(X, embedding, n_neighbors=10))
```

## 🎓 Latihan

Buku catatan [`exercise.ipynb`](./exercise.ipynb) memandu Anda melalui:

1. membuat dan memeriksa data segmen pelanggan,
2. mengalibrasi `sigma` dan memverifikasi entropi target,
3. membangun `P` simetris dan `Q` Student-t,
4. memeriksa gradien analitik dengan selisih hingga,
5. melatih t-SNE dari nol dan membaca riwayat KL,
6. mengukur trustworthiness dan kNN preservation,
7. membandingkan nilai perplexity,
8. mengamati pengaruh early exaggeration.

Setiap sel dapat dijalankan dari atas ke bawah dalam beberapa detik. TODO dan
petunjuk mendorong Anda mengubah nilai dan mengamati hasilnya.

## 🔗 Algoritma Terkait

- **PCA** ([04_pca](../04_pca/)): cepat, linier, mempertahankan varians global;
  langkah pertama yang alami sebelum t-SNE.
- **K-Means** ([01_kmeans](../01_kmeans/)): menghasilkan penetapan klaster keras
  pada fitur asli; t-SNE hanya memvisualisasikannya.
- **DBSCAN** ([02_dbscan](../02_dbscan/)): klasterisasi berbasis kepadatan yang
  dapat menemukan bentuk arbitrer dan memberi label noise.
- **Klasterisasi Hierarkis** ([03_hierarchical](../03_hierarchical/)):
  pengelompokan berbasis dendrogram dengan gagasan jarak eksplisit.
- **Aturan Asosiasi** ([06_association_rules](../06_association_rules/)):
  menemukan pola kemunculan bersama pada data transaksi, tujuan unsupervised yang
  berbeda.
- **Autoencoder** (topik lanjutan): mempelajari embedding nonlinier yang dapat
  menyematkan titik baru, tidak seperti t-SNE.

## 📚 Referensi

- van der Maaten, L., & Hinton, G. (2008). *Visualizing Data using t-SNE*.
  Journal of Machine Learning Research, 9, 2579-2605.
- van der Maaten, L. (2014). *Accelerating t-SNE using Tree-Based Algorithms*.
  Journal of Machine Learning Research, 15, 3221-3245.
- Hinton, G., & Roweis, S. (2002). *Stochastic Neighbor Embedding*. Advances in
  Neural Information Processing Systems.
- [Dokumentasi TSNE scikit-learn](https://scikit-learn.org/stable/modules/generated/sklearn.manifold.TSNE.html)
- [Panduan manifold learning scikit-learn](https://scikit-learn.org/stable/modules/manifold.html)
- [t-SNE (Wikipedia)](https://en.wikipedia.org/wiki/T-distributed_stochastic_neighbor_embedding)
- [How to Use t-SNE Effectively (Distill)](https://distill.pub/2016/misread-tsne/)

---

**Langkah Selanjutnya**:
- Baca `implementation.py` untuk melihat bisection, kernel Student-t, dan
  gradien tervektorisasi.
- Jalankan `sklearn_example.py` dan bandingkan panel PCA dengan t-SNE.
- Selesaikan `exercise.ipynb` dan coba beberapa perplexity pada data yang sama.
- Terapkan t-SNE pada fitur ternormalisasi Anda sendiri, tetapi validasi klaster
  yang tampak pada data asli sebelum bertindak berdasarkan hasilnya.
