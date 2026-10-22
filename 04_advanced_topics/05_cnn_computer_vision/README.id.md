# Convolutional Neural Network (CNN)

**🇮🇩 Bahasa Indonesia | [🇬🇧 English](README.md)**

🏠 Kembali ke Indeks Topik Lanjutan: `../README.md`

Convolutional Neural Network adalah alat utama untuk mempelajari pola dari
gambar. Alih-alih menghubungkan setiap piksel ke setiap neuron, CNN menggeser
filter kecil yang dapat dipelajari ke seluruh gambar, memakai filter yang sama
di setiap posisi, lalu membangun fitur yang makin abstrak lapis demi lapis.
Hasilnya adalah model dengan parameter jauh lebih sedikit daripada jaringan
dense dan memiliki prior bawaan yang kuat terhadap struktur spasial.

Modul ini mengimplementasikan CNN kecil secara **lengkap dari nol dengan
NumPy**: konvolusi melalui `im2col`, max pooling, ReLU, lapisan dense, softmax
cross-entropy yang stabil secara numerik, perambatan balik, dan SGD dengan
momentum. Model kemudian dilatih pada gambar sintetis yang meniru kamera yang
dipakai SPG (pramuniaga) untuk mengaudit rak warung.

## 📚 Daftar Isi

- [Intuisi](#-intuisi)
- [Fondasi Matematis](#-fondasi-matematis)
- [Arsitektur](#-arsitektur)
- [Dataset Rak Warung](#-dataset-rak-warung)
- [Mulai Cepat](#-mulai-cepat)
- [Hasil](#-hasil)
- [Apa yang Dipelajari Jaringan](#-apa-yang-dipelajari-jaringan)
- [Pengujian](#-pengujian)
- [Keuntungan dan Kerugian](#-keuntungan-dan-kerugian)
- [Kapan Menggunakan CNN](#-kapan-menggunakan-cnn)
- [File dalam Direktori Ini](#-file-dalam-direktori-ini)
- [Dari CNN ke deteksi objek (YOLO)](#-dari-cnn-ke-deteksi-objek-yolo)
- [Referensi](#-referensi)

## 🧠 Intuisi

### Mengapa bukan jaringan dense?

Gambar greyscale berukuran 16x16 memiliki 256 piksel. Satu lapisan tersembunyi
dense dengan 64 unit sudah membutuhkan `256 * 64 = 16.384` bobot. Foto RGB
berukuran 224x224 memiliki 150.528 masukan, sehingga lapisan dense pertama
memerlukan jutaan bobot sebelum sempat mempelajari sesuatu yang berguna.
Lapisan dense juga membuang fakta bahwa piksel yang berdekatan saling berkaitan.

Kernel konvolusi jauh lebih efisien. Filter 3x3 hanya memiliki 9 bobot per
kanal masukan, dan filter yang **sama** dipakai ulang di setiap lokasi spasial.
Ini disebut **berbagi bobot** (weight sharing). Ada dua sifat penting yang
muncul:

- **Konektivitas lokal:** setiap keluaran hanya bergantung pada lingkungan kecil.
- **Equivariansi translasi:** jika pola bergeser, respons filter ikut bergeser
  dan tidak langsung hilang.

### Cerita rak warung

Seorang SPG mengunjungi warung, mengarahkan ponsel ke rak, lalu mengambil
potongan gambar. Pihak bisnis ingin mengetahui jenis produk pada potongan
tersebut: kemasan kotak, botol, strip sachet, atau kaleng. Produk dapat berada
di posisi mana saja, dengan ukuran dan kecerahan yang sedikit berbeda, dan
kamera menambahkan noise.

Classifier dense bisa menghafal posisi piksel. CNN mempelajari filter untuk
tepi, garis, dan batas melengkung, sehingga dapat mengenali produk yang sama
meskipun posisinya bergeser. Perbedaan inilah yang disebut **bias induktif** dan
diukur di `sklearn_example.py`.

### Hierarki fitur

```
lapisan awal    -> tepi, sudut, gumpalan kecerahan
lapisan tengah  -> garis, tutup botol, bibir kaleng, bagian kemasan
lapisan dalam   -> objek utuh dan pola tingkat kelas
```

Setiap konvolusi yang diikuti pooling memperbesar medan reseptif sebuah neuron,
sehingga unit yang lebih dalam "melihat" bagian gambar yang lebih luas.

## 📐 Fondasi Matematis

### Ukuran keluaran konvolusi

Untuk masukan dengan tinggi `H` dan lebar `W`, kernel berukuran `K_h x K_w`,
stride `S`, dan zero padding `P`, ukuran keluarannya adalah:

```
H_out = floor((H + 2P - K_h) / S) + 1
W_out = floor((W + 2P - K_w) / S) + 1
```

Pada modul ini kernel, stride, dan padding sering berbentuk persegi, sehingga
rumusnya menjadi `(H + 2P - K) / S + 1` dengan pembagian bilangan bulat. Dengan
masukan `16x16`, kernel `3x3`, dan `padding=1`, konvolusi mempertahankan ukuran
spasial `16x16`. Max pool `2x2` dengan stride 2 setelahnya mengecilkannya
menjadi `8x8`.

### Cross-correlation

Pustaka deep learning menyebut operasinya "konvolusi", tetapi dalam praktik
mereka menghitung **cross-correlation** (kernel tidak dibalik):

```
Y[n, o, i, j] = b[o]
              + jumlah atas c, u, v dari
                X_pad[n, c, i*S + u, j*S + v] * W[o, c, u, v]
```

dengan `n` indeks batch, `o` kanal keluaran, `c` kanal masukan, dan `(u, v)`
posisi di dalam kernel. Modul ini mengikuti konvensi tersebut.

### im2col: konvolusi sebagai perkalian matriks

Implementasi naif membutuhkan enam loop bersarang (batch, kanal keluaran, baris
keluaran, kolom keluaran, kanal masukan, baris kernel, kolom kernel). Itu lambat
di Python. `im2col` menyusun setiap medan reseptif menjadi satu baris matriks:

```
X : (N, C, H, W)
cols = im2col(X, K, S, P)
     : (N * H_out * W_out, C * K_h * K_w)

W_matrix = W.reshape(out_ch, C * K_h * K_w)

Y_flat = cols @ W_matrix.T + b
Y      = Y_flat.reshape(N, H_out, W_out, out_ch).transpose(0, 3, 1, 2)
```

Bagian yang mahal berubah menjadi satu perkalian matriks yang dijalankan NumPy
dalam kode C yang teroptimasi. `col2im` adalah operasi adjoin eksaknya: ia
menjumlahkan setiap gradien patch kembali ke lokasi gambar asalnya.

### Perambatan balik melalui konvolusi

Misalkan `dY` adalah gradien fungsi kerugian terhadap keluaran konvolusi.
Dengan patch yang sama yang disimpan oleh `im2col`:

```
dW[o, c, u, v] = jumlah atas n, i, j dari
                 dY[n, o, i, j] * X_pad[n, c, i*S + u, j*S + v]

db[o] = jumlah atas n, i, j dari dY[n, o, i, j]

dX_pad[n, c, i*S + u, j*S + v] +=
    jumlah atas o dari dY[n, o, i, j] * W[o, c, u, v]
```

`col2im` melakukan akumulasi terakhir, lalu padding dibuang di bagian akhir.

### Perambatan balik melalui max pooling

Untuk satu jendela, max pooling hanya menyimpan nilai terbesar:

```
Y[n, c, i, j] = max atas (u, v) dari X[n, c, i*S + u, j*S + v]
```

Gradien dari operasi max hanya dialirkan ke posisi masukan yang mencapai nilai
maksimum. Posisi lain menerima nol. Lapisan menyimpan indeks argmax saat
perambatan maju dan memakainya saat perambatan balik. Jika beberapa nilai sama,
nilai pertama di dalam jendela yang menerima gradien.

### ReLU dan gradiennya

```
ReLU(x) = max(0, x)
dReLU/dx = 1 jika x > 0, selain itu 0
```

ReLU murah dan tidak jenuh untuk masukan positif, sehingga menjadi pilihan
default yang baik untuk lapisan tersembunyi pada CNN kecil ini.

### Softmax cross-entropy

Untuk logit `z` dan `K` kelas:

```
shifted = z - max(z)                       # stabilitas numerik
p_k     = exp(shifted_k) / jumlah_j exp(shifted_j)
L       = -(1/N) * jumlah_i log(p_i[y_i])
```

Gradien terhadap logit memiliki bentuk ringkas:

```
dL/dz_i = (p_i - one_hot(y_i)) / N
```

Mengurangi logit maksimum mencegah `exp` meluap. Implementasi juga merata-ratakan
fungsi kerugian dan gradien pada mini-batch, sesuai skala yang diharapkan
pengoptimal.

### Inisialisasi He

Bobot diambil dari distribusi Gaussian dengan rata-rata nol dan variansi `2 / fan_in`,
dengan `fan_in` adalah jumlah masukan ke sebuah unit:

```
W ~ Normal(0, sqrt(2 / fan_in))
b = 0
```

Ini menjaga variansi aktivasi relatif stabil melewati lapisan ReLU dan
menghindari sinyal yang menghilang atau meledak akibat bobot awal yang terlalu
kecil atau terlalu besar.

### SGD dengan momentum

Untuk parameter `theta` dan gradiennya `g`:

```
v <- momentum * v + g
theta <- theta - learning_rate * v
```

Momentum mengumpulkan kecepatan pada arah yang terus konsisten, sehingga
menghaluskan pembaruan mini-batch yang berisik. Dengan `momentum=0`,
pembaruannya adalah penurunan gradien biasa.

## 🏗️ Arsitektur

`ShelfCNN` adalah pengklasifikasi konvolusi ringkas dengan dua blok:

```
Masukan: (N, 1, 16, 16) potongan greyscale
   |
   v
Conv2D: 1 -> 8 kanal, kernel 3x3, stride 1, padding 1      -> (N, 8, 16, 16)
ReLU
MaxPool2D: 2x2, stride 2                                   -> (N, 8, 8, 8)
   |
   v
Conv2D: 8 -> 16 kanal, kernel 3x3, stride 1, padding 1     -> (N, 16, 8, 8)
ReLU
MaxPool2D: 2x2, stride 2                                   -> (N, 16, 4, 4)
   |
   v
Flatten                                                    -> (N, 256)
Dense: 256 -> 64
ReLU
Dense: 64 -> 4                                             -> (N, 4)
   |
   v
Softmax cross-entropy
```

`padding=1` pada kedua konvolusi mempertahankan ukuran spasial sebelum setiap
lapisan pooling. Model karenanya memiliki bentuk yang dapat diprediksi dan
bekerja untuk ukuran masukan persegi yang merupakan kelipatan empat.

Setiap lapisan menyediakan:

- `forward(x) -> y`
- `backward(dout) -> dx`
- kamus `.params` dan `.grads` untuk tensor yang dapat dipelajari

`Sequential` cukup memanggil lapisan berurutan saat perambatan maju dan dalam
urutan terbalik saat perambatan balik. `SGD` membaca gradien dari `.grads` dan
memperbarui `.params`, sehingga tidak ada lapisan yang perlu mengetahui detail
pengoptimal.

## 🏪 Dataset Rak Warung

`generate_shelf_images` membuat gambar sintetis yang reproducible. Idenya
terinspirasi potongan gambar dari kamera SPG, tetapi **tidak ada gambar produk
nyata yang dipakai**.

| Kelas | Jenis produk | Bentuk yang digambar |
|-------|--------------|----------------------|
| 0 | kemasan kotak | persegi panjang penuh dengan pita label lebih gelap |
| 1 | botol | badan tinggi dan tipis dengan tutup pendek |
| 2 | strip sachet | tiga batang horizontal yang dipisahkan celah |
| 3 | kaleng | lingkaran penuh dengan bibir lebih terang |

Untuk setiap gambar, generator:

1. Menggambar latar rak gelap dengan gradien vertikal halus.
2. Memilih kelas dan menggambar bentuk yang sesuai di posisi acak.
3. Mengacak ukuran dan kecerahannya.
4. Menambahkan noise piksel Gaussian.
5. Memotong nilai agar berada di `[0, 1]`.

Tanda tangan fungsinya adalah:

```python
X, y = generate_shelf_images(
    n_samples=300,
    image_size=16,
    n_classes=4,     # 3 atau 4
    noise=0.05,
    random_state=42,
)
# X: (n_samples, 1, image_size, image_size), float32
# y: (n_samples,), int
```

Label seimbang secara konstruksi. `random_state` yang sama selalu menghasilkan
gambar dan label yang persis sama, sehingga eksperimen dan pengujian dapat
direproduksi.

Untuk meniru guncangan kamera atau produk yang diletakkan sedikit berbeda,
gunakan:

```python
X_shifted = shift_images(X, shift_y=2, shift_x=2, fill=0.15)
```

Pergeseran tidak melingkar; piksel yang kosong diisi dengan nilai latar. Inilah
uji pergeseran translasi yang dipakai di `sklearn_example.py`.

## 🚀 Mulai Cepat

Jalankan demo dari nol (beberapa detik di CPU):

```bash
python 04_advanced_topics/05_cnn_computer_vision/implementation.py
```

Jalankan perbandingan dengan scikit-learn:

```bash
python 04_advanced_topics/05_cnn_computer_vision/sklearn_example.py
```

Buka latihan:

```bash
jupyter notebook 04_advanced_topics/05_cnn_computer_vision/exercise.ipynb
```

Penggunaan minimal:

```python
from implementation import ShelfCNN, generate_shelf_images, shift_images

X_train, y_train = generate_shelf_images(n_samples=400, random_state=42)
X_test, y_test = generate_shelf_images(n_samples=160, random_state=43)

model = ShelfCNN(n_classes=4, image_size=16, random_state=42)
history = model.fit(
    X_train,
    y_train,
    epochs=12,
    batch_size=32,
    lr=0.02,
    momentum=0.9,
    verbose=0,
)

print("Loss:", history[0], "->", history[-1])
print("Akurasi uji:", model.score(X_test, y_test))
print("Akurasi bergeser:", model.score(shift_images(X_test, 2, 2), y_test))
```

Bekerja dengan lapisan individual:

```python
import numpy as np
from implementation import Conv2D, ReLU, MaxPool2D, Flatten, Dense
from implementation import SoftmaxCrossEntropy, SGD

layers = [
    Conv2D(1, 8, kernel=3, stride=1, padding=1, random_state=0),
    ReLU(),
    MaxPool2D(2, 2),
    Flatten(),
    Dense(8 * 8 * 8, 4, random_state=1),
]
logits = layers[0].forward(np.random.rand(2, 1, 16, 16))
# ... panggil lapisan berikutnya ...
```

File opsional `pytorch_example.py` menunjukkan ide yang sama di PyTorch/Keras.
PyTorch **tidak** terpasang di lingkungan ini dan rangkaian pengujian tidak
pernah mengimpor file tersebut.

## 📊 Hasil

Angka di bawah berasal dari `sklearn_example.py` dengan seed default: 400 gambar
latih, 160 gambar uji bersih, dan 160 gambar yang sama digeser 2 piksel ke bawah
dan ke kanan. Tingkat peluang acak adalah 25%.

| Model | Uji bersih | Uji bergeser | Penurunan | Waktu latih |
|-------|-----------:|-------------:|----------:|------------:|
| **CNN (NumPy, dari nol)** | **98,1%** | **88,8%** | **9,4%** | **~2 dtk** |
| MLP (scikit-learn, piksel diratakan) | 90,0% | 61,3% | 28,7% | 0,3 dtk |
| Regresi logistik (piksel diratakan) | 75,0% | 50,6% | 24,4% | 0,1 dtk |

Yang ditunjukkan tabel:

- Ketiga model mengalahkan tingkat peluang acak pada data uji bersih.
- CNN paling akurat dan paling sedikit menurun saat gambar digeser.
- MLP dense kehilangan hampir 30 poin persentase setelah pergeseran karena ia
  mempelajari bobot yang terikat pada koordinat piksel absolut.
- Regresi logistik berguna sebagai patokan pemeriksaan, tetapi batas linear
  pada ruang piksel mentah tidak cukup untuk memisahkan keempat bentuk secara
  andal.

Demo `__main__` memakai pembagian berbeda (360 gambar latih / 120 gambar uji)
dan mencapai akurasi uji 100% serta sekitar 91% pada data bergeser. Perbedaan
kecil wajar karena demo memakai seed dan jumlah epoch yang berbeda.

Mereproduksi tabel:

```bash
python 04_advanced_topics/05_cnn_computer_vision/sklearn_example.py
```

## 🔍 Apa yang Dipelajari Jaringan

Setelah pelatihan, periksa lapisan konvolusi pertama:

```python
first_conv = next(layer for layer in model.layers if isinstance(layer, Conv2D))
print(first_conv.params["W"].shape)   # (8, 1, 3, 3)
```

Delapan filter `3x3` biasanya berkembang menjadi pendeteksi tepi serta
pendeteksi garis dan gumpalan. Memvisualisasikannya sebagai peta panas kecil
menunjukkan arah yang ditanggapi setiap filter. Peta fitur setelah konvolusi
pertama menunjukkan di mana pola tersebut muncul pada potongan gambar tertentu.
Notebook latihan memplot keduanya.

Konvolusi kedua menggabungkan respons lapisan pertama menjadi bagian yang lebih
kompleks, seperti tutup botol atau bibir kaleng. Classifier dense kemudian
memetakan vektor fitur yang telah di-pool ke empat skor kelas.

## 🧪 Pengujian

Jalankan pengujian modul:

```bash
pytest tests/test_cnn_computer_vision.py -q
```

Rangkaian pengujian memeriksa:

- Bentuk `im2col` / `col2im` dan sifat adjoin
  `<im2col(x), r> = <x, col2im(r)>`.
- Keluaran konvolusi terhadap implementasi loop naif, termasuk stride dan
  padding.
- Pemeriksaan gradien numerik untuk `Conv2D` (masukan dan bobot), `MaxPool2D`,
  `Dense`, dan softmax cross-entropy, semuanya di bawah galat relatif `1e-5`
  pada float64.
- Bentuk keluaran lapisan dan API `ShelfCNN`.
- Bentuk data, rentang nilai, keseimbangan label, determinisme, dan pergeseran
  dari generator.
- Pelatihan ujung ke ujung: fungsi kerugian menurun, akurasi uji jauh di atas
  tingkat peluang acak, dan proses selesai di bawah 10 detik.
- API `SimpleCNN`, `generate_product_images`, dan `gradient_check_cnn` yang
  tetap kompatibel ke belakang.

Pengujian memuat `implementation.py` dengan
`importlib.util.spec_from_file_location`, sehingga tidak bentrok dengan banyak
modul lain yang juga menyertakan `implementation.py`.

## ✅ Keuntungan dan Kerugian

### Keuntungan

- **Efisien secara parameter:** filter yang sama dipakai ulang di setiap posisi.
- **Tanggap terhadap translasi:** pooling dan berbagi bobot membuat prediksi
  lebih tahan terhadap pergeseran kecil.
- **Pembelajaran fitur otomatis:** tidak perlu deskriptor tepi atau tekstur
  buatan tangan.
- **Representasi hierarkis:** lapisan yang lebih dalam menggabungkan bagian
  sederhana menjadi objek.
- **Prior kuat untuk gambar:** struktur lokal dan kehalusan spasial sudah
  tertanam dalam arsitektur.
- **Ramah perangkat keras:** konvolusi sebagian besar berupa perkalian matriks
  padat (terutama melalui `im2col`).

### Kerugian

- **Butuh banyak data:** CNN biasanya memerlukan banyak gambar berlabel agar
  generalisasinya baik.
- **Butuh komputasi besar:** melatih jaringan dalam jauh lebih mahal daripada
  regresi logistik atau MLP kecil.
- **Banyak pilihan desain:** jumlah lapisan, kanal, ukuran kernel, stride,
  pooling, normalisasi, dan learning rate saling memengaruhi.
- **Kurang dapat diinterpretasikan:** filter yang dipelajari dapat
  divisualisasikan, tetapi proses keputusan keseluruhan sulit dijelaskan.
- **Tidak universal:** data tabular, fitur renggang, dan dataset sangat kecil
  sering lebih cocok ditangani kumpulan pohon keputusan atau model linear.
- **Implementasi ini bersifat edukatif:** tidak dioptimalkan untuk kecepatan,
  presisi numerik, atau gambar berskala produksi.

## 🎯 Kapan Menggunakan CNN

Gunakan CNN ketika:

- Masukan memiliki struktur spasial atau grid (gambar, bingkai video,
  spektrogram).
- Pola lokal penting dan pola yang sama dapat muncul di mana saja.
- Anda memiliki cukup data berlabel, atau dapat memakai pembelajaran transfer.
- Ketahanan terhadap translasi penting.
- Anda mampu menyediakan komputasi pelatihan.

Pertimbangkan alternatif ketika:

- Data berbentuk tabular tanpa struktur spasial; gunakan gradient boosting,
  random forest, atau MLP kecil.
- Dataset sangat kecil; fitur klasik ditambah pengklasifikasi sederhana bisa
  memberikan generalisasi lebih baik.
- Anda membutuhkan model yang sangat mudah diinterpretasikan.
- Latensi dan memori sangat ketat; alur fitur buatan tangan bisa jadi masih
  lebih unggul.
- Tugasnya deteksi atau segmentasi objek; CNN klasifikasi hanyalah salah satu
  blok bangunan, bukan solusi lengkap.

## 📁 File dalam Direktori Ini

- **`README.md`** — teori dan penggunaan dalam bahasa Inggris.
- **`README.id.md`** (file ini) — materi yang sama dalam Bahasa Indonesia.
- **`implementation.py`** — lapisan, fungsi kerugian, pengoptimal, `ShelfCNN`,
  `SimpleCNN`, generator data, dan demo `__main__` yang hanya memakai NumPy.
- **`sklearn_example.py`** — perbandingan CNN vs MLP vs regresi logistik,
  termasuk uji pergeseran translasi (tanpa tampilan grafis, mencetak tabel).
- **`exercise.ipynb`** — notebook latihan: data, konvolusi, pelatihan, kurva
  kerugian, filter, matriks konfusi, pergeseran translasi, dan latihan IoU.
- **`pytorch_example.py`** — perbandingan opsional dengan PyTorch/Keras. File
  ini tidak diimpor oleh pengujian, dan PyTorch tidak terpasang di sini.

## 🔗 Dari CNN ke deteksi objek (YOLO)

Semua isi modul ini adalah **klasifikasi gambar**: satu label untuk satu
potongan gambar. Sistem audit rak yang nyata juga harus menjawab **di mana**
setiap produk berada. Itulah **deteksi objek**.

### Kotak pembatas

Detektor mengeluarkan kotak pembatas untuk setiap objek, biasanya dalam format:

```
(x1, y1, x2, y2)          format sudut
(x_center, y_center, w, h) format pusat
```

Kotak melokalisasi produk; label kelas menyatakan jenisnya; skor keyakinan
menyatakan seberapa yakin model tersebut.

### Sel grid

YOLO ("You Only Look Once") membagi gambar menjadi grid `S x S`. Setiap sel grid
bertanggung jawab atas objek yang pusatnya berada di dalam sel tersebut. Sebuah
sel memprediksi:

- `B` kotak pembatas, masing-masing dengan offset `(x, y, w, h)` dan skor
  keberadaan objek,
- `C` probabilitas kelas bersyarat.

Skor keberadaan objek dilatih untuk mencerminkan dua hal: "apakah ada objek?"
dan seberapa baik kotak prediksi tumpang tindih dengan acuan sebenarnya. Saat
inferensi, skor kelas sebuah kotak sering dihitung sebagai
`objectness * probabilitas_kelas`.

Karena satu perambatan maju memprediksi semua sel sekaligus, YOLO cukup cepat
untuk umpan kamera waktu nyata.

### Intersection over Union (IoU)

IoU mengukur seberapa besar dua kotak tumpang tindih:

```
intersection = luas daerah tumpang tindih
union        = luas(A) + luas(B) - intersection
IoU          = intersection / union
```

`IoU = 0` berarti tidak ada tumpang tindih, `IoU = 1` berarti kotaknya identik.
IoU dipakai sebagai metrik evaluasi sekaligus di dalam pelatihan dan
pascapemrosesan.

### Penekanan non-maksimum (NMS)

Detektor sering menghasilkan banyak kotak yang tumpang tindih untuk produk yang
sama. NMS membuang duplikatnya:

1. Urutkan semua kotak kandidat berdasarkan keyakinan, tertinggi lebih dulu.
2. Pertahankan kotak dengan skor tertinggi.
3. Buang setiap kotak tersisa yang IoU-nya dengan kotak yang dipertahankan
   melebihi ambang (misalnya `0.5`).
4. Ulangi dengan kotak berskor tertinggi berikutnya sampai tidak ada kandidat
   tersisa.

Hasilnya adalah daftar pendek deteksi yang yakin dan tidak saling tumpang
tindih.

Modul berikutnya akan membangun detektor bergaya YOLO kecil di atas primitif
konvolusi dan pooling dari modul ini. Lompatan dari klasifikasi ke deteksi
sebagian besar adalah menambahkan **bagian lokalisasi** (regresi kotak),
merancang **kotak anchor atau target grid**, serta mengimplementasikan
**IoU + NMS** untuk pascapemrosesan.

## 📚 Referensi

1. LeCun, Y., Bottou, L., Bengio, Y., dan Haffner, P. (1998).
   *Gradient-Based Learning Applied to Document Recognition.*
   Proceedings of the IEEE, 86(11), 2278–2324.
   <http://yann.lecun.com/exdb/publis/pdf/lecun-01a.pdf>

2. Krizhevsky, A., Sutskever, I., dan Hinton, G. E. (2012).
   *ImageNet Classification with Deep Convolutional Neural Networks.*
   Advances in Neural Information Processing Systems 25.
   <https://papers.nips.cc/paper/2012/hash/c399862d3b9d6b76c8436e924a68c45b-Abstract.html>

3. Redmon, J., Divvala, S., Girshick, R., dan Farhadi, A. (2016).
   *You Only Look Once: Unified, Real-Time Object Detection.*
   <https://arxiv.org/abs/1506.02640>

4. Catatan kuliah CS231n, *Convolutional Neural Networks for Visual
   Recognition.*
   <https://cs231n.github.io/convolutional-networks/>

---

**Langkah Selanjutnya**

- Baca `implementation.py` dan telusuri satu perambatan maju dan balik secara
  manual.
- Jalankan `sklearn_example.py` untuk melihat selisih akibat pergeseran
  translasi.
- Selesaikan `exercise.ipynb`, termasuk visualisasi filter dan helper IoU.
- Coba arsitektur yang lebih dalam atau jumlah kelas yang berbeda.
- Lanjutkan ke deteksi: kotak pembatas, IoU, dan NMS.
