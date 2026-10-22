# Deteksi Objek

**🇬🇧 [English](README.md) | 🇮🇩 Bahasa Indonesia**

🏠 Kembali ke Topik Lanjutan: `../README.md`

Sebuah klasifikator menjawab *apa isi gambar ini*. Detektor juga menjawab *di
mana lokasinya*. Pertanyaan tambahan itu mengubah banyak hal: satu gambar bisa
berisi beberapa objek, objek dapat saling menutupi, prediksi harus dicocokkan
dengan ground truth sebelum dinilai, dan jumlah kemungkinan kotak sangat besar.

Modul ini membangun inti detektor satu tahap dengan **NumPy saja**:

- konversi kotak antara format `xyxy`, `xywh`, dan `cxcywh`,
- IoU berpasangan yang tervektorisasi dan non-maximum suppression (NMS),
- dataset sintetis rak toko kecil berukuran 48x48 dengan tiga kelas produk,
- detektor mini bergaya YOLO-v1 (`S = 6`, satu kotak per sel) dengan
  konvolusi im2col, backpropagation manual, dan optimizer Adam,
- encoding target, decoding, NMS per kelas, dan mAP@0.5,
- baseline sliding window dengan scikit-learn sebagai pembanding.

Semua data bersifat sintetis dan ramah CPU. Tidak ada PyTorch, tidak ada bobot
pretrained, dan tidak ada gambar eksternal yang dipakai.

## 📚 Daftar Isi

- [Mengapa Deteksi Sulit](#-mengapa-deteksi-sulit)
- [Format Kotak](#-format-kotak)
- [Intersection over Union](#-intersection-over-union)
- [Non-Maximum Suppression](#-non-maximum-suppression)
- [Dataset Rak Toko Sintetis](#-dataset-rak-toko-sintetis)
- [YOLO dalam Satu Halaman](#-yolo-dalam-satu-halaman)
- [Encoding Target](#-encoding-target)
- [Loss YOLO](#-loss-yolo)
- [Arsitektur Mini YOLO](#-arsitektur-mini-yolo)
- [Decoding Prediksi](#-decoding-prediksi)
- [Mean Average Precision pada IoU 0,5](#-mean-average-precision-pada-iou-05)
- [Mulai Cepat](#-mulai-cepat)
- [Baseline Sliding Window vs Mini YOLO](#-baseline-sliding-window-vs-mini-yolo)
- [Latihan Notebook](#-latihan-notebook)
- [Pengujian](#-pengujian)
- [Dari mainan ke produksi: Ultralytics YOLO pada foto berlabel Anda sendiri](#-dari-mainan-ke-produksi-ultralytics-yolo-pada-foto-berlabel-anda-sendiri)
- [File di Direktori Ini](#-file-di-direktori-ini)
- [Kelebihan dan Keterbatasan](#-kelebihan-dan-keterbatasan)
- [Referensi](#-referensi)

## 🧠 Mengapa Deteksi Sulit

Pada masalah klasifikasi, label hanya satu nilai. Pada deteksi, label adalah
*himpunan* kotak dan kelas yang jumlahnya berubah-ubah antar gambar. Empat
masalah langsung muncul:

1. **Lokalisasi:** model harus menghasilkan koordinat kontinu, bukan hanya
   indeks kelas.
2. **Banyak objek:** label one-hot tidak lagi cukup; keluaran harus mewakili
   beberapa objek sekaligus.
3. **Pencocokan:** sebelum menghitung loss atau metrik, setiap prediksi harus
   dipasangkan ke kotak ground truth, biasanya memakai IoU.
4. **Duplikat:** detektor padat menghasilkan banyak kotak di sekitar objek yang
   sama; NMS diperlukan untuk membuang duplikat.

Detektor karena itu adalah gabungan dari head regresi, head klasifikasi, aturan
pencocokan, decoder, dan protokol evaluasi. Modul ini menjaga setiap bagian
tetap kecil agar mudah diperiksa.

## 📦 Format Kotak

Tiga format dipakai di seluruh repositori:

| Nama | Arti | Disimpan sebagai |
|------|------|------------------|
| `xyxy` | sudut kiri atas dan kanan bawah | `(x1, y1, x2, y2)` |
| `xywh` | sudut kiri atas dan ukuran | `(x, y, w, h)` |
| `cxcywh` | titik pusat dan ukuran | `(cx, cy, w, h)` |

Konversinya adalah aljabar murni:

```
w  = x2 - x1                 x1 = x
h  = y2 - y1                 y1 = y
cx = (x1 + x2) / 2           x2 = x + w
cy = (y1 + y2) / 2           y2 = y + h
```

Keenam fungsi publik menerima satu kotak berbentuk `(4,)` atau batch berbentuk
`(N, 4)`, dan semuanya tervektorisasi:

```python
from implementation import (
    xyxy_to_xywh, xywh_to_xyxy,
    xyxy_to_cxcywh, cxcywh_to_xyxy,
    xywh_to_cxcywh, cxcywh_to_xywh,
)
```

Fungsi-fungsi ini adalah invers eksak untuk kotak yang sah dengan lebar dan
tinggi positif. Modul tidak memotong kotak secara diam-diam saat konversi,
karena kotak yang terpotong bukan lagi round trip yang setia.

## 🎯 Intersection over Union

IoU mengukur seberapa besar dua kotak bertumpang tindih:

```
IoU(A, B) = luas(A ∩ B) / luas(A ∪ B)
```

dengan

```
luas(A ∪ B) = luas(A) + luas(B) - luas(A ∩ B)
```

Irisan dihitung dari koordinat sudut yang bertumpang tindih:

```
inter_x1 = max(x1_A, x1_B)      inter_y1 = max(y1_A, y1_B)
inter_x2 = min(x2_A, x2_B)      inter_y2 = min(y2_A, y2_B)
inter_w  = max(0, inter_x2 - inter_x1)
inter_h  = max(0, inter_y2 - inter_y1)
inter    = inter_w * inter_h
```

`iou_matrix(boxes_a, boxes_b)` menyiarkan array `(N, 4)` terhadap array
`(M, 4)` dan mengembalikan matriks `(N, M)`. Ada dua detail penting:

- kotak yang hanya bersentuhan di tepi memiliki irisan nol,
- jika gabungan luasnya nol (dua kotak degenerat), IoU didefinisikan sebagai `0`
  agar tidak terjadi pembagian dengan nol.

Fungsi yang sama dipakai oleh NMS, diagnostik target, dan mAP. Satu
implementasi mencegah perbedaan halus antara proses training dan evaluasi.

## ✂️ Non-Maximum Suppression

Detektor sering menghasilkan beberapa kotak di sekitar produk yang sama. NMS
greedy mengubah sekumpulan kotak berskor menjadi himpunan yang lebih kecil:

```
1. urutkan kotak berdasarkan skor menurun
2. ambil kotak berskor tertinggi dan pertahankan
3. buang setiap kotak tersisa yang IoU-nya dengan kotak terpilih > threshold
4. ulangi sampai tidak ada kotak tersisa
```

Keluarannya adalah array **indeks yang terurut berdasarkan skor**. Mengurutkan
kotak terlebih dahulu membuat hasil deterministik, dan mengembalikan indeks
memudahkan pemetaan label serta skor yang sesuai.

NMS per kelas adalah prosedur yang sama yang diterapkan terpisah untuk setiap
kelas:

```
untuk setiap kelas c:
    keep_c = NMS(boxes[labels == c], scores[labels == c])
gabungkan semua keep_c lalu urutkan berdasarkan skor
```

Dua kotak dengan label berbeda tidak pernah saling menekan, walaupun keduanya
bertumpang tindih sepenuhnya. Ini penting ketika sebuah botol berdiri di depan
karton: keduanya objek nyata, jadi kedua kotaknya harus bertahan.

`per_class_nms(boxes, scores, labels, iou_threshold=0.5)` mengimplementasikan
hal ini dan juga menerima alias `batched_nms`.

## 🏪 Dataset Rak Toko Sintetis

`generate_shelf_dataset(...)` mengembalikan:

```
images      : array float32 (N, C, 48, 48), nilai dalam [0, 1]
annotations : daftar (boxes, labels)
              boxes  : array float32 (K, 4) dalam koordinat piksel xyxy
              labels : array int64 (K,), nilai 0, 1, 2
```

Tiga kelas sengaja dibuat berbeda bentuk dan teksturnya:

| Kelas | Bentuk | Tekstur |
|-------|--------|---------|
| `box` | karton lebar | garis diagonal dan label terang di tengah |
| `bottle` | leher sempit dan tutup di atas badan lebar | bayangan samping dan pita label terang |
| `can` | silinder | tutup, alas, dan garis-garis horizontal |

Setiap gambar berisi satu sampai empat produk yang ditempatkan secara acak
dengan jarak aman. Kotak pembatasnya dijamin tidak bertumpang tindih, dan titik
pusat objek ditempatkan pada sel grid 6x6 yang berbeda. Batasan kedua membuat
target satu kotak per sel menjadi tidak ambigu; ini sifat dataset pengajaran
ini, bukan sifat YOLO secara umum.

Sifat lainnya:

- **Ber-seed:** `seed` yang sama menghasilkan gambar dan kotak yang persis sama.
- **Kanal:** `channels=1` menghasilkan grayscale, `channels=3` menghasilkan RGB.
- **Noise:** noise Gaussian ditambahkan agar tugasnya tidak terlalu bersih.
- **Konteks rak:** papan rak horizontal digambar pada latar belakang.
- **Hanya sintetis:** tidak ada foto pribadi, tidak ada gambar klien, tidak ada
  scraping.

```python
images, annotations = generate_shelf_dataset(n_samples=240, seed=42)
boxes, labels = annotations[0]
```

## 🧠 YOLO dalam Satu Halaman

YOLO ("You Only Look Once") memperlakukan deteksi sebagai satu masalah regresi.
Gambar dibagi menjadi grid `S x S`. Setiap sel memprediksi `B` kotak dan
probabilitas kelas. Pada modul ini `S = 6` dan `B = 1`.

Untuk setiap sel, jaringan mengeluarkan:

```
[objectness, x_offset, y_offset, sqrt_w, sqrt_h, logit kelas...]
```

- `objectness` adalah logit; `sigmoid(objectness)` adalah probabilitas bahwa sel
  berisi objek.
- `x_offset` dan `y_offset` adalah offset langsung dalam `[0, 1]` relatif
  terhadap sel. Keduanya **tidak** dilewatkan sigmoid pada implementasi ini.
- `sqrt_w` dan `sqrt_h` memprediksi akar kuadrat ukuran kotak yang
  dinormalisasi.
- Kanal kelas adalah logit mentah yang dilewatkan softmax.

Titik pusat dan ukuran sel di-decode sebagai:

```
cx = (col + x_offset) * cell_size
cy = (row + y_offset) * cell_size
w  = sqrt_w ** 2 * image_size
h  = sqrt_h ** 2 * image_size
x1 = cx - w / 2      y1 = cy - h / 2
x2 = cx + w / 2      y2 = cy + h / 2
```

Memprediksi `sqrt(w)` alih-alih `w` adalah trik YOLO asli: target regresi tetap
berada pada rentang numerik yang lebih kecil dan kotak besar tidak terlalu
mendominasi loss.

## 🎯 Encoding Target

`encode_targets` mengubah kotak satu gambar menjadi tensor `(S, S, 5 + C)`.

Untuk setiap kotak:

1. hitung pusat `(cx, cy)` dan ukuran `(w, h)`,
2. tentukan sel `col = floor(cx / cell_size)`, `row = floor(cy / cell_size)`,
3. tulis `objectness = 1`,
4. tulis `x_offset = cx / cell_size - col` dan `y_offset` dengan cara yang sama,
5. tulis `sqrt_w = sqrt(w / image_size)` dan `sqrt_h = sqrt(h / image_size)`,
6. tulis vektor one-hot kelas.

Jika dua pusat kotak jatuh pada sel yang sama, kotak yang lebih besar menang.
Dataset sintetis menghindari kasus ini, tetapi aturan tersebut menjaga encoder
tetap deterministik.

Fungsi kebalikannya, `decode_targets`, merekonstruksi kotak dari tensor target.
Fungsi ini berguna untuk memeriksa bahwa encoding tidak kehilangan informasi.

## 📉 Loss YOLO

Misalkan `M_ij` bernilai `1` ketika sel `(i, j)` berisi objek dan `0` jika tidak.
Misalkan `o_ij` adalah logit objectness, `(x, y, sqrt_w, sqrt_h)` adalah empat
prediksi koordinat, dan `c_ij` adalah logit kelas. Dengan ukuran batch `N`:

```
L = (1 / N) * jumlah atas gambar dan sel [
        M_ij       * BCE(sigmoid(o_ij), 1)
      + (1 - M_ij) * lambda_noobj * BCE(sigmoid(o_ij), 0)
      + M_ij * lambda_coord * [
            (x - x_hat)^2 + (y - y_hat)^2
          + (sqrt_w - sqrt_w_hat)^2
          + (sqrt_h - sqrt_h_hat)^2
        ]
      + M_ij * CE(softmax(c_ij), kelas_ij)
    ]
```

dengan nilai bawaan

```
lambda_coord = 5.0
lambda_noobj = 0.5
```

dan binary cross-entropy yang stabil secara numerik

```
BCE(z, t) = max(z, 0) - z * t + log(1 + exp(-|z|))
```

dengan `z` adalah logit dan `t` adalah target. Inilah loss yang persis
diimplementasikan di `yolo_loss_and_grad`:

- objectness memakai **sigmoid + BCE**, bukan squared error,
- error koordinat memakai squared error biasa pada offset dan akar kuadrat
  ukuran,
- cross-entropy kelas hanya diterapkan pada **sel objek**,
- total dibagi ukuran batch lalu dijumlahkan atas grid dan kanal.

`yolo_loss_and_grad(raw, targets)` mengembalikan loss skalar dan gradiennya
terhadap keluaran head mentah. Karena itu fungsi ini bisa diperiksa langsung
dengan beda hingga sentral, terlepas dari backbone konvolusinya.

Pembobotan ini penting. `lambda_coord = 5` membuat error lokalisasi mahal.
`lambda_noobj = 0.5` mencegah sel latar belakang yang sangat banyak mendominasi
training. Meskipun begitu, sel latar tetap mayoritas: dengan `S = 6` ada 36 sel
per gambar dan biasanya hanya satu sampai empat objek.

## 🧱 Arsitektur Mini YOLO

`MiniYOLO` memakai tumpukan konvolusi kecil:

```
input C x 48 x 48
  conv 3x3, 16 kanal + ReLU + maxpool 2x2   -> 24 x 24
  conv 3x3, 32 kanal + ReLU + maxpool 2x2   -> 12 x 12
  conv 3x3, 32 kanal + ReLU                 -> 12 x 12
  conv 3x3, 5 + C kanal, stride 2           -> 6 x 6
```

Konvolusi terakhir adalah head deteksi. Lapisan ini menghasilkan satu vektor
`(5 + C)` untuk setiap 36 sel; tidak diperlukan lapisan dense.

Detail implementasi:

- Konvolusi memakai **im2col**: setiap receptive field menjadi satu baris,
  sehingga forward pass adalah perkalian matriks.
- Max pooling berukuran 2x2 tanpa tumpang tindih dan menyimpan argmax untuk
  backward pass.
- ReLU, konvolusi, pooling, dan head semuanya memiliki backward pass manual.
- Optimizer-nya adalah **Adam** dengan momen pertama dan kedua per parameter.
- Parameter bawaan adalah `float32`; gunakan `dtype=np.float64` untuk pemeriksaan
  gradien.

Satu langkah training untuk satu batch gambar adalah:

```
logits = model.forward(images)
loss, dlogits = yolo_loss_and_grad(logits, targets)
grads = model.backward(dlogits)
adam.step(model.parameters(), grads)
```

## 🔍 Decoding Prediksi

`decode_single` mengubah satu keluaran mentah `(S, S, 5 + C)` menjadi kotak:

1. `objectness = sigmoid(raw[..., 0])`,
2. `class_probs = softmax(raw[..., 5:])`,
3. pilih kelas terbaik dan tetapkan
   `score = objectness * max(class_probs)`,
4. buang sel dengan `score < conf_threshold`,
5. decode pusat dan ukuran dalam piksel,
6. potong kotak agar berada di dalam gambar,
7. terapkan **NMS per kelas**.

Skor adalah hasil perkalian, bukan hanya probabilitas kelas. Ini adalah
confidence YOLO standar: sebuah kotak harus berisi objek sekaligus
mengklasifikasikannya dengan baik.

`decode_predictions` menerima satu head mentah atau batch dan selalu
mengembalikan daftar tuple `(boxes, scores, labels)`. Tuple tersebut diurutkan
berdasarkan skor.

## 📏 Mean Average Precision pada IoU 0,5

Average precision dihitung per kelas dari daftar deteksi yang sudah diurutkan.

Untuk setiap kelas:

1. gabungkan semua deteksi dari seluruh gambar,
2. urutkan berdasarkan skor menurun,
3. telusuri daftar dan cocokkan setiap deteksi secara greedy ke kotak ground
   truth yang belum terpakai dengan IoU tertinggi,
4. hitung true positive jika IoU terbaik minimal `0,5`; jika tidak, hitung
   false positive,
5. setiap kotak ground truth hanya boleh dicocokkan satu kali.

Dengan true positive kumulatif `TP(k)` dan false positive kumulatif `FP(k)`:

```
recall(k)    = TP(k) / num_gt
precision(k) = TP(k) / (TP(k) + FP(k))
```

PASCAL VOC memakai **all-point interpolation**. Selubung precision dibuat
monoton dari kanan ke kiri:

```
p_interp(k) = max_{k' >= k} precision(k')
```

dan AP adalah luas di bawah kurva yang dihasilkan:

```
AP = jumlah_k (recall(k) - recall(k-1)) * p_interp(k),   recall(-1) = 0
```

Akhirnya:

```
mAP@0.5 = (1 / C) * jumlah_c AP_c
```

Kelas tanpa ground truth dilaporkan sebagai `0.0` tetapi tidak disertakan dalam
rata-rata. Jika tidak ada kelas yang memiliki ground truth, hasilnya `0.0`.

Dua konsekuensi perlu diingat:

- false positive yang muncul **sebelum** true positive terakhir menurunkan AP,
  karena precision pada recall tinggi ikut turun;
- ground truth yang terlewat membatasi recall di bawah `1` sehingga AP juga
  turun.

## 🚀 Mulai Cepat

Dari direktori ini:

```bash
python implementation.py
python sklearn_example.py
```

Pipeline deteksi minimal:

```python
import numpy as np

from implementation import (
    MiniYOLO,
    generate_shelf_dataset,
    mean_average_precision,
)

images, annotations = generate_shelf_dataset(n_samples=240, seed=42)
train_images, train_annotations = images[:200], annotations[:200]
test_images, test_annotations = images[200:], annotations[200:]

model = MiniYOLO(seed=42)
history = model.fit(
    train_images,
    train_annotations,
    epochs=30,
    batch_size=16,
    lr=3e-3,
)
print("loss akhir:", round(history[-1], 3))

predictions = model.predict(test_images, conf_threshold=0.2)
print("mAP@0.5:", round(mean_average_precision(predictions, test_annotations), 3))
```

Blok bangunan yang sama bisa dipakai langsung:

```python
from implementation import iou_matrix, nms, per_class_nms

ious = iou_matrix(boxes_a, boxes_b)       # (N, M)
keep = nms(boxes, scores, 0.5)            # satu kelas
keep = per_class_nms(boxes, scores, labels, 0.5)
```

## 🤖 Baseline Sliding Window vs Mini YOLO

`sklearn_example.py` membandingkan dua detektor pada split sintetis yang sama
(200 gambar training, 40 gambar held-out):

1. **Sliding window + LogisticRegression.** Empat ukuran jendela persegi panjang
   digeser pada setiap gambar dengan stride 8. Setiap patch dijelaskan oleh
   thumbnail intensitas 8x8 ditambah histogram gradien mirip HOG (grid spasial
   2x2, masing-masing 9 bin orientasi). Regresi logistik multinomial dilatih
   dengan kelas background (0) dan tiga kelas produk (1, 2, 3). Saat inferensi,
   probabilitas non-background terbaik menjadi skor, lalu NMS per kelas membuang
   duplikat.
2. **Mini YOLO.** Detektor dari nol dilatih selama 30 epoch dengan Adam dan
   dievaluasi memakai decoder serta kode mAP yang sama.

### Hasil

Skrip dijalankan pada CPU dengan environment repositori. Nilai berikut adalah
keluaran aktual dari satu run; waktu inferensi dan training berubah tergantung
perangkat keras dan beban sistem:

| Model | mAP@0.5 | Inference (ms/gambar) | Train (detik) |
|---|---:|---:|---:|
| Sliding window + LogisticRegression | 0,372 | 8,7 | 4,6 |
| Mini YOLO (NumPy, dari nol) | 0,645 | 2,8 | 16,5 |

AP@0.5 per kelas:

| Kelas | Sliding window | Mini YOLO |
|---|---:|---:|
| box | 0,583 | 0,808 |
| bottle | 0,169 | 0,630 |
| can | 0,362 | 0,497 |

Mengapa ada selisih?

- Sliding window hanya bisa mengeluarkan kotak dari himpunan jendela tetap.
  Botol yang tinggi kurang tercakup oleh jendela persegi, sehingga AP-nya
  rendah.
- Mini YOLO meregresi pusat dan ukuran secara langsung, jadi kotaknya bisa lebih
  pas untuk setiap objek.
- Baseline mengevaluasi sekitar 85 patch per gambar dengan scikit-learn.
  Detektor hanya melakukan satu pass konvolusi kecil, sehingga lebih cepat saat
  inferensi.
- Baseline tetap berharga: sederhana, mudah diinterpretasikan, dan tidak
  memerlukan backpropagation khusus.

Angka ini berasal dari data sintetis dan tidak boleh dibaca sebagai benchmark
dunia nyata. Waktu training dan inferensi berubah tergantung perangkat keras dan
beban sistem.

## 📓 Latihan Notebook

`exercise.ipynb` berisi sembilan latihan beserta solusinya:

1. mengonversi kotak antara ketiga format,
2. menghitung IoU dan menjalankan NMS greedy secara manual,
3. memeriksa NMS per kelas,
4. memeriksa dataset sintetis dan reproduktibilitasnya,
5. melakukan encode dan decode target YOLO,
6. memeriksa `yolo_loss_and_grad` dengan beda hingga,
7. melatih detektor kecil selama 20 epoch,
8. men-decode prediksi dan mencocokkannya dengan ground truth,
9. menyapu confidence threshold dan mendiskusikan pemilihan model.

Notebook berjalan di CPU jauh di bawah dua menit.

## 🧪 Pengujian

`tests/test_object_detection.py` memuat implementasi dengan
`importlib.util.spec_from_file_location` memakai nama unik
`object_detection_implementation` dan mencakup:

- kasus IoU yang diketahui: identik, terpisah, setengah tumpang tindih (`1/3`),
  tepi bersentuhan, dan kotak berluas nol,
- urutan NMS, penekanan, dan perilaku per kelas,
- round trip konversi kotak untuk kotak tunggal dan batch,
- round trip encoding dan decoding target,
- pemeriksaan gradien numerik loss YOLO (error relatif di bawah `1e-5`) dan
  backbone konvolusi pada input kecil,
- decoding, perhitungan skor, threshold, dan NMS per kelas,
- mAP pada kasus buatan: prediksi sempurna, satu false positive, dan ground truth
  yang terlewat,
- training singkat yang loss-nya turun dan mAP-nya jelas di atas baseline
  trivial.

Jalankan dengan:

```bash
python -m pytest tests/test_object_detection.py -q
```

Seluruh suite selesai dalam beberapa detik.

## 🚀 Dari mainan ke produksi: Ultralytics YOLO pada foto berlabel Anda sendiri

Detektor pada modul ini sengaja dibuat sangat kecil. Saat beralih ke foto rak
yang nyata, gunakan implementasi yang terpelihara seperti
[Ultralytics YOLO](https://docs.ultralytics.com), bukan kode NumPy ini.

### 1. Beri label pada foto Anda sendiri

Gunakan [Label Studio](https://labelstud.io/) atau
[CVAT](https://www.cvat.ai/) lalu ekspor dalam **format YOLO**:

```
dataset/
  images/
    train/shelf_0001.jpg
    val/shelf_0002.jpg
  labels/
    train/shelf_0001.txt
    val/shelf_0002.txt
```

Setiap file label berisi satu baris per objek:

```
<class_id> <x_center> <y_center> <width> <height>
```

Keempat koordinat dinormalisasi ke `[0, 1]`. Jaga urutan daftar kelas tetap dan
tulis nama yang sama di `data.yaml`:

```yaml
path: /absolute/path/to/dataset
train: images/train
val: images/val

names:
  0: box
  1: bottle
  2: can
```

Saran pelabelan:

- gambar kotak yang rapat di sekitar produk yang terlihat, termasuk tutup botol,
- gunakan satu kelas per jenis produk dan tulis definisinya,
- sertakan produk yang terlihat sebagian jika manusia masih bisa menghitungnya,
- sertakan area rak kosong dan latar yang ramai,
- simpan validation set terpisah dari hari atau toko yang berbeda.

### 2. Latih model

```bash
yolo train model=yolo11n.pt data=data.yaml epochs=100 imgsz=640 batch=16
```

Mulai dari model pretrained kecil, lalu ukur. Untuk eksperimen pengajaran
berbasis CPU, kurangi `imgsz` dan `epochs`; untuk produksi, gunakan GPU. Selalu
evaluasi pada split yang tidak pernah dipakai untuk keputusan pelabelan atau
tuning hyperparameter.

### 3. Kepatuhan planogram dan penghitungan stok

Dua tugas retail yang umum merupakan perluasan alami dari deteksi:

- **Kepatuhan planogram:** setiap posisi rak memiliki produk dan jumlah facing
  yang diharapkan. Deteksi produk, petakan kotak ke area rak, lalu bandingkan
  tata letak yang teramati dengan planogram. Laporkan facing yang hilang, produk
  yang salah, dan barang yang salah tempat.
- **Penghitungan stok:** hitung deteksi per kelas dan per rak, lalu gabungkan
  dengan aturan replenishment. Detektor memberi hitungan, bukan kebenaran
  inventaris, jadi tambahkan ambang confidence dan langkah review manusia untuk
  barang yang rendah confidence atau tertutup.

Praktik yang berguna:

- kalibrasi confidence threshold pada validation set,
- laporkan precision dan recall per kelas, bukan hanya mAP,
- lacak performa per toko, kondisi cahaya, dan sudut kamera,
- versikan model, data, dan pedoman pelabelan secara bersamaan,
- pantau drift ketika kamera, produk, atau tata letak rak berubah.

### 4. Jaga privasi foto Anda sendiri

Foto rak dapat mengungkap tata letak toko, rencana promosi, dan aktivitas
pelanggan. Perlakukan sebagai data sensitif:

- simpan foto Anda sendiri secara privat di bucket dengan kontrol akses,
- jangan mengunggah gambar klien ke demo publik atau notebook pihak ketiga,
- hapus wajah, orang, dan plat nomor kecuali memang diperlukan,
- periksa kebijakan data dari layanan training yang dipakai,
- hapus salinan sementara dan hindari commit gambar ke git,
- **jangan pernah memakai data klien tanpa izin tertulis.**

Generator sintetis pada modul ini ada justru agar algoritme dapat dipelajari
tanpa memaparkan foto siapa pun.

## 📂 File di Direktori Ini

| File | Fungsi |
|------|--------|
| `implementation.py` | kotak, IoU, NMS, dataset, Mini YOLO, loss, decoding, dan mAP dengan NumPy |
| `sklearn_example.py` | baseline sliding window LogisticRegression vs Mini YOLO, dengan tabel hasil markdown |
| `exercise.ipynb` | latihan beserta solusi |
| `README.md` | panduan bahasa Inggris |
| `README.id.md` | panduan bahasa Indonesia ini |
| `../../tests/test_object_detection.py` | suite pytest cepat untuk modul |

## ⚖️ Kelebihan dan Keterbatasan

### Kelebihan

- **Transparan:** setiap konvolusi, target, dan gradien terlihat.
- **Mandiri:** detektor hanya membutuhkan NumPy.
- **Cukup cepat untuk eksperimen:** training kecil selesai dalam hitungan detik.
- **Evaluasi benar:** IoU, NMS, VOC AP, dan mAP mengikuti resep standar.
- **Dapat dibandingkan:** baseline scikit-learn memakai data dan metrik yang
  sama.

### Keterbatasan

- `S = 6` dan `B = 1` sangat kasar; dua objek dalam satu sel tidak bisa
  diprediksi keduanya.
- Dataset sintetis, resolusi rendah, dan bebas dari perspektif, motion blur, serta
  oklusi berat.
- Detektor tidak memakai anchor box, feature pyramid, batch normalization,
  multi-scale training, atau test-time augmentation.
- Logit kelas pada sel latar tidak diawasi; hanya sel objek yang menerima
  cross-entropy kelas.
- Jaringan kecil dan cepat overfit pada beberapa lusin gambar.
- mAP@0.5 hanyalah satu ringkasan angka; sistem retail nyata juga memerlukan
  recall per kelas, kalibrasi, dan alur review.

Keterbatasan ini disengaja. Tujuannya adalah membuat mekanismenya mudah dibaca
sebelum beralih ke pustaka produksi.

## 📚 Referensi

- Redmon, J., Divvala, S., Girshick, R., & Farhadi, A. (2016).
  *You Only Look Once: Unified, Real-Time Object Detection.*
  <https://arxiv.org/abs/1506.02640>
- Everingham, M., Van Gool, L., Williams, C. K. I., Winn, J., & Zisserman, A.
  (2010). *The PASCAL Visual Object Classes (VOC) Challenge.* International
  Journal of Computer Vision.
  <https://link.springer.com/article/10.1007/s11263-009-0275-4>
- Neubeck, A., & Van Gool, L. (2006). *Efficient Non-Maximum Suppression.*
  International Conference on Pattern Recognition.
  <https://ieeexplore.ieee.org/document/1699659>
- Dokumentasi Ultralytics. <https://docs.ultralytics.com>

**Selamat Belajar! 🎓** | *[🇬🇧 English version](README.md)*
