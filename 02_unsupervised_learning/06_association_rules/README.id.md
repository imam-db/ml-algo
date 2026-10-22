# Penambangan Aturan Asosiasi (Apriori & FP-Growth)

**[🇬🇧 English](README.md) | 🇮🇩 Bahasa Indonesia**

**🏠 [Kembali ke Beranda](../../README.id.md)** | **📁 [Pembelajaran Unsupervised](../README.id.md)** | **⬅️ [Sebelumnya: PCA](../04_pca/README.id.md)** | **🧠 [Terkait: t-SNE](../05_tsne/)** *(direncanakan)*

## 🗺️ Akses Cepat

| Berkas | Deskripsi | Cara Menjalankan |
|--------|-----------|------------------|
| [`implementation.py`](./implementation.py) | Implementasi Apriori, FP-Growth, metrik, dan pembuat data warung dari nol | `python3 implementation.py` |
| [`sklearn_example.py`](./sklearn_example.py) | Membandingkan hasil kita dengan `mlxtend` (dependensi opsional) | `python3 sklearn_example.py` |
| [`exercise.ipynb`](./exercise.ipynb) | Latihan terpandu dengan TODO dan petunjuk | `jupyter lab exercise.ipynb` |

## 📚 Daftar Isi

- [Teori & Konsep](#-teori--konsep)
- [Dasar Matematika](#-dasar-matematika)
- [Algoritma Apriori](#-algoritma-apriori)
- [Algoritma FP-Growth](#-algoritma-fp-growth)
- [Apriori vs FP-Growth](#-apriori-vs-fp-growth)
- [Dataset Warung](#-dataset-warung)
- [Metrik dalam Praktik](#-metrik-dalam-praktik)
- [Detail Implementasi](#-detail-implementasi)
- [Kelebihan & Kekurangan](#-kelebihan--kekurangan)
- [Kapan Menggunakan](#-kapan-menggunakan)
- [Kesalahan Umum & Praktik Terbaik](#-kesalahan-umum--praktik-terbaik)
- [Berkas di Direktori Ini](#-berkas-di-direktori-ini)
- [Mulai Cepat](#-mulai-cepat)
- [Latihan](#-latihan)
- [Algoritma Terkait](#-algoritma-terkait)
- [Referensi](#-referensi)

## 🧠 Teori & Konsep

### Apa itu Penambangan Aturan Asosiasi?

Penambangan aturan asosiasi menemukan barang-barang yang sering muncul bersama
dalam data transaksi. Sebuah **transaksi** adalah satu keranjang, struk, sesi
klik, atau troli belanja. **Itemset** adalah himpunan barang apa pun. Tujuannya
adalah menemukan aturan berbentuk:

```
X -> Y
```

dengan `X` (**anteseden**) dan `Y` (**konsekuen**) adalah itemset tidak kosong
yang saling lepas. Contoh klasiknya:

```
{Indomie Goreng} -> {Telur Ayam}
```

Jika pelanggan membeli Indomie, seberapa besar kemungkinan mereka juga membeli
telur? Pertanyaan ini berguna untuk penempatan rak, bundling, penjualan silang,
perancangan promosi, dan sistem rekomendasi.

Berbeda dari pembelajaran terbimbing (supervised), penambangan aturan asosiasi
**tidak memiliki label target**. Metode ini adalah penemuan pola tanpa
supervisi: ia mencari keteraturan dalam kemunculan bersama, bukan memprediksi
keluaran yang sudah ditentukan.

### Transaksi, Itemset, dan Aturan

| Istilah | Arti | Contoh |
|---------|------|--------|
| Item | Satu produk atau kejadian | `Indomie Goreng` |
| Transaksi | Satu keranjang berisi barang | `{Indomie, Telur, Rokok}` |
| Itemset | Himpunan barang apa pun | `{Kopi Sachet, Gula Pasir}` |
| k-itemset | Itemset dengan `k` barang | `{Roti, Susu}` adalah 2-itemset |
| Aturan | Implikasi antar-itemset | `{Rokok} -> {Korek Api}` |
| Support | Seberapa sering itemset muncul | 20% keranjang berisi keduanya |
| Confidence | Seberapa sering `Y` muncul saat `X` muncul | 65% keranjang rokok berisi korek |

### Mengapa Support dan Confidence Tidak Cukup

Confidence saja bisa menyesatkan. Aturan `{Roti} -> {Susu}` bisa memiliki
confidence tinggi hanya karena **susu ada di hampir semua keranjang**. Roti
tidak menambah banyak informasi. **Lift** mengoreksi hal ini dengan
membandingkan kemunculan bersama yang diamati dengan yang diperkirakan jika
kedua barang independen:

```
lift = confidence / support(Y)
     = support(X ∪ Y) / (support(X) * support(Y))
```

- `lift > 1`: barang berasosiasi positif (dibeli bersama lebih sering daripada
  kebetulan).
- `lift = 1`: independen.
- `lift < 1`: berasosiasi negatif (substitusi atau saling eksklusif).

Dua metrik lain juga berguna:

- **Leverage** mengukur selisih absolut dari independensi:
  `support(X ∪ Y) - support(X) * support(Y)`.
- **Conviction** mengukur seberapa kuat aturan tersirat:
  `(1 - support(Y)) / (1 - confidence)`. Nilainya tak hingga untuk aturan
  sempurna.

### Itemset Frequent adalah Fondasinya

Alih-alih menguji setiap kemungkinan aturan secara langsung, penambangan aturan
asosiasi bekerja dalam dua tahap:

1. **Penambangan frequent itemset**: temukan semua itemset yang support-nya
   minimal `min_support`.
2. **Pembuatan aturan**: pecah setiap frequent itemset menjadi pasangan
   anteseden/konsekuen dan pertahankan aturan yang confidence-nya minimal
   `min_confidence`.

Desain dua tahap ini efisien karena jumlah frequent itemset biasanya jauh lebih
kecil daripada jumlah aturan yang mungkin.

## 📐 Dasar Matematika

Misalkan `D` adalah dataset berisi `N` transaksi, dan `X`, `Y` adalah itemset.

### Dukungan (Support)

```
support(X) = |{t in D : X ⊆ t}| / N
```

Support adalah pecahan antara 0 dan 1. Beberapa alat menerima jumlah absolut;
modul ini menggunakan ambang **relatif** di `(0, 1]`.

### Keyakinan (Confidence)

```
confidence(X -> Y) = support(X ∪ Y) / support(X)
```

Confidence tidak terdefinisi jika `support(X) = 0`. Sifatnya asimetris:
umumnya `confidence(X -> Y) != confidence(Y -> X)`.

### Lift

```
lift(X -> Y) = confidence(X -> Y) / support(Y)
             = support(X ∪ Y) / (support(X) * support(Y))
```

Lift simetris terhadap `X` dan `Y` meskipun confidence tidak.

### Leverage

```
leverage(X -> Y) = support(X ∪ Y) - support(X) * support(Y)
```

Leverage juga simetris. Nilai positif berarti kedua barang muncul bersama lebih
sering daripada yang diperkirakan independensi. Leverage sering lebih mudah
ditafsirkan daripada lift ketika salah satu barang sangat langka, karena tidak
dibesarkan oleh penyebut yang sangat kecil.

### Conviction

```
conviction(X -> Y) = (1 - support(Y)) / (1 - confidence(X -> Y))
```

- `conviction = 1`: aturan tidak memberi informasi melebihi laju dasar `Y`.
- `conviction > 1`: aturan informatif.
- `conviction = ∞`: confidence bernilai 1 (aturan sempurna), asalkan
  `support(Y) < 1`.
- Jika confidence dan `support(Y)` sama-sama 1, rasio menjadi `0/0`;
  implementasi ini mendefinisikan conviction sebagai `1.0` pada kasus khusus
  tersebut.

### Sifat Anti-Monoton Dukungan

Properti kunci di balik Apriori adalah:

> Jika sebuah itemset frequent, semua subset-nya juga frequent. Dengan kata
> lain, jika sebuah itemset infrequent, semua superset-nya juga infrequent.

Sketsa bukti: jika `X ⊆ Z`, maka setiap transaksi yang memuat `Z` juga memuat
`X`. Karena itu `support(X) >= support(Z)`. Jika `support(X) < min_support`,
maka `support(Z)` juga pasti di bawah ambang.

Sifat ini memungkinkan Apriori membuang kandidat dalam jumlah besar tanpa perlu
menghitungnya sama sekali.

## ⚙️ Algoritma Apriori

Apriori (Agrawal & Srikant, 1994) menambang frequent itemset **tingkat demi
tingkat**.

### Langkah 1: 1-itemset Frequent

Hitung setiap barang tunggal dan pertahankan yang support-nya minimal
`min_support`. Sebut himpunan ini `L1`.

### Langkah 2: Langkah Join

Untuk membangun kandidat berukuran `k`, gabungkan pasangan frequent
`(k-1)`-itemset yang memiliki `k-2` item pertama yang sama. Dengan item
terurut di dalam setiap itemset, jika

```
A = {a1, ..., a(k-2), a(k-1)}
B = {a1, ..., a(k-2), b(k-1)}
```

maka join menghasilkan:

```
C = {a1, ..., a(k-2), a(k-1), b(k-1)}
```

Prefiks yang sama menjaga jumlah pasangan kandidat tetap terkendali.

### Langkah 3: Pemangkasan Anti-Monoton

Sebelum menghitung kandidat, periksa setiap subset berukuran `(k-1)`. Jika ada
subset yang tidak ada di `L(k-1)`, kandidat tidak mungkin frequent dan langsung
dibuang. Inilah **pemeriksaan subset** yang menjadi ciri khas Apriori.

### Langkah 4: Hitung dan Saring

Pindai transaksi sekali dan hitung berapa banyak yang memuat setiap kandidat
yang bertahan. Pertahankan kandidat dengan support minimal `min_support`.
Ulangi sampai tidak ada frequent itemset baru atau `max_len` tercapai.

### Contoh Perhitungan

Perhatikan lima keranjang mainan:

```
1. {Bread, Milk}
2. {Bread, Diaper, Beer, Eggs}
3. {Milk, Diaper, Beer, Cola}
4. {Bread, Milk, Diaper, Beer}
5. {Bread, Milk, Diaper, Cola}
```

Pada `min_support = 0.2` (minimal satu keranjang):

| Itemset | Jumlah | Support |
|---------|--------|---------|
| `{Bread}` | 4 | 0.8 |
| `{Milk}` | 4 | 0.8 |
| `{Diaper}` | 4 | 0.8 |
| `{Beer}` | 3 | 0.6 |
| `{Eggs}` | 1 | 0.2 |
| `{Cola}` | 2 | 0.4 |
| `{Bread, Milk}` | 3 | 0.6 |
| `{Diaper, Beer}` | 3 | 0.6 |

Aturan `{Diaper} -> {Beer}` memiliki:

```
support    = 0.6
confidence = 0.6 / 0.8 = 0.75
lift       = 0.75 / 0.6 = 1.25
leverage   = 0.6 - 0.8 * 0.6 = 0.12
conviction = (1 - 0.6) / (1 - 0.75) = 1.6
```

Karena setiap keranjang yang berisi beer juga berisi diaper, aturan kebalikannya
`{Beer} -> {Diaper}` memiliki confidence `1.0` dan conviction tak hingga.

### Kompleksitas

Kasus terburuk bersifat eksponensial terhadap jumlah barang berbeda, karena ada
`2^m - 1` itemset yang mungkin. Dalam praktik, pemangkasan anti-monoton dan
`min_support` yang wajar menjaga ruang pencarian tetap kecil. Penghitungan
kandidat mendominasi waktu jalan: setiap tingkat memindai transaksi dan
memeriksa kandidat yang bertahan.

## 🌳 Algoritma FP-Growth

FP-Growth (Han, Pei & Yin, 2000) menghindari pembuatan kandidat sepenuhnya. Ia
memadatkan basis data menjadi pohon lalu menambang pohon itu secara rekursif.

### Langkah 1: Hitung Item dan Bangun Tabel Header

Pindai basis data sekali dan hitung frekuensi item. Buang item di bawah
`min_support`. Urutkan item yang tersisa berdasarkan frekuensi menurun dan
bangun header table dengan tautan ke setiap node untuk item tersebut.

### Langkah 2: Bangun FP-Tree

Untuk setiap transaksi:

1. pertahankan hanya item yang frequent,
2. urutkan berdasarkan urutan header table,
3. sisipkan ke pohon, berbagi prefiks dengan jalur yang sudah ada,
4. tambahkan hitungan pada setiap node di jalur tersebut.

Hasilnya adalah pohon prefiks yang padat. Transaksi yang berbagi prefiks dengan
transaksi yang sudah ada tidak perlu membuat cabang baru.

### Langkah 3: Tambang Basis Pola Bersyarat (Conditional Pattern Base)

Untuk setiap item di header table, telusuri tautan node-nya naik ke akar.
Setiap jalur adalah **conditional pattern base**: himpunan item yang muncul
bersama item sufiks, dengan bobot sebesar hitungan node.

### Langkah 4: Bangun FP-Tree Bersyarat dan Rekursi

Hitung item pada pattern base, buang yang di bawah `min_support`, lalu bangun
**conditional FP-tree** yang lebih kecil. Lakukan rekursi dengan item sufiks
ditambahkan ke prefiks saat ini. Setiap panggilan rekursif menghasilkan satu
frequent itemset.

Karena FP-Growth tidak pernah menyebutkan kandidat, ia sering lebih skalabel
daripada Apriori pada data padat. Konsekuensinya adalah implementasi yang lebih
rumit dan memori tambahan untuk pohon serta tautan header.

## ⚖️ Apriori vs FP-Growth

| Aspek | Apriori | FP-Growth |
|-------|---------|-----------|
| Ide utama | Pembuatan kandidat tingkat demi tingkat | FP-tree terpadatkan, tanpa kandidat |
| Pemindaian basis data | Satu per tingkat (beberapa kali) | Dua kali (bangun pohon, lalu tambang) |
| Pembuatan kandidat | Ya (join + prune) | Tidak |
| Memori | Himpunan kandidat bisa besar | Pohon + tautan header |
| Cocok untuk | Data renggang, itemset pendek, pengajaran | Data padat, banyak pola berulang |
| Kompleksitas implementasi | Sedang | Lebih tinggi |
| Hasil | Semua frequent itemset | Semua frequent itemset |

Kedua algoritma mengembalikan frequent itemset dan support yang sama persis
untuk `min_support` yang sama. Rangkaian tes memeriksa kesetaraan ini, termasuk
pada data warung sintetis.

## 🏪 Dataset Warung

`generate_warung_transactions(n_transactions=1000, random_state=42)` membuat
keranjang sintetis yang dapat direproduksi dari warung/minimarket Indonesia:

- **18 produk** seperti Indomie Goreng, telur, kopi sachet, gula pasir, rokok,
  korek api, roti tawar, susu UHT, air mineral, sabun, dan shampoo sachet.
- Setiap keranjang dimulai dengan 1–3 produk acak berbobot.
- Dengan probabilitas 0.70, satu **asosiasi yang ditanam** ditambahkan:

| Pasangan yang ditanam | Cerita |
|-----------------------|--------|
| `Indomie Goreng` + `Telur Ayam` | Makan mi goreng klasik |
| `Kopi Sachet` + `Gula Pasir` | Kopi manis di rumah |
| `Rokok` + `Korek Api` | Rokok butuh pemantik |
| `Roti Tawar` + `Susu UHT` | Kombinasi sarapan |

Pembuat data mempertahankan paling banyak satu pasangan yang ditanam per
keranjang agar kombinasi antar-pasangan tidak mendominasi aturan yang
ditambang. Hasil tipikal pada 800 keranjang dengan `min_support = 0.02`:

```
Apriori  : 137 frequent itemset
FP-Growth: 137 frequent itemset
Identik   : True

{Indomie Goreng} -> {Telur Ayam}  conf ≈ 0.62  lift ≈ 2.0
{Kopi Sachet}    -> {Gula Pasir}   conf ≈ 0.66  lift ≈ 2.5
{Rokok}          -> {Korek Api}    conf ≈ 0.55  lift ≈ 2.6
{Roti Tawar}     -> {Susu UHT}     conf ≈ 0.67  lift ≈ 2.4
```

Angka pastinya bergantung pada `n_transactions` dan `random_state`, tetapi
pasangan yang ditanam selalu memiliki lift jauh di atas 1.

## 📊 Metrik dalam Praktik

### Memilih `min_support`

`min_support` mengendalikan ukuran pencarian:

```
Terlalu rendah -> banyak itemset, waktu jalan lambat, aturan berisik
Terlalu tinggi -> sedikit itemset, asosiasi berguna terlewat
```

Alur kerja praktis:

1. Encode keranjang dan periksa frekuensi item.
2. Mulai dengan ambang support yang menjaga jumlah itemset di kisaran ratusan
   atau beberapa ribu.
3. Tambang itemset dengan Apriori atau FP-Growth.
4. Buat aturan dengan `min_confidence` yang bermakna.
5. Urutkan berdasarkan lift atau leverage, lalu periksa support agar aturan
   tidak hanya berdasar segelintir keranjang.
6. Validasi aturan terbaik pada periode hold-out atau dengan eksperimen bisnis.

### Membaca Tabel Aturan

| Metrik | Nilai tinggi berarti | Perhatikan |
|--------|----------------------|------------|
| Support | Pola tersebut umum | Support tinggi bisa trivial (susu di setiap keranjang) |
| Confidence | `Y` biasanya mengikuti `X` | Bisa tinggi hanya karena `Y` sangat populer |
| Lift | Asosiasi positif yang kuat | Barang sangat langka bisa menghasilkan lift besar yang tidak stabil |
| Leverage | Efek absolut yang besar | Lebih sulit dibandingkan antar-dataset dengan ukuran berbeda |
| Conviction | Implikasi yang kuat | Tak hingga untuk aturan sempurna; tidak selalu sebanding antar-aturan |

### Aturan Bukan Sebab-Akibat

`{Rokok} -> {Korek Api}` tidak berarti rokok *menyebabkan* pembelian korek api.
Artinya kedua barang muncul bersama. Gunakan pengetahuan domain, eksperimen,
dan validasi temporal sebelum bertindak berdasarkan aturan.

## 🛠️ Detail Implementasi

### API Publik

```python
from implementation import (
    encode_transactions, one_hot_encode,
    support, confidence, lift, leverage, conviction, rule_metrics,
    Apriori, apriori,
    FPGrowth, fpgrowth,
    generate_rules, association_rules,
    generate_warung_transactions,
)
```

### Menambang Itemset Frequent

```python
transactions = generate_warung_transactions(800, random_state=42)

model = Apriori(min_support=0.03, max_len=None).fit(transactions)
print(len(model.frequent_itemsets_))
print(model.to_dataframe().head())

# Bentuk fungsi: mengembalikan {frozenset(items): support}
frequent = apriori(transactions, min_support=0.03)

# FP-Growth mengembalikan pemetaan yang sama persis
frequent_fp = fpgrowth(transactions, min_support=0.03)
assert frequent == frequent_fp
```

### Membuat Aturan

```python
rules = generate_rules(
    model.frequent_itemsets_,
    min_confidence=0.50,
    min_lift=1.0,
)

print(rules[["antecedents", "consequents", "support",
             "confidence", "lift"]].head())
```

DataFrame aturan berisi:

```
antecedents, consequents,
antecedent_support, consequent_support, support,
confidence, lift, leverage, conviction
```

### Parameter

| Parameter | Arti | Nilai tipikal |
|-----------|------|---------------|
| `min_support` | Support relatif minimum | `0.01`–`0.10` |
| `min_confidence` | Confidence aturan minimum | `0.30`–`0.80` |
| `min_lift` | Lift minimum | `1.0`–`2.0` |
| `max_len` | Ukuran itemset maksimum | `2`–`4` agar mudah ditafsirkan |

### Kasus Khusus

- Daftar transaksi kosong mengembalikan tanpa itemset dan tanpa aturan.
- Keranjang kosong dipertahankan agar penyebut support tetap benar.
- Barang duplikat dalam satu keranjang dihapus oleh `encode_transactions`.
- `min_support` harus di `(0, 1]`; nilai `0` akan membuat setiap subset yang
  mungkin dianggap "frequent" dan sengaja ditolak.
- Confidence dengan anteseden bersupport nol memunculkan `ValueError` yang
  jelas.
- Aturan sempurna memiliki conviction tak hingga, kecuali konsekuennya juga
  bersupport 1.

## ✅ Kelebihan & Kekurangan

### ✅ Kelebihan

- Mudah ditafsirkan: aturan dinyatakan dalam produk nyata.
- Tidak memerlukan label; bekerja langsung pada log transaksi.
- Apriori sederhana untuk diimplementasikan dan dijelaskan.
- FP-Growth cepat pada data padat dan menghindari ledakan kandidat.
- Metrik seperti lift dan leverage memisahkan asosiasi sejati dari barang yang
  sekadar populer.
- Secara alami menghasilkan keluaran yang dapat ditindaklanjuti untuk ritel,
  analitik web, dan sistem rekomendasi.

### ❌ Kekurangan

- Penambangan frequent itemset bisa mahal: jumlah itemset tumbuh eksponensial
  terhadap jumlah barang.
- Memilih `min_support` dan `min_confidence` memerlukan eksperimen.
- Aturan untuk barang langka bisa memiliki lift besar yang tidak stabil atau
  menyesatkan.
- Algoritma klasik hanya memakai ada/tidak ada, bukan kuantitas atau harga.
- Aturan bersifat korelasional, bukan kausal.
- Urutan waktu diabaikan kecuali transaksi dimodelkan secara eksplisit menurut
  waktu.
- Katalog yang sangat besar mungkin memerlukan penambangan terbatas (misalnya
  hanya pasangan, atau hanya aturan yang memuat barang target).

## 🎯 Kapan Menggunakan

### ✅ Gunakan aturan asosiasi ketika

- Anda memiliki data transaksi: struk, keranjang, klik, atau log kejadian.
- Anda ingin pola "jika X maka Y" yang mudah dibaca manusia.
- Anda perlu mendukung keputusan penempatan rak, bundling, atau penjualan
  silang.
- Anda ingin mengeksplorasi kemunculan bersama sebelum membangun recommender.
- Katalognya sedang, atau Anda dapat membatasi `max_len` / support.

### ❌ Pertimbangkan alternatif ketika

- Anda perlu memprediksi target tertentu: gunakan klasifikasi atau regresi.
- Anda perlu peringkat yang dipersonalisasi: gunakan collaborative filtering
  atau sistem rekomendasi.
- Data bersifat padat dan kontinu, bukan transaksional: gunakan klasterisasi
  atau analisis korelasi.
- Anda membutuhkan efek kausal: gunakan eksperimen terkontrol atau inferensi
  kausal.
- Katalog barang sangat besar dan volume transaksi sangat tinggi:
  pertimbangkan pustaka frequent-pattern khusus, sampling, atau penambangan
  terbatas.

## 💡 Kesalahan Umum & Praktik Terbaik

1. **Jangan hanya memakai confidence.** Selalu periksa lift atau leverage.
2. **Periksa support.** Aturan dengan lift 20 yang hanya berdasar tiga keranjang
   tidak dapat ditindaklanjuti.
3. **Waspadai konsekuen populer.** Susu, air mineral, dan rokok bisa membuat
   confidence terlihat mengesankan.
4. **Gunakan `max_len` untuk keterbacaan.** Aturan pasangan biasanya lebih mudah
   ditindaklanjuti daripada aturan enam barang.
5. **Validasi di luar periode data.** Aturan yang ditambang dari satu bulan
   belum tentu berlaku di bulan berikutnya.
6. **Terjemahkan aturan menjadi tindakan.** "Letakkan korek api di dekat rokok"
   lebih berguna daripada tabel angka.
7. **Perhatikan batas ambang.** Support bersifat inklusif: tepat sama dengan
   `min_support` dihitung frequent.
8. **Ingat sifat anti-monoton.** Itulah alasan pemangkasan aman dan alasan
   ambang rendah bisa meledak.

## 📁 Berkas di Direktori Ini

- **`README.md`** — teori dan konsep dalam bahasa Inggris
- **`README.id.md`** (berkas ini) — dokumentasi lengkap dalam Bahasa Indonesia
- **`implementation.py`** — encoding transaksi, support/confidence/lift/
  leverage/conviction, Apriori, FP-Growth, pembuatan aturan, data warung
  sintetis, dan demo `__main__`
- **`sklearn_example.py`** — membandingkan hasil kita dengan `mlxtend`; jika
  `mlxtend` belum terpasang, skrip mencetak pesan lalu keluar dengan sukses
- **`exercise.ipynb`** — latihan langsung dengan TODO dan petunjuk

## 🚀 Mulai Cepat

Jalankan demo dari nol:

```bash
python3 02_unsupervised_learning/06_association_rules/implementation.py
```

Jalankan perbandingan `mlxtend` opsional:

```bash
python3 02_unsupervised_learning/06_association_rules/sklearn_example.py
```

Jalankan tes:

```bash
pytest tests/test_association_rules.py -v
```

Penggunaan minimal:

```python
from implementation import Apriori, generate_rules, generate_warung_transactions

transactions = generate_warung_transactions(800, random_state=42)
model = Apriori(min_support=0.03).fit(transactions)
rules = generate_rules(model.frequent_itemsets_, min_confidence=0.50)

print(rules.sort_values("lift", ascending=False).head(10))
```

## 🎓 Latihan

Buku catatan [`exercise.ipynb`](./exercise.ipynb) memandu Anda melalui:

1. membuat dan memeriksa keranjang warung,
2. meng-encode transaksi dan menghitung frekuensi barang,
3. menghitung lima metrik secara manual,
4. menambang itemset dengan Apriori,
5. memverifikasi bahwa FP-Growth memberi hasil sama,
6. membuat dan menyaring aturan,
7. menemukan kembali asosiasi yang ditanam,
8. mengukur pengaruh `min_support`,
9. mengamati pemangkasan anti-monoton secara langsung.

Setiap sel dapat dijalankan dari atas ke bawah; TODO dan petunjuk mendorong
Anda mengubah ambang dan mengamati hasilnya.

## 🔗 Algoritma Terkait

- **K-Means** ([01_kmeans](../01_kmeans/)): mengelompokkan transaksi atau
  pelanggan yang mirip; aturan asosiasi menjelaskan kemunculan bersama di dalam
  keranjang.
- **PCA** ([04_pca](../04_pca/)): mereduksi dimensi fitur; aturan asosiasi
  bekerja pada ada/tidak ada yang bersifat kategorikal.
- **DBSCAN** ([02_dbscan](../02_dbscan/)) dan **Klasterisasi Hierarkis**
  ([03_hierarchical](../03_hierarchical/)): metode pengelompokan tanpa
  supervisi alternatif.
- **Naive Bayes** ([06_naive_bayes](../../01_supervised_learning/06_naive_bayes/)):
  klasifikasi probabilistik; dapat dikombinasikan dengan fitur keranjang.
- **Sistem rekomendasi** ([06_playground](../../06_playground/)): biasanya
  memakai collaborative filtering, tetapi aturan asosiasi adalah baseline yang
  kuat dan mudah dijelaskan.

## 📚 Referensi

- Agrawal, R., & Srikant, R. (1994). *Fast Algorithms for Mining Association
  Rules*. Proceedings of VLDB.
- Han, J., Pei, J., & Yin, Y. (2000). *Mining Frequent Patterns without
  Candidate Generation*. SIGMOD.
- [Dokumentasi frequent pattern mining mlxtend](https://rasbt.github.io/mlxtend/user_guide/frequent_patterns/)
- [Association rule learning (Wikipedia)](https://en.wikipedia.org/wiki/Association_rule_learning)
- [Algoritma Apriori (Wikipedia)](https://en.wikipedia.org/wiki/Apriori_algorithm)
- [FP-Growth (Wikipedia)](https://en.wikipedia.org/wiki/FP-growth_algorithm)

---

**Langkah Selanjutnya**:
- Baca `implementation.py` untuk melihat langkah join, pemangkasan subset, dan
  rekursi FP-tree.
- Jalankan `sklearn_example.py` untuk membandingkan dengan `mlxtend` (pasang
  dulu dengan `pip install mlxtend`).
- Selesaikan `exercise.ipynb` dan coba analisis `max_len=2` untuk penempatan
  rak.
- Terapkan alur kerja yang sama pada data transaksi Anda sendiri.
