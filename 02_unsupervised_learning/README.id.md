# Unsupervised Learning

**[🇬🇧 English](README.md) | 🇮🇩 Bahasa Indonesia**

🏠 Kembali ke Utama: [`../README.md`](../README.md)

Algoritma unsupervised untuk menemukan pola tanpa label. Pola folder:
- `README.md` — ringkasan teori dan kapan digunakan
- `implementation.py` — implementasi from-scratch (NumPy)
- `sklearn_example.py` — contoh penggunaan scikit-learn
- `exercise.ipynb` — latihan langsung

Modul (tersedia/direncanakan):
- **[`01_kmeans/`](01_kmeans/)** — K-Means Clustering (tersedia)
- **[`02_dbscan/`](02_dbscan/)** — DBSCAN (tersedia)
- **[`03_hierarchical/`](03_hierarchical/)** — Hierarchical Clustering (tersedia)
- **[`04_pca/`](04_pca/)** — Principal Component Analysis (tersedia)
- **[`05_tsne/`](05_tsne/)** — t-SNE (tersedia)
- **[`06_association_rules/`](06_association_rules/)** — Aturan Asosiasi (Apriori & FP-Growth) (tersedia)

Keenam modul unsupervised sudah diimplementasikan; tidak ada yang direncanakan saat ini.

## Dataset Bersama

Modul [K-Means](01_kmeans/) dan [PCA](04_pca/) menggunakan dataset bersama: **data outlet retail SPG (Sales Promoter)** untuk FMCG Indonesia. Dataset ini berisi 500 outlet di Denpasar/Bali dengan fitur seperti penjualan bulanan, jumlah SKU, frekuensi kunjungan, tingkat out-of-stock, dan rata-rata basket size.

**Cerita:** Seorang supervisor ingin menyegmentasikan outlet untuk mengoptimalkan perencanaan kunjungan SPG. K-Means mengidentifikasi segmen alami, PCA memvisualisasikannya dalam 2D.
