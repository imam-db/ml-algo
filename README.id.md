# Repositori Pembelajaran Algoritma Machine Learning

**[🇬🇧 English](README.md) | 🇮🇩 Bahasa Indonesia**

[![Python Version](https://img.shields.io/badge/python-3.11%2B-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Tests](https://github.com/imam-db/ml-algo/actions/workflows/tests.yml/badge.svg)](https://github.com/imam-db/ml-algo/actions/workflows/tests.yml)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
[![Contributions Welcome](https://img.shields.io/badge/contributions-welcome-brightgreen.svg?style=flat)](CONTRIBUTING.id.md)

Selamat datang di repositori pembelajaran algoritma machine learning! 🤖📚

**Sumber belajar machine learning yang lengkap, berisi implementasi dari nol, teori lengkap dalam Bahasa Indonesia, dan latihan interaktif.**

## 🗺️ Navigasi Cepat

| Kategori | Ramah Pemula | Menengah | Lanjutan |
|----------|--------------|----------|----------|
| **Supervised** | [Regresi Linear](./01_supervised_learning/01_linear_regression/) | [Random Forest](./01_supervised_learning/04_random_forest/) | [Jaringan Saraf Tiruan](./01_supervised_learning/08_neural_networks/) |
| **Unsupervised** | [K-Means](./02_unsupervised_learning/01_kmeans/) | [PCA](./02_unsupervised_learning/04_pca/) | [Klasterisasi Hierarkis](./02_unsupervised_learning/03_hierarchical/) |
| **Lanjutan** | [Evaluasi Model](./04_advanced_topics/03_model_evaluation/) | [Gradient Boosting](./01_supervised_learning/09_gradient_boosting/) | [XGBoost](./04_advanced_topics/04_xgboost/) |
| **🧠 Model Lainnya** | [Q-Learning](./03_reinforcement_learning/01_q_learning/) | [SVM](./01_supervised_learning/10_svm/) | [CNN & Computer Vision](./04_advanced_topics/05_cnn_computer_vision/) |
| **📈 Deret Waktu** | [Peramalan](./04_advanced_topics/06_time_series_forecasting/) | [Gradient Boosting](./01_supervised_learning/09_gradient_boosting/) | [XGBoost](./04_advanced_topics/04_xgboost/) |
| **🎮 Interaktif** | [Balapan Algoritma](./06_playground/README.id.md) | [Taman Bermain Parameter](./06_playground/README.id.md) | [Rekayasa Fitur](./06_playground/README.id.md) |

**🤔 Bingung memilih algoritma?** ➡️ [**Panduan Memilih Algoritma**](./ALGORITHM_COMPARISON.md)
**🎆 Ingin memantau kemajuan belajarmu?** ➡️ [**Pelacak Kemajuan**](./PROGRESS_TRACKER.md)  
**🎮 Ingin bereksperimen secara interaktif?** ➡️ [**Taman Bermain Algoritma**](./06_playground/README.id.md)
**🧾 Butuh referensi cepat?** ➡️ [**Kumpulan Lembar Contekan**](./08_cheat_sheets/README.id.md)

**Berkas Penting:**
- 📚 [GLOSSARY.md](./GLOSSARY.md) - Rujukan lengkap istilah ML
- 🔬 [ALGORITHM_COMPARISON.md](./ALGORITHM_COMPARISON.md) - Panduan memilih algoritma
- 🎯 [PROGRESS_TRACKER.md](./PROGRESS_TRACKER.md) - Kemajuan belajar dan pencapaian
- 📁 [Struktur Proyek](#-struktur-repositori) - Susunan folder secara rinci
- 🚀 [Mulai Cepat](#-mulai-cepat) - Siap jalan dalam 3 langkah

## 📋 Struktur Repositori

### 1. **Supervised Learning** ([`01_supervised_learning/`](./01_supervised_learning/))
- **[01_linear_regression](./01_supervised_learning/01_linear_regression/)** - Regresi Linear ✅
- **[02_logistic_regression](./01_supervised_learning/02_logistic_regression/)** - Regresi Logistik ✅
- **[03_decision_trees](./01_supervised_learning/03_decision_trees/)** - Pohon Keputusan ✅
- **[04_random_forest](./01_supervised_learning/04_random_forest/)** - Random Forest ✅
- *05_svm — dipindah ke [10_svm](./01_supervised_learning/10_svm/)*
- **[06_naive_bayes](./01_supervised_learning/06_naive_bayes/)** - Naive Bayes (Gaussian & Multinomial) ✅
- **[07_knn](./01_supervised_learning/07_knn/)** - K-Nearest Neighbors (Klasifikasi & Regresi) ✅
- **[08_neural_networks](./01_supervised_learning/08_neural_networks/)** - Jaringan Saraf Tiruan (Feedforward) ✅
- **[09_gradient_boosting](./01_supervised_learning/09_gradient_boosting/)** - Gradient Boosting (Regresi & Klasifikasi) ✅
- **[10_svm](./01_supervised_learning/10_svm/)** - Support Vector Machine (Pegasos Linear & SMO Kernel) ✅

### 2. **Unsupervised Learning** ([`02_unsupervised_learning/`](./02_unsupervised_learning/))
- **[01_kmeans](./02_unsupervised_learning/01_kmeans/)** - Klasterisasi K-Means ✅
- **[02_dbscan](./02_unsupervised_learning/02_dbscan/)** - DBSCAN (Klasterisasi Berbasis Kepadatan) ✅
- **[03_hierarchical](./02_unsupervised_learning/03_hierarchical/)** - Klasterisasi Hierarkis (Aglomeratif) ✅
- **[04_pca](./02_unsupervised_learning/04_pca/)** - Analisis Komponen Utama (PCA) ✅
- **[05_tsne](./02_unsupervised_learning/05_tsne/)** - t-SNE (t-Distributed Stochastic Neighbor Embedding) ✅
- **[06_association_rules](./02_unsupervised_learning/06_association_rules/)** - Aturan Asosiasi (Apriori & FP-Growth) ✅

### 3. **Reinforcement Learning** ([`03_reinforcement_learning/`](./03_reinforcement_learning/))
- **[01_q_learning](./03_reinforcement_learning/01_q_learning/)** - Q-Learning (TD Off-Policy) ✅
- **[02_sarsa](./03_reinforcement_learning/02_sarsa/)** - SARSA (TD On-Policy) ✅
- **🚧 03_deep_q_networks** - Deep Q-Network (direncanakan)

### 4. **Topik Lanjutan** ([`04_advanced_topics/`](./04_advanced_topics/))
- **🚧 01_ensemble_methods** - Metode Ensemble (direncanakan)
- **🚧 02_deep_learning** - Deep Learning (direncanakan)
- **[03_model_evaluation](./04_advanced_topics/03_model_evaluation/)** - Evaluasi Model (Metrik & Validasi Silang) ✅
- **[04_xgboost](./04_advanced_topics/04_xgboost/)** - XGBoost (eXtreme Gradient Boosting) ✅
- **[05_cnn_computer_vision](./04_advanced_topics/05_cnn_computer_vision/)** - CNN & Computer Vision (Konvolusi 2D, Pooling & Backprop) ✅
- **[06_time_series_forecasting](./04_advanced_topics/06_time_series_forecasting/)** - Peramalan Deret Waktu (Model Klasik, Fitur Lag & Backtesting) ✅

### 5. **🚧 Dataset** (`05_datasets/`, direncanakan)
Dataset bersama untuk belajar (saat ini setiap modul membuat data sintetisnya sendiri)

### 6. **🎮 Taman Bermain Algoritma** ([`06_playground/`](./06_playground/README.id.md))
Alat interaktif untuk bereksperimen dengan algoritma ML:
- **🏁 Balapan Algoritma** - Bandingkan beberapa algoritma berdampingan
- **🎛️ Taman Bermain Parameter** - Atur hyperparameter secara interaktif
- **📊 Visualisasi Model** - Lihat batas keputusan dan perilaku model
- **📈 Kurva Belajar** - Amati bagaimana algoritma belajar dari waktu ke waktu
- **🧪 Lab Rekayasa Fitur** - Bereksperimen dengan transformasi fitur
- **🎬 Animasi Algoritma** - Animasi visual langkah demi langkah
- **🎲 Pembuat Data** - Buat dataset sintetis untuk pengujian

### 7. **🚧 Utilitas** (`06_utils/`, direncanakan)
Fungsi utilitas dan kode pembantu bersama

### 8. **Lembar Contekan** ([`08_cheat_sheets/`](./08_cheat_sheets/README.id.md))
Referensi cepat untuk mempercepat belajar dan implementasi:
- 🐍 [Referensi Cepat Scikit-Learn](./08_cheat_sheets/02_python_sklearn/sklearn_quick_reference.md)
- 🐼 [Dasar-Dasar Pandas & NumPy](./08_cheat_sheets/02_python_sklearn/pandas_numpy_essentials.md)
- 🧭 [Diagram Alur Memilih Algoritma](./08_cheat_sheets/01_algorithms/algorithm_selection_flowchart.md)
- 🤖 [Referensi Cepat Algoritma](./08_cheat_sheets/01_algorithms/algorithms_quick_reference.md)
- 📊 [Metrik Klasifikasi](./08_cheat_sheets/04_model_evaluation/classification_metrics.md)
- 🧮 [Rumus Penting Matematika & Statistika](./08_cheat_sheets/03_math_statistics/essential_formulas.md)
- 🧹 [Alur Pra-pemrosesan Data](./08_cheat_sheets/05_data_preprocessing/preprocessing_pipeline.md)
- 🛠️ [Error Umum & Solusinya](./08_cheat_sheets/07_troubleshooting/common_errors_solutions.md)
- **🚧 Lembar contekan visualisasi** (direncanakan)

## 🚀 Mulai Cepat

1. **Pasang Dependensi:**
   ```bash
   # Dependensi inti + pengembangan
   uv sync --extra dev
   ```

   Untuk dependensi opsional notebook, pasang semua extra:
   ```bash
   uv sync --all-extras
   ```

2. **Aktifkan Lingkungan Virtual:**
   ```bash
   source .venv/bin/activate
   ```
   Atau lewati aktivasi dan jalankan perintah dengan `uv run` (misalnya, `uv run python`).

3. **Jalankan Jupyter Lab:**
   ```bash
   uv run jupyter lab
   ```

4. **🎮 Buka Taman Bermain Algoritma:**
   ```bash
   uv run python 06_playground/playground_launcher.py
   ```
   Atau coba alatnya satu per satu:
   ```bash
   # Perbandingan algoritma secara cepat
   uv run python 06_playground/algorithm_race.py --quick_test
   
   # Pengaturan parameter secara interaktif
   uv run python 06_playground/parameter_playground.py
   
   # Visualisasi batas keputusan
   uv run python 06_playground/model_visualizer.py --algorithm svm
   ```

### Mode Tanpa Tampilan (tanpa jendela GUI)
Gunakan flag berikut di lingkungan tanpa layar atau saat berjalan di CI:
- Visualisasi Model: `--no_show`
- Taman Bermain Parameter: `--no_show`
- Kurva Belajar: `--no_show`
- Animasi Algoritma: `--no_show`
- Pembuat Data: `--no_show`
- Tutorial XGBoost: `--no_show`
- Balapan Algoritma: `--no_plots`

## 📚 Cara Belajar

Setiap algoritma memiliki struktur folder yang sama:
- `README.md` / `README.id.md` - Penjelasan teori dan konsep (Inggris/Indonesia)
- `implementation.py` - Implementasi dari nol
- `sklearn_example.py` - Contoh penggunaan scikit-learn
- `exercise.ipynb` - Latihan praktik
- `datasets/` - Dataset khusus untuk algoritma tersebut

## 🎯 Jalur Belajar

### Pemula
1. Regresi Linear
2. Regresi Logistik
3. Pohon Keputusan
4. Klasterisasi K-Means

### Menengah
1. Random Forest
2. SVM
3. Naive Bayes
4. PCA

### Lanjutan
1. Jaringan Saraf Tiruan
2. Gradient Boosting
3. XGBoost
4. CNN & Computer Vision
5. Peramalan Deret Waktu
6. Reinforcement Learning (Q-Learning, SARSA)

## 📖 Sumber Belajar

- [Dokumentasi Scikit-learn](https://scikit-learn.org/)
- [Kursus Machine Learning - Andrew Ng](https://www.coursera.org/learn/machine-learning)
- [Hands-On Machine Learning](https://github.com/ageron/handson-ml2)

## 📊 Status Implementasi

| Kategori | Sudah Ada | Direncanakan | Kemajuan |
|----------|-----------|--------------|----------|
| Supervised Learning | 9 | 0 | 🟢 100% |
| Unsupervised Learning | 4 | 2 | 🟢 67% |
| Reinforcement Learning | 2 | 1 | 🟢 67% |
| Topik Lanjutan | 4 | 2 | 🟢 67% |

## 🧪 Pengujian

Repositori ini dilengkapi rangkaian pengujian yang lengkap:

```bash
# Jalankan semua pengujian
uv run pytest

# Jalankan satu berkas pengujian
uv run pytest tests/test_reinforcement_learning.py -v

# Jalankan dengan laporan cakupan
uv run pytest --cov
```

**Cakupan Pengujian:** 200+ pengujian, semuanya lulus ✅

## 🤝 Kontribusi

Kontribusi sangat diterima! Lihat [CONTRIBUTING.id.md](CONTRIBUTING.id.md) untuk panduan lengkapnya.

**Langkah Cepat untuk Kontributor:**

1. Fork repositori ini
2. Buat branch fitur (`git checkout -b feature/AmazingFeature`)
3. Commit perubahanmu (`git commit -m 'feat: Add AmazingFeature'`)
4. Push ke branch tersebut (`git push origin feature/AmazingFeature`)
5. Buka Pull Request

**Catatan:** Setiap modul baru wajib menyertakan dokumentasi bahasa Inggris (README.md) dan bahasa Indonesia (README.id.md).

## 📜 Lisensi

Proyek ini menggunakan Lisensi MIT. Lihat [LICENSE](LICENSE) untuk detailnya.

## 📧 Kontak & Dukungan

- **Masalah**: [GitHub Issues](https://github.com/imam-db/ml-algo/issues)
- **Diskusi**: [GitHub Discussions](https://github.com/imam-db/ml-algo/discussions)
- **Kontribusi**: [Panduan Kontribusi](CONTRIBUTING.id.md)
- **Kode Etik**: [Kode Etik](CODE_OF_CONDUCT.md)

## 🌟 Ucapan Terima Kasih

- Terima kasih untuk semua kontributor
- Terinspirasi dari dokumentasi [scikit-learn](https://scikit-learn.org/)
- Sumber belajar dari [Andrew Ng](https://www.coursera.org/learn/machine-learning) dan [Fast.ai](https://www.fast.ai/)

---

**Selamat Belajar! 🎉**

*Repositori ini dibuat dengan ❤️ untuk para pembelajar machine learning*
