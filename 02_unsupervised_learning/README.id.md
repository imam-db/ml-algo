# Unsupervised Learning

**[🇬🇧 English](README.md) | 🇮🇩 Bahasa Indonesia**

🏠 Back to Main: [`../README.md`](../README.md)

Unsupervised algorithms for discovering patterns without labels. Folder pattern:
- `README.md` — theory overview and when to use
- `implementation.py` — from-scratch (NumPy) implementation
- `sklearn_example.py` — scikit-learn usage examples
- `exercise.ipynb` — hands-on exercises

Modules (available/planned):
- **[`01_kmeans/`](01_kmeans/)** — K-Means Clustering (tersedia)
- **[`02_dbscan/`](02_dbscan/)** — DBSCAN (tersedia)
- **[`03_hierarchical/`](03_hierarchical/)** — Hierarchical Clustering (tersedia)
- **[`04_pca/`](04_pca/)** — Principal Component Analysis (tersedia)
- **[`05_tsne/`](05_tsne/)** — t-SNE (tersedia)
- **[`06_association_rules/`](06_association_rules/)** — Aturan Asosiasi (Apriori & FP-Growth) (tersedia)

All six unsupervised modules are implemented; none are currently planned.

## Shared Dataset

[K-Means](01_kmeans/) and [PCA](04_pca/) modules use a shared dataset: **SPG (Sales Promoter) retail outlet data** for Indonesian FMCG. The dataset contains 500 outlets in Denpasar/Bali with features like monthly sales, SKU count, visit frequency, out-of-stock rate, and average basket size.

**Story:** A supervisor wants to segment outlets to optimize SPG visit planning. K-Means identifies natural segments, PCA visualizes them in 2D.
