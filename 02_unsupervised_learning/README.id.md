# Unsupervised Learning

**[🇬🇧 English](README.md) | 🇮🇩 Bahasa Indonesia**

🏠 Back to Main: `../README.md`

Unsupervised algorithms for discovering patterns without labels. Folder pattern:
- `README.md` — theory overview and when to use
- `implementation.py` — from-scratch (NumPy) implementation
- `sklearn_example.py` — scikit-learn usage examples
- `exercise.ipynb` — hands-on exercises

Modules (available/planned):
- **[`01_kmeans/`](01_kmeans/)** — K-Means Clustering (available)
- `02_hierarchical_clustering/` — Hierarchical Clustering (coming soon)
- `03_dbscan/` — DBSCAN (coming soon)
- **[`04_pca/`](04_pca/)** — Principal Component Analysis (available)
- `05_tsne/` — t-SNE (coming soon)
- `06_association_rules/` — Association Rules (coming soon)

## Shared Dataset

K-Means and PCA modules use a shared dataset: **SPG (Sales Promoter) retail outlet data** for Indonesian FMCG. The dataset contains 500 outlets in Denpasar/Bali with features like monthly sales, SKU count, visit frequency, out-of-stock rate, and average basket size.

**Story:** A supervisor wants to segment outlets to optimize SPG visit planning. K-Means identifies natural segments, PCA visualizes them in 2D.

If a link is not yet available, the content is under development.
