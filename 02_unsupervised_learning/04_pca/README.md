# PCA (Principal Component Analysis)

**🇬🇧 English | [🇮🇩 Bahasa Indonesia](README.id.md)**

## Overview

Principal Component Analysis (PCA) is an unsupervised learning technique for dimensionality reduction. PCA transforms data to a new coordinate system where axes (principal components) are ordered by the variance they explain.

## Mathematical Foundation

PCA uses Singular Value Decomposition (SVD):
1. Center the data (subtract mean)
2. Compute SVD: X = U Σ V^T
3. Principal components = rows of V^T

## Key Concepts

- **Variance maximization**: Find directions of maximum variance
- **Dimensionality reduction**: Reduce features while preserving information
- **Orthogonal components**: Principal components are uncorrelated

**Explained Variance Ratio:** Percentage of information retained by each component. Example: [0.45, 0.25, 0.15, 0.10, 0.05] means PC1 explains 45% variance, PC1+PC2 explain 70%.

## Best Practices

- ALWAYS perform feature scaling before PCA!
- Use cumulative variance ≥ 90-95% to choose n_components
- Interpret component loadings to understand feature contributions

## Implementation

```python
from implementation import PCA

pca = PCA(n_components=2)
X_transformed = pca.fit_transform(X)
explained_variance = pca.explained_variance_ratio_
```

## Case Study

Dataset of 500 retail outlets with 5 features (sales, SKU count, visit frequency, OOS rate, basket size). Use PCA to:
1. Reduce from 5D to 2D for visualization
2. Check what % variance is retained by 2 PCs
3. Identify outlet patterns in PC space
4. Understand which features contribute to each PC

**Exercise:** Determine how many components are needed to capture 90% variance, then interpret loadings (original feature contributions) in each PC.

## When to Use

- High-dimensional data visualization
- Feature extraction before modeling
- Noise reduction
- Data compression

## Related Files

- `implementation.py` - PCA from scratch
- `sklearn_example.py` - Comparison with sklearn
- `exercise.ipynb` - Interactive exercises

**Happy Learning! 🎓** | *[🇮🇩 Bahasa Indonesia](README.id.md)*
