"""
t-SNE with scikit-learn
=======================

This script compares scikit-learn's t-SNE with PCA on a synthetic
outlet/customer-segment dataset and shows how perplexity changes the embedding.

It produces two figures next to this file:

- ``tsne_sklearn_vs_pca.png``: PCA and t-SNE side by side,
- ``tsne_perplexity_effect.png``: t-SNE with low, medium and high perplexity.

The script uses the non-interactive Agg backend and small samples so that it
runs quickly in CI or a headless environment.

Run from the repository root:

    python3 02_unsupervised_learning/05_tsne/sklearn_example.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

# Allow both "python3 path/to/sklearn_example.py" and execution from elsewhere.
try:
    from implementation import (
        SEGMENT_NAMES,
        generate_customer_segments,
        knn_preservation,
        trustworthiness,
    )
except ImportError:  # pragma: no cover - only relevant to unusual launch modes
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from implementation import (  # type: ignore
        SEGMENT_NAMES,
        generate_customer_segments,
        knn_preservation,
        trustworthiness,
    )


OUTPUT_DIR = Path(__file__).resolve().parent


def make_tsne(perplexity: float) -> TSNE:
    """Build a fast scikit-learn t-SNE model across sklearn versions."""
    kwargs = dict(
        n_components=2,
        perplexity=perplexity,
        init="pca",
        random_state=42,
        learning_rate="auto",
    )
    try:
        return TSNE(max_iter=500, **kwargs)
    except TypeError:  # older scikit-learn releases
        return TSNE(n_iter=500, **kwargs)  # type: ignore[call-arg]


def scatter(ax, embedding, labels, title):
    """Draw one two-dimensional embedding coloured by the known segment."""
    unique = np.unique(labels)
    cmap = plt.get_cmap("tab10")
    for index, label in enumerate(unique):
        mask = labels == label
        name = SEGMENT_NAMES[index] if index < len(SEGMENT_NAMES) else str(label)
        ax.scatter(
            embedding[mask, 0],
            embedding[mask, 1],
            s=22,
            alpha=0.85,
            color=cmap(index % 10),
            label=name,
            edgecolors="none",
        )
    ax.set_title(title)
    ax.set_xticks([])
    ax.set_yticks([])


def main() -> int:
    print("t-SNE with scikit-learn")
    print("=" * 72)

    X, labels = generate_customer_segments(
        n_samples=240,
        n_features=12,
        n_clusters=5,
        random_state=42,
    )
    print(f"Data shape      : {X.shape}")
    print(f"Known segments  : {len(np.unique(labels))}")

    # --- PCA baseline -------------------------------------------------------
    pca_embedding = PCA(n_components=2, random_state=42).fit_transform(X)

    # --- t-SNE at a medium perplexity --------------------------------------
    tsne_embedding = make_tsne(perplexity=30.0).fit_transform(X)

    fig, axes = plt.subplots(1, 2, figsize=(11, 5))
    scatter(
        axes[0],
        pca_embedding,
        labels,
        "PCA (linear projection)",
    )
    scatter(
        axes[1],
        tsne_embedding,
        labels,
        "t-SNE (perplexity = 30)",
    )
    handles, legend_labels = axes[0].get_legend_handles_labels()
    fig.legend(
        handles,
        legend_labels,
        loc="lower center",
        ncol=5,
        frameon=False,
        fontsize=8,
    )
    fig.suptitle("Customer segments: PCA vs t-SNE", fontsize=13)
    fig.tight_layout(rect=(0, 0.08, 1, 0.95))
    comparison_path = OUTPUT_DIR / "tsne_sklearn_vs_pca.png"
    fig.savefig(comparison_path, dpi=140)
    plt.close(fig)
    print(f"\nSaved: {comparison_path.name}")

    pca_trust = trustworthiness(X, pca_embedding, n_neighbors=10)
    tsne_trust = trustworthiness(X, tsne_embedding, n_neighbors=10)
    print("\nNeighborhood quality (10 nearest neighbors)")
    print(f"  PCA   trustworthiness: {pca_trust:.3f}")
    print(f"  t-SNE trustworthiness: {tsne_trust:.3f}")

    # --- Perplexity effect --------------------------------------------------
    perplexities = (5.0, 30.0, 80.0)
    fig, axes = plt.subplots(1, len(perplexities), figsize=(15, 4.6))
    print("\nPerplexity sweep")
    for ax, perplexity in zip(axes, perplexities):
        embedding = make_tsne(perplexity=perplexity).fit_transform(X)
        score = trustworthiness(X, embedding, n_neighbors=10)
        preservation = knn_preservation(X, embedding, n_neighbors=10)
        scatter(
            ax,
            embedding,
            labels,
            f"perplexity = {perplexity:.0f}",
        )
        print(
            f"  perplexity {perplexity:5.0f}: "
            f"trustworthiness={score:.3f}, "
            f"kNN preservation={preservation:.3f}"
        )

    handles, legend_labels = axes[0].get_legend_handles_labels()
    fig.legend(
        handles,
        legend_labels,
        loc="lower center",
        ncol=5,
        frameon=False,
        fontsize=8,
    )
    fig.suptitle(
        "Perplexity controls the local/global trade-off in t-SNE",
        fontsize=13,
    )
    fig.tight_layout(rect=(0, 0.10, 1, 0.94))
    sweep_path = OUTPUT_DIR / "tsne_perplexity_effect.png"
    fig.savefig(sweep_path, dpi=140)
    plt.close(fig)
    print(f"\nSaved: {sweep_path.name}")

    print("\nInterpretation")
    print("  - PCA preserves global variance but may collapse overlapping groups.")
    print("  - t-SNE preserves local neighborhoods and reveals tight clusters.")
    print("  - Low perplexity emphasizes very local structure; high perplexity")
    print("    keeps more global relationships but can blur small clusters.")
    print("  - Inter-cluster distances and cluster sizes in t-SNE are not")
    print("    meaningful; do not read them as real-world magnitudes.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
