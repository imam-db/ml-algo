# t-SNE (t-distributed Stochastic Neighbor Embedding)

**🇬🇧 English | [🇮🇩 Bahasa Indonesia](README.id.md)**

**🏠 [Back to Main](../../README.md)** | **📁 [Unsupervised Learning](../README.md)** | **⬅️ [Previous: PCA](../04_pca/README.md)** | **➡️ [Next: Association Rules](../06_association_rules/README.md)**

## 🗺️ Quick Access

| File | Description | Quick Run |
|------|-------------|-----------|
| [`implementation.py`](./implementation.py) | From-scratch NumPy t-SNE, affinity calibration, KL gradient, neighborhood metrics, and a customer-segment generator | `python3 implementation.py` |
| [`sklearn_example.py`](./sklearn_example.py) | scikit-learn t-SNE vs PCA plus a perplexity sweep; saves PNG figures with the Agg backend | `python3 sklearn_example.py` |
| [`exercise.ipynb`](./exercise.ipynb) | Guided exercises with TODOs and hints | `jupyter lab exercise.ipynb` |

## 📚 Table of Contents

- [Theory & Concepts](#-theory--concepts)
- [Mathematical Foundation](#-mathematical-foundation)
- [Perplexity and the Binary Search on Sigma](#-perplexity-and-the-binary-search-on-sigma)
- [Symmetric P and Student-t Q](#-symmetric-p-and-student-t-q)
- [KL Divergence and Its Gradient](#-kl-divergence-and-its-gradient)
- [Optimization: Early Exaggeration, Momentum, Gains](#-optimization-early-exaggeration-momentum-gains)
- [The Algorithm Step by Step](#-the-algorithm-step-by-step)
- [The Customer-Segment Dataset](#-the-customer-segment-dataset)
- [Implementation Details](#-implementation-details)
- [The scikit-learn Example](#-the-scikit-learn-example)
- [Advantages & Disadvantages](#-advantages--disadvantages)
- [When to Use](#-when-to-use)
- [Common Pitfalls & Best Practices](#-common-pitfalls--best-practices)
- [Files in This Directory](#-files-in-this-directory)
- [Quick Start](#-quick-start)
- [Exercises](#-exercises)
- [Related Algorithms](#-related-algorithms)
- [References](#-references)

## 🧠 Theory & Concepts

### What is t-SNE?

t-distributed Stochastic Neighbor Embedding (t-SNE) is a **nonlinear
dimensionality-reduction and visualization** method. It takes points in a
high-dimensional feature space and places them in a low-dimensional map — almost
always two dimensions — so that points that were close in the original space
stay close in the map.

PCA is a linear projection: it preserves directions of maximum variance. t-SNE
optimizes a different objective. It focuses on **local neighborhoods** and is
therefore much better at revealing groups, curved manifolds, and overlapping
clouds that a linear projection would merge.

The method is unsupervised: it needs only the feature matrix `X`, no labels. It
is used for:

- exploratory data analysis,
- checking whether clusters or subpopulations exist,
- inspecting embeddings from neural networks,
- finding labelling mistakes or outliers,
- communicating structure in high-dimensional data.

t-SNE is a **visualization tool**, not a clustering algorithm. It produces
coordinates, not cluster assignments. Running K-Means on a t-SNE embedding is
usually a mistake: the embedding deliberately distorts density and distances, so
clusters found there do not correspond to reliable groups in the original data.

### High-Dimensional Similarities: Gaussians

For every pair of input points `x_i` and `x_j`, t-SNE defines a conditional
probability that `i` would pick `j` as a neighbor:

```
p_{j|i} = exp(-||x_i - x_j||^2 / (2 * sigma_i^2))
          / sum_{k != i} exp(-||x_i - x_k||^2 / (2 * sigma_i^2))
```

Each point has its own bandwidth `sigma_i`. Dense regions need a small `sigma`;
sparse regions need a large one. If one global bandwidth were used, dense
clusters would collapse into a single neighbor and sparse clusters would become
an undifferentiated cloud.

The bandwidth is not chosen by hand for every point. Instead, every point is
calibrated to the same **perplexity**, a smooth measure of the effective number
of neighbors.

### Low-Dimensional Similarities: Student-t

In the two-dimensional map, t-SNE uses a heavy-tailed Student-t kernel with one
degree of freedom:

```
q_{ij} = (1 + ||y_i - y_j||^2)^-1
         / sum_{k != l} (1 + ||y_k - y_l||^2)^-1
```

The heavy tail solves the **crowding problem**. In two dimensions there is not
enough room to preserve all pairwise distances from a high-dimensional space.
With a Gaussian in the map, moderately distant points would be pushed far apart
and the optimization would be unstable. The Student-t tail allows points at
moderate distances to coexist without dominating the objective.

### The Objective: KL Divergence

t-SNE matches the two similarity distributions by minimizing the Kullback-Leibler
divergence:

```
KL(P || Q) = sum_{i != j} p_{ij} * log(p_{ij} / q_{ij})
```

The asymmetry matters. Large `p_{ij}` with small `q_{ij}` is heavily penalized:
**nearby points in the input must stay nearby in the map**. Small `p_{ij}` with
large `q_{ij}` is penalized much less, so t-SNE may place unrelated points
anywhere as long as the local neighborhoods are preserved. This is exactly why
global distances in a t-SNE plot are not trustworthy.

### What t-SNE Preserves — and What It Does Not

| Property | Preserved? | Comment |
|----------|------------|---------|
| Local neighborhoods | Yes, approximately | Controlled by perplexity and the KL objective |
| Global distances | No | Inter-cluster gaps are not calibrated |
| Cluster sizes | No | A dense cluster may look small, a sparse one may look large |
| Density | No | The Student-t kernel expands sparse regions |
| Axes and orientation | No | Arbitrary rotation, reflection, and scaling |
| Cluster identity | No | t-SNE does not output labels or assignments |

If you need a quantitative embedding for downstream models, prefer PCA or a
method with an explicit out-of-sample transform. If you need a fast nonlinear
visualization, UMAP is a popular alternative; it is mentioned here only as text
and is not implemented in this module.

## 📐 Mathematical Foundation

### Pairwise Distances

All similarities are based on squared Euclidean distances:

```
d_ij^2 = ||x_i - x_j||^2
```

Expanding the norm gives a vectorized computation:

```
d_ij^2 = ||x_i||^2 + ||x_j||^2 - 2 * x_i . x_j
```

This is much faster than a Python double loop. Floating-point cancellation can
produce tiny negative values, so the implementation clips distances at zero.

### Conditional Probabilities

Let `beta_i = 1 / (2 * sigma_i^2)`. Then:

```
p_{j|i} = exp(-beta_i * d_ij^2) / sum_{k != i} exp(-beta_i * d_ik^2)
```

The diagonal is set to zero: a point is not its own neighbor. Every row is a
probability distribution over the other `n - 1` points and sums to one.

### Entropy

The Shannon entropy of one conditional distribution is:

```
H(P_i) = -sum_{j != i} p_{j|i} * log(p_{j|i})
```

Entropy measures how spread out the neighbor distribution is. A tiny `sigma`
puts almost all mass on the nearest point and gives entropy near zero. A huge
`sigma` spreads mass over every point and gives entropy near `log(n - 1)`.

### Perplexity

Perplexity is defined as:

```
perplexity(P_i) = exp(H(P_i))
```

It has an intuitive interpretation: the **effective number of neighbors**.
`perplexity = 5` means each point behaves as if it has about five meaningful
neighbors; `perplexity = 30` means about thirty.

Typical values are `5` to `50`, with `30` as the common default. Small values
emphasize very local structure; large values incorporate more global context.
Perplexity must be smaller than the number of samples, and it should be smaller
than roughly `n / 3` for a stable estimate of entropy.

### Gradient of KL with Respect to Y

The exact gradient is:

```
dKL/dy_i = 4 * sum_j (p_ij - q_ij) * (y_i - y_j)
           / (1 + ||y_i - y_j||^2)
```

Every term has a clear meaning:

- `p_ij - q_ij` is the similarity mismatch. If the input similarity is larger
  than the map similarity, the pair is pulled together; if smaller, it is pushed
  apart.
- `(y_i - y_j)` gives the direction of the force.
- `1 / (1 + ||y_i - y_j||^2)` is the Student-t derivative. It weakens forces
  between points that are already far apart, preventing explosions.

In matrix form, with `W_ij = (p_ij - q_ij) / (1 + ||y_i - y_j||^2)`:

```
grad_i = 4 * (sum_j W_ij) * y_i - 4 * sum_j W_ij * y_j
```

The implementation uses this vectorized form. Because `P` and `Q` are symmetric
and both diagonals are zero, the gradient is well behaved and can be checked
against finite differences.

## 🎯 Perplexity and the Binary Search on Sigma

### Why a Search Is Needed

For point `i`, the conditional probabilities depend on `beta_i`. There is no
closed-form solution for `beta_i` such that `H(P_i) = log(perplexity)`, because
the distances are arbitrary. The entropy is, however, **monotonically
decreasing** in `beta`:

- increase `beta` -> narrower Gaussian -> fewer effective neighbors -> lower
  entropy,
- decrease `beta` -> wider Gaussian -> more effective neighbors -> higher
  entropy.

Monotonicity makes a binary search reliable.

### The Bisection Procedure

For every point independently:

1. Compute the squared distances to every other point.
2. Start with a bracket `beta_min = 1e-20`, `beta_max = 1e20` and an initial
   guess `beta = 1`.
3. Compute the conditional row and its entropy.
4. If the entropy is above the target, the Gaussian is too wide: move
   `beta_min` up and increase `beta`.
5. If the entropy is below the target, the Gaussian is too narrow: move
   `beta_max` down and decrease `beta`.
6. Stop when `|H - log(perplexity)| <= tol`, or after `max_iter` iterations.
7. Store `p_{j|i}` and `sigma_i = 1 / sqrt(beta_i)`.

The implementation combines multiplicative bracketing with bisection, which
converges quickly even when the useful precision spans many orders of magnitude.

### Numerical Stability

Two details keep the search stable:

- **Shift before exponentiating.** Subtract the smallest distance in the row
  before applying `exp(-beta * d)`. The constant cancels in the normalization,
  but it prevents overflow for large `beta`.
- **Zero the diagonal.** A point's self-distance is zero, so without masking it
  would receive the largest weight. The diagonal is removed before
  normalization.

### Reading the Result

`compute_pairwise_affinities` returns:

- `conditional_P` with shape `(n, n)`, where every row sums to one and the
  diagonal is zero,
- `sigmas` with shape `(n,)`, the bandwidth found for each point.

A useful diagnostic is `exp(H(P_i))`: it should equal the requested perplexity
for every row, up to the entropy tolerance.

## 🔁 Symmetric P and Student-t Q

### Symmetrizing the Conditional Probabilities

The conditional matrix is not symmetric: `p_{j|i}` and `p_{i|j}` can differ
because each point has its own bandwidth. t-SNE uses the joint distribution:

```
p_ij = (p_{j|i} + p_{i|j}) / (2 * n)
```

Properties:

- `p_ij = p_ji` (symmetric),
- `sum_{i != j} p_ij = 1`,
- the diagonal is zero.

The factor `1 / n` makes the total mass one, and the factor `1 / 2` averages the
two directions. Before normalization, tiny probabilities are floored at
`1e-12` so that `log(p_ij / q_ij)` never sees an exact zero.

### Student-t Similarities in the Map

For an embedding `Y`, compute squared distances `||y_i - y_j||^2` and define:

```
w_ij = 1 / (1 + ||y_i - y_j||^2)
q_ij = w_ij / sum_{k != l} w_kl
```

The diagonal is zero and `Q` sums to one. The kernel has heavier tails than a
Gaussian: even a moderately far pair retains non-negligible probability. That
gives the optimizer room to arrange far-apart groups without an enormous
repulsive force.

## 📉 KL Divergence and Its Gradient

### The Divergence

```
KL(P || Q) = sum_{i != j} p_ij * log(p_ij / q_ij)
```

Useful facts:

- `KL(P || P) = 0`,
- `KL(P || Q) >= 0`,
- it is not symmetric: `KL(P || Q) != KL(Q || P)` in general,
- terms with `p_ij = 0` contribute nothing,
- a positive `p_ij` with `q_ij = 0` gives infinite divergence (the
  implementation avoids this by flooring probabilities).

### The Gradient

```
dKL/dy_i = 4 * sum_j (p_ij - q_ij) * (y_i - y_j)
           / (1 + ||y_i - y_j||^2)
```

The factor `4` comes from differentiating the squared distance inside the
Student-t kernel. The finite-difference test in the test suite checks this
formula directly: perturbing one coordinate of `Y` and comparing
`(KL(Y + eps) - KL(Y - eps)) / (2 * eps)` against the analytic gradient gives
agreement to roughly `1e-9` relative error on small problems.

### Why the Gradient Works

Suppose `p_ij` is large but `q_ij` is small: the pair is close in the input but
far in the map. Then `p_ij - q_ij > 0`, and the force pulls `y_i` toward `y_j`.
Suppose `p_ij` is small but `q_ij` is large: the pair is far in the input but
close in the map. Then `p_ij - q_ij < 0`, and the force pushes them apart. The
denominator shrinks both forces as the pair separates.

## 🚀 Optimization: Early Exaggeration, Momentum, Gains

t-SNE minimizes a non-convex objective with gradient descent. Three practical
tricks make the optimization reliable.

### Early Exaggeration

For the first `early_exaggeration_iter` iterations, multiply the joint
probabilities:

```
P_exaggerated = early_exaggeration * P
```

The default multiplier is `12`. This strengthens attraction and encourages the
map to form tight, well-separated clusters early. After the exaggeration phase,
the optimizer uses the true `P` and refines the layout. Without early
exaggeration, clusters can remain merged or drift apart too slowly.

### Momentum

Vanilla gradient descent zig-zags in narrow valleys. Momentum accumulates a
velocity:

```
update_t = momentum * update_{t-1} - learning_rate * gains * gradient_t
Y_t = Y_{t-1} + update_t
```

Typical values are `0.5` during early exaggeration and `0.8` afterwards. The
initial lower momentum avoids overshooting while the exaggerated forces are
large; the later higher momentum accelerates convergence.

### Adaptive Gains

Each coordinate has its own gain factor, starting at `1`:

- if the gradient keeps the same sign as the previous update, the gain grows by
  `0.2`,
- if the gradient changes sign, the gain shrinks by multiplying by `0.8`,
- gains are clipped below by `min_gain` (default `0.01`).

This is a diagonal, per-coordinate step-size adaptation. It lets coordinates
that consistently move in one direction take larger steps while damping
coordinates that oscillate.

### Recentering

After every update the embedding is translated so that its mean is zero. Only
relative positions matter, so removing translation drift does not change the
objective but keeps the numbers well conditioned.

### Learning Rate

The default is `200`. With `learning_rate="auto"`, the implementation uses:

```
learning_rate = max(n_samples / early_exaggeration / 4, 50)
```

Too large a learning rate produces a collapsed or exploding map; too small a
rate leaves clusters merged and takes many iterations to converge.

### Early Stopping?

The reference implementation runs a fixed number of iterations and records the
KL divergence at every step. In practice you can inspect `kl_history_` and stop
when it plateaus. The history is **not** guaranteed to decrease at every single
iteration: momentum and early exaggeration can cause small temporary increases.
What matters is the overall downward trend and the quality of the final
neighborhood preservation.

## ⚙️ The Algorithm Step by Step

1. **Validate and standardize the input.** Features with wildly different scales
   dominate Euclidean distances, so standardize them first.
2. **Compute pairwise squared distances** in the input space.
3. **Calibrate one Gaussian per point** with a binary search on `beta` until its
   entropy equals `log(perplexity)`. Store the conditional probabilities and
   `sigma_i`.
4. **Symmetrize and normalize** the conditional probabilities into `P`.
5. **Initialize `Y`.** PCA initialization (the default) places the map along the
   leading variance directions and is usually more stable than random
   initialization. The embedding is scaled down so the initial similarities are
   nearly uniform.
6. **Run gradient descent.** For each iteration:
   - apply early exaggeration when appropriate,
   - compute `Q` with the Student-t kernel,
   - compute the KL divergence and its gradient,
   - update gains, velocity, and coordinates,
   - recenter `Y`,
   - record the raw KL divergence.
7. **Return the embedding.** The final `P`, `Q`, `sigmas`, and `kl_history_`
   remain available on the fitted object.

### Complexity

With exact pairwise affinities, one iteration costs `O(n^2 * d)` time and
`O(n^2)` memory. The calibration phase also costs `O(n^2)` distance computations
plus a small number of bisection iterations per point. For large `n`, use an
approximate method such as Barnes-Hut t-SNE or a library implementation; this
module deliberately implements the exact algorithm for clarity.

## 🛍️ The Customer-Segment Dataset

`generate_customer_segments(n_samples=300, n_features=12, n_clusters=5,
cluster_std=0.7, separation=3.0, random_state=42)` creates a reproducible
synthetic outlet/customer dataset with known segments:

- **12 behavioural features** in the intended 10-20 dimensional range:

  | Feature | Meaning |
  |---------|---------|
  | `monthly_spend` | Average monthly spending |
  | `visit_frequency` | Visits per month |
  | `avg_basket_size` | Items or value per basket |
  | `days_since_last_visit` | Recency |
  | `product_diversity` | Number of distinct product categories |
  | `promo_sensitivity` | Response to promotions |
  | `weekend_share` | Share of weekend visits |
  | `online_share` | Share of online orders |
  | `loyalty_score` | Loyalty-program engagement |
  | `basket_value_std` | Variability of basket value |
  | `fresh_product_share` | Share of fresh products |
  | `snack_share` | Share of snacks and drinks |

- **Five latent segments**: budget families, convenience shoppers, premium
  loyalists, promo hunters, and weekend bulk buyers.
- **Balanced, shuffled labels** so the segment order does not leak into the
  feature matrix.
- **Clearly separated centers** with Gaussian noise, which makes t-SNE, PCA,
  and K-Means easy to compare.

The function returns `(X, labels)`. `labels` are known only because the data is
synthetic; the t-SNE model never sees them. Use them to colour plots and to
sanity-check neighborhood preservation, not as model input.

A typical run on 240 samples with perplexity `30` yields trustworthiness around
`0.97-0.99` and clearly separated islands. Exact numbers depend on the seed and
the number of iterations.

## 🛠️ Implementation Details

### Public API

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

### Minimal Example

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
print(model.kl_history_[0])   # initial KL
print(model.kl_history_[-1])  # final KL
print(trustworthiness(X, embedding, n_neighbors=10))
```

### Affinity Calibration

```python
conditional_P, sigmas = compute_pairwise_affinities(
    X, perplexity=30.0, tol=1e-5
)
P = joint_probabilities(conditional_P)

print(conditional_P.shape)  # (n, n)
print(P.sum())              # 1.0
print(sigmas.min(), sigmas.max())
```

### Lower-Level Objective

```python
import numpy as np

rng = np.random.default_rng(0)
Y = rng.normal(size=(X.shape[0], 2))
Q = q_distribution(Y)
loss = kl_divergence(P, Q)
gradient = kl_gradient(P, Q, Y)
```

### Parameters

| Parameter | Meaning | Typical values |
|-----------|---------|----------------|
| `perplexity` | Effective neighbors per point | `5`-`50`, default `30` |
| `learning_rate` | Gradient step size, or `"auto"` | `50`-`1000`, default `200` |
| `n_iter` | Number of optimization iterations | `500`-`2000` |
| `max_iter` | Optional alias that overrides `n_iter` | integer or `None` |
| `early_exaggeration` | Attraction multiplier at the start | `4`-`20`, default `12` |
| `early_exaggeration_iter` | Length of the exaggeration phase | `100`-`300` |
| `momentum` | Momentum before exaggeration ends | `0.5` |
| `final_momentum` | Momentum afterwards | `0.8` |
| `min_gain` | Gain floor | `0.01` |
| `init` | Initialization strategy | `"pca"` or `"random"` |
| `random_state` | Seed for reproducibility | integer or `None` |
| `tol` | Entropy tolerance | `1e-5` |

### Fitted Attributes

| Attribute | Shape | Meaning |
|-----------|-------|---------|
| `embedding_` | `(n, 2)` | Learned low-dimensional coordinates |
| `P_` | `(n, n)` | Symmetric joint input affinities |
| `Q_` | `(n, n)` | Final Student-t similarities |
| `sigmas_` | `(n,)` | Per-point Gaussian bandwidths |
| `kl_history_` | list | Raw KL divergence once per iteration; the last entry belongs to the returned embedding |
| `n_iter_` | int | Iterations actually run |

### Neighborhood Helpers

```python
trust = trustworthiness(X, embedding, n_neighbors=10)
preserved = knn_preservation(X, embedding, n_neighbors=10)
```

- **Trustworthiness** penalizes points that are neighbors in the map but were
  far away in the input space. It is `1` for a perfect embedding and decreases
  as false neighborhoods appear.
- **kNN preservation** is the average overlap between the `k` nearest neighbors
  in the input and in the embedding. It is easy to interpret but stricter than
  trustworthiness.

Both helpers are useful sanity checks, not substitutes for looking at the plot
and the original data.

### Edge Cases and Validation

- `perplexity` must be greater than `1` and smaller than `n_samples`.
- `n_iter` must be positive; if the early-exaggeration phase is not shorter than
  `n_iter`, it is reduced to a quarter of the run so a refinement phase remains.
- `init` must be `"pca"` or `"random"`.
- Non-finite features raise a clear `ValueError`.
- `joint_probabilities` rejects negative or non-square matrices.
- The gradient check is easiest on small `n` because the exact algorithm is
  `O(n^2)`.

## 📊 The scikit-learn Example

`sklearn_example.py` uses the installed scikit-learn implementation for a
practical comparison:

1. generate 240 outlet/customer rows with five known segments,
2. project them to two dimensions with PCA,
3. embed them with scikit-learn t-SNE at perplexity `30`,
4. save `tsne_sklearn_vs_pca.png`,
5. sweep perplexities `5`, `30`, and `80`, print trustworthiness and kNN
   preservation, and save `tsne_perplexity_effect.png`.

The script calls `matplotlib.use("Agg")` before importing `pyplot`, so it works
on a headless server or in CI. It uses `max_iter=500` and a small sample to stay
fast.

Typical output:

```
PCA   trustworthiness: 0.955
t-SNE trustworthiness: 0.978

perplexity     5: trustworthiness=0.967, kNN preservation=0.498
perplexity    30: trustworthiness=0.978, kNN preservation=0.568
perplexity    80: trustworthiness=0.967, kNN preservation=0.445
```

The exact numbers vary slightly with library versions. The qualitative lesson is
stable: t-SNE preserves local neighborhoods better than PCA, and perplexity
controls the local/global trade-off.

## ✅ Advantages & Disadvantages

### ✅ Advantages

- Reveals nonlinear local structure that PCA can miss.
- Produces compact, visually interpretable clusters.
- Handles curved manifolds and overlapping clouds.
- No labels required; works directly on a feature matrix.
- The objective and gradient are mathematically clean and easy to implement.
- Perplexity is a more intuitive knob than a raw bandwidth.
- The heavy-tailed kernel reduces the crowding problem in low dimensions.
- Excellent for exploration, debugging, and communicating structure.

### ❌ Disadvantages

- **Cluster sizes and inter-cluster distances are not meaningful.**
- Global geometry is not preserved; the map can rotate, reflect, and distort.
- It does not produce cluster labels, centroids, or a reusable transform for new
  points.
- Exact t-SNE is `O(n^2)` per iteration, which is expensive for large datasets.
- The objective is non-convex; different seeds can produce different layouts.
- Results depend on perplexity, learning rate, initialization, and iteration
  count.
- No natural out-of-sample extension: adding a point usually requires rerunning
  the whole embedding.
- High-dimensional distance concentration can make neighborhoods noisy.
- Two-dimensional plots can hide structure that needs three dimensions.

## 🎯 When to Use

### ✅ Use t-SNE when

- You want to **explore** high-dimensional data visually.
- You suspect local clusters or subpopulations and want to see them.
- You need a nonlinear alternative to PCA for a two-dimensional plot.
- You are inspecting learned representations such as image or text embeddings.
- You can afford the exact algorithm or have access to an approximate
  implementation.
- You will validate findings on the original features and with quantitative
  neighborhood metrics.

### ❌ Consider alternatives when

- You need a linear, interpretable projection with explained variance: use PCA.
- You need a fast nonlinear visualization for very large data: use an
  approximate t-SNE implementation or UMAP.
- You need cluster assignments: use K-Means, DBSCAN, hierarchical clustering, or
  a mixture model on the original features.
- You need to embed new points without refitting: use PCA, a parametric
  autoencoder, or a method with an explicit transform.
- You need to preserve global distances: use PCA, classical MDS, or a method
  designed for that objective.
- You need causal or predictive conclusions: use supervised learning and
  validation, not a visualization.

## 💡 Common Pitfalls & Best Practices

1. **Do not interpret cluster sizes.** t-SNE expands sparse regions and
   compresses dense ones. A large-looking island is not necessarily a large
   segment.
2. **Do not interpret distances between clusters.** The gap between two islands
   does not measure how different the groups are. Only local neighborhoods carry
   meaning.
3. **Do not treat t-SNE as clustering.** It returns coordinates, not labels.
   Run a clustering algorithm on the original features and compare the result.
4. **Standardize features.** A feature measured in millions will dominate a
   feature measured in fractions. Use `StandardScaler` or an equivalent
   transformation.
5. **Tune perplexity thoughtfully.** Try several values between `5` and `50`.
   Low perplexity highlights tiny local groups and noise; high perplexity can
   merge small clusters.
6. **Use multiple seeds.** The objective is non-convex. Stable structure across
   seeds is more credible than a single beautiful plot.
7. **Do not compare absolute coordinates across runs.** The map can rotate,
   reflect, and rescale. Compare neighborhoods, not `x`/`y` values.
8. **Give early exaggeration enough iterations.** Too short and clusters remain
   merged; too long and the map can become artificially tight.
9. **Check the KL history.** It should trend downward, but small temporary
   increases are normal because of momentum and exaggeration.
10. **Watch runtime and memory.** Exact t-SNE is quadratic. For more than a few
    thousand points, use an approximate method.
11. **Validate on original features.** A pretty plot is a hypothesis. Confirm
    group differences with statistics or a supervised model.
12. **Remember high-dimensional distance concentration.** In very high
    dimensions, nearest and farthest neighbors can have similar distances,
    making local structure less reliable.
13. **Be careful with categorical features.** Encode them thoughtfully; raw
    one-hot Euclidean distances may not reflect meaningful similarity.
14. **Do not read causation into the map.** A cluster is a similarity pattern,
    not a causal mechanism.

## 📁 Files in This Directory

- **`README.md`** (this file) — theory and concepts in English
- **`README.id.md`** — complete documentation in Bahasa Indonesia
- **`implementation.py`** — affinity calibration, symmetric `P`, Student-t `Q`,
  KL divergence and gradient, `TSNE`, trustworthiness, kNN preservation,
  synthetic customer segments, and a `__main__` demo
- **`sklearn_example.py`** — scikit-learn t-SNE vs PCA and a perplexity sweep,
  saving PNG figures with the Agg backend
- **`exercise.ipynb`** — hands-on exercises with TODOs and hints

## 🚀 Quick Start

Run the from-scratch demo:

```bash
python3 02_unsupervised_learning/05_tsne/implementation.py
```

Run the scikit-learn comparison and save the figures:

```bash
python3 02_unsupervised_learning/05_tsne/sklearn_example.py
```

Run the tests:

```bash
pytest tests/test_tsne.py -v
```

Minimal usage:

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

## 🎓 Exercises

The notebook [`exercise.ipynb`](./exercise.ipynb) walks through:

1. generating and inspecting customer-segment data,
2. calibrating `sigma` and verifying the target entropy,
3. building symmetric `P` and Student-t `Q`,
4. checking the analytic gradient against finite differences,
5. training from-scratch t-SNE and reading the KL history,
6. measuring trustworthiness and kNN preservation,
7. comparing perplexity values,
8. observing the effect of early exaggeration.

Every cell runs top-to-bottom in a few seconds. TODO prompts and hints encourage
you to change the values and observe the result.

## 🔗 Related Algorithms

- **PCA** ([04_pca](../04_pca/)): fast, linear, preserves global variance; a
  natural first look before t-SNE.
- **K-Means** ([01_kmeans](../01_kmeans/)): produces hard cluster assignments on
  the original features; t-SNE only visualizes them.
- **DBSCAN** ([02_dbscan](../02_dbscan/)): density-based clustering that can find
  arbitrary shapes and label noise.
- **Hierarchical Clustering** ([03_hierarchical](../03_hierarchical/)):
  dendrogram-based grouping with an explicit distance notion.
- **Association Rules** ([06_association_rules](../06_association_rules/)):
  discovers co-occurrence patterns in transactional data, a different
  unsupervised objective.
- **Autoencoders** (advanced topics): learn nonlinear embeddings that can embed
  new points, unlike t-SNE.

## 📚 References

- van der Maaten, L., & Hinton, G. (2008). *Visualizing Data using t-SNE*.
  Journal of Machine Learning Research, 9, 2579-2605.
- van der Maaten, L. (2014). *Accelerating t-SNE using Tree-Based Algorithms*.
  Journal of Machine Learning Research, 15, 3221-3245.
- Hinton, G., & Roweis, S. (2002). *Stochastic Neighbor Embedding*. Advances in
  Neural Information Processing Systems.
- [scikit-learn TSNE documentation](https://scikit-learn.org/stable/modules/generated/sklearn.manifold.TSNE.html)
- [scikit-learn manifold learning guide](https://scikit-learn.org/stable/modules/manifold.html)
- [t-SNE (Wikipedia)](https://en.wikipedia.org/wiki/T-distributed_stochastic_neighbor_embedding)
- [How to Use t-SNE Effectively (Distill)](https://distill.pub/2016/misread-tsne/)

---

**Next Steps**:
- Read `implementation.py` to see the bisection, the Student-t kernel, and the
  vectorized gradient.
- Run `sklearn_example.py` and compare the PCA and t-SNE panels.
- Complete `exercise.ipynb` and try several perplexities on the same data.
- Apply t-SNE to your own normalized features, but validate any apparent
  clusters on the original data before acting on them.
