# Support Vector Machines (SVM)

[🇮🇩 Baca dalam Bahasa Indonesia](README.id.md) | 🇬🇧 English

🏠 Back to Supervised Index: `../README.md`

Support Vector Machines are margin-based classifiers. Instead of only fitting a
decision boundary that separates the classes, an SVM looks for the boundary with
the largest possible margin around it. With the kernel trick, the same idea can
produce highly non-linear boundaries while still being trained through a convex
optimization problem.

This module implements two binary SVMs from scratch with NumPy:

- a **linear SVM** trained with **Pegasos** (hinge loss + L2 regularization),
- a **kernel SVM** trained with a **simplified SMO** solver for linear, RBF and
  polynomial kernels.

All examples use synthetic outlet data: the goal is to classify an outlet as
**high-performing** or **low-performing** from `daily_sales`, `foot_traffic` and
`staff_count`.

## 📚 Table of Contents
- [Theory & Concepts](#-theory--concepts)
- [Mathematical Foundation](#-mathematical-foundation)
- [Pegasos for the Linear SVM](#-pegasos-for-the-linear-svm)
- [Kernel SVM and Simplified SMO](#-kernel-svm-and-simplified-smo)
- [The Outlet Dataset](#-the-outlet-dataset)
- [Choosing C and Gamma](#-choosing-c-and-gamma)
- [Advantages & Disadvantages](#-advantages--disadvantages)
- [When to Use](#-when-to-use)
- [Files in This Directory](#-files-in-this-directory)
- [Quick Start](#-quick-start)
- [Related Algorithms](#-related-algorithms)

## 🧠 Theory & Concepts

### What is a Support Vector Machine?

A linear classifier can be written as:

```
f(x) = w · x + b
```

The predicted class is `+1` when `f(x) >= 0` and `-1` otherwise. Many different
hyperplanes can separate two classes. An SVM chooses the one whose closest
training points — the **support vectors** — are as far away as possible.

The distance from a point to the hyperplane is `|w · x + b| / ||w||`. For a
correctly classified point with margin `y_i (w · x_i + b) >= 1`, the geometric
margin is `1 / ||w||`. Maximizing the margin is therefore equivalent to
minimizing `||w||²`.

### Hard Margin vs Soft Margin

A **hard-margin** SVM requires every training point to be on the correct side of
the margin. Real data is usually noisy or overlapping, so the **soft-margin**
formulation introduces slack variables `ξ_i >= 0`:

```
minimize    (1/2) ||w||² + C Σ_i ξ_i
subject to  y_i (w · x_i + b) >= 1 - ξ_i
            ξ_i >= 0
```

`C` controls the trade-off:

- **Small C**: strong regularization, wider margin, more margin violations
  (higher bias, lower variance).
- **Large C**: weak regularization, narrower margin, fewer violations
  (lower bias, higher variance, overfitting risk).

### Hinge Loss + L2 Regularization

The soft-margin problem is equivalent to minimizing the unconstrained objective:

```
J(w, b) = (lambda / 2) ||w||² + (1/n) Σ_i max(0, 1 - y_i (w · x_i + b))
```

The second term is the **hinge loss**. It is zero for points outside the margin
and grows linearly for points inside the margin or on the wrong side. The first
term is **L2 regularization**, controlled by `lambda = 1 / (C n)` in the
standard parameterization.

### Support Vectors

Only points with `y_i (w · x_i + b) <= 1` contribute a non-zero hinge-loss
gradient. These are the support vectors. Moving other training points does not
change the fitted boundary, which is one reason SVMs can generalize well.

### The Kernel Trick

If the data is not linearly separable, map it to a higher-dimensional feature
space `φ(x)` and apply a linear SVM there. The optimization only needs inner
products `φ(x_i) · φ(x_j)`, so we can replace them with a kernel function:

```
K(x_i, x_j) = φ(x_i) · φ(x_j)
```

The decision function becomes:

```
f(x) = Σ_i α_i y_i K(x_i, x) + b
```

where `α_i` are dual variables. Training does not require explicitly computing
`φ(x)`.

## 📐 Mathematical Foundation

### Primal Problem

```
minimize    (1/2) ||w||² + C Σ_i ξ_i
subject to  y_i (w · x_i + b) >= 1 - ξ_i
            ξ_i >= 0
```

### Dual Problem

For a kernel `K`, the dual is:

```
maximize    Σ_i α_i - (1/2) Σ_i Σ_j α_i α_j y_i y_j K(x_i, x_j)
subject to  0 <= α_i <= C
            Σ_i α_i y_i = 0
```

The primal weights can be recovered as `w = Σ_i α_i y_i φ(x_i)`. For a linear
kernel this becomes `w = Σ_i α_i y_i x_i`.

### KKT Conditions

At the optimum, every training point satisfies one of:

```
α_i = 0      =>  y_i f(x_i) >= 1       (outside the margin, not a support vector)
0 < α_i < C  =>  y_i f(x_i) = 1        (on the margin, free support vector)
α_i = C      =>  y_i f(x_i) <= 1       (inside the margin or misclassified)
```

SMO uses violations of these conditions to decide which variables to update.

## ⚙️ Pegasos for the Linear SVM

Pegasos (**Primal Estimated sub-GrAdient SOlver**) minimizes the regularized
hinge loss directly with stochastic subgradient descent.

At iteration `t` the learning rate is:

```
eta_t = 1 / (lambda * t)
```

For a randomly selected training point `(x_i, y_i)`:

```
if y_i (w · x_i + b) < 1:
    w <- (1 - eta_t * lambda) * w + eta_t * y_i * x_i
    b <- b + eta_t * y_i
else:
    w <- (1 - eta_t * lambda) * w
```

The `(1 - eta_t * lambda)` factor comes from the L2 regularizer. The bias is not
regularized in this implementation. With a small `lambda`, the initial steps are
large; this is expected and the decay brings the weights down over time.

`LinearSVM` also exposes the full objective after each epoch in
`train_losses_`.

## 🌐 Kernel SVM and Simplified SMO

The `KernelSVM` class solves the dual problem with **Sequential Minimal
Optimization**. SMO updates two dual variables at a time because the equality
constraint `Σ_i α_i y_i = 0` links them.

This implementation uses a deliberately simplified loop:

1. Visit every `α_i` in a shuffled order.
2. Check the KKT conditions within `tol`.
3. Pick the partner `α_j` with the largest error difference
   `|E_i - E_j|`, falling back to the next-best partner if the box constraints
   leave no feasible step.
4. Update the pair analytically and clip it to `[0, C]`.
5. Recompute the intercept `b` from the updated pair.
6. Stop when a full pass makes no pair updates, or after `max_iter` passes.

No working-set cache, shrinking heuristic or second-order information is used.
That keeps the code readable while still solving the dual accurately on small
and medium-sized datasets.

### Supported Kernels

| Kernel | Formula | Important parameters |
|--------|---------|----------------------|
| Linear | `K(x, z) = x · z` | none |
| RBF | `K(x, z) = exp(-gamma * ||x - z||²)` | `gamma` |
| Polynomial | `K(x, z) = (gamma * x · z + coef0)^degree` | `gamma`, `coef0`, `degree` |

For RBF and polynomial kernels, `gamma` can be:

- `'scale'`: `1 / (n_features * X.var())` (default, adapts to the data),
- `'auto'`: `1 / n_features`,
- a positive float.

## 🏪 The Outlet Dataset

`generate_outlet_data(n_samples=300, random_state=42)` creates a reproducible
synthetic dataset:

| Feature | Meaning | Typical range |
|---------|---------|---------------|
| `daily_sales` | Revenue for the day | hundreds to thousands |
| `foot_traffic` | Visitors during the day | 250–1800 |
| `staff_count` | Staff working that day | 6–35 |

The label is based on **sales per staff member** compared with a
traffic-dependent benchmark:

```
sales_per_staff = 55 + 0.035 * foot_traffic
                  + 10 * sin(foot_traffic / 350) + noise
benchmark       = 88 + 0.00004 * (foot_traffic - 900)²
high_performing = sales_per_staff > benchmark
```

Because the benchmark is quadratic in foot traffic, the boundary is non-linear
in the raw features. A linear SVM is a strong baseline; an RBF kernel can bend
around the benchmark more closely. The feature scales differ by orders of
magnitude, which is why both implementations standardize features internally by
default.

## 🎛️ Choosing C and Gamma

### C (both linear and kernel SVMs)

```
Too small: underfitting, many margin violations, many support vectors
Balanced:  good validation accuracy, moderate number of support vectors
Too large: overfitting risk, few margin violations, usually fewer support vectors
```

### Gamma (RBF kernel)

```
Too small: kernel is almost linear / very smooth -> underfitting
Balanced:  local but overlapping regions -> good generalization
Too large: each point only influences itself -> overfitting
```

A practical workflow:

1. Standardize features.
2. Sweep `C` on a logarithmic scale (`0.01, 0.1, 1, 10, 100`).
3. Sweep `gamma` on a logarithmic scale (`0.01, 0.03, 0.1, 0.3, 1, 3`).
4. Select the configuration with the best cross-validation score, not the best
   training score.
5. Refit on the full training set and evaluate once on the test set.

## ✅ Advantages & Disadvantages

### ✅ Advantages

- Strong performance on small and medium-sized datasets.
- Effective in high-dimensional spaces, including text classification.
- Convex objective for the standard SVM formulation: no local minima.
- Kernel trick handles non-linear boundaries without explicit feature maps.
- Support vectors provide a compact representation of the decision boundary.
- Only a subset of training points influences the final model.

### ❌ Disadvantages

- Training time grows quickly with dataset size (especially kernel SVMs).
- Sensitive to feature scaling and hyperparameters (`C`, `gamma`).
- Does not directly provide calibrated probabilities.
- Standard SVM is inherently binary; multi-class requires one-vs-one or
  one-vs-rest strategies.
- Kernel SVMs require storing support vectors and computing kernels at
  prediction time.
- Less interpretable than a linear/logistic model when kernels are used.

## 🎯 When to Use

### ✅ Use an SVM when

- The dataset is small or medium-sized.
- The number of features is large relative to the number of samples.
- You need a strong non-linear boundary and can tune `C` and `gamma`.
- You want a convex, well-understood optimization problem.
- A compact support-vector representation is useful.

### ❌ Consider alternatives when

- The dataset is very large — use linear models, SGD, or tree ensembles.
- You need calibrated probabilities — use logistic regression or probability
  calibration.
- You need highly interpretable coefficients — use a linear model or a single
  decision tree.
- Training speed is critical — consider LinearSVC, LightGBM or XGBoost.
- The data is mostly categorical with many high-cardinality levels — tree-based
  models often handle it more naturally.

## 📁 Files in This Directory

- **`README.md`** (this file) — theory and concepts in English
- **`README.id.md`** — dokumentasi lengkap dalam Bahasa Indonesia
- **`implementation.py`** — `LinearSVM` (Pegasos), `KernelSVM` (simplified SMO),
  synthetic outlet data generator and a `__main__` demo
- **`sklearn_example.py`** — scikit-learn `SVC` experiments showing the effect
  of `C` and `gamma` on the outlet data (headless, prints accuracies)
- **`exercise.ipynb`** — hands-on exercises and experiments

## 🚀 Quick Start

Run the from-scratch demo:

```bash
python 01_supervised_learning/10_svm/implementation.py
```

Run the scikit-learn comparison:

```bash
python 01_supervised_learning/10_svm/sklearn_example.py
```

Run the tests:

```bash
pytest tests/test_svm.py -v
```

Minimal usage:

```python
from implementation import LinearSVM, KernelSVM, generate_outlet_data

X, y = generate_outlet_data(n_samples=300, random_state=42)

linear = LinearSVM(lambda_=1e-3, n_epochs=60, random_state=42)
linear.fit(X, y)
print(linear.score(X, y))

rbf = KernelSVM(C=10.0, kernel="rbf", gamma="scale", random_state=42)
rbf.fit(X, y)
print(rbf.score(X, y), rbf.n_support_)
```

## 🔗 Related Algorithms

- **Logistic Regression** ([02_logistic_regression](../02_logistic_regression/)):
  also a linear classifier, but optimizes log-loss and produces probabilities.
- **K-Nearest Neighbors** ([07_knn](../07_knn/)): another distance-based method;
  no training phase, but prediction requires the whole dataset.
- **Decision Trees** ([03_decision_trees](../03_decision_trees/)) and
  **Random Forests** ([04_random_forest](../04_random_forest/)): non-linear
  models that do not require feature scaling.
- **Gradient Boosting** ([09_gradient_boosting](../09_gradient_boosting/)):
  often the strongest choice on tabular data, but less suited to very
  high-dimensional sparse problems than linear SVMs.
- **Neural Networks** ([08_neural_networks](../08_neural_networks/)): flexible
  non-linear models that can learn feature representations, at the cost of more
  data and tuning.

---

**Next Steps**:
- Read `implementation.py` to see Pegasos and SMO in action.
- Run `sklearn_example.py` to explore the `C` and `gamma` trade-offs.
- Complete `exercise.ipynb` to practice tuning and comparing kernels.
- Try the same workflow on your own tabular classification problem.
