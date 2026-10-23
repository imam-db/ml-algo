"""
Optimizers from scratch with NumPy
==================================

Gradient-based optimizers used to train machine-learning models, written so
that every update rule fits on a few readable lines:

* ``SGD``          -- plain (stochastic) gradient descent
* ``Momentum``     -- heavy-ball momentum (Polyak, 1964)
* ``Nesterov``     -- Nesterov accelerated gradient (Nesterov, 1983)
* ``AdaGrad``      -- per-parameter accumulated scaling (Duchi et al., 2011)
* ``RMSProp``      -- exponential moving average of squared gradients
                      (Tieleman & Hinton, 2012)
* ``Adam``         -- adaptive moments with bias correction (Kingma & Ba, 2014)
* ``AdamW``        -- Adam with decoupled weight decay (Loshchilov & Hutter, 2019)

plus learning-rate schedules (constant, step, cosine, linear warmup), classic
test functions (Rosenbrock, ill-conditioned quadratic, saddle), a
mini-batch training loop for logistic regression and a tiny MLP on synthetic
data, and a numerical gradient checker.

Every optimizer exposes ``step(params, grads)`` where ``params`` and ``grads``
are lists of NumPy arrays; parameters are updated **in place**.

Author: ML-Algo Repository
"""

from __future__ import annotations

import math
from collections.abc import Callable, Sequence

import numpy as np

Array = np.ndarray
Schedule = Callable[[int], float]


# ---------------------------------------------------------------------------
# Learning-rate schedules
# ---------------------------------------------------------------------------
def constant_lr(lr: float) -> Schedule:
    """Return a schedule that always yields ``lr``."""
    return lambda t: float(lr)


def step_decay(lr: float, step_size: int, gamma: float = 0.5) -> Schedule:
    """Multiply ``lr`` by ``gamma`` every ``step_size`` steps (t starts at 0)."""
    if step_size <= 0:
        raise ValueError("step_size must be positive")
    return lambda t: float(lr * gamma ** (t // step_size))


def cosine_decay(lr: float, total_steps: int, min_lr: float = 0.0) -> Schedule:
    """Cosine annealing from ``lr`` at t=0 to ``min_lr`` at t=total_steps."""
    if total_steps <= 0:
        raise ValueError("total_steps must be positive")

    def schedule(t: int) -> float:
        t = min(max(t, 0), total_steps)
        return float(min_lr + 0.5 * (lr - min_lr) * (1 + math.cos(math.pi * t / total_steps)))

    return schedule


def warmup(schedule: Schedule, warmup_steps: int) -> Schedule:
    """Linear warmup: ramp from lr/warmup_steps up to ``schedule`` then follow it.

    During warmup the rate is ``schedule(0) * (t + 1) / warmup_steps``; after
    warmup the wrapped schedule is evaluated at ``t - warmup_steps``.
    """
    if warmup_steps <= 0:
        return schedule

    def wrapped(t: int) -> float:
        if t < warmup_steps:
            return float(schedule(0) * (t + 1) / warmup_steps)
        return float(schedule(t - warmup_steps))

    return wrapped


# ---------------------------------------------------------------------------
# Optimizers
# ---------------------------------------------------------------------------
class Optimizer:
    """Base class. Subclasses implement ``_update(i, p, g, lr)``."""

    name = "optimizer"

    def __init__(self, lr: float | Schedule = 0.01):
        self.schedule: Schedule = lr if callable(lr) else constant_lr(lr)
        self.t = 0  # number of completed steps
        self.state: dict[int, dict[str, Array]] = {}

    @property
    def lr(self) -> float:
        return self.schedule(self.t)

    def _slot(self, i: int, key: str, like: Array) -> Array:
        st = self.state.setdefault(i, {})
        if key not in st:
            st[key] = np.zeros_like(like, dtype=float)
        return st[key]

    def step(self, params: Sequence[Array], grads: Sequence[Array]) -> None:
        if len(params) != len(grads):
            raise ValueError("params and grads must have the same length")
        lr = self.lr
        self.t += 1
        for i, (p, g) in enumerate(zip(params, grads)):
            self._update(i, p, np.asarray(g, dtype=float), lr)

    def _update(self, i: int, p: Array, g: Array, lr: float) -> None:  # pragma: no cover
        raise NotImplementedError

    def reset(self) -> None:
        self.t = 0
        self.state = {}


class SGD(Optimizer):
    """theta <- theta - lr * g  (optionally with coupled L2 weight decay)."""

    name = "SGD"

    def __init__(self, lr: float | Schedule = 0.01, weight_decay: float = 0.0):
        super().__init__(lr)
        self.weight_decay = weight_decay

    def _update(self, i, p, g, lr):
        if self.weight_decay:
            g = g + self.weight_decay * p
        p -= lr * g


class Momentum(Optimizer):
    """Heavy ball: v <- beta v + g ; theta <- theta - lr v."""

    name = "Momentum"

    def __init__(self, lr: float | Schedule = 0.01, beta: float = 0.9):
        super().__init__(lr)
        self.beta = beta

    def _update(self, i, p, g, lr):
        v = self._slot(i, "v", p)
        v *= self.beta
        v += g
        p -= lr * v


class Nesterov(Momentum):
    """Nesterov momentum in the 'look-ahead' reformulation used by most libraries.

    v <- beta v + g ; theta <- theta - lr (g + beta v). This is algebraically
    equivalent to evaluating the gradient at the look-ahead point
    theta - lr * beta * v (Sutskever et al., 2013).
    """

    name = "Nesterov"

    def _update(self, i, p, g, lr):
        v = self._slot(i, "v", p)
        v *= self.beta
        v += g
        p -= lr * (g + self.beta * v)


class AdaGrad(Optimizer):
    """G <- G + g^2 ; theta <- theta - lr g / (sqrt(G) + eps)."""

    name = "AdaGrad"

    def __init__(self, lr: float | Schedule = 0.1, eps: float = 1e-8):
        super().__init__(lr)
        self.eps = eps

    def _update(self, i, p, g, lr):
        G = self._slot(i, "G", p)
        G += g * g
        p -= lr * g / (np.sqrt(G) + self.eps)


class RMSProp(Optimizer):
    """s <- rho s + (1-rho) g^2 ; theta <- theta - lr g / (sqrt(s) + eps)."""

    name = "RMSProp"

    def __init__(self, lr: float | Schedule = 0.001, rho: float = 0.9, eps: float = 1e-8):
        super().__init__(lr)
        self.rho = rho
        self.eps = eps

    def _update(self, i, p, g, lr):
        s = self._slot(i, "s", p)
        s *= self.rho
        s += (1 - self.rho) * g * g
        p -= lr * g / (np.sqrt(s) + self.eps)


class Adam(Optimizer):
    """Adam (Kingma & Ba, 2014) with bias-corrected first and second moments.

    ``weight_decay`` here is the *coupled* L2 penalty (added to the gradient),
    which is what AdamW fixes.
    """

    name = "Adam"

    def __init__(
        self,
        lr: float | Schedule = 0.001,
        beta1: float = 0.9,
        beta2: float = 0.999,
        eps: float = 1e-8,
        weight_decay: float = 0.0,
        bias_correction: bool = True,
    ):
        super().__init__(lr)
        self.beta1, self.beta2, self.eps = beta1, beta2, eps
        self.weight_decay = weight_decay
        self.bias_correction = bias_correction

    def _moments(self, i, p, g):
        m = self._slot(i, "m", p)
        v = self._slot(i, "v", p)
        m *= self.beta1
        m += (1 - self.beta1) * g
        v *= self.beta2
        v += (1 - self.beta2) * g * g
        if self.bias_correction:
            m_hat = m / (1 - self.beta1 ** self.t)
            v_hat = v / (1 - self.beta2 ** self.t)
        else:
            m_hat, v_hat = m, v
        return m_hat, v_hat

    def _update(self, i, p, g, lr):
        if self.weight_decay:
            g = g + self.weight_decay * p
        m_hat, v_hat = self._moments(i, p, g)
        p -= lr * m_hat / (np.sqrt(v_hat) + self.eps)


class AdamW(Adam):
    """AdamW (Loshchilov & Hutter, 2019): weight decay applied directly to theta.

    theta <- theta - lr * (m_hat / (sqrt(v_hat) + eps) + weight_decay * theta)
    """

    name = "AdamW"

    def __init__(self, lr: float | Schedule = 0.001, beta1: float = 0.9, beta2: float = 0.999,
                 eps: float = 1e-8, weight_decay: float = 0.01):
        super().__init__(lr, beta1, beta2, eps, weight_decay=0.0)
        self.decoupled_wd = weight_decay

    def _update(self, i, p, g, lr):
        m_hat, v_hat = self._moments(i, p, g)
        p -= lr * (m_hat / (np.sqrt(v_hat) + self.eps) + self.decoupled_wd * p)


OPTIMIZERS = {cls.name: cls for cls in (SGD, Momentum, Nesterov, AdaGrad, RMSProp, Adam, AdamW)}


# ---------------------------------------------------------------------------
# Test functions: each returns (value, gradient)
# ---------------------------------------------------------------------------
def rosenbrock(x: Array, a: float = 1.0, b: float = 100.0) -> tuple[float, Array]:
    """f(x, y) = (a - x)^2 + b (y - x^2)^2, minimum at (a, a^2)."""
    x0, x1 = float(x[0]), float(x[1])
    f = (a - x0) ** 2 + b * (x1 - x0**2) ** 2
    g = np.array([-2 * (a - x0) - 4 * b * x0 * (x1 - x0**2), 2 * b * (x1 - x0**2)])
    return f, g


def make_quadratic(condition: float = 50.0) -> Callable[[Array], tuple[float, Array]]:
    """f(x) = 0.5 x^T A x with A = diag(1, condition), minimum at 0."""
    diag = np.array([1.0, float(condition)])

    def f(x: Array) -> tuple[float, Array]:
        x = np.asarray(x, dtype=float)
        return float(0.5 * np.sum(diag * x * x)), diag * x

    return f


def saddle(x: Array) -> tuple[float, Array]:
    """f(x, y) = x^2 - y^2 + 0.25 y^4. Saddle at (0,0), minima at (0, +-sqrt(2))."""
    x0, x1 = float(x[0]), float(x[1])
    return x0**2 - x1**2 + 0.25 * x1**4, np.array([2 * x0, -2 * x1 + x1**3])


def numerical_gradient(f: Callable[[Array], float], x: Array, h: float = 1e-6) -> Array:
    """Central-difference gradient of a scalar function ``f`` at ``x``."""
    x = np.asarray(x, dtype=float)
    grad = np.zeros_like(x)
    for idx in np.ndindex(x.shape):
        e = np.zeros_like(x)
        e[idx] = h
        grad[idx] = (f(x + e) - f(x - e)) / (2 * h)
    return grad


def optimize(
    fn: Callable[[Array], tuple[float, Array]],
    x0: Sequence[float],
    optimizer: Optimizer,
    n_steps: int = 1000,
    tol: float | None = None,
    f_star: float = 0.0,
) -> dict[str, object]:
    """Run ``optimizer`` on ``fn`` from ``x0``.

    Returns the trajectory (n+1, d), function values and ``converged_at``: the
    first step at which ``f - f_star < tol`` (or ``None``).
    """
    x = np.array(x0, dtype=float)
    path, values = [x.copy()], []
    converged_at = None
    for t in range(n_steps):
        f, g = fn(x)
        values.append(f)
        if tol is not None and f - f_star < tol:
            converged_at = t
            break
        if not np.all(np.isfinite(x)) or abs(f) > 1e12:
            break
        optimizer.step([x], [g])
        path.append(x.copy())
    else:
        values.append(fn(x)[0])
    return {"path": np.array(path), "values": np.array(values), "converged_at": converged_at, "x": x}


LR_GRID: tuple[float, ...] = (0.0003, 0.001, 0.003, 0.01, 0.03, 0.1, 0.3, 1.0)

BENCHMARK_PROBLEMS: dict[str, tuple[Callable, list[float], float]] = {
    "rosenbrock": (rosenbrock, [-1.5, 2.0], 0.0),
    "quadratic_k50": (make_quadratic(50.0), [5.0, 1.0], 0.0),
    "saddle": (saddle, [1.0, 1e-3], -1.0),
}


def benchmark_test_functions(
    n_steps: int = 5000, tol: float = 1e-6, lr_grid: Sequence[float] = LR_GRID
) -> list[dict[str, object]]:
    """Steps-to-tolerance for every optimizer on three 2-D problems.

    For fairness each optimizer gets its *best* learning rate from ``lr_grid``
    (fewest steps to reach ``f - f* < tol``; ties broken by final loss). All
    other hyper-parameters stay at their library defaults.
    """
    rows = []
    for pname, (fn, x0, f_star) in BENCHMARK_PROBLEMS.items():
        for oname, cls in OPTIMIZERS.items():
            best = None
            for lr in lr_grid:
                kwargs = {"lr": lr}
                if cls is AdamW:
                    kwargs["weight_decay"] = 0.0  # pure optimizer comparison
                res = optimize(fn, x0, cls(**kwargs), n_steps=n_steps, tol=tol, f_star=f_star)
                final = float(res["values"][-1]) - f_star
                if not np.isfinite(final):
                    continue
                key = (res["converged_at"] if res["converged_at"] is not None else n_steps + 1, final)
                if best is None or key < best[0]:
                    best = (key, lr, res["converged_at"], final)
            rows.append({
                "problem": pname,
                "optimizer": oname,
                "lr": best[1] if best else None,
                "steps": best[2] if best else None,
                "final_loss": best[3] if best else float("nan"),
            })
    return rows


# ---------------------------------------------------------------------------
# Synthetic data and models
# ---------------------------------------------------------------------------
def make_blobs_binary(n: int = 400, seed: int = 0) -> tuple[Array, Array]:
    """Two overlapping Gaussian blobs in 2-D with labels 0/1."""
    rng = np.random.default_rng(seed)
    X0 = rng.normal([-1.0, -1.0], 1.0, size=(n // 2, 2))
    X1 = rng.normal([1.0, 1.0], 1.0, size=(n - n // 2, 2))
    X = np.vstack([X0, X1])
    y = np.r_[np.zeros(n // 2), np.ones(n - n // 2)]
    idx = rng.permutation(n)
    return X[idx], y[idx]


def make_moons_np(n: int = 400, noise: float = 0.15, seed: int = 0) -> tuple[Array, Array]:
    """NumPy-only two-moons dataset (non-linear decision boundary)."""
    rng = np.random.default_rng(seed)
    n0 = n // 2
    t0 = rng.uniform(0, np.pi, n0)
    t1 = rng.uniform(0, np.pi, n - n0)
    X = np.vstack([np.c_[np.cos(t0), np.sin(t0)], np.c_[1 - np.cos(t1), 0.5 - np.sin(t1)]])
    X += rng.normal(0, noise, X.shape)
    y = np.r_[np.zeros(n0), np.ones(n - n0)]
    idx = rng.permutation(n)
    return X[idx], y[idx]


def _sigmoid(z: Array) -> Array:
    return 1.0 / (1.0 + np.exp(-np.clip(z, -500, 500)))


def bce(p: Array, y: Array) -> float:
    p = np.clip(p, 1e-12, 1 - 1e-12)
    return float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p)))


class LogisticRegressionModel:
    """Logistic regression with parameters [w, b]."""

    def __init__(self, n_features: int, seed: int = 0):
        rng = np.random.default_rng(seed)
        self.params = [rng.normal(0, 0.01, n_features), np.zeros(1)]

    def predict_proba(self, X: Array) -> Array:
        w, b = self.params
        return _sigmoid(X @ w + b[0])

    def loss_and_grads(self, X: Array, y: Array) -> tuple[float, list[Array]]:
        p = self.predict_proba(X)
        err = (p - y) / len(y)
        return bce(p, y), [X.T @ err, np.array([err.sum()])]


class MLPModel:
    """One-hidden-layer MLP (tanh) for binary classification."""

    def __init__(self, n_features: int, hidden: int = 16, seed: int = 0):
        rng = np.random.default_rng(seed)
        self.params = [
            rng.normal(0, 1 / np.sqrt(n_features), (n_features, hidden)),
            np.zeros(hidden),
            rng.normal(0, 1 / np.sqrt(hidden), hidden),
            np.zeros(1),
        ]

    def _forward(self, X: Array):
        W1, b1, w2, b2 = self.params
        h = np.tanh(X @ W1 + b1)
        return h, _sigmoid(h @ w2 + b2[0])

    def predict_proba(self, X: Array) -> Array:
        return self._forward(X)[1]

    def loss_and_grads(self, X: Array, y: Array) -> tuple[float, list[Array]]:
        _W1, _b1, w2, _b2 = self.params
        h, p = self._forward(X)
        dz2 = (p - y) / len(y)
        dw2 = h.T @ dz2
        db2 = np.array([dz2.sum()])
        dh = np.outer(dz2, w2) * (1 - h * h)
        return bce(p, y), [X.T @ dh, dh.sum(axis=0), dw2, db2]


def accuracy(model, X: Array, y: Array) -> float:
    return float(np.mean((model.predict_proba(X) >= 0.5) == y))


def train(
    model,
    X: Array,
    y: Array,
    optimizer: Optimizer,
    epochs: int = 50,
    batch_size: int | None = 32,
    seed: int = 0,
) -> list[float]:
    """Mini-batch training. ``batch_size=None`` = full batch, ``1`` = pure SGD.

    Returns the full-dataset loss after every epoch.
    """
    rng = np.random.default_rng(seed)
    n = len(y)
    bs = n if batch_size is None else batch_size
    history = []
    for _ in range(epochs):
        order = rng.permutation(n)
        for start in range(0, n, bs):
            idx = order[start:start + bs]
            _, grads = model.loss_and_grads(X[idx], y[idx])
            optimizer.step(model.params, grads)
        history.append(model.loss_and_grads(X, y)[0])
    return history


def compare_on_mlp(epochs: int = 60, seed: int = 0) -> list[dict[str, object]]:
    """Train the same MLP on two-moons with each optimizer (batch size 32)."""
    X, y = make_moons_np(400, seed=seed)
    lrs = {"SGD": 0.1, "Momentum": 0.05, "Nesterov": 0.05, "AdaGrad": 0.1,
           "RMSProp": 0.01, "Adam": 0.01, "AdamW": 0.01}
    rows = []
    for name, lr in lrs.items():
        model = MLPModel(2, hidden=16, seed=seed)
        hist = train(model, X, y, OPTIMIZERS[name](lr=lr), epochs=epochs, batch_size=32, seed=seed)
        rows.append({"optimizer": name, "final_loss": hist[-1], "accuracy": accuracy(model, X, y),
                     "history": hist})
    return rows


# ---------------------------------------------------------------------------
# Plotting (optional, matplotlib)
# ---------------------------------------------------------------------------
def plot_trajectories(fn, x0, configs: dict[str, dict[str, object]], n_steps: int = 300,
                      bounds=((-2, 2), (-1, 3)), path: str | None = None, log_levels: bool = True):
    """Draw contour lines of ``fn`` and each optimizer's path. Returns the figure."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    xs = np.linspace(*bounds[0], 200)
    ys = np.linspace(*bounds[1], 200)
    Z = np.array([[fn(np.array([a, b]))[0] for a in xs] for b in ys])
    fig, ax = plt.subplots(figsize=(7, 5))
    if log_levels:
        ax.contour(xs, ys, np.log1p(Z - Z.min()), levels=30, cmap="Greys", linewidths=0.6)
    else:
        ax.contour(xs, ys, Z, levels=30, cmap="Greys", linewidths=0.6)
    for name, cfg in configs.items():
        res = optimize(fn, x0, OPTIMIZERS[name](**cfg), n_steps=n_steps)
        p = res["path"]
        ax.plot(p[:, 0], p[:, 1], label=name, lw=1.4)
    ax.plot(*x0, "ko")
    ax.set_xlim(*bounds[0])
    ax.set_ylim(*bounds[1])
    ax.legend(fontsize=8)
    if path:
        fig.savefig(path, dpi=110, bbox_inches="tight")
    return fig


def main() -> None:
    print("## Steps to reach f - f* < 1e-6 (max 5000)\n")
    print("| Problem | Optimizer | Best lr | Steps | Final f - f* |")
    print("|---|---|---|---|---|")
    for r in benchmark_test_functions():
        steps = r["steps"] if r["steps"] is not None else "not reached"
        print(f"| {r['problem']} | {r['optimizer']} | {r['lr']} | {steps} | {r['final_loss']:.2e} |")
    print("\n## MLP on two-moons (60 epochs, batch 32)\n")
    print("| Optimizer | Final loss | Accuracy |")
    print("|---|---|---|")
    for r in compare_on_mlp():
        print(f"| {r['optimizer']} | {r['final_loss']:.4f} | {r['accuracy']:.3f} |")


if __name__ == "__main__":
    main()
