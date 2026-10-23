# Optimizers: How Models Actually Learn

**🇬🇧 English | [🇮🇩 Read in Bahasa Indonesia](README.id.md)**

🏠 Back to Advanced Topics: `../README.md`

Every model in this repository that is trained with gradients (logistic
regression, neural networks, the CNN and the mini YOLO detector) depends on one
small piece of code: the **optimizer**, the rule that turns a gradient into a
parameter update. This module implements the most important optimizers from
scratch in **NumPy only**, compares them on classic test functions and on small
models, and explains when each one is a good choice.

What you will build:

- SGD, mini-batch SGD, Momentum (heavy ball), Nesterov momentum,
- AdaGrad, RMSProp, Adam (with bias correction) and AdamW,
- learning-rate schedules: step decay, cosine annealing and linear warmup,
- three 2-D test problems: Rosenbrock, an ill-conditioned quadratic and a saddle,
- a mini-batch training loop for logistic regression and a small MLP,
- a numerical gradient checker and trajectory plots.

All data is synthetic, everything runs on a CPU in seconds.

## 📚 Table of Contents

- [The Basic Idea](#-the-basic-idea)
- [Batch, Mini-Batch and Stochastic](#-batch-mini-batch-and-stochastic)
- [Momentum and Nesterov](#-momentum-and-nesterov)
- [Adaptive Methods: AdaGrad and RMSProp](#-adaptive-methods-adagrad-and-rmsprop)
- [Adam and Bias Correction](#-adam-and-bias-correction)
- [AdamW: Decoupled Weight Decay](#-adamw-decoupled-weight-decay)
- [Learning-Rate Schedules](#-learning-rate-schedules)
- [Test Functions](#-test-functions)
- [Quick Start](#-quick-start)
- [Results](#-results)
- [Choosing an Optimizer](#-choosing-an-optimizer)
- [Notebook Exercises](#-notebook-exercises)
- [Tests](#-tests)
- [Files in This Directory](#-files-in-this-directory)
- [References](#-references)

## 🧠 The Basic Idea

Training means minimizing a loss $L(\theta)$ over parameters $\theta$. The
gradient $g_t = \nabla L(\theta_t)$ points uphill, so gradient descent takes a
step downhill:

$$
\theta_{t+1} = \theta_t - \eta \, g_t
$$

The learning rate $\eta$ is the single most important hyper-parameter. Too
small and training crawls; too large and the iterates oscillate or diverge.
For a quadratic with curvatures $\lambda_{\min} \le \dots \le \lambda_{\max}$,
plain gradient descent is stable only when $\eta < 2/\lambda_{\max}$, and the
slow direction then shrinks by a factor of about $1 - \lambda_{\min}/\lambda_{\max}$
per step. The ratio $\kappa = \lambda_{\max}/\lambda_{\min}$ is the
**condition number**; a large $\kappa$ means a long, narrow valley, and that is
exactly the problem most optimizers below try to fix.

## 🎲 Batch, Mini-Batch and Stochastic

The full gradient is an average over all $n$ training examples. Computing it
every step is expensive, so in practice we estimate it on a random
**mini-batch** $B$:

$$
g_t \approx \frac{1}{|B|} \sum_{i \in B} \nabla \ell_i(\theta_t)
$$

- `batch_size=None`: full-batch gradient descent (smooth but slow per update),
- `batch_size=1`: pure stochastic gradient descent (very noisy),
- `batch_size=32` to `256`: the usual compromise, and what GPUs like.

The noise is not only a cost: it helps escape saddle points and flat regions.
In `implementation.py` the batch size is an argument of `train(...)`, so every
optimizer can be used in any of the three modes.

## 🏃 Momentum and Nesterov

**Momentum** (Polyak's heavy ball, 1964) keeps a running velocity so that
consistent gradient directions accumulate and oscillating ones cancel:

$$
v_{t+1} = \beta v_t + g_t, \qquad \theta_{t+1} = \theta_t - \eta \, v_{t+1}
$$

With $\beta = 0.9$ the effective step in a consistent direction is about ten
times larger, while zig-zagging across a narrow valley is damped.

**Nesterov's accelerated gradient** (1983) evaluates the gradient at the point
where momentum is about to take us, which corrects overshooting earlier. The
form used by most libraries (and here) is

$$
v_{t+1} = \beta v_t + g_t, \qquad \theta_{t+1} = \theta_t - \eta \,(g_t + \beta v_{t+1})
$$

For convex smooth problems Nesterov's method achieves the optimal
$O(1/t^2)$ rate among first-order methods, versus $O(1/t)$ for plain
gradient descent.

## 📐 Adaptive Methods: AdaGrad and RMSProp

Instead of one global learning rate, adaptive methods give each parameter its
own step size based on the history of its gradients.

**AdaGrad** (Duchi, Hazan & Singer, 2011) divides by the root of the
accumulated squared gradients:

$$
G_t = G_{t-1} + g_t^2, \qquad \theta_{t+1} = \theta_t - \frac{\eta}{\sqrt{G_t} + \epsilon} \, g_t
$$

(all operations element-wise). Rare features get large steps, frequent ones
small steps, which works well for sparse data. Because $G_t$ only grows, the
effective learning rate keeps shrinking and training can stall.

**RMSProp** (Tieleman & Hinton, 2012, Coursera lecture 6.5) replaces the sum by
an exponential moving average so old gradients are forgotten:

$$
s_t = \rho \, s_{t-1} + (1 - \rho) \, g_t^2, \qquad
\theta_{t+1} = \theta_t - \frac{\eta}{\sqrt{s_t} + \epsilon} \, g_t
$$

## ⚙️ Adam and Bias Correction

**Adam** (Kingma & Ba, 2014) combines momentum (first moment) with RMSProp
scaling (second moment):

$$
m_t = \beta_1 m_{t-1} + (1 - \beta_1) g_t, \qquad
v_t = \beta_2 v_{t-1} + (1 - \beta_2) g_t^2
$$

Both moments start at zero, so early on they are biased toward zero. Unrolling
the recursion gives $\mathbb{E}[m_t] \approx (1 - \beta_1^t)\,\mathbb{E}[g]$,
which motivates the **bias correction**

$$
\hat m_t = \frac{m_t}{1 - \beta_1^t}, \qquad \hat v_t = \frac{v_t}{1 - \beta_2^t}, \qquad
\theta_{t+1} = \theta_t - \eta \, \frac{\hat m_t}{\sqrt{\hat v_t} + \epsilon}
$$

A useful consequence: at the very first step $\hat m_1 = g_1$ and
$\hat v_1 = g_1^2$, so every parameter moves by almost exactly $\eta$
regardless of the gradient scale. The tests check this property. Without the
correction ($\beta_2 = 0.999$) the first step would be about
$0.1/\sqrt{0.001} \approx 3.16$ times larger than intended.
Defaults: $\beta_1 = 0.9$, $\beta_2 = 0.999$, $\epsilon = 10^{-8}$.

## 🪶 AdamW: Decoupled Weight Decay

With plain SGD, adding an L2 penalty $\tfrac{\lambda}{2}\lVert\theta\rVert^2$
to the loss is the same as shrinking the weights each step. With Adam it is
**not**: the penalty gradient $\lambda\theta$ is divided by $\sqrt{\hat v_t}$,
so parameters with large gradients are barely regularized. Loshchilov & Hutter
(2019) proposed applying the decay directly to the weights instead:

$$
\theta_{t+1} = \theta_t - \eta \left( \frac{\hat m_t}{\sqrt{\hat v_t} + \epsilon} + \lambda \, \theta_t \right)
$$

In the code, `Adam(weight_decay=...)` is the coupled (L2) version and
`AdamW(weight_decay=...)` is the decoupled one, so you can compare them. AdamW
is the default optimizer for training transformers today.

## 📉 Learning-Rate Schedules

A schedule is a function $t \mapsto \eta_t$. Every optimizer in
`implementation.py` accepts either a number or a schedule as `lr`.

| Schedule | Formula | Function |
|---|---|---|
| Constant | $\eta_t = \eta_0$ | `constant_lr(lr)` |
| Step decay | $\eta_t = \eta_0 \, \gamma^{\lfloor t / s \rfloor}$ | `step_decay(lr, step_size, gamma)` |
| Cosine | $\eta_t = \eta_{\min} + \tfrac{1}{2}(\eta_0 - \eta_{\min})(1 + \cos(\pi t / T))$ | `cosine_decay(lr, total_steps, min_lr)` |
| Linear warmup | $\eta_t = \eta_0 (t+1)/W$ for $t < W$, then the wrapped schedule | `warmup(schedule, warmup_steps)` |

Warmup matters most for Adam-style methods on large models: early gradient
statistics are noisy, and a few hundred small steps prevent a bad start.
Cosine with warmup is a common default for modern deep learning.

```python
from implementation import Adam, cosine_decay, warmup

opt = Adam(lr=warmup(cosine_decay(1e-3, total_steps=900), warmup_steps=100))
```

## 🗺️ Test Functions

| Function | Formula | Minimum | What it tests |
|---|---|---|---|
| Rosenbrock | $(1 - x)^2 + 100 (y - x^2)^2$ | $(1, 1)$, $f = 0$ | curved narrow valley |
| Ill-conditioned quadratic | $\tfrac{1}{2}(x^2 + 50 y^2)$ | $(0, 0)$, $f = 0$ | condition number $\kappa = 50$ |
| Saddle | $x^2 - y^2 + \tfrac{1}{4} y^4$ | $(0, \pm\sqrt 2)$, $f = -1$ | escaping a saddle at the origin |

The saddle run starts at $(1, 10^{-3})$, almost exactly on the ridge, so the
optimizer must amplify a tiny component to escape.

![Trajectories on the Rosenbrock function](trajectories_rosenbrock.png)

![Trajectories on the ill-conditioned quadratic](trajectories_quadratic.png)

![Trajectories on the saddle function](trajectories_saddle.png)

## 🚀 Quick Start

```bash
# From the repository root
uv run python 04_advanced_topics/08_optimizers/implementation.py
uv run python 04_advanced_topics/08_optimizers/sklearn_example.py
uv run pytest tests/test_optimizers.py -v
```

```python
import numpy as np
from implementation import Adam, MLPModel, make_moons_np, train, accuracy

X, y = make_moons_np(400, seed=0)
model = MLPModel(n_features=2, hidden=16)
history = train(model, X, y, Adam(lr=0.01), epochs=60, batch_size=32)
print(history[-1], accuracy(model, X, y))
```

## 📊 Results

**Steps to reach $f - f^\star < 10^{-6}$** (max 5000 steps). Each optimizer
uses its best learning rate from the grid
$\{3\cdot10^{-4}, 10^{-3}, 3\cdot10^{-3}, 0.01, 0.03, 0.1, 0.3, 1\}$; all other
hyper-parameters are defaults. AdamW runs with zero weight decay here, so it
matches Adam exactly.

| Optimizer | Rosenbrock | Quadratic ($\kappa = 50$) | Saddle |
|---|---|---|---|
| SGD | not reached ($6.5 \times 10^{-3}$) | 269 | 19 |
| Momentum | 1484 | 95 | 120 |
| Nesterov | 1379 | 97 | 25 |
| AdaGrad | not reached ($1.6 \times 10^{-3}$) | 71 | 7 |
| RMSProp | not reached ($7.8 \times 10^{-4}$) | 32 | 8 |
| Adam / AdamW | **706** | 132 | 97 |

**One-hidden-layer MLP (16 tanh units) on two moons**, 60 epochs, batch 32:

| Optimizer | lr | Final loss | Accuracy |
|---|---|---|---|
| SGD | 0.1 | 0.214 | 0.910 |
| Momentum | 0.05 | 0.056 | 0.988 |
| Nesterov | 0.05 | 0.055 | 0.988 |
| AdaGrad | 0.1 | 0.083 | 0.973 |
| RMSProp | 0.01 | 0.051 | 0.993 |
| Adam | 0.01 | 0.059 | 0.988 |
| AdamW | 0.01 | 0.062 | 0.985 |

How to read these numbers:

- Plain SGD never solves Rosenbrock in 5000 steps: the valley floor is almost
  flat while the walls are steep. Momentum and Nesterov fix this; Adam is fastest.
- The quadratic is axis-aligned, which flatters the per-coordinate methods
  (RMSProp, AdaGrad): their diagonal scaling exactly matches the curvature. A
  rotated quadratic would remove that advantage (try it in the notebook).
- On the saddle, RMSProp and AdaGrad escape quickly because they rescale the
  tiny $y$ gradient into a full-size step. Heavy momentum overshoots the minima
  and needs extra steps to settle.
- RMSProp with a constant learning rate tends to hover near the minimum instead
  of converging exactly; a decaying schedule fixes that.
- On the small MLP, every method with momentum or adaptivity clearly beats
  plain SGD at the same budget.

## 🧭 Choosing an Optimizer

| Situation | Good first choice |
|---|---|
| Default for neural networks | Adam ($\eta = 10^{-3}$) |
| Training transformers or using weight decay | AdamW + warmup + cosine |
| CNNs where best final accuracy matters | SGD + Nesterov momentum + step or cosine schedule |
| Sparse features (text, recommender) | AdaGrad |
| Recurrent networks, non-stationary objectives | RMSProp or Adam |
| Small convex problems (logistic regression) | L-BFGS (`sklearn.linear_model.LogisticRegression`) |

## 📝 Notebook Exercises

`exercise.ipynb` contains guided exercises:

1. Find the largest stable learning rate for SGD on the quadratic and compare it
   with $2/\lambda_{\max}$.
2. Rotate the quadratic by 45° and rerun the benchmark: who loses their advantage?
3. Turn Adam's bias correction off and measure the size of the first step.
4. Compare `Adam(weight_decay=...)` with `AdamW(weight_decay=...)` on the MLP.
5. Plot step, cosine and warmup schedules, then train with each.
6. Implement a new optimizer (for example AMSGrad or Lion) and add it to the race.

## 🧪 Tests

```bash
uv run pytest tests/test_optimizers.py -v
```

The tests cover convergence on a quadratic for every optimizer, the exact
values of Adam's first steps with and without bias correction, decoupled vs
coupled weight decay, all schedules, gradient checks for the test functions and
both models, mini-batch training and the benchmark table.

## 📁 Files in This Directory

| File | Description |
|---|---|
| `README.md` | This guide (English) |
| `README.id.md` | Indonesian guide |
| `implementation.py` | Optimizers, schedules, test functions, models, training loop, benchmark |
| `sklearn_example.py` | NumPy optimizers vs scikit-learn solvers on the same data |
| `exercise.ipynb` | Hands-on exercises |
| `trajectories_*.png` | Optimizer paths on the three test functions |

## 📚 References

- Polyak, B. T. (1964). *Some methods of speeding up the convergence of
  iteration methods.* USSR Computational Mathematics and Mathematical Physics,
  4(5), 1–17. <https://doi.org/10.1016/0041-5553(64)90137-5>
- Nesterov, Y. (1983). *A method for solving the convex programming problem
  with convergence rate $O(1/k^2)$.* Soviet Mathematics Doklady, 27, 372–376.
- Duchi, J., Hazan, E., & Singer, Y. (2011). *Adaptive Subgradient Methods for
  Online Learning and Stochastic Optimization.* JMLR, 12, 2121–2159.
  <https://jmlr.org/papers/v12/duchi11a.html>
- Tieleman, T., & Hinton, G. (2012). *Lecture 6.5 — RMSProp: Divide the
  gradient by a running average of its recent magnitude.* COURSERA: Neural
  Networks for Machine Learning.
- Sutskever, I., Martens, J., Dahl, G., & Hinton, G. (2013). *On the importance
  of initialization and momentum in deep learning.* ICML.
  <https://proceedings.mlr.press/v28/sutskever13.html>
- Kingma, D. P., & Ba, J. (2014). *Adam: A Method for Stochastic Optimization.*
  <https://arxiv.org/abs/1412.6980>
- Loshchilov, I., & Hutter, F. (2017). *SGDR: Stochastic Gradient Descent with
  Warm Restarts.* <https://arxiv.org/abs/1608.03983>
- Loshchilov, I., & Hutter, F. (2019). *Decoupled Weight Decay
  Regularization.* ICLR. <https://arxiv.org/abs/1711.05101>
- Goyal, P., et al. (2017). *Accurate, Large Minibatch SGD: Training ImageNet
  in 1 Hour.* (linear warmup) <https://arxiv.org/abs/1706.02677>

**Happy Learning! 🎓** | *[🇮🇩 Indonesian version](README.id.md)*
