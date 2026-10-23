"""
Tests for the optimizers module (04_advanced_topics/08_optimizers).

The implementation is loaded with ``importlib.util.spec_from_file_location``
under a unique module name, following ``tests/test_object_detection.py``.  The
suite covers every update rule, Adam's bias correction, coupled vs decoupled
weight decay, learning-rate schedules, gradient checks for the test functions
and models, mini-batch training and the benchmark helpers.
"""

import importlib.util
import itertools
import math
from pathlib import Path

import numpy as np
import pytest

MODULE_DIR = (
    Path(__file__).resolve().parent.parent / "04_advanced_topics" / "08_optimizers"
)
_SPEC = importlib.util.spec_from_file_location(
    "optimizers_implementation", MODULE_DIR / "implementation.py"
)
assert _SPEC is not None and _SPEC.loader is not None
impl = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(impl)

QUAD_LRS = {
    "SGD": 0.03,
    "Momentum": 0.01,
    "Nesterov": 0.01,
    "AdaGrad": 1.0,
    "RMSProp": 0.03,
    "Adam": 0.3,
    "AdamW": 0.3,
}


# --------------------------------------------------------------------------- #
# Convergence
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("name", list(impl.OPTIMIZERS))
def test_converges_on_ill_conditioned_quadratic(name):
    kwargs = {"lr": QUAD_LRS[name]}
    if name == "AdamW":
        kwargs["weight_decay"] = 0.0
    if name == "RMSProp":
        # constant-lr RMSProp hovers around the minimum; decay the rate
        kwargs["lr"] = impl.cosine_decay(QUAD_LRS[name], total_steps=1000)
    opt = impl.OPTIMIZERS[name](**kwargs)
    res = impl.optimize(impl.make_quadratic(50.0), [5.0, 1.0], opt, n_steps=3000, tol=1e-3)
    assert res["converged_at"] is not None, name
    assert np.linalg.norm(res["x"]) < 0.1


def test_momentum_beats_sgd_on_rosenbrock():
    sgd = impl.optimize(impl.rosenbrock, [-1.5, 2.0], impl.SGD(lr=1e-3), n_steps=3000)
    mom = impl.optimize(impl.rosenbrock, [-1.5, 2.0], impl.Momentum(lr=1e-3), n_steps=3000)
    assert mom["values"][-1] < sgd["values"][-1]
    assert np.allclose(mom["x"], [1.0, 1.0], atol=1e-2)


def test_adam_solves_rosenbrock():
    res = impl.optimize(impl.rosenbrock, [-1.5, 2.0], impl.Adam(lr=1.0), n_steps=5000, tol=1e-6)
    assert res["converged_at"] is not None
    assert np.allclose(res["x"], [1.0, 1.0], atol=1e-2)


def test_escapes_saddle():
    res = impl.optimize(impl.saddle, [1.0, 1e-3], impl.RMSProp(lr=impl.cosine_decay(0.3, 300)), n_steps=500)
    assert abs(abs(res["x"][1]) - math.sqrt(2)) < 1e-2
    assert abs(res["x"][0]) < 1e-2


def test_sgd_diverges_above_stability_limit():
    quad = impl.make_quadratic(50.0)
    ok = impl.optimize(quad, [5.0, 1.0], impl.SGD(lr=0.039), n_steps=300)
    bad = impl.optimize(quad, [5.0, 1.0], impl.SGD(lr=0.041), n_steps=300)
    assert ok["values"][-1] < 1e-6
    assert bad["values"][-1] > 1.0


# --------------------------------------------------------------------------- #
# Exact update rules
# --------------------------------------------------------------------------- #
def test_sgd_single_step():
    p = np.array([1.0, -2.0])
    impl.SGD(lr=0.1).step([p], [np.array([0.5, 1.0])])
    np.testing.assert_allclose(p, [0.95, -2.1])


def test_momentum_and_nesterov_two_steps():
    g = np.array([1.0])
    p_m, p_n = np.zeros(1), np.zeros(1)
    m, n = impl.Momentum(lr=0.1, beta=0.9), impl.Nesterov(lr=0.1, beta=0.9)
    for _ in range(2):
        m.step([p_m], [g])
        n.step([p_n], [g])
    # momentum: v1=1, v2=1.9 -> -0.1 - 0.19
    np.testing.assert_allclose(p_m, [-0.29])
    # nesterov: step1 = 0.1*(1+0.9*1)=0.19, step2 = 0.1*(1+0.9*1.9)=0.271
    np.testing.assert_allclose(p_n, [-0.461])


def test_adagrad_and_rmsprop_first_step():
    g = np.array([2.0, -0.5])
    p = np.zeros(2)
    impl.AdaGrad(lr=0.1, eps=0.0).step([p], [g])
    np.testing.assert_allclose(p, [-0.1, 0.1])
    p = np.zeros(2)
    impl.RMSProp(lr=0.1, rho=0.9, eps=0.0).step([p], [g])
    np.testing.assert_allclose(p, -0.1 * np.sign(g) / math.sqrt(0.1))


def test_adam_bias_correction_first_step_is_lr():
    g = np.array([1e-4, 1.0, 1e4, -3.0])
    p = np.zeros(4)
    impl.Adam(lr=0.01).step([p], [g])
    np.testing.assert_allclose(p, -0.01 * np.sign(g), rtol=1e-3)


def test_adam_without_bias_correction_is_larger():
    g = np.array([1.0])
    p = np.zeros(1)
    impl.Adam(lr=0.01, bias_correction=False).step([p], [g])
    np.testing.assert_allclose(-p, 0.01 * 0.1 / math.sqrt(0.001), rtol=1e-6)


def test_adam_matches_manual_recursion():
    rng = np.random.default_rng(0)
    grads = rng.normal(size=(5, 3))
    p = np.zeros(3)
    opt = impl.Adam(lr=0.05, beta1=0.8, beta2=0.95, eps=1e-8)
    m = v = np.zeros(3)
    q = np.zeros(3)
    for t, g in enumerate(grads, start=1):
        opt.step([p], [g])
        m = 0.8 * m + 0.2 * g
        v = 0.95 * v + 0.05 * g * g
        q = q - 0.05 * (m / (1 - 0.8**t)) / (np.sqrt(v / (1 - 0.95**t)) + 1e-8)
    np.testing.assert_allclose(p, q)


def test_adamw_decoupled_vs_coupled_decay():
    # zero gradient: AdamW still shrinks weights; coupled Adam turns decay into a
    # normalized step of size lr regardless of the weight magnitude.
    p_w = np.array([10.0])
    impl.AdamW(lr=0.1, weight_decay=0.1).step([p_w], [np.zeros(1)])
    np.testing.assert_allclose(p_w, [10.0 - 0.1 * 0.1 * 10.0])
    p_a = np.array([10.0])
    impl.Adam(lr=0.1, weight_decay=0.1).step([p_a], [np.zeros(1)])
    np.testing.assert_allclose(p_a, [9.9], rtol=1e-6)


def test_adamw_zero_decay_equals_adam():
    rng = np.random.default_rng(1)
    a, b = np.ones(4), np.ones(4)
    oa, ob = impl.Adam(lr=0.01), impl.AdamW(lr=0.01, weight_decay=0.0)
    for _ in range(10):
        g = rng.normal(size=4)
        oa.step([a], [g])
        ob.step([b], [g])
    np.testing.assert_allclose(a, b)


def test_step_validates_lengths_and_reset():
    opt = impl.Adam()
    with pytest.raises(ValueError):
        opt.step([np.zeros(2)], [])
    opt.step([np.zeros(2)], [np.ones(2)])
    assert opt.t == 1 and opt.state
    opt.reset()
    assert opt.t == 0 and not opt.state


# --------------------------------------------------------------------------- #
# Schedules
# --------------------------------------------------------------------------- #
def test_step_decay():
    s = impl.step_decay(1.0, step_size=10, gamma=0.5)
    assert [s(0), s(9), s(10), s(25)] == [1.0, 1.0, 0.5, 0.25]
    with pytest.raises(ValueError):
        impl.step_decay(1.0, 0)


def test_cosine_decay():
    s = impl.cosine_decay(1.0, total_steps=100, min_lr=0.1)
    assert s(0) == pytest.approx(1.0)
    assert s(50) == pytest.approx(0.55)
    assert s(100) == pytest.approx(0.1)
    assert s(500) == pytest.approx(0.1)
    vals = [s(t) for t in range(101)]
    assert all(a >= b for a, b in itertools.pairwise(vals))


def test_warmup():
    s = impl.warmup(impl.constant_lr(1.0), warmup_steps=4)
    assert [s(t) for t in range(6)] == [0.25, 0.5, 0.75, 1.0, 1.0, 1.0]
    base = impl.constant_lr(0.3)
    assert impl.warmup(base, 0) is base


def test_optimizer_uses_schedule():
    opt = impl.SGD(lr=impl.step_decay(1.0, step_size=1, gamma=0.5))
    p = np.zeros(1)
    for _ in range(3):
        opt.step([p], [np.ones(1)])
    np.testing.assert_allclose(p, [-(1 + 0.5 + 0.25)])


# --------------------------------------------------------------------------- #
# Gradient checks
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize(
    "fn", [impl.rosenbrock, impl.make_quadratic(50.0), impl.saddle], ids=["rosen", "quad", "saddle"]
)
def test_test_function_gradients(fn):
    rng = np.random.default_rng(0)
    for _ in range(5):
        x = rng.uniform(-2, 2, 2)
        num = impl.numerical_gradient(lambda z: fn(z)[0], x)
        np.testing.assert_allclose(fn(x)[1], num, rtol=1e-5, atol=1e-5)


@pytest.mark.parametrize("model_cls", [impl.LogisticRegressionModel, impl.MLPModel])
def test_model_gradients(model_cls):
    X, y = impl.make_moons_np(30, seed=3)
    model = model_cls(2, seed=3)
    _, grads = model.loss_and_grads(X, y)
    for k, p in enumerate(model.params):
        def f(v, k=k):
            old = model.params[k]
            model.params[k] = v
            loss = model.loss_and_grads(X, y)[0]
            model.params[k] = old
            return loss
        np.testing.assert_allclose(grads[k], impl.numerical_gradient(f, p.copy()), atol=1e-6)


# --------------------------------------------------------------------------- #
# Training and benchmarks
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("batch_size", [None, 1, 32])
def test_logistic_regression_training_modes(batch_size):
    X, y = impl.make_blobs_binary(200, seed=0)
    model = impl.LogisticRegressionModel(2)
    hist = impl.train(model, X, y, impl.SGD(lr=0.1), epochs=20, batch_size=batch_size)
    assert hist[-1] < hist[0] or hist[-1] < 0.3
    assert impl.accuracy(model, X, y) > 0.85


def test_mlp_adam_learns_moons():
    X, y = impl.make_moons_np(300, seed=0)
    model = impl.MLPModel(2, hidden=16)
    hist = impl.train(model, X, y, impl.Adam(lr=0.01), epochs=40, batch_size=32)
    assert hist[-1] < 0.5 * hist[0]
    assert impl.accuracy(model, X, y) > 0.9


def test_compare_on_mlp_adaptive_beats_sgd():
    rows = {r["optimizer"]: r for r in impl.compare_on_mlp(epochs=30)}
    assert set(rows) == set(impl.OPTIMIZERS)
    assert rows["Adam"]["final_loss"] < rows["SGD"]["final_loss"]


def test_benchmark_table_shape():
    rows = impl.benchmark_test_functions(n_steps=500, lr_grid=(0.01, 0.1))
    assert len(rows) == 3 * len(impl.OPTIMIZERS)
    assert {r["problem"] for r in rows} == set(impl.BENCHMARK_PROBLEMS)
    saddle_sgd = next(r for r in rows if r["problem"] == "saddle" and r["optimizer"] == "SGD")
    assert saddle_sgd["steps"] is not None


def test_datasets_are_deterministic_and_balanced():
    X1, y1 = impl.make_moons_np(100, seed=5)
    X2, _y2 = impl.make_moons_np(100, seed=5)
    np.testing.assert_array_equal(X1, X2)
    assert y1.sum() == 50
    Xb, yb = impl.make_blobs_binary(101, seed=0)
    assert Xb.shape == (101, 2) and set(np.unique(yb)) == {0.0, 1.0}


def test_plot_trajectories(tmp_path):
    pytest.importorskip("matplotlib")
    out = tmp_path / "traj.png"
    fig = impl.plot_trajectories(impl.rosenbrock, [-1.5, 2.0], {"SGD": {"lr": 1e-3}}, n_steps=20, path=str(out))
    assert out.exists() and fig is not None
