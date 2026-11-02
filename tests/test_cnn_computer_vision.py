"""
Tests for the CNN computer-vision module (05_cnn_computer_vision).

The implementation is loaded with ``importlib`` (the same pattern used by
``tests/test_svm.py``) so that several modules can ship an ``implementation.py``
without clashing in ``sys.modules``.

The suite covers:

- ``im2col`` / ``col2im`` shapes and the adjoint property,
- convolution against a naive loop implementation,
- numerical gradient checks for conv, pooling, dense and softmax-CE,
- output shapes of the layers and the CNN,
- reproducibility and validity of the synthetic shelf data,
- end-to-end training that beats chance by a wide margin,
- the backwards-compatible ``SimpleCNN`` / ``generate_product_images`` API.
"""

import importlib.util
import time
from pathlib import Path

import numpy as np
import pytest

# ---------------------------------------------------------------------------
# Load implementation.py by path (avoids clashes with other modules)
# ---------------------------------------------------------------------------
MODULE_PATH = (
    Path(__file__).resolve().parent.parent
    / "04_advanced_topics"
    / "05_cnn_computer_vision"
    / "implementation.py"
)
spec = importlib.util.spec_from_file_location("cnn_impl", MODULE_PATH)
cnn_impl = importlib.util.module_from_spec(spec)
assert spec.loader is not None
spec.loader.exec_module(cnn_impl)

Conv2D = cnn_impl.Conv2D
MaxPool2D = cnn_impl.MaxPool2D
ReLU = cnn_impl.ReLU
Flatten = cnn_impl.Flatten
Dense = cnn_impl.Dense
SoftmaxCrossEntropy = cnn_impl.SoftmaxCrossEntropy
SGD = cnn_impl.SGD
Sequential = cnn_impl.Sequential
ShelfCNN = cnn_impl.ShelfCNN
SimpleCNN = cnn_impl.SimpleCNN
im2col = cnn_impl.im2col
col2im = cnn_impl.col2im
conv2d = cnn_impl.conv2d
max_pool2d = cnn_impl.max_pool2d
conv2d_backward = cnn_impl.conv2d_backward
numerical_gradient_check = cnn_impl.numerical_gradient_check
generate_shelf_images = cnn_impl.generate_shelf_images
generate_product_images = cnn_impl.generate_product_images
gradient_check_cnn = cnn_impl.gradient_check_cnn
shift_images = cnn_impl.shift_images


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _naive_conv_nchw(x, W, b, stride=1, padding=0):
    """Reference convolution with explicit loops, NCHW layout."""
    n, c, h, w = x.shape
    f, kc, kh, kw = W.shape
    assert c == kc
    xp = np.pad(x, ((0, 0), (0, 0), (padding, padding), (padding, padding)))
    out_h = (h + 2 * padding - kh) // stride + 1
    out_w = (w + 2 * padding - kw) // stride + 1
    out = np.zeros((n, f, out_h, out_w))
    for ni in range(n):
        for fi in range(f):
            for i in range(out_h):
                for j in range(out_w):
                    patch = xp[ni, :, i * stride:i * stride + kh,
                               j * stride:j * stride + kw]
                    out[ni, fi, i, j] = np.sum(patch * W[fi]) + b[fi]
    return out


def _relative_error(a, b):
    """Element-wise relative error with a safe denominator."""
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    return np.max(np.abs(a - b) / np.maximum(np.abs(a) + np.abs(b), 1e-12))


def _small_dataset(n_train=240, n_test=80, random_state=0):
    X_train, y_train = generate_shelf_images(
        n_samples=n_train, random_state=random_state
    )
    X_test, y_test = generate_shelf_images(
        n_samples=n_test, random_state=random_state + 1
    )
    return X_train, X_test, y_train, y_test


# ---------------------------------------------------------------------------
# im2col / col2im
# ---------------------------------------------------------------------------
class TestIm2Col:
    def test_shapes_nchw(self):
        x = np.random.randn(2, 3, 5, 5)
        cols = im2col(x, kernel_size=3, stride=2, padding=1)
        # (5 + 2*1 - 3)//2 + 1 = 3 -> 2*3*3 patches, 3*3*3 values each
        assert cols.shape == (2 * 3 * 3, 3 * 3 * 3)

        dx = col2im(cols, x.shape, kernel_size=3, stride=2, padding=1)
        assert dx.shape == x.shape

    def test_patch_order_matches_nchw_layout(self):
        x = np.arange(1 * 1 * 4 * 4, dtype=float).reshape(1, 1, 4, 4)
        cols = im2col(x, kernel_size=2, stride=2, padding=0)
        # Four non-overlapping 2x2 patches in row-major spatial order.
        np.testing.assert_array_equal(
            cols,
            np.array(
                [
                    [0, 1, 4, 5],
                    [2, 3, 6, 7],
                    [8, 9, 12, 13],
                    [10, 11, 14, 15],
                ],
                dtype=float,
            ),
        )

    @pytest.mark.parametrize(
        "kernel,stride,padding",
        [
            (3, 1, 0),
            (3, 2, 1),
            ((2, 3), (1, 2), (2, 1)),
        ],
    )
    def test_adjoint_property(self, kernel, stride, padding):
        rng = np.random.default_rng(0)
        x = rng.normal(size=(2, 3, 6, 7))
        cols = im2col(x, kernel, stride, padding)
        r = rng.normal(size=cols.shape)
        dx = col2im(r, x.shape, kernel, stride, padding)
        # <im2col(x), r> == <x, col2im(r)>
        assert np.sum(cols * r) == pytest.approx(np.sum(x * dx), rel=1e-12)


# ---------------------------------------------------------------------------
# Convolution
# ---------------------------------------------------------------------------
class TestConvolution:
    @pytest.mark.parametrize("stride,padding", [(1, 0), (2, 1), (1, 2)])
    def test_conv2d_matches_naive_loop(self, stride, padding):
        rng = np.random.default_rng(1)
        x = rng.normal(size=(2, 3, 7, 6))
        W = rng.normal(size=(4, 3, 3, 2))
        b = rng.normal(size=4)
        layer = Conv2D(3, 4, kernel=(3, 2), stride=stride, padding=padding)
        layer.params["W"] = W.copy()
        layer.params["b"] = b.copy()

        out = layer.forward(x)
        ref = _naive_conv_nchw(x, W, b, stride, padding)
        np.testing.assert_allclose(out, ref, rtol=1e-12, atol=1e-12)

    def test_legacy_2d_convolution(self):
        image = np.ones((3, 3))
        kernel = np.ones((2, 2))
        result = conv2d(image, kernel, stride=1, padding=0)
        assert result.shape == (2, 2)
        np.testing.assert_allclose(result, 4.0)

    def test_output_size_formula(self):
        layer = Conv2D(1, 2, kernel=3, stride=2, padding=1)
        out = layer.forward(np.random.randn(5, 1, 9, 11))
        # (9 + 2 - 3)//2 + 1 = 5, (11 + 2 - 3)//2 + 1 = 6
        assert out.shape == (5, 2, 5, 6)
        assert layer.out_shape((5, 1, 9, 11)) == (5, 2, 5, 6)

    def test_conv2d_backward_matches_finite_differences(self):
        rng = np.random.default_rng(2)
        image = rng.normal(size=(5, 5))
        kernel = rng.normal(size=(3, 3))
        dout = rng.normal(size=(3, 3))

        d_input, d_kernel = conv2d_backward(dout, image, kernel)
        eps = 1e-5

        def loss(im, ke):
            return np.sum(conv2d(im, ke) * dout)

        numerical_input = np.zeros_like(image)
        for idx in np.ndindex(image.shape):
            original = image[idx]
            image[idx] = original + eps
            plus = loss(image, kernel)
            image[idx] = original - eps
            minus = loss(image, kernel)
            image[idx] = original
            numerical_input[idx] = (plus - minus) / (2 * eps)

        numerical_kernel = np.zeros_like(kernel)
        for idx in np.ndindex(kernel.shape):
            original = kernel[idx]
            kernel[idx] = original + eps
            plus = loss(image, kernel)
            kernel[idx] = original - eps
            minus = loss(image, kernel)
            kernel[idx] = original
            numerical_kernel[idx] = (plus - minus) / (2 * eps)

        assert _relative_error(numerical_input, d_input) < 1e-6
        assert _relative_error(numerical_kernel, d_kernel) < 1e-6


# ---------------------------------------------------------------------------
# Pooling and activations
# ---------------------------------------------------------------------------
class TestPoolingAndActivation:
    def test_max_pool2d_legacy_and_batch(self):
        image = np.array(
            [[1, 2, 3, 4],
             [5, 6, 7, 8],
             [9, 10, 11, 12],
             [13, 14, 15, 16]],
            dtype=float,
        )
        pooled = max_pool2d(image, pool_size=2, stride=2)
        assert pooled.shape == (2, 2)
        np.testing.assert_allclose(pooled, [[6, 8], [14, 16]])

        batch = np.stack([image, image + 100.0])[:, None, :, :]
        pooled_batch = max_pool2d(batch, pool_size=2, stride=2)
        assert pooled_batch.shape == (2, 1, 2, 2)
        np.testing.assert_allclose(pooled_batch[1, 0], [[106, 108], [114, 116]])

    def test_max_pool_backward_routes_to_argmax(self):
        x = np.zeros((1, 1, 2, 2))
        x[0, 0, 0, 1] = 5.0
        layer = MaxPool2D(2, 2)
        out = layer.forward(x)
        assert out[0, 0, 0, 0] == 5.0
        dx = layer.backward(np.ones_like(out))
        expected = np.zeros_like(x)
        expected[0, 0, 0, 1] = 1.0
        np.testing.assert_array_equal(dx, expected)

    def test_relu_and_flatten(self):
        x = np.array([[-1.0, 0.0, 2.0]])
        relu = ReLU()
        y = relu.forward(x)
        np.testing.assert_array_equal(y, [[0.0, 0.0, 2.0]])
        np.testing.assert_array_equal(relu.backward(np.ones_like(y)), [[0, 0, 1]])

        flat = Flatten()
        h = np.arange(2 * 2 * 3, dtype=float).reshape(2, 2, 3)
        f = flat.forward(h)
        assert f.shape == (2, 6)
        np.testing.assert_array_equal(flat.backward(f), h)


# ---------------------------------------------------------------------------
# Numerical gradient checks
# ---------------------------------------------------------------------------
class TestNumericalGradients:
    def test_conv2d_input_and_weights(self):
        rng = np.random.default_rng(3)
        layer = Conv2D(2, 3, kernel=3, stride=2, padding=1, random_state=0)
        x = rng.normal(size=(2, 2, 7, 7))
        dout = rng.normal(size=layer.forward(x).shape)

        details = numerical_gradient_check(layer, x, dout, epsilon=1e-5,
                                           return_details=True)
        assert details["input"] < 1e-5
        assert details["W"] < 1e-5
        assert details["b"] < 1e-5

    def test_conv2d_scalar_return(self):
        rng = np.random.default_rng(4)
        layer = Conv2D(1, 2, kernel=3, stride=1, padding=0, random_state=0)
        x = rng.normal(size=(1, 1, 5, 5))
        error = numerical_gradient_check(layer, x, epsilon=1e-5)
        assert isinstance(error, float)
        assert error < 1e-5

    def test_maxpool2d(self):
        rng = np.random.default_rng(5)
        layer = MaxPool2D(2, 2)
        x = rng.normal(size=(2, 3, 6, 6))
        dout = rng.normal(size=layer.forward(x).shape)
        assert numerical_gradient_check(layer, x, dout) < 1e-6

    def test_dense(self):
        rng = np.random.default_rng(6)
        layer = Dense(5, 4, random_state=0)
        x = rng.normal(size=(3, 5))
        dout = rng.normal(size=(3, 4))
        details = numerical_gradient_check(layer, x, dout, return_details=True)
        assert details["input"] < 1e-6
        assert details["W"] < 1e-6
        assert details["b"] < 1e-6

    def test_softmax_cross_entropy(self):
        rng = np.random.default_rng(7)
        logits = rng.normal(size=(5, 4))
        y = rng.integers(0, 4, size=5)
        criterion = SoftmaxCrossEntropy()

        loss, grad = criterion.loss_and_grad(logits, y)
        assert np.isscalar(loss)
        assert grad.shape == logits.shape

        # The returned object is float-like and unpackable.
        value = criterion.forward(logits, y)
        assert float(value) == pytest.approx(loss)
        unpacked_loss, unpacked_grad = value
        assert unpacked_loss == pytest.approx(loss)
        np.testing.assert_allclose(unpacked_grad, grad)

        numerical = np.zeros_like(logits)
        eps = 1e-6
        for idx in np.ndindex(logits.shape):
            original = logits[idx]
            logits[idx] = original + eps
            plus = float(criterion.forward(logits, y))
            logits[idx] = original - eps
            minus = float(criterion.forward(logits, y))
            logits[idx] = original
            numerical[idx] = (plus - minus) / (2 * eps)

        assert _relative_error(numerical, grad) < 1e-5

    def test_softmax_is_numerically_stable(self):
        logits = np.array([[1000.0, 1001.0, 999.0]])
        probs = SoftmaxCrossEntropy.softmax(logits)
        assert np.all(np.isfinite(probs))
        assert probs.sum() == pytest.approx(1.0)
        assert np.all(probs >= 0.0)


# ---------------------------------------------------------------------------
# Optimiser and model shapes
# ---------------------------------------------------------------------------
class TestOptimizerAndShapes:
    def test_sgd_momentum_update(self):
        layer = Dense(2, 2, random_state=0)
        layer.params["W"] = np.zeros((2, 2))
        layer.grads["W"] = np.ones((2, 2))
        layer.grads["b"] = np.zeros(2)

        opt = SGD(lr=0.1, momentum=0.9)
        opt.step([layer])
        np.testing.assert_allclose(layer.params["W"], -0.1)
        opt.step([layer])
        # v = 0.9*1 + 1 = 1.9, total update = 0.1 + 0.19
        np.testing.assert_allclose(layer.params["W"], -0.29)

    def test_sequential_output_shapes(self):
        model = Sequential(
            [
                Conv2D(1, 4, 3, 1, 1, random_state=0),
                ReLU(),
                MaxPool2D(2, 2),
                Flatten(),
                Dense(4 * 8 * 8, 3, random_state=1),
            ],
            random_state=0,
        )
        X = np.random.rand(7, 1, 16, 16)
        probs = model.predict_proba(X)
        assert probs.shape == (7, 3)
        np.testing.assert_allclose(probs.sum(axis=1), 1.0)

    def test_shelfcnn_fit_predict(self):
        X_train, X_test, y_train, y_test = _small_dataset(120, 40)
        model = ShelfCNN(n_classes=4, image_size=16, random_state=0)
        history = model.fit(
            X_train, y_train, epochs=2, batch_size=32, lr=0.02, momentum=0.9
        )
        assert len(history) == 2
        assert model.loss_history_ == history
        probs = model.predict_proba(X_test)
        preds = model.predict(X_test)
        assert probs.shape == (len(X_test), 4)
        assert preds.shape == (len(X_test),)
        assert set(np.unique(preds)).issubset({0, 1, 2, 3})
        assert 0.0 <= model.score(X_test, y_test) <= 1.0


# ---------------------------------------------------------------------------
# Synthetic data
# ---------------------------------------------------------------------------
class TestDataGeneration:
    def test_generate_images_legacy(self):
        images, labels = generate_product_images(n_samples=10, random_state=42)
        assert len(images) == 10
        assert len(labels) == 10
        assert images[0].shape == (28, 28)
        assert set(labels).issubset({0, 1})

    def test_generate_shelf_images_shape_and_range(self):
        X, y = generate_shelf_images(n_samples=64, random_state=42)
        assert X.shape == (64, 1, 16, 16)
        assert X.dtype == np.float32
        assert y.shape == (64,)
        assert np.issubdtype(y.dtype, np.integer)
        assert np.all(X >= 0.0) and np.all(X <= 1.0)
        assert np.all(np.isfinite(X))
        assert set(np.unique(y)) == {0, 1, 2, 3}
        assert set(np.bincount(y)) == {16}  # balanced classes

    def test_generate_shelf_images_three_classes(self):
        X, y = generate_shelf_images(n_samples=30, n_classes=3, random_state=1)
        assert X.shape == (30, 1, 16, 16)
        assert set(np.unique(y)) == {0, 1, 2}

    def test_generator_is_deterministic(self):
        X1, y1 = generate_shelf_images(n_samples=32, random_state=7)
        X2, y2 = generate_shelf_images(n_samples=32, random_state=7)
        X3, y3 = generate_shelf_images(n_samples=32, random_state=8)
        np.testing.assert_array_equal(X1, X2)
        np.testing.assert_array_equal(y1, y2)
        assert not np.array_equal(X1, X3)
        assert not np.array_equal(y1, y3)

    def test_shift_images_translates_without_wrapping(self):
        X = np.zeros((1, 1, 5, 5), dtype=np.float32)
        X[0, 0, 2, 2] = 1.0
        shifted = shift_images(X, shift_y=1, shift_x=2, fill=0.0)
        assert shifted.shape == X.shape
        assert shifted[0, 0, 3, 4] == 1.0
        assert shifted.sum() == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# End-to-end training
# ---------------------------------------------------------------------------
class TestTraining:
    @pytest.mark.slow
    def test_training_reduces_loss_and_beats_chance(self):
        X_train, X_test, y_train, y_test = _small_dataset(240, 80, random_state=0)

        model = ShelfCNN(n_classes=4, image_size=16, random_state=42)
        started = time.perf_counter()
        history = model.fit(
            X_train,
            y_train,
            epochs=10,
            batch_size=32,
            lr=0.02,
            momentum=0.9,
        )
        elapsed = time.perf_counter() - started

        assert elapsed < 10.0, f"training took {elapsed:.2f}s (expected < 10s)"
        assert len(history) == 10
        assert history[-1] < history[0]
        # Chance level is 25%; require a wide margin.
        accuracy = model.score(X_test, y_test)
        assert accuracy > 0.75, f"test accuracy only {accuracy:.3f}"

        shifted = shift_images(X_test, shift_y=2, shift_x=2)
        shifted_accuracy = model.score(shifted, y_test)
        assert shifted_accuracy > 0.5, (
            f"shifted test accuracy only {shifted_accuracy:.3f}"
        )


# ---------------------------------------------------------------------------
# Backwards-compatible SimpleCNN
# ---------------------------------------------------------------------------
class TestSimpleCNN:
    def test_initialization(self):
        cnn = SimpleCNN(input_shape=(28, 28), random_state=42)
        assert cnn.input_shape == (28, 28)

    def test_predict(self):
        cnn = SimpleCNN(input_shape=(28, 28), random_state=42)
        image = np.random.randn(28, 28)
        assert cnn.predict(image) in [0, 1]
        proba = cnn.predict_proba(image)
        assert 0.0 <= proba <= 1.0

    def test_gradient_check(self):
        cnn = SimpleCNN(input_shape=(28, 28), random_state=42)
        X = np.random.default_rng(11).normal(size=(28, 28))
        max_diff, rel_diff = gradient_check_cnn(cnn, X, 1.0, epsilon=1e-5)
        assert np.isfinite(max_diff)
        assert rel_diff < 1e-4, f"relative gradient error {rel_diff:.2e}"

    def test_loss_decreases_with_training(self):
        X_train, y_train = generate_product_images(n_samples=30, random_state=42)
        cnn = SimpleCNN(input_shape=(28, 28), learning_rate=0.05, random_state=42)
        cnn.fit(X_train, y_train, epochs=30)
        assert len(cnn.loss_history_) == 30
        assert cnn.loss_history_[-1] < cnn.loss_history_[0]


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
