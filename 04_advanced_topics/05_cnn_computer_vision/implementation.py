"""
Convolutional Neural Networks from Scratch
==========================================

This module builds a small convolutional neural network using only NumPy.  It
is the implementation companion of the ``05_cnn_computer_vision`` lesson and is
written for readability first: every operation used by the network (im2col,
convolution, pooling, ReLU, dense layers, softmax cross-entropy and momentum
SGD) is implemented explicitly.

The running example is an SPG (Sales Promotion Girl) camera pointed at a warung
shelf.  ``generate_shelf_images`` creates tiny synthetic greyscale images of
four product types (box pack, bottle, sachet strip and can).  A small CNN learns
to recognise the product type from the image.

Public API
----------
Core helpers
    ``im2col``, ``col2im``, ``conv2d``, ``max_pool2d``, ``conv2d_backward``
Layers (``forward`` / ``backward``, parameters in ``.params`` / ``.grads``)
    ``Conv2D``, ``MaxPool2D``, ``ReLU``, ``Flatten``, ``Dense``
Loss and optimiser
    ``SoftmaxCrossEntropy``, ``SGD``
Models
    ``Sequential``, ``ShelfCNN``, ``SimpleCNN`` (backwards compatible)
Utilities
    ``numerical_gradient_check``, ``gradient_check_cnn``, ``shift_images``
Data
    ``generate_shelf_images``, ``generate_product_images`` (backwards compatible)

Author: ML-Algo Repository
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from typing import Self, Union

import numpy as np

Array = np.ndarray
IntPair = int | tuple[int, int]


# ---------------------------------------------------------------------------
# Small shape helpers
# ---------------------------------------------------------------------------
def _pair(value: IntPair) -> tuple[int, int]:
    """Return ``value`` as a ``(vertical, horizontal)`` pair."""
    if isinstance(value, (tuple, list)):
        if len(value) != 2:
            raise ValueError("Expected an int or a pair of ints")
        return int(value[0]), int(value[1])
    return int(value), int(value)


def _out_size(size: int, kernel: int, stride: int, padding: int) -> int:
    """Output size of a convolution/pooling window."""
    return (size + 2 * padding - kernel) // stride + 1


def _as_nchw(x: Array) -> tuple[Array, bool]:
    """Convert ``(H, W)`` or ``(H, W, C)`` input to ``(1, C, H, W)``.

    Returns the converted array and a flag telling whether the original input
    was 2-D (used by the backwards-compatible ``conv2d``/``max_pool2d``
    helpers so that they can squeeze their result again).
    """
    x = np.asarray(x)
    if x.ndim == 2:
        return x[np.newaxis, np.newaxis, :, :], True
    if x.ndim == 3:
        return np.transpose(x, (2, 0, 1))[np.newaxis, :, :, :], False
    if x.ndim == 4:
        return x, False
    raise ValueError("Expected an array with 2, 3 or 4 dimensions")


# ---------------------------------------------------------------------------
# im2col / col2im
# ---------------------------------------------------------------------------
def im2col(
    x: Array,
    kernel_size: IntPair,
    stride: IntPair = 1,
    padding: IntPair = 0,
) -> Array:
    """Rearrange image patches into columns (NCHW layout).

    Parameters
    ----------
    x : ndarray of shape (N, C, H, W)
        Input feature maps.
    kernel_size : int or (k_h, k_w)
        Spatial size of each patch.
    stride : int or (s_h, s_w), default=1
        Step between neighbouring patches.
    padding : int or (p_h, p_w), default=0
        Zero padding applied to the height and width.

    Returns
    -------
    cols : ndarray of shape (N * out_h * out_w, C * k_h * k_w)
        Each row contains one flattened patch.  The row order is
        ``n, out_h, out_w`` and the column order is ``c, k_h, k_w``, which is
        exactly the layout expected by :class:`Conv2D`.
    """
    x = np.asarray(x, dtype=float)
    if x.ndim != 4:
        raise ValueError("im2col expects an array of shape (N, C, H, W)")

    kh, kw = _pair(kernel_size)
    sh, sw = _pair(stride)
    ph, pw = _pair(padding)
    if min(kh, kw, sh, sw) <= 0:
        raise ValueError("kernel_size and stride must be positive")
    if min(ph, pw) < 0:
        raise ValueError("padding must be non-negative")

    n, c, h, w = x.shape
    out_h = _out_size(h, kh, sh, ph)
    out_w = _out_size(w, kw, sw, pw)
    if out_h <= 0 or out_w <= 0:
        raise ValueError(
            "The convolution window is larger than the padded input: "
            f"input={(h, w)}, kernel={(kh, kw)}, padding={(ph, pw)}"
        )

    if ph or pw:
        x = np.pad(x, ((0, 0), (0, 0), (ph, ph), (pw, pw)), mode="constant")

    cols = np.empty((n, out_h, out_w, c, kh, kw), dtype=x.dtype)
    for i in range(out_h):
        for j in range(out_w):
            h0 = i * sh
            w0 = j * sw
            cols[:, i, j] = x[:, :, h0:h0 + kh, w0:w0 + kw]
    return cols.reshape(n * out_h * out_w, c * kh * kw)


def col2im(
    cols: Array,
    x_shape: tuple[int, int, int, int],
    kernel_size: IntPair,
    stride: IntPair = 1,
    padding: IntPair = 0,
) -> Array:
    """Scatter columns back to an image (the adjoint of :func:`im2col`).

    Parameters
    ----------
    cols : ndarray of shape (N * out_h * out_w, C * k_h * k_w)
        Patch matrix produced by :func:`im2col` (or a gradient with the same
        shape).
    x_shape : tuple (N, C, H, W)
        Shape of the original, *unpadded* input.
    kernel_size, stride, padding :
        Must match the values used by the corresponding :func:`im2col` call.

    Returns
    -------
    x : ndarray of shape (N, C, H, W)
        Sum of all patches placed back at their original locations.
    """
    cols = np.asarray(cols, dtype=float)
    n, c, h, w = (int(v) for v in x_shape)
    kh, kw = _pair(kernel_size)
    sh, sw = _pair(stride)
    ph, pw = _pair(padding)
    out_h = _out_size(h, kh, sh, ph)
    out_w = _out_size(w, kw, sw, pw)

    expected = (n * out_h * out_w, c * kh * kw)
    if cols.shape != expected:
        raise ValueError(f"cols has shape {cols.shape}, expected {expected}")

    cols = cols.reshape(n, out_h, out_w, c, kh, kw)
    padded = np.zeros((n, c, h + 2 * ph, w + 2 * pw), dtype=cols.dtype)
    for i in range(out_h):
        for j in range(out_w):
            h0 = i * sh
            w0 = j * sw
            padded[:, :, h0:h0 + kh, w0:w0 + kw] += cols[:, i, j]

    if ph or pw:
        return padded[:, :, ph:ph + h, pw:pw + w]
    return padded


# ---------------------------------------------------------------------------
# Layers
# ---------------------------------------------------------------------------
class Layer:
    """Base class for a differentiable layer.

    Subclasses implement :meth:`forward` and :meth:`backward`.  Learnable
    tensors live in ``self.params`` and their gradients in ``self.grads``; both
    are plain dictionaries so that an optimiser can iterate over them without
    knowing anything about the layer internals.
    """

    params: dict[str, Array]
    grads: dict[str, Array]

    def __init__(self) -> None:
        self.params = {}
        self.grads = {}

    def forward(self, x: Array) -> Array:
        """Compute the layer output for a batch ``x``."""
        raise NotImplementedError

    def backward(self, dout: Array) -> Array:
        """Backpropagate ``dout`` and return the gradient w.r.t. the input."""
        raise NotImplementedError


class Conv2D(Layer):
    """2-D convolution implemented with :func:`im2col`.

    The operation is cross-correlation (the usual deep-learning convention):
    for every output location the layer takes the dot product between a patch
    of the padded input and a filter.  Weights have shape
    ``(out_ch, in_ch, k_h, k_w)`` and are initialised with He scaling.

    Parameters
    ----------
    in_ch : int
        Number of input channels.
    out_ch : int
        Number of filters / output channels.
    kernel : int or (k_h, k_w), default=3
        Spatial filter size.
    stride : int or (s_h, s_w), default=1
        Convolution stride.
    padding : int or (p_h, p_w), default=0
        Zero padding.
    random_state : int or None, default=None
        Seed for reproducible He initialisation.

    Attributes
    ----------
    params : dict
        ``{"W": (out_ch, in_ch, k_h, k_w), "b": (out_ch,)}``.
    grads : dict
        Gradients with the same keys and shapes.
    """

    def __init__(
        self,
        in_ch: int,
        out_ch: int,
        kernel: IntPair = 3,
        stride: IntPair = 1,
        padding: IntPair = 0,
        random_state: int | None = None,
    ) -> None:
        super().__init__()
        if in_ch <= 0 or out_ch <= 0:
            raise ValueError("in_ch and out_ch must be positive")
        self.in_ch = int(in_ch)
        self.out_ch = int(out_ch)
        self.kernel = _pair(kernel)
        self.stride = _pair(stride)
        self.padding = _pair(padding)
        if min(self.kernel) <= 0 or min(self.stride) <= 0:
            raise ValueError("kernel and stride must be positive")
        if min(self.padding) < 0:
            raise ValueError("padding must be non-negative")

        rng = np.random.default_rng(random_state)
        kh, kw = self.kernel
        fan_in = self.in_ch * kh * kw
        scale = np.sqrt(2.0 / fan_in)  # He initialisation for ReLU networks
        self.params = {
            "W": rng.normal(0.0, scale, size=(self.out_ch, self.in_ch, kh, kw)),
            "b": np.zeros(self.out_ch, dtype=float),
        }
        self.grads = {
            "W": np.zeros_like(self.params["W"]),
            "b": np.zeros_like(self.params["b"]),
        }
        self._cache: tuple[Array, tuple[int, int, int, int], Array] | None = None

    def out_shape(self, in_shape: tuple[int, int, int, int]) -> tuple[int, int, int, int]:
        """Return the output shape for an ``(N, C, H, W)`` input shape."""
        n, _, h, w = in_shape
        kh, kw = self.kernel
        sh, sw = self.stride
        ph, pw = self.padding
        return (
            n,
            self.out_ch,
            _out_size(h, kh, sh, ph),
            _out_size(w, kw, sw, pw),
        )

    def forward(self, x: Array) -> Array:
        """Convolve a batch ``x`` of shape ``(N, in_ch, H, W)``."""
        x = np.asarray(x, dtype=float)
        if x.ndim != 4:
            raise ValueError("Conv2D expects input of shape (N, C, H, W)")
        if x.shape[1] != self.in_ch:
            raise ValueError(
                f"Conv2D expected {self.in_ch} input channels, got {x.shape[1]}"
            )

        n, _, h, w = x.shape
        kh, kw = self.kernel
        sh, sw = self.stride
        ph, pw = self.padding
        out_h = _out_size(h, kh, sh, ph)
        out_w = _out_size(w, kw, sw, pw)
        if out_h <= 0 or out_w <= 0:
            raise ValueError(
                "Conv2D output would be empty; reduce the kernel or padding"
            )

        cols = im2col(x, self.kernel, self.stride, self.padding)
        w_mat = self.params["W"].reshape(self.out_ch, -1)
        out = cols @ w_mat.T + self.params["b"]
        out = out.reshape(n, out_h, out_w, self.out_ch).transpose(0, 3, 1, 2)
        self._cache = (cols, x.shape, (out_h, out_w))
        return out

    def backward(self, dout: Array) -> Array:
        """Return ``dL/dx`` and store ``dL/dW`` and ``dL/db`` in ``self.grads``."""
        if self._cache is None:
            raise RuntimeError("Call forward before backward")
        cols, x_shape, (out_h, out_w) = self._cache
        n = x_shape[0]

        dout = np.asarray(dout, dtype=float)
        dout_flat = dout.transpose(0, 2, 3, 1).reshape(n * out_h * out_w, self.out_ch)

        w_mat = self.params["W"].reshape(self.out_ch, -1)
        self.grads["W"] = (dout_flat.T @ cols).reshape(self.params["W"].shape)
        self.grads["b"] = dout_flat.sum(axis=0)

        dcols = dout_flat @ w_mat
        return col2im(dcols, x_shape, self.kernel, self.stride, self.padding)


class MaxPool2D(Layer):
    """Max pooling with an exact argmax backward pass.

    Parameters
    ----------
    size : int or (p_h, p_w), default=2
        Pooling window size.
    stride : int or (s_h, s_w) or None, default=None
        Pooling stride.  ``None`` means "same as ``size``" (non-overlapping
        windows).

    Attributes
    ----------
    params, grads : dict
        Empty dictionaries; pooling has no learnable parameters.
    """

    def __init__(self, size: IntPair = 2, stride: IntPair | None = None) -> None:
        super().__init__()
        self.size = _pair(size)
        self.stride = self.size if stride is None else _pair(stride)
        if min(self.size) <= 0 or min(self.stride) <= 0:
            raise ValueError("size and stride must be positive")

    def out_shape(self, in_shape: tuple[int, int, int, int]) -> tuple[int, int, int, int]:
        """Return the output shape for an ``(N, C, H, W)`` input shape."""
        n, c, h, w = in_shape
        ph, pw = self.size
        sh, sw = self.stride
        return (
            n,
            c,
            _out_size(h, ph, sh, 0),
            _out_size(w, pw, sw, 0),
        )

    def forward(self, x: Array) -> Array:
        """Pool a batch ``x`` of shape ``(N, C, H, W)``."""
        x = np.asarray(x, dtype=float)
        if x.ndim != 4:
            raise ValueError("MaxPool2D expects input of shape (N, C, H, W)")
        n, c, h, w = x.shape
        ph, pw = self.size
        sh, sw = self.stride
        out_h = _out_size(h, ph, sh, 0)
        out_w = _out_size(w, pw, sw, 0)
        if out_h <= 0 or out_w <= 0:
            raise ValueError("MaxPool2D output would be empty")

        out = np.empty((n, c, out_h, out_w), dtype=x.dtype)
        arg_h = np.empty((n, c, out_h, out_w), dtype=np.int64)
        arg_w = np.empty((n, c, out_h, out_w), dtype=np.int64)

        for i in range(out_h):
            h0 = i * sh
            for j in range(out_w):
                w0 = j * sw
                window = x[:, :, h0:h0 + ph, w0:w0 + pw]
                flat = window.reshape(n, c, ph * pw)
                idx = np.argmax(flat, axis=2)
                out[:, :, i, j] = np.take_along_axis(
                    flat, idx[:, :, None], axis=2
                )[:, :, 0]
                arg_h[:, :, i, j] = idx // pw
                arg_w[:, :, i, j] = idx % pw

        self._cache = (x.shape, (out_h, out_w), arg_h, arg_w)
        return out

    def backward(self, dout: Array) -> Array:
        """Route each upstream gradient to the position that achieved the max."""
        if self._cache is None:
            raise RuntimeError("Call forward before backward")
        x_shape, (out_h, out_w), arg_h, arg_w = self._cache
        n, c, h, w = x_shape
        dout = np.asarray(dout, dtype=float)

        dx = np.zeros((n, c, h, w), dtype=dout.dtype)
        n_idx = np.arange(n)[:, None]
        c_idx = np.arange(c)[None, :]
        _ph, _pw = self.size
        sh, sw = self.stride

        for i in range(out_h):
            for j in range(out_w):
                rows = i * sh + arg_h[:, :, i, j]
                cols = j * sw + arg_w[:, :, i, j]
                np.add.at(dx, (n_idx, c_idx, rows, cols), dout[:, :, i, j])
        return dx


class ReLU(Layer):
    """Element-wise rectified linear unit: ``max(0, x)``."""

    def __init__(self) -> None:
        super().__init__()
        self._mask: Array | None = None

    def forward(self, x: Array) -> Array:
        """Return ``max(0, x)`` and remember the activation mask."""
        x = np.asarray(x, dtype=float)
        self._mask = x > 0.0
        return np.maximum(0.0, x)

    def backward(self, dout: Array) -> Array:
        """Pass gradients through positions where the input was positive."""
        if self._mask is None:
            raise RuntimeError("Call forward before backward")
        return np.asarray(dout, dtype=float) * self._mask


class Flatten(Layer):
    """Flatten every sample to a vector while remembering the input shape."""

    def __init__(self) -> None:
        super().__init__()
        self._input_shape: tuple[int, ...] | None = None

    def forward(self, x: Array) -> Array:
        """Reshape ``(N, ...)`` to ``(N, prod(...))``."""
        x = np.asarray(x, dtype=float)
        self._input_shape = x.shape
        return x.reshape(x.shape[0], -1)

    def backward(self, dout: Array) -> Array:
        """Reshape a flat gradient back to the original input shape."""
        if self._input_shape is None:
            raise RuntimeError("Call forward before backward")
        return np.asarray(dout, dtype=float).reshape(self._input_shape)


class Dense(Layer):
    """Fully connected layer ``y = x @ W + b``.

    Weights are initialised with He scaling, which pairs well with ReLU.

    Parameters
    ----------
    in_features : int
        Size of the flattened input.
    out_features : int
        Number of output units.
    random_state : int or None, default=None
        Seed for reproducible initialisation.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        random_state: int | None = None,
    ) -> None:
        super().__init__()
        if in_features <= 0 or out_features <= 0:
            raise ValueError("in_features and out_features must be positive")
        self.in_features = int(in_features)
        self.out_features = int(out_features)

        rng = np.random.default_rng(random_state)
        scale = np.sqrt(2.0 / self.in_features)
        self.params = {
            "W": rng.normal(0.0, scale, size=(self.in_features, self.out_features)),
            "b": np.zeros(self.out_features, dtype=float),
        }
        self.grads = {
            "W": np.zeros_like(self.params["W"]),
            "b": np.zeros_like(self.params["b"]),
        }
        self._x: Array | None = None

    def forward(self, x: Array) -> Array:
        """Apply the affine transformation to a ``(N, in_features)`` batch."""
        x = np.asarray(x, dtype=float)
        if x.ndim != 2 or x.shape[1] != self.in_features:
            raise ValueError(
                f"Dense expected shape (N, {self.in_features}), got {x.shape}"
            )
        self._x = x
        return x @ self.params["W"] + self.params["b"]

    def backward(self, dout: Array) -> Array:
        """Return ``dL/dx`` and store ``dL/dW`` / ``dL/db`` in ``self.grads``."""
        if self._x is None:
            raise RuntimeError("Call forward before backward")
        dout = np.asarray(dout, dtype=float)
        self.grads["W"] = self._x.T @ dout
        self.grads["b"] = dout.sum(axis=0)
        return dout @ self.params["W"].T


# ---------------------------------------------------------------------------
# Loss
# ---------------------------------------------------------------------------
class _LossValue(float):
    """A loss value that also unpacks as ``(loss, gradient)``.

    ``SoftmaxCrossEntropy.forward`` returns this small float subclass so that
    both common calling styles work::

        loss = criterion.forward(logits, y)          # float-like
        loss, grad = criterion.forward(logits, y)     # tuple-style

    The gradient is available as ``loss.grad`` as well.
    """

    __slots__ = ("grad",)

    def __new__(cls, value: float, grad: Array) -> Self:
        obj = float.__new__(cls, value)
        obj.grad = grad
        return obj

    def __iter__(self):
        yield float(self)
        yield self.grad


class SoftmaxCrossEntropy:
    """Numerically stable softmax + cross-entropy loss.

    For a batch of logits ``z`` and integer labels ``y`` the loss is::

        L = -(1/N) * sum_i log( softmax(z_i)[y_i] )

    The gradient with respect to the logits has the compact form
    ``(softmax(z) - one_hot(y)) / N``.  Subtracting the per-sample maximum
    logit before exponentiation prevents overflow.
    """

    def __init__(self) -> None:
        self._grad: Array | None = None
        self.probs_: Array | None = None
        self.loss_: float | None = None

    @property
    def grad(self) -> Array:
        """The gradient stored by the most recent forward call."""
        if self._grad is None:
            raise RuntimeError("Call forward first")
        return self._grad

    @staticmethod
    def softmax(logits: Array) -> Array:
        """Return a numerically stable softmax over the last axis."""
        logits = np.asarray(logits, dtype=float)
        shifted = logits - np.max(logits, axis=1, keepdims=True)
        exp = np.exp(shifted)
        return exp / np.sum(exp, axis=1, keepdims=True)

    def forward(self, logits: Array, y: Array) -> _LossValue:
        """Compute the mean cross-entropy and remember its gradient.

        Parameters
        ----------
        logits : ndarray of shape (N, n_classes)
            Unnormalised class scores.
        y : ndarray of shape (N,)
            Integer class indices in ``[0, n_classes)``.
        """
        logits = np.asarray(logits, dtype=float)
        y = np.asarray(y, dtype=np.int64).ravel()
        if logits.ndim != 2:
            raise ValueError("logits must have shape (N, n_classes)")
        if y.shape[0] != logits.shape[0]:
            raise ValueError("logits and y must contain the same number of samples")
        if np.any(y < 0) or np.any(y >= logits.shape[1]):
            raise ValueError("y contains a class index outside the logits")

        probs = self.softmax(logits)
        n = logits.shape[0]
        loss = -np.mean(np.log(probs[np.arange(n), y] + 1e-12))

        grad = probs.copy()
        grad[np.arange(n), y] -= 1.0
        grad /= n
        self.probs_ = probs
        self._grad = grad
        self.loss_ = float(loss)
        return _LossValue(float(loss), grad)

    def backward(self, dout: Array | None = None) -> Array:
        """Return the stored ``dL/dlogits`` (already averaged over the batch)."""
        if self._grad is None:
            raise RuntimeError("Call forward before backward")
        if dout is None:
            return self._grad
        return self._grad * np.asarray(dout, dtype=float)

    def loss_and_grad(self, logits: Array, y: Array) -> tuple[float, Array]:
        """Return ``(loss, grad)`` as plain Python/NumPy objects."""
        value = self.forward(logits, y)
        return float(value), self._grad

    def __call__(self, logits: Array, y: Array) -> _LossValue:
        """Alias for :meth:`forward`."""
        return self.forward(logits, y)


# ---------------------------------------------------------------------------
# Optimiser
# ---------------------------------------------------------------------------
class SGD:
    """Stochastic gradient descent with optional momentum.

    The update for a parameter ``p`` with gradient ``g`` is::

        v <- momentum * v + g
        p <- p - lr * v

    With ``momentum=0`` this reduces to plain SGD.  The optimiser stores one
    velocity buffer per ``(layer, parameter name)`` pair.

    Parameters
    ----------
    lr : float, default=0.01
        Learning rate.
    momentum : float, default=0.9
        Momentum coefficient in ``[0, 1)``.
    """

    def __init__(self, lr: float = 0.01, momentum: float = 0.9) -> None:
        if lr <= 0:
            raise ValueError("lr must be positive")
        if not 0.0 <= momentum < 1.0:
            raise ValueError("momentum must be in [0, 1)")
        self.lr = float(lr)
        self.momentum = float(momentum)
        self.velocity: dict[tuple[int, str], Array] = {}

    def step(self, layers: Iterable[Layer]) -> None:
        """Apply one update to every parameter of ``layers``."""
        for layer in layers:
            for name, param in layer.params.items():
                grad = layer.grads.get(name)
                if grad is None:
                    continue
                key = (id(layer), name)
                if self.momentum:
                    if key not in self.velocity:
                        self.velocity[key] = np.zeros_like(param)
                    self.velocity[key] = (
                        self.momentum * self.velocity[key] + grad
                    )
                    param -= self.lr * self.velocity[key]
                else:
                    param -= self.lr * grad

    def zero_velocity(self) -> None:
        """Forget all momentum buffers."""
        self.velocity.clear()


# ---------------------------------------------------------------------------
# Sequential model
# ---------------------------------------------------------------------------
class Sequential:
    """A minimal feed-forward CNN made of a list of layers.

    Parameters
    ----------
    layers : sequence of Layer
        Layers applied in order.
    loss_fn : SoftmaxCrossEntropy or None
        Loss used by :meth:`fit`.  A new ``SoftmaxCrossEntropy`` is created
        when omitted.
    random_state : int or None, default=None
        Seed used to shuffle mini-batches during training.
    """

    def __init__(
        self,
        layers: Sequence[Layer],
        loss_fn: SoftmaxCrossEntropy | None = None,
        random_state: int | None = None,
    ) -> None:
        self.layers: list[Layer] = list(layers)
        self.loss_fn = loss_fn if loss_fn is not None else SoftmaxCrossEntropy()
        self.random_state = random_state
        self.loss_history_: list[float] = []
        self.classes_: Array | None = None

    # -- forward / backward -------------------------------------------------
    def _forward(self, x: Array) -> Array:
        for layer in self.layers:
            x = layer.forward(x)
        return x

    def forward(self, x: Array) -> Array:
        """Return raw logits for a batch of inputs."""
        return self._forward(x)

    def _backward(self, dout: Array) -> Array:
        for layer in reversed(self.layers):
            dout = layer.backward(dout)
        return dout

    # -- inference ----------------------------------------------------------
    def predict_proba(self, X: Array) -> Array:
        """Return softmax class probabilities of shape ``(N, n_classes)``."""
        logits = self._forward(self._prepare_images(X))
        return SoftmaxCrossEntropy.softmax(logits)

    def predict(self, X: Array) -> Array:
        """Return predicted class labels (original label values)."""
        if self.classes_ is None:
            raise RuntimeError("Call fit before predict")
        proba = self.predict_proba(X)
        return self.classes_[np.argmax(proba, axis=1)]

    def score(self, X: Array, y: Array) -> float:
        """Return the mean accuracy on ``(X, y)``."""
        y = np.asarray(y).ravel()
        return float(np.mean(self.predict(X) == y))

    # -- training -----------------------------------------------------------
    def _prepare_images(self, X: Array) -> Array:
        """Convert accepted image layouts to float ``(N, C, H, W)``."""
        X = np.asarray(X, dtype=float)
        if X.ndim == 2:
            X = X[np.newaxis, np.newaxis, :, :]
        elif X.ndim == 3:
            # Accept both (N, H, W) and (C, H, W).
            if getattr(self, "image_size", None) is not None:
                h, w = self.image_size
                if X.shape[1:] != (h, w):
                    raise ValueError(f"Could not interpret input shape {X.shape}")
                if X.shape[0] == self.in_channels and X.shape[0] in (1, 3, 4):
                    # A single (C, H, W) image.
                    X = X[np.newaxis, :, :, :]
                else:
                    # A batch of (N, H, W) greyscale images.
                    X = X[:, np.newaxis, :, :]
            else:
                X = X[:, np.newaxis, :, :]
        elif X.ndim != 4:
            raise ValueError("X must have 2, 3 or 4 dimensions")
        return X.astype(float)

    def fit(
        self,
        X: Array,
        y: Array,
        epochs: int = 10,
        batch_size: int = 32,
        lr: float = 0.01,
        momentum: float = 0.9,
        verbose: int = 0,
    ) -> list[float]:
        """Train the network with mini-batch momentum SGD.

        Parameters
        ----------
        X : array-like
            Images with shape ``(N, C, H, W)``, ``(N, H, W)`` or a single
            ``(H, W)`` image.
        y : array-like
            Labels with shape ``(N,)``.  Any label values are accepted; they
            are mapped to contiguous class indices internally.
        epochs : int, default=10
            Number of passes over the training set.
        batch_size : int, default=32
            Mini-batch size.
        lr : float, default=0.01
            Learning rate.
        momentum : float, default=0.9
            Momentum coefficient.
        verbose : int, default=0
            Print one line per epoch when greater than zero.

        Returns
        -------
        history : list of float
            Mean training loss for every epoch.  The same list is stored in
            ``self.loss_history_``.
        """
        if epochs < 1:
            raise ValueError("epochs must be at least 1")
        if batch_size < 1:
            raise ValueError("batch_size must be at least 1")

        X = self._prepare_images(X)
        y = np.asarray(y).ravel()
        if X.shape[0] != y.shape[0]:
            raise ValueError("X and y must contain the same number of samples")

        self.classes_ = np.unique(y)
        y_idx = np.searchsorted(self.classes_, y).astype(np.int64)

        optimizer = SGD(lr=lr, momentum=momentum)
        rng = np.random.default_rng(self.random_state)
        n = X.shape[0]
        self.loss_history_ = []

        for epoch in range(epochs):
            order = rng.permutation(n)
            epoch_loss = 0.0
            seen = 0
            for start in range(0, n, batch_size):
                batch = order[start:start + batch_size]
                logits = self._forward(X[batch])
                loss_value = self.loss_fn.forward(logits, y_idx[batch])
                self._backward(self.loss_fn.backward())
                optimizer.step(self.layers)

                epoch_loss += float(loss_value) * len(batch)
                seen += len(batch)

            mean_loss = epoch_loss / max(seen, 1)
            self.loss_history_.append(mean_loss)
            if verbose:
                print(f"Epoch {epoch + 1:>3}/{epochs} - loss: {mean_loss:.5f}")

        return self.loss_history_


class ShelfCNN(Sequential):
    """Small CNN for the synthetic warung-shelf product images.

    Architecture (default 16x16 input)::

        Conv2D(1 -> 8, 3x3, padding=1) -> ReLU -> MaxPool2D(2)   # 8x8
        Conv2D(8 -> 16, 3x3, padding=1) -> ReLU -> MaxPool2D(2)  # 4x4
        Flatten -> Dense(256 -> 64) -> ReLU -> Dense(64 -> n_classes)

    The two ``padding=1`` convolutions keep the spatial size unchanged before
    every pooling layer.  All weights use He initialisation.

    Parameters
    ----------
    n_classes : int, default=4
        Number of product classes.
    image_size : int, default=16
        Height and width of the square input images.
    in_channels : int, default=1
        Number of input channels (1 for greyscale).
    random_state : int or None, default=42
        Seed used for weight initialisation and mini-batch shuffling.
    """

    def __init__(
        self,
        n_classes: int = 4,
        image_size: int = 16,
        in_channels: int = 1,
        random_state: int | None = 42,
    ) -> None:
        if n_classes < 2:
            raise ValueError("n_classes must be at least 2")
        if image_size < 8:
            raise ValueError("image_size must be at least 8")

        self.n_classes = int(n_classes)
        self.image_size = (int(image_size), int(image_size))
        self.input_shape = self.image_size
        self.in_channels = int(in_channels)
        self.random_state = random_state

        seed = 0 if random_state is None else int(random_state)
        h = image_size
        layers: list[Layer] = [
            Conv2D(in_channels, 8, kernel=3, stride=1, padding=1, random_state=seed),
            ReLU(),
            MaxPool2D(2, 2),
            Conv2D(8, 16, kernel=3, stride=1, padding=1, random_state=seed + 1),
            ReLU(),
            MaxPool2D(2, 2),
        ]
        h //= 2
        h //= 2
        flat = 16 * h * h
        layers.extend(
            [
                Flatten(),
                Dense(flat, 64, random_state=seed + 2),
                ReLU(),
                Dense(64, n_classes, random_state=seed + 3),
            ]
        )
        super().__init__(layers, random_state=random_state)

    def summary(self) -> str:
        """Return a compact human-readable architecture summary."""
        lines = [f"ShelfCNN: {self.n_classes} classes, input {self.image_size}"]
        for i, layer in enumerate(self.layers):
            lines.append(f"  [{i:>2}] {layer.__class__.__name__}")
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Gradient checking
# ---------------------------------------------------------------------------
def _relative_error(numerical: Array, analytical: Array) -> float:
    """Element-wise relative error with a safe denominator."""
    numerical = np.asarray(numerical, dtype=float)
    analytical = np.asarray(analytical, dtype=float)
    denom = np.maximum(np.abs(numerical) + np.abs(analytical), 1e-12)
    return float(np.max(np.abs(numerical - analytical) / denom))


def numerical_gradient_check(
    layer: Layer,
    x: Array,
    dout: Array | None = None,
    epsilon: float = 1e-5,
    max_checks: int | None = None,
    return_details: bool = False,
) -> float | dict[str, float]:
    """Compare a layer's analytical gradients with central differences.

    The scalar objective is ``sum(forward(x) * dout)``.  When ``dout`` is
    omitted it defaults to all ones, so the objective is simply the sum of the
    outputs.  Both the gradient with respect to the input and the gradients in
    ``layer.grads`` are checked.

    Parameters
    ----------
    layer : Layer
        Layer to check.  It must implement ``forward`` and ``backward``.
    x : ndarray
        A single input batch.
    dout : ndarray or None, default=None
        Upstream gradient defining the scalar objective.  Defaults to ones.
    epsilon : float, default=1e-5
        Central-difference step.
    max_checks : int or None, default=None
        Optional cap on the number of numerical evaluations per tensor.
    return_details : bool, default=False
        If true, return a dictionary with the maximum relative error for the
        input and for every parameter instead of a single float.

    Returns
    -------
    error : float or dict
        Maximum relative error (``float``), or a dictionary of errors when
        ``return_details=True``.
    """
    if epsilon <= 0:
        raise ValueError("epsilon must be positive")

    x = np.asarray(x, dtype=np.float64)
    original_dtypes = {name: p.dtype for name, p in layer.params.items()}
    for name in layer.params:
        layer.params[name] = layer.params[name].astype(np.float64)

    try:
        out = layer.forward(x)
        dout = np.ones_like(out, dtype=np.float64) if dout is None else np.asarray(
            dout, dtype=np.float64
        )
        if dout.shape != out.shape:
            raise ValueError("dout must have the same shape as the layer output")

        analytical_input = np.asarray(layer.backward(dout), dtype=np.float64)
        analytical_params = {k: np.array(v, dtype=np.float64) for k, v in layer.grads.items()}

        def objective() -> float:
            return float(np.sum(layer.forward(x) * dout))

        details: dict[str, float] = {}

        # Numerical gradient with respect to the input.
        numerical_input = np.zeros_like(x, dtype=np.float64)
        count = 0
        for idx in np.ndindex(*x.shape):
            if max_checks is not None and count >= max_checks:
                break
            original = x[idx]
            x[idx] = original + epsilon
            plus = objective()
            x[idx] = original - epsilon
            minus = objective()
            x[idx] = original
            numerical_input[idx] = (plus - minus) / (2.0 * epsilon)
            count += 1  # noqa: SIM113
        details["input"] = _relative_error(numerical_input, analytical_input)

        # Numerical gradients with respect to every learnable parameter.
        for name, param in layer.params.items():
            analytical = analytical_params.get(name)
            if analytical is None:
                continue
            numerical = np.zeros_like(param, dtype=np.float64)
            count = 0
            for idx in np.ndindex(*param.shape):
                if max_checks is not None and count >= max_checks:
                    break
                original = param[idx]
                param[idx] = original + epsilon
                plus = objective()
                param[idx] = original - epsilon
                minus = objective()
                param[idx] = original
                numerical[idx] = (plus - minus) / (2.0 * epsilon)
                count += 1  # noqa: SIM113
            details[name] = _relative_error(numerical, analytical)

        if return_details:
            return details
        return float(max(details.values(), default=0.0))
    finally:
        for name, dtype in original_dtypes.items():
            layer.params[name] = layer.params[name].astype(dtype)


# ---------------------------------------------------------------------------
# Backwards-compatible functional helpers
# ---------------------------------------------------------------------------
def _parse_kernel(kernel: Array, in_ch: int) -> tuple[Array, bool]:
    """Convert an old/new kernel layout to ``(F, C, k_h, k_w)``.

    Returns the converted kernel and whether the original input was 2-D.
    """
    kernel = np.asarray(kernel, dtype=float)
    if kernel.ndim == 2:
        return kernel[np.newaxis, np.newaxis, :, :], True
    if kernel.ndim == 3:
        # (C, k_h, k_w) when the first axis matches the input channels,
        # otherwise treat it as (k_h, k_w, F) for a single-channel image.
        if kernel.shape[0] == in_ch:
            return kernel[np.newaxis, :, :, :], False
        return kernel.transpose(2, 0, 1)[:, np.newaxis, :, :], False
    if kernel.ndim == 4:
        # Legacy layout is (k_h, k_w, C, F); also accept (F, C, k_h, k_w).
        if kernel.shape[2] == in_ch:
            return np.transpose(kernel, (3, 2, 0, 1)), False
        if kernel.shape[1] == in_ch:
            return kernel, False
        raise ValueError("Could not match the kernel channels to the input")
    raise ValueError("kernel must have 2, 3 or 4 dimensions")


def conv2d(
    image: Array,
    kernel: Array,
    stride: int = 1,
    padding: int = 0,
) -> Array:
    """Functional 2-D convolution (cross-correlation).

    This helper keeps the original module API.  It accepts a single image with
    shape ``(H, W)`` or ``(H, W, C)`` and a kernel with shape ``(k_h, k_w)`` or
    ``(k_h, k_w, C, F)``.  Four-dimensional ``(N, C, H, W)`` inputs are also
    accepted as a convenience.  The output is squeezed when the input was a
    single-channel image, matching the historical behaviour.
    """
    image_arr = np.asarray(image, dtype=float)
    was_2d = image_arr.ndim == 2
    if image_arr.ndim == 4:
        x = image_arr
        nchw = False
    else:
        x, _ = _as_nchw(image_arr)
        nchw = image_arr.ndim == 3
    n, c, h, w = x.shape
    filters, _ = _parse_kernel(kernel, c)
    f, kc, kh, kw = filters.shape
    if kc != c:
        raise ValueError(
            f"Kernel expects {kc} input channels but the image has {c}"
        )

    out_h = _out_size(h, kh, stride, padding)
    out_w = _out_size(w, kw, stride, padding)
    if out_h <= 0 or out_w <= 0:
        raise ValueError("Convolution output would be empty")

    xp = np.pad(x, ((0, 0), (0, 0), (padding, padding), (padding, padding)))
    out = np.empty((n, f, out_h, out_w), dtype=float)
    for i in range(out_h):
        for j in range(out_w):
            patch = xp[:, :, i * stride:i * stride + kh, j * stride:j * stride + kw]
            out[:, :, i, j] = np.tensordot(
                patch, filters, axes=([1, 2, 3], [1, 2, 3])
            )

    if image_arr.ndim == 4:
        return out
    if nchw:
        result = out.transpose(0, 2, 3, 1)[0]
    else:
        result = out.transpose(0, 2, 3, 1)
    return result.squeeze() if (was_2d or f == 1) else result


def max_pool2d(
    image: Array,
    pool_size: int = 2,
    stride: int | None = None,
) -> Array:
    """Functional max pooling.

    Accepts ``(H, W)``, ``(H, W, C)`` or ``(N, C, H, W)`` arrays.  A 2-D input
    is squeezed back to 2-D, preserving the original module API.
    """
    image_arr = np.asarray(image, dtype=float)
    if image_arr.ndim == 4:
        x = image_arr
        channels_first = True
    else:
        x, _ = _as_nchw(image_arr)
        channels_first = image_arr.ndim == 3

    layer = MaxPool2D(pool_size, stride)
    out = layer.forward(x)
    if image_arr.ndim == 4:
        return out
    if channels_first:
        return out.transpose(0, 2, 3, 1)[0]
    return out[0, 0]


def conv2d_backward(
    dL_dout: Array,
    input_cache: Array,
    kernel: Array,
    stride: int = 1,
    padding: int = 0,
) -> tuple[Array, Array]:
    """Backward pass for :func:`conv2d`.

    Parameters
    ----------
    dL_dout : ndarray
        Gradient with respect to the convolution output, shape ``(out_h,
        out_w)``, ``(out_h, out_w, F)`` or ``(N, F, out_h, out_w)``.
    input_cache : ndarray
        The original input image(s).
    kernel : ndarray
        The kernel used in the forward pass.
    stride, padding : int
        Same values used in the forward pass.

    Returns
    -------
    dL_dinput : ndarray
        Gradient with respect to the input, in the original layout.
    dL_dkernel : ndarray
        Gradient with respect to the kernel, in the original layout.
    """
    input_arr = np.asarray(input_cache, dtype=float)
    was_2d = input_arr.ndim == 2
    if input_arr.ndim == 4:
        x = input_arr
    else:
        x, _ = _as_nchw(input_arr)
    n, c, h, w = x.shape

    kernel_4d, _ = _parse_kernel(kernel, c)
    f, kc, kh, kw = kernel_4d.shape
    if kc != c:
        raise ValueError("Kernel and input channels do not match")

    dout = np.asarray(dL_dout, dtype=float)
    if dout.ndim == 2:
        dout = dout[np.newaxis, np.newaxis, :, :]
    elif dout.ndim == 3:
        dout = dout.transpose(2, 0, 1)[np.newaxis, :, :, :]
    if dout.shape[0] != n or dout.shape[1] != f:
        raise ValueError("dL_dout has an incompatible shape")

    out_h = _out_size(h, kh, stride, padding)
    out_w = _out_size(w, kw, stride, padding)
    cols = im2col(x, (kh, kw), stride, padding)
    dout_flat = dout.transpose(0, 2, 3, 1).reshape(n * out_h * out_w, f)
    dkernel = (dout_flat.T @ cols).reshape(kernel_4d.shape)
    dcols = dout_flat @ kernel_4d.reshape(f, -1)
    dinput = col2im(dcols, x.shape, (kh, kw), stride, padding)

    # Return gradients in the same layout the user provided.
    if input_arr.ndim == 4:
        dinput_out = dinput
    else:
        dinput_out = dinput.transpose(0, 2, 3, 1)
        if was_2d or (input_arr.ndim == 2):
            dinput_out = dinput_out[0, :, :, 0]
        elif input_arr.ndim == 3:
            dinput_out = dinput_out[0]

    original_kernel = np.asarray(kernel)
    if original_kernel.ndim == 2:
        dkernel_out = dkernel[0, 0]
    elif original_kernel.ndim == 3:
        if original_kernel.shape[0] == c:
            dkernel_out = dkernel[0]
        else:
            dkernel_out = dkernel[0].transpose(1, 2, 0)
    else:
        # Legacy (k_h, k_w, C, F) layout.
        dkernel_out = dkernel.transpose(2, 3, 1, 0)
    return dinput_out, dkernel_out


# ---------------------------------------------------------------------------
# Synthetic data: warung shelf product packs
# ---------------------------------------------------------------------------
PRODUCT_CLASS_NAMES = ("box_pack", "bottle", "sachet_strip", "can")


def _draw_box(img: Array, rng: np.random.Generator) -> None:
    """Draw a filled box pack (a rectangle with a label band)."""
    h, w = img.shape
    bw = int(rng.integers(6, min(9, w - 2) + 1))
    bh = int(rng.integers(6, min(9, h - 2) + 1))
    x0 = int(rng.integers(1, w - bw))
    y0 = int(rng.integers(1, h - bh))
    intensity = float(rng.uniform(0.68, 0.95))
    img[y0:y0 + bh, x0:x0 + bw] = intensity
    # A darker label band gives the pack some internal structure.
    band = y0 + max(1, bh // 2)
    img[band:band + 1, x0 + 1:x0 + bw - 1] *= 0.55


def _draw_bottle(img: Array, rng: np.random.Generator) -> None:
    """Draw a tall bottle: a thin body plus a short cap on top."""
    h, w = img.shape
    bw = int(rng.integers(3, min(4, w - 2) + 1))
    body_h = int(rng.integers(8, max(9, min(12, h - 3)) + 1))
    cap_h = 2
    cap_w = max(2, bw - 1)
    total_h = body_h + cap_h
    x0 = int(rng.integers(1, w - bw))
    y0 = int(rng.integers(1, max(2, h - total_h)))
    img[y0:y0 + cap_h, x0 + (bw - cap_w) // 2:x0 + (bw - cap_w) // 2 + cap_w] = (
        float(rng.uniform(0.45, 0.65))
    )
    body = float(rng.uniform(0.70, 0.92))
    img[y0 + cap_h:y0 + total_h, x0:x0 + bw] = body
    # Vertical highlight on the bottle body.
    highlight = x0 + max(0, bw // 2)
    img[y0 + cap_h + 1:y0 + total_h - 1, highlight:highlight + 1] = min(
        1.0, body + 0.10
    )


def _draw_sachet(img: Array, rng: np.random.Generator) -> None:
    """Draw a sachet strip: several horizontal bars separated by gaps."""
    h, w = img.shape
    bar_w = int(rng.integers(8, min(10, w - 2) + 1))
    bar_h = int(rng.integers(2, 3) + 1)
    n_bars = 3
    gap = 1
    total_h = n_bars * bar_h + (n_bars - 1) * gap
    x0 = int(rng.integers(1, w - bar_w))
    y0 = int(rng.integers(1, max(2, h - total_h)))
    base = float(rng.uniform(0.55, 0.75))
    for b in range(n_bars):
        y = y0 + b * (bar_h + gap)
        intensity = min(1.0, base + 0.12 * b + float(rng.uniform(-0.03, 0.03)))
        img[y:y + bar_h, x0:x0 + bar_w] = intensity


def _draw_can(img: Array, rng: np.random.Generator) -> None:
    """Draw a filled circle with a lighter rim, representing a can."""
    h, w = img.shape
    r = int(rng.integers(3, min(4, (h - 2) // 2) + 1))
    cx = int(rng.integers(r + 1, w - r - 1))
    cy = int(rng.integers(r + 1, h - r - 1))
    yy, xx = np.ogrid[:h, :w]
    dist2 = (yy - cy) ** 2 + (xx - cx) ** 2
    body = float(rng.uniform(0.68, 0.92))
    img[dist2 <= r * r] = body
    # Top rim and a small highlight.
    img[cy - r:cy - r + 1, cx - r + 1:cx + r] = min(1.0, body + 0.12)
    img[cy - 1:cy + 1, cx - r + 1:cx - r + 2] = min(1.0, body + 0.15)


_DRAWERS = (_draw_box, _draw_bottle, _draw_sachet, _draw_can)


def generate_shelf_images(
    n_samples: int = 300,
    image_size: int = 16,
    n_classes: int = 4,
    noise: float = 0.05,
    random_state: int | None = None,
) -> tuple[Array, Array]:
    """Generate synthetic warung-shelf product images.

    Each image is a tiny greyscale crop from an SPG camera mounted in front of
    a warung shelf.  The four classes are simple shapes:

    =====  ==========================================
    Class  Product type  Shape
    =====  ==========================================
    0      box pack      filled rectangle with a label band
    1      bottle        tall thin rectangle with a cap
    2      sachet strip  three horizontal bars
    3      can           filled circle with a rim
    =====  ==========================================

    Position, size and brightness are jittered, then Gaussian noise is added.
    The data is entirely synthetic; no real product images are used.

    Parameters
    ----------
    n_samples : int, default=300
        Number of images to generate.
    image_size : int, default=16
        Height and width of the square images.
    n_classes : int, default=4
        Number of classes, either 3 (no can) or 4.
    noise : float, default=0.05
        Standard deviation of the additive Gaussian pixel noise.
    random_state : int or None, default=None
        Seed for reproducibility.

    Returns
    -------
    X : ndarray of shape (n_samples, 1, image_size, image_size), float32
        Greyscale images with values in ``[0, 1]``.
    y : ndarray of shape (n_samples,), int
        Class labels in ``[0, n_classes)``.
    """
    if n_samples < 1:
        raise ValueError("n_samples must be at least 1")
    if image_size < 8:
        raise ValueError("image_size must be at least 8")
    if n_classes not in (3, 4):
        raise ValueError("n_classes must be 3 or 4")
    if noise < 0:
        raise ValueError("noise must be non-negative")

    rng = np.random.default_rng(random_state)
    y = np.arange(n_samples, dtype=int) % n_classes
    rng.shuffle(y)

    X = np.zeros((n_samples, 1, image_size, image_size), dtype=np.float32)
    for i, label in enumerate(y):
        # Shelf background: dark grey with a subtle top-to-bottom gradient.
        img = np.full((image_size, image_size), rng.uniform(0.12, 0.22))
        img += np.linspace(-0.03, 0.03, image_size)[:, None]
        _DRAWERS[int(label)](img, rng)
        img *= rng.uniform(0.92, 1.08)
        img += rng.normal(0.0, noise, size=img.shape)
        X[i, 0] = np.clip(img, 0.0, 1.0)

    return X, y


def shift_images(
    X: Array,
    shift_y: int = 0,
    shift_x: int = 0,
    fill: float = 0.15,
) -> Array:
    """Translate images without wrapping (used for the inductive-bias test).

    Parameters
    ----------
    X : ndarray of shape (N, C, H, W), (N, H, W) or (H, W)
        Images to shift.
    shift_y, shift_x : int, default=0
        Number of pixels to move down and right.  Negative values move up and
        left.  Vacated pixels are filled with ``fill``.
    fill : float, default=0.15
        Background intensity used for the vacated border.

    Returns
    -------
    shifted : ndarray
        A new array with the same shape as ``X``.
    """
    X = np.asarray(X)
    if X.ndim == 2:
        return shift_images(X[None, None], shift_y, shift_x, fill)[0, 0]
    if X.ndim == 3:
        return shift_images(X[:, None], shift_y, shift_x, fill)[:, 0]
    if X.ndim != 4:
        raise ValueError("X must have 2, 3 or 4 dimensions")

    out = np.full_like(X, fill, dtype=np.float32)
    h, w = X.shape[-2:]
    if abs(shift_y) >= h or abs(shift_x) >= w:
        return out

    src_y0 = max(0, -shift_y)
    src_y1 = min(h, h - shift_y)
    dst_y0 = max(0, shift_y)
    dst_y1 = min(h, h + shift_y)
    src_x0 = max(0, -shift_x)
    src_x1 = min(w, w - shift_x)
    dst_x0 = max(0, shift_x)
    dst_x1 = min(w, w + shift_x)

    out[..., dst_y0:dst_y1, dst_x0:dst_x1] = X[
        ..., src_y0:src_y1, src_x0:src_x1
    ]
    return out


# ---------------------------------------------------------------------------
# Backwards-compatible binary CNN and data generator
# ---------------------------------------------------------------------------
class SimpleCNN:
    """Binary CNN kept for backwards compatibility with the original module.

    Architecture::

        Conv2D -> ReLU -> MaxPool2D -> Flatten -> Dense -> Sigmoid

    The original class only trained the dense layer.  This version trains every
    layer with a correct backward pass, while keeping the old public methods
    ``forward``, ``predict``, ``predict_proba``, ``train_step`` and ``fit``.
    """

    def __init__(
        self,
        input_shape: tuple[int, int],
        n_filters: int = 8,
        kernel_size: int = 3,
        pool_size: int = 2,
        learning_rate: float = 0.01,
        random_state: int | None = None,
    ) -> None:
        self.input_shape = tuple(int(v) for v in input_shape)
        self.n_filters = int(n_filters)
        self.kernel_size = int(kernel_size)
        self.pool_size = int(pool_size)
        self.learning_rate = float(learning_rate)
        self.random_state = random_state

        seed = 0 if random_state is None else int(random_state)
        conv_h = self.input_shape[0] - self.kernel_size + 1
        conv_w = self.input_shape[1] - self.kernel_size + 1
        pool_h = conv_h // self.pool_size
        pool_w = conv_w // self.pool_size
        self.dense_input_size = pool_h * pool_w * self.n_filters

        self.layers: list[Layer] = [
            Conv2D(1, self.n_filters, self.kernel_size, 1, 0, random_state=seed),
            ReLU(),
            MaxPool2D(self.pool_size, self.pool_size),
            Flatten(),
            Dense(self.dense_input_size, 1, random_state=seed + 1),
        ]
        self.optimizer = SGD(lr=self.learning_rate, momentum=0.9)
        self.loss_history_: list[float] = []

    # -- helpers ------------------------------------------------------------
    @staticmethod
    def _single(x: Array) -> Array:
        """Convert a single image to ``(1, 1, H, W)``."""
        x = np.asarray(x, dtype=float)
        if x.ndim == 2:
            return x[np.newaxis, np.newaxis, :, :]
        if x.ndim == 3:
            return x[np.newaxis, :, :, :]
        if x.ndim == 4:
            return x
        raise ValueError("Expected an image with 2, 3 or 4 dimensions")

    @staticmethod
    def _sigmoid(z: Array) -> Array:
        z = np.clip(z, -500.0, 500.0)
        return 1.0 / (1.0 + np.exp(-z))

    # -- forward / training -------------------------------------------------
    def forward(self, X: Array) -> tuple[Array, dict]:
        """Run a single image and return ``(probability, cache)``."""
        x = self._single(X)
        h = x
        for layer in self.layers:
            h = layer.forward(h)
        output = self._sigmoid(h)
        return output, {"logits": h, "input": x}

    def _loss_and_grads_batch(self, X: Array, y: Array) -> float:
        """Return mean binary cross-entropy and fill all parameter gradients."""
        X = np.asarray(X, dtype=float)
        if X.ndim == 2:
            X = X[np.newaxis, np.newaxis, :, :]
        elif X.ndim == 3:
            X = X[:, np.newaxis, :, :]
        y = np.asarray(y, dtype=float).reshape(-1, 1)
        if y.shape[0] != X.shape[0]:
            raise ValueError("X and y must contain the same number of samples")

        h = X
        for layer in self.layers:
            h = layer.forward(h)
        out = self._sigmoid(h)
        n = X.shape[0]
        eps = 1e-12
        loss = -np.mean(y * np.log(out + eps) + (1.0 - y) * np.log(1.0 - out + eps))
        grad = (out - y) / n
        for layer in reversed(self.layers):
            grad = layer.backward(grad)
        return float(loss)

    def _loss_and_grads(self, x: Array, y: float) -> tuple[float, Array]:
        """Return binary cross-entropy and input gradient for one image."""
        x = self._single(x)
        loss = self._loss_and_grads_batch(x, np.array([float(y)]))
        return loss, np.empty(0)

    def train_step(self, X: Array, y: float) -> float:
        """Perform one gradient step on a single image and return the loss."""
        loss = self._loss_and_grads_batch(self._single(X), np.array([float(y)]))
        self.optimizer.step(self.layers)
        return loss

    def fit(
        self,
        X_list: Sequence[Array],
        y_list: Sequence[float],
        epochs: int = 100,
        verbose: int = 0,
    ) -> SimpleCNN:
        """Train on a list of images, keeping the original ``fit`` signature."""
        if len(X_list) != len(y_list):
            raise ValueError("X_list and y_list must have the same length")
        X = np.stack([self._single(x)[0] for x in X_list], axis=0)
        y = np.asarray(y_list, dtype=float)

        self.loss_history_ = []
        rng = np.random.default_rng(self.random_state)
        n = X.shape[0]
        for epoch in range(epochs):
            order = rng.permutation(n)
            epoch_loss = 0.0
            for start in range(0, n, 32):
                batch = order[start:start + 32]
                epoch_loss += self._loss_and_grads_batch(
                    X[batch], y[batch]
                ) * len(batch)
                self.optimizer.step(self.layers)
            mean_loss = epoch_loss / max(n, 1)
            self.loss_history_.append(mean_loss)
            if verbose:
                print(f"Epoch {epoch + 1:>3}/{epochs} - loss: {mean_loss:.5f}")
        return self

    # -- inference ----------------------------------------------------------
    def predict_proba(self, X: Array) -> float:
        """Return the probability of class 1 for a single image."""
        output, _ = self.forward(X)
        return float(output[0, 0])

    def predict(self, X: Array) -> int:
        """Return the predicted binary label (0 or 1)."""
        return int(self.predict_proba(X) > 0.5)


def generate_product_images(
    n_samples: int = 100,
    image_size: tuple[int, int] = (28, 28),
    n_products_range: tuple[int, int] = (1, 4),
    random_state: int | None = None,
) -> tuple[list[Array], list[int]]:
    """Generate the original synthetic shelf images (binary stock-out task).

    This helper is kept for backwards compatibility.  Images contain between
    ``n_products_range[0]`` and ``n_products_range[1]`` bright rectangles on a
    dark shelf.  The binary label is 1 when at least two products are present
    (well stocked) and 0 otherwise (few products / stock-out).

    Returns
    -------
    images : list of ndarray
        Each image has shape ``image_size`` and float values.
    labels : list of int
        Binary labels.
    """
    if n_samples < 1:
        raise ValueError("n_samples must be at least 1")
    rng = np.random.default_rng(random_state)
    images: list[Array] = []
    labels: list[int] = []

    for _ in range(n_samples):
        img = np.full(image_size, 0.2, dtype=float)
        n_products = int(rng.integers(n_products_range[0], n_products_range[1] + 1))
        label = 1 if n_products >= 2 else 0

        for _ in range(n_products):
            ph = int(rng.integers(5, 12))
            pw = int(rng.integers(4, 10))
            y0 = int(rng.integers(0, max(1, image_size[0] - ph)))
            x0 = int(rng.integers(0, max(1, image_size[1] - pw)))
            intensity = float(rng.uniform(0.5, 1.0))
            img[y0:y0 + ph, x0:x0 + pw] = intensity

        images.append(img)
        labels.append(label)

    return images, labels


def gradient_check_cnn(
    cnn: SimpleCNN,
    X: Array,
    y: float,
    epsilon: float = 1e-5,
) -> tuple[float, float]:
    """Numerically check the gradients of ``SimpleCNN``.

    A subset of at most 20 parameters across all layers is compared with central
    differences of the binary cross-entropy loss.

    Returns
    -------
    max_diff : float
        Maximum absolute difference between numerical and analytical values.
    relative_diff : float
        Maximum absolute difference scaled by the largest gradient magnitude.
    """
    _, _ = cnn._loss_and_grads(X, float(y))
    analytical = []
    params = []
    for layer in cnn.layers:
        for name, param in layer.params.items():
            grad = layer.grads.get(name)
            if grad is None:
                continue
            params.append((param, grad))
            analytical.append(grad.ravel())

    analytical_flat = np.concatenate(analytical) if analytical else np.array([])
    total = analytical_flat.size
    if total == 0:
        return 0.0, 0.0

    rng = np.random.default_rng(0)
    n_checks = min(20, total)
    chosen = rng.choice(total, size=n_checks, replace=False)

    numerical = np.empty(n_checks)
    for k, flat_idx in enumerate(chosen):
        offset = 0
        for param, _ in params:
            size = param.size
            if offset + size > flat_idx:
                local = flat_idx - offset
                idx = np.unravel_index(local, param.shape)
                original = param[idx]
                param[idx] = original + epsilon
                plus, _ = cnn._loss_and_grads(X, float(y))
                param[idx] = original - epsilon
                minus, _ = cnn._loss_and_grads(X, float(y))
                param[idx] = original
                numerical[k] = (plus - minus) / (2.0 * epsilon)
                break
            offset += size

    analytical_selected = analytical_flat[chosen]
    diff = np.abs(numerical - analytical_selected)
    max_diff = float(np.max(diff))
    relative_diff = max_diff / (
        float(np.max(np.abs(numerical))) + float(np.max(np.abs(analytical_selected)))
        + 1e-8
    )
    return max_diff, relative_diff


# ---------------------------------------------------------------------------
# Demo
# ---------------------------------------------------------------------------
def _demo() -> None:
    """Train a small CNN for a few seconds and print its accuracy."""
    print("=" * 66)
    print("CNN Computer Vision - Warung Shelf Product Recognition")
    print("=" * 66)

    X_train, y_train = generate_shelf_images(
        n_samples=360, image_size=16, n_classes=4, noise=0.05, random_state=42
    )
    X_test, y_test = generate_shelf_images(
        n_samples=120, image_size=16, n_classes=4, noise=0.05, random_state=7
    )
    print(f"Training images: {X_train.shape}  Test images: {X_test.shape}")
    print(f"Classes: {PRODUCT_CLASS_NAMES[:4]}")

    model = ShelfCNN(n_classes=4, image_size=16, random_state=42)
    print("\nTraining ShelfCNN with momentum SGD...")
    history = model.fit(
        X_train,
        y_train,
        epochs=12,
        batch_size=32,
        lr=0.02,
        momentum=0.9,
        verbose=1,
    )

    train_acc = model.score(X_train, y_train)
    test_acc = model.score(X_test, y_test)
    print(f"\nFirst/last loss : {history[0]:.4f} -> {history[-1]:.4f}")
    print(f"Training accuracy: {train_acc:.4f}")
    print(f"Test accuracy    : {test_acc:.4f}")

    shifted = shift_images(X_test, shift_y=2, shift_x=2, fill=0.15)
    print(f"Shifted test acc : {model.score(shifted, y_test):.4f}")
    print("\nDone.")


if __name__ == "__main__":
    _demo()
