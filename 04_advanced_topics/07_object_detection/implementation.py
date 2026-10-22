"""
Object Detection from Scratch
=============================

A compact, dependency-free (NumPy only) toolkit for the object-detection
lesson.  The module follows the same spirit as the rest of the repository:
small, readable code that exposes the mechanics instead of hiding them behind
a framework.

What is inside
--------------
Box utilities
    ``xyxy_to_xywh``, ``xywh_to_xyxy``, ``xyxy_to_cxcywh``,
    ``cxcywh_to_xyxy``, ``xywh_to_cxcywh``, ``cxcywh_to_xywh``,
    ``box_area``, ``iou_matrix`` (pairwise IoU), ``nms``, ``per_class_nms``.
Data
    ``generate_shelf_dataset`` -- synthetic 48x48 small-shop shelf images with
    1--4 products of three classes (box, bottle, can).
Mini YOLO
    ``MiniYOLO`` -- a small convolutional detector with S=6, B=1 and the
    per-cell output ``[objectness, x_offset, y_offset, sqrt_w, sqrt_h,
    class logits]``.  Convolutions use im2col, ReLU and max pooling; gradients
    are written by hand and optimised with Adam.
Targets and loss
    ``encode_targets``, ``encode_targets_batch``, ``decode_targets``,
    ``decode_targets_batch``, ``yolo_loss_and_grad``.
Prediction
    ``decode_single``, ``decode_predictions``.
Evaluation
    ``voc_average_precision``, ``mean_average_precision``.

Everything is CPU friendly.  The script at the bottom prints a short
end-to-end demonstration when the file is executed directly.

Author: ML-Algo Repository
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np

Array = np.ndarray

CLASS_NAMES: tuple[str, ...] = ("box", "bottle", "can")
NUM_CLASSES: int = len(CLASS_NAMES)
DEFAULT_IMAGE_SIZE: int = 48
DEFAULT_GRID_SIZE: int = 6
LAMBDA_COORD: float = 5.0
LAMBDA_NOOBJ: float = 0.5

_EPS = 1e-12


# ---------------------------------------------------------------------------
# Validation helpers
# ---------------------------------------------------------------------------
def _positive_int(value: int, name: str, minimum: int = 1) -> int:
    """Validate an integer scalar and return it as a Python ``int``."""
    if isinstance(value, bool) or not isinstance(value, (int, np.integer)):
        raise TypeError(f"{name} must be an integer")
    value = int(value)
    if value < minimum:
        raise ValueError(f"{name} must be at least {minimum}")
    return value


def _as_box_matrix(boxes: Any, name: str = "boxes") -> tuple[Array, bool]:
    """Return ``boxes`` as a float ``(N, 4)`` array and a single-box flag."""
    arr = np.asarray(boxes, dtype=float)
    if arr.size == 0:
        return np.empty((0, 4), dtype=float), False
    if arr.ndim == 1:
        if arr.size != 4:
            raise ValueError(f"{name} must have exactly 4 coordinates")
        arr = arr.reshape(1, 4)
        single = True
    elif arr.ndim == 2 and arr.shape[1] == 4:
        single = False
    else:
        raise ValueError(f"{name} must have shape (4,) or (N, 4)")
    if not np.all(np.isfinite(arr)):
        raise ValueError(f"{name} contains NaN or infinite values")
    return arr, single


def _restore_boxes(arr: Array, single: bool) -> Array:
    return arr[0] if single else arr


def _as_1d(values: Any, name: str) -> Array:
    arr = np.asarray(values)
    if arr.size == 0:
        return np.empty((0,), dtype=float)
    return arr.reshape(-1)


def _check_iou_threshold(value: float) -> float:
    value = float(value)
    if not 0.0 <= value <= 1.0:
        raise ValueError("iou_threshold must be in [0, 1]")
    return value


def _sigmoid(x: Array) -> Array:
    """Numerically stable logistic sigmoid."""
    out = np.empty_like(x, dtype=float)
    positive = x >= 0
    out[positive] = 1.0 / (1.0 + np.exp(-x[positive]))
    exp_x = np.exp(x[~positive])
    out[~positive] = exp_x / (1.0 + exp_x)
    return out


def _softmax(x: Array, axis: int = -1) -> Array:
    """Numerically stable softmax along ``axis``."""
    shifted = x - np.max(x, axis=axis, keepdims=True)
    exp_x = np.exp(shifted)
    return exp_x / np.sum(exp_x, axis=axis, keepdims=True)


def _log_softmax(x: Array, axis: int = -1) -> Array:
    shifted = x - np.max(x, axis=axis, keepdims=True)
    return shifted - np.log(np.sum(np.exp(shifted), axis=axis, keepdims=True))


# ---------------------------------------------------------------------------
# Box formats
# ---------------------------------------------------------------------------
def xyxy_to_xywh(boxes: Any) -> Array:
    """Convert ``(x1, y1, x2, y2)`` boxes to ``(x, y, w, h)``.

    The function accepts a single box ``(4,)`` or a batch ``(N, 4)`` and
    returns the same leading shape.  Width and height are not clipped, so the
    conversion is an exact algebraic round trip for legal boxes.
    """
    arr, single = _as_box_matrix(boxes)
    out = np.empty_like(arr)
    out[:, 0] = arr[:, 0]
    out[:, 1] = arr[:, 1]
    out[:, 2] = arr[:, 2] - arr[:, 0]
    out[:, 3] = arr[:, 3] - arr[:, 1]
    return _restore_boxes(out, single)


def xywh_to_xyxy(boxes: Any) -> Array:
    """Convert ``(x, y, w, h)`` boxes to ``(x1, y1, x2, y2)``."""
    arr, single = _as_box_matrix(boxes)
    out = np.empty_like(arr)
    out[:, 0] = arr[:, 0]
    out[:, 1] = arr[:, 1]
    out[:, 2] = arr[:, 0] + arr[:, 2]
    out[:, 3] = arr[:, 1] + arr[:, 3]
    return _restore_boxes(out, single)


def xyxy_to_cxcywh(boxes: Any) -> Array:
    """Convert ``(x1, y1, x2, y2)`` boxes to center-based ``(cx, cy, w, h)``."""
    arr, single = _as_box_matrix(boxes)
    out = np.empty_like(arr)
    out[:, 0] = (arr[:, 0] + arr[:, 2]) / 2.0
    out[:, 1] = (arr[:, 1] + arr[:, 3]) / 2.0
    out[:, 2] = arr[:, 2] - arr[:, 0]
    out[:, 3] = arr[:, 3] - arr[:, 1]
    return _restore_boxes(out, single)


def cxcywh_to_xyxy(boxes: Any) -> Array:
    """Convert center-based ``(cx, cy, w, h)`` boxes to ``(x1, y1, x2, y2)``."""
    arr, single = _as_box_matrix(boxes)
    half_w = arr[:, 2] / 2.0
    half_h = arr[:, 3] / 2.0
    out = np.empty_like(arr)
    out[:, 0] = arr[:, 0] - half_w
    out[:, 1] = arr[:, 1] - half_h
    out[:, 2] = arr[:, 0] + half_w
    out[:, 3] = arr[:, 1] + half_h
    return _restore_boxes(out, single)


def xywh_to_cxcywh(boxes: Any) -> Array:
    """Convert ``(x, y, w, h)`` boxes to ``(cx, cy, w, h)``."""
    arr, single = _as_box_matrix(boxes)
    out = arr.copy()
    out[:, 0] = arr[:, 0] + arr[:, 2] / 2.0
    out[:, 1] = arr[:, 1] + arr[:, 3] / 2.0
    return _restore_boxes(out, single)


def cxcywh_to_xywh(boxes: Any) -> Array:
    """Convert ``(cx, cy, w, h)`` boxes to top-left ``(x, y, w, h)``."""
    arr, single = _as_box_matrix(boxes)
    out = arr.copy()
    out[:, 0] = arr[:, 0] - arr[:, 2] / 2.0
    out[:, 1] = arr[:, 1] - arr[:, 3] / 2.0
    return _restore_boxes(out, single)


def box_area(boxes: Any) -> Array:
    """Area of ``(x1, y1, x2, y2)`` boxes, clipped at zero."""
    arr, single = _as_box_matrix(boxes)
    widths = np.maximum(arr[:, 2] - arr[:, 0], 0.0)
    heights = np.maximum(arr[:, 3] - arr[:, 1], 0.0)
    area = widths * heights
    return area[0] if single else area


# ---------------------------------------------------------------------------
# IoU and non-maximum suppression
# ---------------------------------------------------------------------------
def iou_matrix(boxes_a: Any, boxes_b: Any) -> Array:
    """Pairwise intersection-over-union, ``(N, 4) x (M, 4) -> (N, M)``.

    Boxes are ``(x1, y1, x2, y2)``.  Intersections with zero (or negative)
    width or height contribute nothing, and a zero union area yields IoU 0.
    Both inputs may also be a single ``(4,)`` box; the result is still a
    matrix with shape ``(1, 1)`` in that case.
    """
    a, _ = _as_box_matrix(boxes_a, "boxes_a")
    b, _ = _as_box_matrix(boxes_b, "boxes_b")
    area_a = np.maximum(a[:, 2] - a[:, 0], 0.0) * np.maximum(
        a[:, 3] - a[:, 1], 0.0
    )
    area_b = np.maximum(b[:, 2] - b[:, 0], 0.0) * np.maximum(
        b[:, 3] - b[:, 1], 0.0
    )

    top_left = np.maximum(a[:, None, :2], b[None, :, :2])
    bottom_right = np.minimum(a[:, None, 2:], b[None, :, 2:])
    inter_wh = np.maximum(bottom_right - top_left, 0.0)
    inter = inter_wh[..., 0] * inter_wh[..., 1]
    union = area_a[:, None] + area_b[None, :] - inter
    iou = np.zeros_like(inter)
    np.divide(inter, union, out=iou, where=union > _EPS)
    return iou


def nms(
    boxes: Any,
    scores: Any,
    iou_threshold: float = 0.5,
) -> Array:
    """Greedy non-maximum suppression for one class.

    Boxes are sorted by descending score.  The highest-scoring box is kept and
    every remaining box whose IoU with it exceeds ``iou_threshold`` is
    suppressed.  The process repeats until no boxes remain.

    Returns
    -------
    kept : ndarray of int
        Indices into the input arrays, ordered by descending score.
    """
    arr, _ = _as_box_matrix(boxes)
    score_arr = _as_1d(scores, "scores").astype(float)
    if arr.shape[0] != score_arr.size:
        raise ValueError("boxes and scores must have the same length")
    if not np.all(np.isfinite(score_arr)):
        raise ValueError("scores contains NaN or infinite values")
    threshold = _check_iou_threshold(iou_threshold)
    if arr.shape[0] == 0:
        return np.empty((0,), dtype=int)

    order = np.argsort(-score_arr, kind="stable")
    keep: list[int] = []
    while order.size > 0:
        current = int(order[0])
        keep.append(current)
        if order.size == 1:
            break
        ious = iou_matrix(arr[current : current + 1], arr[order[1:]])[0]
        order = order[1:][ious <= threshold]
    return np.asarray(keep, dtype=int)


def per_class_nms(
    boxes: Any,
    scores: Any,
    labels: Any,
    iou_threshold: float = 0.5,
) -> Array:
    """Class-aware NMS.

    Boxes with different labels never suppress each other, even when they
    overlap heavily.  All per-class kept indices are merged and sorted by
    descending score.
    """
    arr, _ = _as_box_matrix(boxes)
    score_arr = _as_1d(scores, "scores").astype(float)
    label_arr = _as_1d(labels, "labels").astype(int)
    if not (arr.shape[0] == score_arr.size == label_arr.size):
        raise ValueError("boxes, scores and labels must have the same length")
    threshold = _check_iou_threshold(iou_threshold)
    if arr.shape[0] == 0:
        return np.empty((0,), dtype=int)

    kept: list[int] = []
    for label in np.unique(label_arr):
        class_indices = np.flatnonzero(label_arr == label)
        local = nms(arr[class_indices], score_arr[class_indices], threshold)
        kept.extend(class_indices[local].tolist())
    if not kept:
        return np.empty((0,), dtype=int)
    kept_arr = np.asarray(kept, dtype=int)
    return kept_arr[np.argsort(-score_arr[kept_arr], kind="stable")]


#: Alias used in some detection codebases.
batched_nms = per_class_nms


# ---------------------------------------------------------------------------
# Synthetic small-shop shelf dataset
# ---------------------------------------------------------------------------
def _class_colors(channels: int) -> tuple[tuple[float, ...], ...]:
    """Base RGB / grey colours for box, bottle and can."""
    if channels == 1:
        return (0.78,), (0.64,), (0.86,)
    return (0.80, 0.38, 0.26), (0.28, 0.62, 0.70), (0.88, 0.74, 0.30)


def _fill_rect(
    image: Array,
    x0: int,
    y0: int,
    x1: int,
    y1: int,
    color: Sequence[float],
) -> None:
    """Fill a clipped half-open rectangle in an ``(H, W, C)`` image."""
    h, w = image.shape[:2]
    x0, x1 = max(0, int(x0)), min(w, int(x1))
    y0, y1 = max(0, int(y0)), min(h, int(y1))
    if x0 >= x1 or y0 >= y1:
        return
    image[y0:y1, x0:x1] = np.asarray(color, dtype=float).reshape(1, 1, -1)


def _shade(color: Sequence[float], factor: float) -> tuple[float, ...]:
    return tuple(float(np.clip(c * factor, 0.0, 1.0)) for c in color)


def _render_object(
    image: Array,
    class_id: int,
    x0: int,
    y0: int,
    width: int,
    height: int,
) -> None:
    """Draw one product with a class-specific silhouette and texture."""
    channels = image.shape[2]
    base = _class_colors(channels)[class_id]
    if channels == 1:
        dark = _shade(base, 0.72)
        light = _shade(base, 1.18)
        accent = _shade(base, 1.35)
        shadow = _shade(base, 0.45)
    else:
        dark = _shade(base, 0.70)
        light = _shade(base, 1.20)
        accent = _shade(base, 1.45)
        shadow = _shade(base, 0.45)

    x1, y1 = x0 + width, y0 + height

    if class_id == 0:
        # A carton box: solid body, diagonal stripes and a bright label.
        _fill_rect(image, x0, y0, x1, y1, base)
        for yy in range(y0, y1):
            for xx in range(x0, x1):
                if (xx - x0 + yy - y0) % 5 < 2:
                    image[yy, xx] = np.asarray(dark)
        _fill_rect(image, x0, y0, x1, y0 + 1, light)
        _fill_rect(image, x0, y0, x0 + 1, y1, light)
        _fill_rect(image, x1 - 1, y0, x1, y1, shadow)
        _fill_rect(image, x0, y1 - 1, x1, y1, shadow)
        label_w = max(2, width // 2)
        label_h = max(2, height // 3)
        lx = x0 + (width - label_w) // 2
        ly = y0 + (height - label_h) // 2
        _fill_rect(image, lx, ly, lx + label_w, ly + label_h, accent)
    elif class_id == 1:
        # A bottle: narrow neck and cap above a wider body.
        body_top = y0 + max(4, height // 4)
        neck_w = max(2, width // 3)
        neck_x = x0 + (width - neck_w) // 2
        cap_w = max(3, width // 2)
        cap_x = x0 + (width - cap_w) // 2
        _fill_rect(image, cap_x, y0, cap_x + cap_w, y0 + max(2, height // 10), accent)
        _fill_rect(image, neck_x, y0 + max(2, height // 10), neck_x + neck_w,
                   body_top, base)
        _fill_rect(image, x0, body_top, x1, y1, base)
        _fill_rect(image, x0, body_top, x0 + max(1, width // 5), y1, dark)
        _fill_rect(image, x1 - max(1, width // 5), body_top, x1, y1, light)
        band_y0 = body_top + max(2, (y1 - body_top) // 4)
        band_y1 = y1 - max(2, (y1 - body_top) // 5)
        _fill_rect(image, x0 + max(1, width // 4), band_y0,
                   x1 - max(1, width // 4), band_y1, accent)
        _fill_rect(image, x0, body_top, x0 + 1, y1, light)
        _fill_rect(image, x1 - 1, body_top, x1, y1, shadow)
    else:
        # A can: cylinder with a lid, base and horizontal ridges.
        _fill_rect(image, x0, y0, x1, y1, base)
        _fill_rect(image, x0, y0, x1, y0 + max(2, height // 6), light)
        _fill_rect(image, x0, y1 - max(2, height // 6), x1, y1, dark)
        ridge_start = y0 + max(3, height // 4)
        for yy in range(ridge_start, y1 - 2, 4):
            _fill_rect(image, x0 + 1, yy, x1 - 1, yy + 1, dark)
        _fill_rect(image, x0 + 1, y0, x0 + max(2, width // 4), y1, light)
        _fill_rect(image, x1 - max(2, width // 5), y0, x1 - 1, y1, shadow)
        _fill_rect(image, x0, y0, x0 + 1, y1, light)
        _fill_rect(image, x1 - 1, y0, x1, y1, shadow)
        # Round the four corners by carving them back to the background.
        background = image[max(0, y0 - 2), max(0, x0 - 2)].copy()
        for dx, dy in ((0, 0), (width - 1, 0), (0, height - 1),
                       (width - 1, height - 1)):
            yy, xx = y0 + dy, x0 + dx
            if 0 <= yy < image.shape[0] and 0 <= xx < image.shape[1]:
                image[yy, xx] = background


def _sample_size(class_id: int, rng: np.random.Generator) -> tuple[int, int]:
    if class_id == 0:
        return int(rng.integers(10, 17)), int(rng.integers(10, 17))
    if class_id == 1:
        return int(rng.integers(7, 12)), int(rng.integers(14, 23))
    return int(rng.integers(9, 14)), int(rng.integers(11, 17))


def _overlaps(
    candidate: Array,
    boxes: Array,
    margin: float = 2.0,
) -> bool:
    """Return True when ``candidate`` overlaps any box after inflation."""
    if boxes.size == 0:
        return False
    inflated = boxes.copy()
    inflated[:, :2] -= margin
    inflated[:, 2:] += margin
    inter_w = np.minimum(inflated[:, 2], candidate[2]) - np.maximum(
        inflated[:, 0], candidate[0]
    )
    inter_h = np.minimum(inflated[:, 3], candidate[3]) - np.maximum(
        inflated[:, 1], candidate[1]
    )
    return bool(np.any((inter_w > 0) & (inter_h > 0)))


def generate_shelf_dataset(
    n_samples: int = 200,
    image_size: int = DEFAULT_IMAGE_SIZE,
    channels: int = 1,
    min_objects: int = 1,
    max_objects: int = 4,
    seed: int = 0,
    noise: float = 0.03,
    grid_size: int = DEFAULT_GRID_SIZE,
) -> tuple[Array, list[tuple[Array, Array]]]:
    """Generate synthetic small-shop shelf images with product boxes.

    Each image is a greyscale or RGB ``image_size x image_size`` scene with a
    simple shelf background and one to four products.  The three classes have
    deliberately distinct silhouettes and textures:

    ==========  ===========================================================
    class       appearance
    ==========  ===========================================================
    ``box``     carton with diagonal stripes and a bright centre label
    ``bottle``  narrow neck and cap above a wider, banded body
    ``can``     cylinder with a lid, base and horizontal ridges
    ==========  ===========================================================

    Object centres are also placed in distinct grid cells.  This keeps the
    one-box-per-cell target encoding unambiguous on this dataset.

    Parameters
    ----------
    n_samples : int, default=200
        Number of images to generate.
    image_size : int, default=48
        Square image side.  Must be divisible by ``4 * grid_size``.
    channels : {1, 3}, default=1
        Greyscale or RGB images.
    min_objects, max_objects : int, default=1, 4
        Inclusive range for the number of products per image.
    seed : int, default=0
        Seed for ``numpy.random.default_rng``; identical seeds reproduce the
        exact same images and annotations.
    noise : float, default=0.03
        Standard deviation of Gaussian sensor noise.
    grid_size : int, default=6
        Detector grid size used to enforce distinct centre cells.

    Returns
    -------
    images : ndarray of shape (N, C, H, W), float32
        Channels-first images in ``[0, 1]``.
    annotations : list of (boxes, labels)
        ``boxes`` has shape ``(K, 4)`` in ``(x1, y1, x2, y2)`` pixel
        coordinates and ``labels`` has shape ``(K,)`` with values ``0, 1, 2``
        for box, bottle and can.
    """
    n_samples = _positive_int(n_samples, "n_samples")
    image_size = _positive_int(image_size, "image_size", minimum=8)
    if channels not in (1, 3):
        raise ValueError("channels must be 1 (grayscale) or 3 (RGB)")
    min_objects = _positive_int(min_objects, "min_objects")
    max_objects = _positive_int(max_objects, "max_objects")
    if min_objects > max_objects:
        raise ValueError("min_objects must not exceed max_objects")
    if max_objects > 8:
        raise ValueError("max_objects must be at most 8")
    if not isinstance(seed, (int, np.integer)) or isinstance(seed, bool):
        raise TypeError("seed must be an integer")
    noise = float(noise)
    if noise < 0:
        raise ValueError("noise must be non-negative")
    grid_size = _positive_int(grid_size, "grid_size")
    if image_size % (4 * grid_size) != 0:
        raise ValueError("image_size must be divisible by 4 * grid_size")

    rng = np.random.default_rng(int(seed))
    cell = image_size // grid_size
    images = np.empty((n_samples, channels, image_size, image_size),
                      dtype=np.float32)
    annotations: list[tuple[Array, Array]] = []

    for sample_index in range(n_samples):
        scene = None
        for _attempt in range(40):
            image = np.full((image_size, image_size, channels), 0.20, dtype=float)
            # Horizontal shelf boards give the detector a plausible context.
            for board_y in range(4, image_size, 12):
                image[board_y:board_y + 1, :, :] += 0.10
            image += rng.normal(0.0, noise, size=image.shape)
            if channels == 3:
                image[..., 0] *= 1.02
                image[..., 2] *= 0.97

            n_objects = int(rng.integers(min_objects, max_objects + 1))
            boxes: list[Array] = []
            labels: list[int] = []
            used_cells: set[tuple[int, int]] = set()
            placed = 0
            for _ in range(n_objects):
                for _try in range(120):
                    class_id = int(rng.integers(0, NUM_CLASSES))
                    width, height = _sample_size(class_id, rng)
                    if width >= image_size - 2 or height >= image_size - 2:
                        continue
                    x0 = int(rng.integers(1, image_size - width - 1))
                    y0 = int(rng.integers(1, image_size - height - 1))
                    x1, y1 = x0 + width, y0 + height
                    cell_x = min(int(((x0 + x1) / 2) // cell), grid_size - 1)
                    cell_y = min(int(((y0 + y1) / 2) // cell), grid_size - 1)
                    if (cell_y, cell_x) in used_cells:
                        continue
                    candidate = np.array([x0, y0, x1, y1], dtype=float)
                    if _overlaps(candidate, np.asarray(boxes) if boxes else
                                 np.empty((0, 4))):
                        continue
                    boxes.append(candidate)
                    labels.append(class_id)
                    used_cells.add((cell_y, cell_x))
                    _render_object(image, class_id, x0, y0, width, height)
                    placed += 1
                    break
            if placed == n_objects and placed > 0:
                scene = (image, boxes, labels)
                break
        if scene is None:
            raise RuntimeError("Could not place non-overlapping products")

        image, boxes, labels = scene
        image += rng.normal(0.0, noise * 0.5, size=image.shape)
        np.clip(image, 0.0, 1.0, out=image)
        images[sample_index] = np.transpose(image, (2, 0, 1)).astype(np.float32)
        box_array = np.asarray(boxes, dtype=np.float32).reshape(-1, 4)
        label_array = np.asarray(labels, dtype=np.int64)
        annotations.append((box_array, label_array))

    return images, annotations


#: Friendly alias for the dataset generator.
generate_warung_shelf = generate_shelf_dataset


# ---------------------------------------------------------------------------
# YOLO target encoding
# ---------------------------------------------------------------------------
def _validate_target_shape(raw: Array, grid_size: int, num_classes: int) -> None:
    expected = 5 + num_classes
    if raw.ndim != 3 or raw.shape != (grid_size, grid_size, expected):
        raise ValueError(
            "raw predictions must have shape "
            f"({grid_size}, {grid_size}, {expected})"
        )


def encode_targets(
    boxes: Any,
    labels: Any,
    image_size: int = DEFAULT_IMAGE_SIZE,
    grid_size: int = DEFAULT_GRID_SIZE,
    num_classes: int = NUM_CLASSES,
) -> Array:
    """Encode one image's boxes into the YOLO target tensor.

    The output has shape ``(S, S, 5 + C)``.  For every object the cell that
    contains its centre is selected.  If two centres fall in the same cell the
    larger box wins (the dataset generator avoids this case).  The channels
    are:

    ``[objectness, x_offset, y_offset, sqrt_w, sqrt_h, class one-hot]``

    where ``x_offset, y_offset`` are in ``[0, 1)`` relative to the cell and
    ``sqrt_w, sqrt_h`` are ``sqrt(width / image_size)`` and
    ``sqrt(height / image_size)``.
    """
    image_size = _positive_int(image_size, "image_size")
    grid_size = _positive_int(grid_size, "grid_size")
    num_classes = _positive_int(num_classes, "num_classes")
    box_arr, _ = _as_box_matrix(boxes)
    label_arr = _as_1d(labels, "labels").astype(int)
    if box_arr.shape[0] != label_arr.size:
        raise ValueError("boxes and labels must have the same length")
    if np.any(label_arr < 0) or np.any(label_arr >= num_classes):
        raise ValueError("labels must be in [0, num_classes)")

    targets = np.zeros((grid_size, grid_size, 5 + num_classes), dtype=float)
    cell = image_size / grid_size
    areas = np.zeros((grid_size, grid_size), dtype=float)

    for box, label in zip(box_arr, label_arr, strict=True):
        clipped = np.array(
            [
                np.clip(box[0], 0.0, image_size),
                np.clip(box[1], 0.0, image_size),
                np.clip(box[2], 0.0, image_size),
                np.clip(box[3], 0.0, image_size),
            ],
            dtype=float,
        )
        width = clipped[2] - clipped[0]
        height = clipped[3] - clipped[1]
        if width <= 0 or height <= 0:
            continue
        cx = (clipped[0] + clipped[2]) / 2.0
        cy = (clipped[1] + clipped[3]) / 2.0
        col = min(int(cx // cell), grid_size - 1)
        row = min(int(cy // cell), grid_size - 1)
        area = width * height
        if targets[row, col, 0] > 0.5 and areas[row, col] >= area:
            continue
        targets[row, col, 0] = 1.0
        targets[row, col, 1] = np.clip(cx / cell - col, 0.0, 1.0)
        targets[row, col, 2] = np.clip(cy / cell - row, 0.0, 1.0)
        targets[row, col, 3] = np.sqrt(width / image_size)
        targets[row, col, 4] = np.sqrt(height / image_size)
        targets[row, col, 5:] = 0.0
        targets[row, col, 5 + label] = 1.0
        areas[row, col] = area
    return targets


def encode_targets_batch(
    annotations: Sequence[tuple[Any, Any]],
    image_size: int = DEFAULT_IMAGE_SIZE,
    grid_size: int = DEFAULT_GRID_SIZE,
    num_classes: int = NUM_CLASSES,
) -> Array:
    """Stack :func:`encode_targets` over a sequence of annotations."""
    encoded = [
        encode_targets(boxes, labels, image_size, grid_size, num_classes)
        for boxes, labels in annotations
    ]
    if not encoded:
        raise ValueError("annotations must not be empty")
    return np.stack(encoded, axis=0)


def decode_targets(
    target: Array,
    image_size: int = DEFAULT_IMAGE_SIZE,
    grid_size: int = DEFAULT_GRID_SIZE,
    num_classes: int = NUM_CLASSES,
) -> tuple[Array, Array]:
    """Decode a single target tensor back to ``(boxes, labels)``.

    This is the inverse of :func:`encode_targets` for boxes that lie inside
    the image.  It is mainly useful for tests and for visual debugging.
    """
    target = np.asarray(target, dtype=float)
    if target.shape != (grid_size, grid_size, 5 + num_classes):
        raise ValueError("target has an unexpected shape")
    cell = image_size / grid_size
    boxes: list[Array] = []
    labels: list[int] = []
    rows, cols = np.nonzero(target[..., 0] > 0.5)
    for row, col in zip(rows, cols, strict=True):
        cx = (col + target[row, col, 1]) * cell
        cy = (row + target[row, col, 2]) * cell
        width = target[row, col, 3] ** 2 * image_size
        height = target[row, col, 4] ** 2 * image_size
        boxes.append(
            np.array(
                [cx - width / 2, cy - height / 2,
                 cx + width / 2, cy + height / 2],
                dtype=float,
            )
        )
        labels.append(int(np.argmax(target[row, col, 5:])))
    if not boxes:
        return np.empty((0, 4), dtype=float), np.empty((0,), dtype=int)
    return np.stack(boxes), np.asarray(labels, dtype=int)


def decode_targets_batch(
    targets: Array,
    image_size: int = DEFAULT_IMAGE_SIZE,
    grid_size: int = DEFAULT_GRID_SIZE,
    num_classes: int = NUM_CLASSES,
) -> list[tuple[Array, Array]]:
    """Decode a batch of target tensors."""
    targets = np.asarray(targets, dtype=float)
    if targets.ndim != 4:
        raise ValueError("targets must have shape (N, S, S, 5 + C)")
    return [
        decode_targets(target, image_size, grid_size, num_classes)
        for target in targets
    ]


# ---------------------------------------------------------------------------
# YOLO loss
# ---------------------------------------------------------------------------
def yolo_loss_and_grad(
    raw: Array,
    targets: Array,
    lambda_coord: float = LAMBDA_COORD,
    lambda_noobj: float = LAMBDA_NOOBJ,
) -> tuple[float, Array]:
    """YOLO-v1-style loss and its gradient with respect to raw head logits.

    Parameters
    ----------
    raw : ndarray of shape (N, S, S, 5 + C)
        Untransformed network output.  Channel 0 is the objectness logit,
        channels 1--4 are ``x_offset, y_offset, sqrt_w, sqrt_h`` and the last
        ``C`` channels are class logits.
    targets : ndarray of the same shape
        Output of :func:`encode_targets_batch`.

    Returns
    -------
    loss : float
    gradient : ndarray of the same shape as ``raw``

    Notes
    -----
    With ``N`` images, object mask :math:`M_{ij}` and background mask
    :math:`\\bar M_{ij}`, the exact loss is

    .. math::

        L = \\frac{1}{N}\\sum_{i,j}\\Big[
            M_{ij}\\,\\mathrm{BCE}(o_{ij}, 1)
            + \\bar M_{ij}\\,\\lambda_{noobj}\\,
              \\mathrm{BCE}(o_{ij}, 0)
            + M_{ij}\\,\\lambda_{coord}
              \\big[(x-\\hat x)^2+(y-\\hat y)^2
              +(\\sqrt w-\\sqrt{\\hat w})^2
              +(\\sqrt h-\\sqrt{\\hat h})^2\\big]
            + M_{ij}\\,\\mathrm{CE}(\\mathrm{softmax}(c), y)
        \\Big]

    with ``lambda_coord = 5`` and ``lambda_noobj = 0.5`` by default.  BCE is
    the binary cross-entropy of ``sigmoid(objectness)`` and is implemented in
    a numerically stable log-sum-exp form.  Class cross-entropy is applied
    only to object cells.  The loss is averaged over the batch and summed over
    the grid and channels.
    """
    raw = np.asarray(raw, dtype=float)
    targets = np.asarray(targets, dtype=float)
    if raw.ndim != 4 or raw.shape != targets.shape:
        raise ValueError("raw and targets must have the same 4-D shape")
    if raw.shape[-1] < 6:
        raise ValueError("the last dimension must contain 5 + num_classes")
    n_images = raw.shape[0]
    if n_images == 0:
        raise ValueError("the batch must contain at least one image")

    objectness_logit = raw[..., 0]
    objectness_target = targets[..., 0]
    object_mask = objectness_target > 0.5
    object_float = object_mask.astype(float)
    background_float = 1.0 - object_float

    # Stable BCE: max(o, 0) - o*t + log(1 + exp(-|o|)).
    bce = (
        np.maximum(objectness_logit, 0.0)
        - objectness_logit * objectness_target
        + np.log1p(np.exp(-np.abs(objectness_logit)))
    )
    object_loss = np.sum(bce * object_float)
    background_loss = np.sum(bce * background_float)

    coord_error = np.sum(
        (raw[..., 1:5] - targets[..., 1:5]) ** 2 * object_float[..., None]
    )

    class_logits = raw[..., 5:]
    log_probs = _log_softmax(class_logits, axis=-1)
    class_loss = -np.sum(
        targets[..., 5:] * log_probs * object_float[..., None]
    )

    total = (
        lambda_coord * coord_error
        + object_loss
        + lambda_noobj * background_loss
        + class_loss
    ) / n_images

    gradient = np.zeros_like(raw, dtype=float)
    sigmoid_obj = _sigmoid(objectness_logit)
    gradient[..., 0] = (
        (sigmoid_obj - objectness_target) * object_float
        + lambda_noobj * (sigmoid_obj - objectness_target) * background_float
    ) / n_images
    gradient[..., 1:5] = (
        2.0
        * lambda_coord
        * (raw[..., 1:5] - targets[..., 1:5])
        * object_float[..., None]
        / n_images
    )
    softmax_probs = np.exp(log_probs)
    gradient[..., 5:] = (
        (softmax_probs - targets[..., 5:]) * object_float[..., None] / n_images
    )
    return float(total), gradient


#: Alias for checking the loss directly on raw head output.
loss_and_grad = yolo_loss_and_grad


# ---------------------------------------------------------------------------
# Convolution building blocks (im2col + manual backprop)
# ---------------------------------------------------------------------------
def im2col(
    x: Array,
    kernel_size: int,
    stride: int = 1,
    padding: int = 0,
) -> Array:
    """Extract square patches into a ``(N*out_h*out_w, C*k*k)`` matrix.

    The input uses NCHW layout.  Row order is ``n, out_h, out_w`` and column
    order is ``c, k_h, k_w``.
    """
    x = np.asarray(x, dtype=float)
    if x.ndim != 4:
        raise ValueError("im2col expects an (N, C, H, W) array")
    if kernel_size <= 0 or stride <= 0 or padding < 0:
        raise ValueError("kernel_size/stride must be positive, padding >= 0")
    n, c, h, w = x.shape
    out_h = (h + 2 * padding - kernel_size) // stride + 1
    out_w = (w + 2 * padding - kernel_size) // stride + 1
    if out_h <= 0 or out_w <= 0:
        raise ValueError("the kernel is larger than the padded input")
    if padding:
        x = np.pad(x, ((0, 0), (0, 0), (padding, padding),
                       (padding, padding)), mode="constant")
    cols = np.empty((n, out_h, out_w, c, kernel_size, kernel_size),
                    dtype=x.dtype)
    for i in range(out_h):
        for j in range(out_w):
            cols[:, i, j] = x[
                :, :, i * stride:i * stride + kernel_size,
                j * stride:j * stride + kernel_size
            ]
    return cols.reshape(n * out_h * out_w, c * kernel_size * kernel_size)


def col2im(
    cols: Array,
    input_shape: tuple[int, int, int, int],
    kernel_size: int,
    stride: int = 1,
    padding: int = 0,
) -> Array:
    """Adjoint of :func:`im2col`; scatter-add columns back to an image."""
    cols = np.asarray(cols, dtype=float)
    n, c, h, w = input_shape
    out_h = (h + 2 * padding - kernel_size) // stride + 1
    out_w = (w + 2 * padding - kernel_size) // stride + 1
    patch = cols.reshape(n, out_h, out_w, c, kernel_size, kernel_size)
    padded = np.zeros((n, c, h + 2 * padding, w + 2 * padding), dtype=float)
    for i in range(kernel_size):
        for j in range(kernel_size):
            padded[
                :, :, i:i + out_h * stride:stride,
                j:j + out_w * stride:stride
            ] += patch[:, :, :, :, i, j].transpose(0, 3, 1, 2)
    if padding:
        return padded[:, :, padding:padding + h, padding:padding + w]
    return padded


class Conv2D:
    """A square convolution layer with explicit im2col forward/backward."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        stride: int = 1,
        padding: int = 1,
        rng: np.random.Generator | None = None,
        dtype: np.dtype = np.float64,
    ) -> None:
        rng = np.random.default_rng(0) if rng is None else rng
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        fan_in = in_channels * kernel_size * kernel_size
        scale = np.sqrt(2.0 / fan_in)
        self.W = rng.normal(0.0, scale,
                            size=(out_channels, in_channels, kernel_size,
                                  kernel_size)).astype(dtype)
        self.b = np.zeros(out_channels, dtype=dtype)
        self.dW = np.zeros_like(self.W)
        self.db = np.zeros_like(self.b)
        self._cache: tuple[Array, tuple[int, int, int, int]] | None = None

    def forward(self, x: Array) -> Array:
        x = np.asarray(x)
        cols = im2col(x, self.kernel_size, self.stride, self.padding)
        n, _, h, w = x.shape
        out_h = (h + 2 * self.padding - self.kernel_size) // self.stride + 1
        out_w = (w + 2 * self.padding - self.kernel_size) // self.stride + 1
        out = cols @ self.W.reshape(self.out_channels, -1).T + self.b
        out = out.reshape(n, out_h, out_w, self.out_channels).transpose(0, 3, 1, 2)
        self._cache = (cols, (n, self.in_channels, h, w))
        return out

    def backward(self, dout: Array) -> Array:
        if self._cache is None:
            raise RuntimeError("forward must be called before backward")
        cols, input_shape = self._cache
        dout_flat = np.ascontiguousarray(
            dout.transpose(0, 2, 3, 1)
        ).reshape(-1, self.out_channels)
        self.dW = (dout_flat.T @ cols).reshape(self.W.shape)
        self.db = dout_flat.sum(axis=0)
        dcols = dout_flat @ self.W.reshape(self.out_channels, -1)
        return col2im(
            dcols,
            input_shape,
            self.kernel_size,
            self.stride,
            self.padding,
        )


class MaxPool2D:
    """Non-overlapping 2x2 max pooling with argmax backpropagation."""

    def __init__(self) -> None:
        self._cache: tuple[Array, tuple[int, ...]] | None = None

    def forward(self, x: Array) -> Array:
        x = np.asarray(x, dtype=float)
        n, c, h, w = x.shape
        if h % 2 or w % 2:
            raise ValueError("MaxPool2D expects even spatial dimensions")
        out_h, out_w = h // 2, w // 2
        windows = (
            x.reshape(n, c, out_h, 2, out_w, 2)
            .transpose(0, 1, 2, 4, 3, 5)
            .reshape(n, c, out_h, out_w, 4)
        )
        indices = np.argmax(windows, axis=-1)
        out = np.take_along_axis(windows, indices[..., None], axis=-1)[..., 0]
        self._cache = (indices, x.shape)
        return out

    def backward(self, dout: Array) -> Array:
        if self._cache is None:
            raise RuntimeError("forward must be called before backward")
        indices, (n, c, h, w) = self._cache
        out_h, out_w = h // 2, w // 2
        dflat = np.zeros((n, c, out_h, out_w, 4), dtype=float)
        np.put_along_axis(dflat, indices[..., None], dout[..., None], axis=-1)
        return (
            dflat.reshape(n, c, out_h, out_w, 2, 2)
            .transpose(0, 1, 2, 4, 3, 5)
            .reshape(n, c, h, w)
        )


def relu_forward(x: Array) -> Array:
    return np.maximum(x, 0.0)


def relu_backward(dout: Array, x: Array) -> Array:
    return dout * (x > 0)


class Adam:
    """Adam optimiser with lazy per-parameter state."""

    def __init__(
        self,
        lr: float = 1e-3,
        beta1: float = 0.9,
        beta2: float = 0.999,
        eps: float = 1e-8,
    ) -> None:
        self.lr = float(lr)
        self.beta1 = float(beta1)
        self.beta2 = float(beta2)
        self.eps = float(eps)
        self.m: dict[str, Array] = {}
        self.v: dict[str, Array] = {}
        self.t = 0

    def step(self, params: dict[str, Array], grads: dict[str, Array]) -> None:
        self.t += 1
        for key, param in params.items():
            grad = grads.get(key)
            if grad is None:
                continue
            if key not in self.m:
                self.m[key] = np.zeros_like(param)
                self.v[key] = np.zeros_like(param)
            self.m[key] = self.beta1 * self.m[key] + (1 - self.beta1) * grad
            self.v[key] = self.beta2 * self.v[key] + (1 - self.beta2) * grad * grad
            m_hat = self.m[key] / (1 - self.beta1 ** self.t)
            v_hat = self.v[key] / (1 - self.beta2 ** self.t)
            param -= self.lr * m_hat / (np.sqrt(v_hat) + self.eps)


# ---------------------------------------------------------------------------
# Mini YOLO detector
# ---------------------------------------------------------------------------
class MiniYOLO:
    """A tiny YOLO-v1-style detector implemented with NumPy.

    Architecture (input ``C x H x W``, defaults ``1 x 48 x 48``)::

        conv3x3(C -> c1) + ReLU + maxpool2
        conv3x3(c1 -> c2) + ReLU + maxpool2
        conv3x3(c2 -> c3) + ReLU
        conv3x3(c3 -> 5 + num_classes, stride=H/(4*S)) -> S x S grid

    The last convolution produces one prediction per grid cell.  ``B = 1``
    box per cell; the channels are ``[objectness, x_offset, y_offset,
    sqrt_w, sqrt_h, class logits]``.  ``x_offset`` and ``y_offset`` are direct
    (not sigmoided) cell-relative offsets, while ``sqrt_w`` and ``sqrt_h``
    decode to pixel size as ``(sqrt_w ** 2) * image_size``.
    """

    def __init__(
        self,
        num_classes: int = NUM_CLASSES,
        grid_size: int = DEFAULT_GRID_SIZE,
        image_size: int = DEFAULT_IMAGE_SIZE,
        in_channels: int = 1,
        base_channels: tuple[int, int] = (16, 32),
        head_channels: int = 32,
        seed: int = 0,
        dtype: Any = np.float32,
    ) -> None:
        self.num_classes = _positive_int(num_classes, "num_classes")
        self.grid_size = _positive_int(grid_size, "grid_size")
        self.image_size = _positive_int(image_size, "image_size", minimum=8)
        self.in_channels = _positive_int(in_channels, "in_channels")
        if image_size % 4 != 0 or (image_size // 4) % grid_size != 0:
            raise ValueError(
                "image_size must be divisible by 4 and (image_size / 4) "
                "must be divisible by grid_size"
            )
        c1, c2 = (int(base_channels[0]), int(base_channels[1]))
        if c1 <= 0 or c2 <= 0 or head_channels <= 0:
            raise ValueError("channel counts must be positive")
        head_channels = int(head_channels)
        self.dtype = np.dtype(dtype)
        self.base_channels = (c1, c2)
        self.head_channels = head_channels
        self.head_stride = (image_size // 4) // grid_size
        rng = np.random.default_rng(seed)
        self.conv1 = Conv2D(in_channels, c1, 3, 1, 1, rng, self.dtype)
        self.conv2 = Conv2D(c1, c2, 3, 1, 1, rng, self.dtype)
        self.conv3 = Conv2D(c2, head_channels, 3, 1, 1, rng, self.dtype)
        self.conv4 = Conv2D(
            head_channels,
            5 + self.num_classes,
            3,
            self.head_stride,
            1,
            rng,
            self.dtype,
        )
        self.pool1 = MaxPool2D()
        self.pool2 = MaxPool2D()
        self._cache: dict[str, Array | None] = {}

    # -- parameters and forward/backward -----------------------------------
    def parameters(self) -> dict[str, Array]:
        """Return a name -> array mapping of all trainable parameters."""
        return {
            "conv1_W": self.conv1.W,
            "conv1_b": self.conv1.b,
            "conv2_W": self.conv2.W,
            "conv2_b": self.conv2.b,
            "conv3_W": self.conv3.W,
            "conv3_b": self.conv3.b,
            "conv4_W": self.conv4.W,
            "conv4_b": self.conv4.b,
        }

    def _prepare_images(self, images: Array) -> Array:
        arr = np.asarray(images, dtype=self.dtype)
        if arr.ndim == 2:
            arr = arr[None, None]
        elif arr.ndim == 3:
            arr = arr[:, None]
        if arr.ndim != 4:
            raise ValueError("images must have shape (N, C, H, W)")
        if arr.shape[1] != self.in_channels:
            raise ValueError(
                f"expected {self.in_channels} input channels, got {arr.shape[1]}"
            )
        if arr.shape[2:] != (self.image_size, self.image_size):
            raise ValueError(
                f"expected {self.image_size}x{self.image_size} images, "
                f"got {arr.shape[2:]}"
            )
        return arr

    def forward(self, images: Array) -> Array:
        """Run the convolutional backbone and return raw ``(N, S, S, 5+C)``."""
        x = self._prepare_images(images)
        c1 = self.conv1.forward(x)
        r1 = relu_forward(c1)
        p1 = self.pool1.forward(r1)
        c2 = self.conv2.forward(p1)
        r2 = relu_forward(c2)
        p2 = self.pool2.forward(r2)
        c3 = self.conv3.forward(p2)
        r3 = relu_forward(c3)
        c4 = self.conv4.forward(r3)
        raw = c4.transpose(0, 2, 3, 1)
        self._cache = {"c1": c1, "c2": c2, "c3": c3}
        return raw

    def backward(self, dlogits: Array) -> dict[str, Array]:
        """Backpropagate ``dL/draw`` and return parameter and input grads."""
        if not self._cache:
            raise RuntimeError("forward must be called before backward")
        d = np.asarray(dlogits, dtype=float).transpose(0, 3, 1, 2)
        d = self.conv4.backward(d)
        d = relu_backward(d, self._cache["c3"])  # type: ignore[arg-type]
        d = self.conv3.backward(d)
        d = self.pool2.backward(d)
        d = relu_backward(d, self._cache["c2"])  # type: ignore[arg-type]
        d = self.conv2.backward(d)
        d = self.pool1.backward(d)
        d = relu_backward(d, self._cache["c1"])  # type: ignore[arg-type]
        d = self.conv1.backward(d)
        return {
            "conv1_W": self.conv1.dW,
            "conv1_b": self.conv1.db,
            "conv2_W": self.conv2.dW,
            "conv2_b": self.conv2.db,
            "conv3_W": self.conv3.dW,
            "conv3_b": self.conv3.db,
            "conv4_W": self.conv4.dW,
            "conv4_b": self.conv4.db,
            "input": d,
        }

    def loss_and_grad(
        self,
        images: Array,
        targets: Array,
        lambda_coord: float = LAMBDA_COORD,
        lambda_noobj: float = LAMBDA_NOOBJ,
    ) -> tuple[float, dict[str, Array]]:
        """Compute the YOLO loss and all parameter gradients for a batch."""
        logits = self.forward(images)
        loss, dlogits = yolo_loss_and_grad(
            logits, targets, lambda_coord, lambda_noobj
        )
        return loss, self.backward(dlogits)

    def fit(
        self,
        images: Array,
        annotations: Sequence[tuple[Any, Any]],
        epochs: int = 10,
        batch_size: int = 16,
        lr: float = 1e-3,
        seed: int = 0,
        verbose: bool = False,
    ) -> list[float]:
        """Train with Adam and return the per-epoch mean loss."""
        epochs = _positive_int(epochs, "epochs")
        batch_size = _positive_int(batch_size, "batch_size")
        images = self._prepare_images(images)
        if len(images) != len(annotations):
            raise ValueError("images and annotations must have the same length")
        targets = encode_targets_batch(
            annotations, self.image_size, self.grid_size, self.num_classes
        )
        if not np.isfinite(float(lr)) or lr <= 0:
            raise ValueError("lr must be a positive finite number")
        rng = np.random.default_rng(seed)
        optimizer = Adam(lr=lr)
        history: list[float] = []
        num_images = len(images)

        for epoch in range(epochs):
            order = rng.permutation(num_images)
            running = 0.0
            seen = 0
            for start in range(0, num_images, batch_size):
                index = order[start:start + batch_size]
                loss, grads = self.loss_and_grad(images[index], targets[index])
                optimizer.step(self.parameters(), grads)
                running += loss * len(index)
                seen += len(index)
            epoch_loss = running / max(seen, 1)
            history.append(float(epoch_loss))
            if verbose:
                print(f"epoch {epoch + 1:02d}/{epochs}: loss={epoch_loss:.4f}")
        return history

    # -- inference ---------------------------------------------------------
    def predict(
        self,
        images: Array,
        conf_threshold: float = 0.25,
        iou_threshold: float = 0.5,
        apply_nms: bool = True,
    ) -> list[tuple[Array, Array, Array]]:
        """Detect objects in one image or a batch of images."""
        raw = self.forward(images)
        return decode_predictions(
            raw,
            image_size=self.image_size,
            grid_size=self.grid_size,
            num_classes=self.num_classes,
            conf_threshold=conf_threshold,
            iou_threshold=iou_threshold,
            apply_nms=apply_nms,
        )


# ---------------------------------------------------------------------------
# Prediction decoding
# ---------------------------------------------------------------------------
def decode_single(
    raw: Array,
    image_size: int = DEFAULT_IMAGE_SIZE,
    grid_size: int = DEFAULT_GRID_SIZE,
    num_classes: int = NUM_CLASSES,
    conf_threshold: float = 0.25,
    iou_threshold: float = 0.5,
    apply_nms: bool = True,
) -> tuple[Array, Array, Array]:
    """Decode one ``(S, S, 5+C)`` raw head into detections.

    Each cell proposes one box.  The objectness probability is multiplied by
    the largest softmax class probability to obtain the detection score.  Low
    scores are dropped, boxes are converted to pixel ``(x1, y1, x2, y2)`` and
    class-aware NMS is applied.

    Returns
    -------
    boxes : ndarray of shape (M, 4)
    scores : ndarray of shape (M,)
    labels : ndarray of shape (M,)
        All three arrays are sorted by descending score.
    """
    arr = np.asarray(raw, dtype=float)
    _validate_target_shape(arr, grid_size, num_classes)
    if not 0.0 <= conf_threshold <= 1.0:
        raise ValueError("conf_threshold must be in [0, 1]")
    iou_threshold = _check_iou_threshold(iou_threshold)

    objectness = _sigmoid(arr[..., 0])
    class_probs = _softmax(arr[..., 5:], axis=-1)
    labels = np.argmax(class_probs, axis=-1)
    class_score = np.max(class_probs, axis=-1)
    scores = objectness * class_score
    mask = scores >= float(conf_threshold)
    if not np.any(mask):
        return (
            np.empty((0, 4), dtype=float),
            np.empty((0,), dtype=float),
            np.empty((0,), dtype=int),
        )

    cell = image_size / grid_size
    rows, cols = np.meshgrid(
        np.arange(grid_size), np.arange(grid_size), indexing="ij"
    )
    cx = (cols + arr[..., 1]) * cell
    cy = (rows + arr[..., 2]) * cell
    width = arr[..., 3] ** 2 * image_size
    height = arr[..., 4] ** 2 * image_size
    boxes = np.stack(
        [
            cx - width / 2.0,
            cy - height / 2.0,
            cx + width / 2.0,
            cy + height / 2.0,
        ],
        axis=-1,
    )

    boxes = boxes[mask]
    scores = scores[mask]
    labels = labels[mask]
    np.clip(boxes, 0.0, float(image_size), out=boxes)

    if apply_nms and boxes.shape[0] > 0:
        keep = per_class_nms(boxes, scores, labels, iou_threshold)
        boxes, scores, labels = boxes[keep], scores[keep], labels[keep]
    else:
        order = np.argsort(-scores, kind="stable")
        boxes, scores, labels = boxes[order], scores[order], labels[order]
    return boxes, scores, labels.astype(int)


def decode_predictions(
    raw: Array,
    image_size: int = DEFAULT_IMAGE_SIZE,
    grid_size: int = DEFAULT_GRID_SIZE,
    num_classes: int = NUM_CLASSES,
    conf_threshold: float = 0.25,
    iou_threshold: float = 0.5,
    apply_nms: bool = True,
) -> list[tuple[Array, Array, Array]]:
    """Decode a single raw head or a batch of raw heads.

    A ``(S, S, 5+C)`` input is treated as one image; ``(N, S, S, 5+C)`` is
    treated as a batch.  The return value is always a list with one
    ``(boxes, scores, labels)`` tuple per image.
    """
    arr = np.asarray(raw, dtype=float)
    if arr.ndim == 3:
        return [
            decode_single(
                arr, image_size, grid_size, num_classes,
                conf_threshold, iou_threshold, apply_nms,
            )
        ]
    if arr.ndim == 4:
        return [
            decode_single(
                image, image_size, grid_size, num_classes,
                conf_threshold, iou_threshold, apply_nms,
            )
            for image in arr
        ]
    raise ValueError("raw must have shape (S, S, 5+C) or (N, S, S, 5+C)")


# ---------------------------------------------------------------------------
# Mean average precision
# ---------------------------------------------------------------------------
def voc_average_precision(tp: Any, fp: Any, num_gt: int) -> float:
    """PASCAL VOC all-point average precision from matched detections.

    Parameters
    ----------
    tp, fp : array-like of bool
        Detection outcomes already ordered by descending score.
    num_gt : int
        Number of ground-truth boxes for this class.

    The precision envelope is made monotone from right to left, then AP is the
    area under the resulting precision/recall curve:
    ``sum((r_i - r_{i-1}) * p_i)``.
    """
    num_gt = int(num_gt)
    tp_arr = np.asarray(tp, dtype=bool).reshape(-1)
    fp_arr = np.asarray(fp, dtype=bool).reshape(-1)
    if tp_arr.size != fp_arr.size:
        raise ValueError("tp and fp must have the same length")
    if num_gt <= 0:
        return 0.0
    if tp_arr.size == 0:
        return 0.0
    tp_cum = np.cumsum(tp_arr)
    fp_cum = np.cumsum(fp_arr)
    recall = tp_cum / float(num_gt)
    precision = tp_cum / np.maximum(tp_cum + fp_cum, 1)
    precision = np.maximum.accumulate(precision[::-1])[::-1]
    delta_recall = np.diff(np.concatenate(([0.0], recall)))
    return float(np.sum(delta_recall * precision))


def mean_average_precision(
    predictions: Sequence[tuple[Any, Any, Any]],
    ground_truths: Sequence[tuple[Any, Any]],
    num_classes: int = NUM_CLASSES,
    iou_threshold: float = 0.5,
    return_per_class: bool = False,
) -> float | tuple[float, dict[int, float]]:
    """Mean average precision at a single IoU threshold (mAP@0.5).

    Detections from all images are pooled per class and sorted by score.  Each
    detection greedily matches the highest-IoU unmatched ground-truth box of
    the same class, provided that IoU is at least ``iou_threshold``.  A second
    detection on an already matched GT is a false positive.  AP uses VOC
    all-point interpolation.

    Classes without any ground truth are reported as 0.0 but excluded from the
    mean.  If no class has ground truth the result is 0.0.

    Parameters
    ----------
    predictions : sequence of (boxes, scores, labels)
    ground_truths : sequence of (boxes, labels)
    num_classes : int, default=3
    iou_threshold : float, default=0.5
    return_per_class : bool, default=False
        When True, return ``(map_value, {class_index: ap})``.
    """
    num_classes = _positive_int(num_classes, "num_classes")
    iou_threshold = _check_iou_threshold(iou_threshold)
    if len(predictions) != len(ground_truths):
        raise ValueError("predictions and ground_truths must have the same length")

    gt_by_class: dict[int, list[Array]] = {
        class_id: [] for class_id in range(num_classes)
    }
    for gt_boxes, gt_labels in ground_truths:
        box_arr, _ = _as_box_matrix(gt_boxes, "gt_boxes")
        label_arr = _as_1d(gt_labels, "gt_labels").astype(int)
        if box_arr.shape[0] != label_arr.size:
            raise ValueError("ground-truth boxes and labels differ in length")
        for class_id in range(num_classes):
            gt_by_class[class_id].append(box_arr[label_arr == class_id])

    matched = {
        class_id: [np.zeros(len(boxes), dtype=bool)
                   for boxes in gt_by_class[class_id]]
        for class_id in range(num_classes)
    }

    per_class: dict[int, float] = {}
    for class_id in range(num_classes):
        gt_count = int(sum(len(boxes) for boxes in gt_by_class[class_id]))
        detections: list[tuple[float, int, Array]] = []
        for image_index, (pred_boxes, pred_scores, pred_labels) in enumerate(
            predictions
        ):
            box_arr, _ = _as_box_matrix(pred_boxes, "pred_boxes")
            score_arr = _as_1d(pred_scores, "pred_scores").astype(float)
            label_arr = _as_1d(pred_labels, "pred_labels").astype(int)
            if not (box_arr.shape[0] == score_arr.size == label_arr.size):
                raise ValueError(
                    "prediction boxes, scores and labels differ in length"
                )
            for box, score, label in zip(
                box_arr, score_arr, label_arr, strict=True
            ):
                if label == class_id:
                    detections.append((float(score), image_index, box))
        detections.sort(key=lambda item: -item[0])
        tp = np.zeros(len(detections), dtype=bool)
        fp = np.zeros(len(detections), dtype=bool)
        for det_index, (_, image_index, box) in enumerate(detections):
            class_gt = gt_by_class[class_id][image_index]
            if class_gt.size == 0:
                fp[det_index] = True
                continue
            ious = iou_matrix(box[None, :], class_gt)[0]
            available = np.flatnonzero(~matched[class_id][image_index])
            if available.size == 0:
                fp[det_index] = True
                continue
            best_local = int(np.argmax(ious[available]))
            best_index = int(available[best_local])
            if ious[best_index] >= iou_threshold:
                tp[det_index] = True
                matched[class_id][image_index][best_index] = True
            else:
                fp[det_index] = True
        per_class[class_id] = voc_average_precision(tp, fp, gt_count)

    valid = [per_class[class_id] for class_id in range(num_classes)
             if sum(len(boxes) for boxes in gt_by_class[class_id]) > 0]
    map_value = float(np.mean(valid)) if valid else 0.0
    if return_per_class:
        return map_value, per_class
    return map_value


#: Short aliases used in the lesson text.
map_at_05 = mean_average_precision
evaluate_map = mean_average_precision


# ---------------------------------------------------------------------------
# Small end-to-end demonstration
# ---------------------------------------------------------------------------
def _demo() -> None:
    images, annotations = generate_shelf_dataset(n_samples=200, seed=7)
    train_images, train_annotations = images[:160], annotations[:160]
    test_images, test_annotations = images[160:], annotations[160:]

    model = MiniYOLO(seed=3)
    print("Training the MiniYOLO detector on 160 synthetic shelf images ...")
    history = model.fit(
        train_images,
        train_annotations,
        epochs=20,
        batch_size=16,
        lr=3e-3,
        seed=1,
        verbose=True,
    )
    predictions = model.predict(test_images, conf_threshold=0.2)
    map_value, per_class = mean_average_precision(
        predictions,
        test_annotations,
        return_per_class=True,
    )
    print(f"\nloss: {history[0]:.3f} -> {history[-1]:.3f}")
    print(f"mAP@0.5: {map_value:.3f}")
    for class_id, ap in per_class.items():
        print(f"  {CLASS_NAMES[class_id]:<7} AP={ap:.3f}")


if __name__ == "__main__":
    _demo()
