"""
Sliding-window baseline vs the NumPy MiniYOLO detector
======================================================

This script compares two object detectors on exactly the same synthetic
small-shop shelf images:

1. **Sliding window + LogisticRegression** -- a classical baseline.  A few
   rectangular window sizes are slid over the image with a fixed stride.  Each
   patch is described by raw downsampled pixels plus a small HOG-like gradient
   histogram, and a multinomial logistic regression classifies it as
   background or one of the three product classes.  Windows are then filtered
   by score and passed through per-class NMS.
2. **MiniYOLO** -- the from-scratch NumPy detector in ``implementation.py``
   (``S = 6``, one box per cell, manual backpropagation and Adam).

Both models are trained on one synthetic split and evaluated on a held-out
split with mAP@0.5 and measured inference time per image.  Everything runs on
the CPU and finishes in well under a minute.  Only a markdown table and a few
notes are printed, so the script is safe to run in a headless environment.

Author: ML-Algo Repository
"""

from __future__ import annotations

import os
import sys
import time
import warnings

import numpy as np
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from implementation import (
    CLASS_NAMES,
    MiniYOLO,
    generate_shelf_dataset,
    iou_matrix,
    mean_average_precision,
    per_class_nms,
)

warnings.filterwarnings("ignore", category=ConvergenceWarning)

RANDOM_STATE = 42
IMAGE_SIZE = 48
N_TRAIN = 200
N_TEST = 40
# (height, width) windows; two aspect ratios cover cartons and tall bottles.
WINDOWS = ((12, 12), (16, 14), (20, 16), (12, 20))
STRIDE = 8
PATCH_SIZE = 10
ORIENTATION_BINS = 9
SCORE_THRESHOLD = 0.35
NMS_IOU = 0.5


# ---------------------------------------------------------------------------
# Sliding-window baseline
# ---------------------------------------------------------------------------
def _resize_patch(patch: np.ndarray, size: int) -> np.ndarray:
    """Nearest-neighbour resize of a small 2-D patch."""
    rows = np.linspace(0, patch.shape[0] - 1, size).round().astype(int)
    cols = np.linspace(0, patch.shape[1] - 1, size).round().astype(int)
    return patch[np.ix_(rows, cols)]


def patch_features(patch: np.ndarray) -> np.ndarray:
    """Raw downsampled pixels plus a compact HOG-like descriptor.

    The patch is converted to grayscale, resized to ``PATCH_SIZE`` and then to
    an 8x8 intensity thumbnail.  Gradients are binned into nine orientations
    inside each quadrant of a 2x2 grid, giving 36 additional shape features.
    The vector is deliberately small so the classical baseline stays fast.
    """
    if patch.ndim == 3:
        patch = patch.mean(axis=0)
    small = _resize_patch(patch, PATCH_SIZE)
    thumbnail = _resize_patch(small, 8)
    grad_y, grad_x = np.gradient(small)
    magnitude = np.sqrt(grad_x * grad_x + grad_y * grad_y)
    angle = np.mod(np.arctan2(grad_y, grad_x), np.pi)
    bins = np.minimum(
        (angle / np.pi * ORIENTATION_BINS).astype(int),
        ORIENTATION_BINS - 1,
    )
    histograms = []
    half = PATCH_SIZE // 2
    for row_start in (0, half):
        for col_start in (0, half):
            cell_bins = bins[row_start:row_start + half,
                             col_start:col_start + half]
            cell_magnitude = magnitude[row_start:row_start + half,
                                       col_start:col_start + half]
            hist = np.bincount(
                cell_bins.ravel(),
                weights=cell_magnitude.ravel(),
                minlength=ORIENTATION_BINS,
            )
            histograms.append(hist / (hist.sum() + 1e-9))
    return np.concatenate([thumbnail.ravel()] + histograms)


def window_box(
    image_size: int,
    row: int,
    col: int,
    height: int,
    width: int,
) -> np.ndarray:
    """Return a window as an ``(x1, y1, x2, y2)`` box."""
    del image_size  # kept in the signature for readability at call sites
    return np.array(
        [col, row, col + width, row + height], dtype=float
    )


def iter_windows(image_size: int):
    """Yield ``(row, col, height, width)`` for every sliding window."""
    for height, width in WINDOWS:
        for row in range(0, image_size - height + 1, STRIDE):
            for col in range(0, image_size - width + 1, STRIDE):
                yield row, col, height, width


def collect_baseline_data(
    images: np.ndarray,
    annotations,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray]:
    """Build patch features and labels (0 = background, 1..3 = product)."""
    features: list[np.ndarray] = []
    labels: list[int] = []
    for image, (boxes, box_labels) in zip(images, annotations, strict=True):
        background: list[np.ndarray] = []
        for row, col, height, width in iter_windows(IMAGE_SIZE):
            patch = image[:, row:row + height, col:col + width]
            feature = patch_features(patch)
            window = window_box(IMAGE_SIZE, row, col, height, width)
            overlaps = iou_matrix(window[None, :], boxes)[0]
            best = int(np.argmax(overlaps)) if overlaps.size else -1
            if best >= 0 and overlaps[best] >= 0.5:
                features.append(feature)
                labels.append(int(box_labels[best]) + 1)
            else:
                background.append(feature)
        # Keep only a random subset of the easy background windows.  The
        # classifier still sees every window at inference time.
        keep = min(len(background), 12)
        if keep:
            chosen = rng.choice(len(background), size=keep, replace=False)
            for index in chosen:
                features.append(background[int(index)])
                labels.append(0)
    return np.stack(features), np.asarray(labels, dtype=int)


def train_sliding_window(
    images: np.ndarray,
    annotations,
    seed: int = RANDOM_STATE,
) -> tuple[StandardScaler, LogisticRegression, float]:
    """Fit the feature scaler and multinomial logistic-regression baseline."""
    rng = np.random.default_rng(seed)
    started = time.perf_counter()
    features, labels = collect_baseline_data(images, annotations, rng)
    scaler = StandardScaler().fit(features)
    classifier = LogisticRegression(
        max_iter=400,
        C=2.0,
        class_weight="balanced",
        random_state=seed,
    )
    classifier.fit(scaler.transform(features), labels)
    elapsed = time.perf_counter() - started
    return scaler, classifier, elapsed


def sliding_window_detect(
    image: np.ndarray,
    scaler: StandardScaler,
    classifier: LogisticRegression,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Run the sliding-window classifier and per-class NMS on one image."""
    windows: list[np.ndarray] = []
    features: list[np.ndarray] = []
    for row, col, height, width in iter_windows(IMAGE_SIZE):
        windows.append(window_box(IMAGE_SIZE, row, col, height, width))
        features.append(patch_features(image[:, row:row + height,
                                             col:col + width]))
    probabilities = classifier.predict_proba(scaler.transform(features))
    classes = classifier.classes_
    product_columns = np.flatnonzero(classes > 0)
    boxes = np.asarray(windows, dtype=float)
    if product_columns.size == 0:
        return (
            np.empty((0, 4), dtype=float),
            np.empty((0,), dtype=float),
            np.empty((0,), dtype=int),
        )
    product_probs = probabilities[:, product_columns]
    best_local = np.argmax(product_probs, axis=1)
    scores = product_probs[np.arange(len(windows)), best_local]
    labels = classes[product_columns][best_local] - 1
    keep = scores >= SCORE_THRESHOLD
    boxes, scores, labels = boxes[keep], scores[keep], labels[keep]
    if boxes.shape[0] == 0:
        return boxes, scores, labels.astype(int)
    kept = per_class_nms(boxes, scores, labels, NMS_IOU)
    return boxes[kept], scores[kept], labels[kept].astype(int)


# ---------------------------------------------------------------------------
# Evaluation helpers
# ---------------------------------------------------------------------------
def measure_inference(
    detect,
    images: np.ndarray,
) -> tuple[list[tuple[np.ndarray, np.ndarray, np.ndarray]], float]:
    """Run ``detect`` per image and return predictions plus ms/image."""
    predictions = []
    started = time.perf_counter()
    for image in images:
        predictions.append(detect(image))
    elapsed = time.perf_counter() - started
    return predictions, 1000.0 * elapsed / len(images)


def format_table(rows: list[dict[str, float | str]]) -> None:
    """Print a markdown comparison table."""
    print("\n### Results on the held-out synthetic set\n")
    print("| Model | mAP@0.5 | Inference (ms/image) | Train (s) |")
    print("|---|---:|---:|---:|")
    for row in rows:
        print(
            f"| {row['model']} | {row['map']:.3f} | "
            f"{row['inference_ms']:.1f} | {row['train_s']:.1f} |"
        )


def print_per_class(model_name: str, per_class: dict[int, float]) -> None:
    values = " | ".join(
        f"{CLASS_NAMES[class_id]} {per_class[class_id]:.3f}"
        for class_id in sorted(per_class)
    )
    print(f"- {model_name}: {values}")


# ---------------------------------------------------------------------------
# Main comparison
# ---------------------------------------------------------------------------
def main() -> None:
    """Train both detectors, evaluate, and print a markdown table."""
    print("=" * 78)
    print("Object detection: sliding window vs MiniYOLO (synthetic shelf data)")
    print("=" * 78)
    images, annotations = generate_shelf_dataset(
        n_samples=N_TRAIN + N_TEST,
        image_size=IMAGE_SIZE,
        channels=1,
        seed=RANDOM_STATE,
    )
    train_images, train_annotations = images[:N_TRAIN], annotations[:N_TRAIN]
    test_images, test_annotations = images[N_TRAIN:], annotations[N_TRAIN:]
    objects = sum(len(labels) for _, labels in annotations)
    print(f"Images       : {len(images)} (train {N_TRAIN}, test {N_TEST})")
    print(f"Objects      : {objects} across {len(CLASS_NAMES)} classes")
    print(f"Image shape  : {images.shape[1:]}")
    print(f"Windows      : {WINDOWS}, stride {STRIDE}")
    print("All images are synthetic; no real photos are used.\n")

    # 1) Sliding window + logistic regression ------------------------------
    print("Training the sliding-window LogisticRegression baseline ...")
    scaler, classifier, baseline_train_s = train_sliding_window(
        train_images, train_annotations
    )
    print(f"  fitted in {baseline_train_s:.1f}s")
    baseline_predictions, baseline_ms = measure_inference(
        lambda image: sliding_window_detect(image, scaler, classifier),
        test_images,
    )
    baseline_map, baseline_per_class = mean_average_precision(
        baseline_predictions,
        test_annotations,
        return_per_class=True,
    )
    print(f"  mAP@0.5 = {baseline_map:.3f}, {baseline_ms:.1f} ms/image")

    # 2) MiniYOLO ----------------------------------------------------------
    print("Training the from-scratch MiniYOLO detector ...")
    detector = MiniYOLO(
        num_classes=len(CLASS_NAMES),
        grid_size=6,
        image_size=IMAGE_SIZE,
        in_channels=1,
        seed=RANDOM_STATE,
    )
    started = time.perf_counter()
    history = detector.fit(
        train_images,
        train_annotations,
        epochs=30,
        batch_size=16,
        lr=3e-3,
        seed=RANDOM_STATE,
        verbose=False,
    )
    detector_train_s = time.perf_counter() - started
    print(
        f"  loss {history[0]:.2f} -> {history[-1]:.2f} "
        f"in {detector_train_s:.1f}s"
    )
    detector_predictions, detector_ms = measure_inference(
        lambda image: detector.predict(image[None], conf_threshold=0.2)[0],
        test_images,
    )
    detector_map, detector_per_class = mean_average_precision(
        detector_predictions,
        test_annotations,
        return_per_class=True,
    )
    print(f"  mAP@0.5 = {detector_map:.3f}, {detector_ms:.1f} ms/image")

    rows = [
        {
            "model": "Sliding window + LogisticRegression",
            "map": baseline_map,
            "inference_ms": baseline_ms,
            "train_s": baseline_train_s,
        },
        {
            "model": "MiniYOLO (NumPy, from scratch)",
            "map": detector_map,
            "inference_ms": detector_ms,
            "train_s": detector_train_s,
        },
    ]
    format_table(rows)
    print("\nPer-class AP@0.5")
    print_per_class("Sliding window", baseline_per_class)
    print_per_class("MiniYOLO", detector_per_class)

    print("\nNotes")
    print("-----")
    print(
        "- The sliding window is a strong, simple baseline, but its boxes are "
        "axis-aligned windows with fixed sizes."
    )
    print(
        "- MiniYOLO predicts the box centre and size directly from a 6x6 grid, "
        "so it can localise objects more tightly after only a few epochs."
    )
    print(
        "- The baseline runs scikit-learn inference on ~85 patches per image; "
        "the detector performs one small convolution pass."
    )
    print(
        "- These numbers come from synthetic data. Real shelves add lighting, "
        "occlusion, perspective and background clutter."
    )


if __name__ == "__main__":
    main()
