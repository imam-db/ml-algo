"""
CNN vs classical models on the synthetic warung-shelf dataset
=============================================================

This script compares three classifiers on exactly the same generated images:

1. ``ShelfCNN`` -- the from-scratch NumPy CNN in ``implementation.py``.
2. ``MLPClassifier`` -- a scikit-learn multilayer perceptron on flattened pixels.
3. ``LogisticRegression`` -- a linear baseline on flattened pixels.

The interesting part is the **translation-shift test set**: every test image is
moved a few pixels down and right.  A convolutional network shares weights
across positions and pools local responses, so a small translation should hurt
it much less than a dense model that only sees absolute pixel coordinates.

Everything runs on the CPU and finishes in well under a minute.  The script
only prints a results table, so it is safe to run in a headless environment.

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
from sklearn.neural_network import MLPClassifier

# Import the shared implementation from this directory.
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from implementation import (
    PRODUCT_CLASS_NAMES,
    ShelfCNN,
    generate_shelf_images,
    shift_images,
)

warnings.filterwarnings("ignore", category=ConvergenceWarning)

RANDOM_STATE = 42
N_TRAIN = 400
N_TEST = 160
SHIFT_Y = 2
SHIFT_X = 2


def make_data() -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Generate the clean test set and a translated copy of it."""
    X_train, y_train = generate_shelf_images(
        n_samples=N_TRAIN, image_size=16, n_classes=4, noise=0.05,
        random_state=RANDOM_STATE,
    )
    X_test, y_test = generate_shelf_images(
        n_samples=N_TEST, image_size=16, n_classes=4, noise=0.05,
        random_state=RANDOM_STATE + 1,
    )
    X_shifted = shift_images(X_test, shift_y=SHIFT_Y, shift_x=SHIFT_X, fill=0.15)

    print("=" * 74)
    print("CNN vs classical models: warung shelf product recognition")
    print("=" * 74)
    print(f"Train images : {X_train.shape}")
    print(f"Test images  : {X_test.shape}")
    print(f"Shifted test : {X_shifted.shape} (moved {SHIFT_Y}px down, {SHIFT_X}px right)")
    print(f"Classes      : {', '.join(PRODUCT_CLASS_NAMES)}")
    print("All images are synthetic 16x16 greyscale crops.\n")
    return X_train, y_train, X_test, y_test, X_shifted


def flatten(X: np.ndarray) -> np.ndarray:
    """Flatten ``(N, 1, H, W)`` images to ``(N, H*W)`` feature vectors."""
    return X.reshape(len(X), -1)


def evaluate(
    name: str,
    predict,
    X_test: np.ndarray,
    y_test: np.ndarray,
    X_shifted: np.ndarray,
    elapsed: float,
) -> dict[str, float]:
    """Evaluate a fitted model on the clean and shifted test sets."""
    clean = float(np.mean(predict(X_test) == y_test))
    shifted = float(np.mean(predict(X_shifted) == y_test))
    drop = clean - shifted
    return {
        "model": name,
        "clean": clean,
        "shifted": shifted,
        "drop": drop,
        "seconds": elapsed,
    }


def main() -> None:
    """Train all three models and print a comparison table."""
    X_train, y_train, X_test, y_test, X_shifted = make_data()
    X_train_flat = flatten(X_train)
    X_test_flat = flatten(X_test)
    X_shifted_flat = flatten(X_shifted)

    results = []

    # 1) From-scratch CNN ---------------------------------------------------
    print("Training from-scratch ShelfCNN ...")
    started = time.perf_counter()
    cnn = ShelfCNN(n_classes=4, image_size=16, random_state=RANDOM_STATE)
    cnn_history = cnn.fit(
        X_train,
        y_train,
        epochs=12,
        batch_size=32,
        lr=0.02,
        momentum=0.9,
    )
    cnn_seconds = time.perf_counter() - started
    results.append(
        evaluate(
            "CNN (NumPy, from scratch)",
            cnn.predict,
            X_test,
            y_test,
            X_shifted,
            cnn_seconds,
        )
    )
    print(
        f"  loss {cnn_history[0]:.3f} -> {cnn_history[-1]:.3f} "
        f"in {cnn_seconds:.2f}s"
    )

    # 2) scikit-learn MLP on flattened pixels --------------------------------
    print("Training sklearn MLPClassifier on flattened pixels ...")
    started = time.perf_counter()
    mlp = MLPClassifier(
        hidden_layer_sizes=(64,),
        activation="relu",
        solver="adam",
        learning_rate_init=1e-3,
        max_iter=400,
        random_state=RANDOM_STATE,
    )
    mlp.fit(X_train_flat, y_train)
    mlp_seconds = time.perf_counter() - started
    results.append(
        evaluate(
            "MLP (sklearn, flattened)",
            mlp.predict,
            X_test_flat,
            y_test,
            X_shifted_flat,
            mlp_seconds,
        )
    )
    print(f"  {mlp.n_iter_} iterations in {mlp_seconds:.2f}s")

    # 3) Logistic regression baseline ---------------------------------------
    print("Training LogisticRegression on flattened pixels ...")
    started = time.perf_counter()
    logistic = LogisticRegression(max_iter=1000, random_state=RANDOM_STATE)
    logistic.fit(X_train_flat, y_train)
    logistic_seconds = time.perf_counter() - started
    results.append(
        evaluate(
            "Logistic regression",
            logistic.predict,
            X_test_flat,
            y_test,
            X_shifted_flat,
            logistic_seconds,
        )
    )
    print(f"  fitted in {logistic_seconds:.2f}s\n")

    # Results table ---------------------------------------------------------
    chance = 1.0 / 4.0
    print("=" * 74)
    print("RESULTS (accuracy, chance level = 25%)")
    print("=" * 74)
    print(
        f"{'Model':<30} {'Clean':>8} {'Shifted':>9} {'Drop':>8} {'Time':>8}"
    )
    print("-" * 74)
    for row in results:
        print(
            f"{row['model']:<30} {row['clean']:>7.1%} {row['shifted']:>8.1%} "
            f"{row['drop']:>7.1%} {row['seconds']:>7.2f}s"
        )
    print("-" * 74)
    print(f"{'Chance level':<30} {chance:>7.1%} {chance:>8.1%}")

    cnn_row = results[0]
    mlp_row = results[1]
    print("\nKey observations")
    print("----------------")
    print(
        f"- The CNN keeps {cnn_row['shifted']:.1%} accuracy after the "
        f"{SHIFT_Y}px/{SHIFT_X}px shift, a drop of only {cnn_row['drop']:.1%}."
    )
    print(
        f"- The MLP falls from {mlp_row['clean']:.1%} to "
        f"{mlp_row['shifted']:.1%} accuracy (drop {mlp_row['drop']:.1%})."
    )
    print(
        "- Dense models treat each pixel coordinate independently, so moving "
        "the product changes their input distribution."
    )
    print(
        "- Convolution shares the same filters at every position and pooling "
        "aggregates local evidence. That inductive bias is why CNNs are the "
        "default choice for images."
    )
    print(
        "- Logistic regression is a useful sanity-check baseline, but it can "
        "only learn a linear boundary in pixel space."
    )
    print("\nNote: this is synthetic data. Real shelf images add lighting,")
    print("occlusion, perspective and background clutter.")


if __name__ == "__main__":
    main()
