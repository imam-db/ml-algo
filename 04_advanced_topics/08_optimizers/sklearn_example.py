"""
NumPy optimizers vs scikit-learn's built-in solvers
===================================================

scikit-learn hides the optimizer inside each estimator: ``SGDClassifier`` runs
plain SGD with a decaying learning rate, ``LogisticRegression`` uses L-BFGS (a
quasi-Newton method) and ``MLPClassifier`` exposes ``solver="sgd"`` (with
optional Nesterov momentum) and ``solver="adam"``.

This script trains the same models with the from-scratch optimizers in
``implementation.py`` and with the matching scikit-learn solvers on synthetic
data, and prints a markdown table of accuracy and log-loss. Everything runs on
the CPU in a few seconds; nothing is plotted.

Author: ML-Algo Repository
"""

from __future__ import annotations

import os
import sys
import warnings

import numpy as np
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.metrics import log_loss
from sklearn.neural_network import MLPClassifier

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from implementation import (
    SGD,
    Adam,
    LogisticRegressionModel,
    MLPModel,
    Nesterov,
    accuracy,
    make_blobs_binary,
    make_moons_np,
    train,
)

warnings.filterwarnings("ignore", category=ConvergenceWarning)
RANDOM_STATE = 42


def row(name: str, acc: float, loss: float) -> str:
    return f"| {name} | {acc:.3f} | {loss:.4f} |"


def main() -> None:
    print("## Logistic regression on two Gaussian blobs\n")
    print("| Model / optimizer | Accuracy | Log-loss |")
    print("|---|---|---|")
    X, y = make_blobs_binary(600, seed=RANDOM_STATE)
    for name, opt in (("NumPy + SGD (lr=0.1)", SGD(lr=0.1)), ("NumPy + Adam (lr=0.05)", Adam(lr=0.05))):
        m = LogisticRegressionModel(2, seed=RANDOM_STATE)
        train(m, X, y, opt, epochs=30, batch_size=32, seed=RANDOM_STATE)
        print(row(name, accuracy(m, X, y), m.loss_and_grads(X, y)[0]))
    sk = SGDClassifier(loss="log_loss", learning_rate="constant", eta0=0.01, alpha=1e-6, max_iter=30, random_state=RANDOM_STATE).fit(X, y)
    print(row("sklearn SGDClassifier", sk.score(X, y), log_loss(y, sk.predict_proba(X))))
    lb = LogisticRegression(C=1e6).fit(X, y)
    print(row("sklearn LogisticRegression (L-BFGS)", lb.score(X, y), log_loss(y, lb.predict_proba(X))))

    print("\n## One-hidden-layer MLP (16 tanh units) on two moons\n")
    print("| Model / optimizer | Accuracy | Log-loss |")
    print("|---|---|---|")
    X, y = make_moons_np(400, seed=RANDOM_STATE)
    for name, opt in (("NumPy + Nesterov (lr=0.05)", Nesterov(lr=0.05)), ("NumPy + Adam (lr=0.01)", Adam(lr=0.01))):
        m = MLPModel(2, hidden=16, seed=RANDOM_STATE)
        train(m, X, y, opt, epochs=60, batch_size=32, seed=RANDOM_STATE)
        print(row(name, accuracy(m, X, y), m.loss_and_grads(X, y)[0]))
    for solver, extra in (("sgd", {"momentum": 0.9, "nesterovs_momentum": True, "learning_rate_init": 0.05}),
                          ("adam", {"learning_rate_init": 0.01})):
        mlp = MLPClassifier(hidden_layer_sizes=(16,), activation="tanh", solver=solver, alpha=0.0,
                            batch_size=32, max_iter=60, random_state=RANDOM_STATE, **extra).fit(X, y)
        print(row(f"sklearn MLPClassifier solver={solver}", mlp.score(X, y), log_loss(y, mlp.predict_proba(X))))

    print("\nNotes: identical data and epochs; sklearn adds its own init and stopping rules,")
    print("so small differences are expected. The point is that the update rules match.")


if __name__ == "__main__":
    np.random.seed(RANDOM_STATE)
    main()
