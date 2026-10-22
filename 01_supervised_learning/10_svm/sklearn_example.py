"""
Support Vector Machines: Scikit-Learn Examples

This script demonstrates how scikit-learn's ``SVC`` behaves on synthetic
outlet-performance data:

1. Linear SVM with different values of the regularization parameter ``C``.
2. RBF SVM with different values of ``C``.
3. RBF SVM with different values of ``gamma`` (kernel width).
4. A small combined ``C`` x ``gamma`` grid and the best configuration.

Only printed accuracies are produced, so the script is safe to run in a
headless environment. If you add plots, select a non-interactive backend
first (``import matplotlib; matplotlib.use("Agg")``).

Author: ML-Algo Repository
"""

import os
import sys

import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

# Import the shared synthetic-data generator.
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from implementation import OUTLET_FEATURE_NAMES, generate_outlet_data


def prepare_outlet_data(n_samples=400, random_state=42):
    """Generate, split and standardize the synthetic outlet dataset."""
    X, y = generate_outlet_data(n_samples=n_samples, random_state=random_state)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=random_state, stratify=y
    )

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    print(f"Dataset: {len(X_train)} training / {len(X_test)} test outlets")
    print(f"Features: {', '.join(OUTLET_FEATURE_NAMES)}")
    print(f"High-performing share (train): {np.mean(y_train):.1%}")
    print(f"High-performing share (test):  {np.mean(y_test):.1%}")
    return X_train_scaled, X_test_scaled, y_train, y_test


def example_1_linear_svm_regularization(X_train, X_test, y_train, y_test):
    """Show the effect of C on a linear SVM."""
    print("\n" + "=" * 78)
    print("Example 1: Linear SVM - effect of C")
    print("=" * 78)
    print(
        "Small C => strong regularization (wider margin, more bias).\n"
        "Large C => weak regularization (narrower margin, more variance).\n"
    )

    print(f"{'C':>8} {'train acc':>11} {'test acc':>10} {'support vectors':>17}")
    print("-" * 50)

    results = []
    for C in [0.01, 0.1, 1.0, 10.0, 100.0]:
        model = SVC(kernel="linear", C=C)
        model.fit(X_train, y_train)

        train_acc = accuracy_score(y_train, model.predict(X_train))
        test_acc = accuracy_score(y_test, model.predict(X_test))
        results.append((C, test_acc, int(np.sum(model.n_support_))))

        print(
            f"{C:>8.2f} {train_acc:>11.4f} {test_acc:>10.4f} "
            f"{int(np.sum(model.n_support_)):>17d}"
        )

    best_C, best_acc, _ = max(results, key=lambda item: item[1])
    print(f"\nBest linear C in this sweep: {best_C:g} (test accuracy {best_acc:.4f})")
    return results


def example_2_rbf_svm_regularization(X_train, X_test, y_train, y_test):
    """Show the effect of C on an RBF SVM with fixed gamma."""
    print("\n" + "=" * 78)
    print("Example 2: RBF SVM - effect of C (gamma='scale')")
    print("=" * 78)
    print(
        "C controls how much the model is allowed to violate the margin.\n"
        "Very large C can fit noise; very small C underfits.\n"
    )

    print(f"{'C':>8} {'train acc':>11} {'test acc':>10} {'support vectors':>17}")
    print("-" * 50)

    results = []
    for C in [0.01, 0.1, 1.0, 10.0, 100.0]:
        model = SVC(kernel="rbf", C=C, gamma="scale")
        model.fit(X_train, y_train)

        train_acc = accuracy_score(y_train, model.predict(X_train))
        test_acc = accuracy_score(y_test, model.predict(X_test))
        results.append((C, test_acc, int(np.sum(model.n_support_))))

        print(
            f"{C:>8.2f} {train_acc:>11.4f} {test_acc:>10.4f} "
            f"{int(np.sum(model.n_support_)):>17d}"
        )

    best_C, best_acc, _ = max(results, key=lambda item: item[1])
    print(f"\nBest RBF C in this sweep: {best_C:g} (test accuracy {best_acc:.4f})")
    return results


def example_3_rbf_svm_gamma(X_train, X_test, y_train, y_test):
    """Show the effect of gamma (kernel width) on an RBF SVM."""
    print("\n" + "=" * 78)
    print("Example 3: RBF SVM - effect of gamma (C=10)")
    print("=" * 78)
    print(
        "Small gamma => smooth, wide influence (underfitting risk).\n"
        "Large gamma => narrow, local influence (overfitting risk).\n"
    )

    print(f"{'gamma':>10} {'train acc':>11} {'test acc':>10} {'support vectors':>17}")
    print("-" * 52)

    results = []
    for gamma in [0.01, 0.03, 0.1, 0.3, 1.0, 3.0]:
        model = SVC(kernel="rbf", C=10.0, gamma=gamma)
        model.fit(X_train, y_train)

        train_acc = accuracy_score(y_train, model.predict(X_train))
        test_acc = accuracy_score(y_test, model.predict(X_test))
        results.append((gamma, test_acc, int(np.sum(model.n_support_))))

        print(
            f"{gamma:>10.3f} {train_acc:>11.4f} {test_acc:>10.4f} "
            f"{int(np.sum(model.n_support_)):>17d}"
        )

    best_gamma, best_acc, _ = max(results, key=lambda item: item[1])
    print(
        f"\nBest gamma in this sweep: {best_gamma:g} "
        f"(test accuracy {best_acc:.4f})"
    )
    return results


def example_4_combined_grid(X_train, X_test, y_train, y_test):
    """Print a small C x gamma grid and report the best configuration."""
    print("\n" + "=" * 78)
    print("Example 4: RBF SVM - combined C x gamma grid")
    print("=" * 78)

    C_values = [0.1, 1.0, 10.0, 100.0]
    gamma_values = [0.01, 0.1, 0.5, 1.0]

    print(f"{'C \\ gamma':>12}", end="")
    for gamma in gamma_values:
        print(f"{gamma:>10.3f}", end="")
    print()
    print("-" * (12 + 10 * len(gamma_values)))

    best = None
    for C in C_values:
        print(f"{C:>12.2f}", end="")
        for gamma in gamma_values:
            model = SVC(kernel="rbf", C=C, gamma=gamma)
            model.fit(X_train, y_train)
            test_acc = accuracy_score(y_test, model.predict(X_test))
            print(f"{test_acc:>10.4f}", end="")
            if best is None or test_acc > best[2]:
                best = (C, gamma, test_acc)
        print()

    print(
        f"\nBest grid configuration: C={best[0]:g}, gamma={best[1]:g}, "
        f"test accuracy={best[2]:.4f}"
    )
    return best


def main():
    """Run all SVM examples."""
    print("=" * 78)
    print("SUPPORT VECTOR MACHINES: Scikit-Learn Examples")
    print("Synthetic outlet performance classification")
    print("=" * 78)

    X_train, X_test, y_train, y_test = prepare_outlet_data(
        n_samples=400, random_state=42
    )

    example_1_linear_svm_regularization(X_train, X_test, y_train, y_test)
    example_2_rbf_svm_regularization(X_train, X_test, y_train, y_test)
    example_3_rbf_svm_gamma(X_train, X_test, y_train, y_test)
    example_4_combined_grid(X_train, X_test, y_train, y_test)

    print("\n" + "=" * 78)
    print("Key takeaways")
    print("=" * 78)
    print("- Standardize features before using an SVM; distance-based kernels")
    print("  otherwise let the largest-scale feature dominate.")
    print("- C balances margin width against training errors.")
    print("- gamma controls the reach of each RBF kernel bump: too small")
    print("  underfits, too large creates isolated, overfit decision regions.")
    print("- The traffic-dependent benchmark is non-linear. Kernel SVMs can")
    print("  bend around it, while a well-tuned linear SVM is already a strong")
    print("  baseline because the underlying signal is informative.")
    print("=" * 78)


if __name__ == "__main__":
    main()
