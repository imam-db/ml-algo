"""
Model Evaluation using scikit-learn
Comparison with from-scratch implementation
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split as sk_train_test_split
from sklearn.model_selection import cross_val_score as sk_cross_val_score
from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.metrics import (
    confusion_matrix as sk_confusion_matrix,
    accuracy_score as sk_accuracy,
    precision_score as sk_precision,
    recall_score as sk_recall,
    f1_score as sk_f1,
    roc_curve as sk_roc_curve,
    roc_auc_score as sk_roc_auc,
    classification_report as sk_classification_report
)
from sklearn.datasets import make_classification
from sklearn.linear_model import LogisticRegression

# Import our implementation
import sys
import os
import importlib.util

eval_path = os.path.join(os.path.dirname(__file__), 'implementation.py')
spec = importlib.util.spec_from_file_location("eval_impl", eval_path)
eval_impl = importlib.util.module_from_spec(spec)
spec.loader.exec_module(eval_impl)


def compare_train_test_split():
    """Compare custom and sklearn train_test_split."""
    print("="*70)
    print("COMPARING TRAIN/TEST SPLIT")
    print("="*70)
    
    # Generate data
    X, y = make_classification(n_samples=100, n_features=5, random_state=42)
    
    # Custom implementation
    X_train_c, X_test_c, y_train_c, y_test_c = eval_impl.train_test_split(
        X, y, test_size=0.3, random_state=42
    )
    
    # Sklearn implementation
    X_train_s, X_test_s, y_train_s, y_test_s = sk_train_test_split(
        X, y, test_size=0.3, random_state=42
    )
    
    print(f"\nCustom Implementation:")
    print(f"  Train size: {len(X_train_c)}, Test size: {len(X_test_c)}")
    
    print(f"\nSklearn Implementation:")
    print(f"  Train size: {len(X_train_s)}, Test size: {len(X_test_s)}")
    
    print(f"\n✓ Both implementations produce {100*(1-0.3):.0f}/{100*0.3:.0f} split")


def compare_metrics():
    """Compare classification metrics."""
    print("\n" + "="*70)
    print("COMPARING CLASSIFICATION METRICS")
    print("="*70)
    
    # Generate predictions
    y_true = np.array([0, 0, 1, 1, 0, 1, 1, 0, 1, 0])
    y_pred = np.array([0, 1, 1, 1, 0, 0, 1, 0, 1, 1])
    
    # Confusion Matrix
    print("\nConfusion Matrix:")
    cm_custom = eval_impl.confusion_matrix(y_true, y_pred)
    cm_sklearn = sk_confusion_matrix(y_true, y_pred)
    print(f"Custom:\n{cm_custom}")
    print(f"Sklearn:\n{cm_sklearn}")
    print(f"Match: {np.array_equal(cm_custom, cm_sklearn)}")
    
    # Accuracy
    print("\nAccuracy:")
    acc_custom = eval_impl.accuracy_score(y_true, y_pred)
    acc_sklearn = sk_accuracy(y_true, y_pred)
    print(f"Custom:  {acc_custom:.4f}")
    print(f"Sklearn: {acc_sklearn:.4f}")
    
    # Precision
    print("\nPrecision:")
    prec_custom = eval_impl.precision_score(y_true, y_pred)
    prec_sklearn = sk_precision(y_true, y_pred)
    print(f"Custom:  {prec_custom:.4f}")
    print(f"Sklearn: {prec_sklearn:.4f}")
    
    # Recall
    print("\nRecall:")
    rec_custom = eval_impl.recall_score(y_true, y_pred)
    rec_sklearn = sk_recall(y_true, y_pred)
    print(f"Custom:  {rec_custom:.4f}")
    print(f"Sklearn: {rec_sklearn:.4f}")
    
    # F1 Score
    print("\nF1 Score:")
    f1_custom = eval_impl.f1_score(y_true, y_pred)
    f1_sklearn = sk_f1(y_true, y_pred)
    print(f"Custom:  {f1_custom:.4f}")
    print(f"Sklearn: {f1_sklearn:.4f}")


def compare_roc_auc():
    """Compare ROC-AUC calculations."""
    print("\n" + "="*70)
    print("COMPARING ROC-AUC")
    print("="*70)
    
    # Generate data with probabilities
    np.random.seed(42)
    y_true = np.array([0, 0, 1, 1, 0, 1, 1, 0, 1, 0])
    y_scores = np.array([0.1, 0.4, 0.35, 0.8, 0.2, 0.6, 0.9, 0.3, 0.7, 0.45])
    
    # Custom AUC
    auc_custom = eval_impl.roc_auc_score(y_true, y_scores)
    
    # Sklearn AUC
    auc_sklearn = sk_roc_auc(y_true, y_scores)
    
    print(f"\nCustom ROC-AUC:  {auc_custom:.4f}")
    print(f"Sklearn ROC-AUC: {auc_sklearn:.4f}")
    print(f"Difference: {abs(auc_custom - auc_sklearn):.6f}")


def stratified_kfold_demo():
    """Demonstrate stratified K-fold for imbalanced data."""
    print("\n" + "="*70)
    print("STRATIFIED K-FOLD FOR IMBALANCED DATA")
    print("="*70)
    
    # Create imbalanced dataset
    X, y = make_classification(
        n_samples=100,
        n_features=10,
        weights=[0.9, 0.1],  # 90% class 0, 10% class 1
        random_state=42
    )
    
    print(f"\nDataset: {len(y)} samples")
    print(f"Class 0: {np.sum(y==0)} samples ({100*np.sum(y==0)/len(y):.1f}%)")
    print(f"Class 1: {np.sum(y==1)} samples ({100*np.sum(y==1)/len(y):.1f}%)")
    
    # Regular K-Fold
    print("\nRegular K-Fold (might not preserve class distribution):")
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    
    for i, (train_idx, val_idx) in enumerate(kf.split(X)):
        y_train, y_val = y[train_idx], y[val_idx]
        train_ratio = np.sum(y_train==1) / len(y_train)
        val_ratio = np.sum(y_val==1) / len(y_val)
        print(f"  Fold {i+1}: Train class 1: {100*train_ratio:.1f}%, Val class 1: {100*val_ratio:.1f}%")
    
    # Stratified K-Fold
    print("\nStratified K-Fold (preserves class distribution):")
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    
    for i, (train_idx, val_idx) in enumerate(skf.split(X, y)):
        y_train, y_val = y[train_idx], y[val_idx]
        train_ratio = np.sum(y_train==1) / len(y_train)
        val_ratio = np.sum(y_val==1) / len(y_val)
        print(f"  Fold {i+1}: Train class 1: {100*train_ratio:.1f}%, Val class 1: {100*val_ratio:.1f}%")


def cross_validation_demo():
    """Demonstrate cross-validation with different scoring metrics."""
    print("\n" + "="*70)
    print("CROSS-VALIDATION WITH DIFFERENT METRICS")
    print("="*70)
    
    # Generate data
    X, y = make_classification(
        n_samples=200,
        n_features=10,
        n_informative=8,
        random_state=42
    )
    
    # Model
    model = LogisticRegression(random_state=42)
    
    # Different scoring metrics
    scoring_metrics = ['accuracy', 'precision', 'recall', 'f1', 'roc_auc']
    
    print(f"\n5-Fold Cross-Validation Results:")
    print(f"{'Metric':<12} | {'Mean':>8} | {'Std':>8}")
    print("-" * 35)
    
    for metric in scoring_metrics:
        scores = sk_cross_val_score(model, X, y, cv=5, scoring=metric)
        print(f"{metric:<12} | {scores.mean():>8.4f} | {scores.std():>8.4f}")


def multiclass_metrics_demo():
    """Demonstrate multiclass classification metrics."""
    print("\n" + "="*70)
    print("MULTICLASS CLASSIFICATION METRICS")
    print("="*70)
    
    # Generate multiclass data
    X, y = make_classification(
        n_samples=300,
        n_features=10,
        n_informative=8,
        n_classes=3,
        random_state=42
    )
    
    X_train, X_test, y_train, y_test = sk_train_test_split(
        X, y, test_size=0.3, random_state=42
    )
    
    # Train model
    model = LogisticRegression(random_state=42, max_iter=200)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    
    # Different averaging strategies
    print("\nF1 Scores with Different Averaging:")
    
    for average in ['macro', 'micro', 'weighted']:
        f1 = sk_f1(y_test, y_pred, average=average)
        print(f"  {average.capitalize():>8}: {f1:.4f}")
    
    # Per-class metrics
    print("\nPer-Class F1 Scores:")
    f1_per_class = sk_f1(y_test, y_pred, average=None)
    for i, f1 in enumerate(f1_per_class):
        print(f"  Class {i}: {f1:.4f}")
    
    # Classification report
    print("\nClassification Report:")
    print(sk_classification_report(y_test, y_pred))


def confusion_matrix_analysis():
    """Analyze confusion matrix in detail."""
    print("\n" + "="*70)
    print("CONFUSION MATRIX DETAILED ANALYSIS")
    print("="*70)
    
    # Generate data
    X, y = make_classification(
        n_samples=200,
        n_features=10,
        random_state=42
    )
    
    X_train, X_test, y_train, y_test = sk_train_test_split(
        X, y, test_size=0.3, random_state=42
    )
    
    # Train model
    model = LogisticRegression(random_state=42)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    
    # Confusion matrix
    cm = sk_confusion_matrix(y_test, y_pred)
    
    print("\nConfusion Matrix:")
    print(cm)
    
    # Extract values
    tn, fp, fn, tp = cm.ravel()
    
    print("\nBreakdown:")
    print(f"  True Negatives  (TN): {tn}")
    print(f"  False Positives (FP): {fp}")
    print(f"  False Negatives (FN): {fn}")
    print(f"  True Positives  (TP): {tp}")
    
    # Derived metrics
    print("\nDerived Metrics:")
    print(f"  Accuracy:  {(tp + tn) / (tp + tn + fp + fn):.4f}")
    print(f"  Precision: {tp / (tp + fp):.4f}" if (tp + fp) > 0 else "  Precision: undefined")
    print(f"  Recall:    {tp / (tp + fn):.4f}" if (tp + fn) > 0 else "  Recall: undefined")
    print(f"  Specificity: {tn / (tn + fp):.4f}" if (tn + fp) > 0 else "  Specificity: undefined")


def threshold_tuning_demo():
    """Demonstrate threshold tuning for precision-recall trade-off."""
    print("\n" + "="*70)
    print("THRESHOLD TUNING: PRECISION-RECALL TRADE-OFF")
    print("="*70)
    
    # Generate data
    X, y = make_classification(
        n_samples=200,
        n_features=10,
        random_state=42
    )
    
    X_train, X_test, y_train, y_test = sk_train_test_split(
        X, y, test_size=0.3, random_state=42
    )
    
    # Train model
    model = LogisticRegression(random_state=42)
    model.fit(X_train, y_train)
    y_proba = model.predict_proba(X_test)[:, 1]
    
    print(f"\n{'Threshold':>10} | {'Precision':>10} | {'Recall':>10} | {'F1':>10}")
    print("-" * 47)
    
    # Test different thresholds
    for threshold in [0.3, 0.4, 0.5, 0.6, 0.7]:
        y_pred_thresh = (y_proba >= threshold).astype(int)
        
        prec = sk_precision(y_test, y_pred_thresh)
        rec = sk_recall(y_test, y_pred_thresh)
        f1 = sk_f1(y_test, y_pred_thresh)
        
        print(f"{threshold:>10.1f} | {prec:>10.4f} | {rec:>10.4f} | {f1:>10.4f}")


if __name__ == "__main__":
    # Run all demonstrations
    compare_train_test_split()
    compare_metrics()
    compare_roc_auc()
    stratified_kfold_demo()
    cross_validation_demo()
    multiclass_metrics_demo()
    confusion_matrix_analysis()
    threshold_tuning_demo()
    
    print("\n" + "="*70)
    print("✓ All sklearn Model Evaluation demonstrations completed!")
    print("="*70)
