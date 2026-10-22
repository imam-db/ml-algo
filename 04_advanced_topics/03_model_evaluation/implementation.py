"""
Model Evaluation Metrics and Techniques - Implementation from Scratch

Train/test split, confusion matrix, classification metrics, ROC-AUC, cross-validation.
"""

import numpy as np
from typing import Tuple, List, Optional, Callable
import matplotlib.pyplot as plt


def train_test_split(X: np.ndarray, y: np.ndarray, 
                     test_size: float = 0.2, 
                     random_state: Optional[int] = None,
                     shuffle: bool = True) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Split arrays into random train and test subsets.
    
    Args:
        X: Features array
        y: Labels array
        test_size: Proportion of dataset in test split
        random_state: Random seed
        shuffle: Whether to shuffle before split
        
    Returns:
        X_train, X_test, y_train, y_test
    """
    if random_state is not None:
        np.random.seed(random_state)
    
    n_samples = len(X)
    n_test = int(n_samples * test_size)
    n_train = n_samples - n_test
    
    indices = np.arange(n_samples)
    if shuffle:
        np.random.shuffle(indices)
    
    # Split indices: first n_train for training, rest for test
    train_indices = indices[:n_train]
    test_indices = indices[n_train:]
    
    X_train = X[train_indices]
    X_test = X[test_indices]
    y_train = y[train_indices]
    y_test = y[test_indices]
    
    return X_train, X_test, y_train, y_test


def confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
    """
    Compute confusion matrix.
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
        
    Returns:
        Confusion matrix where C[i, j] is number of observations 
        known to be in group i but predicted to be in group j
    """
    classes = np.unique(np.concatenate([y_true, y_pred]))
    n_classes = len(classes)
    
    # Map class labels to indices
    class_to_idx = {c: idx for idx, c in enumerate(classes)}
    
    # Initialize confusion matrix
    cm = np.zeros((n_classes, n_classes), dtype=int)
    
    # Fill confusion matrix
    for true_label, pred_label in zip(y_true, y_pred):
        true_idx = class_to_idx[true_label]
        pred_idx = class_to_idx[pred_label]
        cm[true_idx, pred_idx] += 1
    
    return cm


def accuracy_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Calculate accuracy: (TP + TN) / total."""
    return np.mean(y_true == y_pred)


def precision_score(y_true: np.ndarray, y_pred: np.ndarray, 
                   pos_label: int = 1, average: Optional[str] = 'binary') -> float:
    """
    Calculate precision: TP / (TP + FP).
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
        pos_label: Positive class label (for binary)
        average: 'binary', 'macro', 'micro', 'weighted', or None
        
    Returns:
        Precision score(s)
    """
    if average == 'binary':
        tp = np.sum((y_true == pos_label) & (y_pred == pos_label))
        fp = np.sum((y_true != pos_label) & (y_pred == pos_label))
        
        if tp + fp == 0:
            return 0.0
        return tp / (tp + fp)
    
    elif average in ['macro', 'micro', 'weighted']:
        classes = np.unique(y_true)
        precisions = []
        supports = []
        
        for c in classes:
            tp = np.sum((y_true == c) & (y_pred == c))
            fp = np.sum((y_true != c) & (y_pred == c))
            
            if tp + fp == 0:
                precisions.append(0.0)
            else:
                precisions.append(tp / (tp + fp))
            
            supports.append(np.sum(y_true == c))
        
        precisions = np.array(precisions)
        supports = np.array(supports)
        
        if average == 'macro':
            return np.mean(precisions)
        elif average == 'micro':
            # Micro: calculate metrics globally
            tp_total = np.sum([np.sum((y_true == c) & (y_pred == c)) for c in classes])
            fp_total = np.sum([np.sum((y_true != c) & (y_pred == c)) for c in classes])
            return tp_total / (tp_total + fp_total) if (tp_total + fp_total) > 0 else 0.0
        elif average == 'weighted':
            return np.average(precisions, weights=supports)
    
    else:  # None - return array of precisions
        classes = np.unique(y_true)
        precisions = []
        for c in classes:
            tp = np.sum((y_true == c) & (y_pred == c))
            fp = np.sum((y_true != c) & (y_pred == c))
            precisions.append(tp / (tp + fp) if (tp + fp) > 0 else 0.0)
        return np.array(precisions)


def recall_score(y_true: np.ndarray, y_pred: np.ndarray, 
                pos_label: int = 1, average: Optional[str] = 'binary') -> float:
    """
    Calculate recall (sensitivity): TP / (TP + FN).
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
        pos_label: Positive class label (for binary)
        average: 'binary', 'macro', 'micro', 'weighted', or None
        
    Returns:
        Recall score(s)
    """
    if average == 'binary':
        tp = np.sum((y_true == pos_label) & (y_pred == pos_label))
        fn = np.sum((y_true == pos_label) & (y_pred != pos_label))
        
        if tp + fn == 0:
            return 0.0
        return tp / (tp + fn)
    
    elif average in ['macro', 'micro', 'weighted']:
        classes = np.unique(y_true)
        recalls = []
        supports = []
        
        for c in classes:
            tp = np.sum((y_true == c) & (y_pred == c))
            fn = np.sum((y_true == c) & (y_pred != c))
            
            if tp + fn == 0:
                recalls.append(0.0)
            else:
                recalls.append(tp / (tp + fn))
            
            supports.append(np.sum(y_true == c))
        
        recalls = np.array(recalls)
        supports = np.array(supports)
        
        if average == 'macro':
            return np.mean(recalls)
        elif average == 'micro':
            # For recall, micro = accuracy
            return accuracy_score(y_true, y_pred)
        elif average == 'weighted':
            return np.average(recalls, weights=supports)
    
    else:  # None
        classes = np.unique(y_true)
        recalls = []
        for c in classes:
            tp = np.sum((y_true == c) & (y_pred == c))
            fn = np.sum((y_true == c) & (y_pred != c))
            recalls.append(tp / (tp + fn) if (tp + fn) > 0 else 0.0)
        return np.array(recalls)


def f1_score(y_true: np.ndarray, y_pred: np.ndarray, 
            pos_label: int = 1, average: Optional[str] = 'binary') -> float:
    """
    Calculate F1 score: 2 * (precision * recall) / (precision + recall).
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
        pos_label: Positive class label (for binary)
        average: 'binary', 'macro', 'micro', 'weighted', or None
        
    Returns:
        F1 score(s)
    """
    precision = precision_score(y_true, y_pred, pos_label, average)
    recall = recall_score(y_true, y_pred, pos_label, average)
    
    if isinstance(precision, np.ndarray):
        # Element-wise F1 calculation
        f1 = np.zeros_like(precision)
        mask = (precision + recall) > 0
        f1[mask] = 2 * (precision[mask] * recall[mask]) / (precision[mask] + recall[mask])
        return f1
    else:
        if precision + recall == 0:
            return 0.0
        return 2 * (precision * recall) / (precision + recall)


def roc_curve(y_true: np.ndarray, y_scores: np.ndarray, 
             pos_label: int = 1) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute ROC curve (Receiver Operating Characteristic).
    
    Args:
        y_true: True binary labels
        y_scores: Predicted probabilities or decision scores
        pos_label: Label of positive class
        
    Returns:
        fpr: False positive rates
        tpr: True positive rates
        thresholds: Decision thresholds
    """
    # Sort by scores in descending order
    desc_score_indices = np.argsort(y_scores)[::-1]
    y_true_sorted = y_true[desc_score_indices]
    y_scores_sorted = y_scores[desc_score_indices]
    
    # Get distinct thresholds
    distinct_indices = np.where(np.diff(y_scores_sorted))[0]
    threshold_indices = np.concatenate([[0], distinct_indices + 1])
    thresholds = y_scores_sorted[threshold_indices]
    
    # Add threshold for 0 and 1 TPR/FPR
    thresholds = np.concatenate([[thresholds[0] + 1], thresholds])
    
    # Calculate TPR and FPR at each threshold
    tps = np.cumsum(y_true_sorted == pos_label)
    fps = np.cumsum(y_true_sorted != pos_label)
    
    # Add (0,0) point
    tps = np.concatenate([[0], tps[threshold_indices[:-1]], [tps[-1]]])
    fps = np.concatenate([[0], fps[threshold_indices[:-1]], [fps[-1]]])
    
    # Calculate rates
    n_pos = np.sum(y_true == pos_label)
    n_neg = np.sum(y_true != pos_label)
    
    tpr = tps / n_pos if n_pos > 0 else np.zeros_like(tps)
    fpr = fps / n_neg if n_neg > 0 else np.zeros_like(fps)
    
    return fpr, tpr, thresholds


def roc_auc_score(y_true: np.ndarray, y_scores: np.ndarray, 
                 pos_label: int = 1) -> float:
    """
    Calculate ROC-AUC score (Area Under ROC Curve).
    
    Args:
        y_true: True binary labels
        y_scores: Predicted probabilities or scores
        pos_label: Label of positive class
        
    Returns:
        AUC score
    """
    fpr, tpr, _ = roc_curve(y_true, y_scores, pos_label)
    
    # Calculate area using trapezoidal rule
    # Use trapezoid for numpy >= 2.0, fallback to trapz for older versions
    try:
        auc = np.trapezoid(tpr, fpr)
    except AttributeError:
        auc = np.trapz(tpr, fpr)
    
    return auc


def k_fold_split(X: np.ndarray, n_splits: int = 5, 
                shuffle: bool = True, random_state: Optional[int] = None) -> List[Tuple[np.ndarray, np.ndarray]]:
    """
    Generate K-Fold cross-validation indices.
    
    Args:
        X: Data array
        n_splits: Number of folds
        shuffle: Whether to shuffle before splitting
        random_state: Random seed
        
    Returns:
        List of (train_indices, val_indices) tuples
    """
    if random_state is not None:
        np.random.seed(random_state)
    
    n_samples = len(X)
    indices = np.arange(n_samples)
    
    if shuffle:
        np.random.shuffle(indices)
    
    fold_sizes = np.full(n_splits, n_samples // n_splits, dtype=int)
    fold_sizes[:n_samples % n_splits] += 1
    
    current = 0
    folds = []
    
    for fold_size in fold_sizes:
        start, stop = current, current + fold_size
        val_indices = indices[start:stop]
        train_indices = np.concatenate([indices[:start], indices[stop:]])
        folds.append((train_indices, val_indices))
        current = stop
    
    return folds


def cross_val_score(model, X: np.ndarray, y: np.ndarray, 
                   cv: int = 5, scoring: str = 'accuracy',
                   random_state: Optional[int] = None) -> np.ndarray:
    """
    Evaluate model using cross-validation.
    
    Args:
        model: Model with fit and predict methods
        X: Features
        y: Labels
        cv: Number of folds
        scoring: Scoring method ('accuracy', 'precision', 'recall', 'f1', 'roc_auc')
        random_state: Random seed
        
    Returns:
        Array of scores for each fold
    """
    folds = k_fold_split(X, n_splits=cv, shuffle=True, random_state=random_state)
    scores = []
    
    for train_idx, val_idx in folds:
        X_train, X_val = X[train_idx], X[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]
        
        # Train model
        model.fit(X_train, y_train)
        
        # Score
        if scoring == 'accuracy':
            y_pred = model.predict(X_val)
            score = accuracy_score(y_val, y_pred)
        elif scoring == 'precision':
            y_pred = model.predict(X_val)
            score = precision_score(y_val, y_pred)
        elif scoring == 'recall':
            y_pred = model.predict(X_val)
            score = recall_score(y_val, y_pred)
        elif scoring == 'f1':
            y_pred = model.predict(X_val)
            score = f1_score(y_val, y_pred)
        elif scoring == 'roc_auc':
            # Requires predict_proba
            if hasattr(model, 'predict_proba'):
                y_proba = model.predict_proba(X_val)
                if y_proba.ndim > 1:
                    y_proba = y_proba[:, 1]  # Positive class probability
                score = roc_auc_score(y_val, y_proba)
            else:
                raise ValueError("Model must have predict_proba for ROC-AUC")
        else:
            raise ValueError(f"Unknown scoring: {scoring}")
        
        scores.append(score)
    
    return np.array(scores)


def classification_report(y_true: np.ndarray, y_pred: np.ndarray) -> str:
    """
    Generate classification report with precision, recall, F1 for each class.
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
        
    Returns:
        Report string
    """
    classes = np.unique(y_true)
    
    # Calculate metrics for each class
    report_lines = []
    report_lines.append("              precision    recall  f1-score   support\n")
    
    for c in classes:
        precision = precision_score(y_true, y_pred, pos_label=c, average='binary')
        recall = recall_score(y_true, y_pred, pos_label=c, average='binary')
        f1 = f1_score(y_true, y_pred, pos_label=c, average='binary')
        support = np.sum(y_true == c)
        
        report_lines.append(f"{c:>12}  {precision:>9.2f}  {recall:>7.2f}  {f1:>9.2f}  {support:>8}\n")
    
    # Add averages
    report_lines.append("\n")
    
    # Macro average
    macro_precision = precision_score(y_true, y_pred, average='macro')
    macro_recall = recall_score(y_true, y_pred, average='macro')
    macro_f1 = f1_score(y_true, y_pred, average='macro')
    
    report_lines.append(f"{'macro avg':>12}  {macro_precision:>9.2f}  {macro_recall:>7.2f}  {macro_f1:>9.2f}  {len(y_true):>8}\n")
    
    # Weighted average
    weighted_precision = precision_score(y_true, y_pred, average='weighted')
    weighted_recall = recall_score(y_true, y_pred, average='weighted')
    weighted_f1 = f1_score(y_true, y_pred, average='weighted')
    
    report_lines.append(f"{'weighted avg':>12}  {weighted_precision:>9.2f}  {weighted_recall:>7.2f}  {weighted_f1:>9.2f}  {len(y_true):>8}\n")
    
    # Accuracy
    acc = accuracy_score(y_true, y_pred)
    report_lines.append(f"\n{'accuracy':>12}  {acc:>9.2f}  {len(y_true):>27}\n")
    
    return ''.join(report_lines)


def plot_confusion_matrix(cm: np.ndarray, class_names: Optional[List] = None, 
                         normalize: bool = False, cmap: str = 'Blues'):
    """Plot confusion matrix."""
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        fmt = '.2f'
    else:
        fmt = 'd'
    
    fig, ax = plt.subplots(figsize=(8, 6))
    im = ax.imshow(cm, interpolation='nearest', cmap=cmap)
    ax.figure.colorbar(im, ax=ax)
    
    if class_names is None:
        class_names = np.arange(cm.shape[0])
    
    ax.set(xticks=np.arange(cm.shape[1]),
           yticks=np.arange(cm.shape[0]),
           xticklabels=class_names,
           yticklabels=class_names,
           ylabel='True label',
           xlabel='Predicted label')
    
    # Rotate labels
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
    
    # Add text annotations
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], fmt),
                   ha="center", va="center",
                   color="white" if cm[i, j] > thresh else "black")
    
    plt.tight_layout()
    return fig


def plot_roc_curve_viz(fpr, tpr, auc_score=None):
    """Plot ROC curve."""
    fig, ax = plt.subplots(figsize=(8, 6))
    
    label = f'ROC curve (AUC = {auc_score:.2f})' if auc_score else 'ROC curve'
    ax.plot(fpr, tpr, linewidth=2, label=label)
    ax.plot([0, 1], [0, 1], 'k--', linewidth=1, label='Random classifier')
    
    ax.set_xlabel('False Positive Rate', fontsize=12)
    ax.set_ylabel('True Positive Rate', fontsize=12)
    ax.set_title('Receiver Operating Characteristic (ROC) Curve', fontsize=14, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    return fig


if __name__ == "__main__":
    from sklearn.datasets import make_classification
    
    print("="*60)
    print("MODEL EVALUATION DEMONSTRATION")
    print("="*60)
    
    # Generate sample data
    X, y = make_classification(
        n_samples=1000,
        n_features=20,
        n_informative=15,
        n_redundant=5,
        n_classes=2,
        weights=[0.6, 0.4],
        random_state=42
    )
    
    # Train/test split
    print("\nTrain/Test Split:")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42
    )
    print(f"Training samples: {len(X_train)}")
    print(f"Test samples: {len(X_test)}")
    
    # Simple model (for demonstration)
    class SimpleModel:
        def fit(self, X, y):
            self.mean = X.mean(axis=0)
            return self
        
        def predict(self, X):
            # Simple threshold-based prediction
            scores = ((X - self.mean) ** 2).sum(axis=1)
            return (scores < np.median(scores)).astype(int)
        
        def predict_proba(self, X):
            scores = ((X - self.mean) ** 2).sum(axis=1)
            proba_pos = 1 / (1 + scores / scores.max())
            return np.column_stack([1 - proba_pos, proba_pos])
    
    model = SimpleModel()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]
    
    # Confusion Matrix
    print("\n" + "="*60)
    print("CONFUSION MATRIX")
    print("="*60)
    cm = confusion_matrix(y_test, y_pred)
    print("\n", cm)
    
    # Classification Metrics
    print("\n" + "="*60)
    print("CLASSIFICATION METRICS")
    print("="*60)
    print(f"Accuracy:  {accuracy_score(y_test, y_pred):.4f}")
    print(f"Precision: {precision_score(y_test, y_pred):.4f}")
    print(f"Recall:    {recall_score(y_test, y_pred):.4f}")
    print(f"F1 Score:  {f1_score(y_test, y_pred):.4f}")
    
    # Classification Report
    print("\n" + "="*60)
    print("CLASSIFICATION REPORT")
    print("="*60)
    print(classification_report(y_test, y_pred))
    
    # ROC-AUC
    print("="*60)
    print("ROC-AUC ANALYSIS")
    print("="*60)
    fpr, tpr, thresholds = roc_curve(y_test, y_proba)
    auc = roc_auc_score(y_test, y_proba)
    print(f"\nROC-AUC Score: {auc:.4f}")
    
    # Cross-validation
    print("\n" + "="*60)
    print("CROSS-VALIDATION")
    print("="*60)
    cv_scores = cross_val_score(SimpleModel(), X, y, cv=5, scoring='accuracy', random_state=42)
    print(f"\n5-Fold CV Scores: {cv_scores}")
    print(f"Mean CV Score: {cv_scores.mean():.4f} (+/- {cv_scores.std():.4f})")
    
    print("\n✓ Model evaluation demonstration completed!")
