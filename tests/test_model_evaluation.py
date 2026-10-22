"""
Tests for Model Evaluation metrics and techniques.
"""

import pytest
import numpy as np
import sys
import os
import importlib.util

# Import model evaluation implementation directly
eval_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '04_advanced_topics', '03_model_evaluation', 'implementation.py'))
spec = importlib.util.spec_from_file_location("eval_implementation", eval_path)
eval_impl = importlib.util.module_from_spec(spec)
spec.loader.exec_module(eval_impl)

train_test_split = eval_impl.train_test_split
confusion_matrix = eval_impl.confusion_matrix
accuracy_score = eval_impl.accuracy_score
precision_score = eval_impl.precision_score
recall_score = eval_impl.recall_score
f1_score = eval_impl.f1_score
roc_curve = eval_impl.roc_curve
roc_auc_score = eval_impl.roc_auc_score
k_fold_split = eval_impl.k_fold_split
cross_val_score = eval_impl.cross_val_score
classification_report = eval_impl.classification_report


@pytest.fixture
def binary_classification_data():
    """Generate binary classification data."""
    np.random.seed(42)
    X = np.random.randn(100, 5)
    y = (X[:, 0] + X[:, 1] > 0).astype(int)
    return X, y


@pytest.fixture
def multiclass_data():
    """Generate multiclass data."""
    np.random.seed(42)
    X = np.random.randn(90, 5)
    y = np.array([0]*30 + [1]*30 + [2]*30)
    return X, y


class TestTrainTestSplit:
    """Tests for train/test split."""
    
    def test_split_proportions(self, binary_classification_data):
        """Test split proportions."""
        X, y = binary_classification_data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.3, random_state=42
        )
        
        assert len(X_train) == 70
        assert len(X_test) == 30
        assert len(y_train) == 70
        assert len(y_test) == 30
    
    def test_split_reproducibility(self, binary_classification_data):
        """Test that same seed gives same split."""
        X, y = binary_classification_data
        
        X_train1, X_test1, y_train1, y_test1 = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        
        X_train2, X_test2, y_train2, y_test2 = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        
        np.testing.assert_array_equal(X_train1, X_train2)
        np.testing.assert_array_equal(y_train1, y_train2)
    
    def test_no_shuffle(self, binary_classification_data):
        """Test split without shuffling."""
        X, y = binary_classification_data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.3, shuffle=False
        )
        
        # First 70% should be train, last 30% test
        np.testing.assert_array_equal(X_train, X[:70])
        np.testing.assert_array_equal(X_test, X[70:])


class TestConfusionMatrix:
    """Tests for confusion matrix."""
    
    def test_binary_confusion_matrix(self):
        """Test binary classification confusion matrix."""
        y_true = np.array([0, 0, 1, 1, 0, 1])
        y_pred = np.array([0, 1, 1, 1, 0, 0])
        
        cm = confusion_matrix(y_true, y_pred)
        
        assert cm.shape == (2, 2)
        assert cm[0, 0] == 2  # TN
        assert cm[0, 1] == 1  # FP
        assert cm[1, 0] == 1  # FN
        assert cm[1, 1] == 2  # TP
    
    def test_multiclass_confusion_matrix(self):
        """Test multiclass confusion matrix."""
        y_true = np.array([0, 0, 1, 1, 2, 2])
        y_pred = np.array([0, 1, 1, 2, 2, 2])
        
        cm = confusion_matrix(y_true, y_pred)
        
        assert cm.shape == (3, 3)
        assert cm[0, 0] == 1
        assert cm[1, 1] == 1
        assert cm[2, 2] == 2


class TestAccuracy:
    """Tests for accuracy score."""
    
    def test_perfect_accuracy(self):
        """Test perfect accuracy."""
        y_true = np.array([0, 1, 0, 1, 0])
        y_pred = np.array([0, 1, 0, 1, 0])
        
        acc = accuracy_score(y_true, y_pred)
        assert acc == 1.0
    
    def test_zero_accuracy(self):
        """Test zero accuracy."""
        y_true = np.array([0, 0, 0, 0])
        y_pred = np.array([1, 1, 1, 1])
        
        acc = accuracy_score(y_true, y_pred)
        assert acc == 0.0
    
    def test_partial_accuracy(self):
        """Test partial accuracy."""
        y_true = np.array([0, 1, 0, 1, 0])
        y_pred = np.array([0, 1, 1, 1, 0])
        
        acc = accuracy_score(y_true, y_pred)
        assert acc == 0.8


class TestPrecisionRecallF1:
    """Tests for precision, recall, and F1."""
    
    def test_precision_binary(self):
        """Test binary precision."""
        y_true = np.array([0, 0, 1, 1, 1])
        y_pred = np.array([0, 1, 1, 1, 0])
        
        precision = precision_score(y_true, y_pred, pos_label=1)
        
        # TP=2, FP=1 -> precision = 2/3
        assert np.isclose(precision, 2/3)
    
    def test_recall_binary(self):
        """Test binary recall."""
        y_true = np.array([0, 0, 1, 1, 1])
        y_pred = np.array([0, 1, 1, 1, 0])
        
        recall = recall_score(y_true, y_pred, pos_label=1)
        
        # TP=2, FN=1 -> recall = 2/3
        assert np.isclose(recall, 2/3)
    
    def test_f1_binary(self):
        """Test binary F1."""
        y_true = np.array([0, 0, 1, 1, 1])
        y_pred = np.array([0, 1, 1, 1, 0])
        
        f1 = f1_score(y_true, y_pred, pos_label=1)
        
        # Precision=2/3, Recall=2/3 -> F1=2/3
        assert np.isclose(f1, 2/3)
    
    def test_macro_average(self):
        """Test macro averaging."""
        y_true = np.array([0, 0, 1, 1, 2, 2])
        y_pred = np.array([0, 1, 1, 2, 2, 2])
        
        precision = precision_score(y_true, y_pred, average='macro')
        recall = recall_score(y_true, y_pred, average='macro')
        
        assert 0 <= precision <= 1
        assert 0 <= recall <= 1
    
    def test_weighted_average(self):
        """Test weighted averaging."""
        y_true = np.array([0, 0, 1, 1, 2, 2])
        y_pred = np.array([0, 1, 1, 2, 2, 2])
        
        precision = precision_score(y_true, y_pred, average='weighted')
        recall = recall_score(y_true, y_pred, average='weighted')
        
        assert 0 <= precision <= 1
        assert 0 <= recall <= 1


class TestROC:
    """Tests for ROC curve and AUC."""
    
    def test_roc_curve(self):
        """Test ROC curve computation."""
        y_true = np.array([0, 0, 1, 1])
        y_scores = np.array([0.1, 0.4, 0.35, 0.8])
        
        fpr, tpr, thresholds = roc_curve(y_true, y_scores)
        
        assert len(fpr) == len(tpr)
        assert len(fpr) == len(thresholds)
        assert fpr[0] == 0.0  # Start at (0, 0)
        assert tpr[0] == 0.0
        assert np.all(fpr >= 0)
        assert np.all(tpr >= 0)
        assert np.all(fpr <= 1)
        assert np.all(tpr <= 1)
    
    def test_roc_auc_perfect(self):
        """Test perfect AUC."""
        y_true = np.array([0, 0, 1, 1])
        y_scores = np.array([0.1, 0.2, 0.8, 0.9])
        
        auc = roc_auc_score(y_true, y_scores)
        
        assert auc == 1.0
    
    def test_roc_auc_random(self):
        """Test random AUC (around 0.5)."""
        np.random.seed(42)
        y_true = np.random.randint(0, 2, 100)
        y_scores = np.random.rand(100)
        
        auc = roc_auc_score(y_true, y_scores)
        
        # Random classifier should be around 0.5
        assert 0.3 < auc < 0.7


class TestKFoldSplit:
    """Tests for K-Fold cross-validation."""
    
    def test_k_fold_split(self):
        """Test K-Fold splitting."""
        X = np.arange(100)
        folds = k_fold_split(X, n_splits=5, shuffle=False)
        
        assert len(folds) == 5
        
        # Each fold should have train and val indices
        for train_idx, val_idx in folds:
            assert len(train_idx) + len(val_idx) == 100
            # No overlap
            assert len(np.intersect1d(train_idx, val_idx)) == 0
    
    def test_k_fold_coverage(self):
        """Test that K-Fold covers all samples."""
        X = np.arange(100)
        folds = k_fold_split(X, n_splits=5, shuffle=False)
        
        all_val_indices = []
        for _, val_idx in folds:
            all_val_indices.extend(val_idx)
        
        # All indices should be covered exactly once
        assert sorted(all_val_indices) == list(range(100))
    
    def test_k_fold_reproducibility(self):
        """Test reproducibility with random state."""
        X = np.arange(50)
        
        folds1 = k_fold_split(X, n_splits=5, shuffle=True, random_state=42)
        folds2 = k_fold_split(X, n_splits=5, shuffle=True, random_state=42)
        
        for (train1, val1), (train2, val2) in zip(folds1, folds2):
            np.testing.assert_array_equal(train1, train2)
            np.testing.assert_array_equal(val1, val2)


class TestCrossValScore:
    """Tests for cross-validation scoring."""
    
    def test_cross_val_score(self, binary_classification_data):
        """Test cross-validation scoring."""
        X, y = binary_classification_data
        
        # Simple model for testing
        class DummyModel:
            def fit(self, X, y):
                self.mean = y.mean()
                return self
            
            def predict(self, X):
                return (np.ones(len(X)) * (self.mean > 0.5)).astype(int)
        
        scores = cross_val_score(
            DummyModel(), X, y, 
            cv=5, scoring='accuracy', random_state=42
        )
        
        assert len(scores) == 5
        assert all(0 <= s <= 1 for s in scores)


class TestClassificationReport:
    """Tests for classification report."""
    
    def test_classification_report(self):
        """Test classification report generation."""
        y_true = np.array([0, 0, 1, 1, 2, 2])
        y_pred = np.array([0, 1, 1, 2, 2, 2])
        
        report = classification_report(y_true, y_pred)
        
        assert isinstance(report, str)
        assert 'precision' in report
        assert 'recall' in report
        assert 'f1-score' in report
        assert 'support' in report


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
