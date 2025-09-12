#!/usr/bin/env python3
"""
Model Visualizer
================

Visual tool to see how different algorithms create decision boundaries and classifications.
Provides comprehensive visualization of model behavior, feature importance, and performance metrics.

Usage: uv run python model_visualizer.py [options]
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import argparse
import pandas as pd
from sklearn.model_selection import train_test_split, validation_curve
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression, LogisticRegression, Ridge
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.svm import SVC, SVR
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.naive_bayes import GaussianNB
from sklearn.datasets import make_classification, make_regression, make_blobs, make_moons, make_circles
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, 
    mean_squared_error, r2_score, confusion_matrix, 
    classification_report, roc_curve, auc, precision_recall_curve
)
import warnings
warnings.filterwarnings('ignore')

try:
    import xgboost as xgb
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False

class ModelVisualizer:
    """Comprehensive model visualization tool"""
    
    def __init__(self):
        self.models = {}
        self.data = {}
        self.is_classification = None
        self.feature_names = None
        
        # Define color palettes
        self.colors = {
            'primary': ['#FF6B6B', '#4ECDC4', '#45B7D1', '#FFA07A', '#98D8C8'],
            'decision_boundary': 'RdYlBu',
            'regression': 'viridis'
        }
    
    def load_data(self, X, y, feature_names=None, problem_type=None):
        """Load and prepare data for visualization"""
        self.data['X'] = np.array(X)
        self.data['y'] = np.array(y)
        
        # Determine problem type
        if problem_type:
            self.is_classification = (problem_type == 'classification')
        else:
            unique_y = len(np.unique(y))
            self.is_classification = unique_y <= 10 and np.issubdtype(y.dtype, np.integer)
        
        # Set feature names
        if feature_names:
            self.feature_names = feature_names
        else:
            self.feature_names = [f'Feature {i+1}' for i in range(X.shape[1])]
        
        # Split data
        self.data['X_train'], self.data['X_test'], self.data['y_train'], self.data['y_test'] = \
            train_test_split(X, y, test_size=0.2, random_state=42,
                           stratify=y if self.is_classification else None)
        
        # Scale features
        self.scaler = StandardScaler()
        self.data['X_train_scaled'] = self.scaler.fit_transform(self.data['X_train'])
        self.data['X_test_scaled'] = self.scaler.transform(self.data['X_test'])
        
        print(f"✅ Data loaded: {X.shape[0]} samples, {X.shape[1]} features")
        print(f"🎯 Problem type: {'Classification' if self.is_classification else 'Regression'}")
        if self.is_classification:
            print(f"📊 Classes: {len(np.unique(y))} unique")
    
    def get_model(self, algorithm, **params):
        """Get model instance"""
        models = {
            'logistic_regression': LogisticRegression(random_state=42, max_iter=1000, **params),
            'linear_regression': LinearRegression(**params),
            'ridge_regression': Ridge(random_state=42, **params),
            'random_forest': (RandomForestClassifier(random_state=42, **params) 
                            if self.is_classification 
                            else RandomForestRegressor(random_state=42, **params)),
            'svm': (SVC(random_state=42, probability=True, **params) 
                   if self.is_classification 
                   else SVR(**params)),
            'knn': (KNeighborsClassifier(**params) 
                   if self.is_classification 
                   else KNeighborsRegressor(**params)),
            'decision_tree': (DecisionTreeClassifier(random_state=42, **params) 
                            if self.is_classification 
                            else DecisionTreeRegressor(random_state=42, **params)),
            'naive_bayes': GaussianNB(**params) if self.is_classification else None
        }
        
        if XGBOOST_AVAILABLE:
            models['xgboost'] = (xgb.XGBClassifier(random_state=42, eval_metric='logloss', **params) 
                               if self.is_classification 
                               else xgb.XGBRegressor(random_state=42, **params))
        
        if algorithm not in models or models[algorithm] is None:
            available = [k for k, v in models.items() if v is not None]
            raise ValueError(f"Algorithm '{algorithm}' not available. Available: {available}")
        
        return models[algorithm]
    
    def train_model(self, algorithm, use_scaling=True, **params):
        """Train model and store results"""
        model = self.get_model(algorithm, **params)
        
        # Select data (scaled or unscaled)
        X_train = self.data['X_train_scaled'] if use_scaling else self.data['X_train']
        X_test = self.data['X_test_scaled'] if use_scaling else self.data['X_test']
        
        # Train model
        model.fit(X_train, self.data['y_train'])
        
        # Make predictions
        train_pred = model.predict(X_train)
        test_pred = model.predict(X_test)
        
        # Store results
        self.models[algorithm] = {
            'model': model,
            'train_pred': train_pred,
            'test_pred': test_pred,
            'X_train': X_train,
            'X_test': X_test,
            'use_scaling': use_scaling
        }
        
        return model
    
    def create_decision_boundary_2d(self, algorithm, h=0.02, ax=None):
        """Create 2D decision boundary visualization"""
        if algorithm not in self.models:
            raise ValueError(f"Model '{algorithm}' not trained. Call train_model() first.")
        
        model_info = self.models[algorithm]
        model = model_info['model']
        X_train = model_info['X_train']
        
        if X_train.shape[1] != 2:
            raise ValueError("Decision boundary visualization requires 2D data")
        
        if ax is None:
            fig, ax = plt.subplots(figsize=(10, 8))
        
        # Create mesh
        x_min, x_max = X_train[:, 0].min() - 1, X_train[:, 0].max() + 1
        y_min, y_max = X_train[:, 1].min() - 1, X_train[:, 1].max() + 1
        xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                           np.arange(y_min, y_max, h))
        
        # Make predictions on mesh
        mesh_points = np.c_[xx.ravel(), yy.ravel()]
        
        try:
            if self.is_classification:
                Z = model.predict(mesh_points)
                Z = Z.reshape(xx.shape)
                
                # Plot decision boundary
                ax.contourf(xx, yy, Z, alpha=0.4, cmap=self.colors['decision_boundary'])
                ax.contour(xx, yy, Z, colors='black', linewidths=0.5, alpha=0.5)
                
                # Plot training points
                scatter = ax.scatter(X_train[:, 0], X_train[:, 1], 
                                   c=self.data['y_train'], cmap=self.colors['decision_boundary'], 
                                   s=60, alpha=0.9, edgecolors='black', linewidth=0.5)
                
                # Plot test points with different marker
                ax.scatter(self.models[algorithm]['X_test'][:, 0], 
                          self.models[algorithm]['X_test'][:, 1],
                          c=self.data['y_test'], cmap=self.colors['decision_boundary'], 
                          s=60, alpha=0.7, marker='^', edgecolors='black', linewidth=0.5)
                
                plt.colorbar(scatter, ax=ax, label='Class')
                
            else:
                # For regression, show prediction surface
                Z = model.predict(mesh_points)
                Z = Z.reshape(xx.shape)
                
                # Plot surface
                contour = ax.contourf(xx, yy, Z, levels=20, alpha=0.6, cmap=self.colors['regression'])
                ax.contour(xx, yy, Z, levels=20, colors='black', alpha=0.4, linewidths=0.5)
                
                # Plot training points
                scatter = ax.scatter(X_train[:, 0], X_train[:, 1], 
                                   c=self.data['y_train'], cmap='plasma', 
                                   s=60, alpha=0.9, edgecolors='black', linewidth=0.5)
                
                plt.colorbar(contour, ax=ax, label='Predicted Value')
        
        except Exception as e:
            ax.text(0.5, 0.5, f'Error creating boundary: {str(e)}', 
                   ha='center', va='center', transform=ax.transAxes)
        
        ax.set_xlabel(self.feature_names[0])
        ax.set_ylabel(self.feature_names[1])
        ax.set_title(f'{algorithm.replace("_", " ").title()} - Decision Boundary')
        ax.grid(True, alpha=0.3)
        
        return ax
    
    def plot_feature_importance(self, algorithm, ax=None):
        """Plot feature importance if available"""
        if algorithm not in self.models:
            raise ValueError(f"Model '{algorithm}' not trained")
        
        model = self.models[algorithm]['model']
        
        # Get feature importance
        importance = None
        if hasattr(model, 'feature_importances_'):
            importance = model.feature_importances_
            importance_type = "Feature Importance"
        elif hasattr(model, 'coef_'):
            importance = np.abs(model.coef_).flatten()
            importance_type = "Coefficient Magnitude"
        
        if importance is None:
            if ax:
                ax.text(0.5, 0.5, f'Feature importance not available for {algorithm}',
                       ha='center', va='center', transform=ax.transAxes)
                ax.set_title(f'{algorithm.replace("_", " ").title()} - Feature Importance')
            return None
        
        if ax is None:
            fig, ax = plt.subplots(figsize=(10, 6))
        
        # Sort features by importance
        indices = np.argsort(importance)[::-1]
        
        # Plot
        ax.bar(range(len(importance)), importance[indices], 
               color=self.colors['primary'][0], alpha=0.7)
        ax.set_xlabel('Features')
        ax.set_ylabel(importance_type)
        ax.set_title(f'{algorithm.replace("_", " ").title()} - {importance_type}')
        ax.set_xticks(range(len(importance)))
        ax.set_xticklabels([self.feature_names[i] for i in indices], rotation=45)
        ax.grid(True, alpha=0.3)
        
        return ax
    
    def plot_performance_metrics(self, algorithm, ax=None):
        """Plot performance metrics"""
        if algorithm not in self.models:
            raise ValueError(f"Model '{algorithm}' not trained")
        
        model_info = self.models[algorithm]
        train_pred = model_info['train_pred']
        test_pred = model_info['test_pred']
        
        if ax is None:
            fig, ax = plt.subplots(figsize=(8, 6))
        
        if self.is_classification:
            # Classification metrics
            train_acc = accuracy_score(self.data['y_train'], train_pred)
            test_acc = accuracy_score(self.data['y_test'], test_pred)
            train_f1 = f1_score(self.data['y_train'], train_pred, average='weighted')
            test_f1 = f1_score(self.data['y_test'], test_pred, average='weighted')
            
            metrics = ['Accuracy', 'F1-Score']
            train_scores = [train_acc, train_f1]
            test_scores = [test_acc, test_f1]
            
        else:
            # Regression metrics
            train_r2 = r2_score(self.data['y_train'], train_pred)
            test_r2 = r2_score(self.data['y_test'], test_pred)
            train_rmse = np.sqrt(mean_squared_error(self.data['y_train'], train_pred))
            test_rmse = np.sqrt(mean_squared_error(self.data['y_test'], test_pred))
            
            metrics = ['R² Score', 'RMSE']
            train_scores = [train_r2, -train_rmse]  # Negative RMSE for better visualization
            test_scores = [test_r2, -test_rmse]
        
        # Plot metrics
        x = np.arange(len(metrics))
        width = 0.35
        
        ax.bar(x - width/2, train_scores, width, label='Training', 
               color=self.colors['primary'][0], alpha=0.7)
        ax.bar(x + width/2, test_scores, width, label='Test', 
               color=self.colors['primary'][1], alpha=0.7)
        
        ax.set_xlabel('Metrics')
        ax.set_ylabel('Score')
        ax.set_title(f'{algorithm.replace("_", " ").title()} - Performance Metrics')
        ax.set_xticks(x)
        ax.set_xticklabels(metrics)
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Add value labels
        for i, (train_score, test_score) in enumerate(zip(train_scores, test_scores)):
            ax.text(i - width/2, train_score + 0.01, f'{train_score:.3f}', 
                   ha='center', va='bottom')
            ax.text(i + width/2, test_score + 0.01, f'{test_score:.3f}', 
                   ha='center', va='bottom')
        
        return ax
    
    def plot_confusion_matrix(self, algorithm, ax=None):
        """Plot confusion matrix for classification"""
        if not self.is_classification:
            return None
        
        if algorithm not in self.models:
            raise ValueError(f"Model '{algorithm}' not trained")
        
        if ax is None:
            fig, ax = plt.subplots(figsize=(8, 6))
        
        test_pred = self.models[algorithm]['test_pred']
        cm = confusion_matrix(self.data['y_test'], test_pred)
        
        # Normalize confusion matrix
        cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        
        # Plot
        im = ax.imshow(cm_normalized, interpolation='nearest', cmap='Blues')
        ax.figure.colorbar(im, ax=ax)
        
        # Add text annotations
        thresh = cm_normalized.max() / 2.
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                ax.text(j, i, f'{cm[i, j]}\n({cm_normalized[i, j]:.2f})',
                       ha="center", va="center",
                       color="white" if cm_normalized[i, j] > thresh else "black")
        
        ax.set_xlabel('Predicted Label')
        ax.set_ylabel('True Label')
        ax.set_title(f'{algorithm.replace("_", " ").title()} - Confusion Matrix')
        
        return ax
    
    def plot_roc_curve(self, algorithm, ax=None):
        """Plot ROC curve for binary classification"""
        if not self.is_classification or len(np.unique(self.data['y'])) != 2:
            return None
        
        if algorithm not in self.models:
            raise ValueError(f"Model '{algorithm}' not trained")
        
        model = self.models[algorithm]['model']
        
        # Check if model has predict_proba
        if not hasattr(model, 'predict_proba'):
            return None
        
        if ax is None:
            fig, ax = plt.subplots(figsize=(8, 6))
        
        try:
            # Get probabilities
            X_test = self.models[algorithm]['X_test']
            y_proba = model.predict_proba(X_test)[:, 1]
            
            # Calculate ROC curve
            fpr, tpr, _ = roc_curve(self.data['y_test'], y_proba)
            roc_auc = auc(fpr, tpr)
            
            # Plot
            ax.plot(fpr, tpr, color=self.colors['primary'][0], lw=2,
                   label=f'ROC curve (AUC = {roc_auc:.2f})')
            ax.plot([0, 1], [0, 1], color='gray', lw=1, linestyle='--', alpha=0.5)
            
            ax.set_xlim([0.0, 1.0])
            ax.set_ylim([0.0, 1.05])
            ax.set_xlabel('False Positive Rate')
            ax.set_ylabel('True Positive Rate')
            ax.set_title(f'{algorithm.replace("_", " ").title()} - ROC Curve')
            ax.legend(loc="lower right")
            ax.grid(True, alpha=0.3)
            
        except Exception as e:
            ax.text(0.5, 0.5, f'Error creating ROC curve: {str(e)}', 
                   ha='center', va='center', transform=ax.transAxes)
        
        return ax
    
    def create_comprehensive_visualization(self, algorithm, save_path=None, show=True):
        """Create comprehensive visualization dashboard"""
        if algorithm not in self.models:
            raise ValueError(f"Model '{algorithm}' not trained")
        
        # Determine layout based on data dimensions and problem type
        if self.data['X'].shape[1] == 2:
            if self.is_classification:
                fig, axes = plt.subplots(2, 3, figsize=(18, 12))
                fig.suptitle(f'{algorithm.replace("_", " ").title()} - Comprehensive Analysis', 
                           fontsize=16, y=0.95)
                
                # Decision boundary
                self.create_decision_boundary_2d(algorithm, ax=axes[0, 0])
                
                # Feature importance
                self.plot_feature_importance(algorithm, ax=axes[0, 1])
                
                # Performance metrics
                self.plot_performance_metrics(algorithm, ax=axes[0, 2])
                
                # Confusion matrix
                self.plot_confusion_matrix(algorithm, ax=axes[1, 0])
                
                # ROC curve (if binary classification)
                if len(np.unique(self.data['y'])) == 2:
                    self.plot_roc_curve(algorithm, ax=axes[1, 1])
                else:
                    axes[1, 1].text(0.5, 0.5, 'ROC curve only for\nbinary classification',
                                   ha='center', va='center', transform=axes[1, 1].transAxes)
                    axes[1, 1].set_title('ROC Curve')
                
                # Model info
                self._plot_model_info(algorithm, ax=axes[1, 2])
                
            else:
                # Regression
                fig, axes = plt.subplots(2, 2, figsize=(15, 10))
                fig.suptitle(f'{algorithm.replace("_", " ").title()} - Regression Analysis', 
                           fontsize=16, y=0.95)
                
                # Decision boundary (prediction surface)
                self.create_decision_boundary_2d(algorithm, ax=axes[0, 0])
                
                # Feature importance
                self.plot_feature_importance(algorithm, ax=axes[0, 1])
                
                # Performance metrics
                self.plot_performance_metrics(algorithm, ax=axes[1, 0])
                
                # Prediction vs actual
                self._plot_prediction_vs_actual(algorithm, ax=axes[1, 1])
        
        else:
            # Higher dimensional data
            if self.is_classification:
                fig, axes = plt.subplots(2, 2, figsize=(15, 10))
                fig.suptitle(f'{algorithm.replace("_", " ").title()} - Analysis (High-D Data)', 
                           fontsize=16, y=0.95)
                
                # Feature importance
                self.plot_feature_importance(algorithm, ax=axes[0, 0])
                
                # Performance metrics
                self.plot_performance_metrics(algorithm, ax=axes[0, 1])
                
                # Confusion matrix
                self.plot_confusion_matrix(algorithm, ax=axes[1, 0])
                
                # Model info
                self._plot_model_info(algorithm, ax=axes[1, 1])
            
            else:
                fig, axes = plt.subplots(2, 2, figsize=(15, 10))
                fig.suptitle(f'{algorithm.replace("_", " ").title()} - Regression Analysis (High-D Data)', 
                           fontsize=16, y=0.95)
                
                # Feature importance
                self.plot_feature_importance(algorithm, ax=axes[0, 0])
                
                # Performance metrics
                self.plot_performance_metrics(algorithm, ax=axes[0, 1])
                
                # Prediction vs actual
                self._plot_prediction_vs_actual(algorithm, ax=axes[1, 0])
                
                # Residuals plot
                self._plot_residuals(algorithm, ax=axes[1, 1])
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"✅ Visualization saved to {save_path}")
        
        if show:
            plt.show()
    
    def _plot_model_info(self, algorithm, ax):
        """Plot model information and parameters"""
        model = self.models[algorithm]['model']
        
        ax.axis('off')
        
        info_text = f"🤖 {algorithm.replace('_', ' ').title()} Model\n\n"
        
        # Model parameters
        params = model.get_params()
        important_params = []
        
        for key, value in params.items():
            if not key.startswith('_') and value is not None:
                if isinstance(value, (int, float, str, bool)):
                    important_params.append(f"{key}: {value}")
        
        info_text += "📊 Key Parameters:\n"
        for param in important_params[:8]:  # Show top 8 parameters
            info_text += f"  • {param}\n"
        
        if len(important_params) > 8:
            info_text += f"  • ... and {len(important_params) - 8} more\n"
        
        # Training info
        info_text += f"\n🎯 Training Info:\n"
        info_text += f"  • Training samples: {len(self.data['y_train'])}\n"
        info_text += f"  • Test samples: {len(self.data['y_test'])}\n"
        info_text += f"  • Features: {len(self.feature_names)}\n"
        info_text += f"  • Scaling: {'Yes' if self.models[algorithm]['use_scaling'] else 'No'}\n"
        
        ax.text(0.1, 0.9, info_text, transform=ax.transAxes, 
               fontsize=10, verticalalignment='top', fontfamily='monospace')
        ax.set_title('Model Information')
    
    def _plot_prediction_vs_actual(self, algorithm, ax):
        """Plot prediction vs actual for regression"""
        if self.is_classification:
            return
        
        test_pred = self.models[algorithm]['test_pred']
        
        ax.scatter(self.data['y_test'], test_pred, alpha=0.6, 
                  color=self.colors['primary'][0])
        
        # Perfect prediction line
        min_val = min(self.data['y_test'].min(), test_pred.min())
        max_val = max(self.data['y_test'].max(), test_pred.max())
        ax.plot([min_val, max_val], [min_val, max_val], 
               'r--', alpha=0.8, linewidth=2)
        
        ax.set_xlabel('Actual Values')
        ax.set_ylabel('Predicted Values')
        ax.set_title('Prediction vs Actual')
        ax.grid(True, alpha=0.3)
        
        # Add R² score
        r2 = r2_score(self.data['y_test'], test_pred)
        ax.text(0.05, 0.95, f'R² = {r2:.3f}', transform=ax.transAxes,
               bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    def _plot_residuals(self, algorithm, ax):
        """Plot residuals for regression"""
        if self.is_classification:
            return
        
        test_pred = self.models[algorithm]['test_pred']
        residuals = self.data['y_test'] - test_pred
        
        ax.scatter(test_pred, residuals, alpha=0.6, color=self.colors['primary'][1])
        ax.axhline(y=0, color='r', linestyle='--', alpha=0.8)
        
        ax.set_xlabel('Predicted Values')
        ax.set_ylabel('Residuals')
        ax.set_title('Residual Plot')
        ax.grid(True, alpha=0.3)

def generate_sample_data(data_type='moons', n_samples=300, **kwargs):
    """Generate sample data for visualization"""
    generators = {
        'moons': lambda: make_moons(n_samples=n_samples, noise=kwargs.get('noise', 0.1), 
                                   random_state=42),
        'circles': lambda: make_circles(n_samples=n_samples, noise=kwargs.get('noise', 0.1), 
                                       factor=kwargs.get('factor', 0.5), random_state=42),
        'blobs': lambda: make_blobs(n_samples=n_samples, centers=kwargs.get('centers', 3),
                                   n_features=2, random_state=42),
        'classification': lambda: make_classification(n_samples=n_samples, n_features=2, 
                                                     n_redundant=0, n_informative=2,
                                                     n_clusters_per_class=1, random_state=42),
        'regression': lambda: make_regression(n_samples=n_samples, n_features=2, 
                                            noise=kwargs.get('noise', 10), random_state=42)
    }
    
    if data_type not in generators:
        raise ValueError(f"Unknown data type: {data_type}. Available: {list(generators.keys())}")
    
    return generators[data_type]()

def main():
    """Main function with command line interface"""
    parser = argparse.ArgumentParser(
        description="Visualize machine learning model behavior and decision boundaries"
    )
    parser.add_argument('--algorithm', default='random_forest',
                       choices=['logistic_regression', 'linear_regression', 'ridge_regression',
                               'random_forest', 'svm', 'knn', 'decision_tree', 'naive_bayes', 'xgboost'],
                       help='Algorithm to visualize')
    parser.add_argument('--data_type', default='moons',
                       choices=['moons', 'circles', 'blobs', 'classification', 'regression'],
                       help='Type of sample data to generate')
    parser.add_argument('--n_samples', type=int, default=300,
                       help='Number of samples to generate')
    parser.add_argument('--noise', type=float, default=0.1,
                       help='Noise level for generated data')
    parser.add_argument('--save', type=str,
                       help='Save visualization to file')
    parser.add_argument('--no_scaling', action='store_true',
                       help='Disable feature scaling')
    parser.add_argument('--no_show', action='store_true',
                       help='Do not display interactive window (use with --save)')
    
    args = parser.parse_args()
    
    # Create visualizer
    visualizer = ModelVisualizer()
    
    # Generate or load data
    print(f"🎲 Generating {args.data_type} dataset...")
    X, y = generate_sample_data(args.data_type, args.n_samples, noise=args.noise)
    
    # Load data
    visualizer.load_data(X, y)
    
    # Train model
    print(f"🔧 Training {args.algorithm} model...")
    visualizer.train_model(args.algorithm, use_scaling=not args.no_scaling)
    
    # Create visualization
    print("📊 Creating comprehensive visualization...")
    visualizer.create_comprehensive_visualization(
        args.algorithm,
        save_path=args.save,
        show=not args.no_show
    )

if __name__ == "__main__":
    main()
