#!/usr/bin/env python3
"""
Learning Curve Explorer
=======================

Interactive tool to show how algorithms learn over time and iterations.
Visualizes learning curves, validation curves, and convergence patterns.

Usage: uv run python learning_curves.py [options]
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import argparse
import pandas as pd
from sklearn.model_selection import learning_curve, validation_curve, train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression, Ridge, Lasso, SGDClassifier, SGDRegressor
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor, GradientBoostingClassifier, GradientBoostingRegressor
from sklearn.svm import SVC, SVR
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.naive_bayes import GaussianNB
from sklearn.datasets import make_classification, make_regression, make_blobs, make_moons
from sklearn.metrics import accuracy_score, mean_squared_error, r2_score, log_loss
from sklearn.neural_network import MLPClassifier, MLPRegressor
import warnings
warnings.filterwarnings('ignore')

try:
    import xgboost as xgb
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False

class LearningCurveExplorer:
    """Learning curve visualization and analysis tool"""
    
    def __init__(self):
        self.data = {}
        self.models = {}
        self.is_classification = None
        self.feature_names = None
        
        # Color scheme
        self.colors = {
            'train': '#FF6B6B',
            'validation': '#4ECDC4', 
            'test': '#45B7D1',
            'convergence': '#FFA07A',
            'overfitting': '#FFB6C1'
        }
    
    def load_data(self, X, y, feature_names=None, problem_type=None):
        """Load and prepare data"""
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
    
    def get_model(self, algorithm, **params):
        """Get model instance"""
        models = {
            'logistic_regression': LogisticRegression(random_state=42, max_iter=1000, **params),
            'ridge_regression': Ridge(random_state=42, **params),
            'lasso_regression': Lasso(random_state=42, **params),
            'sgd': (SGDClassifier(random_state=42, **params) if self.is_classification 
                   else SGDRegressor(random_state=42, **params)),
            'random_forest': (RandomForestClassifier(random_state=42, **params) 
                            if self.is_classification 
                            else RandomForestRegressor(random_state=42, **params)),
            'gradient_boosting': (GradientBoostingClassifier(random_state=42, **params) 
                                if self.is_classification 
                                else GradientBoostingRegressor(random_state=42, **params)),
            'svm': (SVC(random_state=42, **params) if self.is_classification 
                   else SVR(**params)),
            'knn': (KNeighborsClassifier(**params) if self.is_classification 
                   else KNeighborsRegressor(**params)),
            'decision_tree': (DecisionTreeClassifier(random_state=42, **params) 
                            if self.is_classification 
                            else DecisionTreeRegressor(random_state=42, **params)),
            'mlp': (MLPClassifier(random_state=42, max_iter=1000, **params) 
                   if self.is_classification 
                   else MLPRegressor(random_state=42, max_iter=1000, **params)),
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
    
    def plot_learning_curve(self, algorithm, train_sizes=None, cv=5, use_scaling=True, ax=None, **model_params):
        """Plot learning curve showing performance vs training set size"""
        if train_sizes is None:
            train_sizes = np.linspace(0.1, 1.0, 10)
        
        model = self.get_model(algorithm, **model_params)
        
        # Select data
        X = self.data['X_train_scaled'] if use_scaling else self.data['X_train']
        y = self.data['y_train']
        
        # Calculate learning curve
        scoring = 'accuracy' if self.is_classification else 'r2'
        
        try:
            train_sizes_abs, train_scores, val_scores = learning_curve(
                model, X, y, train_sizes=train_sizes, cv=cv, 
                scoring=scoring, random_state=42, n_jobs=-1
            )
            
            # Calculate means and stds
            train_mean = np.mean(train_scores, axis=1)
            train_std = np.std(train_scores, axis=1)
            val_mean = np.mean(val_scores, axis=1)
            val_std = np.std(val_scores, axis=1)
            
            if ax is None:
                fig, ax = plt.subplots(figsize=(10, 6))
            
            # Plot learning curves
            ax.plot(train_sizes_abs, train_mean, 'o-', color=self.colors['train'], 
                   label='Training Score', linewidth=2, markersize=6)
            ax.fill_between(train_sizes_abs, train_mean - train_std, train_mean + train_std,
                           color=self.colors['train'], alpha=0.2)
            
            ax.plot(train_sizes_abs, val_mean, 'o-', color=self.colors['validation'], 
                   label='Validation Score', linewidth=2, markersize=6)
            ax.fill_between(train_sizes_abs, val_mean - val_std, val_mean + val_std,
                           color=self.colors['validation'], alpha=0.2)
            
            ax.set_xlabel('Training Set Size')
            ax.set_ylabel('Score')
            ax.set_title(f'{algorithm.replace("_", " ").title()} - Learning Curve')
            ax.legend(loc='best')
            ax.grid(True, alpha=0.3)
            
            # Add analysis text
            final_gap = train_mean[-1] - val_mean[-1]
            if final_gap > 0.1:
                ax.text(0.02, 0.98, '⚠️ Possible Overfitting', transform=ax.transAxes,
                       bbox=dict(boxstyle='round', facecolor=self.colors['overfitting'], alpha=0.7),
                       verticalalignment='top')
            elif final_gap < 0.05:
                ax.text(0.02, 0.98, '✅ Good Generalization', transform=ax.transAxes,
                       bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.7),
                       verticalalignment='top')
            
            return train_sizes_abs, train_scores, val_scores
            
        except Exception as e:
            if ax:
                ax.text(0.5, 0.5, f'Error: {str(e)}', ha='center', va='center', 
                       transform=ax.transAxes)
            return None, None, None
    
    def plot_validation_curve(self, algorithm, param_name, param_range, cv=5, 
                            use_scaling=True, ax=None, **fixed_params):
        """Plot validation curve showing performance vs hyperparameter"""
        model = self.get_model(algorithm, **fixed_params)
        
        # Select data
        X = self.data['X_train_scaled'] if use_scaling else self.data['X_train']
        y = self.data['y_train']
        
        scoring = 'accuracy' if self.is_classification else 'r2'
        
        try:
            train_scores, val_scores = validation_curve(
                model, X, y, param_name=param_name, param_range=param_range,
                cv=cv, scoring=scoring, n_jobs=-1
            )
            
            train_mean = np.mean(train_scores, axis=1)
            train_std = np.std(train_scores, axis=1)
            val_mean = np.mean(val_scores, axis=1)
            val_std = np.std(val_scores, axis=1)
            
            if ax is None:
                fig, ax = plt.subplots(figsize=(10, 6))
            
            # Plot validation curves
            ax.plot(param_range, train_mean, 'o-', color=self.colors['train'], 
                   label='Training Score', linewidth=2, markersize=6)
            ax.fill_between(param_range, train_mean - train_std, train_mean + train_std,
                           color=self.colors['train'], alpha=0.2)
            
            ax.plot(param_range, val_mean, 'o-', color=self.colors['validation'], 
                   label='Validation Score', linewidth=2, markersize=6)
            ax.fill_between(param_range, val_mean - val_std, val_mean + val_std,
                           color=self.colors['validation'], alpha=0.2)
            
            # Find optimal parameter
            best_idx = np.argmax(val_mean)
            best_param = param_range[best_idx]
            best_score = val_mean[best_idx]
            
            ax.axvline(x=best_param, color='red', linestyle='--', alpha=0.7,
                      label=f'Optimal {param_name}={best_param}')
            
            ax.set_xlabel(param_name.replace('_', ' ').title())
            ax.set_ylabel('Score')
            ax.set_title(f'{algorithm.replace("_", " ").title()} - Validation Curve ({param_name})')
            ax.legend(loc='best')
            ax.grid(True, alpha=0.3)
            
            # Set x-axis scale (log if wide range)
            if max(param_range) / min(param_range) > 100:
                ax.set_xscale('log')
            
            return param_range, train_scores, val_scores, best_param, best_score
            
        except Exception as e:
            if ax:
                ax.text(0.5, 0.5, f'Error: {str(e)}', ha='center', va='center', 
                       transform=ax.transAxes)
            return None, None, None, None, None
    
    def plot_training_history(self, algorithm, n_epochs=100, use_scaling=True, ax=None, **model_params):
        """Plot training history for iterative algorithms"""
        # Only works for algorithms that support partial_fit or have loss curves
        iterative_algos = ['sgd', 'mlp', 'gradient_boosting']
        
        if algorithm not in iterative_algos:
            if ax:
                ax.text(0.5, 0.5, f'Training history not available for {algorithm}',
                       ha='center', va='center', transform=ax.transAxes)
            return None
        
        # Select data
        X_train = self.data['X_train_scaled'] if use_scaling else self.data['X_train']
        y_train = self.data['y_train']
        X_test = self.data['X_test_scaled'] if use_scaling else self.data['X_test']
        y_test = self.data['y_test']
        
        if ax is None:
            fig, ax = plt.subplots(figsize=(10, 6))
        
        try:
            if algorithm == 'sgd':
                # SGD with partial_fit
                model = self.get_model(algorithm, **model_params)
                
                train_losses = []
                test_losses = []
                epochs = []
                
                # Convert to binary if needed for SGD
                classes = np.unique(y_train) if self.is_classification else None
                
                batch_size = min(100, len(X_train) // 10)
                n_batches = len(X_train) // batch_size
                
                for epoch in range(n_epochs):
                    epoch_loss = 0
                    for i in range(n_batches):
                        start_idx = i * batch_size
                        end_idx = (i + 1) * batch_size
                        
                        X_batch = X_train[start_idx:end_idx]
                        y_batch = y_train[start_idx:end_idx]
                        
                        if self.is_classification:
                            model.partial_fit(X_batch, y_batch, classes=classes)
                        else:
                            model.partial_fit(X_batch, y_batch)
                    
                    if epoch % 5 == 0:  # Record every 5 epochs
                        if self.is_classification:
                            train_score = model.score(X_train, y_train)
                            test_score = model.score(X_test, y_test)
                        else:
                            train_score = model.score(X_train, y_train)
                            test_score = model.score(X_test, y_test)
                        
                        train_losses.append(1 - train_score if self.is_classification else -train_score)
                        test_losses.append(1 - test_score if self.is_classification else -test_score)
                        epochs.append(epoch)
                
                ax.plot(epochs, train_losses, 'o-', color=self.colors['train'], 
                       label='Training Loss', linewidth=2, markersize=4)
                ax.plot(epochs, test_losses, 'o-', color=self.colors['validation'], 
                       label='Test Loss', linewidth=2, markersize=4)
            
            elif algorithm == 'mlp':
                # Neural network loss curve
                model = self.get_model(algorithm, **model_params)
                model.fit(X_train, y_train)
                
                if hasattr(model, 'loss_curve_'):
                    epochs = range(len(model.loss_curve_))
                    ax.plot(epochs, model.loss_curve_, 'o-', color=self.colors['train'],
                           label='Training Loss', linewidth=2, markersize=4)
                else:
                    ax.text(0.5, 0.5, 'Loss curve not available',
                           ha='center', va='center', transform=ax.transAxes)
                    return None
            
            elif algorithm == 'gradient_boosting':
                # Gradient boosting staged prediction
                model = self.get_model(algorithm, **model_params)
                model.fit(X_train, y_train)
                
                if self.is_classification:
                    train_scores = [accuracy_score(y_train, pred) for pred in 
                                   model.staged_predict(X_train)]
                    test_scores = [accuracy_score(y_test, pred) for pred in 
                                  model.staged_predict(X_test)]
                else:
                    train_scores = [r2_score(y_train, pred) for pred in 
                                   model.staged_predict(X_train)]
                    test_scores = [r2_score(y_test, pred) for pred in 
                                  model.staged_predict(X_test)]
                
                epochs = range(1, len(train_scores) + 1)
                ax.plot(epochs, train_scores, 'o-', color=self.colors['train'],
                       label='Training Score', linewidth=2, markersize=3)
                ax.plot(epochs, test_scores, 'o-', color=self.colors['validation'],
                       label='Test Score', linewidth=2, markersize=3)
            
            ax.set_xlabel('Iterations/Epochs')
            ax.set_ylabel('Loss/Score')
            ax.set_title(f'{algorithm.replace("_", " ").title()} - Training History')
            ax.legend(loc='best')
            ax.grid(True, alpha=0.3)
            
            return epochs, train_losses if 'train_losses' in locals() else train_scores
            
        except Exception as e:
            ax.text(0.5, 0.5, f'Error: {str(e)}', ha='center', va='center', 
                   transform=ax.transAxes)
            return None
    
    def plot_convergence_analysis(self, algorithm, use_scaling=True, ax=None, **model_params):
        """Analyze convergence behavior"""
        if ax is None:
            fig, ax = plt.subplots(figsize=(10, 6))
        
        # Multiple runs to show convergence variance
        n_runs = 5
        all_histories = []
        
        X_train = self.data['X_train_scaled'] if use_scaling else self.data['X_train']
        y_train = self.data['y_train']
        
        for run in range(n_runs):
            model_params_copy = model_params.copy()
            model_params_copy['random_state'] = 42 + run
            
            history = self.plot_training_history(
                algorithm, n_epochs=50, use_scaling=use_scaling, 
                ax=None, **model_params_copy
            )
            
            if history:
                epochs, scores = history
                all_histories.append(scores)
        
        if all_histories:
            # Plot multiple runs
            for i, scores in enumerate(all_histories):
                epochs = range(len(scores))
                ax.plot(epochs, scores, alpha=0.3, color=self.colors['train'])
            
            # Plot mean
            min_len = min(len(scores) for scores in all_histories)
            mean_scores = np.mean([scores[:min_len] for scores in all_histories], axis=0)
            std_scores = np.std([scores[:min_len] for scores in all_histories], axis=0)
            
            epochs = range(min_len)
            ax.plot(epochs, mean_scores, color=self.colors['convergence'], 
                   linewidth=3, label='Mean Convergence')
            ax.fill_between(epochs, mean_scores - std_scores, mean_scores + std_scores,
                           color=self.colors['convergence'], alpha=0.2)
        else:
            ax.text(0.5, 0.5, f'Convergence analysis not available for {algorithm}',
                   ha='center', va='center', transform=ax.transAxes)
        
        ax.set_xlabel('Iterations')
        ax.set_ylabel('Score/Loss')
        ax.set_title(f'{algorithm.replace("_", " ").title()} - Convergence Analysis')
        ax.legend(loc='best')
        ax.grid(True, alpha=0.3)
    
    def create_comprehensive_learning_analysis(self, algorithm, save_path=None, show=True, **model_params):
        """Create comprehensive learning curve analysis"""
        fig = plt.figure(figsize=(16, 12))
        fig.suptitle(f'{algorithm.replace("_", " ").title()} - Learning Analysis', 
                    fontsize=16, y=0.95)
        
        # 1. Learning Curve
        ax1 = plt.subplot(2, 3, 1)
        self.plot_learning_curve(algorithm, ax=ax1, **model_params)
        
        # 2. Training History (if available)
        ax2 = plt.subplot(2, 3, 2)
        self.plot_training_history(algorithm, ax=ax2, **model_params)
        
        # 3. Parameter validation curves
        param_configs = {
            'random_forest': [('n_estimators', [10, 50, 100, 200]), 
                            ('max_depth', [3, 5, 10, 15, None])],
            'svm': [('C', [0.1, 1, 10, 100]), ('gamma', [0.01, 0.1, 1, 10])],
            'logistic_regression': [('C', [0.01, 0.1, 1, 10, 100])],
            'ridge_regression': [('alpha', [0.01, 0.1, 1, 10, 100])],
            'lasso_regression': [('alpha', [0.01, 0.1, 1, 10])],
            'decision_tree': [('max_depth', [3, 5, 10, 15, 20, None])],
            'knn': [('n_neighbors', [1, 3, 5, 10, 15, 20])],
            'gradient_boosting': [('n_estimators', [50, 100, 200]), 
                                ('learning_rate', [0.01, 0.1, 0.2, 0.5])],
            'mlp': [('alpha', [0.0001, 0.001, 0.01, 0.1]), 
                   ('learning_rate_init', [0.001, 0.01, 0.1])],
        }
        
        if algorithm in param_configs:
            params_to_test = param_configs[algorithm][:2]  # Test up to 2 parameters
            
            for i, (param_name, param_range) in enumerate(params_to_test):
                ax = plt.subplot(2, 3, 3 + i)
                self.plot_validation_curve(
                    algorithm, param_name, param_range, ax=ax, **model_params
                )
        
        # 5. Convergence Analysis
        ax5 = plt.subplot(2, 3, 5)
        self.plot_convergence_analysis(algorithm, ax=ax5, **model_params)
        
        # 6. Learning Summary
        ax6 = plt.subplot(2, 3, 6)
        self._plot_learning_summary(algorithm, ax=ax6, **model_params)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"✅ Learning analysis saved to {save_path}")
        
        if show:
            plt.show()
    
    def _plot_learning_summary(self, algorithm, ax, **model_params):
        """Plot learning summary and recommendations"""
        ax.axis('off')
        
        # Train model for basic info
        model = self.get_model(algorithm, **model_params)
        X_train = self.data['X_train_scaled']
        X_test = self.data['X_test_scaled']
        y_train = self.data['y_train']
        y_test = self.data['y_test']
        
        model.fit(X_train, y_train)
        
        # Get basic performance
        if self.is_classification:
            train_score = accuracy_score(y_train, model.predict(X_train))
            test_score = accuracy_score(y_test, model.predict(X_test))
            metric = "Accuracy"
        else:
            train_score = r2_score(y_train, model.predict(X_train))
            test_score = r2_score(y_test, model.predict(X_test))
            metric = "R² Score"
        
        # Analysis text
        summary_text = f"📊 {algorithm.replace('_', ' ').title()} Learning Summary\n\n"
        
        summary_text += f"🎯 Performance:\n"
        summary_text += f"  • Training {metric}: {train_score:.3f}\n"
        summary_text += f"  • Test {metric}: {test_score:.3f}\n"
        summary_text += f"  • Gap: {train_score - test_score:.3f}\n\n"
        
        # Analysis and recommendations
        summary_text += f"🔍 Analysis:\n"
        gap = train_score - test_score
        
        if gap > 0.1:
            summary_text += f"  ⚠️ Overfitting detected!\n"
            summary_text += f"  💡 Try: Reduce complexity, add regularization\n"
        elif gap < 0.05:
            summary_text += f"  ✅ Good generalization\n"
            summary_text += f"  💡 Try: Increase complexity for better performance\n"
        else:
            summary_text += f"  📈 Reasonable generalization\n"
            summary_text += f"  💡 Try: Fine-tune hyperparameters\n"
        
        summary_text += f"\n🎮 Dataset Info:\n"
        summary_text += f"  • Training samples: {len(y_train)}\n"
        summary_text += f"  • Test samples: {len(y_test)}\n"
        summary_text += f"  • Features: {len(self.feature_names)}\n"
        
        # Algorithm-specific tips
        tips = {
            'random_forest': "🌲 Try tuning n_estimators and max_depth",
            'svm': "🎯 Tune C and gamma parameters",
            'gradient_boosting': "📈 Adjust learning_rate and n_estimators",
            'mlp': "🧠 Try different hidden_layer_sizes",
            'logistic_regression': "📊 Consider regularization with C parameter"
        }
        
        if algorithm in tips:
            summary_text += f"\n💡 Algorithm Tips:\n  {tips[algorithm]}\n"
        
        ax.text(0.05, 0.95, summary_text, transform=ax.transAxes,
               fontsize=10, verticalalignment='top', fontfamily='monospace',
               bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.1))
        
        ax.set_title('Learning Summary & Tips')

def generate_sample_data(data_type='classification', n_samples=1000, **kwargs):
    """Generate sample data for learning curve analysis"""
    generators = {
        'classification': lambda: make_classification(
            n_samples=n_samples, n_features=kwargs.get('n_features', 10), 
            n_informative=kwargs.get('n_informative', 7), n_redundant=2,
            n_classes=kwargs.get('n_classes', 2), random_state=42
        ),
        'regression': lambda: make_regression(
            n_samples=n_samples, n_features=kwargs.get('n_features', 10),
            noise=kwargs.get('noise', 10), random_state=42
        ),
        'moons': lambda: make_moons(n_samples=n_samples, noise=0.1, random_state=42),
        'blobs': lambda: make_blobs(
            n_samples=n_samples, centers=kwargs.get('centers', 3),
            n_features=kwargs.get('n_features', 5), random_state=42
        )
    }
    
    if data_type not in generators:
        raise ValueError(f"Unknown data type: {data_type}. Available: {list(generators.keys())}")
    
    return generators[data_type]()

def main():
    """Main function with command line interface"""
    parser = argparse.ArgumentParser(
        description="Visualize learning curves and algorithm learning behavior"
    )
    parser.add_argument('--algorithm', default='random_forest',
                       choices=['logistic_regression', 'ridge_regression', 'lasso_regression',
                               'random_forest', 'gradient_boosting', 'svm', 'knn', 
                               'decision_tree', 'mlp', 'sgd', 'naive_bayes', 'xgboost'],
                       help='Algorithm to analyze')
    parser.add_argument('--data_type', default='classification',
                       choices=['classification', 'regression', 'moons', 'blobs'],
                       help='Type of sample data to generate')
    parser.add_argument('--n_samples', type=int, default=1000,
                       help='Number of samples to generate')
    parser.add_argument('--save', type=str,
                       help='Save visualization to file')
    parser.add_argument('--quick', action='store_true',
                       help='Quick learning curve only')
    parser.add_argument('--no_show', action='store_true',
                       help='Run without opening plots (headless)')
    
    args = parser.parse_args()
    
    # Create explorer
    explorer = LearningCurveExplorer()
    
    # Generate data
    print(f"🎲 Generating {args.data_type} dataset...")
    X, y = generate_sample_data(args.data_type, args.n_samples)
    
    # Load data
    explorer.load_data(X, y)
    
    if args.quick:
        # Quick learning curve only
        print(f"📈 Creating learning curve for {args.algorithm}...")
        fig, ax = plt.subplots(figsize=(10, 6))
        explorer.plot_learning_curve(args.algorithm, ax=ax)
        plt.tight_layout()
        if not args.no_show:
            plt.show()
    else:
        # Comprehensive analysis
        print(f"📊 Creating comprehensive learning analysis for {args.algorithm}...")
        explorer.create_comprehensive_learning_analysis(args.algorithm, save_path=args.save, show=not args.no_show)

if __name__ == "__main__":
    main()
