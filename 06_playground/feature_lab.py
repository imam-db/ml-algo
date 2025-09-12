#!/usr/bin/env python3
"""
Feature Engineering Lab
=======================

Interactive playground for experimenting with different feature transformations and their effects.
Provides real-time visualization of how transformations affect model performance.

Usage: uv run python feature_lab.py [options]
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import argparse
import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import (
    StandardScaler, MinMaxScaler, RobustScaler, QuantileTransformer,
    PolynomialFeatures, PowerTransformer, FunctionTransformer
)
from sklearn.feature_selection import (
    SelectKBest, SelectPercentile, RFE, SelectFromModel,
    f_classif, f_regression, mutual_info_classif, mutual_info_regression
)
from sklearn.decomposition import PCA, TruncatedSVD, FastICA
from sklearn.manifold import TSNE
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LogisticRegression, LinearRegression, Lasso, Ridge
from sklearn.svm import SVC, SVR
from sklearn.datasets import make_classification, make_regression, make_blobs
from sklearn.metrics import accuracy_score, r2_score, mean_squared_error
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import warnings
warnings.filterwarnings('ignore')

class FeatureEngineeringLab:
    """Interactive feature engineering experimentation tool"""
    
    def __init__(self):
        self.original_data = {}
        self.transformed_data = {}
        self.transformations = {}
        self.models = {}
        self.results = {}
        self.is_classification = None
        self.feature_names = None
        
        # Color scheme
        self.colors = {
            'original': '#FF6B6B',
            'transformed': '#4ECDC4',
            'improvement': '#45B7D1',
            'degradation': '#FFA07A',
            'neutral': '#98D8C8'
        }
        
        # Available transformations
        self.scalers = {
            'standard': StandardScaler(),
            'minmax': MinMaxScaler(),
            'robust': RobustScaler(),
            'quantile_uniform': QuantileTransformer(output_distribution='uniform'),
            'quantile_normal': QuantileTransformer(output_distribution='normal'),
            'power_yeo_johnson': PowerTransformer(method='yeo-johnson'),
            'power_box_cox': PowerTransformer(method='box-cox')
        }
        
        self.feature_generators = {
            'polynomial_2': PolynomialFeatures(degree=2, include_bias=False),
            'polynomial_3': PolynomialFeatures(degree=3, include_bias=False),
            'log_transform': FunctionTransformer(np.log1p, validate=False),
            'sqrt_transform': FunctionTransformer(np.sqrt, validate=False),
            'reciprocal_transform': FunctionTransformer(lambda x: 1 / (x + 1e-8), validate=False)
        }
        
        self.dimensionality_reducers = {
            'pca_0.95': PCA(n_components=0.95),
            'pca_0.90': PCA(n_components=0.90),
            'pca_10': PCA(n_components=10),
            'svd_10': TruncatedSVD(n_components=10),
            'ica_10': FastICA(n_components=10, random_state=42)
        }
    
    def load_data(self, X, y, feature_names=None, problem_type=None):
        """Load and prepare data"""
        self.original_data['X'] = np.array(X)
        self.original_data['y'] = np.array(y)
        
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
            self.feature_names = [f'feature_{i+1}' for i in range(X.shape[1])]
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42,
            stratify=y if self.is_classification else None
        )
        
        self.original_data.update({
            'X_train': X_train,
            'X_test': X_test,
            'y_train': y_train,
            'y_test': y_test
        })
        
        # Initialize transformed data with original
        self.transformed_data = self.original_data.copy()
        
        print(f"✅ Data loaded: {X.shape[0]} samples, {X.shape[1]} features")
        print(f"🎯 Problem type: {'Classification' if self.is_classification else 'Regression'}")
    
    def apply_scaling(self, scaler_name, fit_on_train=True):
        """Apply scaling transformation"""
        if scaler_name not in self.scalers:
            raise ValueError(f"Unknown scaler: {scaler_name}")
        
        scaler = self.scalers[scaler_name]
        
        try:
            if scaler_name == 'power_box_cox':
                # Box-Cox requires positive values
                X_train = self.transformed_data['X_train'] 
                X_test = self.transformed_data['X_test']
                
                # Add small positive constant if needed
                if np.any(X_train <= 0):
                    offset = np.abs(X_train.min()) + 1
                    X_train = X_train + offset
                    X_test = X_test + offset
            else:
                X_train = self.transformed_data['X_train']
                X_test = self.transformed_data['X_test']
            
            if fit_on_train:
                X_train_scaled = scaler.fit_transform(X_train)
                X_test_scaled = scaler.transform(X_test)
            else:
                X_train_scaled = scaler.fit_transform(X_train)
                X_test_scaled = scaler.fit_transform(X_test)
            
            self.transformed_data['X_train'] = X_train_scaled
            self.transformed_data['X_test'] = X_test_scaled
            
            self.transformations['scaling'] = scaler_name
            
            print(f"✅ Applied {scaler_name} scaling")
            
        except Exception as e:
            print(f"❌ Error applying {scaler_name}: {str(e)}")
    
    def apply_feature_generation(self, generator_name):
        """Apply feature generation transformation"""
        if generator_name not in self.feature_generators:
            raise ValueError(f"Unknown generator: {generator_name}")
        
        generator = self.feature_generators[generator_name]
        
        try:
            X_train = self.transformed_data['X_train']
            X_test = self.transformed_data['X_test']
            
            X_train_gen = generator.fit_transform(X_train)
            X_test_gen = generator.transform(X_test)
            
            self.transformed_data['X_train'] = X_train_gen
            self.transformed_data['X_test'] = X_test_gen
            
            self.transformations['feature_generation'] = generator_name
            
            print(f"✅ Applied {generator_name} feature generation")
            print(f"   Features: {X_train.shape[1]} → {X_train_gen.shape[1]}")
            
        except Exception as e:
            print(f"❌ Error applying {generator_name}: {str(e)}")
    
    def apply_feature_selection(self, method, k_features=10):
        """Apply feature selection"""
        X_train = self.transformed_data['X_train']
        X_test = self.transformed_data['X_test']
        y_train = self.transformed_data['y_train']
        
        try:
            if method == 'selectkbest_f':
                score_func = f_classif if self.is_classification else f_regression
                selector = SelectKBest(score_func=score_func, k=min(k_features, X_train.shape[1]))
            elif method == 'selectkbest_mutual':
                score_func = mutual_info_classif if self.is_classification else mutual_info_regression
                selector = SelectKBest(score_func=score_func, k=min(k_features, X_train.shape[1]))
            elif method == 'rfe':
                estimator = (RandomForestClassifier(n_estimators=50, random_state=42) 
                           if self.is_classification 
                           else RandomForestRegressor(n_estimators=50, random_state=42))
                selector = RFE(estimator, n_features_to_select=min(k_features, X_train.shape[1]))
            elif method == 'model_based':
                estimator = (RandomForestClassifier(n_estimators=50, random_state=42) 
                           if self.is_classification 
                           else RandomForestRegressor(n_estimators=50, random_state=42))
                selector = SelectFromModel(estimator, max_features=min(k_features, X_train.shape[1]))
            else:
                raise ValueError(f"Unknown selection method: {method}")
            
            X_train_selected = selector.fit_transform(X_train, y_train)
            X_test_selected = selector.transform(X_test)
            
            self.transformed_data['X_train'] = X_train_selected
            self.transformed_data['X_test'] = X_test_selected
            
            self.transformations['feature_selection'] = f"{method}_{k_features}"
            
            print(f"✅ Applied {method} feature selection")
            print(f"   Features: {X_train.shape[1]} → {X_train_selected.shape[1]}")
            
        except Exception as e:
            print(f"❌ Error applying {method}: {str(e)}")
    
    def apply_dimensionality_reduction(self, reducer_name):
        """Apply dimensionality reduction"""
        if reducer_name not in self.dimensionality_reducers:
            raise ValueError(f"Unknown reducer: {reducer_name}")
        
        reducer = self.dimensionality_reducers[reducer_name]
        
        try:
            X_train = self.transformed_data['X_train']
            X_test = self.transformed_data['X_test']
            
            X_train_reduced = reducer.fit_transform(X_train)
            X_test_reduced = reducer.transform(X_test)
            
            self.transformed_data['X_train'] = X_train_reduced
            self.transformed_data['X_test'] = X_test_reduced
            
            self.transformations['dimensionality_reduction'] = reducer_name
            
            print(f"✅ Applied {reducer_name} dimensionality reduction")
            print(f"   Features: {X_train.shape[1]} → {X_train_reduced.shape[1]}")
            
            # Show explained variance for PCA
            if hasattr(reducer, 'explained_variance_ratio_'):
                total_var = np.sum(reducer.explained_variance_ratio_)
                print(f"   Explained variance: {total_var:.3f}")
            
        except Exception as e:
            print(f"❌ Error applying {reducer_name}: {str(e)}")
    
    def reset_transformations(self):
        """Reset all transformations to original data"""
        self.transformed_data = self.original_data.copy()
        self.transformations = {}
        print("🔄 Reset to original data")
    
    def evaluate_performance(self, algorithms=None):
        """Evaluate model performance on original vs transformed data"""
        if algorithms is None:
            if self.is_classification:
                algorithms = ['logistic_regression', 'random_forest', 'svm']
            else:
                algorithms = ['linear_regression', 'random_forest', 'svm']
        
        results = {}
        
        for algorithm in algorithms:
            # Original data performance
            original_score = self._train_and_score(
                algorithm, 
                self.original_data['X_train'], 
                self.original_data['X_test'],
                self.original_data['y_train'], 
                self.original_data['y_test']
            )
            
            # Transformed data performance
            transformed_score = self._train_and_score(
                algorithm,
                self.transformed_data['X_train'], 
                self.transformed_data['X_test'],
                self.transformed_data['y_train'], 
                self.transformed_data['y_test']
            )
            
            improvement = transformed_score - original_score
            
            results[algorithm] = {
                'original': original_score,
                'transformed': transformed_score,
                'improvement': improvement,
                'improvement_pct': (improvement / original_score) * 100 if original_score != 0 else 0
            }
        
        self.results = results
        return results
    
    def _train_and_score(self, algorithm, X_train, X_test, y_train, y_test):
        """Train model and return score"""
        models = {
            'logistic_regression': LogisticRegression(random_state=42, max_iter=1000),
            'linear_regression': LinearRegression(),
            'random_forest': (RandomForestClassifier(n_estimators=100, random_state=42) 
                            if self.is_classification 
                            else RandomForestRegressor(n_estimators=100, random_state=42)),
            'svm': (SVC(random_state=42) if self.is_classification else SVR()),
            'ridge': Ridge(random_state=42) if not self.is_classification else LogisticRegression(random_state=42, max_iter=1000),
            'lasso': Lasso(random_state=42) if not self.is_classification else LogisticRegression(random_state=42, max_iter=1000)
        }
        
        if algorithm not in models:
            return 0.0
        
        try:
            model = models[algorithm]
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            
            if self.is_classification:
                return accuracy_score(y_test, y_pred)
            else:
                return r2_score(y_test, y_pred)
                
        except Exception as e:
            print(f"Error training {algorithm}: {str(e)}")
            return 0.0
    
    def plot_transformation_comparison(self, save_path=None):
        """Plot comparison of original vs transformed data"""
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle('Feature Engineering Impact Analysis', fontsize=16, y=0.95)
        
        # 1. Performance comparison
        self._plot_performance_comparison(axes[0, 0])
        
        # 2. Feature distribution comparison (if 2D or less)
        self._plot_feature_distributions(axes[0, 1])
        
        # 3. Correlation matrix comparison
        self._plot_correlation_comparison(axes[0, 2])
        
        # 4. Dimensionality comparison
        self._plot_dimensionality_analysis(axes[1, 0])
        
        # 5. Transformation summary
        self._plot_transformation_summary(axes[1, 1])
        
        # 6. Feature importance comparison
        self._plot_feature_importance_comparison(axes[1, 2])
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"✅ Analysis saved to {save_path}")
        
        plt.show()
    
    def _plot_performance_comparison(self, ax):
        """Plot performance comparison between original and transformed data"""
        if not self.results:
            self.evaluate_performance()
        
        algorithms = list(self.results.keys())
        original_scores = [self.results[algo]['original'] for algo in algorithms]
        transformed_scores = [self.results[algo]['transformed'] for algo in algorithms]
        
        x = np.arange(len(algorithms))
        width = 0.35
        
        bars1 = ax.bar(x - width/2, original_scores, width, label='Original', 
                      color=self.colors['original'], alpha=0.7)
        bars2 = ax.bar(x + width/2, transformed_scores, width, label='Transformed', 
                      color=self.colors['transformed'], alpha=0.7)
        
        # Add improvement indicators
        for i, algo in enumerate(algorithms):
            improvement = self.results[algo]['improvement']
            color = self.colors['improvement'] if improvement > 0 else self.colors['degradation']
            ax.annotate(f'{improvement:+.3f}', xy=(i, max(original_scores[i], transformed_scores[i]) + 0.02),
                       ha='center', fontweight='bold', color=color)
        
        ax.set_xlabel('Algorithms')
        ax.set_ylabel('Score')
        ax.set_title('Performance Comparison')
        ax.set_xticks(x)
        ax.set_xticklabels([algo.replace('_', ' ').title() for algo in algorithms], rotation=45)
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    def _plot_feature_distributions(self, ax):
        """Plot feature distribution comparison"""
        # Show first 3 features for comparison
        n_features_to_show = min(3, self.original_data['X_train'].shape[1], 
                                self.transformed_data['X_train'].shape[1])
        
        if n_features_to_show == 0:
            ax.text(0.5, 0.5, 'No features to compare', ha='center', va='center', transform=ax.transAxes)
            return
        
        # Plot histograms
        for i in range(n_features_to_show):
            ax.hist(self.original_data['X_train'][:, i], bins=20, alpha=0.5, 
                   label=f'Original F{i+1}', color=self.colors['original'])
            
            if i < self.transformed_data['X_train'].shape[1]:
                ax.hist(self.transformed_data['X_train'][:, i], bins=20, alpha=0.5, 
                       label=f'Transformed F{i+1}', color=self.colors['transformed'])
        
        ax.set_xlabel('Feature Values')
        ax.set_ylabel('Frequency')
        ax.set_title('Feature Distributions')
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    def _plot_correlation_comparison(self, ax):
        """Plot correlation matrix comparison"""
        # Calculate correlations
        original_corr = np.corrcoef(self.original_data['X_train'].T)
        transformed_corr = np.corrcoef(self.transformed_data['X_train'].T)
        
        # Show average correlation change
        orig_mean_corr = np.mean(np.abs(original_corr[np.triu_indices_from(original_corr, k=1)]))
        trans_mean_corr = np.mean(np.abs(transformed_corr[np.triu_indices_from(transformed_corr, k=1)]))
        
        categories = ['Original', 'Transformed']
        correlations = [orig_mean_corr, trans_mean_corr]
        
        bars = ax.bar(categories, correlations, color=[self.colors['original'], self.colors['transformed']], alpha=0.7)
        
        ax.set_ylabel('Mean Absolute Correlation')
        ax.set_title('Feature Correlation Comparison')
        ax.grid(True, alpha=0.3)
        
        # Add values on bars
        for bar, corr in zip(bars, correlations):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                   f'{corr:.3f}', ha='center', va='bottom')
    
    def _plot_dimensionality_analysis(self, ax):
        """Plot dimensionality analysis"""
        original_dims = self.original_data['X_train'].shape[1]
        transformed_dims = self.transformed_data['X_train'].shape[1]
        
        categories = ['Original', 'Transformed']
        dimensions = [original_dims, transformed_dims]
        
        bars = ax.bar(categories, dimensions, color=[self.colors['original'], self.colors['transformed']], alpha=0.7)
        
        ax.set_ylabel('Number of Features')
        ax.set_title('Dimensionality Analysis')
        ax.grid(True, alpha=0.3)
        
        # Add values on bars
        for bar, dim in zip(bars, dimensions):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + 0.5,
                   f'{dim}', ha='center', va='bottom')
        
        # Show dimensionality change
        change = transformed_dims - original_dims
        change_pct = (change / original_dims) * 100 if original_dims > 0 else 0
        ax.text(0.5, 0.95, f'Change: {change:+d} ({change_pct:+.1f}%)', 
               transform=ax.transAxes, ha='center', va='top',
               bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.7))
    
    def _plot_transformation_summary(self, ax):
        """Plot transformation summary"""
        ax.axis('off')
        
        summary_text = "🔧 Applied Transformations:\n\n"
        
        if not self.transformations:
            summary_text += "  No transformations applied\n"
        else:
            for transform_type, transform_name in self.transformations.items():
                summary_text += f"  • {transform_type.replace('_', ' ').title()}: {transform_name}\n"
        
        summary_text += f"\n📊 Dataset Info:\n"
        summary_text += f"  • Original shape: {self.original_data['X_train'].shape}\n"
        summary_text += f"  • Current shape: {self.transformed_data['X_train'].shape}\n"
        summary_text += f"  • Problem type: {'Classification' if self.is_classification else 'Regression'}\n"
        
        if self.results:
            summary_text += f"\n🎯 Best Performance:\n"
            best_algo = max(self.results.keys(), 
                          key=lambda x: self.results[x]['transformed'])
            best_score = self.results[best_algo]['transformed']
            best_improvement = self.results[best_algo]['improvement']
            summary_text += f"  • Algorithm: {best_algo.replace('_', ' ').title()}\n"
            summary_text += f"  • Score: {best_score:.4f}\n"
            summary_text += f"  • Improvement: {best_improvement:+.4f}\n"
        
        ax.text(0.05, 0.95, summary_text, transform=ax.transAxes,
               fontsize=10, verticalalignment='top', fontfamily='monospace',
               bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))
        
        ax.set_title('Transformation Summary')
    
    def _plot_feature_importance_comparison(self, ax):
        """Plot feature importance comparison"""
        try:
            # Use Random Forest to get feature importance
            if self.is_classification:
                rf_orig = RandomForestClassifier(n_estimators=50, random_state=42)
                rf_trans = RandomForestClassifier(n_estimators=50, random_state=42)
            else:
                rf_orig = RandomForestRegressor(n_estimators=50, random_state=42)
                rf_trans = RandomForestRegressor(n_estimators=50, random_state=42)
            
            # Train on original data
            rf_orig.fit(self.original_data['X_train'], self.original_data['y_train'])
            orig_importance = rf_orig.feature_importances_
            
            # Train on transformed data
            rf_trans.fit(self.transformed_data['X_train'], self.transformed_data['y_train'])
            trans_importance = rf_trans.feature_importances_
            
            # Show top features
            n_features = min(10, len(orig_importance), len(trans_importance))
            
            x = np.arange(n_features)
            width = 0.35
            
            ax.bar(x - width/2, orig_importance[:n_features], width, 
                  label='Original', color=self.colors['original'], alpha=0.7)
            ax.bar(x + width/2, trans_importance[:n_features], width, 
                  label='Transformed', color=self.colors['transformed'], alpha=0.7)
            
            ax.set_xlabel('Feature Index')
            ax.set_ylabel('Importance')
            ax.set_title('Feature Importance Comparison')
            ax.set_xticks(x)
            ax.set_xticklabels([f'F{i+1}' for i in range(n_features)])
            ax.legend()
            ax.grid(True, alpha=0.3)
            
        except Exception as e:
            ax.text(0.5, 0.5, f'Feature importance\nnot available:\n{str(e)}',
                   ha='center', va='center', transform=ax.transAxes)
    
    def interactive_feature_engineering(self):
        """Interactive feature engineering session"""
        print("🧪 FEATURE ENGINEERING LAB - INTERACTIVE MODE")
        print("=" * 50)
        
        while True:
            print("\n🎛️ Available Operations:")
            print("1. 📏 Apply Scaling")
            print("2. 🔧 Generate Features")
            print("3. 🎯 Select Features")
            print("4. 📉 Reduce Dimensions")
            print("5. 📊 Evaluate Performance")
            print("6. 📈 Show Analysis")
            print("7. 🔄 Reset Transformations")
            print("8. ❌ Exit")
            
            choice = input("\nSelect operation (1-8): ").strip()
            
            try:
                if choice == '1':
                    self._interactive_scaling()
                elif choice == '2':
                    self._interactive_feature_generation()
                elif choice == '3':
                    self._interactive_feature_selection()
                elif choice == '4':
                    self._interactive_dimensionality_reduction()
                elif choice == '5':
                    results = self.evaluate_performance()
                    self._print_performance_results(results)
                elif choice == '6':
                    self.plot_transformation_comparison()
                elif choice == '7':
                    self.reset_transformations()
                elif choice == '8':
                    print("👋 Thanks for using Feature Engineering Lab!")
                    break
                else:
                    print("❌ Invalid choice. Please try again.")
                    
            except KeyboardInterrupt:
                print("\n👋 Exiting Feature Engineering Lab.")
                break
            except Exception as e:
                print(f"❌ Error: {str(e)}")
    
    def _interactive_scaling(self):
        """Interactive scaling selection"""
        print("\n📏 Available Scaling Methods:")
        scalers = list(self.scalers.keys())
        for i, scaler in enumerate(scalers, 1):
            print(f"{i}. {scaler.replace('_', ' ').title()}")
        
        try:
            choice = int(input(f"Select scaler (1-{len(scalers)}): ")) - 1
            if 0 <= choice < len(scalers):
                self.apply_scaling(scalers[choice])
            else:
                print("❌ Invalid selection")
        except ValueError:
            print("❌ Please enter a valid number")
    
    def _interactive_feature_generation(self):
        """Interactive feature generation"""
        print("\n🔧 Available Feature Generators:")
        generators = list(self.feature_generators.keys())
        for i, gen in enumerate(generators, 1):
            print(f"{i}. {gen.replace('_', ' ').title()}")
        
        try:
            choice = int(input(f"Select generator (1-{len(generators)}): ")) - 1
            if 0 <= choice < len(generators):
                self.apply_feature_generation(generators[choice])
            else:
                print("❌ Invalid selection")
        except ValueError:
            print("❌ Please enter a valid number")
    
    def _interactive_feature_selection(self):
        """Interactive feature selection"""
        print("\n🎯 Available Feature Selection Methods:")
        methods = ['selectkbest_f', 'selectkbest_mutual', 'rfe', 'model_based']
        for i, method in enumerate(methods, 1):
            print(f"{i}. {method.replace('_', ' ').title()}")
        
        try:
            choice = int(input(f"Select method (1-{len(methods)}): ")) - 1
            if 0 <= choice < len(methods):
                current_features = self.transformed_data['X_train'].shape[1]
                max_features = min(current_features, 50)
                k = int(input(f"Number of features to select (1-{max_features}): "))
                if 1 <= k <= max_features:
                    self.apply_feature_selection(methods[choice], k)
                else:
                    print(f"❌ Please enter a number between 1 and {max_features}")
            else:
                print("❌ Invalid selection")
        except ValueError:
            print("❌ Please enter valid numbers")
    
    def _interactive_dimensionality_reduction(self):
        """Interactive dimensionality reduction"""
        print("\n📉 Available Dimensionality Reduction Methods:")
        reducers = list(self.dimensionality_reducers.keys())
        for i, reducer in enumerate(reducers, 1):
            print(f"{i}. {reducer.replace('_', ' ').title()}")
        
        try:
            choice = int(input(f"Select method (1-{len(reducers)}): ")) - 1
            if 0 <= choice < len(reducers):
                self.apply_dimensionality_reduction(reducers[choice])
            else:
                print("❌ Invalid selection")
        except ValueError:
            print("❌ Please enter a valid number")
    
    def _print_performance_results(self, results):
        """Print performance results in a nice format"""
        print("\n📊 PERFORMANCE EVALUATION RESULTS")
        print("-" * 50)
        
        for algo, metrics in results.items():
            print(f"\n🤖 {algo.replace('_', ' ').title()}:")
            print(f"   Original:    {metrics['original']:.4f}")
            print(f"   Transformed: {metrics['transformed']:.4f}")
            print(f"   Improvement: {metrics['improvement']:+.4f} ({metrics['improvement_pct']:+.2f}%)")
            
            if metrics['improvement'] > 0:
                print("   Status: ✅ Improved")
            elif metrics['improvement'] < -0.01:
                print("   Status: ⚠️ Degraded")
            else:
                print("   Status: ➖ No significant change")

def generate_sample_data(data_type='classification', n_samples=1000, **kwargs):
    """Generate sample data for feature engineering experiments"""
    generators = {
        'classification': lambda: make_classification(
            n_samples=n_samples, n_features=kwargs.get('n_features', 20), 
            n_informative=kwargs.get('n_informative', 10), n_redundant=5,
            n_classes=kwargs.get('n_classes', 3), random_state=42
        ),
        'regression': lambda: make_regression(
            n_samples=n_samples, n_features=kwargs.get('n_features', 20),
            n_informative=kwargs.get('n_informative', 15), noise=kwargs.get('noise', 10),
            random_state=42
        ),
        'blobs': lambda: make_blobs(
            n_samples=n_samples, centers=kwargs.get('centers', 3),
            n_features=kwargs.get('n_features', 10), random_state=42
        )
    }
    
    if data_type not in generators:
        raise ValueError(f"Unknown data type: {data_type}. Available: {list(generators.keys())}")
    
    return generators[data_type]()

def main():
    """Main function with command line interface"""
    parser = argparse.ArgumentParser(
        description="Interactive feature engineering laboratory"
    )
    parser.add_argument('--data_type', default='classification',
                       choices=['classification', 'regression', 'blobs'],
                       help='Type of sample data to generate')
    parser.add_argument('--n_samples', type=int, default=1000,
                       help='Number of samples to generate')
    parser.add_argument('--n_features', type=int, default=20,
                       help='Number of features to generate')
    parser.add_argument('--interactive', action='store_true',
                       help='Launch interactive mode')
    parser.add_argument('--demo', action='store_true',
                       help='Run automatic demo')
    
    args = parser.parse_args()
    
    # Create lab
    lab = FeatureEngineeringLab()
    
    # Generate data
    print(f"🎲 Generating {args.data_type} dataset...")
    X, y = generate_sample_data(args.data_type, args.n_samples, n_features=args.n_features)
    
    # Load data
    lab.load_data(X, y)
    
    if args.interactive:
        lab.interactive_feature_engineering()
    elif args.demo:
        print("🎭 Running automatic demo...")
        
        # Demo various transformations
        print("\n1. Baseline performance...")
        lab.evaluate_performance()
        
        print("\n2. Applying standard scaling...")
        lab.apply_scaling('standard')
        lab.evaluate_performance()
        
        print("\n3. Adding polynomial features...")
        lab.apply_feature_generation('polynomial_2')
        lab.evaluate_performance()
        
        print("\n4. Selecting best features...")
        lab.apply_feature_selection('selectkbest_f', k=15)
        lab.evaluate_performance()
        
        print("\n📊 Final analysis:")
        lab.plot_transformation_comparison()
        
    else:
        print("💡 Use --interactive for interactive mode or --demo for automatic demo")
        print("📊 Showing baseline analysis...")
        lab.evaluate_performance()
        lab.plot_transformation_comparison()

if __name__ == "__main__":
    main()