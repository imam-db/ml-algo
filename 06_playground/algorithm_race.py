#!/usr/bin/env python3
"""
Algorithm Racing Tool
====================

Compare multiple algorithms side-by-side on the same dataset.
Provides comprehensive performance analysis with timing, accuracy, and statistical testing.

Usage: uv run python algorithm_race.py [options]
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import time
import argparse
from sklearn.model_selection import train_test_split, cross_val_score, KFold
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.svm import SVC, SVR
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, roc_auc_score,
    mean_squared_error, mean_absolute_error, r2_score
)
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

try:
    import xgboost as xgb
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False

class AlgorithmRacer:
    """Compare multiple ML algorithms on the same dataset"""
    
    def __init__(self):
        self.results = {}
        self.is_classification = None
        self.feature_names = None
        
        # Define available algorithms
        self.classification_algorithms = {
            'logistic_regression': LogisticRegression(random_state=42, max_iter=1000),
            'random_forest': RandomForestClassifier(random_state=42, n_estimators=100),
            'svm': SVC(random_state=42, probability=True),
            'naive_bayes': GaussianNB(),
            'knn': KNeighborsClassifier(n_neighbors=5),
            'decision_tree': DecisionTreeClassifier(random_state=42)
        }
        
        self.regression_algorithms = {
            'linear_regression': LinearRegression(),
            'random_forest': RandomForestRegressor(random_state=42, n_estimators=100),
            'svm': SVR(),
            'knn': KNeighborsRegressor(n_neighbors=5),
            'decision_tree': DecisionTreeRegressor(random_state=42)
        }
        
        # Add XGBoost if available
        if XGBOOST_AVAILABLE:
            self.classification_algorithms['xgboost'] = xgb.XGBClassifier(random_state=42, eval_metric='logloss')
            self.regression_algorithms['xgboost'] = xgb.XGBRegressor(random_state=42)
    
    def load_data(self, data_path: str = None, X: np.ndarray = None, y: np.ndarray = None):
        """Load data from file or arrays"""
        if data_path:
            # Load from file
            if data_path.endswith('.csv'):
                df = pd.read_csv(data_path)
                # Assume last column is target
                X = df.iloc[:, :-1].values
                y = df.iloc[:, -1].values
                self.feature_names = df.columns[:-1].tolist()
            else:
                raise ValueError("Only CSV files are supported")
        
        elif X is not None and y is not None:
            X = np.array(X)
            y = np.array(y)
            self.feature_names = [f'feature_{i+1}' for i in range(X.shape[1])]
        
        else:
            raise ValueError("Either data_path or X,y arrays must be provided")
        
        # Determine if classification or regression
        unique_values = len(np.unique(y))
        self.is_classification = unique_values <= 20 and (
            np.issubdtype(y.dtype, np.integer) or 
            isinstance(y[0], (str, bool))
        )
        
        # Encode labels for classification if needed
        if self.is_classification and not np.issubdtype(y.dtype, np.number):
            le = LabelEncoder()
            y = le.fit_transform(y)
        
        return X, y
    
    def prepare_data(self, X: np.ndarray, y: np.ndarray, test_size: float = 0.2, 
                    scale_features: bool = True):
        """Prepare data for training"""
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=42, 
            stratify=y if self.is_classification else None
        )
        
        # Scale features
        if scale_features:
            scaler = StandardScaler()
            X_train = scaler.fit_transform(X_train)
            X_test = scaler.transform(X_test)
        
        return X_train, X_test, y_train, y_test
    
    def race_algorithms(self, X: np.ndarray, y: np.ndarray, algorithms: list = None, 
                       cv_folds: int = 5, scale_features: bool = True):
        """Race multiple algorithms and collect performance metrics"""
        
        # Select algorithms based on problem type
        if self.is_classification:
            available_algos = self.classification_algorithms
            problem_type = "Classification"
        else:
            available_algos = self.regression_algorithms  
            problem_type = "Regression"
        
        if algorithms is None:
            algorithms = list(available_algos.keys())
        
        # Validate algorithm names
        valid_algorithms = []
        for algo in algorithms:
            if algo in available_algos:
                valid_algorithms.append(algo)
            else:
                print(f"⚠️  Algorithm '{algo}' not available for {problem_type.lower()}. Skipping.")
        
        algorithms = valid_algorithms
        if not algorithms:
            raise ValueError(f"No valid algorithms specified for {problem_type}")
        
        print(f"🏁 ALGORITHM RACE - {problem_type}")
        print("=" * 60)
        print(f"📊 Dataset: {X.shape[0]} samples, {X.shape[1]} features")
        print(f"🎯 Racing {len(algorithms)} algorithms: {', '.join(algorithms)}")
        print(f"📈 Evaluation: {cv_folds}-fold cross-validation")
        print()
        
        # Prepare data
        X_train, X_test, y_train, y_test = self.prepare_data(X, y, scale_features=scale_features)
        
        results = {}
        
        for algo_name in algorithms:
            print(f"🏃 Training {algo_name}...")
            
            model = available_algos[algo_name]
            result = {}
            
            try:
                # Training time
                start_time = time.time()
                model.fit(X_train, y_train)
                training_time = time.time() - start_time
                
                # Prediction time
                start_time = time.time()
                y_pred = model.predict(X_test)
                prediction_time = time.time() - start_time
                
                # Cross-validation score
                cv_scores = cross_val_score(
                    model, X_train, y_train, cv=cv_folds, 
                    scoring='accuracy' if self.is_classification else 'r2'
                )
                
                # Performance metrics
                if self.is_classification:
                    result.update({
                        'accuracy': accuracy_score(y_test, y_pred),
                        'precision': precision_score(y_test, y_pred, average='weighted', zero_division=0),
                        'recall': recall_score(y_test, y_pred, average='weighted', zero_division=0),
                        'f1_score': f1_score(y_test, y_pred, average='weighted', zero_division=0)
                    })
                    
                    # ROC AUC for binary classification
                    if len(np.unique(y)) == 2 and hasattr(model, 'predict_proba'):
                        try:
                            y_pred_proba = model.predict_proba(X_test)[:, 1]
                            result['roc_auc'] = roc_auc_score(y_test, y_pred_proba)
                        except:
                            result['roc_auc'] = np.nan
                    else:
                        result['roc_auc'] = np.nan
                
                else:
                    result.update({
                        'r2_score': r2_score(y_test, y_pred),
                        'mse': mean_squared_error(y_test, y_pred),
                        'rmse': np.sqrt(mean_squared_error(y_test, y_pred)),
                        'mae': mean_absolute_error(y_test, y_pred)
                    })
                
                # Common metrics
                result.update({
                    'training_time': training_time,
                    'prediction_time': prediction_time,
                    'cv_score_mean': cv_scores.mean(),
                    'cv_score_std': cv_scores.std(),
                    'model': model,
                    'predictions': y_pred
                })
                
                results[algo_name] = result
                print(f"   ✅ Completed in {training_time:.3f}s")
                
            except Exception as e:
                print(f"   ❌ Failed: {str(e)}")
                continue
        
        self.results = results
        return results
    
    def create_results_table(self):
        """Create formatted results table"""
        if not self.results:
            return pd.DataFrame()
        
        # Prepare data for table
        table_data = []
        
        for algo_name, result in self.results.items():
            row = {'Algorithm': algo_name.replace('_', ' ').title()}
            
            if self.is_classification:
                row.update({
                    'Accuracy': f"{result['accuracy']:.4f}",
                    'Precision': f"{result['precision']:.4f}",
                    'Recall': f"{result['recall']:.4f}",
                    'F1-Score': f"{result['f1_score']:.4f}",
                    'ROC-AUC': f"{result['roc_auc']:.4f}" if not np.isnan(result['roc_auc']) else 'N/A'
                })
            else:
                row.update({
                    'R² Score': f"{result['r2_score']:.4f}",
                    'RMSE': f"{result['rmse']:.4f}",
                    'MAE': f"{result['mae']:.4f}",
                    'MSE': f"{result['mse']:.4f}"
                })
            
            row.update({
                'CV Score': f"{result['cv_score_mean']:.4f} ±{result['cv_score_std']:.4f}",
                'Train Time (s)': f"{result['training_time']:.4f}",
                'Pred Time (s)': f"{result['prediction_time']:.6f}"
            })
            
            table_data.append(row)
        
        return pd.DataFrame(table_data)
    
    def plot_performance_comparison(self):
        """Create performance comparison plots"""
        if not self.results:
            print("❌ No results to plot. Run race_algorithms() first.")
            return
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle('Algorithm Performance Comparison', fontsize=16, y=1.02)
        
        algorithms = list(self.results.keys())
        
        # Plot 1: Main performance metric
        ax1 = axes[0, 0]
        if self.is_classification:
            metric_values = [self.results[algo]['accuracy'] for algo in algorithms]
            metric_name = 'Accuracy'
        else:
            metric_values = [self.results[algo]['r2_score'] for algo in algorithms]
            metric_name = 'R² Score'
        
        colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#FFA07A', '#98D8C8', '#F7DC6F', '#DDA0DD', '#87CEEB']
        bars = ax1.bar(range(len(algorithms)), metric_values, 
                      color=colors[:len(algorithms)])
        ax1.set_xlabel('Algorithms')
        ax1.set_ylabel(metric_name)
        ax1.set_title(f'{metric_name} Comparison')
        ax1.set_xticks(range(len(algorithms)))
        ax1.set_xticklabels([algo.replace('_', ' ').title() for algo in algorithms], rotation=45)
        
        # Add value labels on bars
        for bar, value in zip(bars, metric_values):
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                    f'{value:.3f}', ha='center', va='bottom')
        
        # Plot 2: Training time comparison
        ax2 = axes[0, 1]
        train_times = [self.results[algo]['training_time'] for algo in algorithms]
        bars2 = ax2.bar(range(len(algorithms)), train_times, color='orange', alpha=0.7)
        ax2.set_xlabel('Algorithms')
        ax2.set_ylabel('Training Time (seconds)')
        ax2.set_title('Training Time Comparison')
        ax2.set_xticks(range(len(algorithms)))
        ax2.set_xticklabels([algo.replace('_', ' ').title() for algo in algorithms], rotation=45)
        ax2.set_yscale('log')
        
        # Plot 3: Cross-validation scores with error bars
        ax3 = axes[1, 0]
        cv_means = [self.results[algo]['cv_score_mean'] for algo in algorithms]
        cv_stds = [self.results[algo]['cv_score_std'] for algo in algorithms]
        
        bars3 = ax3.bar(range(len(algorithms)), cv_means, yerr=cv_stds, 
                       capsize=5, color='lightblue', alpha=0.7)
        ax3.set_xlabel('Algorithms')
        ax3.set_ylabel('Cross-Validation Score')
        ax3.set_title('Cross-Validation Performance')
        ax3.set_xticks(range(len(algorithms)))
        ax3.set_xticklabels([algo.replace('_', ' ').title() for algo in algorithms], rotation=45)
        
        # Plot 4: Performance vs Speed scatter
        ax4 = axes[1, 1]
        scatter_x = train_times
        scatter_y = metric_values
        
        scatter = ax4.scatter(scatter_x, scatter_y, 
                            s=[100] * len(algorithms), 
                            c=colors[:len(algorithms)],
                            alpha=0.7)
        
        # Add algorithm labels
        for i, algo in enumerate(algorithms):
            ax4.annotate(algo.replace('_', ' ').title(), 
                        (scatter_x[i], scatter_y[i]),
                        xytext=(5, 5), textcoords='offset points',
                        fontsize=8, ha='left')
        
        ax4.set_xlabel('Training Time (seconds)')
        ax4.set_ylabel(metric_name)
        ax4.set_title(f'{metric_name} vs Training Time')
        ax4.set_xscale('log')
        
        plt.tight_layout()
        plt.show()
    
    def statistical_significance_test(self):
        """Perform statistical significance tests between algorithms"""
        if not self.results or len(self.results) < 2:
            print("❌ Need at least 2 algorithms for statistical testing")
            return
        
        print("\n📊 STATISTICAL SIGNIFICANCE TESTING")
        print("=" * 50)
        
        algorithms = list(self.results.keys())
        
        # Get cross-validation scores for each algorithm
        cv_scores = {}
        for algo in algorithms:
            model = self.results[algo]['model']
            # Re-run CV to get individual fold scores
            scores = cross_val_score(
                model, 
                # Use the same data preparation
                StandardScaler().fit_transform(self.X) if hasattr(self, 'X') else self.X,
                self.y if hasattr(self, 'y') else None,
                cv=5,
                scoring='accuracy' if self.is_classification else 'r2'
            )
            cv_scores[algo] = scores
        
        # Pairwise t-tests
        print("🔬 Pairwise t-test results (p-values):")
        print("-" * 40)
        
        for i, algo1 in enumerate(algorithms):
            for j, algo2 in enumerate(algorithms):
                if i < j:  # Avoid duplicate pairs
                    statistic, p_value = stats.ttest_rel(
                        cv_scores[algo1], cv_scores[algo2]
                    )
                    
                    significance = ""
                    if p_value < 0.001:
                        significance = "*** (highly significant)"
                    elif p_value < 0.01:
                        significance = "** (significant)"
                    elif p_value < 0.05:
                        significance = "* (marginally significant)"
                    else:
                        significance = "(not significant)"
                    
                    algo1_clean = algo1.replace('_', ' ').title()
                    algo2_clean = algo2.replace('_', ' ').title()
                    
                    print(f"{algo1_clean} vs {algo2_clean}: p={p_value:.4f} {significance}")
        
        print("\n* p < 0.05, ** p < 0.01, *** p < 0.001")
    
    def get_winner(self):
        """Determine the best performing algorithm"""
        if not self.results:
            return None
        
        if self.is_classification:
            metric = 'accuracy'
        else:
            metric = 'r2_score'
        
        best_algo = max(self.results.keys(), key=lambda x: self.results[x][metric])
        best_score = self.results[best_algo][metric]
        
        return best_algo, best_score
    
    def print_race_summary(self):
        """Print comprehensive race summary"""
        if not self.results:
            print("❌ No race results available")
            return
        
        print("\n🏆 RACE SUMMARY")
        print("=" * 50)
        
        # Winner
        winner, score = self.get_winner()
        print(f"🥇 Winner: {winner.replace('_', ' ').title()}")
        print(f"   Score: {score:.4f} ({'Accuracy' if self.is_classification else 'R² Score'})")
        print()
        
        # Performance table
        print("📊 DETAILED RESULTS TABLE")
        print("-" * 50)
        results_table = self.create_results_table()
        print(results_table.to_string(index=False))
        
        # Speed analysis
        fastest_train = min(self.results.keys(), key=lambda x: self.results[x]['training_time'])
        fastest_pred = min(self.results.keys(), key=lambda x: self.results[x]['prediction_time'])
        
        print(f"\n⚡ Speed Analysis:")
        print(f"   Fastest Training: {fastest_train.replace('_', ' ').title()} ({self.results[fastest_train]['training_time']:.4f}s)")
        print(f"   Fastest Prediction: {fastest_pred.replace('_', ' ').title()} ({self.results[fastest_pred]['prediction_time']:.6f}s)")

def quick_test():
    """Quick test with generated data"""
    print("🧪 QUICK ALGORITHM RACE TEST")
    print("=" * 40)
    
    # Generate test data
    from sklearn.datasets import make_classification, make_regression
    
    print("🎲 Generating test datasets...")
    
    # Classification test
    X_class, y_class = make_classification(
        n_samples=1000, n_features=10, n_informative=5, 
        n_redundant=2, n_classes=3, random_state=42
    )
    
    # Regression test  
    X_reg, y_reg = make_regression(
        n_samples=1000, n_features=10, noise=0.1, random_state=42
    )
    
    for problem_type, (X, y) in [("Classification", (X_class, y_class)), ("Regression", (X_reg, y_reg))]:
        print(f"\n🏁 {problem_type} Race:")
        print("-" * 30)
        
        racer = AlgorithmRacer()
        racer.load_data(X=X, y=y)
        
        # Race with default algorithms
        results = racer.race_algorithms(X, y)
        
        # Print summary
        racer.print_race_summary()
        
        # Show plots
        print("\n📊 Generating performance plots...")
        racer.plot_performance_comparison()

def main():
    """Main function with command line interface"""
    parser = argparse.ArgumentParser(
        description="Race multiple ML algorithms on the same dataset"
    )
    parser.add_argument('--data_path', type=str, help='Path to dataset CSV file')
    parser.add_argument('--algorithms', type=str, 
                       help='Comma-separated list of algorithms to race')
    parser.add_argument('--cv_folds', type=int, default=5, 
                       help='Number of cross-validation folds')
    parser.add_argument('--no_scaling', action='store_true', 
                       help='Disable feature scaling')
    parser.add_argument('--quick_test', action='store_true', 
                       help='Run quick test with generated data')
    parser.add_argument('--stats_test', action='store_true', 
                       help='Perform statistical significance testing')
    parser.add_argument('--no_plots', action='store_true', 
                       help='Skip performance plots')
    
    args = parser.parse_args()
    
    if args.quick_test:
        quick_test()
        return
    
    if not args.data_path:
        print("❌ Please provide a dataset path or use --quick_test")
        print("\nAvailable algorithms:")
        print("Classification: logistic_regression, random_forest, svm, naive_bayes, knn, decision_tree")
        print("Regression: linear_regression, random_forest, svm, knn, decision_tree")
        if XGBOOST_AVAILABLE:
            print("Both: xgboost (if installed)")
        return
    
    try:
        # Initialize racer
        racer = AlgorithmRacer()
        
        # Load data
        print(f"📂 Loading data from {args.data_path}...")
        X, y = racer.load_data(args.data_path)
        print(f"✅ Loaded dataset: {X.shape[0]} samples, {X.shape[1]} features")
        print(f"🎯 Problem type: {'Classification' if racer.is_classification else 'Regression'}")
        
        # Parse algorithms
        algorithms = None
        if args.algorithms:
            algorithms = [algo.strip() for algo in args.algorithms.split(',')]
        
        # Race algorithms
        results = racer.race_algorithms(
            X, y, 
            algorithms=algorithms,
            cv_folds=args.cv_folds,
            scale_features=not args.no_scaling
        )
        
        # Print results
        racer.print_race_summary()
        
        # Statistical testing
        if args.stats_test and len(results) > 1:
            racer.X, racer.y = X, y  # Store for statistical testing
            racer.statistical_significance_test()
        
        # Plot results
        if not args.no_plots:
            print("\n📊 Generating performance comparison plots...")
            racer.plot_performance_comparison()
        
    except Exception as e:
        print(f"❌ Error: {e}")

if __name__ == "__main__":
    main()