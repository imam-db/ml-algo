#!/usr/bin/env python3
"""
Parameter Playground
====================

Interactive tool to visualize how hyperparameters affect algorithm behavior in real-time.
Provides sliders and controls to tune parameters and see immediate visual feedback.

Usage: uv run python parameter_playground.py [options]
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.widgets import Slider, Button, CheckButtons
import argparse
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression, LogisticRegression, Ridge, Lasso
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.svm import SVC, SVR
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.datasets import make_classification, make_regression, make_blobs, make_moons
from sklearn.metrics import accuracy_score, mean_squared_error, r2_score
import warnings
warnings.filterwarnings('ignore')

try:
    import xgboost as xgb
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False

class ParameterPlayground:
    """Interactive parameter tuning visualization"""
    
    def __init__(self):
        self.current_algorithm = 'linear_regression'
        self.current_data = None
        self.current_model = None
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.scaler = None
        self.is_classification = False
        
        # Define algorithm parameters
        self.algorithm_params = {
            'linear_regression': {},
            'ridge_regression': {'alpha': (0.1, 100.0, 1.0)},
            'lasso_regression': {'alpha': (0.01, 10.0, 1.0)},
            'logistic_regression': {'C': (0.01, 100.0, 1.0)},
            'random_forest': {
                'n_estimators': (10, 200, 100),
                'max_depth': (1, 20, 10),
                'min_samples_split': (2, 20, 2),
                'min_samples_leaf': (1, 10, 1)
            },
            'svm': {
                'C': (0.1, 100.0, 1.0),
                'gamma': (0.001, 10.0, 1.0)
            },
            'knn': {
                'n_neighbors': (1, 50, 5),
                'weights': ['uniform', 'distance']
            },
            'decision_tree': {
                'max_depth': (1, 20, 10),
                'min_samples_split': (2, 20, 2),
                'min_samples_leaf': (1, 10, 1)
            }
        }
        
        if XGBOOST_AVAILABLE:
            self.algorithm_params['xgboost'] = {
                'n_estimators': (10, 300, 100),
                'max_depth': (1, 10, 6),
                'learning_rate': (0.01, 1.0, 0.1),
                'subsample': (0.5, 1.0, 1.0)
            }
        
        # Available algorithms
        self.classification_algorithms = [
            'logistic_regression', 'random_forest', 'svm', 'knn', 'decision_tree'
        ]
        self.regression_algorithms = [
            'linear_regression', 'ridge_regression', 'lasso_regression', 
            'random_forest', 'svm', 'knn', 'decision_tree'
        ]
        
        if XGBOOST_AVAILABLE:
            self.classification_algorithms.append('xgboost')
            self.regression_algorithms.append('xgboost')
    
    def generate_sample_data(self, data_type: str = 'classification', n_samples: int = 300):
        """Generate sample data for testing"""
        if data_type == 'classification':
            # Generate classification data
            if np.random.random() < 0.3:  # 30% chance for moons
                X, y = make_moons(n_samples=n_samples, noise=0.1, random_state=42)
            elif np.random.random() < 0.5:  # 20% chance for blobs
                X, y = make_blobs(n_samples=n_samples, centers=3, n_features=2, 
                                random_state=42, cluster_std=1.0)
            else:  # 50% chance for general classification
                X, y = make_classification(n_samples=n_samples, n_features=2, n_redundant=0,
                                         n_informative=2, n_clusters_per_class=1, 
                                         random_state=42)
            self.is_classification = True
        else:
            # Generate regression data
            X, y = make_regression(n_samples=n_samples, n_features=2, noise=10, 
                                 random_state=42)
            self.is_classification = False
        
        return X, y
    
    def prepare_data(self, X, y, test_size=0.2):
        """Prepare data for training"""
        # Split data
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X, y, test_size=test_size, random_state=42,
            stratify=y if self.is_classification else None
        )
        
        # Scale features
        self.scaler = StandardScaler()
        self.X_train = self.scaler.fit_transform(self.X_train)
        self.X_test = self.scaler.transform(self.X_test)
        
        self.current_data = (X, y)
    
    def get_model(self, algorithm: str, **params):
        """Get model instance with specified parameters"""
        if algorithm == 'linear_regression':
            return LinearRegression()
        elif algorithm == 'ridge_regression':
            return Ridge(alpha=params.get('alpha', 1.0))
        elif algorithm == 'lasso_regression':
            return Lasso(alpha=params.get('alpha', 1.0))
        elif algorithm == 'logistic_regression':
            return LogisticRegression(C=params.get('C', 1.0), random_state=42, max_iter=1000)
        elif algorithm == 'random_forest':
            if self.is_classification:
                return RandomForestClassifier(
                    n_estimators=int(params.get('n_estimators', 100)),
                    max_depth=int(params.get('max_depth', 10)) if params.get('max_depth', 10) < 20 else None,
                    min_samples_split=int(params.get('min_samples_split', 2)),
                    min_samples_leaf=int(params.get('min_samples_leaf', 1)),
                    random_state=42
                )
            else:
                return RandomForestRegressor(
                    n_estimators=int(params.get('n_estimators', 100)),
                    max_depth=int(params.get('max_depth', 10)) if params.get('max_depth', 10) < 20 else None,
                    min_samples_split=int(params.get('min_samples_split', 2)),
                    min_samples_leaf=int(params.get('min_samples_leaf', 1)),
                    random_state=42
                )
        elif algorithm == 'svm':
            if self.is_classification:
                return SVC(C=params.get('C', 1.0), gamma=params.get('gamma', 1.0), 
                          kernel='rbf', random_state=42)
            else:
                return SVR(C=params.get('C', 1.0), gamma=params.get('gamma', 1.0), kernel='rbf')
        elif algorithm == 'knn':
            weights = params.get('weights', 'uniform')
            if isinstance(weights, (list, tuple)):
                weights = weights[0] if weights else 'uniform'
            
            if self.is_classification:
                return KNeighborsClassifier(
                    n_neighbors=int(params.get('n_neighbors', 5)),
                    weights=weights
                )
            else:
                return KNeighborsRegressor(
                    n_neighbors=int(params.get('n_neighbors', 5)),
                    weights=weights
                )
        elif algorithm == 'decision_tree':
            if self.is_classification:
                return DecisionTreeClassifier(
                    max_depth=int(params.get('max_depth', 10)) if params.get('max_depth', 10) < 20 else None,
                    min_samples_split=int(params.get('min_samples_split', 2)),
                    min_samples_leaf=int(params.get('min_samples_leaf', 1)),
                    random_state=42
                )
            else:
                return DecisionTreeRegressor(
                    max_depth=int(params.get('max_depth', 10)) if params.get('max_depth', 10) < 20 else None,
                    min_samples_split=int(params.get('min_samples_split', 2)),
                    min_samples_leaf=int(params.get('min_samples_leaf', 1)),
                    random_state=42
                )
        elif algorithm == 'xgboost' and XGBOOST_AVAILABLE:
            if self.is_classification:
                return xgb.XGBClassifier(
                    n_estimators=int(params.get('n_estimators', 100)),
                    max_depth=int(params.get('max_depth', 6)),
                    learning_rate=params.get('learning_rate', 0.1),
                    subsample=params.get('subsample', 1.0),
                    random_state=42,
                    eval_metric='logloss'
                )
            else:
                return xgb.XGBRegressor(
                    n_estimators=int(params.get('n_estimators', 100)),
                    max_depth=int(params.get('max_depth', 6)),
                    learning_rate=params.get('learning_rate', 0.1),
                    subsample=params.get('subsample', 1.0),
                    random_state=42
                )
        
        raise ValueError(f"Unknown algorithm: {algorithm}")
    
    def create_decision_boundary(self, model, h=0.02):
        """Create decision boundary mesh for visualization"""
        if self.X_train is None:
            return None, None, None
        
        x_min, x_max = self.X_train[:, 0].min() - 1, self.X_train[:, 0].max() + 1
        y_min, y_max = self.X_train[:, 1].min() - 1, self.X_train[:, 1].max() + 1
        xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                           np.arange(y_min, y_max, h))
        
        mesh_points = np.c_[xx.ravel(), yy.ravel()]
        
        try:
            if self.is_classification:
                Z = model.predict(mesh_points)
            else:
                Z = model.predict(mesh_points)
            Z = Z.reshape(xx.shape)
            return xx, yy, Z
        except:
            return None, None, None
    
    def evaluate_model(self, model):
        """Evaluate model performance"""
        try:
            train_pred = model.predict(self.X_train)
            test_pred = model.predict(self.X_test)
            
            if self.is_classification:
                train_score = accuracy_score(self.y_train, train_pred)
                test_score = accuracy_score(self.y_test, test_pred)
                metric_name = 'Accuracy'
            else:
                train_score = r2_score(self.y_train, train_pred)
                test_score = r2_score(self.y_test, test_pred)
                metric_name = 'R² Score'
            
            return train_score, test_score, metric_name
        except:
            return 0.0, 0.0, 'Error'
    
    def launch_interactive_playground(self, algorithm: str = 'random_forest', 
                                   data_type: str = 'classification'):
        """Launch interactive parameter tuning interface"""
        self.current_algorithm = algorithm
        
        # Generate sample data
        X, y = self.generate_sample_data(data_type)
        self.prepare_data(X, y)
        
        # Validate algorithm for data type
        available_algos = (self.classification_algorithms if self.is_classification 
                         else self.regression_algorithms)
        if algorithm not in available_algos:
            print(f"⚠️  Algorithm '{algorithm}' not available for {data_type}")
            algorithm = available_algos[0]
            self.current_algorithm = algorithm
        
        # Create figure and subplots
        fig = plt.figure(figsize=(16, 10))
        
        # Main plot for data and decision boundary
        ax_main = plt.subplot2grid((3, 4), (0, 0), colspan=2, rowspan=2)
        
        # Performance plot
        ax_perf = plt.subplot2grid((3, 4), (0, 2), colspan=2)
        
        # Parameter info
        ax_info = plt.subplot2grid((3, 4), (1, 2), colspan=2)
        ax_info.axis('off')
        
        # Slider area
        slider_ax_start = 0.05
        slider_height = 0.25
        
        fig.suptitle(f'Parameter Playground - {algorithm.replace("_", " ").title()}', 
                    fontsize=16, y=0.95)
        
        # Get algorithm parameters
        param_info = self.algorithm_params.get(algorithm, {})
        sliders = {}
        slider_axes = {}
        
        # Create sliders for parameters
        y_pos = 0.35
        for i, (param_name, param_range) in enumerate(param_info.items()):
            if isinstance(param_range, tuple) and len(param_range) == 3:
                min_val, max_val, default_val = param_range
                
                # Create slider axis
                slider_ax = plt.axes([slider_ax_start, y_pos - i*0.05, 0.3, 0.03])
                slider_axes[param_name] = slider_ax
                
                # Create slider
                slider = Slider(slider_ax, param_name, min_val, max_val, 
                              valinit=default_val, valfmt='%.3f')
                sliders[param_name] = slider
        
        # Initial model and plot
        initial_params = {name: slider.val for name, slider in sliders.items()}
        current_model = self.get_model(algorithm, **initial_params)
        
        def update_plots():
            """Update all plots with current parameters"""
            # Get current parameter values
            current_params = {name: slider.val for name, slider in sliders.items()}
            
            # Create new model
            model = self.get_model(algorithm, **current_params)
            
            try:
                # Fit model
                model.fit(self.X_train, self.y_train)
                
                # Clear main plot
                ax_main.clear()
                
                # Plot training data
                if self.is_classification:
                    scatter = ax_main.scatter(self.X_train[:, 0], self.X_train[:, 1], 
                                            c=self.y_train, cmap='viridis', alpha=0.6, s=50)
                    
                    # Plot decision boundary
                    xx, yy, Z = self.create_decision_boundary(model)
                    if xx is not None:
                        ax_main.contourf(xx, yy, Z, alpha=0.3, cmap='viridis')
                        ax_main.contour(xx, yy, Z, colors='black', linewidths=0.5, alpha=0.5)
                else:
                    # For regression, create a surface plot
                    xx, yy, Z = self.create_decision_boundary(model, h=0.1)
                    if xx is not None:
                        contour = ax_main.contourf(xx, yy, Z, levels=20, alpha=0.6, cmap='viridis')
                        ax_main.contour(xx, yy, Z, levels=20, colors='black', alpha=0.3, linewidths=0.5)
                    
                    # Scatter plot with color-coded target values
                    scatter = ax_main.scatter(self.X_train[:, 0], self.X_train[:, 1], 
                                            c=self.y_train, cmap='plasma', s=50, alpha=0.8)
                
                ax_main.set_xlabel('Feature 1')
                ax_main.set_ylabel('Feature 2')
                ax_main.set_title(f'{algorithm.replace("_", " ").title()} - Decision Boundary')
                ax_main.grid(True, alpha=0.3)
                
                # Evaluate model
                train_score, test_score, metric_name = self.evaluate_model(model)
                
                # Update performance plot
                ax_perf.clear()
                scores = [train_score, test_score]
                labels = ['Training', 'Test']
                colors = ['skyblue', 'orange']
                
                bars = ax_perf.bar(labels, scores, color=colors, alpha=0.7)
                ax_perf.set_ylabel(metric_name)
                ax_perf.set_title('Model Performance')
                ax_perf.set_ylim(0, 1.0 if self.is_classification else max(1.0, max(scores) + 0.1))
                
                # Add value labels on bars
                for bar, score in zip(bars, scores):
                    height = bar.get_height()
                    ax_perf.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                               f'{score:.3f}', ha='center', va='bottom')
                
                ax_perf.grid(True, alpha=0.3)
                
                # Update parameter info
                ax_info.clear()
                ax_info.axis('off')
                
                info_text = f"📊 Current Parameters:\n"
                for param_name, value in current_params.items():
                    if isinstance(value, float):
                        info_text += f"  {param_name}: {value:.3f}\n"
                    else:
                        info_text += f"  {param_name}: {value}\n"
                
                info_text += f"\n🎯 Performance:\n"
                info_text += f"  Training {metric_name}: {train_score:.3f}\n"
                info_text += f"  Test {metric_name}: {test_score:.3f}\n"
                
                # Add overfitting warning
                if train_score - test_score > 0.1:
                    info_text += f"\n⚠️  Possible overfitting detected!\n"
                elif test_score > train_score:
                    info_text += f"\n✅ Good generalization!\n"
                
                ax_info.text(0.1, 0.9, info_text, transform=ax_info.transAxes, 
                           fontsize=10, verticalalignment='top', fontfamily='monospace')
                
            except Exception as e:
                ax_main.clear()
                ax_main.text(0.5, 0.5, f'Error: {str(e)}', ha='center', va='center', 
                           transform=ax_main.transAxes)
            
            plt.draw()
        
        # Connect slider events
        for slider in sliders.values():
            slider.on_changed(lambda val: update_plots())
        
        # Add reset button
        reset_ax = plt.axes([0.05, 0.02, 0.1, 0.04])
        reset_button = Button(reset_ax, 'Reset')
        
        def reset_parameters(event):
            for param_name, slider in sliders.items():
                param_range = param_info[param_name]
                if isinstance(param_range, tuple) and len(param_range) == 3:
                    default_val = param_range[2]
                    slider.reset()
            update_plots()
        
        reset_button.on_clicked(reset_parameters)
        
        # Add new data button
        new_data_ax = plt.axes([0.2, 0.02, 0.15, 0.04])
        new_data_button = Button(new_data_ax, 'New Data')
        
        def generate_new_data(event):
            X, y = self.generate_sample_data('classification' if self.is_classification else 'regression')
            self.prepare_data(X, y)
            update_plots()
        
        new_data_button.on_clicked(generate_new_data)
        
        # Initial plot update
        update_plots()
        
        plt.tight_layout()
        plt.subplots_adjust(bottom=0.1)
        
        print("🎛️ PARAMETER PLAYGROUND LAUNCHED!")
        print("=" * 40)
        print(f"📊 Algorithm: {algorithm.replace('_', ' ').title()}")
        print(f"🎯 Problem Type: {'Classification' if self.is_classification else 'Regression'}")
        print(f"📈 Dataset: {len(self.y_train)} training samples")
        print("\n🎮 Interactive Controls:")
        print("  • Use sliders to adjust parameters")
        print("  • Click 'Reset' to restore defaults")
        print("  • Click 'New Data' to generate new dataset")
        print("  • Watch decision boundary and performance change in real-time!")
        print("\n💡 Tips:")
        print("  • Look for overfitting when training score >> test score")
        print("  • Try different parameter combinations")
        print("  • Close window to exit")
        
        plt.show()

def quick_demo():
    """Quick demonstration of parameter playground"""
    playground = ParameterPlayground()
    
    print("🎛️ PARAMETER PLAYGROUND - QUICK DEMO")
    print("=" * 45)
    
    algorithms_to_demo = ['random_forest', 'svm', 'decision_tree']
    
    for algorithm in algorithms_to_demo:
        print(f"\n🔧 Demoing {algorithm.replace('_', ' ').title()}...")
        
        # Generate sample data
        X, y = playground.generate_sample_data('classification')
        playground.prepare_data(X, y)
        
        # Test with default parameters
        default_params = {}
        param_info = playground.algorithm_params.get(algorithm, {})
        for param_name, param_range in param_info.items():
            if isinstance(param_range, tuple) and len(param_range) == 3:
                default_params[param_name] = param_range[2]
        
        model = playground.get_model(algorithm, **default_params)
        model.fit(playground.X_train, playground.y_train)
        
        train_score, test_score, metric_name = playground.evaluate_model(model)
        
        print(f"   📊 Default Performance:")
        print(f"      Training {metric_name}: {train_score:.3f}")
        print(f"      Test {metric_name}: {test_score:.3f}")
        print(f"      Parameters: {default_params}")

def main():
    """Main function with command line interface"""
    parser = argparse.ArgumentParser(
        description="Interactive parameter tuning playground"
    )
    parser.add_argument('--algorithm', default='random_forest',
                       choices=['linear_regression', 'ridge_regression', 'lasso_regression',
                               'logistic_regression', 'random_forest', 'svm', 'knn', 
                               'decision_tree', 'xgboost'],
                       help='Algorithm to tune')
    parser.add_argument('--data_type', default='classification',
                       choices=['classification', 'regression'],
                       help='Type of problem')
    parser.add_argument('--demo', action='store_true',
                       help='Run quick demo')
    
    args = parser.parse_args()
    
    if args.demo:
        quick_demo()
    else:
        playground = ParameterPlayground()
        playground.launch_interactive_playground(args.algorithm, args.data_type)

if __name__ == "__main__":
    main()