#!/usr/bin/env python3
"""
Algorithm Animator
==================

Step-by-step visual animations showing how algorithms work internally.
Provides educational animations for understanding algorithm mechanics.

Usage: uv run python algorithm_animator.py [options]
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import argparse
from sklearn.datasets import make_blobs, make_classification, make_regression
import warnings
warnings.filterwarnings('ignore')

class AlgorithmAnimator:
    """Animated visualizations of algorithm internals"""
    
    def __init__(self):
        self.fig = None
        self.ax = None
        self.animation = None
        self.data = {}
        
        # Color schemes
        self.colors = {
            'clusters': ['#FF6B6B', '#4ECDC4', '#45B7D1', '#FFA07A', '#98D8C8', '#F7DC6F'],
            'points': '#2C3E50',
            'centroids': '#E74C3C',
            'lines': '#34495E',
            'decision': '#9B59B6',
            'gradient': '#E67E22'
        }
    
    def load_data(self, X, y=None):
        """Load data for animation"""
        self.data['X'] = np.array(X)
        if y is not None:
            self.data['y'] = np.array(y)
        else:
            self.data['y'] = None
    
    def animate_kmeans(self, n_clusters=3, max_iters=20, interval=1000, save_path=None, show=True):
        """Animate K-Means clustering algorithm"""
        print(f"🎬 Animating K-Means with {n_clusters} clusters...")
        
        X = self.data['X']
        
        # Initialize centroids randomly
        centroids_history = []
        assignments_history = []
        
        # Random initial centroids
        x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
        y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
        
        centroids = np.array([
            [np.random.uniform(x_min, x_max), np.random.uniform(y_min, y_max)]
            for _ in range(n_clusters)
        ])
        
        # Run K-means and record each step
        for iteration in range(max_iters):
            # Assign points to nearest centroid
            distances = np.sqrt(((X - centroids[:, np.newaxis])**2).sum(axis=2))
            assignments = np.argmin(distances, axis=0)
            
            centroids_history.append(centroids.copy())
            assignments_history.append(assignments.copy())
            
            # Update centroids
            new_centroids = np.array([
                X[assignments == k].mean(axis=0) if np.sum(assignments == k) > 0 else centroids[k]
                for k in range(n_clusters)
            ])
            
            # Check for convergence
            if np.allclose(centroids, new_centroids):
                centroids_history.append(new_centroids)
                assignments_history.append(assignments)
                break
            
            centroids = new_centroids
        
        # Create animation
        self.fig, self.ax = plt.subplots(figsize=(12, 8))
        
        def animate(frame):
            self.ax.clear()
            
            if frame < len(centroids_history):
                current_centroids = centroids_history[frame]
                current_assignments = assignments_history[frame]
                
                # Plot points colored by assignment
                for k in range(n_clusters):
                    mask = current_assignments == k
                    if np.any(mask):
                        self.ax.scatter(X[mask, 0], X[mask, 1], 
                                      c=self.colors['clusters'][k % len(self.colors['clusters'])], 
                                      alpha=0.7, s=50, label=f'Cluster {k+1}')
                
                # Plot centroids
                self.ax.scatter(current_centroids[:, 0], current_centroids[:, 1], 
                               c=self.colors['centroids'], s=200, marker='x', 
                               linewidths=3, label='Centroids')
                
                # Draw lines from points to their assigned centroids
                for i, point in enumerate(X):
                    assigned_centroid = current_centroids[current_assignments[i]]
                    self.ax.plot([point[0], assigned_centroid[0]], 
                               [point[1], assigned_centroid[1]], 
                               c=self.colors['clusters'][current_assignments[i] % len(self.colors['clusters'])], 
                               alpha=0.3, linewidth=1)
                
                self.ax.set_title(f'K-Means Animation - Iteration {frame + 1}', fontsize=16)
            
            self.ax.set_xlabel('Feature 1')
            self.ax.set_ylabel('Feature 2')
            self.ax.legend()
            self.ax.grid(True, alpha=0.3)
            
        self.animation = animation.FuncAnimation(
            self.fig, animate, frames=len(centroids_history), 
            interval=interval, repeat=True, blit=False
        )
        
        if save_path:
            print(f"💾 Saving animation to {save_path}...")
            try:
                self.animation.save(save_path, writer='pillow', fps=1)
            except Exception as e:
                print(f"❌ Could not save animation: {e}")
        
        plt.tight_layout()
        if show:
            plt.show()
        
        return self.animation
    
    def animate_gradient_descent(self, learning_rate=0.01, max_iters=100, interval=200, save_path=None, show=True):
        """Animate gradient descent for linear regression"""
        print(f"🎬 Animating Gradient Descent (lr={learning_rate})...")
        
        X = self.data['X'][:, 0:1]  # Use only first feature for simplicity
        y = self.data['y']
        
        # Add bias term
        X_with_bias = np.c_[np.ones(X.shape[0]), X]
        
        # Initialize parameters
        theta_history = []
        cost_history = []
        
        # Random initialization
        theta = np.random.normal(0, 1, 2)
        
        # Gradient descent steps
        for i in range(max_iters):
            # Predictions
            predictions = X_with_bias.dot(theta)
            
            # Cost
            cost = np.mean((predictions - y) ** 2) / 2
            
            # Gradients
            gradients = X_with_bias.T.dot(predictions - y) / len(y)
            
            # Store history
            theta_history.append(theta.copy())
            cost_history.append(cost)
            
            # Update parameters
            theta = theta - learning_rate * gradients
            
            # Early stopping if converged
            if i > 0 and abs(cost_history[-2] - cost_history[-1]) < 1e-8:
                break
        
        # Create animation with subplots
        self.fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
        
        def animate(frame):
            if frame >= len(theta_history):
                frame = len(theta_history) - 1
            
            current_theta = theta_history[frame]
            
            # Left plot: Data and regression line
            ax1.clear()
            
            # Plot data points
            ax1.scatter(X, y, c=self.colors['points'], alpha=0.7, s=50)
            
            # Plot regression line
            x_line = np.linspace(X.min(), X.max(), 100)
            y_line = current_theta[0] + current_theta[1] * x_line
            ax1.plot(x_line, y_line, c=self.colors['gradient'], linewidth=3, 
                    label=f'y = {current_theta[0]:.2f} + {current_theta[1]:.2f}x')
            
            # Plot residuals
            predictions = current_theta[0] + current_theta[1] * X.flatten()
            for i, (xi, yi, pi) in enumerate(zip(X.flatten(), y, predictions)):
                ax1.plot([xi, xi], [yi, pi], c=self.colors['gradient'], 
                        alpha=0.4, linewidth=1)
            
            ax1.set_title(f'Gradient Descent - Iteration {frame + 1}')
            ax1.set_xlabel('Feature')
            ax1.set_ylabel('Target')
            ax1.legend()
            ax1.grid(True, alpha=0.3)
            
            # Right plot: Cost function
            ax2.clear()
            
            iterations = range(1, len(cost_history[:frame+1]) + 1)
            ax2.plot(iterations, cost_history[:frame+1], c=self.colors['gradient'], 
                    linewidth=2, marker='o', markersize=4)
            ax2.scatter(frame + 1, cost_history[frame], c=self.colors['centroids'], 
                       s=100, zorder=5)
            
            ax2.set_title('Cost Function Over Time')
            ax2.set_xlabel('Iteration')
            ax2.set_ylabel('Cost')
            ax2.grid(True, alpha=0.3)
            
        self.animation = animation.FuncAnimation(
            self.fig, animate, frames=len(theta_history), 
            interval=interval, repeat=True, blit=False
        )
        
        if save_path:
            print(f"💾 Saving animation to {save_path}...")
            try:
                self.animation.save(save_path, writer='pillow', fps=5)
            except Exception as e:
                print(f"❌ Could not save animation: {e}")
        
        plt.tight_layout()
        if show:
            plt.show()
        
        return self.animation
    
    def animate_knn_classification(self, k=5, query_point=None, interval=1500, save_path=None, show=True):
        """Animate K-NN classification process"""
        print(f"🎬 Animating K-NN Classification (k={k})...")
        
        X = self.data['X']
        y = self.data['y']
        
        if query_point is None:
            # Generate a query point in the middle of the data
            query_point = np.array([X[:, 0].mean(), X[:, 1].mean()])
        
        # Calculate distances to all points
        distances = np.sqrt(np.sum((X - query_point)**2, axis=1))
        sorted_indices = np.argsort(distances)
        
        # Create animation steps
        steps = []
        
        # Step 1: Show all data and query point
        steps.append({'type': 'initial', 'data': 'all'})
        
        # Steps 2-k+1: Highlight each nearest neighbor one by one
        for i in range(k):
            steps.append({'type': 'highlight_neighbor', 'neighbor_idx': sorted_indices[i], 'count': i+1})
        
        # Final step: Show classification result
        neighbor_classes = y[sorted_indices[:k]]
        prediction = np.bincount(neighbor_classes).argmax()
        steps.append({'type': 'final_prediction', 'prediction': prediction})
        
        # Create animation
        self.fig, self.ax = plt.subplots(figsize=(12, 8))
        
        def animate(frame):
            self.ax.clear()
            
            current_step = steps[frame]
            
            # Plot all data points
            unique_classes = np.unique(y)
            for i, class_val in enumerate(unique_classes):
                mask = y == class_val
                self.ax.scatter(X[mask, 0], X[mask, 1], 
                               c=self.colors['clusters'][i % len(self.colors['clusters'])], 
                               alpha=0.6, s=50, label=f'Class {class_val}')
            
            # Plot query point
            self.ax.scatter(query_point[0], query_point[1], 
                           c=self.colors['centroids'], s=200, marker='*', 
                           edgecolors='black', linewidth=2, label='Query Point', zorder=10)
            
            if current_step['type'] == 'initial':
                self.ax.set_title(f'K-NN Classification - Step 1: Query Point Introduced')
                
            elif current_step['type'] == 'highlight_neighbor':
                neighbor_idx = current_step['neighbor_idx']
                count = current_step['count']
                
                # Highlight nearest neighbors found so far
                for i in range(count):
                    idx = sorted_indices[i]
                    self.ax.scatter(X[idx, 0], X[idx, 1], 
                                   s=200, facecolors='none', 
                                   edgecolors=self.colors['decision'], linewidth=3)
                    
                    # Draw line to query point
                    self.ax.plot([query_point[0], X[idx, 0]], 
                               [query_point[1], X[idx, 1]], 
                               c=self.colors['decision'], linewidth=2, alpha=0.7)
                    
                    # Add distance annotation
                    distance = distances[idx]
                    mid_x = (query_point[0] + X[idx, 0]) / 2
                    mid_y = (query_point[1] + X[idx, 1]) / 2
                    self.ax.annotate(f'd={distance:.2f}', (mid_x, mid_y), 
                                   fontsize=10, ha='center', 
                                   bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8))
                
                self.ax.set_title(f'K-NN Classification - Step {frame + 1}: Found {count}/{k} Nearest Neighbors')
                
            elif current_step['type'] == 'final_prediction':
                prediction = current_step['prediction']
                
                # Highlight all k nearest neighbors
                for i in range(k):
                    idx = sorted_indices[i]
                    self.ax.scatter(X[idx, 0], X[idx, 1], 
                                   s=200, facecolors='none', 
                                   edgecolors=self.colors['decision'], linewidth=3)
                
                # Change query point color to predicted class
                self.ax.scatter(query_point[0], query_point[1], 
                               c=self.colors['clusters'][prediction % len(self.colors['clusters'])], 
                               s=300, marker='*', edgecolors='black', linewidth=3, 
                               label=f'Predicted: Class {prediction}', zorder=10)
                
                # Show voting results
                vote_counts = np.bincount(y[sorted_indices[:k]], minlength=len(unique_classes))
                vote_text = ', '.join([f'Class {i}: {count}' for i, count in enumerate(vote_counts) if count > 0])
                
                self.ax.set_title(f'K-NN Classification - Final: Votes ({vote_text}) → Prediction: Class {prediction}')
            
            self.ax.set_xlabel('Feature 1')
            self.ax.set_ylabel('Feature 2')
            self.ax.legend()
            self.ax.grid(True, alpha=0.3)
        
        self.animation = animation.FuncAnimation(
            self.fig, animate, frames=len(steps), 
            interval=interval, repeat=True, blit=False
        )
        
        if save_path:
            print(f"💾 Saving animation to {save_path}...")
            try:
                self.animation.save(save_path, writer='pillow', fps=0.67)
            except Exception as e:
                print(f"❌ Could not save animation: {e}")
        
        plt.tight_layout()
        if show:
            plt.show()
        
        return self.animation
    
    def animate_linear_regression_fitting(self, interval=500, save_path=None, show=True):
        """Animate linear regression line fitting process"""
        print("🎬 Animating Linear Regression Fitting...")
        
        X = self.data['X'][:, 0] if len(self.data['X'].shape) > 1 else self.data['X']
        y = self.data['y']
        
        # Calculate best fit parameters
        X_mean, y_mean = X.mean(), y.mean()
        best_slope = np.sum((X - X_mean) * (y - y_mean)) / np.sum((X - X_mean)**2)
        best_intercept = y_mean - best_slope * X_mean
        
        # Create animation sequence approaching the best fit
        animation_params = []
        
        # Random exploration phase
        for _ in range(15):
            slope = np.random.uniform(-1, 2)
            intercept = np.random.uniform(y.min(), y.max())
            mse = np.mean((y - (slope * X + intercept))**2)
            animation_params.append({'slope': slope, 'intercept': intercept, 'mse': mse, 'phase': 'explore'})
        
        # Converging phase
        for i in range(10):
            alpha = (10 - i) / 10  # Decreasing from 1 to 0
            slope = alpha * animation_params[-1]['slope'] + (1 - alpha) * best_slope
            intercept = alpha * animation_params[-1]['intercept'] + (1 - alpha) * best_intercept
            mse = np.mean((y - (slope * X + intercept))**2)
            animation_params.append({'slope': slope, 'intercept': intercept, 'mse': mse, 'phase': 'converge'})
        
        # Final best fit
        mse = np.mean((y - (best_slope * X + best_intercept))**2)
        animation_params.append({'slope': best_slope, 'intercept': best_intercept, 'mse': mse, 'phase': 'final'})
        
        # Create animation
        self.fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
        
        def animate(frame):
            current_params = animation_params[frame]
            slope = current_params['slope']
            intercept = current_params['intercept']
            mse = current_params['mse']
            phase = current_params['phase']
            
            # Left plot: Data and regression line
            ax1.clear()
            
            # Plot data points
            ax1.scatter(X, y, c=self.colors['points'], alpha=0.7, s=50)
            
            # Plot regression line
            x_line = np.linspace(X.min() - 0.5, X.max() + 0.5, 100)
            y_line = slope * x_line + intercept
            
            color = self.colors['gradient']
            if phase == 'final':
                color = self.colors['centroids']
                linewidth = 4
            else:
                linewidth = 2
            
            ax1.plot(x_line, y_line, c=color, linewidth=linewidth, 
                    label=f'y = {slope:.2f}x + {intercept:.2f}')
            
            # Plot residuals
            predictions = slope * X + intercept
            for xi, yi, pi in zip(X, y, predictions):
                ax1.plot([xi, xi], [yi, pi], c=color, alpha=0.4, linewidth=1)
            
            phase_names = {'explore': 'Exploring', 'converge': 'Converging', 'final': 'Best Fit'}
            ax1.set_title(f'Linear Regression - {phase_names.get(phase, "Fitting")}: MSE = {mse:.2f}')
            ax1.set_xlabel('Feature')
            ax1.set_ylabel('Target')
            ax1.legend()
            ax1.grid(True, alpha=0.3)
            
            # Right plot: MSE over time
            ax2.clear()
            
            frames_so_far = range(1, frame + 2)
            mse_history = [p['mse'] for p in animation_params[:frame+1]]
            
            ax2.plot(frames_so_far, mse_history, c=self.colors['gradient'], 
                    linewidth=2, marker='o', markersize=4)
            ax2.scatter(frame + 1, mse, c=self.colors['centroids'], s=100, zorder=5)
            
            ax2.set_title('Mean Squared Error Over Time')
            ax2.set_xlabel('Iteration')
            ax2.set_ylabel('MSE')
            ax2.grid(True, alpha=0.3)
        
        self.animation = animation.FuncAnimation(
            self.fig, animate, frames=len(animation_params), 
            interval=interval, repeat=True, blit=False
        )
        
        if save_path:
            print(f"💾 Saving animation to {save_path}...")
            try:
                self.animation.save(save_path, writer='pillow', fps=2)
            except Exception as e:
                print(f"❌ Could not save animation: {e}")
        
        plt.tight_layout()
        if show:
            plt.show()
        
        return self.animation

def generate_sample_data(algorithm_type, n_samples=200, **kwargs):
    """Generate appropriate sample data for different algorithms"""
    if algorithm_type == 'kmeans':
        X, _ = make_blobs(n_samples=n_samples, centers=kwargs.get('centers', 3), 
                         n_features=2, cluster_std=1.5, random_state=42)
        return X, None
    
    elif algorithm_type == 'gradient_descent':
        X, y = make_regression(n_samples=n_samples, n_features=1, 
                             noise=kwargs.get('noise', 10), random_state=42)
        return X, y
    
    elif algorithm_type == 'knn':
        X, y = make_classification(n_samples=n_samples, n_features=2, n_redundant=0,
                                 n_informative=2, n_classes=kwargs.get('classes', 3),
                                 n_clusters_per_class=1, cluster_std=1.5, random_state=42)
        return X, y
    
    elif algorithm_type == 'linear_regression':
        X, y = make_regression(n_samples=n_samples, n_features=1, 
                             noise=kwargs.get('noise', 15), random_state=42)
        return X, y
    
    else:
        raise ValueError(f"Unknown algorithm type: {algorithm_type}")

def main():
    """Main function with command line interface"""
    parser = argparse.ArgumentParser(
        description="Animated visualizations of machine learning algorithms"
    )
    parser.add_argument('--algorithm', default='kmeans',
                       choices=['kmeans', 'gradient_descent', 'knn', 'linear_regression'],
                       help='Algorithm to animate')
    parser.add_argument('--n_samples', type=int, default=200,
                       help='Number of samples to generate')
    parser.add_argument('--save', type=str,
                       help='Save animation as GIF file')
    parser.add_argument('--interval', type=int, default=1000,
                       help='Animation interval in milliseconds')
    parser.add_argument('--no_show', action='store_true',
                       help='Run without opening animation windows (headless)')
    
    # Algorithm-specific parameters
    parser.add_argument('--k', type=int, default=3,
                       help='Number of clusters for K-means or neighbors for K-NN')
    parser.add_argument('--learning_rate', type=float, default=0.01,
                       help='Learning rate for gradient descent')
    parser.add_argument('--max_iters', type=int, default=50,
                       help='Maximum iterations for iterative algorithms')
    
    args = parser.parse_args()
    
    # Create animator
    animator = AlgorithmAnimator()
    
    # Generate appropriate data
    print(f"🎲 Generating data for {args.algorithm} animation...")
    X, y = generate_sample_data(args.algorithm, args.n_samples)
    animator.load_data(X, y)
    
    # Run animation based on algorithm
    print(f"🎬 Starting {args.algorithm} animation...")
    
    if args.algorithm == 'kmeans':
        animator.animate_kmeans(n_clusters=args.k, max_iters=args.max_iters, 
                               interval=args.interval, save_path=args.save, show=not args.no_show)
    
    elif args.algorithm == 'gradient_descent':
        animator.animate_gradient_descent(learning_rate=args.learning_rate, 
                                        max_iters=args.max_iters,
                                        interval=args.interval, save_path=args.save, show=not args.no_show)
    
    elif args.algorithm == 'knn':
        animator.animate_knn_classification(k=args.k, interval=args.interval, 
                                          save_path=args.save, show=not args.no_show)
    
    elif args.algorithm == 'linear_regression':
        animator.animate_linear_regression_fitting(interval=args.interval, 
                                                 save_path=args.save, show=not args.no_show)
    
    print("🎉 Animation complete! Close the window to exit.")

if __name__ == "__main__":
    main()
