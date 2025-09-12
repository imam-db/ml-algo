#!/usr/bin/env python3
"""
Data Generator Lab
==================

Create synthetic datasets for algorithm testing and experimentation.
Supports various data patterns and distributions for comprehensive algorithm testing.

Usage: uv run python data_generator.py [options]
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import argparse
import pandas as pd
from sklearn.datasets import (
    make_classification, make_regression, make_blobs, make_circles, 
    make_moons, make_swiss_roll, make_s_curve
)
from typing import Tuple, Optional, Dict, Any
import warnings
warnings.filterwarnings('ignore')

class DataGenerator:
    """Synthetic data generation for ML algorithm testing"""
    
    def __init__(self):
        self.datasets = {}
        self.available_types = [
            'linear', 'polynomial', 'clusters', 'moons', 'circles', 
            'blobs', 'spiral', 'wave', 'checkerboard', 'swiss_roll'
        ]
    
    def generate_linear(self, n_samples: int = 100, noise: float = 0.1, 
                       bias: float = 2.0, slope: float = 1.5) -> Tuple[np.ndarray, np.ndarray]:
        """Generate linear relationship data"""
        np.random.seed(42)
        X = np.random.uniform(-3, 3, n_samples).reshape(-1, 1)
        y = slope * X.ravel() + bias + np.random.normal(0, noise, n_samples)
        return X, y
    
    def generate_polynomial(self, n_samples: int = 100, noise: float = 0.1, 
                          degree: int = 3) -> Tuple[np.ndarray, np.ndarray]:
        """Generate polynomial relationship data"""
        np.random.seed(42)
        X = np.random.uniform(-2, 2, n_samples).reshape(-1, 1)
        
        # Create polynomial features
        if degree == 2:
            y = 0.5 * X.ravel()**2 + 0.3 * X.ravel() + 1
        elif degree == 3:
            y = 0.2 * X.ravel()**3 - 0.5 * X.ravel()**2 + 0.3 * X.ravel() + 1
        else:
            # General polynomial
            coeffs = np.random.uniform(-0.5, 0.5, degree + 1)
            y = sum(coeffs[i] * X.ravel()**i for i in range(degree + 1))
        
        y += np.random.normal(0, noise, n_samples)
        return X, y
    
    def generate_clusters(self, n_samples: int = 300, n_clusters: int = 3, 
                         n_features: int = 2, noise: float = 0.1) -> Tuple[np.ndarray, np.ndarray]:
        """Generate clustered classification data"""
        X, y = make_blobs(
            n_samples=n_samples, 
            centers=n_clusters, 
            n_features=n_features,
            cluster_std=1.0 + noise,
            random_state=42
        )
        return X, y
    
    def generate_moons(self, n_samples: int = 300, noise: float = 0.1) -> Tuple[np.ndarray, np.ndarray]:
        """Generate half-moon shaped classification data"""
        X, y = make_moons(n_samples=n_samples, noise=noise, random_state=42)
        return X, y
    
    def generate_circles(self, n_samples: int = 300, noise: float = 0.1, 
                        factor: float = 0.5) -> Tuple[np.ndarray, np.ndarray]:
        """Generate concentric circles classification data"""
        X, y = make_circles(n_samples=n_samples, noise=noise, factor=factor, random_state=42)
        return X, y
    
    def generate_spiral(self, n_samples: int = 300, noise: float = 0.1) -> Tuple[np.ndarray, np.ndarray]:
        """Generate spiral pattern classification data"""
        np.random.seed(42)
        n_per_class = n_samples // 2
        
        # Generate two spirals
        theta1 = np.linspace(0, 4*np.pi, n_per_class)
        theta2 = np.linspace(0, 4*np.pi, n_per_class) + np.pi
        
        r1 = theta1 / (2*np.pi)
        r2 = theta2 / (2*np.pi)
        
        x1 = r1 * np.cos(theta1) + np.random.normal(0, noise, n_per_class)
        y1 = r1 * np.sin(theta1) + np.random.normal(0, noise, n_per_class)
        
        x2 = r2 * np.cos(theta2) + np.random.normal(0, noise, n_per_class)
        y2 = r2 * np.sin(theta2) + np.random.normal(0, noise, n_per_class)
        
        X = np.vstack([np.column_stack([x1, y1]), np.column_stack([x2, y2])])
        y = np.hstack([np.zeros(n_per_class), np.ones(n_per_class)])
        
        return X, y
    
    def generate_wave(self, n_samples: int = 300, noise: float = 0.1) -> Tuple[np.ndarray, np.ndarray]:
        """Generate wave pattern classification data"""
        np.random.seed(42)
        X = np.random.uniform(-3, 3, (n_samples, 2))
        
        # Create wave boundary: y = sin(2*x1) + cos(x1)
        boundary = np.sin(2 * X[:, 0]) + 0.5 * np.cos(X[:, 0])
        y = (X[:, 1] > boundary).astype(int)
        
        # Add noise
        X += np.random.normal(0, noise, X.shape)
        
        return X, y
    
    def generate_checkerboard(self, n_samples: int = 400, noise: float = 0.05) -> Tuple[np.ndarray, np.ndarray]:
        """Generate checkerboard pattern classification data"""
        np.random.seed(42)
        X = np.random.uniform(-3, 3, (n_samples, 2))
        
        # Create checkerboard pattern
        x_grid = np.floor(X[:, 0] + 3).astype(int) % 2
        y_grid = np.floor(X[:, 1] + 3).astype(int) % 2
        y = (x_grid + y_grid) % 2
        
        # Add noise
        X += np.random.normal(0, noise, X.shape)
        
        return X, y
    
    def generate_swiss_roll(self, n_samples: int = 1000, noise: float = 0.1) -> Tuple[np.ndarray, np.ndarray]:
        """Generate Swiss roll manifold data"""
        X, color = make_swiss_roll(n_samples=n_samples, noise=noise, random_state=42)
        # Use only first 2 dimensions for visualization
        X_2d = X[:, [0, 2]]
        return X_2d, color
    
    def generate_dataset(self, data_type: str, **kwargs) -> Tuple[np.ndarray, np.ndarray]:
        """Generate dataset of specified type"""
        generators = {
            'linear': self.generate_linear,
            'polynomial': self.generate_polynomial,
            'clusters': self.generate_clusters,
            'moons': self.generate_moons,
            'circles': self.generate_circles,
            'spiral': self.generate_spiral,
            'wave': self.generate_wave,
            'checkerboard': self.generate_checkerboard,
            'swiss_roll': self.generate_swiss_roll
        }
        
        if data_type not in generators:
            raise ValueError(f"Unknown data type: {data_type}. Available: {list(generators.keys())}")
        
        return generators[data_type](**kwargs)
    
    def visualize_dataset(self, X: np.ndarray, y: np.ndarray, title: str = "Generated Dataset", show: bool = True):
        """Visualize generated dataset"""
        plt.figure(figsize=(10, 6))
        
        if X.shape[1] == 1:
            # 1D data - regression plot
            plt.subplot(1, 2, 1)
            plt.scatter(X.ravel(), y, alpha=0.6, c='blue')
            plt.xlabel('X')
            plt.ylabel('y')
            plt.title(f'{title} - Scatter Plot')
            plt.grid(True, alpha=0.3)
            
            # Distribution plot
            plt.subplot(1, 2, 2)
            plt.hist(y, bins=30, alpha=0.7, color='green')
            plt.xlabel('y values')
            plt.ylabel('Frequency')
            plt.title(f'{title} - Target Distribution')
            plt.grid(True, alpha=0.3)
        
        else:
            # 2D data - classification plot
            plt.subplot(1, 2, 1)
            if len(np.unique(y)) <= 10:  # Classification
                scatter = plt.scatter(X[:, 0], X[:, 1], c=y, cmap='viridis', alpha=0.6)
                plt.colorbar(scatter)
            else:  # Regression with continuous target
                scatter = plt.scatter(X[:, 0], X[:, 1], c=y, cmap='viridis', alpha=0.6)
                plt.colorbar(scatter, label='Target Value')
            
            plt.xlabel('Feature 1')
            plt.ylabel('Feature 2')
            plt.title(f'{title} - Feature Space')
            plt.grid(True, alpha=0.3)
            
            # Class distribution
            plt.subplot(1, 2, 2)
            if len(np.unique(y)) <= 10:
                unique, counts = np.unique(y, return_counts=True)
                plt.bar(unique, counts, alpha=0.7, color='orange')
                plt.xlabel('Class')
                plt.ylabel('Count')
                plt.title(f'{title} - Class Distribution')
            else:
                plt.hist(y, bins=30, alpha=0.7, color='orange')
                plt.xlabel('Target Value')
                plt.ylabel('Frequency')
                plt.title(f'{title} - Target Distribution')
            plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        if show:
            plt.show()
    
    def save_dataset(self, X: np.ndarray, y: np.ndarray, filename: str):
        """Save dataset to CSV file"""
        if X.shape[1] == 1:
            df = pd.DataFrame({
                'feature': X.ravel(),
                'target': y
            })
        else:
            feature_cols = {f'feature_{i+1}': X[:, i] for i in range(X.shape[1])}
            df = pd.DataFrame({**feature_cols, 'target': y})
        
        df.to_csv(filename, index=False)
        print(f"✅ Dataset saved to {filename}")
        print(f"   Shape: {X.shape[0]} samples, {X.shape[1]} features")
        print(f"   Target type: {'Classification' if len(np.unique(y)) <= 10 else 'Regression'}")

def interactive_mode():
    """Interactive data generation mode"""
    generator = DataGenerator()
    
    print("🎲 DATA GENERATOR LAB")
    print("=" * 40)
    print(f"Available dataset types: {', '.join(generator.available_types)}")
    
    while True:
        print("\n🎯 What would you like to do?")
        print("1. 📊 Generate Dataset")
        print("2. 🎨 Visualize Generated Data")
        print("3. 💾 Save Dataset")
        print("4. 🔍 Show Dataset Info")
        print("5. ❌ Exit")
        
        choice = input("\nEnter your choice (1-5): ").strip()
        
        if choice == '1':
            generate_interactive_dataset(generator)
        elif choice == '2':
            visualize_current_dataset(generator)
        elif choice == '3':
            save_current_dataset(generator)
        elif choice == '4':
            show_dataset_info(generator)
        elif choice == '5':
            print("👋 Thanks for using Data Generator Lab!")
            break
        else:
            print("❌ Invalid choice. Please try again.")

def generate_interactive_dataset(generator: DataGenerator):
    """Generate dataset interactively"""
    print("\n📊 Dataset Generation")
    print("-" * 30)
    
    data_type = input(f"Enter dataset type ({'/'.join(generator.available_types)}): ").strip().lower()
    
    if data_type not in generator.available_types:
        print(f"❌ Invalid type. Available: {generator.available_types}")
        return
    
    # Get common parameters
    try:
        n_samples = int(input("Number of samples (default 300): ") or "300")
        noise = float(input("Noise level 0.0-1.0 (default 0.1): ") or "0.1")
    except ValueError:
        print("❌ Invalid input. Using defaults.")
        n_samples, noise = 300, 0.1
    
    # Type-specific parameters
    kwargs = {'n_samples': n_samples, 'noise': noise}
    
    if data_type == 'clusters':
        n_clusters = int(input("Number of clusters (default 3): ") or "3")
        kwargs['n_clusters'] = n_clusters
    elif data_type == 'polynomial':
        degree = int(input("Polynomial degree (default 3): ") or "3")
        kwargs['degree'] = degree
    elif data_type == 'circles':
        factor = float(input("Inner circle factor 0.0-1.0 (default 0.5): ") or "0.5")
        kwargs['factor'] = factor
    
    try:
        X, y = generator.generate_dataset(data_type, **kwargs)
        generator.datasets[data_type] = (X, y)
        
        print(f"✅ Generated {data_type} dataset:")
        print(f"   Samples: {X.shape[0]}")
        print(f"   Features: {X.shape[1]}")
        print(f"   Target classes: {len(np.unique(y))}")
        
        # Auto-visualize
        generator.visualize_dataset(X, y, f"{data_type.title()} Dataset")
        
    except Exception as e:
        print(f"❌ Error generating dataset: {e}")

def visualize_current_dataset(generator: DataGenerator):
    """Visualize currently generated datasets"""
    if not generator.datasets:
        print("❌ No datasets generated yet. Generate a dataset first.")
        return
    
    print("\n🎨 Available datasets for visualization:")
    for i, name in enumerate(generator.datasets.keys(), 1):
        print(f"{i}. {name}")
    
    try:
        choice = int(input("Select dataset to visualize: ")) - 1
        dataset_names = list(generator.datasets.keys())
        
        if 0 <= choice < len(dataset_names):
            name = dataset_names[choice]
            X, y = generator.datasets[name]
            generator.visualize_dataset(X, y, f"{name.title()} Dataset")
        else:
            print("❌ Invalid selection")
    except ValueError:
        print("❌ Please enter a valid number")

def save_current_dataset(generator: DataGenerator):
    """Save currently generated dataset"""
    if not generator.datasets:
        print("❌ No datasets generated yet. Generate a dataset first.")
        return
    
    print("\n💾 Available datasets for saving:")
    for i, name in enumerate(generator.datasets.keys(), 1):
        print(f"{i}. {name}")
    
    try:
        choice = int(input("Select dataset to save: ")) - 1
        dataset_names = list(generator.datasets.keys())
        
        if 0 <= choice < len(dataset_names):
            name = dataset_names[choice]
            X, y = generator.datasets[name]
            
            filename = input(f"Enter filename (default: {name}_data.csv): ").strip()
            if not filename:
                filename = f"{name}_data.csv"
            
            if not filename.endswith('.csv'):
                filename += '.csv'
            
            generator.save_dataset(X, y, filename)
        else:
            print("❌ Invalid selection")
    except ValueError:
        print("❌ Please enter a valid number")

def show_dataset_info(generator: DataGenerator):
    """Show information about generated datasets"""
    if not generator.datasets:
        print("❌ No datasets generated yet.")
        return
    
    print("\n🔍 Generated Datasets Summary:")
    print("-" * 40)
    
    for name, (X, y) in generator.datasets.items():
        print(f"\n📊 {name.upper()}")
        print(f"   Samples: {X.shape[0]}")
        print(f"   Features: {X.shape[1]}")
        print(f"   Target classes/range: {len(np.unique(y))} unique values")
        print(f"   Feature ranges: {X.min(axis=0)} to {X.max(axis=0)}")
        if len(np.unique(y)) <= 10:
            print(f"   Class distribution: {dict(zip(*np.unique(y, return_counts=True)))}")

def main():
    """Main function with command line interface"""
    parser = argparse.ArgumentParser(
        description="Generate synthetic datasets for ML algorithm testing"
    )
    parser.add_argument('--type', choices=[
        'linear', 'polynomial', 'clusters', 'moons', 'circles', 
        'blobs', 'spiral', 'wave', 'checkerboard', 'swiss_roll'
    ], help='Type of dataset to generate')
    parser.add_argument('--n_samples', type=int, default=300, help='Number of samples')
    parser.add_argument('--noise', type=float, default=0.1, help='Noise level')
    parser.add_argument('--n_clusters', type=int, default=3, help='Number of clusters (for clusters type)')
    parser.add_argument('--degree', type=int, default=3, help='Polynomial degree (for polynomial type)')
    parser.add_argument('--factor', type=float, default=0.5, help='Inner circle factor (for circles type)')
    parser.add_argument('--save', type=str, help='Save dataset to file')
    parser.add_argument('--visualize', action='store_true', help='Show visualization')
    parser.add_argument('--no_show', action='store_true', help='Run without opening plots (headless)')
    parser.add_argument('--interactive', action='store_true', help='Launch interactive mode')
    parser.add_argument('--show_all', action='store_true', help='Generate and show all dataset types')
    
    args = parser.parse_args()
    
    if args.interactive:
        interactive_mode()
        return
    
    generator = DataGenerator()
    
    if args.show_all:
        # Show all dataset types
        print("🎲 ALL DATASET TYPES SHOWCASE")
        print("=" * 50)
        
        fig, axes = plt.subplots(3, 4, figsize=(16, 12))
        axes = axes.flatten()
        
        for i, data_type in enumerate(generator.available_types[:10]):  # First 10 types
            try:
                kwargs = {'n_samples': 200, 'noise': 0.1}
                if data_type == 'clusters':
                    kwargs['n_clusters'] = 3
                elif data_type == 'polynomial':
                    kwargs['degree'] = 3
                elif data_type == 'circles':
                    kwargs['factor'] = 0.5
                
                X, y = generator.generate_dataset(data_type, **kwargs)
                
                ax = axes[i]
                if X.shape[1] == 1:
                    ax.scatter(X.ravel(), y, alpha=0.6)
                    ax.set_xlabel('X')
                    ax.set_ylabel('y')
                else:
                    scatter = ax.scatter(X[:, 0], X[:, 1], c=y, cmap='viridis', alpha=0.6)
                    ax.set_xlabel('Feature 1')
                    ax.set_ylabel('Feature 2')
                
                ax.set_title(f'{data_type.title()}')
                ax.grid(True, alpha=0.3)
                
            except Exception as e:
                axes[i].text(0.5, 0.5, f'Error:\n{str(e)}', 
                           ha='center', va='center', transform=axes[i].transAxes)
                axes[i].set_title(f'{data_type.title()} (Error)')
        
        # Hide unused subplots
        for i in range(len(generator.available_types), len(axes)):
            axes[i].set_visible(False)
        
        plt.tight_layout()
        plt.suptitle('Dataset Types Showcase', y=1.02, fontsize=16)
        if not args.no_show:
            plt.show()
        return
    
    if not args.type:
        print("❌ Please specify a dataset type or use --interactive mode")
        print(f"Available types: {', '.join(generator.available_types)}")
        return
    
    # Generate specific dataset
    kwargs = {
        'n_samples': args.n_samples,
        'noise': args.noise
    }
    
    if args.type == 'clusters':
        kwargs['n_clusters'] = args.n_clusters
    elif args.type == 'polynomial':
        kwargs['degree'] = args.degree
    elif args.type == 'circles':
        kwargs['factor'] = args.factor
    
    try:
        X, y = generator.generate_dataset(args.type, **kwargs)
        
        print(f"✅ Generated {args.type} dataset:")
        print(f"   Samples: {X.shape[0]}")
        print(f"   Features: {X.shape[1]}")
        print(f"   Target classes: {len(np.unique(y))}")
        
        if args.visualize:
            generator.visualize_dataset(X, y, f"{args.type.title()} Dataset", show=not args.no_show)
        
        if args.save:
            filename = args.save if args.save.endswith('.csv') else f"{args.save}.csv"
            generator.save_dataset(X, y, filename)
        
    except Exception as e:
        print(f"❌ Error: {e}")

if __name__ == "__main__":
    main()
