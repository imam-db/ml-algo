#!/usr/bin/env python3
"""
Algorithm Racing Tool Demo
==========================

Demonstrates various usage patterns and capabilities of the algorithm racing tool.

Usage: uv run python demo_algorithm_race.py
"""

import numpy as np
from sklearn.datasets import make_classification, make_regression, load_iris, load_wine
from algorithm_race import AlgorithmRacer
import pandas as pd

def demo_classification_algorithms():
    """Demo classification algorithm racing"""
    print("🎯 CLASSIFICATION ALGORITHM DEMO")
    print("=" * 50)
    
    # Generate multi-class classification data
    X, y = make_classification(
        n_samples=800, 
        n_features=15, 
        n_informative=10,
        n_redundant=3,
        n_classes=4, 
        n_clusters_per_class=1,
        random_state=42
    )
    
    print(f"📊 Generated dataset: {X.shape[0]} samples, {X.shape[1]} features, {len(np.unique(y))} classes")
    
    # Initialize racer
    racer = AlgorithmRacer()
    racer.load_data(X=X, y=y)
    
    # Race specific algorithms
    algorithms_to_test = ['logistic_regression', 'random_forest', 'svm', 'xgboost']
    
    print(f"\n🏁 Racing {len(algorithms_to_test)} algorithms...")
    results = racer.race_algorithms(X, y, algorithms=algorithms_to_test)
    
    # Print comprehensive results
    racer.print_race_summary()
    
    # Statistical significance testing
    print("\n🔬 Statistical Testing:")
    racer.X, racer.y = X, y
    racer.statistical_significance_test()
    
    return racer, results

def demo_regression_algorithms():
    """Demo regression algorithm racing"""
    print("\n\n🎯 REGRESSION ALGORITHM DEMO")
    print("=" * 50)
    
    # Generate regression data with some noise
    X, y = make_regression(
        n_samples=600,
        n_features=8,
        noise=10,
        random_state=42
    )
    
    print(f"📊 Generated dataset: {X.shape[0]} samples, {X.shape[1]} features")
    
    # Initialize racer
    racer = AlgorithmRacer()
    racer.load_data(X=X, y=y)
    
    # Race all regression algorithms
    results = racer.race_algorithms(X, y)
    
    # Print results
    racer.print_race_summary()
    
    return racer, results

def demo_real_world_dataset():
    """Demo with real-world dataset"""
    print("\n\n🎯 REAL-WORLD DATASET DEMO (Wine Classification)")
    print("=" * 60)
    
    # Load wine dataset
    wine = load_wine()
    X, y = wine.data, wine.target
    
    print(f"📊 Wine dataset: {X.shape[0]} samples, {X.shape[1]} features, {len(np.unique(y))} classes")
    print(f"🍷 Features: {', '.join(wine.feature_names[:3])}...")
    
    # Initialize racer
    racer = AlgorithmRacer()
    racer.load_data(X=X, y=y)
    racer.feature_names = wine.feature_names
    
    # Race algorithms with cross-validation
    results = racer.race_algorithms(X, y, cv_folds=10)
    
    # Print detailed results
    racer.print_race_summary()
    
    return racer, results

def demo_speed_vs_accuracy_analysis():
    """Demo speed vs accuracy trade-off analysis"""
    print("\n\n⚡ SPEED VS ACCURACY ANALYSIS")
    print("=" * 40)
    
    # Create larger dataset to see timing differences
    X, y = make_classification(
        n_samples=5000, 
        n_features=50,
        n_informative=30,
        n_classes=2,
        random_state=42
    )
    
    print(f"📊 Large dataset: {X.shape[0]} samples, {X.shape[1]} features")
    
    racer = AlgorithmRacer()
    racer.load_data(X=X, y=y)
    
    # Test algorithms known for different speed/accuracy trade-offs
    speed_test_algos = ['naive_bayes', 'logistic_regression', 'knn', 'svm', 'random_forest']
    
    results = racer.race_algorithms(X, y, algorithms=speed_test_algos)
    
    # Analyze speed vs accuracy trade-offs
    print("\n⚖️  SPEED VS ACCURACY TRADE-OFF ANALYSIS:")
    print("-" * 45)
    
    for algo_name, result in results.items():
        accuracy = result['accuracy']
        train_time = result['training_time']
        pred_time = result['prediction_time']
        
        # Calculate efficiency score (accuracy per second of training)
        efficiency = accuracy / max(train_time, 0.001)  # Avoid division by zero
        
        print(f"{algo_name.replace('_', ' ').title():>18}: "
              f"Acc={accuracy:.3f}, Train={train_time:.3f}s, "
              f"Pred={pred_time:.6f}s, Efficiency={efficiency:.1f}")
    
    return racer, results

def demo_custom_algorithm_selection():
    """Demo custom algorithm selection based on problem characteristics"""
    print("\n\n🎯 CUSTOM ALGORITHM SELECTION DEMO")
    print("=" * 45)
    
    # Create different types of problems
    datasets = {
        'Linear Separable': make_classification(n_samples=500, n_features=2, n_redundant=0, 
                                              n_informative=2, n_clusters_per_class=1, random_state=42),
        'High Dimensional': make_classification(n_samples=300, n_features=100, n_informative=50, 
                                              random_state=42),
        'Small Dataset': make_classification(n_samples=50, n_features=5, random_state=42),
    }
    
    for problem_name, (X, y) in datasets.items():
        print(f"\n📊 Problem: {problem_name}")
        print(f"   Dataset size: {X.shape[0]} samples, {X.shape[1]} features")
        
        racer = AlgorithmRacer()
        racer.load_data(X=X, y=y)
        
        # Select appropriate algorithms based on problem characteristics
        if X.shape[1] > X.shape[0]:  # High dimensional
            selected_algos = ['logistic_regression', 'svm', 'naive_bayes']
        elif X.shape[0] < 100:  # Small dataset
            selected_algos = ['naive_bayes', 'knn', 'decision_tree']
        else:  # General case
            selected_algos = ['logistic_regression', 'random_forest', 'svm']
        
        print(f"   Selected algorithms: {', '.join(selected_algos)}")
        
        # Quick race
        results = racer.race_algorithms(X, y, algorithms=selected_algos, cv_folds=3)
        
        # Show winner
        winner, score = racer.get_winner()
        print(f"   🥇 Winner: {winner.replace('_', ' ').title()} (Score: {score:.3f})")

def main():
    """Run all demos"""
    print("🏁 ALGORITHM RACING TOOL - COMPREHENSIVE DEMO")
    print("=" * 55)
    print()
    
    # Demo 1: Classification algorithms
    demo_classification_algorithms()
    
    # Demo 2: Regression algorithms  
    demo_regression_algorithms()
    
    # Demo 3: Real-world dataset
    demo_real_world_dataset()
    
    # Demo 4: Speed vs accuracy analysis
    demo_speed_vs_accuracy_analysis()
    
    # Demo 5: Custom algorithm selection
    demo_custom_algorithm_selection()
    
    print("\n\n🎉 DEMO COMPLETE!")
    print("=" * 20)
    print("✅ Classification racing demonstrated")
    print("✅ Regression racing demonstrated")
    print("✅ Real-world dataset tested")
    print("✅ Speed vs accuracy analysis shown")
    print("✅ Custom algorithm selection illustrated")
    print("\n🚀 Try your own datasets with:")
    print("   uv run python algorithm_race.py --data_path your_data.csv")

if __name__ == "__main__":
    main()