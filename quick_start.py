#!/usr/bin/env python3
"""
Quick Start Script for ML Algorithms Learning Repository
=======================================================

Use this script to test your dependency installation and environment setup.
Run it to ensure everything is working correctly.
"""

import sys
import importlib
from typing import List, Tuple

def check_python_version() -> bool:
    """Check if Python version is compatible"""
    required_version = (3, 11)
    current_version = sys.version_info[:2]
    
    print(f"🐍 Python version: {sys.version}")
    
    if current_version >= required_version:
        print(f"✅ Python version OK (>= {required_version[0]}.{required_version[1]})")
        return True
    else:
        print(f"❌ Python version too old. Required >= {required_version[0]}.{required_version[1]}")
        return False

def check_dependencies() -> List[Tuple[str, bool, str]]:
    """Check if all required dependencies are installed"""
    dependencies = [
        "numpy",
        "pandas", 
        "matplotlib",
        "seaborn",
        "sklearn",
        "jupyter",
        "tqdm",
        "joblib",
        "requests"
    ]
    
    results = []
    
    print("\n📦 Checking dependencies...")
    
    for dep in dependencies:
        try:
            module = importlib.import_module(dep)
            version = getattr(module, '__version__', 'Unknown')
            print(f"✅ {dep:15} v{version}")
            results.append((dep, True, version))
        except ImportError:
            print(f"❌ {dep:15} NOT FOUND")
            results.append((dep, False, "Not installed"))
    
    return results

def test_basic_functionality():
    """Test basic ML functionality"""
    print("\n🧪 Testing basic functionality...")
    
    try:
        # Test NumPy
        import numpy as np
        arr = np.array([1, 2, 3, 4, 5])
        assert arr.mean() == 3.0
        print("✅ NumPy basic operations OK")
        
        # Test Pandas
        import pandas as pd
        df = pd.DataFrame({'a': [1, 2, 3], 'b': [4, 5, 6]})
        assert len(df) == 3
        print("✅ Pandas basic operations OK")
        
        # Test Matplotlib
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots()
        plt.close(fig)
        print("✅ Matplotlib plotting OK")
        
        # Test Scikit-learn
        from sklearn.linear_model import LinearRegression
        from sklearn.datasets import make_regression
        
        X, y = make_regression(n_samples=100, n_features=1, noise=0.1, random_state=42)
        model = LinearRegression()
        model.fit(X, y)
        score = model.score(X, y)
        assert score > 0.95
        print("✅ Scikit-learn LinearRegression OK")
        
        print("\n🎉 All basic functionality tests passed!")
        return True
        
    except Exception as e:
        print(f"❌ Basic functionality test failed: {str(e)}")
        return False

def run_linear_regression_demo():
    """Run a quick Linear Regression demo"""
    print("\n📊 Running Linear Regression demo...")
    
    try:
        import numpy as np
        import matplotlib.pyplot as plt
        from sklearn.linear_model import LinearRegression
        from sklearn.metrics import r2_score
        
        # Generate sample data
        np.random.seed(42)
        X = np.linspace(0, 10, 50).reshape(-1, 1)
        y = 2 * X.ravel() + 3 + np.random.normal(0, 1, 50)
        
        # Train model
        model = LinearRegression()
        model.fit(X, y)
        
        # Make predictions
        y_pred = model.predict(X)
        r2 = r2_score(y, y_pred)
        
        print(f"✅ Model trained successfully!")
        print(f"   Slope: {model.coef_[0]:.4f}")
        print(f"   Intercept: {model.intercept_:.4f}")
        print(f"   R² Score: {r2:.4f}")
        
        # Create simple plot (don't show, just test)
        plt.figure(figsize=(8, 6))
        plt.scatter(X, y, alpha=0.7, label='Data points')
        plt.plot(X, y_pred, 'r-', linewidth=2, label='Regression line')
        plt.xlabel('X')
        plt.ylabel('y')
        plt.title('Linear Regression Demo')
        plt.legend()
        plt.savefig('linear_regression_demo.png', dpi=100, bbox_inches='tight')
        plt.close()
        
        print("📊 Plot saved as 'linear_regression_demo.png'")
        return True
        
    except Exception as e:
        print(f"❌ Linear Regression demo failed: {str(e)}")
        return False

def print_next_steps():
    """Print next steps for learning"""
    print("\n" + "="*60)
    print("🚀 NEXT STEPS")
    print("="*60)
    print("""
Getting Started:
1. 📚 Read the main README.md for an overview
2. 📂 Navigate to 01_supervised_learning/01_linear_regression/
3. 📖 Start with README.md for theory
4. 💻 Run implementation.py to see the from-scratch implementation
5. 📊 Try sklearn_example.py for practical examples  
6. 🧪 Open exercise.ipynb in Jupyter for hands-on practice

Commands to run:
  uv run python 01_supervised_learning/01_linear_regression/implementation.py
  uv run python 01_supervised_learning/01_linear_regression/sklearn_example.py
  uv run jupyter lab exercise.ipynb

Learning Path:
📈 Beginner:    Linear Regression → Logistic Regression → Decision Trees
📊 Intermediate: Random Forest → SVM → Naive Bayes → KNN
🧠 Advanced:    Neural Networks → Ensemble Methods → Deep Learning

Happy Learning! 🎓
    """)

def main():
    """Main function"""
    print("="*60)
    print("🤖 ML ALGORITHMS LEARNING - QUICK START")
    print("="*60)
    
    # Check Python version
    python_ok = check_python_version()
    if not python_ok:
        print("\n❌ Please upgrade Python and try again.")
        return False
    
    # Check dependencies
    dep_results = check_dependencies()
    missing_deps = [dep for dep, ok, _ in dep_results if not ok]
    
    if missing_deps:
        print(f"\n❌ Missing dependencies: {missing_deps}")
        print("Run: uv sync")
        return False
    
    # Test functionality
    func_ok = test_basic_functionality()
    if not func_ok:
        return False
    
    # Run demo
    demo_ok = run_linear_regression_demo()
    if not demo_ok:
        return False
    
    # Print next steps
    print_next_steps()
    
    print("\n✅ Quick start completed successfully!")
    return True

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
