# WARP.md

This file provides guidance to WARP (warp.dev) when working with code in this repository.

## Repository Overview

This is a comprehensive machine learning algorithms learning repository (`ml-algorithms-learning`) designed for educational purposes. It contains implementations of various ML algorithms from scratch using NumPy, alongside examples using scikit-learn and other popular libraries.

## Project Structure and Architecture

### Core Directory Structure
- `01_supervised_learning/` - Supervised learning algorithms (Linear Regression, Logistic Regression, Decision Trees, Random Forest, SVM, Naive Bayes, KNN, Neural Networks)
- `02_unsupervised_learning/` - Unsupervised learning algorithms (K-Means, Hierarchical Clustering, DBSCAN, PCA, t-SNE, Association Rules)
- `03_reinforcement_learning/` - Reinforcement learning algorithms (Q-Learning, Deep Q-Networks)
- `04_advanced_topics/` - Advanced topics (Ensemble Methods, Deep Learning, Model Evaluation, XGBoost)
- `05_datasets/` - Datasets used for learning
- `06_utils/` - Utility functions and helper code

### Algorithm Directory Pattern
Each algorithm follows a consistent structure:
- `README.md` - Theory, concepts, mathematical foundations, when to use, pros/cons
- `implementation.py` - From-scratch implementation using NumPy
- `sklearn_example.py` - Examples using scikit-learn
- `exercise.ipynb` - Practical exercises with real datasets

## Development Commands

### Environment Setup
```powershell
# Install dependencies using UV (recommended)
uv sync

# Activate virtual environment and run Python
uv run python

# Start Jupyter Lab for interactive development
uv run jupyter lab
```

### Common Development Tasks
```powershell
# Run a specific Python implementation
uv run python 01_supervised_learning/01_linear_regression/implementation.py

# Run sklearn examples
uv run python 01_supervised_learning/01_linear_regression/sklearn_example.py

# Open Jupyter notebooks for exercises
uv run jupyter lab 01_supervised_learning/01_linear_regression/exercise.ipynb
```

## Key Dependencies and Tools

### Core Libraries (from pyproject.toml)
- **NumPy** (>=1.24.0) - Core numerical computing for from-scratch implementations
- **Pandas** (>=1.5.0) - Data manipulation and analysis
- **Matplotlib** (>=3.6.0), **Seaborn** (>=0.11.0), **Plotly** (>=5.10.0) - Visualization
- **Scikit-learn** (>=1.2.0) - Reference implementations and utilities
- **XGBoost** (>=1.7.0) - Gradient boosting framework
- **Jupyter** (>=1.0.0) - Interactive development environment
- **tqdm**, **joblib**, **requests** - Utilities

### Python Version
- Requires Python >=3.11

## Architecture Insights

### Implementation Philosophy
1. **Dual Implementation Approach**: Each algorithm has both from-scratch (NumPy) and library-based (scikit-learn) implementations to understand the underlying mathematics
2. **Educational Focus**: Code is extensively documented with mathematical explanations, assumptions, and use cases
3. **Practical Learning**: Each algorithm includes exercises with real datasets

### Code Organization Patterns
- **From-scratch implementations** use object-oriented design with fit/predict patterns matching scikit-learn API
- **Mathematical foundations** are explained in docstrings and README files
- **Comparison utilities** (e.g., `compare_methods()` in Linear Regression) demonstrate different optimization approaches
- **Visualization functions** are integrated for understanding algorithm behavior

### Key Implementation Details
- **Linear Regression**: Implements both Normal Equation and Gradient Descent optimization methods
- **XGBoost**: Comprehensive coverage including hyperparameter tuning, cross-validation, and feature importance analysis
- **Consistent API**: All from-scratch implementations follow scikit-learn patterns with `.fit()`, `.predict()`, and `.score()` methods

## Learning Path Structure

### Beginner Track
Start with: Linear Regression → Logistic Regression → Decision Trees → K-Means

### Intermediate Track  
Continue with: Random Forest → SVM → Naive Bayes → PCA

### Advanced Track
Progress to: Neural Networks → Ensemble Methods → XGBoost → Deep Learning → Reinforcement Learning

## Development Notes

### Working with Algorithms
- **Start with README.md** for theoretical understanding and scenario overview
- **Run scenarios** using the provided commands (e.g., `uv run python sklearn_example.py --scenario house_prices`)
- **Compare implementations**: from-scratch vs production (sklearn)
- **Use exercise.ipynb** for interactive hands-on practice
- **Follow navigation links** between related algorithms

### Navigation Patterns
- **Main README.md** has quick navigation table for different skill levels
- **Algorithm READMEs** include "Quick Access" tables with run commands
- **Cross-references** link related algorithms and prerequisites
- **Learning progression** sections guide the next steps

### Running Scenarios
Each algorithm includes multiple real-world scenarios:

**Linear Regression scenarios:**
```powershell
uv run python sklearn_example.py --scenario house_prices
uv run python sklearn_example.py --scenario temperature
uv run python sklearn_example.py --scenario salary
uv run python implementation.py --scenario stocks
```

**XGBoost scenarios:**
```powershell
uv run python sklearn_example.py --scenario kaggle
uv run python sklearn_example.py --scenario credit_risk
uv run python sklearn_example.py --scenario sales_forecast
uv run python hyperparameter_tuning.py --method optuna --budget production
```

### Adding New Algorithms
- Follow the established directory structure pattern
- Include comprehensive README with theory, math, pros/cons, and use cases
- Implement both from-scratch and library versions
- Create practical exercises with real datasets
- Document assumptions and mathematical foundations thoroughly

### Windows/PowerShell Specific
- Use `uv` package manager for dependency management
- PowerShell commands are preferred for Windows environment
- Virtual environment located in `.venv/` directory