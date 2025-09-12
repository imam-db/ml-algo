# 🎮 Algorithm Playground

**🏠 [Back to Main](../README.md)** | **🎯 [Progress Tracker](../PROGRESS_TRACKER.md)** | **🔬 [Algorithm Comparison](../ALGORITHM_COMPARISON.md)**

Interactive tools for experimenting with machine learning algorithms, visualizing their behavior, and understanding how they work under the hood.

---

## 🎨 Interactive Tools

### **🎛️ Parameter Playground**
**[`parameter_playground.py`](./parameter_playground.py)**

Visualize how hyperparameters affect algorithm behavior in real-time.

```bash
# Interactive parameter tuning with live visualization
uv run python parameter_playground.py

# Try different algorithms
uv run python parameter_playground.py --algorithm linear_regression
uv run python parameter_playground.py --algorithm xgboost
```

**Features:**
- Real-time parameter adjustment with sliders
- Live visualization of model changes
- Performance metrics updates
- Side-by-side comparison mode

---

### **🎲 Data Generator Lab**
**[`data_generator.py`](./data_generator.py)**

Create synthetic datasets for algorithm testing and experimentation.

```bash
# Generate different dataset types
uv run python data_generator.py --type linear
uv run python data_generator.py --type clusters --n_clusters 4
uv run python data_generator.py --type moons --noise 0.2
```

**Dataset Types:**
- **Linear**: Simple linear relationships
- **Polynomial**: Non-linear polynomial curves
- **Clusters**: Gaussian clusters for classification
- **Moons**: Half-moon shaped data
- **Circles**: Concentric circles
- **Blobs**: Random blob clusters
- **Spiral**: Spiral patterns
- **Custom**: Define your own patterns

---

### **🎯 Model Visualizer**
**[`model_visualizer.py`](./model_visualizer.py)**

See decision boundaries and classification regions for different algorithms.

```bash
# Visualize decision boundaries
uv run python model_visualizer.py --algorithm svm
uv run python model_visualizer.py --algorithm random_forest --data_type moons
```

**Visualizations:**
- Decision boundaries for classification
- Regression lines and confidence intervals
- Feature importance plots
- Confusion matrices
- ROC curves and precision-recall curves

---

### **🏁 Algorithm Racing**
**[`algorithm_race.py`](./algorithm_race.py)**

Compare multiple algorithms side-by-side on the same dataset with comprehensive performance analysis.

```bash
# Quick test with generated data
uv run python algorithm_race.py --quick_test

# Race with your own dataset
uv run python algorithm_race.py --data_path your_dataset.csv

# Race specific algorithms
uv run python algorithm_race.py --data_path data.csv --algorithms "random_forest,svm,logistic_regression"

# Include statistical significance testing
uv run python algorithm_race.py --data_path data.csv --stats_test
```

**Features:**
- **Automatic Problem Detection**: Classification vs regression
- **Comprehensive Metrics**: Accuracy, precision, recall, F1, R², RMSE, etc.
- **Performance Visualizations**: 4-panel comparison charts
- **Statistical Testing**: Pairwise t-tests between algorithms
- **Speed Analysis**: Training and prediction time benchmarks
- **Cross-validation**: Robust performance evaluation
- **Winner Declaration**: Best algorithm with detailed analysis

**Supported Algorithms:**
- Classification: Logistic Regression, Random Forest, SVM, Naive Bayes, KNN, Decision Tree, XGBoost
- Regression: Linear Regression, Random Forest, SVM, KNN, Decision Tree, XGBoost

---

### **📈 Learning Curve Explorer**
**[`learning_curves.py`](./learning_curves.py)**

Watch how algorithms learn over time and iterations.

```bash
# Visualize learning process
uv run python learning_curves.py --algorithm gradient_descent
uv run python learning_curves.py --algorithm xgboost --show_validation
```

**Visualizations:**
- Training vs validation curves
- Cost function optimization paths
- Convergence analysis
- Overfitting detection
- Early stopping visualization

---

### **⚗️ Feature Engineering Lab**
**[`feature_lab.py`](./feature_lab.py)**

Experiment with different feature transformations and see their effects.

```bash
# Interactive feature engineering
uv run python feature_lab.py
uv run python feature_lab.py --transformations polynomial,scaling,selection
```

**Transformations:**
- Polynomial features
- Feature scaling (StandardScaler, MinMaxScaler)
- Feature selection (SelectKBest, RFE)
- PCA dimensionality reduction
- Custom transformations
- Interaction features

---

### **🎬 Algorithm Animations**
**[`algorithm_animator.py`](./algorithm_animator.py)**

Step-by-step visual animations of how algorithms work internally.

```bash
# Animate algorithm steps
uv run python algorithm_animator.py --algorithm kmeans
uv run python algorithm_animator.py --algorithm gradient_descent --steps 50
```

**Available Animations:**
- K-Means clustering iterations
- Gradient descent optimization
- Decision tree splitting
- Linear regression fitting
- Neural network training

---

## 🚀 Quick Start Guide

### **1. 🎯 First-Time Setup**
```bash
# Install additional visualization dependencies
uv add matplotlib seaborn plotly ipywidgets

# Test the playground
uv run python parameter_playground.py --test
```

### **2. 🎮 Interactive Mode**
```bash
# Launch interactive playground
uv run python 06_playground/playground_launcher.py
```

### **3. 🔬 Experiment with Algorithms**
```bash
# Quick algorithm comparison
uv run python algorithm_race.py --quick_test

# Generate test data
uv run python data_generator.py --interactive

# Visualize any algorithm
uv run python model_visualizer.py --interactive
```

---

## 🎓 Educational Use Cases

### **👶 For Beginners**
- **Understanding Basics**: Use `parameter_playground.py` to see how changing learning rate affects linear regression
- **Data Exploration**: Generate different datasets with `data_generator.py` to understand data patterns
- **Visual Learning**: Watch algorithms work with `algorithm_animator.py`

### **📚 For Intermediate Learners**
- **Parameter Tuning**: Master hyperparameter effects with interactive sliders
- **Algorithm Comparison**: Race algorithms to understand trade-offs
- **Feature Engineering**: Experiment with transformations in `feature_lab.py`

### **🚀 For Advanced Users**
- **Algorithm Analysis**: Deep dive into learning curves and convergence
- **Custom Experiments**: Modify playground tools for specific research
- **Teaching Tools**: Use visualizations for presentations and tutorials

---

## 🛠️ Playground Architecture

### **Core Components**
```
06_playground/
├── README.md                    # This file
├── playground_launcher.py       # Main interactive launcher
├── parameter_playground.py      # Parameter visualization tool
├── data_generator.py           # Synthetic data creation
├── model_visualizer.py         # Decision boundary plots  
├── algorithm_race.py           # Multi-algorithm comparison
├── learning_curves.py          # Learning curve analysis
├── feature_lab.py              # Feature engineering experiments
├── algorithm_animator.py       # Step-by-step animations
├── utils/                      # Shared utilities
│   ├── __init__.py
│   ├── visualization.py       # Common plotting functions
│   ├── data_utils.py          # Data generation utilities
│   └── algorithm_utils.py     # Algorithm wrapper utilities
└── examples/                   # Example notebooks
    ├── playground_tour.ipynb   # Interactive tour
    ├── parameter_tuning.ipynb  # Parameter exploration
    └── visual_learning.ipynb   # Visual algorithm guide
```

### **Integration Points**
- **Algorithm Implementations**: Uses algorithms from `01_supervised_learning/`, `02_unsupervised_learning/`
- **Datasets**: Can load data from `05_datasets/`
- **Progress Tracking**: Integration with `progress_tracker.py`
- **Terminology**: Links to `GLOSSARY.md` for explanations

---

## 🎯 Learning Objectives

After using the Algorithm Playground, you will:

### **🎓 Conceptual Understanding**
- [ ] **Visualize** how different algorithms create decision boundaries
- [ ] **Understand** the effect of hyperparameters on model behavior
- [ ] **Compare** algorithm performance across different data types
- [ ] **Recognize** overfitting and underfitting patterns

### **🔬 Practical Skills**  
- [ ] **Generate** synthetic datasets for algorithm testing
- [ ] **Experiment** with feature engineering transformations
- [ ] **Analyze** learning curves and convergence patterns
- [ ] **Optimize** hyperparameters through visual feedback

### **🧠 Analytical Thinking**
- [ ] **Predict** which algorithms work best for specific data patterns
- [ ] **Diagnose** model problems through visualization
- [ ] **Design** experiments to test algorithmic hypotheses
- [ ] **Communicate** algorithm behavior through visual evidence

---

## 🎮 Interactive Examples

### **Example 1: Understanding Overfitting**
```bash
# Generate polynomial data
uv run python data_generator.py --type polynomial --noise 0.1 --save overfitting_data.csv

# Compare different model complexities
uv run python parameter_playground.py --data overfitting_data.csv --algorithm polynomial_regression

# Adjust polynomial degree and watch validation performance
# Degree 1: Underfitting (high bias)
# Degree 3: Good fit  
# Degree 15: Overfitting (high variance)
```

### **Example 2: Algorithm Selection**
```bash
# Create classification dataset  
uv run python data_generator.py --type moons --noise 0.3 --save moons_data.csv

# Race different classifiers
uv run python algorithm_race.py --data moons_data.csv --algorithms logistic_regression,svm,random_forest

# Visualize decision boundaries
uv run python model_visualizer.py --data moons_data.csv --algorithm svm --kernel rbf
```

### **Example 3: Feature Engineering Impact**
```bash
# Generate non-linear data
uv run python data_generator.py --type circles --save circles_data.csv

# Test linear model (will fail)
uv run python model_visualizer.py --data circles_data.csv --algorithm logistic_regression

# Apply polynomial features and see improvement  
uv run python feature_lab.py --data circles_data.csv --transform polynomial --degree 2
```

---

## 💡 Pro Tips

### **🎯 Effective Experimentation**
1. **Start Simple**: Begin with linear data and simple algorithms
2. **One Variable**: Change one parameter at a time to understand effects
3. **Visual First**: Always visualize before diving into metrics
4. **Compare**: Use racing tool to understand trade-offs

### **🔬 Scientific Approach**
1. **Hypothesis**: Form predictions before running experiments
2. **Control**: Keep other variables constant when testing
3. **Repeat**: Run experiments multiple times with different random seeds
4. **Document**: Save interesting findings and parameter combinations

### **🎨 Visualization Best Practices**
1. **Appropriate Scale**: Use log scale for wide-ranging metrics
2. **Color Coding**: Consistent colors for algorithms across plots
3. **Clear Labels**: Always label axes and include legends
4. **Interactive**: Use sliders and widgets for real-time exploration

---

## 🔗 Related Resources

### **Learning Materials**
- **[Algorithm Comparison](../ALGORITHM_COMPARISON.md)** - Choose the right algorithm
- **[Glossary](../GLOSSARY.md)** - Understand terminology
- **[Progress Tracker](../PROGRESS_TRACKER.md)** - Track your learning

### **Implementation References**
- **[Linear Regression](../01_supervised_learning/01_linear_regression/)** - Theory and implementation
- **[XGBoost](../04_advanced_topics/04_xgboost/)** - Advanced ensemble methods
- **[Model Evaluation](../04_advanced_topics/03_model_evaluation/)** - Performance metrics

### **Development Tools**
- **[WARP.md](../WARP.md)** - Development workflow
- **Jupyter Notebooks** - Interactive development environment

---

**🎮 Ready to Play?** Start with the playground launcher and explore the interactive world of machine learning!

```bash
uv run python 06_playground/playground_launcher.py
```

**🏠 [Back to Main README](../README.md)** | **🎯 [Progress Tracker](../PROGRESS_TRACKER.md)** | **🚀 [Start Learning](../README.md#-quick-navigation)**