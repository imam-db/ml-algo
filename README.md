# Machine Learning Algorithms Learning Repository

Welcome to the machine learning algorithms learning repository! 🤖📚

## 🗺️ Quick Navigation

| Category | Beginner Friendly | Intermediate | Advanced |
|----------|------------------|--------------|----------|
| **Supervised** | [Linear Regression](./01_supervised_learning/01_linear_regression/) | [Random Forest](./01_supervised_learning/04_random_forest/) | [Neural Networks](./01_supervised_learning/08_neural_networks/) |
| **Unsupervised** | [K-Means](./02_unsupervised_learning/01_kmeans/) | [PCA](./02_unsupervised_learning/04_pca/) | [t-SNE](./02_unsupervised_learning/05_tsne/) |
| **Advanced** | [Model Evaluation](./04_advanced_topics/03_model_evaluation/) | [Ensemble Methods](./04_advanced_topics/01_ensemble_methods/) | [XGBoost](./04_advanced_topics/04_xgboost/) |
| **🎮 Interactive** | [Algorithm Racing](./06_playground/) | [Parameter Playground](./06_playground/) | [Feature Engineering](./06_playground/) |

**🤔 Not sure which algorithm to choose?** ➡️ [**Algorithm Selection Guide**](./ALGORITHM_COMPARISON.md)
**🎆 Want to track your learning progress?** ➡️ [**Progress Tracker**](./PROGRESS_TRACKER.md)  
**🎮 Want to experiment interactively?** ➡️ [**Algorithm Playground**](./06_playground/)
**🧾 Need quick references?** ➡️ [**Cheat Sheets Collection**](./08_cheat_sheets/README.md)

**Essential Files:**
- 📝 [WARP.md](./WARP.md) - Development guidance for Warp terminal
- 📚 [GLOSSARY.md](./GLOSSARY.md) - Complete ML terminology reference
- 🔬 [ALGORITHM_COMPARISON.md](./ALGORITHM_COMPARISON.md) - Algorithm selection guide
- 🎯 [PROGRESS_TRACKER.md](./PROGRESS_TRACKER.md) - Learning progress and achievements
- 📁 [Project Structure](#-repository-structure) - Detailed folder organization
- 🚀 [Quick Start](#-quick-start) - Get running in 3 steps

## 📋 Repository Structure

### 1. **Supervised Learning** ([`01_supervised_learning/`](./01_supervised_learning/))
- **[01_linear_regression](./01_supervised_learning/01_linear_regression/)** - Linear Regression
- **[02_logistic_regression](./01_supervised_learning/02_logistic_regression/)** - Logistic Regression  
- **[03_decision_trees](./01_supervised_learning/03_decision_trees/)** - Decision Trees
- **[04_random_forest](./01_supervised_learning/04_random_forest/)** - Random Forest
- **🚧 05_svm** - Support Vector Machine (planned)
- **🚧 06_naive_bayes** - Naive Bayes (planned)
- **🚧 07_knn** - K-Nearest Neighbors (planned)
- **🚧 08_neural_networks** - Neural Networks (planned)

### 2. **Unsupervised Learning** ([`02_unsupervised_learning/`](./02_unsupervised_learning/))
- **🚧 01_kmeans** - K-Means Clustering (planned)
- **🚧 02_hierarchical_clustering** - Hierarchical Clustering (planned)
- **🚧 03_dbscan** - DBSCAN (planned)
- **🚧 04_pca** - Principal Component Analysis (planned)
- **🚧 05_tsne** - t-SNE (planned)
- **🚧 06_association_rules** - Association Rules (planned)

### 3. **Reinforcement Learning** ([`03_reinforcement_learning/`](./03_reinforcement_learning/))
- **🚧 01_q_learning** - Q-Learning (planned)
- **🚧 02_deep_q_networks** - Deep Q-Networks (planned)

### 4. **Advanced Topics** ([`04_advanced_topics/`](./04_advanced_topics/))
- **🚧 01_ensemble_methods** - Ensemble Methods (planned)
- **🚧 02_deep_learning** - Deep Learning (planned)
- **🚧 03_model_evaluation** - Model Evaluation (planned)
- **[04_xgboost](./04_advanced_topics/04_xgboost/)** - XGBoost (eXtreme Gradient Boosting)

### 5. **Datasets** ([`05_datasets/`](./05_datasets/))
Datasets used for learning

### 6. **🎮 Algorithm Playground** ([`06_playground/`](./06_playground/))
Interactive tools for experimenting with ML algorithms:
- **🏁 Algorithm Racing** - Compare multiple algorithms side-by-side
- **🎛️ Parameter Playground** - Interactive hyperparameter tuning
- **📊 Model Visualizer** - Visualize decision boundaries and model behavior
- **📈 Learning Curves** - Analyze how algorithms learn over time
- **🧪 Feature Engineering Lab** - Experiment with feature transformations
- **🎬 Algorithm Animator** - Step-by-step visual animations
- **🎲 Data Generator** - Create synthetic datasets for testing

### 7. **Utils** ([`06_utils/`](./06_utils/))
Utility functions and helper code

### 8. **Cheat Sheets** ([`08_cheat_sheets/`](./08_cheat_sheets/))
Quick references to accelerate learning and implementation:
- 🐍 [Scikit-Learn Quick Reference](./08_cheat_sheets/02_python_sklearn/sklearn_quick_reference.md)
- 🐼 [Pandas & NumPy Essentials](./08_cheat_sheets/02_python_sklearn/pandas_numpy_essentials.md)
- 🧭 [Algorithm Selection Flowchart](./08_cheat_sheets/01_algorithms/algorithm_selection_flowchart.md)
- 🤖 [Algorithms Quick Reference](./08_cheat_sheets/01_algorithms/algorithms_quick_reference.md)
- 📊 [Classification Metrics](./08_cheat_sheets/04_model_evaluation/classification_metrics.md)
- 🧮 [Essential Math & Statistics](./08_cheat_sheets/03_math_statistics/essential_formulas.md)
- **🚧 Visualization cheat sheets** (planned)

## 🚀 Quick Start

1. **Install Dependencies:**
   ```bash
   uv sync
   ```

2. **Activate Virtual Environment:**
   ```bash
   uv run python
   ```

3. **Run Jupyter Lab:**
   ```bash
   uv run jupyter lab
   ```

4. **🎮 Launch Algorithm Playground:**
   ```bash
   uv run python 06_playground/playground_launcher.py
   ```
   Or try individual tools:
   ```bash
   # Quick algorithm comparison
   uv run python 06_playground/algorithm_race.py --quick_test
   
   # Interactive parameter tuning
   uv run python 06_playground/parameter_playground.py
   
   # Visualize decision boundaries
   uv run python 06_playground/model_visualizer.py --algorithm svm
   ```

### Headless Mode (no GUI windows)
Use these flags in environments without a display or when running in CI:
- Model Visualizer: `--no_show`
- Parameter Playground: `--no_show`
- Learning Curves: `--no_show`
- Algorithm Animator: `--no_show`
- Data Generator: `--no_show`
- XGBoost Tutorial: `--no_show`
- Algorithm Race: `--no_plots`

## 📚 How to Learn

Each algorithm has the same folder structure:
- `README.md` - Theory and concept explanations
- `implementation.py` - Implementation from scratch
- `sklearn_example.py` - Examples using scikit-learn
- `exercise.ipynb` - Practical exercises
- `datasets/` - Specific datasets for the algorithm

## 🎯 Learning Path

### Beginner
1. Linear Regression
2. Logistic Regression
3. Decision Trees
4. K-Means Clustering

### Intermediate
1. Random Forest
2. SVM
3. Naive Bayes
4. PCA

### Advanced
1. Neural Networks
2. Ensemble Methods
3. XGBoost (Gradient Boosting)
4. Deep Learning
5. Reinforcement Learning

## 📖 Resources

- [Scikit-learn Documentation](https://scikit-learn.org/)
- [Machine Learning Course - Andrew Ng](https://www.coursera.org/learn/machine-learning)
- [Hands-On Machine Learning](https://github.com/ageron/handson-ml2)

## 🤝 Contributing

Feel free to contribute by adding new algorithms or improving existing implementations!

---
Happy Learning! 🎉
