# Machine Learning Algorithms Learning Repository

**🇬🇧 English | [🇮🇩 Bahasa Indonesia](README.id.md)**

[![Python Version](https://img.shields.io/badge/python-3.11%2B-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Tests](https://github.com/imam-db/ml-algo/actions/workflows/tests.yml/badge.svg)](https://github.com/imam-db/ml-algo/actions/workflows/tests.yml)
[![Ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json)](https://github.com/astral-sh/ruff)
[![Contributions Welcome](https://img.shields.io/badge/contributions-welcome-brightgreen.svg?style=flat)](CONTRIBUTING.md)

Welcome to the machine learning algorithms learning repository! 🤖📚

**A comprehensive learning resource for machine learning with from-scratch implementations, complete theory, and interactive exercises.**

## 🗺️ Quick Navigation

| Category | Beginner Friendly | Intermediate | Advanced |
|----------|------------------|--------------|----------|
| **Supervised** | [Linear Regression](./01_supervised_learning/01_linear_regression/) | [Random Forest](./01_supervised_learning/04_random_forest/) | [Neural Networks](./01_supervised_learning/08_neural_networks/) |
| **Unsupervised** | [K-Means](./02_unsupervised_learning/01_kmeans/) | [PCA](./02_unsupervised_learning/04_pca/) | [Hierarchical Clustering](./02_unsupervised_learning/03_hierarchical/) |
| **Advanced** | [Model Evaluation](./04_advanced_topics/03_model_evaluation/) | [Gradient Boosting](./01_supervised_learning/09_gradient_boosting/) | [XGBoost](./04_advanced_topics/04_xgboost/) |
| **🧠 More Models** | [Q-Learning](./03_reinforcement_learning/01_q_learning/) | [SVM](./01_supervised_learning/10_svm/) | [CNN & Computer Vision](./04_advanced_topics/05_cnn_computer_vision/) |
| **📈 Time Series** | [Forecasting](./04_advanced_topics/06_time_series_forecasting/) | [Object Detection](./04_advanced_topics/07_object_detection/) | [XGBoost](./04_advanced_topics/04_xgboost/) |
| **🎮 Interactive** | [Algorithm Racing](./06_playground/) | [Parameter Playground](./06_playground/) | [Feature Engineering](./06_playground/) |

**🤔 Not sure which algorithm to choose?** ➡️ [**Algorithm Selection Guide**](./ALGORITHM_COMPARISON.md)
**🎆 Want to track your learning progress?** ➡️ [**Progress Tracker**](./PROGRESS_TRACKER.md)  
**🎮 Want to experiment interactively?** ➡️ [**Algorithm Playground**](./06_playground/)
**🧾 Need quick references?** ➡️ [**Cheat Sheets Collection**](./08_cheat_sheets/README.md)

**Essential Files:**
- 📚 [GLOSSARY.md](./GLOSSARY.md) - Complete ML terminology reference
- 🔬 [ALGORITHM_COMPARISON.md](./ALGORITHM_COMPARISON.md) - Algorithm selection guide
- 🎯 [PROGRESS_TRACKER.md](./PROGRESS_TRACKER.md) - Learning progress and achievements
- 📁 [Project Structure](#-repository-structure) - Detailed folder organization
- 🚀 [Quick Start](#-quick-start) - Get running in 4 steps

## 🌐 Documentation Site

Prefer reading in the browser? All modules are published as a searchable website with math rendering, dark mode, and an English/Indonesian language toggle:

**➡️ https://imam-db.github.io/ml-algo/**

To preview it locally:

```bash
pip install -r requirements-docs.txt
python scripts/build_docs.py && mkdocs serve
```

## 📋 Repository Structure

### 1. **Supervised Learning** ([`01_supervised_learning/`](./01_supervised_learning/))
- **[01_linear_regression](./01_supervised_learning/01_linear_regression/)** - Linear Regression ✅
- **[02_logistic_regression](./01_supervised_learning/02_logistic_regression/)** - Logistic Regression ✅
- **[03_decision_trees](./01_supervised_learning/03_decision_trees/)** - Decision Trees ✅
- **[04_random_forest](./01_supervised_learning/04_random_forest/)** - Random Forest ✅
- *05_svm — moved to [10_svm](./01_supervised_learning/10_svm/)*
- **[06_naive_bayes](./01_supervised_learning/06_naive_bayes/)** - Naive Bayes (Gaussian & Multinomial) ✅
- **[07_knn](./01_supervised_learning/07_knn/)** - K-Nearest Neighbors (Classifier & Regressor) ✅
- **[08_neural_networks](./01_supervised_learning/08_neural_networks/)** - Neural Networks (Feedforward) ✅
- **[09_gradient_boosting](./01_supervised_learning/09_gradient_boosting/)** - Gradient Boosting (Regression & Classification) ✅
- **[10_svm](./01_supervised_learning/10_svm/)** - Support Vector Machine (Linear Pegasos & Kernel SMO) ✅

### 2. **Unsupervised Learning** ([`02_unsupervised_learning/`](./02_unsupervised_learning/))
- **[01_kmeans](./02_unsupervised_learning/01_kmeans/)** - K-Means Clustering ✅
- **[02_dbscan](./02_unsupervised_learning/02_dbscan/)** - DBSCAN (Density-Based Clustering) ✅
- **[03_hierarchical](./02_unsupervised_learning/03_hierarchical/)** - Hierarchical (Agglomerative) Clustering ✅
- **[04_pca](./02_unsupervised_learning/04_pca/)** - Principal Component Analysis ✅
- **[05_tsne](./02_unsupervised_learning/05_tsne/)** - t-SNE (t-Distributed Stochastic Neighbor Embedding) ✅
- **[06_association_rules](./02_unsupervised_learning/06_association_rules/)** - Association Rules (Apriori & FP-Growth) ✅

### 3. **Reinforcement Learning** ([`03_reinforcement_learning/`](./03_reinforcement_learning/))
- **[01_q_learning](./03_reinforcement_learning/01_q_learning/)** - Q-Learning (Off-Policy TD) ✅
- **[02_sarsa](./03_reinforcement_learning/02_sarsa/)** - SARSA (On-Policy TD) ✅
- **🚧 03_deep_q_networks** - Deep Q-Networks (planned)

### 4. **Advanced Topics** ([`04_advanced_topics/`](./04_advanced_topics/))
- **🚧 01_ensemble_methods** - Ensemble Methods (planned)
- **🚧 02_deep_learning** - Deep Learning (planned)
- **[03_model_evaluation](./04_advanced_topics/03_model_evaluation/)** - Model Evaluation (Metrics & Cross-Validation) ✅
- **[04_xgboost](./04_advanced_topics/04_xgboost/)** - XGBoost (eXtreme Gradient Boosting) ✅
- **[05_cnn_computer_vision](./04_advanced_topics/05_cnn_computer_vision/)** - CNN & Computer Vision (im2col Conv2D, MaxPool, trainable shelf-product CNN, YOLO bridge) ✅
- **[06_time_series_forecasting](./04_advanced_topics/06_time_series_forecasting/)** - Time Series Forecasting (Classical Models, Lag Features & Backtesting) ✅
- **[07_object_detection](./04_advanced_topics/07_object_detection/)** - Object Detection (box formats, vectorized IoU, NMS, mini YOLO-v1 grid detector, mAP@0.5) ✅

### 5. **🚧 Datasets** (`05_datasets/`, planned)
Shared datasets for learning (each module currently generates its own synthetic data)

### 6. **🎮 Algorithm Playground** ([`06_playground/`](./06_playground/))
Interactive tools for experimenting with ML algorithms:
- **🏁 Algorithm Racing** - Compare multiple algorithms side-by-side
- **🎛️ Parameter Playground** - Interactive hyperparameter tuning
- **📊 Model Visualizer** - Visualize decision boundaries and model behavior
- **📈 Learning Curves** - Analyze how algorithms learn over time
- **🧪 Feature Engineering Lab** - Experiment with feature transformations
- **🎬 Algorithm Animator** - Step-by-step visual animations
- **🎲 Data Generator** - Create synthetic datasets for testing

### 7. **🚧 Utils** (`06_utils/`, planned)
Shared utility functions and helper code

### 8. **Cheat Sheets** ([`08_cheat_sheets/`](./08_cheat_sheets/))
Quick references to accelerate learning and implementation:
- 🐍 [Scikit-Learn Quick Reference](./08_cheat_sheets/02_python_sklearn/sklearn_quick_reference.md)
- 🐼 [Pandas & NumPy Essentials](./08_cheat_sheets/02_python_sklearn/pandas_numpy_essentials.md)
- 🧭 [Algorithm Selection Flowchart](./08_cheat_sheets/01_algorithms/algorithm_selection_flowchart.md)
- 🤖 [Algorithms Quick Reference](./08_cheat_sheets/01_algorithms/algorithms_quick_reference.md)
- 📊 [Classification Metrics](./08_cheat_sheets/04_model_evaluation/classification_metrics.md)
- 🧮 [Essential Math & Statistics](./08_cheat_sheets/03_math_statistics/essential_formulas.md)
- 🧹 [Data Preprocessing Pipeline](./08_cheat_sheets/05_data_preprocessing/preprocessing_pipeline.md)
- 🛠️ [Common Errors & Solutions](./08_cheat_sheets/07_troubleshooting/common_errors_solutions.md)
- **🚧 Visualization cheat sheets** (planned)

## 🚀 Quick Start

1. **Install Dependencies:**
   ```bash
   # Core + development dependencies
   uv sync --extra dev
   ```

   For optional notebook dependencies, install every extra:
   ```bash
   uv sync --all-extras
   ```

2. **Activate Virtual Environment:**
   ```bash
   source .venv/bin/activate
   ```
   Or skip activation and run commands with `uv run` (for example, `uv run python`).

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
- `README.md` / `README.id.md` - Theory and concept explanations (English/Indonesian)
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
2. Gradient Boosting
3. XGBoost
4. CNN & Computer Vision
5. Time Series Forecasting
6. Reinforcement Learning (Q-Learning, SARSA)

## 📖 Resources

- [Scikit-learn Documentation](https://scikit-learn.org/)
- [Machine Learning Course - Andrew Ng](https://www.coursera.org/learn/machine-learning)
- [Hands-On Machine Learning](https://github.com/ageron/handson-ml2)

## 📊 Implementation Status

| Category | Implemented | Planned | Progress |
|----------|-------------|---------|----------|
| Supervised Learning | 9 | 0 | 🟢 100% |
| Unsupervised Learning | 6 | 0 | 🟢 100% |
| Reinforcement Learning | 2 | 1 | 🟢 67% |
| Advanced Topics | 4 | 2 | 🟢 67% |
| **Total** | **21** | **3** | **🟢 88%** |

## 🧪 Testing

This repository includes a comprehensive test suite:

```bash
# Run all tests
uv run pytest

# Run specific test file
uv run pytest tests/test_reinforcement_learning.py -v

# Run with coverage
uv run pytest --cov
```

**Test Coverage:** 270+ tests, all passing ✅

## 🤝 Contributing

Contributions are very welcome! See [CONTRIBUTING.md](CONTRIBUTING.md) for complete guidelines.

**Quick Start for Contributors:**

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'feat: Add AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

**Note:** All new modules must include both English (README.md) and Indonesian (README.id.md) documentation.

## 📜 License

This project is licensed under the MIT License - see [LICENSE](LICENSE) for details.

## 📧 Contact & Support

- **Issues**: [GitHub Issues](https://github.com/imam-db/ml-algo/issues)
- **Discussions**: [GitHub Discussions](https://github.com/imam-db/ml-algo/discussions)
- **Contributing**: [Contributing Guidelines](CONTRIBUTING.md)
- **Code of Conduct**: [Code of Conduct](CODE_OF_CONDUCT.md)

## 🌟 Acknowledgments

- Thanks to all contributors
- Inspired by [scikit-learn](https://scikit-learn.org/) documentation
- Educational resources from [Andrew Ng](https://www.coursera.org/learn/machine-learning) and [Fast.ai](https://www.fast.ai/)

---

**Happy Learning! 🎉**

*This repository is created with ❤️ for machine learning learners*
