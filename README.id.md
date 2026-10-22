# Machine Learning Algorithms Learning Repository

**[🇬🇧 English](README.md) | 🇮🇩 Bahasa Indonesia**

[![Python Version](https://img.shields.io/badge/python-3.11%2B-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Tests](https://github.com/imam-db/ml-algo/actions/workflows/tests.yml/badge.svg)](https://github.com/imam-db/ml-algo/actions/workflows/tests.yml)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
[![Contributions Welcome](https://img.shields.io/badge/contributions-welcome-brightgreen.svg?style=flat)](CONTRIBUTING.md)

Selamat datang di repositori pembelajaran algoritma machine learning! 🤖📚

**Resource pembelajaran komprehensif untuk machine learning dengan implementasi dari nol, teori lengkap dalam Bahasa Indonesia, dan latihan interaktif.**

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
- 📚 [GLOSSARY.md](./GLOSSARY.md) - Complete ML terminology reference
- 🔬 [ALGORITHM_COMPARISON.md](./ALGORITHM_COMPARISON.md) - Algorithm selection guide
- 🎯 [PROGRESS_TRACKER.md](./PROGRESS_TRACKER.md) - Learning progress and achievements
- 📁 [Project Structure](#-repository-structure) - Detailed folder organization
- 🚀 [Quick Start](#-quick-start) - Get running in 3 steps

## 📋 Repository Structure

### 1. **Supervised Learning** ([`01_supervised_learning/`](./01_supervised_learning/))
- **[01_linear_regression](./01_supervised_learning/01_linear_regression/)** - Linear Regression ✅
- **[02_logistic_regression](./01_supervised_learning/02_logistic_regression/)** - Logistic Regression ✅
- **[03_decision_trees](./01_supervised_learning/03_decision_trees/)** - Decision Trees ✅
- **[04_random_forest](./01_supervised_learning/04_random_forest/)** - Random Forest ✅
- **🚧 05_svm** - Support Vector Machine (planned)
- **[06_naive_bayes](./01_supervised_learning/06_naive_bayes/)** - Naive Bayes (Gaussian & Multinomial) ✅
- **[07_knn](./01_supervised_learning/07_knn/)** - K-Nearest Neighbors (Classifier & Regressor) ✅
- **[08_neural_networks](./01_supervised_learning/08_neural_networks/)** - Neural Networks (Feedforward) ✅
- **[09_gradient_boosting](./01_supervised_learning/09_gradient_boosting/)** - Gradient Boosting (Regression & Classification) ✅

### 2. **Unsupervised Learning** ([`02_unsupervised_learning/`](./02_unsupervised_learning/))
- **[01_kmeans](./02_unsupervised_learning/01_kmeans/)** - K-Means Clustering ✅
- **[02_dbscan](./02_unsupervised_learning/02_dbscan/)** - DBSCAN (Density-Based Clustering) ✅
- **[03_hierarchical](./02_unsupervised_learning/03_hierarchical/)** - Hierarchical (Agglomerative) Clustering ✅
- **[04_pca](./02_unsupervised_learning/04_pca/)** - Principal Component Analysis ✅
- **🚧 05_tsne** - t-SNE (planned)
- **🚧 06_association_rules** - Association Rules (planned)

### 3. **Reinforcement Learning** ([`03_reinforcement_learning/`](./03_reinforcement_learning/))
- **[01_q_learning](./03_reinforcement_learning/01_q_learning/)** - Q-Learning (Off-Policy TD) ✅
- **[02_sarsa](./03_reinforcement_learning/02_sarsa/)** - SARSA (On-Policy TD) ✅
- **🚧 03_deep_q_networks** - Deep Q-Networks (planned)

### 4. **Advanced Topics** ([`04_advanced_topics/`](./04_advanced_topics/))
- **🚧 01_ensemble_methods** - Ensemble Methods (planned)
- **🚧 02_deep_learning** - Deep Learning (planned)
- **[03_model_evaluation](./04_advanced_topics/03_model_evaluation/)** - Model Evaluation (Metrics & Cross-Validation) ✅
- **[04_xgboost](./04_advanced_topics/04_xgboost/)** - XGBoost (eXtreme Gradient Boosting) ✅

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

## 📊 Status Implementasi

| Category | Implemented | Planned | Progress |
|----------|-------------|---------|----------|
| Supervised Learning | 6 | 2 | 🟢 75% |
| Unsupervised Learning | 4 | 2 | 🟢 67% |
| Reinforcement Learning | 2 | 1 | 🟢 67% |
| Advanced Topics | 2 | 2 | 🟢 50% |

## 🧪 Testing

Repository ini dilengkapi dengan comprehensive test suite:

```bash
# Run all tests
python -m pytest tests/ -v

# Run specific test file
python -m pytest tests/test_reinforcement_learning.py -v

# Run with coverage
python -m pytest tests/ --cov --cov-report=html
```

**Test Coverage:** 160+ tests, semua passing ✅

## 🤝 Contributing

Kontribusi sangat diterima! Lihat [CONTRIBUTING.md](CONTRIBUTING.md) untuk panduan lengkap.

**Quick Start untuk Contributors:**

1. Fork repository
2. Buat feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit changes (`git commit -m 'feat: Add AmazingFeature'`)
4. Push ke branch (`git push origin feature/AmazingFeature`)
5. Buka Pull Request

## 📜 License

Project ini dilisensikan under MIT License - lihat [LICENSE](LICENSE) untuk details.

## 📧 Contact & Support

- **Issues**: [GitHub Issues](https://github.com/YOUR_USERNAME/ml-algorithms-learning/issues)
- **Discussions**: [GitHub Discussions](https://github.com/YOUR_USERNAME/ml-algorithms-learning/discussions)
- **Contributing**: [Contributing Guidelines](CONTRIBUTING.md)
- **Code of Conduct**: [Code of Conduct](CODE_OF_CONDUCT.md)

## 🌟 Acknowledgments

- Terima kasih kepada semua contributors
- Inspired by [scikit-learn](https://scikit-learn.org/) documentation
- Educational resources dari [Andrew Ng](https://www.coursera.org/learn/machine-learning) dan [Fast.ai](https://www.fast.ai/)

---

**Happy Learning! 🎉**

*Repository ini dibuat dengan ❤️ untuk pembelajar machine learning*
