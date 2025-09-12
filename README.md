# Machine Learning Algorithms Learning Repository

Welcome to the machine learning algorithms learning repository! 🤖📚

## 🗺️ Quick Navigation

| Category | Beginner Friendly | Intermediate | Advanced |
|----------|------------------|--------------|----------|
| **Supervised** | [Linear Regression](./01_supervised_learning/01_linear_regression/) | [Random Forest](./01_supervised_learning/04_random_forest/) | [Neural Networks](./01_supervised_learning/08_neural_networks/) |
| **Unsupervised** | [K-Means](./02_unsupervised_learning/01_kmeans/) | [PCA](./02_unsupervised_learning/04_pca/) | [t-SNE](./02_unsupervised_learning/05_tsne/) |
| **Advanced** | [Model Evaluation](./04_advanced_topics/03_model_evaluation/) | [Ensemble Methods](./04_advanced_topics/01_ensemble_methods/) | [XGBoost](./04_advanced_topics/04_xgboost/) |

**Essential Files:**
- 📝 [WARP.md](./WARP.md) - Development guidance for Warp terminal
- 📚 [GLOSSARY.md](./GLOSSARY.md) - Complete ML terminology reference
- 📁 [Project Structure](#-repository-structure) - Detailed folder organization
- 🚀 [Quick Start](#-quick-start) - Get running in 3 steps

## 📋 Repository Structure

### 1. **Supervised Learning** ([`01_supervised_learning/`](./01_supervised_learning/))
- **[01_linear_regression](./01_supervised_learning/01_linear_regression/)** - Linear Regression
- **[02_logistic_regression](./01_supervised_learning/02_logistic_regression/)** - Logistic Regression  
- **[03_decision_trees](./01_supervised_learning/03_decision_trees/)** - Decision Trees
- **[04_random_forest](./01_supervised_learning/04_random_forest/)** - Random Forest
- **[05_svm](./01_supervised_learning/05_svm/)** - Support Vector Machine
- **[06_naive_bayes](./01_supervised_learning/06_naive_bayes/)** - Naive Bayes
- **[07_knn](./01_supervised_learning/07_knn/)** - K-Nearest Neighbors
- **[08_neural_networks](./01_supervised_learning/08_neural_networks/)** - Neural Networks

### 2. **Unsupervised Learning** ([`02_unsupervised_learning/`](./02_unsupervised_learning/))
- **[01_kmeans](./02_unsupervised_learning/01_kmeans/)** - K-Means Clustering
- **[02_hierarchical_clustering](./02_unsupervised_learning/02_hierarchical_clustering/)** - Hierarchical Clustering
- **[03_dbscan](./02_unsupervised_learning/03_dbscan/)** - DBSCAN
- **[04_pca](./02_unsupervised_learning/04_pca/)** - Principal Component Analysis
- **[05_tsne](./02_unsupervised_learning/05_tsne/)** - t-SNE
- **[06_association_rules](./02_unsupervised_learning/06_association_rules/)** - Association Rules

### 3. **Reinforcement Learning** ([`03_reinforcement_learning/`](./03_reinforcement_learning/))
- **[01_q_learning](./03_reinforcement_learning/01_q_learning/)** - Q-Learning
- **[02_deep_q_networks](./03_reinforcement_learning/02_deep_q_networks/)** - Deep Q-Networks

### 4. **Advanced Topics** ([`04_advanced_topics/`](./04_advanced_topics/))
- **[01_ensemble_methods](./04_advanced_topics/01_ensemble_methods/)** - Ensemble Methods
- **[02_deep_learning](./04_advanced_topics/02_deep_learning/)** - Deep Learning
- **[03_model_evaluation](./04_advanced_topics/03_model_evaluation/)** - Model Evaluation
- **[04_xgboost](./04_advanced_topics/04_xgboost/)** - XGBoost (eXtreme Gradient Boosting)

### 5. **Datasets** ([`05_datasets/`](./05_datasets/))
Datasets used for learning

### 6. **Utils** ([`06_utils/`](./06_utils/))
Utility functions and helper code

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