# 📚 Machine Learning Glossary

**🏠 [Back to Main](./README.md)** | **🔍 Quick Search**: Use `Ctrl+F` to find any term

A comprehensive reference for machine learning terminology used throughout this repository.

---

## 📖 Table of Contents

- [🧠 Core ML Concepts](#-core-ml-concepts)
- [📊 Evaluation Metrics](#-evaluation-metrics)  
- [🔢 Mathematical Terms](#-mathematical-terms)
- [⚙️ Model Training](#️-model-training)
- [📈 Data & Features](#-data--features)
- [🏗️ Model Architecture](#️-model-architecture)
- [🎯 Problem Types](#-problem-types)
- [🔧 Tools & Techniques](#-tools--techniques)

---

## 🧠 Core ML Concepts

### **Algorithm**
A set of rules or instructions that a computer follows to solve problems. In ML, algorithms learn patterns from data to make predictions.

### **Artificial Intelligence (AI)**
The broader field of making machines smart. Machine learning is a subset of AI.

### **Bias**
1. **Statistical Bias**: Systematic error in predictions
2. **Model Bias**: The intercept term in linear models (β₀)
3. **Algorithmic Bias**: Unfair treatment of certain groups

### **Cross-Validation (CV)**
A technique to assess model performance by splitting data into multiple train/test sets. Common types:
- **K-Fold CV**: Split data into K parts, train on K-1, test on 1
- **Stratified CV**: Maintains class distribution in each fold

### **Feature**
An individual measurable property of observed phenomena. Also called variables, attributes, or predictors.
- **Example**: In house price prediction, features might be size, location, bedrooms

### **Ground Truth**
The actual, correct answer that we're trying to predict. Also called labels or target values.

### **Machine Learning (ML)**
A method of data analysis that automates analytical model building. Systems learn from data without being explicitly programmed.

### **Model**
The output of an algorithm trained on data. It represents the patterns learned and can make predictions on new data.

### **Overfitting**
When a model learns the training data too well, including noise and random fluctuations. Results in:
- ✅ High performance on training data
- ❌ Poor performance on new, unseen data
- **Solution**: Regularization, cross-validation, more data

### **Underfitting**
When a model is too simple to capture underlying patterns in data. Results in:
- ❌ Poor performance on training data  
- ❌ Poor performance on test data
- **Solution**: More complex model, better features

---

## 📊 Evaluation Metrics

### **Regression Metrics**

#### **MAE (Mean Absolute Error)**
Average of absolute differences between predicted and actual values.
- **Formula**: `MAE = (1/n) × Σ|yi - ŷi|`
- **Range**: 0 to ∞ (lower is better)
- **Good for**: Easy interpretation, robust to outliers

#### **MSE (Mean Squared Error)**
Average of squared differences between predicted and actual values.
- **Formula**: `MSE = (1/n) × Σ(yi - ŷi)²`
- **Range**: 0 to ∞ (lower is better)
- **Good for**: Penalizes large errors more heavily

#### **RMSE (Root Mean Squared Error)**
Square root of MSE, in same units as the target variable.
- **Formula**: `RMSE = √MSE`
- **Range**: 0 to ∞ (lower is better)
- **Good for**: Interpretable, same scale as predictions

#### **R² (R-squared / Coefficient of Determination)**
Proportion of variance in target variable explained by the model.
- **Formula**: `R² = 1 - (SS_res / SS_tot)`
- **Range**: -∞ to 1 (higher is better)
- **Interpretation**: 
  - 1.0 = Perfect predictions
  - 0.0 = No better than predicting the mean
  - Negative = Worse than predicting the mean

### **Classification Metrics**

#### **Accuracy**
Percentage of correct predictions out of all predictions.
- **Formula**: `Accuracy = (TP + TN) / (TP + TN + FP + FN)`
- **Range**: 0 to 1 (higher is better)
- **Issue**: Can be misleading with imbalanced datasets

#### **Precision**
Of all positive predictions, how many were actually positive?
- **Formula**: `Precision = TP / (TP + FP)`
- **Use case**: When false positives are costly (e.g., spam detection)

#### **Recall (Sensitivity)**
Of all actual positives, how many did we correctly identify?
- **Formula**: `Recall = TP / (TP + FN)`
- **Use case**: When false negatives are costly (e.g., disease detection)

#### **F1-Score**
Harmonic mean of precision and recall.
- **Formula**: `F1 = 2 × (Precision × Recall) / (Precision + Recall)`
- **Use case**: When you need balance between precision and recall

#### **AUC-ROC**
Area Under the Receiver Operating Characteristic curve. Measures ability to distinguish between classes.
- **Range**: 0 to 1 (higher is better)
- **Interpretation**: 0.5 = Random guessing, 1.0 = Perfect classifier

---

## 🔢 Mathematical Terms

### **Coefficient**
The multiplier for each feature in linear models (β₁, β₂, etc.). Shows the relationship strength between feature and target.

### **Cost Function (Loss Function)**
Mathematical function that measures how wrong a model's predictions are. The goal is to minimize this.

### **Derivative**
Rate of change of a function. Used in optimization to find the direction to adjust parameters.

### **Gradient**
Vector of all partial derivatives. Points in the direction of steepest increase of a function.

### **Gradient Descent**
Optimization algorithm that iteratively moves toward the minimum of a cost function by following the negative gradient.

### **Intercept**
The y-axis intercept in linear models (β₀). The predicted value when all features equal zero.

### **Learning Rate**
Step size in gradient descent. Controls how big steps the algorithm takes toward the minimum.
- Too high: May overshoot the minimum
- Too low: Slow convergence

### **Linear Combination**
Sum of variables each multiplied by a coefficient: `y = β₀ + β₁x₁ + β₂x₂ + ...`

### **Matrix**
Rectangular array of numbers. Used to represent datasets and perform calculations efficiently.

### **Normal Equation**
Analytical solution to find optimal parameters in linear regression: `θ = (X^T × X)^(-1) × X^T × y`

### **Optimization**
Process of finding the best parameters for a model to minimize the cost function.

---

## ⚙️ Model Training

### **Batch Size**
Number of training examples processed before updating model parameters.

### **Convergence**
When the optimization algorithm stops improving (reaches minimum of cost function).

### **Epoch**
One complete pass through the entire training dataset.

### **Hyperparameter**
Configuration settings for algorithms (not learned from data). Examples: learning rate, number of trees, regularization strength.

### **Hyperparameter Tuning**
Process of finding optimal hyperparameter values. Methods:
- **Grid Search**: Try all combinations
- **Random Search**: Try random combinations  
- **Bayesian Optimization**: Use previous results to guide search

### **Iteration**
Single update step in an optimization algorithm.

### **Parameter**
Values learned by the model from training data (weights, coefficients).

### **Regularization**
Technique to prevent overfitting by adding penalty for complexity:
- **L1 (Lasso)**: Adds sum of absolute values of parameters
- **L2 (Ridge)**: Adds sum of squared parameters

### **Training**
Process of teaching the algorithm using labeled data.

### **Validation**
Evaluating model performance on data not used for training, to check for overfitting.

---

## 📈 Data & Features

### **Dataset**
Collection of data used for training and testing machine learning models.

### **Feature Engineering**
Creating new features or transforming existing ones to improve model performance.

### **Feature Scaling**
Normalizing features to similar ranges. Types:
- **Normalization**: Scale to 0-1 range
- **Standardization**: Zero mean, unit variance

### **Feature Selection**
Choosing the most relevant features for model training.

### **Imbalanced Dataset**
When classes are not represented equally (e.g., 95% class A, 5% class B).

### **Outlier**
Data point significantly different from other observations. Can skew model performance.

### **Target Variable**
The variable we're trying to predict. Also called dependent variable or label.

### **Training Set**
Data used to train the model (typically 70-80% of total data).

### **Test Set** 
Data held back to evaluate final model performance (typically 20-30% of total data).

### **Validation Set**
Data used during training to tune hyperparameters and prevent overfitting.

---

## 🏗️ Model Architecture

### **Ensemble Method**
Combining multiple models to get better performance than individual models.

### **Boosting**
Ensemble technique where models are trained sequentially, each trying to correct previous model's mistakes.

### **Bagging**
Ensemble technique where multiple models are trained on different subsets of data and predictions are averaged.

### **Decision Tree**
Model that makes predictions by asking a series of yes/no questions.

### **Random Forest**
Ensemble of many decision trees, each trained on random subsets of data and features.

### **Neural Network**
Model inspired by biological neurons, with interconnected layers of nodes.

### **Linear Model**
Model that assumes linear relationship between features and target.

---

## 🎯 Problem Types

### **Supervised Learning**
Learning with labeled examples (input-output pairs).

### **Unsupervised Learning**
Finding patterns in data without labels.

### **Classification**
Predicting categories or classes (e.g., spam/not spam, cat/dog).

### **Regression**
Predicting continuous numerical values (e.g., price, temperature).

### **Clustering**
Grouping similar data points together.

### **Time Series**
Data with temporal ordering (e.g., stock prices over time).

---

## 🔧 Tools & Techniques

### **Cross-Validation**
See [Core ML Concepts](#cross-validation-cv)

### **Early Stopping**
Stopping training when validation performance stops improving, to prevent overfitting.

### **Feature Importance**
Measure of how much each feature contributes to model predictions.

### **Grid Search**
Exhaustive search over specified hyperparameter values.

### **Pipeline**
Sequence of data processing steps and model training combined into one object.

### **SHAP (SHapley Additive exPlanations)**
Method to explain individual predictions by showing contribution of each feature.

### **Stratified Sampling**
Sampling that maintains the same proportion of classes as in the original dataset.

---

## 🔗 Quick References by Algorithm

### Linear Regression Key Terms
- [Coefficient](#coefficient), [Intercept](#intercept), [MSE](#mse-mean-squared-error), [R²](#r²-r-squared--coefficient-of-determination)
- [Normal Equation](#normal-equation), [Gradient Descent](#gradient-descent), [Overfitting](#overfitting)

### XGBoost Key Terms  
- [Boosting](#boosting), [Ensemble Method](#ensemble-method), [Feature Importance](#feature-importance)
- [Hyperparameter Tuning](#hyperparameter-tuning), [Cross-Validation](#cross-validation-cv), [SHAP](#shap-shapley-additive-explanations)

---

## 💡 How to Use This Glossary

1. **While Reading**: Click term links in READMEs to get quick definitions
2. **Quick Search**: Use `Ctrl+F` to find any term instantly  
3. **Study Reference**: Review relevant sections before starting new algorithms
4. **Context Learning**: Check algorithm-specific quick references

---

**📝 Contributing**: Found a term that needs explanation? Feel free to add it to this glossary!

**🏠 [Back to Main README](./README.md)** | **🎯 [Start Learning](./README.md#-quick-navigation)**