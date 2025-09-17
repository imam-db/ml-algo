# Logistic Regression

🏠 Back to Supervised Index: `../README.md`

Logistic Regression is a fundamental classification algorithm that uses the logistic function to model the probability of class membership. Despite its name, it's used for classification, not regression.

## 📚 Table of Contents
- [Theory & Concepts](#-theory--concepts)
- [Mathematical Foundation](#-mathematical-foundation)
- [Assumptions](#-assumptions)
- [Implementation](#-implementation)
- [Advantages & Disadvantages](#-advantages--disadvantages)
- [When to Use](#-when-to-use)
- [Files in This Directory](#-files-in-this-directory)

## 🧠 Theory & Concepts

### What is Logistic Regression?
Logistic Regression predicts the probability that an instance belongs to a particular category. It uses the **sigmoid (logistic) function** to map any real number to a value between 0 and 1, making it perfect for binary classification.

### Key Concepts:
1. **Sigmoid Function**: Maps input to probability (0-1)
2. **Decision Boundary**: Linear boundary that separates classes
3. **Maximum Likelihood Estimation**: Method to find optimal parameters
4. **Log-Odds**: Natural logarithm of odds ratio

### Types of Logistic Regression:
- **Binary**: Two classes (yes/no, spam/not spam)
- **Multinomial**: Multiple classes (cat/dog/bird)
- **Ordinal**: Ordered classes (low/medium/high)

## 📐 Mathematical Foundation

### The Sigmoid Function
```
σ(z) = 1 / (1 + e^(-z))
where z = β₀ + β₁x₁ + β₂x₂ + ... + βₙxₙ
```

### Probability and Odds
```
# Probability
P(y=1|x) = σ(β₀ + β₁x₁ + β₂x₂ + ... + βₙxₙ)

# Odds
Odds = P(y=1|x) / (1 - P(y=1|x))

# Log-Odds (Logit)
logit(p) = ln(odds) = β₀ + β₁x₁ + β₂x₂ + ... + βₙxₙ
```

### Cost Function (Log-Likelihood)
```
J(θ) = -(1/m) * Σ[y*log(h(x)) + (1-y)*log(1-h(x))]
where h(x) = sigmoid function
```

### Gradient Descent Update
```
θⱼ := θⱼ - α * (1/m) * Σ[(h(x⁽ⁱ⁾) - y⁽ⁱ⁾) * xⱼ⁽ⁱ⁾]
```

## ⚖️ Assumptions

1. **Linear Relationship**: Between logit of outcome and features
2. **Independence**: Observations are independent
3. **No Perfect Multicollinearity**: Features aren't perfectly correlated
4. **Large Sample Size**: More data improves accuracy
5. **No Extreme Outliers**: Can affect the model significantly

## 🔧 Implementation

### From Scratch Implementation
See `implementation.py` for step-by-step implementation including:
- Sigmoid function
- Cost function calculation
- Gradient descent optimization
- Prediction method

### Using scikit-learn
See `sklearn_example.py` for practical examples:
- Data preparation
- Model training
- Evaluation metrics
- Feature importance

## ✅ Advantages & Disadvantages

### ✅ Advantages:
- **Probabilistic Output**: Provides probability estimates
- **No Tuning**: Fewer hyperparameters to tune
- **Less Overfitting**: With low-dimensional data
- **Interpretable**: Coefficients show feature importance
- **Fast**: Quick training and prediction
- **No Scaling Required**: Robust to feature scales

### ❌ Disadvantages:
- **Linear Decision Boundary**: Can't capture complex relationships
- **Sensitive to Outliers**: Extreme values can skew results
- **Large Sample Size**: Needs substantial data for stable results
- **Perfect Separation**: Problems when classes are perfectly separable
- **Multicollinearity**: Struggles with correlated features

## 🎯 When to Use

### ✅ Use Logistic Regression When:
- Binary or multi-class classification needed
- You need probability estimates, not just predictions
- Dataset is linearly separable or nearly so
- Interpretability is important
- You have limited training time
- Features have linear relationship with log-odds

### ❌ Don't Use When:
- You need to capture complex non-linear patterns
- Dataset has perfect multicollinearity
- Classes are perfectly separable (causes numerical issues)
- You have very small datasets

### Real-World Applications:
- **Medical Diagnosis**: Disease prediction based on symptoms
- **Marketing**: Customer conversion prediction
- **Finance**: Credit approval, fraud detection
- **Email**: Spam classification
- **Social Media**: Sentiment analysis

## 📁 Files in This Directory

- **`README.md`** (this file) - Theory and concepts
- **`implementation.py`** - Logistic regression from scratch
- **`sklearn_example.py`** - Practical examples using scikit-learn
- **`exercise.ipynb`** - Hands-on exercises and experiments
- **`datasets/`** - Sample datasets for practice

## 🚀 Quick Start

1. **Understand the Theory**: Read this README
2. **See Implementation**: Check `implementation.py`
3. **Try Examples**: Run `sklearn_example.py`
4. **Practice**: Complete `exercise.ipynb`
5. **Experiment**: Use your own datasets

## 📊 Performance Metrics

For classification problems, evaluate using:
- **Accuracy**: Overall correctness
- **Precision**: True positives / (True positives + False positives)
- **Recall**: True positives / (True positives + False negatives)
- **F1-Score**: Harmonic mean of precision and recall
- **ROC-AUC**: Area under ROC curve
- **Confusion Matrix**: Detailed breakdown of predictions

## 🔗 Related Algorithms

- **Linear Regression**: The linear version for continuous outputs
- **Decision Trees**: Non-linear alternative for classification
- **SVM**: Another linear classifier with different approach
- **Naive Bayes**: Probabilistic classifier based on Bayes' theorem

---

**Next Steps**: 
- Try the implementation in `sklearn_example.py`
- Complete exercises in `exercise.ipynb`
- Compare with other algorithms in `../`
