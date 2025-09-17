# Random Forest

🏠 Back to Supervised Index: `../README.md`

Random Forest is a powerful ensemble learning method that combines multiple decision trees to create a more robust and accurate model. It's one of the most popular and effective machine learning algorithms.

## 📚 Table of Contents
- [Theory & Concepts](#-theory--concepts)
- [Mathematical Foundation](#-mathematical-foundation)
- [Ensemble Methods](#-ensemble-methods)
- [Bootstrap Aggregating (Bagging)](#-bootstrap-aggregating-bagging)
- [Advantages & Disadvantages](#-advantages--disadvantages)
- [When to Use](#-when-to-use)
- [Files in This Directory](#-files-in-this-directory)

## 🧠 Theory & Concepts

### What is Random Forest?
Random Forest is an ensemble method that constructs multiple decision trees during training and outputs the average prediction (regression) or majority vote (classification) of the individual trees. It combines the concepts of **bagging** and **random feature selection**.

### Key Concepts:
1. **Ensemble Learning**: Combining multiple weak learners to create a strong learner
2. **Bootstrap Sampling**: Creating different datasets by sampling with replacement
3. **Random Feature Selection**: Each tree uses a random subset of features
4. **Voting/Averaging**: Combining predictions from all trees
5. **Out-of-Bag (OOB) Error**: Error estimation using samples not used in training each tree
6. **Feature Importance**: Measuring importance based on impurity decrease across all trees

### How Random Forest Works:
```
1. Create B bootstrap samples from training data
2. For each bootstrap sample:
   - Train a decision tree
   - At each split, consider only m random features (m < total features)
   - Grow tree without pruning
3. For prediction:
   - Classification: Majority vote from all trees
   - Regression: Average prediction from all trees
```

## 📐 Mathematical Foundation

### Bootstrap Sampling
```
Given dataset D with n samples:
- Create bootstrap sample D_i by sampling n samples WITH replacement
- Each D_i contains ~63.2% unique samples from D
- Remaining ~36.8% are "out-of-bag" (OOB) samples
```

### Random Feature Selection
```
At each node split:
- Total features: p
- Random features considered: m = √p (classification) or m = p/3 (regression)
- Best split chosen from these m features only
```

### Ensemble Prediction

#### For Classification:
```
Final_Prediction = mode(Tree_1, Tree_2, ..., Tree_B)

Or with probabilities:
P(class_k) = (1/B) * Σ P_i(class_k)
where P_i(class_k) is probability from tree i
```

#### For Regression:
```
Final_Prediction = (1/B) * Σ Tree_i_Prediction
```

### Out-of-Bag (OOB) Error
```
For each sample x_i:
- Find all trees where x_i was NOT in training set
- Make prediction using only those trees
- Compare with true label to compute OOB error

OOB_Error = (1/n) * Σ L(y_i, ŷ_i^OOB)
where L is loss function and ŷ_i^OOB is OOB prediction
```

### Feature Importance
```
For each feature j:
1. Calculate impurity decrease when feature j is used for splitting
2. Average across all trees and all nodes where feature j is used
3. Normalize by total impurity decrease

Importance_j = Σ(trees) Σ(nodes using feature j) Impurity_Decrease / Total_Impurity_Decrease
```

## 🌳 Ensemble Methods

### Types of Ensemble Methods:

1. **Bagging (Bootstrap Aggregating)**
   - Train models on different subsets of data
   - Combine predictions by voting/averaging
   - Reduces variance (overfitting)
   - Random Forest uses this

2. **Boosting**
   - Train models sequentially
   - Each model learns from previous model's errors
   - Reduces bias (underfitting)
   - Examples: AdaBoost, Gradient Boosting

3. **Stacking**
   - Train multiple different algorithms
   - Use meta-learner to combine predictions
   - Can reduce both bias and variance

### Why Ensemble Works:

#### Bias-Variance Decomposition:
```
Total Error = Bias² + Variance + Irreducible Error

Random Forest:
- Individual trees: High variance, low bias
- Ensemble: Reduces variance while maintaining low bias
```

#### Wisdom of Crowds:
- Individual trees make different errors
- Averaging reduces overall error
- Works best when trees are diverse and uncorrelated

## 🎒 Bootstrap Aggregating (Bagging)

### Bootstrap Process:
1. **Original Dataset**: D = {(x₁,y₁), (x₂,y₂), ..., (xₙ,yₙ)}
2. **Bootstrap Sample**: Sample n instances WITH replacement
3. **Multiple Samples**: Create B bootstrap samples
4. **Train Models**: One model per bootstrap sample
5. **Aggregate**: Combine predictions

### Benefits of Bagging:
- **Variance Reduction**: Averaging reduces prediction variance
- **Stability**: Less sensitive to small data changes  
- **Parallel Training**: Each model can be trained independently
- **OOB Evaluation**: Built-in validation without separate test set

### Mathematical Intuition:
```
If each tree has error variance σ²:
- Single tree variance: σ²
- Average of B independent trees: σ²/B
- With correlation ρ between trees: ρσ² + (1-ρ)σ²/B

Random feature selection reduces correlation ρ
```

## ✅ Advantages & Disadvantages

### ✅ Advantages:
- **High Accuracy**: Often provides excellent performance
- **Robust to Overfitting**: Averaging reduces overfitting risk
- **Handles Missing Values**: Can maintain accuracy with missing data
- **Feature Importance**: Built-in feature importance ranking
- **No Scaling Required**: Works with features on different scales
- **Parallel Training**: Trees can be trained simultaneously
- **OOB Validation**: Built-in model validation
- **Versatile**: Works for both classification and regression
- **Handles Large Datasets**: Scales well with data size
- **Robust to Outliers**: Individual trees may be affected, but ensemble is robust

### ❌ Disadvantages:
- **Less Interpretable**: Harder to understand than single decision tree
- **Memory Intensive**: Stores multiple trees
- **Slower Prediction**: Must query all trees for prediction
- **Can Overfit**: With very noisy data or too many trees
- **Biased Toward Categorical Features**: With many categories
- **Not Great for Linear Relationships**: Like individual trees
- **Hyperparameter Tuning**: More parameters to optimize

## 🎯 When to Use

### ✅ Use Random Forest When:
- You need high accuracy without much tuning
- Dataset has mixed feature types (numerical + categorical)
- You want feature importance rankings
- Interpretability is less important than performance
- You have sufficient training data
- You want a robust, general-purpose algorithm
- You need to handle missing values gracefully
- You want built-in validation (OOB error)

### ❌ Don't Use When:
- You need highly interpretable models
- You have very limited memory/computational resources
- Dataset is very small (< 1000 samples)
- Prediction speed is critical
- Relationships are primarily linear
- You need probability calibration (though this can be fixed)

### Real-World Applications:
- **Finance**: Credit scoring, algorithmic trading, risk assessment
- **Healthcare**: Disease prediction, drug discovery, medical imaging
- **Technology**: Recommendation systems, image classification, NLP
- **Marketing**: Customer segmentation, churn prediction, A/B testing
- **Manufacturing**: Quality control, predictive maintenance
- **Bioinformatics**: Gene expression analysis, protein structure prediction

## 🔧 Random Forest Variants

### 1. **Standard Random Forest**
- Random bootstrap sampling
- Random feature selection at each split
- No pruning of individual trees

### 2. **Extremely Randomized Trees (Extra Trees)**
- Random feature selection
- Random thresholds (not optimal splits)
- Uses entire dataset (no bootstrap)
- Faster training, potentially better generalization

### 3. **Balanced Random Forest**
- Handles imbalanced datasets
- Balances bootstrap samples
- Good for skewed class distributions

### 4. **Rotation Forest**
- Applies PCA to feature subsets
- Rotates feature space for each tree
- Can improve performance on some datasets

## 📁 Files in This Directory

- **`README.md`** (this file) - Theory and concepts
- **`implementation.py`** - Random Forest from scratch
- **`sklearn_example.py`** - Practical examples using scikit-learn
- **`exercise.ipynb`** - Hands-on exercises and experiments
- **`datasets/`** - Sample datasets for practice

## 🚀 Quick Start

1. **Understand the Theory**: Read this README
2. **See Implementation**: Check `implementation.py`
3. **Try Examples**: Run `sklearn_example.py`
4. **Practice**: Complete `exercise.ipynb`
5. **Experiment**: Use your own datasets

## 🎛️ Important Hyperparameters

### Scikit-learn Parameters:
```python
RandomForestClassifier(
    n_estimators=100,          # Number of trees
    criterion='gini',          # Split criterion
    max_depth=None,           # Maximum depth per tree
    min_samples_split=2,      # Min samples to split node
    min_samples_leaf=1,       # Min samples in leaf
    max_features='sqrt',      # Features per split
    bootstrap=True,           # Use bootstrap sampling
    oob_score=False,          # Calculate OOB score
    n_jobs=None,              # Parallel jobs (-1 for all cores)
    random_state=None,        # Random seed
    warm_start=False,         # Add trees incrementally
    class_weight=None,        # Handle class imbalance
)
```

### Key Parameters Explained:
- **n_estimators**: More trees = better performance, diminishing returns after ~100-500
- **max_features**: Controls randomness and overfitting (sqrt for classification, 1/3 for regression)
- **max_depth**: Controls individual tree complexity
- **min_samples_split/leaf**: Prevents overfitting by requiring minimum samples
- **bootstrap**: Whether to use bootstrap sampling (True for RF, False for Extra Trees)
- **oob_score**: Whether to calculate out-of-bag validation score

## 📊 Performance Tuning

### Hyperparameter Tuning Strategy:
1. **Start with defaults**: Often work well
2. **Tune n_estimators**: Find point where performance plateaus
3. **Tune max_features**: Try sqrt(p), log2(p), p/3
4. **Tune tree depth**: max_depth, min_samples_split, min_samples_leaf
5. **Handle class imbalance**: class_weight, balanced sampling

### Cross-Validation Grid:
```python
param_grid = {
    'n_estimators': [100, 200, 500],
    'max_features': ['sqrt', 'log2', None],
    'max_depth': [3, 5, 7, None],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4]
}
```

## 🔗 Related Algorithms

- **Decision Trees**: Building blocks of Random Forest
- **Extra Trees**: Extremely randomized trees variant
- **Gradient Boosting**: Sequential ensemble method
- **XGBoost/LightGBM**: Optimized gradient boosting
- **Bagging**: General bootstrap aggregating method

---

**Next Steps**: 
- Try the implementation in `sklearn_example.py`
- Complete exercises in `exercise.ipynb`
- Compare with other ensemble methods
- Explore feature importance analysis