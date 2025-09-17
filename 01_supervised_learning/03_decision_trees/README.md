# Decision Trees

🏠 Back to Supervised Index: `../README.md`

Decision Trees are one of the most intuitive and interpretable machine learning algorithms. They create a model that predicts target values by learning simple decision rules inferred from data features.

## 📚 Table of Contents
- [Theory & Concepts](#-theory--concepts)
- [Mathematical Foundation](#-mathematical-foundation)
- [Tree Construction Process](#-tree-construction-process)
- [Splitting Criteria](#-splitting-criteria)
- [Advantages & Disadvantages](#-advantages--disadvantages)
- [When to Use](#-when-to-use)
- [Files in This Directory](#-files-in-this-directory)

## 🧠 Theory & Concepts

### What are Decision Trees?
Decision Trees are a non-parametric supervised learning method used for both classification and regression. The goal is to create a model that predicts the value of a target variable by learning simple decision rules inferred from the data features.

### Key Concepts:
1. **Root Node**: The topmost node representing the entire dataset
2. **Internal Nodes**: Nodes that have child nodes, representing feature tests
3. **Leaf Nodes**: Terminal nodes with no children, containing predictions
4. **Branches**: Connections between nodes representing decision paths
5. **Depth**: The longest path from root to leaf
6. **Pruning**: Removing nodes to prevent overfitting

### How Decision Trees Work:
```
Is Feature_A > threshold?
├─ Yes → Is Feature_B > threshold?
│          ├─ Yes → Predict Class A
│          └─ No → Predict Class B
└─ No → Predict Class C
```

## 📐 Mathematical Foundation

### Information Theory Concepts

#### Entropy (for Classification)
```
Entropy(S) = -Σ(p_i * log2(p_i))

Where:
- S is the dataset
- p_i is the proportion of examples in class i
- Lower entropy = more pure/homogeneous data
```

#### Gini Impurity
```
Gini(S) = 1 - Σ(p_i²)

Where:
- p_i is the proportion of examples in class i
- Measures the probability of misclassifying a randomly chosen element
```

#### Information Gain
```
Information_Gain(S, A) = Entropy(S) - Σ(|Sv|/|S| * Entropy(Sv))

Where:
- S is the original dataset
- A is the attribute/feature
- Sv is the subset of S where attribute A has value v
```

#### Gain Ratio (C4.5 Algorithm)
```
Gain_Ratio(S, A) = Information_Gain(S, A) / Split_Information(S, A)

Split_Information(S, A) = -Σ(|Sv|/|S| * log2(|Sv|/|S|))
```

### For Regression Trees

#### Mean Squared Error (MSE)
```
MSE = (1/n) * Σ(yi - ŷ)²

Where:
- yi is the actual value
- ŷ is the predicted value (mean of leaf)
```

#### Variance Reduction
```
Variance_Reduction = Var(parent) - Σ(|child_i|/|parent| * Var(child_i))
```

## 🌳 Tree Construction Process

### 1. **Recursive Binary Splitting**
- Start with the root node containing all training data
- Find the best feature and split point that maximizes information gain
- Create two child nodes based on the split
- Recursively apply the same process to child nodes

### 2. **Stopping Criteria**
- Maximum depth reached
- Minimum samples per leaf
- Minimum samples to split
- No further information gain possible
- All samples have the same target value

### 3. **Prediction Process**
- Start at the root node
- Follow the decision rules down the tree
- Reach a leaf node and return its prediction

## ⚖️ Splitting Criteria

### For Classification:
1. **Information Gain (ID3)**
   - Uses entropy to measure impurity
   - Prefers attributes with more levels
   - Can lead to overfitting

2. **Gain Ratio (C4.5)**
   - Normalizes information gain by split information
   - Reduces bias towards multi-valued attributes
   - More balanced splitting

3. **Gini Index (CART)**
   - Measures impurity of a dataset
   - Computationally efficient
   - Less sensitive to outliers

### For Regression:
1. **Mean Squared Error (MSE)**
   - Minimizes the variance in leaf nodes
   - Standard approach for regression trees

2. **Mean Absolute Error (MAE)**
   - More robust to outliers
   - Alternative to MSE

## ✅ Advantages & Disadvantages

### ✅ Advantages:
- **Interpretable**: Easy to understand and visualize
- **No Assumptions**: Works with non-linear relationships
- **Handles Mixed Data**: Both numerical and categorical features
- **Feature Selection**: Implicit feature importance
- **Non-parametric**: No assumptions about data distribution
- **Missing Values**: Can handle missing data naturally
- **Fast Prediction**: O(log n) prediction time

### ❌ Disadvantages:
- **Overfitting**: Prone to overfitting, especially with deep trees
- **Instability**: Small changes in data can result in different trees
- **Bias**: Biased toward features with more levels
- **Poor Extrapolation**: Cannot predict beyond training data range
- **Difficulty with Linear Relationships**: Inefficient for simple linear patterns
- **High Variance**: Small data changes can dramatically change the tree

## 🎯 When to Use

### ✅ Use Decision Trees When:
- Interpretability is crucial (medical diagnosis, loan approval)
- You have mixed data types (numerical and categorical)
- The relationship between features and target is non-linear
- You need feature importance rankings
- You want a baseline model that's easy to understand
- Data has missing values that are difficult to impute

### ❌ Don't Use When:
- You have high-dimensional data with many irrelevant features
- The relationship is primarily linear (use linear models instead)
- You have very noisy data
- Dataset is very small
- You need the highest possible accuracy (use ensemble methods)

### Real-World Applications:
- **Medical Diagnosis**: Symptom-based diagnosis trees
- **Finance**: Credit scoring and loan approval
- **Marketing**: Customer segmentation and targeting
- **Manufacturing**: Quality control and fault detection
- **HR**: Resume screening and employee evaluation
- **Sports**: Game strategy and player evaluation

## 🔧 Tree Variants

### 1. **ID3 (Iterative Dichotomiser 3)**
- Uses information gain for splitting
- Only handles categorical features
- No pruning mechanism
- Can overfit easily

### 2. **C4.5**
- Improvement over ID3
- Uses gain ratio for splitting
- Handles continuous and missing values
- Includes pruning mechanism

### 3. **CART (Classification and Regression Trees)**
- Used by scikit-learn
- Binary splits only
- Uses Gini index for classification, MSE for regression
- Built-in pruning with cost-complexity

### 4. **CHAID (Chi-squared Automatic Interaction Detector)**
- Uses chi-squared test for splitting
- Allows multi-way splits
- Good for categorical targets

## 📁 Files in This Directory

- **`README.md`** (this file) - Theory and concepts
- **`implementation.py`** - Decision tree from scratch
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
DecisionTreeClassifier(
    criterion='gini',           # 'gini', 'entropy', 'log_loss'
    max_depth=None,            # Maximum depth of tree
    min_samples_split=2,       # Minimum samples to split internal node
    min_samples_leaf=1,        # Minimum samples in leaf node
    max_features=None,         # Number of features for best split
    max_leaf_nodes=None,       # Maximum number of leaf nodes
    min_impurity_decrease=0.0, # Minimum impurity decrease for split
    random_state=None          # Random seed for reproducibility
)
```

### Key Parameters Explained:
- **max_depth**: Controls overfitting, deeper = more complex
- **min_samples_split**: Higher values prevent overfitting
- **min_samples_leaf**: Ensures leaves have minimum samples
- **max_features**: Controls randomness and speed
- **criterion**: Different measures of split quality

## 📊 Evaluation Metrics

### Classification:
- **Accuracy**: Overall correctness
- **Precision/Recall**: Class-specific performance
- **F1-Score**: Balanced precision and recall
- **ROC-AUC**: True positive vs false positive rate
- **Feature Importance**: Which features are most useful

### Regression:
- **Mean Squared Error (MSE)**: Average squared differences
- **Mean Absolute Error (MAE)**: Average absolute differences
- **R² Score**: Coefficient of determination
- **Feature Importance**: Impact of each feature

## 🌲 Tree Visualization

Decision trees can be visualized using:
```python
from sklearn.tree import plot_tree, export_text
import matplotlib.pyplot as plt

# Plot the tree
plt.figure(figsize=(20,10))
plot_tree(model, feature_names=feature_names, 
          class_names=class_names, filled=True)

# Text representation
tree_rules = export_text(model, feature_names=feature_names)
print(tree_rules)
```

## 🔗 Related Algorithms

- **Random Forest**: Ensemble of decision trees
- **Gradient Boosting**: Sequential tree learning
- **Extra Trees**: Extremely randomized trees
- **XGBoost/LightGBM**: Optimized gradient boosting

---

**Next Steps**: 
- Try the implementation in `sklearn_example.py`
- Complete exercises in `exercise.ipynb`
- Compare with ensemble methods in `../04_random_forest/`