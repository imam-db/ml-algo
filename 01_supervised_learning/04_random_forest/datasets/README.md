
# Random Forest Practice Datasets 📊

This folder contains sample datasets for practicing Random Forest algorithms. Each dataset is designed to demonstrate different aspects of Random Forest applications.

## 📁 Available Datasets

### 1. Customer Churn (customer_churn.csv)
- **Type**: Binary Classification
- **Target**: `churn` (0/1)
- **Size**: ~2,000 samples, 13 features
- **Use Case**: Predicting customer churn for subscription services
- **Key Features**: Age, tenure, charges, contract type, services
- **Challenge**: Moderate class imbalance (~30% churn rate)

### 2. House Prices (house_prices.csv)
- **Type**: Regression
- **Target**: `price` (continuous)
- **Size**: ~1,500 samples, 13 features
- **Use Case**: Real estate price prediction
- **Key Features**: Square footage, bedrooms, bathrooms, location, age
- **Challenge**: Mixed numerical and categorical features

### 3. Credit Approval (credit_approval.csv)
- **Type**: Binary Classification (Imbalanced)
- **Target**: `approved` (0/1)
- **Size**: ~1,000 samples, 10 features
- **Use Case**: Credit application approval prediction
- **Key Features**: Credit score, income, debt-to-income ratio
- **Challenge**: Heavily imbalanced (~25% approval rate)

### 4. Marketing Campaign (marketing_campaign.csv)
- **Type**: Multi-class Classification
- **Target**: `response` (0: No response, 1: Inquired, 2: Purchased)
- **Size**: ~1,200 samples, 11 features
- **Use Case**: Marketing campaign response prediction
- **Key Features**: Customer demographics, purchase history, engagement
- **Challenge**: Multi-class with imbalanced classes

### 5. Employee Satisfaction (employee_satisfaction.csv)
- **Type**: Regression
- **Target**: `satisfaction` (1-10 scale)
- **Size**: ~800 samples, 13 features
- **Use Case**: Predicting employee satisfaction scores
- **Key Features**: Salary, work hours, benefits, department
- **Challenge**: Mixed feature types, ordinal target

## 🚀 Usage Examples

### Loading Data
```python
import pandas as pd

# Load any dataset
df = pd.read_csv('customer_churn.csv')
print(df.head())
print(df.info())
```

### Basic Random Forest Example
```python
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

# Load data
df = pd.read_csv('customer_churn.csv')

# Prepare features (handle categorical variables)
# ... preprocessing steps ...

# Train model
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(X_train, y_train)
```

## 🎯 Learning Objectives

Use these datasets to practice:

1. **Data Preprocessing**: Handle categorical variables, missing values
2. **Feature Engineering**: Create new features, handle different data types
3. **Model Training**: Different Random Forest configurations
4. **Hyperparameter Tuning**: Grid search, random search
5. **Model Evaluation**: Different metrics for different problem types
6. **Feature Importance**: Understand which features matter most
7. **Imbalanced Data**: Handle datasets with uneven class distribution
8. **Multi-class Problems**: Extend beyond binary classification
9. **Regression**: Apply Random Forest to continuous targets
10. **Real-world Applications**: Practice on realistic business problems

## 📚 Suggested Exercises

1. **Compare Algorithms**: Random Forest vs. Decision Tree vs. other algorithms
2. **Feature Selection**: Use Random Forest importance for feature selection
3. **Cross-Validation**: Robust model evaluation
4. **Ensemble Analysis**: Study how number of trees affects performance
5. **Interpretation**: Use feature importance and partial dependence plots
6. **Production Pipeline**: Build complete ML pipeline from data to prediction

## 🔧 Data Generation

All datasets are synthetically generated using realistic statistical models. The generation script (`generate_datasets.py`) can be modified to create datasets with different characteristics:

- Sample sizes
- Number of features
- Class distributions
- Noise levels
- Feature relationships

Happy learning! 🌲🎓
