"""
Dataset Generator for Random Forest Practice
===========================================

This script generates various sample datasets for practicing Random Forest algorithms.
Includes classification and regression datasets with different characteristics.

Author: ML Learning Project
Date: 2024
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import make_classification, make_regression, make_blobs
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
import os
import warnings
warnings.filterwarnings('ignore')

def generate_customer_churn_dataset(n_samples=2000, save_csv=True):
    """
    Generate realistic customer churn dataset.
    """
    np.random.seed(42)
    print("🏢 Generating Customer Churn Dataset...")
    
    # Customer demographics
    age = np.random.normal(40, 15, n_samples).astype(int)
    age = np.clip(age, 18, 80)
    
    tenure = np.random.exponential(2, n_samples)
    tenure = np.clip(tenure, 0, 10)
    
    # Service usage
    monthly_charges = np.random.normal(65, 20, n_samples)
    monthly_charges = np.clip(monthly_charges, 20, 150)
    
    total_charges = monthly_charges * tenure * 12 + np.random.normal(0, 100, n_samples)
    total_charges = np.clip(total_charges, 0, None)
    
    # Categorical features
    contract_types = np.random.choice(['Month-to-month', 'One year', 'Two year'], 
                                    n_samples, p=[0.5, 0.3, 0.2])
    
    payment_methods = np.random.choice(['Electronic check', 'Mailed check', 
                                      'Bank transfer', 'Credit card'], 
                                     n_samples, p=[0.3, 0.2, 0.25, 0.25])
    
    internet_service = np.random.choice(['DSL', 'Fiber optic', 'No'], 
                                       n_samples, p=[0.4, 0.4, 0.2])
    
    # Binary services
    online_security = np.random.choice([0, 1], n_samples, p=[0.4, 0.6])
    online_backup = np.random.choice([0, 1], n_samples, p=[0.35, 0.65])
    device_protection = np.random.choice([0, 1], n_samples, p=[0.3, 0.7])
    tech_support = np.random.choice([0, 1], n_samples, p=[0.3, 0.7])
    streaming_tv = np.random.choice([0, 1], n_samples, p=[0.4, 0.6])
    streaming_movies = np.random.choice([0, 1], n_samples, p=[0.4, 0.6])
    
    # Target variable (churn) - influenced by features
    churn_prob = (
        0.1 +  # Base probability
        0.3 * (contract_types == 'Month-to-month') +  # Contract effect
        0.2 * (monthly_charges > 80) / 80 +  # High charges
        0.15 * (tenure < 1) +  # New customers
        0.1 * (payment_methods == 'Electronic check') +  # Payment method
        0.05 * (age < 30) +  # Young customers
        -0.1 * online_security +  # Security reduces churn
        -0.05 * tech_support  # Support reduces churn
    )
    
    churn_prob = np.clip(churn_prob, 0, 1)
    churn = np.random.binomial(1, churn_prob, n_samples)
    
    # Create DataFrame
    df = pd.DataFrame({
        'age': age,
        'tenure': tenure,
        'monthly_charges': monthly_charges,
        'total_charges': total_charges,
        'contract': contract_types,
        'payment_method': payment_methods,
        'internet_service': internet_service,
        'online_security': online_security,
        'online_backup': online_backup,
        'device_protection': device_protection,
        'tech_support': tech_support,
        'streaming_tv': streaming_tv,
        'streaming_movies': streaming_movies,
        'churn': churn
    })
    
    if save_csv:
        filepath = os.path.join(os.path.dirname(__file__), 'customer_churn.csv')
        df.to_csv(filepath, index=False)
        print(f"   ✅ Saved to: {filepath}")
    
    print(f"   📊 Shape: {df.shape}")
    print(f"   🎯 Churn rate: {churn.mean():.2%}")
    print(f"   📈 Features: {df.columns.tolist()}")
    
    return df

def generate_house_prices_dataset(n_samples=1500, save_csv=True):
    """
    Generate realistic house prices dataset for regression.
    """
    np.random.seed(42)
    print("\n🏠 Generating House Prices Dataset...")
    
    # House characteristics
    sqft_living = np.random.normal(2000, 800, n_samples)
    sqft_living = np.clip(sqft_living, 500, 8000)
    
    bedrooms = np.random.poisson(3, n_samples)
    bedrooms = np.clip(bedrooms, 1, 6)
    
    bathrooms = np.random.normal(2.5, 1, n_samples)
    bathrooms = np.clip(bathrooms, 1, 6)
    
    floors = np.random.choice([1, 1.5, 2, 2.5, 3], n_samples, p=[0.3, 0.15, 0.4, 0.1, 0.05])
    
    # Age and condition
    yr_built = np.random.randint(1900, 2020, n_samples)
    age = 2024 - yr_built
    
    condition = np.random.randint(1, 6, n_samples)  # 1-5 scale
    grade = np.random.randint(4, 12, n_samples)     # 4-11 scale
    
    # Location features
    waterfront = np.random.choice([0, 1], n_samples, p=[0.95, 0.05])
    view = np.random.randint(0, 5, n_samples)
    
    # Lot size
    sqft_lot = np.random.lognormal(9, 0.8, n_samples).astype(int)
    sqft_lot = np.clip(sqft_lot, 1000, 50000)
    
    # Above ground vs basement
    sqft_above = sqft_living * np.random.uniform(0.7, 1.0, n_samples)
    sqft_basement = sqft_living - sqft_above
    sqft_basement = np.clip(sqft_basement, 0, None)
    
    # Price calculation (realistic model)
    price = (
        100 * sqft_living +  # Base price per sqft
        10000 * bedrooms +   # Bedroom premium
        15000 * bathrooms +  # Bathroom premium
        5000 * floors +      # Floor premium
        -500 * age +         # Age depreciation
        10000 * condition +  # Condition adjustment
        15000 * grade +      # Grade premium
        200000 * waterfront + # Waterfront premium
        10000 * view +       # View premium
        2 * sqft_lot +       # Lot size value
        np.random.normal(0, 50000, n_samples)  # Noise
    )
    
    price = np.clip(price, 50000, 2000000)  # Reasonable price range
    
    # Create DataFrame
    df = pd.DataFrame({
        'sqft_living': sqft_living.astype(int),
        'bedrooms': bedrooms,
        'bathrooms': bathrooms.round(1),
        'floors': floors,
        'sqft_lot': sqft_lot,
        'sqft_above': sqft_above.astype(int),
        'sqft_basement': sqft_basement.astype(int),
        'yr_built': yr_built,
        'age': age,
        'condition': condition,
        'grade': grade,
        'waterfront': waterfront,
        'view': view,
        'price': price.astype(int)
    })
    
    if save_csv:
        filepath = os.path.join(os.path.dirname(__file__), 'house_prices.csv')
        df.to_csv(filepath, index=False)
        print(f"   ✅ Saved to: {filepath}")
    
    print(f"   📊 Shape: {df.shape}")
    print(f"   💰 Price range: ${df['price'].min():,} - ${df['price'].max():,}")
    print(f"   📈 Features: {df.columns.tolist()}")
    
    return df

def generate_credit_approval_dataset(n_samples=1000, save_csv=True):
    """
    Generate credit approval dataset (imbalanced classification).
    """
    np.random.seed(42)
    print("\n💳 Generating Credit Approval Dataset...")
    
    # Applicant demographics
    age = np.random.normal(35, 12, n_samples)
    age = np.clip(age, 18, 70).astype(int)
    
    income = np.random.lognormal(10.5, 0.8, n_samples)
    income = np.clip(income, 20000, 300000)
    
    employment_length = np.random.exponential(5, n_samples)
    employment_length = np.clip(employment_length, 0, 40)
    
    # Credit history
    credit_score = np.random.normal(650, 100, n_samples)
    credit_score = np.clip(credit_score, 300, 850).astype(int)
    
    debt_to_income = np.random.beta(2, 5, n_samples) * 0.8  # 0-0.8 ratio
    
    # Loan details
    loan_amount = np.random.uniform(5000, 100000, n_samples)
    loan_purpose = np.random.choice(['home', 'auto', 'personal', 'business'], 
                                   n_samples, p=[0.3, 0.25, 0.3, 0.15])
    
    # Previous defaults
    previous_defaults = np.random.poisson(0.3, n_samples)
    previous_defaults = np.clip(previous_defaults, 0, 5)
    
    # Categorical features
    education = np.random.choice(['High School', 'College', 'Graduate'], 
                                n_samples, p=[0.3, 0.5, 0.2])
    
    marital_status = np.random.choice(['Single', 'Married', 'Divorced'], 
                                     n_samples, p=[0.4, 0.5, 0.1])
    
    # Approval probability (imbalanced - more rejections)
    approval_prob = (
        0.05 +  # Base probability (low)
        0.4 * (credit_score > 700) / 850 +
        0.3 * (income > 50000) / 100000 +
        0.15 * (employment_length > 2) / 10 +
        0.1 * (debt_to_income < 0.3) +
        -0.2 * (previous_defaults > 0) +
        0.05 * (education == 'Graduate') +
        0.05 * (marital_status == 'Married')
    )
    
    approval_prob = np.clip(approval_prob, 0, 1)
    approved = np.random.binomial(1, approval_prob, n_samples)
    
    # Create DataFrame
    df = pd.DataFrame({
        'age': age,
        'income': income.astype(int),
        'employment_length': employment_length.round(1),
        'credit_score': credit_score,
        'debt_to_income': debt_to_income.round(3),
        'loan_amount': loan_amount.astype(int),
        'loan_purpose': loan_purpose,
        'previous_defaults': previous_defaults,
        'education': education,
        'marital_status': marital_status,
        'approved': approved
    })
    
    if save_csv:
        filepath = os.path.join(os.path.dirname(__file__), 'credit_approval.csv')
        df.to_csv(filepath, index=False)
        print(f"   ✅ Saved to: {filepath}")
    
    print(f"   📊 Shape: {df.shape}")
    print(f"   ✅ Approval rate: {approved.mean():.2%}")
    print(f"   📈 Features: {df.columns.tolist()}")
    
    return df

def generate_marketing_campaign_dataset(n_samples=1200, save_csv=True):
    """
    Generate marketing campaign response dataset (multi-class).
    """
    np.random.seed(42)
    print("\n📢 Generating Marketing Campaign Dataset...")
    
    # Customer demographics
    age = np.random.normal(45, 15, n_samples)
    age = np.clip(age, 18, 80).astype(int)
    
    income = np.random.lognormal(10.8, 0.6, n_samples)
    income = np.clip(income, 25000, 250000)
    
    # Purchase history
    recency = np.random.exponential(50, n_samples)  # Days since last purchase
    recency = np.clip(recency, 1, 365).astype(int)
    
    frequency = np.random.poisson(8, n_samples)  # Number of purchases
    frequency = np.clip(frequency, 1, 50)
    
    monetary = np.random.lognormal(7, 1, n_samples)  # Total spent
    monetary = np.clip(monetary, 100, 10000)
    
    # Engagement metrics
    website_visits = np.random.poisson(12, n_samples)
    email_opens = np.random.poisson(5, n_samples)
    social_media_follows = np.random.choice([0, 1], n_samples, p=[0.3, 0.7])
    
    # Campaign exposure
    previous_campaigns = np.random.poisson(3, n_samples)
    previous_campaigns = np.clip(previous_campaigns, 0, 10)
    
    channel_preference = np.random.choice(['email', 'phone', 'sms', 'mail'], 
                                         n_samples, p=[0.4, 0.2, 0.3, 0.1])
    
    # Product categories
    preferred_category = np.random.choice(['electronics', 'clothing', 'books', 'home'], 
                                        n_samples, p=[0.3, 0.25, 0.2, 0.25])
    
    # Response levels (0: No response, 1: Inquired, 2: Purchased)
    # Calculate probabilities for each sample
    response_prob_1 = (
        0.2 +  # Base inquiry probability
        0.1 * (frequency > 10) / 20 +
        0.05 * (recency < 30) / 30 +
        0.05 * (website_visits > 15) / 30 +
        0.05 * social_media_follows
    )
    
    response_prob_2 = (
        0.1 +  # Base purchase probability
        0.15 * (monetary > 2000) / 5000 +
        0.1 * (frequency > 15) / 30 +
        0.05 * (income > 75000) / 100000 +
        0.05 * (email_opens > 8) / 15
    )
    
    # Ensure probabilities are valid
    response_prob_1 = np.clip(response_prob_1, 0, 0.4)
    response_prob_2 = np.clip(response_prob_2, 0, 0.3)
    response_prob_0 = 1 - response_prob_1 - response_prob_2
    response_prob_0 = np.clip(response_prob_0, 0.3, 1)
    
    # Renormalize
    total_prob = response_prob_0 + response_prob_1 + response_prob_2
    response_prob_0 /= total_prob
    response_prob_1 /= total_prob
    response_prob_2 /= total_prob
    
    # Generate response for each sample
    response = np.zeros(n_samples, dtype=int)
    for i in range(n_samples):
        probs = [response_prob_0[i], response_prob_1[i], response_prob_2[i]]
        response[i] = np.random.choice([0, 1, 2], p=probs)
    
    # Create DataFrame
    df = pd.DataFrame({
        'age': age,
        'income': income.astype(int),
        'recency': recency,
        'frequency': frequency,
        'monetary': monetary.astype(int),
        'website_visits': website_visits,
        'email_opens': email_opens,
        'social_media_follows': social_media_follows,
        'previous_campaigns': previous_campaigns,
        'channel_preference': channel_preference,
        'preferred_category': preferred_category,
        'response': response  # 0: No response, 1: Inquired, 2: Purchased
    })
    
    if save_csv:
        filepath = os.path.join(os.path.dirname(__file__), 'marketing_campaign.csv')
        df.to_csv(filepath, index=False)
        print(f"   ✅ Saved to: {filepath}")
    
    print(f"   📊 Shape: {df.shape}")
    print(f"   📈 Response distribution: {np.bincount(response) / len(response)}")
    print(f"   📈 Features: {df.columns.tolist()}")
    
    return df

def generate_employee_satisfaction_dataset(n_samples=800, save_csv=True):
    """
    Generate employee satisfaction dataset for regression.
    """
    np.random.seed(42)
    print("\n👥 Generating Employee Satisfaction Dataset...")
    
    # Employee characteristics
    age = np.random.normal(35, 10, n_samples)
    age = np.clip(age, 22, 65).astype(int)
    
    tenure = np.random.exponential(3, n_samples)
    tenure = np.clip(tenure, 0.1, 20).round(1)
    
    salary = np.random.normal(65000, 20000, n_samples)
    salary = np.clip(salary, 30000, 150000)
    
    # Work characteristics
    hours_per_week = np.random.normal(42, 8, n_samples)
    hours_per_week = np.clip(hours_per_week, 30, 70)
    
    commute_time = np.random.exponential(25, n_samples)  # minutes
    commute_time = np.clip(commute_time, 5, 120)
    
    team_size = np.random.poisson(8, n_samples)
    team_size = np.clip(team_size, 2, 20)
    
    # Benefits and perks
    flexible_hours = np.random.choice([0, 1], n_samples, p=[0.4, 0.6])
    remote_work = np.random.choice([0, 1], n_samples, p=[0.3, 0.7])
    health_benefits = np.random.choice([0, 1], n_samples, p=[0.2, 0.8])
    retirement_plan = np.random.choice([0, 1], n_samples, p=[0.3, 0.7])
    
    # Categorical features
    department = np.random.choice(['Engineering', 'Sales', 'Marketing', 'HR', 'Finance'], 
                                 n_samples, p=[0.3, 0.2, 0.15, 0.1, 0.25])
    
    education = np.random.choice(['High School', 'Bachelor', 'Master', 'PhD'], 
                                n_samples, p=[0.15, 0.5, 0.3, 0.05])
    
    management_level = np.random.choice(['Individual', 'Team Lead', 'Manager', 'Director'], 
                                       n_samples, p=[0.6, 0.2, 0.15, 0.05])
    
    # Satisfaction score (1-10 scale)
    satisfaction = (
        5.0 +  # Base satisfaction
        1.0 * (salary > 70000) +
        0.8 * flexible_hours +
        0.6 * remote_work +
        0.5 * health_benefits +
        0.4 * retirement_plan +
        -0.02 * (hours_per_week - 40) +  # Penalty for long hours
        -0.01 * (commute_time - 20) +    # Penalty for long commute
        0.1 * (tenure > 5) +             # Loyalty bonus
        0.3 * (department == 'Engineering') +  # Department effect
        0.2 * (education == 'Master') +
        0.4 * (management_level != 'Individual') +
        np.random.normal(0, 1.2, n_samples)  # Random variation
    )
    
    satisfaction = np.clip(satisfaction, 1, 10).round(1)
    
    # Create DataFrame
    df = pd.DataFrame({
        'age': age,
        'tenure': tenure,
        'salary': salary.astype(int),
        'hours_per_week': hours_per_week.round(1),
        'commute_time': commute_time.astype(int),
        'team_size': team_size,
        'flexible_hours': flexible_hours,
        'remote_work': remote_work,
        'health_benefits': health_benefits,
        'retirement_plan': retirement_plan,
        'department': department,
        'education': education,
        'management_level': management_level,
        'satisfaction': satisfaction
    })
    
    if save_csv:
        filepath = os.path.join(os.path.dirname(__file__), 'employee_satisfaction.csv')
        df.to_csv(filepath, index=False)
        print(f"   ✅ Saved to: {filepath}")
    
    print(f"   📊 Shape: {df.shape}")
    print(f"   😊 Satisfaction range: {df['satisfaction'].min()} - {df['satisfaction'].max()}")
    print(f"   📈 Features: {df.columns.tolist()}")
    
    return df

def create_dataset_summary():
    """
    Create a summary file describing all generated datasets.
    """
    summary = """
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
"""
    
    filepath = os.path.join(os.path.dirname(__file__), 'README.md')
    with open(filepath, 'w', encoding='utf-8') as f:
        f.write(summary)
    
    print(f"\n📋 Dataset summary created: {filepath}")

def main():
    """
    Generate all sample datasets for Random Forest practice.
    """
    print("🌲 Random Forest Practice Dataset Generator")
    print("=" * 60)
    print("Generating comprehensive datasets for Random Forest learning...")
    
    # Generate all datasets
    customer_churn_df = generate_customer_churn_dataset()
    house_prices_df = generate_house_prices_dataset()
    credit_approval_df = generate_credit_approval_dataset()
    marketing_campaign_df = generate_marketing_campaign_dataset()
    employee_satisfaction_df = generate_employee_satisfaction_dataset()
    
    # Create summary
    create_dataset_summary()
    
    print("\n" + "=" * 60)
    print("✅ All datasets generated successfully!")
    print("\n📊 Dataset Summary:")
    print(f"   🏢 Customer Churn: {customer_churn_df.shape[0]} samples, {customer_churn_df.shape[1]} features")
    print(f"   🏠 House Prices: {house_prices_df.shape[0]} samples, {house_prices_df.shape[1]} features")
    print(f"   💳 Credit Approval: {credit_approval_df.shape[0]} samples, {credit_approval_df.shape[1]} features")
    print(f"   📢 Marketing Campaign: {marketing_campaign_df.shape[0]} samples, {marketing_campaign_df.shape[1]} features")
    print(f"   👥 Employee Satisfaction: {employee_satisfaction_df.shape[0]} samples, {employee_satisfaction_df.shape[1]} features")
    
    print("\n🎯 Next Steps:")
    print("   1. Explore the generated datasets")
    print("   2. Try different Random Forest configurations")
    print("   3. Practice feature engineering and selection")
    print("   4. Compare with other algorithms")
    print("   5. Build complete ML pipelines")
    
    return {
        'customer_churn': customer_churn_df,
        'house_prices': house_prices_df,
        'credit_approval': credit_approval_df,
        'marketing_campaign': marketing_campaign_df,
        'employee_satisfaction': employee_satisfaction_df
    }

if __name__ == "__main__":
    datasets = main()