"""
Sample Dataset Generator for Logistic Regression

This module generates various datasets suitable for practicing
logistic regression concepts.

Author: ML Learning Repository
Date: 2024
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import make_classification, make_blobs
from sklearn.preprocessing import StandardScaler
import os


def generate_binary_classification_dataset(n_samples=1000, n_features=2, 
                                         noise=0.1, random_state=42):
    """Generate binary classification dataset"""
    X, y = make_classification(
        n_samples=n_samples,
        n_features=n_features,
        n_redundant=0,
        n_informative=n_features,
        n_clusters_per_class=1,
        random_state=random_state
    )
    
    # Add some noise
    X += np.random.normal(0, noise, X.shape)
    
    return X, y


def generate_linearly_separable_data(n_samples=500, random_state=42):
    """Generate perfectly linearly separable data"""
    np.random.seed(random_state)
    
    # Generate two well-separated clusters
    X1 = np.random.multivariate_normal([2, 2], [[0.5, 0.1], [0.1, 0.5]], n_samples//2)
    X2 = np.random.multivariate_normal([-1, -1], [[0.5, -0.1], [-0.1, 0.5]], n_samples//2)
    
    X = np.vstack((X1, X2))
    y = np.hstack((np.ones(n_samples//2), np.zeros(n_samples//2)))
    
    # Shuffle
    indices = np.random.permutation(n_samples)
    return X[indices], y[indices]


def generate_non_linearly_separable_data(n_samples=500, random_state=42):
    """Generate data that's not linearly separable"""
    np.random.seed(random_state)
    
    # Generate circular pattern
    theta = np.random.uniform(0, 2*np.pi, n_samples//2)
    r1 = np.random.normal(1, 0.1, n_samples//2)
    r2 = np.random.normal(2.5, 0.2, n_samples//2)
    
    # Inner circle (class 0)
    X1 = np.column_stack((r1 * np.cos(theta), r1 * np.sin(theta)))
    # Outer circle (class 1)
    X2 = np.column_stack((r2 * np.cos(theta), r2 * np.sin(theta)))
    
    X = np.vstack((X1, X2))
    y = np.hstack((np.zeros(n_samples//2), np.ones(n_samples//2)))
    
    # Shuffle
    indices = np.random.permutation(n_samples)
    return X[indices], y[indices]


def generate_medical_diagnosis_dataset(n_samples=1000, random_state=42):
    """Generate synthetic medical diagnosis dataset"""
    np.random.seed(random_state)
    
    # Features: age, BMI, blood_pressure, cholesterol, glucose
    age = np.random.normal(50, 15, n_samples)
    age = np.clip(age, 18, 90)
    
    bmi = np.random.normal(25, 5, n_samples)
    bmi = np.clip(bmi, 15, 45)
    
    bp = np.random.normal(120, 20, n_samples)
    bp = np.clip(bp, 90, 180)
    
    cholesterol = np.random.normal(200, 40, n_samples)
    cholesterol = np.clip(cholesterol, 150, 300)
    
    glucose = np.random.normal(90, 15, n_samples)
    glucose = np.clip(glucose, 70, 150)
    
    # Create target based on logical rules
    risk_score = (
        0.02 * (age - 40) +
        0.1 * (bmi - 25) +
        0.01 * (bp - 120) +
        0.005 * (cholesterol - 200) +
        0.02 * (glucose - 90)
    )
    
    # Add some randomness
    risk_score += np.random.normal(0, 1, n_samples)
    
    # Convert to binary classification
    y = (risk_score > 0).astype(int)
    
    X = np.column_stack((age, bmi, bp, cholesterol, glucose))
    
    # Create DataFrame with meaningful names
    feature_names = ['age', 'bmi', 'blood_pressure', 'cholesterol', 'glucose']
    df = pd.DataFrame(X, columns=feature_names)
    df['diagnosis'] = y
    
    return df


def generate_marketing_dataset(n_samples=800, random_state=42):
    """Generate customer conversion dataset"""
    np.random.seed(random_state)
    
    # Features: age, income, website_visits, email_opens, time_on_site
    age = np.random.normal(35, 12, n_samples)
    age = np.clip(age, 18, 65)
    
    income = np.random.lognormal(10, 0.5, n_samples)
    income = np.clip(income, 20000, 150000)
    
    website_visits = np.random.poisson(3, n_samples)
    website_visits = np.clip(website_visits, 0, 15)
    
    email_opens = np.random.poisson(2, n_samples)
    email_opens = np.clip(email_opens, 0, 10)
    
    time_on_site = np.random.gamma(2, 2, n_samples)
    time_on_site = np.clip(time_on_site, 0, 20)
    
    # Create conversion probability
    conversion_prob = (
        -3 +
        0.01 * (age - 35) +
        0.00002 * (income - 50000) +
        0.2 * website_visits +
        0.3 * email_opens +
        0.1 * time_on_site +
        np.random.normal(0, 0.5, n_samples)
    )
    
    # Convert to probabilities and then binary
    conversion_prob = 1 / (1 + np.exp(-conversion_prob))
    y = np.random.binomial(1, conversion_prob)
    
    X = np.column_stack((age, income, website_visits, email_opens, time_on_site))
    
    feature_names = ['age', 'income', 'website_visits', 'email_opens', 'time_on_site']
    df = pd.DataFrame(X, columns=feature_names)
    df['converted'] = y
    
    return df


def save_datasets():
    """Generate and save all sample datasets"""
    output_dir = os.path.dirname(os.path.abspath(__file__))
    
    print("Generating sample datasets for Logistic Regression...")
    
    # 1. Basic binary classification
    X1, y1 = generate_binary_classification_dataset()
    df1 = pd.DataFrame(X1, columns=['feature1', 'feature2'])
    df1['target'] = y1
    df1.to_csv(os.path.join(output_dir, 'binary_classification.csv'), index=False)
    print("✅ Created binary_classification.csv")
    
    # 2. Linearly separable
    X2, y2 = generate_linearly_separable_data()
    df2 = pd.DataFrame(X2, columns=['feature1', 'feature2'])
    df2['target'] = y2
    df2.to_csv(os.path.join(output_dir, 'linearly_separable.csv'), index=False)
    print("✅ Created linearly_separable.csv")
    
    # 3. Non-linearly separable
    X3, y3 = generate_non_linearly_separable_data()
    df3 = pd.DataFrame(X3, columns=['feature1', 'feature2'])
    df3['target'] = y3
    df3.to_csv(os.path.join(output_dir, 'non_linearly_separable.csv'), index=False)
    print("✅ Created non_linearly_separable.csv")
    
    # 4. Medical diagnosis
    df4 = generate_medical_diagnosis_dataset()
    df4.to_csv(os.path.join(output_dir, 'medical_diagnosis.csv'), index=False)
    print("✅ Created medical_diagnosis.csv")
    
    # 5. Marketing conversion
    df5 = generate_marketing_dataset()
    df5.to_csv(os.path.join(output_dir, 'marketing_conversion.csv'), index=False)
    print("✅ Created marketing_conversion.csv")
    
    return {
        'binary_classification': df1,
        'linearly_separable': df2,
        'non_linearly_separable': df3,
        'medical_diagnosis': df4,
        'marketing_conversion': df5
    }


def visualize_datasets():
    """Visualize all generated datasets"""
    datasets = save_datasets()
    
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    axes = axes.ravel()
    
    # Binary classification
    df = datasets['binary_classification']
    axes[0].scatter(df[df['target']==0]['feature1'], df[df['target']==0]['feature2'], 
                   c='red', alpha=0.6, label='Class 0')
    axes[0].scatter(df[df['target']==1]['feature1'], df[df['target']==1]['feature2'], 
                   c='blue', alpha=0.6, label='Class 1')
    axes[0].set_title('Binary Classification')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # Linearly separable
    df = datasets['linearly_separable']
    axes[1].scatter(df[df['target']==0]['feature1'], df[df['target']==0]['feature2'], 
                   c='red', alpha=0.6, label='Class 0')
    axes[1].scatter(df[df['target']==1]['feature1'], df[df['target']==1]['feature2'], 
                   c='blue', alpha=0.6, label='Class 1')
    axes[1].set_title('Linearly Separable')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    # Non-linearly separable
    df = datasets['non_linearly_separable']
    axes[2].scatter(df[df['target']==0]['feature1'], df[df['target']==0]['feature2'], 
                   c='red', alpha=0.6, label='Class 0')
    axes[2].scatter(df[df['target']==1]['feature1'], df[df['target']==1]['feature2'], 
                   c='blue', alpha=0.6, label='Class 1')
    axes[2].set_title('Non-Linearly Separable')
    axes[2].legend()
    axes[2].grid(True, alpha=0.3)
    
    # Medical diagnosis
    df = datasets['medical_diagnosis']
    axes[3].scatter(df[df['diagnosis']==0]['age'], df[df['diagnosis']==0]['bmi'], 
                   c='green', alpha=0.6, label='Healthy')
    axes[3].scatter(df[df['diagnosis']==1]['age'], df[df['diagnosis']==1]['bmi'], 
                   c='orange', alpha=0.6, label='At Risk')
    axes[3].set_xlabel('Age')
    axes[3].set_ylabel('BMI')
    axes[3].set_title('Medical Diagnosis (Age vs BMI)')
    axes[3].legend()
    axes[3].grid(True, alpha=0.3)
    
    # Marketing conversion
    df = datasets['marketing_conversion']
    axes[4].scatter(df[df['converted']==0]['income'], df[df['converted']==0]['website_visits'], 
                   c='gray', alpha=0.6, label='No Conversion')
    axes[4].scatter(df[df['converted']==1]['income'], df[df['converted']==1]['website_visits'], 
                   c='purple', alpha=0.6, label='Converted')
    axes[4].set_xlabel('Income')
    axes[4].set_ylabel('Website Visits')
    axes[4].set_title('Marketing Conversion')
    axes[4].legend()
    axes[4].grid(True, alpha=0.3)
    
    # Hide the last subplot
    axes[5].set_visible(False)
    
    plt.tight_layout()
    plt.suptitle('Sample Datasets for Logistic Regression Practice', 
                 fontsize=16, y=0.98)
    plt.show()
    
    # Display dataset summaries
    print("\n" + "="*60)
    print("DATASET SUMMARIES")
    print("="*60)
    
    for name, df in datasets.items():
        print(f"\n{name.upper()}:")
        print(f"  Shape: {df.shape}")
        target_col = 'target' if 'target' in df.columns else ('diagnosis' if 'diagnosis' in df.columns else 'converted')
        print(f"  Class distribution: {df[target_col].value_counts().to_dict()}")
        print(f"  Features: {[col for col in df.columns if col != target_col]}")


if __name__ == "__main__":
    # Generate and visualize datasets
    visualize_datasets()
    
    print("\n" + "="*60)
    print("All datasets generated successfully!")
    print("Use these datasets to practice logistic regression concepts.")
    print("="*60)