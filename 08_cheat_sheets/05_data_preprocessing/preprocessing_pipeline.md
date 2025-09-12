# 🔧 Data Preprocessing Pipeline Cheat Sheet

**🏠 [Cheat Sheets Home](../README.md)** | **🐼 [Pandas Essentials](../02_python_sklearn/pandas_numpy_essentials.md)** | **🤖 [Algorithm Reference](../01_algorithms/)**

---

## 🎯 Quick Summary

Complete data preprocessing workflow from raw data to ML-ready datasets, with code examples and best practices.

---

## ⚡ Essential Imports

```python
# Core libraries
import pandas as pd
import numpy as np

# Preprocessing
from sklearn.preprocessing import (
    StandardScaler, MinMaxScaler, RobustScaler,
    LabelEncoder, OneHotEncoder, TargetEncoder,
    PolynomialFeatures, PowerTransformer
)

# Feature selection
from sklearn.feature_selection import (
    SelectKBest, SelectPercentile, RFE, RFECV,
    VarianceThreshold, mutual_info_classif
)

# Model validation
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

# Visualization
import matplotlib.pyplot as plt
import seaborn as sns
```

---

## 🔄 Complete Preprocessing Workflow

### **📊 1. Data Loading & Initial Exploration**

```python
def initial_data_exploration(df):
    """Quick data exploration summary"""
    print("=== DATASET OVERVIEW ===")
    print(f"Shape: {df.shape}")
    print(f"Memory Usage: {df.memory_usage(deep=True).sum() / 1024**2:.2f} MB")
    
    print("\n=== DATA TYPES ===")
    print(df.dtypes.value_counts())
    
    print("\n=== MISSING VALUES ===")
    missing = df.isnull().sum()
    missing_pct = (missing / len(df)) * 100
    missing_df = pd.DataFrame({
        'Missing': missing,
        'Percentage': missing_pct
    }).sort_values('Percentage', ascending=False)
    print(missing_df[missing_df['Missing'] > 0])
    
    print("\n=== NUMERICAL COLUMNS SUMMARY ===")
    print(df.describe())
    
    print("\n=== CATEGORICAL COLUMNS ===")
    cat_cols = df.select_dtypes(include=['object', 'category']).columns
    for col in cat_cols:
        print(f"{col}: {df[col].nunique()} unique values")
        if df[col].nunique() <= 10:
            print(f"  Values: {df[col].value_counts().to_dict()}")
        print()

# Usage
initial_data_exploration(df)
```

### **🚨 2. Missing Data Analysis & Treatment**

#### **Missing Data Patterns**
```python
def analyze_missing_patterns(df):
    """Analyze missing data patterns"""
    import missingno as msno
    
    # Missing data matrix
    plt.figure(figsize=(15, 6))
    msno.matrix(df)
    plt.title('Missing Data Pattern')
    plt.show()
    
    # Missing data correlation
    plt.figure(figsize=(10, 8))
    msno.heatmap(df)
    plt.title('Missing Data Correlation')
    plt.show()
    
    # Missing data by column
    missing_summary = pd.DataFrame({
        'Column': df.columns,
        'Missing_Count': df.isnull().sum(),
        'Missing_Percentage': (df.isnull().sum() / len(df)) * 100
    }).sort_values('Missing_Percentage', ascending=False)
    
    return missing_summary[missing_summary['Missing_Count'] > 0]

missing_analysis = analyze_missing_patterns(df)
```

#### **Missing Data Imputation Strategies**
```python
def handle_missing_data(df, strategy='auto'):
    """
    Handle missing data with various strategies
    """
    df_processed = df.copy()
    
    # Separate numeric and categorical columns
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    categorical_cols = df.select_dtypes(include=['object', 'category']).columns
    
    # Numeric columns
    for col in numeric_cols:
        missing_pct = (df[col].isnull().sum() / len(df)) * 100
        
        if missing_pct > 0:
            if missing_pct < 5:
                # Low missing: use mean/median
                if df[col].skew() > 1:
                    df_processed[col].fillna(df[col].median(), inplace=True)
                else:
                    df_processed[col].fillna(df[col].mean(), inplace=True)
            elif missing_pct < 20:
                # Medium missing: use interpolation or forward fill
                df_processed[col] = df_processed[col].interpolate()
            else:
                # High missing: create indicator and fill with median
                df_processed[f'{col}_missing'] = df[col].isnull().astype(int)
                df_processed[col].fillna(df[col].median(), inplace=True)
    
    # Categorical columns
    for col in categorical_cols:
        missing_pct = (df[col].isnull().sum() / len(df)) * 100
        
        if missing_pct > 0:
            if missing_pct < 5:
                # Use mode
                df_processed[col].fillna(df[col].mode()[0], inplace=True)
            else:
                # Create 'Unknown' category
                df_processed[col].fillna('Unknown', inplace=True)
    
    return df_processed

# Apply missing data handling
df_clean = handle_missing_data(df)
```

### **🔍 3. Outlier Detection & Treatment**

```python
def detect_outliers(df, method='iqr'):
    """
    Detect outliers using various methods
    """
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    outlier_info = {}
    
    for col in numeric_cols:
        if method == 'iqr':
            Q1 = df[col].quantile(0.25)
            Q3 = df[col].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            
            outliers = df[(df[col] < lower_bound) | (df[col] > upper_bound)][col]
            outlier_info[col] = {
                'count': len(outliers),
                'percentage': len(outliers) / len(df) * 100,
                'lower_bound': lower_bound,
                'upper_bound': upper_bound
            }
        
        elif method == 'zscore':
            from scipy import stats
            z_scores = np.abs(stats.zscore(df[col].dropna()))
            outliers = df[z_scores > 3][col]
            outlier_info[col] = {
                'count': len(outliers),
                'percentage': len(outliers) / len(df) * 100
            }
    
    return outlier_info

def treat_outliers(df, method='cap', threshold=0.05):
    """
    Treat outliers using various methods
    """
    df_treated = df.copy()
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    
    for col in numeric_cols:
        if method == 'cap':
            # Cap at percentiles
            lower_cap = df[col].quantile(threshold)
            upper_cap = df[col].quantile(1 - threshold)
            df_treated[col] = df_treated[col].clip(lower=lower_cap, upper=upper_cap)
        
        elif method == 'remove':
            # Remove extreme outliers (use carefully)
            Q1 = df[col].quantile(0.25)
            Q3 = df[col].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            df_treated = df_treated[(df_treated[col] >= lower_bound) & 
                                  (df_treated[col] <= upper_bound)]
        
        elif method == 'transform':
            # Log transformation for right-skewed data
            if df[col].min() > 0:  # Only if all values are positive
                df_treated[col] = np.log1p(df[col])
    
    return df_treated

# Detect and treat outliers
outlier_info = detect_outliers(df_clean)
df_outliers_treated = treat_outliers(df_clean, method='cap')
```

---

## 🔢 Feature Scaling & Normalization

### **📏 Scaling Methods Comparison**

```python
def compare_scaling_methods(df, target_col=None):
    """
    Compare different scaling methods visually
    """
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    if target_col and target_col in numeric_cols:
        numeric_cols = numeric_cols.drop(target_col)
    
    # Sample a few columns for visualization
    sample_cols = numeric_cols[:3]  # First 3 numeric columns
    
    # Original data
    fig, axes = plt.subplots(3, 4, figsize=(20, 15))
    
    scalers = {
        'Original': None,
        'StandardScaler': StandardScaler(),
        'MinMaxScaler': MinMaxScaler(),
        'RobustScaler': RobustScaler()
    }
    
    for i, col in enumerate(sample_cols):
        for j, (name, scaler) in enumerate(scalers.items()):
            if scaler is None:
                data = df[col]
            else:
                data = scaler.fit_transform(df[[col]]).flatten()
            
            axes[i, j].hist(data, bins=30, alpha=0.7)
            axes[i, j].set_title(f'{col} - {name}')
            axes[i, j].set_xlabel('Value')
            axes[i, j].set_ylabel('Frequency')
    
    plt.tight_layout()
    plt.show()

# Compare scaling methods
compare_scaling_methods(df_outliers_treated)
```

### **⚖️ Choosing the Right Scaler**

```python
def choose_scaler_automatically(df, target_col=None):
    """
    Automatically choose the best scaler for each column
    """
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    if target_col and target_col in numeric_cols:
        numeric_cols = numeric_cols.drop(target_col)
    
    scaler_recommendations = {}
    
    for col in numeric_cols:
        data = df[col].dropna()
        
        # Calculate statistics
        skewness = abs(data.skew())
        outlier_ratio = len(data[(data < data.quantile(0.25) - 1.5 * data.quantile(0.75)) |
                                (data > data.quantile(0.75) + 1.5 * data.quantile(0.25))]) / len(data)
        
        # Recommendation logic
        if outlier_ratio > 0.1:  # High outliers
            recommendation = 'RobustScaler'
        elif skewness > 2:  # Highly skewed
            recommendation = 'PowerTransformer'
        elif data.min() >= 0:  # All positive, need bounded range
            recommendation = 'MinMaxScaler'
        else:  # Default choice
            recommendation = 'StandardScaler'
        
        scaler_recommendations[col] = {
            'recommended_scaler': recommendation,
            'skewness': skewness,
            'outlier_ratio': outlier_ratio * 100
        }
    
    return pd.DataFrame(scaler_recommendations).T

# Get scaler recommendations
scaler_recommendations = choose_scaler_automatically(df_outliers_treated)
print("Scaler Recommendations:")
print(scaler_recommendations)
```

---

## 📝 Categorical Data Encoding

### **🎯 Encoding Strategy Selection**

```python
def analyze_categorical_columns(df):
    """
    Analyze categorical columns to recommend encoding strategy
    """
    cat_cols = df.select_dtypes(include=['object', 'category']).columns
    analysis = {}
    
    for col in cat_cols:
        unique_count = df[col].nunique()
        mode_frequency = df[col].value_counts().iloc[0] / len(df)
        
        # Recommend encoding strategy
        if unique_count == 2:
            recommendation = 'Binary/Label Encoding'
        elif unique_count <= 10:
            recommendation = 'One-Hot Encoding'
        elif unique_count <= 50:
            recommendation = 'Target Encoding'
        else:
            recommendation = 'Frequency/Target Encoding'
        
        analysis[col] = {
            'unique_values': unique_count,
            'mode_frequency': mode_frequency,
            'recommended_encoding': recommendation,
            'sample_values': list(df[col].value_counts().head().index)
        }
    
    return pd.DataFrame(analysis).T

# Analyze categorical columns
cat_analysis = analyze_categorical_columns(df_outliers_treated)
print("Categorical Encoding Recommendations:")
print(cat_analysis)
```

### **🔄 Encoding Implementation**

```python
def encode_categorical_features(df, target_col=None):
    """
    Apply appropriate encoding to categorical features
    """
    df_encoded = df.copy()
    cat_cols = df.select_dtypes(include=['object', 'category']).columns
    
    encoding_applied = {}
    
    for col in cat_cols:
        unique_count = df[col].nunique()
        
        if unique_count == 2:
            # Binary encoding
            le = LabelEncoder()
            df_encoded[col] = le.fit_transform(df[col].astype(str))
            encoding_applied[col] = 'LabelEncoder'
        
        elif unique_count <= 10:
            # One-hot encoding
            dummies = pd.get_dummies(df[col], prefix=col, drop_first=True)
            df_encoded = pd.concat([df_encoded.drop(col, axis=1), dummies], axis=1)
            encoding_applied[col] = f'OneHotEncoder ({len(dummies.columns)} features)'
        
        elif unique_count <= 50 and target_col is not None:
            # Target encoding
            target_means = df.groupby(col)[target_col].mean()
            df_encoded[f'{col}_target_encoded'] = df[col].map(target_means)
            df_encoded.drop(col, axis=1, inplace=True)
            encoding_applied[col] = 'TargetEncoder'
        
        else:
            # Frequency encoding
            freq_map = df[col].value_counts().to_dict()
            df_encoded[f'{col}_frequency'] = df[col].map(freq_map)
            df_encoded.drop(col, axis=1, inplace=True)
            encoding_applied[col] = 'FrequencyEncoder'
    
    print("Encoding Applied:")
    for col, encoding in encoding_applied.items():
        print(f"  {col}: {encoding}")
    
    return df_encoded, encoding_applied

# Apply categorical encoding
df_encoded, encoding_info = encode_categorical_features(df_outliers_treated, target_col='target')
```

---

## 🎯 Feature Engineering

### **📊 Automated Feature Creation**

```python
def create_engineered_features(df, target_col=None):
    """
    Create new features automatically
    """
    df_featured = df.copy()
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    if target_col and target_col in numeric_cols:
        numeric_cols = numeric_cols.drop(target_col)
    
    # 1. Polynomial features (degree 2) for highly correlated pairs
    if len(numeric_cols) >= 2:
        corr_matrix = df[numeric_cols].corr().abs()
        
        # Find highly correlated pairs
        high_corr_pairs = []
        for i in range(len(corr_matrix.columns)):
            for j in range(i+1, len(corr_matrix.columns)):
                if 0.3 < corr_matrix.iloc[i, j] < 0.95:  # Moderate correlation
                    high_corr_pairs.append((corr_matrix.columns[i], corr_matrix.columns[j]))
        
        # Create interaction features for top pairs
        for col1, col2 in high_corr_pairs[:5]:  # Limit to top 5 pairs
            df_featured[f'{col1}_x_{col2}'] = df[col1] * df[col2]
            df_featured[f'{col1}_div_{col2}'] = df[col1] / (df[col2] + 1e-8)  # Avoid division by zero
    
    # 2. Binning continuous variables
    for col in numeric_cols[:3]:  # Limit to first 3 columns
        if df[col].nunique() > 20:  # Only bin if many unique values
            df_featured[f'{col}_binned'] = pd.qcut(df[col], q=5, labels=['Low', 'Low-Med', 'Med', 'Med-High', 'High'], duplicates='drop')
    
    # 3. Statistical aggregations if there are grouping columns
    cat_cols = df.select_dtypes(include=['object', 'category']).columns
    if len(cat_cols) > 0 and len(numeric_cols) > 0:
        # Group by first categorical column and create aggregations
        group_col = cat_cols[0]
        target_num_col = numeric_cols[0]
        
        group_stats = df.groupby(group_col)[target_num_col].agg(['mean', 'std', 'median']).reset_index()
        group_stats.columns = [group_col, f'{group_col}_{target_num_col}_mean', 
                              f'{group_col}_{target_num_col}_std', f'{group_col}_{target_num_col}_median']
        
        df_featured = df_featured.merge(group_stats, on=group_col, how='left')
    
    print(f"Original features: {df.shape[1]}")
    print(f"After feature engineering: {df_featured.shape[1]}")
    print(f"New features created: {df_featured.shape[1] - df.shape[1]}")
    
    return df_featured

# Create engineered features
df_featured = create_engineered_features(df_encoded)
```

### **🔍 Feature Selection**

```python
def feature_selection_pipeline(X, y, max_features=50):
    """
    Comprehensive feature selection pipeline
    """
    from sklearn.feature_selection import (
        VarianceThreshold, SelectKBest, f_classif, mutual_info_classif,
        RFE, RandomForestClassifier
    )
    
    print("=== FEATURE SELECTION PIPELINE ===")
    print(f"Starting with {X.shape[1]} features")
    
    # 1. Remove low-variance features
    variance_selector = VarianceThreshold(threshold=0.01)
    X_variance = variance_selector.fit_transform(X)
    selected_variance = variance_selector.get_support()
    print(f"After variance filtering: {X_variance.shape[1]} features")
    
    # 2. Univariate feature selection
    k_best = min(max_features, X_variance.shape[1])
    univariate_selector = SelectKBest(score_func=f_classif, k=k_best)
    X_univariate = univariate_selector.fit_transform(X_variance, y)
    print(f"After univariate selection: {X_univariate.shape[1]} features")
    
    # 3. Recursive feature elimination
    rf = RandomForestClassifier(n_estimators=50, random_state=42)
    rfe_features = min(max_features//2, X_univariate.shape[1])
    rfe_selector = RFE(rf, n_features_to_select=rfe_features)
    X_rfe = rfe_selector.fit_transform(X_univariate, y)
    print(f"After RFE: {X_rfe.shape[1]} features")
    
    # Get final selected feature names
    feature_names = X.columns
    
    # Apply all selections in sequence
    variance_mask = selected_variance
    univariate_mask = univariate_selector.get_support()
    rfe_mask = rfe_selector.get_support()
    
    # Combine masks
    final_mask = np.zeros(len(feature_names), dtype=bool)
    variance_idx = 0
    for i, include_var in enumerate(variance_mask):
        if include_var:
            univariate_idx = variance_idx
            if univariate_mask[univariate_idx]:
                rfe_idx = np.sum(univariate_mask[:univariate_idx+1]) - 1
                if rfe_mask[rfe_idx]:
                    final_mask[i] = True
            variance_idx += 1
    
    selected_features = feature_names[final_mask]
    
    print(f"\nFinal selected features ({len(selected_features)}):")
    for feature in selected_features:
        print(f"  - {feature}")
    
    return X[selected_features], selected_features

# Apply feature selection (assuming we have a target variable)
if 'target' in df_featured.columns:
    X = df_featured.drop('target', axis=1)
    y = df_featured['target']
    X_selected, selected_features = feature_selection_pipeline(X, y)
```

---

## 🏗️ Complete Preprocessing Pipeline

### **🔧 Production Pipeline**

```python
def create_preprocessing_pipeline(numeric_features, categorical_features, 
                                target_col=None, scale_method='auto'):
    """
    Create a complete preprocessing pipeline using sklearn Pipeline
    """
    from sklearn.compose import ColumnTransformer
    from sklearn.pipeline import Pipeline
    from sklearn.impute import SimpleImputer
    
    # Numeric preprocessing
    if scale_method == 'standard':
        numeric_transformer = Pipeline([
            ('imputer', SimpleImputer(strategy='median')),
            ('scaler', StandardScaler())
        ])
    elif scale_method == 'minmax':
        numeric_transformer = Pipeline([
            ('imputer', SimpleImputer(strategy='median')),
            ('scaler', MinMaxScaler())
        ])
    else:  # robust
        numeric_transformer = Pipeline([
            ('imputer', SimpleImputer(strategy='median')),
            ('scaler', RobustScaler())
        ])
    
    # Categorical preprocessing
    categorical_transformer = Pipeline([
        ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
        ('onehot', OneHotEncoder(handle_unknown='ignore', drop='first'))
    ])
    
    # Combine preprocessing steps
    preprocessor = ColumnTransformer([
        ('num', numeric_transformer, numeric_features),
        ('cat', categorical_transformer, categorical_features)
    ])
    
    return preprocessor

# Example usage
numeric_features = df_featured.select_dtypes(include=[np.number]).columns.drop('target')
categorical_features = df_featured.select_dtypes(include=['object', 'category']).columns

preprocessing_pipeline = create_preprocessing_pipeline(
    numeric_features, categorical_features, scale_method='robust'
)

# Fit and transform
X = df_featured.drop('target', axis=1)
y = df_featured['target']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
X_train_processed = preprocessing_pipeline.fit_transform(X_train)
X_test_processed = preprocessing_pipeline.transform(X_test)

print(f"Original shape: {X_train.shape}")
print(f"Processed shape: {X_train_processed.shape}")
```

---

## 📋 Preprocessing Checklist

### **✅ Pre-Processing Checklist**

```python
def preprocessing_quality_check(df_original, df_processed, target_col):
    """
    Comprehensive quality check for preprocessing
    """
    print("=== PREPROCESSING QUALITY CHECK ===\n")
    
    # 1. Shape comparison
    print("📊 SHAPE COMPARISON:")
    print(f"Original: {df_original.shape}")
    print(f"Processed: {df_processed.shape}")
    print(f"Features added: {df_processed.shape[1] - df_original.shape[1]}\n")
    
    # 2. Missing values
    print("🚨 MISSING VALUES CHECK:")
    missing_original = df_original.isnull().sum().sum()
    missing_processed = df_processed.isnull().sum().sum()
    print(f"Original missing values: {missing_original}")
    print(f"Processed missing values: {missing_processed}")
    if missing_processed == 0:
        print("✅ All missing values handled")
    else:
        print("❌ Missing values still present")
    print()
    
    # 3. Data types
    print("📝 DATA TYPES:")
    print("Original dtypes:")
    print(df_original.dtypes.value_counts())
    print("\nProcessed dtypes:")
    print(df_processed.dtypes.value_counts())
    print()
    
    # 4. Target distribution (if applicable)
    if target_col in df_original.columns and target_col in df_processed.columns:
        print("🎯 TARGET DISTRIBUTION:")
        print("Original target distribution:")
        print(df_original[target_col].value_counts(normalize=True).sort_index())
        print("\nProcessed target distribution:")
        print(df_processed[target_col].value_counts(normalize=True).sort_index())
        print()
    
    # 5. Feature correlation with target
    if target_col in df_processed.columns:
        numeric_cols = df_processed.select_dtypes(include=[np.number]).columns
        if len(numeric_cols) > 1:
            target_corr = df_processed[numeric_cols].corr()[target_col].abs().sort_values(ascending=False)
            print("🔗 TOP FEATURES CORRELATED WITH TARGET:")
            print(target_corr.head(10))
            print()
    
    # 6. Memory usage
    print("💾 MEMORY USAGE:")
    memory_original = df_original.memory_usage(deep=True).sum() / 1024**2
    memory_processed = df_processed.memory_usage(deep=True).sum() / 1024**2
    print(f"Original memory: {memory_original:.2f} MB")
    print(f"Processed memory: {memory_processed:.2f} MB")
    print(f"Memory change: {((memory_processed - memory_original) / memory_original * 100):+.1f}%")
    
    return {
        'shape_original': df_original.shape,
        'shape_processed': df_processed.shape,
        'missing_handled': missing_processed == 0,
        'memory_original_mb': memory_original,
        'memory_processed_mb': memory_processed
    }

# Run quality check
quality_report = preprocessing_quality_check(df, df_featured, 'target')
```

---

## 🎯 Quick Preprocessing Templates

### **🚀 Fast Track Templates**

```python
# Template 1: Basic Cleaning
def quick_clean(df):
    """Quick and dirty preprocessing for rapid prototyping"""
    # Handle missing values
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    categorical_cols = df.select_dtypes(include=['object']).columns
    
    # Fill numeric with median, categorical with mode
    for col in numeric_cols:
        df[col].fillna(df[col].median(), inplace=True)
    
    for col in categorical_cols:
        df[col].fillna(df[col].mode()[0] if len(df[col].mode()) > 0 else 'Unknown', inplace=True)
    
    # Simple encoding
    for col in categorical_cols:
        if df[col].nunique() <= 10:
            df = pd.get_dummies(df, columns=[col], drop_first=True)
        else:
            le = LabelEncoder()
            df[col] = le.fit_transform(df[col])
    
    return df

# Template 2: Competition Ready
def competition_preprocessing(df, target_col):
    """Comprehensive preprocessing for competitions"""
    # Advanced missing value handling
    df = handle_missing_data(df)
    
    # Outlier treatment
    df = treat_outliers(df, method='cap', threshold=0.01)
    
    # Feature engineering
    df = create_engineered_features(df, target_col)
    
    # Encoding
    df, _ = encode_categorical_features(df, target_col)
    
    # Feature selection
    if target_col in df.columns:
        X = df.drop(target_col, axis=1)
        y = df[target_col]
        X_selected, _ = feature_selection_pipeline(X, y)
        df = pd.concat([X_selected, y], axis=1)
    
    return df

# Template 3: Production Pipeline
class ProductionPreprocessor:
    def __init__(self):
        self.numeric_imputer = SimpleImputer(strategy='median')
        self.categorical_imputer = SimpleImputer(strategy='constant', fill_value='Unknown')
        self.scaler = RobustScaler()
        self.encoder = OneHotEncoder(handle_unknown='ignore', drop='first')
        self.feature_selector = SelectKBest(k=20)
        
    def fit(self, X, y=None):
        numeric_cols = X.select_dtypes(include=[np.number]).columns
        categorical_cols = X.select_dtypes(include=['object']).columns
        
        # Fit transformers
        if len(numeric_cols) > 0:
            X_num_imputed = self.numeric_imputer.fit_transform(X[numeric_cols])
            X_num_scaled = self.scaler.fit_transform(X_num_imputed)
        
        if len(categorical_cols) > 0:
            X_cat_imputed = self.categorical_imputer.fit_transform(X[categorical_cols])
            X_cat_encoded = self.encoder.fit_transform(X_cat_imputed)
        
        # Combine and fit feature selector
        if len(numeric_cols) > 0 and len(categorical_cols) > 0:
            X_combined = np.hstack([X_num_scaled, X_cat_encoded.toarray()])
        elif len(numeric_cols) > 0:
            X_combined = X_num_scaled
        else:
            X_combined = X_cat_encoded.toarray()
        
        if y is not None:
            self.feature_selector.fit(X_combined, y)
        
        return self
    
    def transform(self, X):
        numeric_cols = X.select_dtypes(include=[np.number]).columns
        categorical_cols = X.select_dtypes(include=['object']).columns
        
        # Transform
        if len(numeric_cols) > 0:
            X_num_imputed = self.numeric_imputer.transform(X[numeric_cols])
            X_num_scaled = self.scaler.transform(X_num_imputed)
        
        if len(categorical_cols) > 0:
            X_cat_imputed = self.categorical_imputer.transform(X[categorical_cols])
            X_cat_encoded = self.encoder.transform(X_cat_imputed)
        
        # Combine
        if len(numeric_cols) > 0 and len(categorical_cols) > 0:
            X_combined = np.hstack([X_num_scaled, X_cat_encoded.toarray()])
        elif len(numeric_cols) > 0:
            X_combined = X_num_scaled
        else:
            X_combined = X_cat_encoded.toarray()
        
        # Feature selection
        return self.feature_selector.transform(X_combined)
```

---

## 🔗 Related Cheat Sheets

- **[Pandas & NumPy Essentials](../02_python_sklearn/pandas_numpy_essentials.md)** - Data manipulation foundations
- **[Scikit-Learn Quick Reference](../02_python_sklearn/sklearn_quick_reference.md)** - ML implementation syntax
- **[Feature Engineering Lab](../../06_playground/feature_lab.py)** - Interactive feature engineering
- **[Essential Math & Statistics](../03_math_statistics/essential_formulas.md)** - Mathematical foundations

---

## 💡 Key Takeaways

### **🎯 Best Practices**
1. **Always explore data first** - Understand before transforming
2. **Handle missing data thoughtfully** - Don't just drop everything
3. **Choose scalers based on data distribution** - No one-size-fits-all
4. **Encode categoricals appropriately** - Consider cardinality
5. **Create meaningful features** - Domain knowledge is key
6. **Use pipelines** - Prevent data leakage and ensure reproducibility

### **🚨 Common Pitfalls**
- ❌ Scaling before train/test split (data leakage)
- ❌ Dropping too many rows with missing values
- ❌ Using inappropriate encoding for high-cardinality categoricals
- ❌ Not handling unseen categories in test data
- ❌ Creating too many features without selection
- ❌ Ignoring the target variable during preprocessing

---

**📋 Keep this pipeline handy** for consistent preprocessing! **🔍 Use Ctrl+F** to find specific techniques quickly.

**🏠 [Back to Cheat Sheets](../README.md)** | **🎮 [Try Feature Lab](../../06_playground/feature_lab.py)** | **🤖 [Algorithm Selection](../01_algorithms/algorithm_selection_flowchart.md)**