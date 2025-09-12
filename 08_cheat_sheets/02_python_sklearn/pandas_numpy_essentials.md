# 🐼 Pandas & NumPy Essentials for ML

**🏠 [Cheat Sheets Home](../README.md)** | **🐍 [Scikit-Learn Quick Reference](sklearn_quick_reference.md)** | **🔧 [Data Preprocessing](../05_data_preprocessing/)**

---

## 🎯 Quick Summary

Essential pandas and NumPy operations for machine learning data manipulation, from loading data to feature engineering.

---

## 🚀 Quick Start Imports

```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Display options
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', 100)
```

---

## 📁 Data Loading & Initial Exploration

### **📊 Loading Data**
```python
# CSV files
df = pd.read_csv('data.csv')
df = pd.read_csv('data.csv', index_col=0)  # First column as index
df = pd.read_csv('data.csv', parse_dates=['date_column'])

# Other formats
df = pd.read_excel('data.xlsx', sheet_name='Sheet1')
df = pd.read_json('data.json')
df = pd.read_parquet('data.parquet')

# From URL
df = pd.read_csv('https://example.com/data.csv')

# From SQL
import sqlite3
conn = sqlite3.connect('database.db')
df = pd.read_sql_query("SELECT * FROM table_name", conn)
```

### **🔍 Quick Data Overview**
```python
# Basic info
print(f"Shape: {df.shape}")
print(f"Columns: {list(df.columns)}")
print(f"Data types:\n{df.dtypes}")
print(f"Memory usage: {df.memory_usage(deep=True).sum() / 1024**2:.1f} MB")

# Statistical summary
df.describe()                    # Numerical columns
df.describe(include='all')       # All columns
df.info()                        # Data types and null counts

# First/last rows
df.head(10)                      # First 10 rows
df.tail(5)                       # Last 5 rows
df.sample(10)                    # Random 10 rows
```

### **🚨 Missing Data Analysis**
```python
# Missing data overview
missing_data = df.isnull().sum()
missing_percent = 100 * df.isnull().sum() / len(df)
missing_table = pd.DataFrame({
    'Missing Count': missing_data,
    'Percentage': missing_percent
}).sort_values('Percentage', ascending=False)

print(missing_table[missing_table['Missing Count'] > 0])

# Visualize missing data
import missingno as msno  # pip install missingno
msno.matrix(df)
msno.heatmap(df)
```

---

## 🔍 Data Selection & Filtering

### **📋 Selecting Data**
```python
# Select columns
df['column_name']                # Single column (Series)
df[['col1', 'col2']]            # Multiple columns (DataFrame)
df.loc[:, 'col1':'col3']        # Range of columns
df.select_dtypes(include=['number'])  # By data type

# Select rows
df.loc[0]                       # Single row by label
df.iloc[0]                      # Single row by position
df.loc[0:5]                     # Range of rows
df.head(10)                     # First n rows

# Select both
df.loc[0:5, 'col1':'col3']      # Rows and columns by label
df.iloc[0:5, 0:3]               # Rows and columns by position
```

### **🎯 Filtering Data**
```python
# Simple conditions
df[df['age'] > 25]
df[df['category'] == 'A']
df[df['score'].between(50, 80)]

# Multiple conditions
df[(df['age'] > 25) & (df['score'] > 80)]  # AND
df[(df['category'] == 'A') | (df['category'] == 'B')]  # OR
df[~(df['age'] > 65)]          # NOT (negation)

# String operations
df[df['name'].str.contains('John')]
df[df['email'].str.endswith('.com')]
df[df['text'].str.len() > 10]

# Filtering with isin()
categories_of_interest = ['A', 'B', 'C']
df[df['category'].isin(categories_of_interest)]

# Query method (alternative syntax)
df.query('age > 25 and score > 80')
df.query('category in ["A", "B"]')
```

---

## 🔄 Data Manipulation

### **➕ Adding/Modifying Columns**
```python
# Simple calculations
df['age_squared'] = df['age'] ** 2
df['total_score'] = df['math'] + df['english']
df['average'] = df[['math', 'english', 'science']].mean(axis=1)

# Conditional columns
df['grade'] = df['score'].apply(lambda x: 'A' if x >= 90 else 'B' if x >= 80 else 'C')

# Using np.where (vectorized)
df['pass_fail'] = np.where(df['score'] >= 60, 'Pass', 'Fail')

# Multiple conditions with np.select
conditions = [
    df['score'] >= 90,
    df['score'] >= 80,
    df['score'] >= 70
]
choices = ['A', 'B', 'C']
df['letter_grade'] = np.select(conditions, choices, default='F')

# Map values
grade_mapping = {1: 'Freshman', 2: 'Sophomore', 3: 'Junior', 4: 'Senior'}
df['class_level'] = df['year'].map(grade_mapping)
```

### **🔄 Transforming Data**
```python
# Apply functions
df['log_income'] = df['income'].apply(np.log)
df['name_length'] = df['name'].apply(len)

# Apply to multiple columns
df[['math', 'english']] = df[['math', 'english']].apply(lambda x: x / 100)

# String operations
df['name_upper'] = df['name'].str.upper()
df['first_name'] = df['name'].str.split().str[0]
df['domain'] = df['email'].str.split('@').str[1]

# Date operations
df['date'] = pd.to_datetime(df['date_string'])
df['year'] = df['date'].dt.year
df['month'] = df['date'].dt.month
df['day_of_week'] = df['date'].dt.day_name()
```

---

## 🧮 NumPy Essential Operations

### **📊 Array Creation**
```python
# From lists/data
arr = np.array([1, 2, 3, 4, 5])
arr_2d = np.array([[1, 2], [3, 4]])

# Special arrays
np.zeros(5)                     # [0, 0, 0, 0, 0]
np.ones((2, 3))                 # 2x3 array of ones
np.full((2, 2), 7)              # 2x2 array filled with 7
np.eye(3)                       # 3x3 identity matrix
np.arange(0, 10, 2)             # [0, 2, 4, 6, 8]
np.linspace(0, 1, 5)            # [0, 0.25, 0.5, 0.75, 1]

# Random arrays
np.random.rand(3, 3)            # Random numbers [0,1)
np.random.randn(3, 3)           # Standard normal distribution
np.random.randint(1, 10, size=5) # Random integers
np.random.choice(['A', 'B', 'C'], size=10)  # Random choice
```

### **🔢 Array Operations**
```python
# Basic math
arr + 5                         # Add scalar
arr * 2                         # Multiply by scalar
arr1 + arr2                     # Element-wise addition
arr1 * arr2                     # Element-wise multiplication
arr1 @ arr2                     # Matrix multiplication

# Statistical operations
np.mean(arr)                    # Average
np.median(arr)                  # Median
np.std(arr)                     # Standard deviation
np.min(arr), np.max(arr)        # Min/Max
np.sum(arr)                     # Sum
np.cumsum(arr)                  # Cumulative sum

# Aggregation with axis
arr_2d = np.array([[1, 2, 3], [4, 5, 6]])
np.mean(arr_2d, axis=0)         # Mean along columns
np.mean(arr_2d, axis=1)         # Mean along rows
```

### **🎯 Array Indexing & Slicing**
```python
# 1D arrays
arr[0]                          # First element
arr[-1]                         # Last element
arr[1:4]                        # Elements 1, 2, 3
arr[::2]                        # Every 2nd element

# 2D arrays
arr_2d[0, 1]                    # Row 0, Column 1
arr_2d[:, 1]                    # All rows, Column 1
arr_2d[0, :]                    # Row 0, All columns
arr_2d[arr_2d > 3]              # Boolean indexing

# Fancy indexing
arr[[0, 2, 4]]                  # Select specific indices
arr_2d[[0, 1], [1, 2]]          # Select (0,1) and (1,2)
```

---

## 📊 Aggregation & Grouping

### **🔍 GroupBy Operations**
```python
# Basic grouping
df.groupby('category').mean()
df.groupby('category').size()
df.groupby('category')['score'].mean()

# Multiple columns
df.groupby(['category', 'region']).sum()

# Multiple aggregations
df.groupby('category').agg({
    'score': ['mean', 'std'],
    'age': ['min', 'max'],
    'income': 'sum'
})

# Custom aggregations
df.groupby('category').agg({
    'score': lambda x: x.max() - x.min(),  # Range
    'age': lambda x: x.quantile(0.75)      # 75th percentile
})

# Apply custom functions
def score_summary(group):
    return pd.Series({
        'count': len(group),
        'mean_score': group['score'].mean(),
        'top_scorer': group.loc[group['score'].idxmax(), 'name']
    })

df.groupby('category').apply(score_summary)
```

### **🔄 Pivot Tables & Cross-tabs**
```python
# Pivot table
pivot = df.pivot_table(
    values='score',
    index='category',
    columns='region',
    aggfunc='mean',
    fill_value=0
)

# Multiple values
pivot_multi = df.pivot_table(
    values=['score', 'age'],
    index='category',
    columns='region',
    aggfunc={'score': 'mean', 'age': 'median'}
)

# Cross-tabulation
pd.crosstab(df['category'], df['region'], margins=True)
pd.crosstab(df['category'], df['pass_fail'], normalize='index')  # Percentages
```

---

## 🛠️ Data Cleaning

### **🚨 Handling Missing Values**
```python
# Drop missing values
df.dropna()                     # Drop all rows with any NaN
df.dropna(subset=['important_col'])  # Drop only if specific column is NaN
df.dropna(axis=1)               # Drop columns with any NaN
df.dropna(thresh=5)             # Drop rows with less than 5 non-NaN values

# Fill missing values
df.fillna(0)                    # Fill with 0
df.fillna(df.mean())            # Fill with mean
df['col'].fillna(df['col'].median())  # Fill specific column with median
df.fillna(method='ffill')       # Forward fill
df.fillna(method='bfill')       # Backward fill

# Fill by group
df['score'] = df.groupby('category')['score'].transform(lambda x: x.fillna(x.mean()))

# Interpolation
df['score'].interpolate()       # Linear interpolation
df['score'].interpolate(method='polynomial', order=2)
```

### **🔄 Removing Duplicates**
```python
# Check for duplicates
df.duplicated().sum()           # Count duplicates
df[df.duplicated()]             # Show duplicate rows

# Remove duplicates
df.drop_duplicates()            # Remove all duplicates
df.drop_duplicates(subset=['name'])  # Remove based on specific columns
df.drop_duplicates(keep='last') # Keep last occurrence
```

### **🎯 Data Type Conversions**
```python
# Convert data types
df['age'] = df['age'].astype(int)
df['score'] = df['score'].astype(float)
df['category'] = df['category'].astype('category')  # Categorical

# Convert to datetime
df['date'] = pd.to_datetime(df['date'])
df['date'] = pd.to_datetime(df['date'], format='%Y-%m-%d')

# Convert strings to numbers
df['numeric_str'] = pd.to_numeric(df['string_col'], errors='coerce')  # NaN for invalid

# Boolean conversion
df['is_active'] = df['status'].map({'active': True, 'inactive': False})
```

---

## 📈 Feature Engineering Essentials

### **🔢 Numerical Features**
```python
# Binning/Discretization
df['age_group'] = pd.cut(df['age'], bins=[0, 18, 35, 50, 100], labels=['Teen', 'Young', 'Middle', 'Senior'])
df['score_quartile'] = pd.qcut(df['score'], q=4, labels=['Q1', 'Q2', 'Q3', 'Q4'])

# Scaling (using pandas)
df['score_normalized'] = (df['score'] - df['score'].min()) / (df['score'].max() - df['score'].min())
df['score_standardized'] = (df['score'] - df['score'].mean()) / df['score'].std()

# Log transformation
df['log_income'] = np.log1p(df['income'])  # log(1 + x) to handle zeros

# Polynomial features
df['age_squared'] = df['age'] ** 2
df['age_income_interaction'] = df['age'] * df['income']
```

### **📝 Categorical Features**
```python
# One-hot encoding
dummies = pd.get_dummies(df['category'], prefix='category')
df = pd.concat([df, dummies], axis=1)

# Label encoding (manual)
category_map = {cat: i for i, cat in enumerate(df['category'].unique())}
df['category_encoded'] = df['category'].map(category_map)

# Frequency encoding
freq_map = df['category'].value_counts().to_dict()
df['category_freq'] = df['category'].map(freq_map)

# Target encoding (mean target by category)
target_means = df.groupby('category')['target'].mean()
df['category_target_encoded'] = df['category'].map(target_means)
```

### **📅 Date/Time Features**
```python
df['date'] = pd.to_datetime(df['date'])

# Extract components
df['year'] = df['date'].dt.year
df['month'] = df['date'].dt.month
df['day'] = df['date'].dt.day
df['dayofweek'] = df['date'].dt.dayofweek
df['quarter'] = df['date'].dt.quarter

# Create features
df['is_weekend'] = df['date'].dt.dayofweek.isin([5, 6])
df['is_month_start'] = df['date'].dt.is_month_start
df['days_since_epoch'] = (df['date'] - pd.Timestamp('1970-01-01')).dt.days

# Time differences
df['days_since_last_event'] = (df['current_date'] - df['last_event_date']).dt.days
```

---

## 🎨 Quick Visualization

### **📊 Basic Plots with Pandas**
```python
# Distribution plots
df['score'].hist(bins=30)
df['score'].plot(kind='box')
df['category'].value_counts().plot(kind='bar')

# Relationships
df.plot.scatter(x='age', y='income')
df.groupby('category')['score'].mean().plot(kind='bar')

# Multiple subplots
df[['math', 'english', 'science']].hist(bins=20, figsize=(15, 5))
```

### **🔍 Quick Data Profiling**
```python
# Correlation matrix
corr_matrix = df.select_dtypes(include=[np.number]).corr()
plt.figure(figsize=(10, 8))
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', center=0)

# Distribution overview
df.describe().T.plot(kind='bar', subplots=True, layout=(2, 2), figsize=(15, 10))
```

---

## ⚡ Performance Tips

### **🚀 Speed Optimizations**
```python
# Use vectorized operations instead of loops
# ❌ Slow
df['result'] = df.apply(lambda row: row['a'] + row['b'], axis=1)

# ✅ Fast
df['result'] = df['a'] + df['b']

# Use .loc for setting values
# ❌ Slow
df[df['score'] > 80]['grade'] = 'A'  # Creates a copy

# ✅ Fast
df.loc[df['score'] > 80, 'grade'] = 'A'

# Use categorical data type for repeated strings
df['category'] = df['category'].astype('category')

# Use eval() for complex expressions
df.eval('result = (a + b) * c - d', inplace=True)
```

### **💾 Memory Optimization**
```python
# Optimize data types
def optimize_dtypes(df):
    for col in df.columns:
        if df[col].dtype == 'object':
            try:
                df[col] = pd.to_datetime(df[col])
            except:
                if df[col].nunique() / len(df) < 0.5:  # Less than 50% unique
                    df[col] = df[col].astype('category')
        elif df[col].dtype == 'int64':
            if df[col].min() >= 0 and df[col].max() < 255:
                df[col] = df[col].astype('uint8')
            elif df[col].min() >= -128 and df[col].max() < 127:
                df[col] = df[col].astype('int8')
    return df

df = optimize_dtypes(df)
```

---

## 🔗 Related Cheat Sheets

- **[Scikit-Learn Quick Reference](sklearn_quick_reference.md)** - ML implementation
- **[Data Preprocessing](../05_data_preprocessing/preprocessing_pipeline.md)** - Advanced preprocessing
- **[Data Visualization](../06_visualization/matplotlib_seaborn_essentials.md)** - Plotting and EDA
- **[Feature Engineering Lab](../../06_playground/feature_lab.py)** - Interactive feature engineering

---

## 💡 Pro Tips

### **🎯 Best Practices**
- **Use `.copy()`** when you don't want to modify the original DataFrame
- **Chain operations** for cleaner code: `df.dropna().groupby('category').mean()`
- **Use `pd.options.mode.chained_assignment = None`** to suppress warnings when you know what you're doing
- **Set `parse_dates`** when reading CSV files with dates
- **Use `category` dtype** for strings with limited unique values
- **Profile your data** with `df.info()` and `df.describe()` before starting

### **🚨 Common Pitfalls**
- ❌ Modifying DataFrames during iteration
- ❌ Not using `.copy()` when needed (SettingWithCopyWarning)
- ❌ Forgetting to reset index after filtering/grouping
- ❌ Not handling missing values before mathematical operations
- ❌ Using loops instead of vectorized operations

---

**📋 Print this cheat sheet** for quick offline reference! **🔍 Use Ctrl+F** to find specific operations quickly.

**🏠 [Back to Cheat Sheets](../README.md)** | **🎮 [Try Interactive Tools](../../06_playground/)** | **📊 [Visualization Guide](../06_visualization/)**