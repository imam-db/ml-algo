# Seaborn Quick Reference for ML

## Table of Contents
- [Basic Setup & Import](#basic-setup--import)
- [Distribution Plots](#distribution-plots)
- [Relationship Plots](#relationship-plots)
- [Categorical Plots](#categorical-plots)
- [Matrix Plots](#matrix-plots)
- [Grid Plots](#grid-plots)
- [Statistical Estimation](#statistical-estimation)
- [Styling & Customization](#styling--customization)

---

## Basic Setup & Import

```python
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

# Set style and palette
sns.set_style("whitegrid")  # whitegrid, darkgrid, white, dark, ticks
sns.set_palette("husl")     # husl, deep, muted, bright, pastel, dark, colorblind
sns.set_context("notebook") # paper, notebook, talk, poster

# Figure size
plt.figure(figsize=(10, 6))
```

## Distribution Plots

### Histogram and KDE
```python
# Basic histogram
sns.histplot(data=df, x='feature', bins=30, kde=True)
plt.title('Distribution of Feature')

# Multiple distributions
sns.histplot(data=df, x='feature', hue='target', kde=True, alpha=0.7)

# KDE only
sns.kdeplot(data=df, x='feature', hue='target', fill=True, alpha=0.6)

# Rug plot (marginal ticks)
sns.histplot(data=df, x='feature', kde=True)
sns.rugplot(data=df, x='feature', height=0.02)
```

### Distribution Comparison
```python
# Box plot
sns.boxplot(data=df, x='category', y='value')
plt.xticks(rotation=45)

# Violin plot (combines boxplot + KDE)
sns.violinplot(data=df, x='category', y='value', inner='quart')

# Strip plot (scatter with categories)
sns.stripplot(data=df, x='category', y='value', jitter=True, alpha=0.7)

# Swarm plot (non-overlapping points)
sns.swarmplot(data=df, x='category', y='value', size=4)
```

## Relationship Plots

### Scatter Plots
```python
# Basic scatter
sns.scatterplot(data=df, x='feature1', y='feature2', hue='target', 
                size='feature3', style='category', alpha=0.7)

# Regression plot
sns.regplot(data=df, x='feature1', y='feature2', scatter_kws={'alpha':0.6})

# Multiple regression lines
sns.lmplot(data=df, x='feature1', y='feature2', hue='target', 
           col='category', height=4)
```

### Line Plots
```python
# Time series or ordered data
sns.lineplot(data=df, x='time', y='value', hue='group', 
             marker='o', markersize=8)

# Error bands (confidence intervals)
sns.lineplot(data=df, x='epoch', y='loss', hue='model', 
             err_style='band', ci=95)
```

## Categorical Plots

### Bar Plots
```python
# Count plot
sns.countplot(data=df, x='category', hue='target')
plt.xticks(rotation=45)

# Bar plot (with aggregation)
sns.barplot(data=df, x='category', y='value', hue='target', 
            ci=95, capsize=0.05)

# Point plot (for trends)
sns.pointplot(data=df, x='category', y='value', hue='target', 
              join=True, markers=['o', 's'])
```

### Advanced Categorical
```python
# Factor plot (versatile categorical plotting)
sns.catplot(data=df, x='category', y='value', hue='target',
            kind='box', col='condition', height=4, aspect=1.2)

# Available kinds: strip, swarm, box, violin, boxen, point, bar, count
```

## Matrix Plots

### Correlation Heatmap
```python
# Correlation matrix
corr = df.corr()
plt.figure(figsize=(10, 8))
sns.heatmap(corr, annot=True, cmap='coolwarm', center=0,
            square=True, linewidths=0.5, cbar_kws={'shrink': 0.8})
plt.title('Correlation Matrix')

# Custom colormap and formatting
mask = np.triu(np.ones_like(corr, dtype=bool))  # Mask upper triangle
sns.heatmap(corr, mask=mask, annot=True, fmt='.2f', 
            cmap='RdBu_r', vmin=-1, vmax=1)
```

### Pivot Tables and Clustermap
```python
# Heatmap from pivot table
pivot_table = df.pivot_table(values='score', index='model', 
                             columns='dataset', aggfunc='mean')
sns.heatmap(pivot_table, annot=True, fmt='.3f', cmap='YlOrRd')

# Clustermap (with hierarchical clustering)
sns.clustermap(df.corr(), cmap='coolwarm', center=0,
               linewidths=0.5, figsize=(10, 10))
```

## Grid Plots

### Pair Plots
```python
# Pairwise relationships
sns.pairplot(df, hue='target', diag_kind='kde', 
             plot_kws={'alpha': 0.7}, diag_kws={'alpha': 0.7})

# Custom pair plot
g = sns.PairGrid(df, hue='target')
g.map_diag(sns.histplot, alpha=0.7)
g.map_upper(sns.scatterplot, alpha=0.7)
g.map_lower(sns.kdeplot, alpha=0.7)
g.add_legend()
```

### Facet Grids
```python
# Custom grid
g = sns.FacetGrid(df, col='category', row='condition', 
                  height=4, aspect=1.2)
g.map(sns.scatterplot, 'x', 'y', alpha=0.7)
g.add_legend()

# With different plot types
g = sns.FacetGrid(df, col='model', hue='target', height=4)
g.map(sns.histplot, 'score', alpha=0.7)
g.add_legend()
```

## Statistical Estimation

### Regression Plots
```python
# Linear regression with confidence interval
sns.regplot(data=df, x='feature', y='target', 
            scatter_kws={'alpha': 0.6}, line_kws={'color': 'red'})

# Polynomial regression
sns.regplot(data=df, x='feature', y='target', order=2,
            scatter_kws={'alpha': 0.6})

# Robust regression
sns.regplot(data=df, x='feature', y='target', robust=True,
            scatter_kws={'alpha': 0.6})
```

### Joint Distributions
```python
# Joint plot (scatter + marginal distributions)
sns.jointplot(data=df, x='feature1', y='feature2', kind='scatter',
              height=8, marginal_kws={'bins': 30})

# Different joint plot types
kinds = ['scatter', 'reg', 'resid', 'kde', 'hex']
for kind in kinds:
    sns.jointplot(data=df, x='feature1', y='feature2', kind=kind)
```

## Styling & Customization

### Themes and Palettes
```python
# Built-in styles
styles = ['darkgrid', 'whitegrid', 'dark', 'white', 'ticks']
for style in styles:
    sns.set_style(style)

# Color palettes
palettes = ['deep', 'muted', 'bright', 'pastel', 'dark', 'colorblind']
sns.set_palette('husl', n_colors=8)

# Custom palette
custom_colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4', '#FFEAA7']
sns.set_palette(custom_colors)

# Diverging palette
sns.diverging_palette(220, 20, as_cmap=True)
```

### Context and Scaling
```python
# Context for different outputs
contexts = ['paper', 'notebook', 'talk', 'poster']
sns.set_context('talk', font_scale=1.2, rc={'lines.linewidth': 2})

# Custom parameters
sns.set_theme(style='whitegrid', 
              rc={'figure.figsize': (10, 6),
                  'axes.titlesize': 16,
                  'axes.labelsize': 14})
```

## ML-Specific Patterns

### Model Performance Visualization
```python
# Learning curves comparison
melted_df = pd.melt(learning_curves_df, id_vars=['train_size'], 
                    value_vars=['train_score', 'val_score'],
                    var_name='score_type', value_name='score')

sns.lineplot(data=melted_df, x='train_size', y='score', 
             hue='score_type', marker='o', markersize=8)
plt.title('Learning Curves')
plt.xlabel('Training Set Size')
plt.ylabel('Score')
```

### Feature Analysis
```python
# Feature importance plot
importance_df = pd.DataFrame({
    'feature': feature_names,
    'importance': model.feature_importances_
}).sort_values('importance', ascending=True)

plt.figure(figsize=(10, 8))
sns.barplot(data=importance_df.tail(15), y='feature', x='importance')
plt.title('Top 15 Feature Importance')

# Feature correlation with target
feature_corr = df.corr()['target'].abs().sort_values(ascending=True)
plt.figure(figsize=(8, 10))
sns.barplot(x=feature_corr.values, y=feature_corr.index)
plt.title('Feature Correlation with Target')
```

### Model Comparison
```python
# Model performance comparison
results_df = pd.DataFrame({
    'model': ['RF', 'SVM', 'XGB', 'LR'],
    'accuracy': [0.85, 0.82, 0.88, 0.80],
    'precision': [0.83, 0.80, 0.86, 0.78],
    'recall': [0.87, 0.84, 0.90, 0.82]
})

# Melt for visualization
melted_results = pd.melt(results_df, id_vars=['model'], 
                         var_name='metric', value_name='score')

sns.barplot(data=melted_results, x='model', y='score', hue='metric')
plt.title('Model Performance Comparison')
plt.ylim(0, 1)

# Or use grouped bar plot
fig, ax = plt.subplots(figsize=(10, 6))
x = np.arange(len(results_df))
width = 0.25

metrics = ['accuracy', 'precision', 'recall']
for i, metric in enumerate(metrics):
    ax.bar(x + i*width, results_df[metric], width, label=metric)

ax.set_xlabel('Models')
ax.set_ylabel('Scores')
ax.set_title('Model Performance Comparison')
ax.set_xticks(x + width)
ax.set_xticklabels(results_df['model'])
ax.legend()
```

### Residual Analysis
```python
# Residuals plot for regression
residuals_df = pd.DataFrame({
    'predicted': y_pred,
    'residuals': y_true - y_pred,
    'actual': y_true
})

# Residuals vs predicted
plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
sns.scatterplot(data=residuals_df, x='predicted', y='residuals', alpha=0.7)
plt.axhline(y=0, color='red', linestyle='--')
plt.title('Residuals vs Predicted')

# Q-Q plot using seaborn
plt.subplot(1, 2, 2)
from scipy import stats
stats.probplot(residuals_df['residuals'], dist="norm", plot=plt)
plt.title('Q-Q Plot')
plt.tight_layout()
```

### Confusion Matrix Styling
```python
from sklearn.metrics import confusion_matrix

# Stylized confusion matrix
cm = confusion_matrix(y_true, y_pred)
plt.figure(figsize=(8, 6))

sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=class_names, yticklabels=class_names,
            cbar_kws={'label': 'Count'})
plt.title('Confusion Matrix')
plt.xlabel('Predicted')
plt.ylabel('Actual')

# Normalized confusion matrix
cm_norm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
sns.heatmap(cm_norm, annot=True, fmt='.2f', cmap='Blues',
            xticklabels=class_names, yticklabels=class_names)
```

## Advanced Techniques

### Statistical Annotations
```python
from scipy import stats

# Add statistical test results
g = sns.boxplot(data=df, x='group', y='score')
pairs = [('A', 'B'), ('A', 'C'), ('B', 'C')]

for i, (group1, group2) in enumerate(pairs):
    data1 = df[df['group'] == group1]['score']
    data2 = df[df['group'] == group2]['score']
    stat, p_value = stats.ttest_ind(data1, data2)
    
    # Add significance annotation
    if p_value < 0.05:
        plt.annotate(f'p = {p_value:.3f}*', 
                     xy=(i+0.5, max(df['score']) + 0.1))
```

### Multi-level Grouping
```python
# Complex grouping visualization
plt.figure(figsize=(12, 8))
sns.boxplot(data=df, x='algorithm', y='accuracy', hue='dataset')
plt.xticks(rotation=45)
plt.title('Algorithm Performance Across Datasets')

# Add swarm plot overlay
sns.swarmplot(data=df, x='algorithm', y='accuracy', hue='dataset',
              size=3, alpha=0.7, dodge=True)
```

---

## Quick Tips

1. **Use `sns.despine()`** to remove chart borders for cleaner look
2. **Set figure size with `plt.figure(figsize=(w, h))`** before seaborn plots
3. **Use `hue` parameter** for categorical color encoding
4. **Use `col` and `row`** in FacetGrid for subplots
5. **Combine seaborn with matplotlib** for fine-tuned control
6. **Use `alpha` parameter** for transparency with overlapping data
7. **Rotate x-labels** with `plt.xticks(rotation=45)` when needed
8. **Save plots** with `plt.savefig('plot.png', dpi=300, bbox_inches='tight')`

## Common Parameters

```python
# Color parameters
hue='category'          # Color by category
palette='viridis'       # Color palette
color='blue'           # Single color

# Size parameters  
size='continuous_var'   # Size by continuous variable
sizes=(50, 200)        # Size range

# Style parameters
style='category'        # Marker style by category
markers=['o', 's', '^'] # Custom markers

# Transparency
alpha=0.7              # Transparency level

# Error representation
ci=95                  # Confidence interval
err_style='band'       # Error style: band or bars
capsize=0.05          # Error bar cap size
```