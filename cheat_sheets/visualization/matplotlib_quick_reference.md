# Matplotlib Quick Reference for ML

## Table of Contents
- [Basic Setup & Import](#basic-setup--import)
- [Data Visualization](#data-visualization)
- [Model Visualization](#model-visualization)
- [Statistical Plots](#statistical-plots)
- [Advanced Customization](#advanced-customization)
- [Common ML Patterns](#common-ml-patterns)

---

## Basic Setup & Import

```python
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import ListedColormap
import seaborn as sns

# Set style and figure size
plt.style.use('seaborn-v0_8')  # or 'default', 'ggplot'
plt.rcParams['figure.figsize'] = (10, 6)
plt.rcParams['font.size'] = 12
```

## Data Visualization

### Basic Plots
```python
# Line plot
plt.plot(x, y, label='Data', linewidth=2, color='blue')
plt.xlabel('X Label')
plt.ylabel('Y Label')
plt.title('Title')
plt.legend()
plt.grid(True, alpha=0.3)
plt.show()

# Scatter plot
plt.scatter(x, y, c=colors, s=50, alpha=0.7, cmap='viridis')
plt.colorbar(label='Color Scale')

# Histogram
plt.hist(data, bins=30, alpha=0.7, density=True, color='skyblue')
plt.axvline(np.mean(data), color='red', linestyle='--', label='Mean')
```

### Multiple Subplots
```python
fig, axes = plt.subplots(2, 2, figsize=(12, 10))
axes = axes.ravel()  # Flatten for easy iteration

for i, ax in enumerate(axes):
    ax.plot(x, y[i])
    ax.set_title(f'Plot {i+1}')
    ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()
```

## Model Visualization

### Decision Boundaries (2D)
```python
def plot_decision_boundary(X, y, model, resolution=0.02):
    # Create mesh
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, resolution),
                         np.arange(y_min, y_max, resolution))
    
    # Predict on mesh
    mesh_points = np.c_[xx.ravel(), yy.ravel()]
    Z = model.predict(mesh_points)
    Z = Z.reshape(xx.shape)
    
    # Plot
    plt.contourf(xx, yy, Z, alpha=0.3, cmap='viridis')
    scatter = plt.scatter(X[:, 0], X[:, 1], c=y, cmap='viridis', edgecolors='black')
    plt.colorbar(scatter)
    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')
    plt.title('Decision Boundary')
```

### Learning Curves
```python
def plot_learning_curves(train_sizes, train_scores, val_scores):
    train_mean = np.mean(train_scores, axis=1)
    train_std = np.std(train_scores, axis=1)
    val_mean = np.mean(val_scores, axis=1)
    val_std = np.std(val_scores, axis=1)
    
    plt.figure(figsize=(10, 6))
    plt.plot(train_sizes, train_mean, 'o-', color='blue', label='Training Score')
    plt.fill_between(train_sizes, train_mean - train_std, 
                     train_mean + train_std, alpha=0.1, color='blue')
    
    plt.plot(train_sizes, val_mean, 'o-', color='red', label='Validation Score')
    plt.fill_between(train_sizes, val_mean - val_std, 
                     val_mean + val_std, alpha=0.1, color='red')
    
    plt.xlabel('Training Set Size')
    plt.ylabel('Score')
    plt.title('Learning Curves')
    plt.legend()
    plt.grid(True, alpha=0.3)
```

### Validation Curves
```python
def plot_validation_curve(param_range, train_scores, val_scores, param_name):
    train_mean = np.mean(train_scores, axis=1)
    train_std = np.std(train_scores, axis=1)
    val_mean = np.mean(val_scores, axis=1)
    val_std = np.std(val_scores, axis=1)
    
    plt.figure(figsize=(10, 6))
    plt.semilogx(param_range, train_mean, 'o-', color='blue', label='Training Score')
    plt.fill_between(param_range, train_mean - train_std, 
                     train_mean + train_std, alpha=0.1, color='blue')
    
    plt.semilogx(param_range, val_mean, 'o-', color='red', label='Validation Score')
    plt.fill_between(param_range, val_mean - val_std, 
                     val_mean + val_std, alpha=0.1, color='red')
    
    plt.xlabel(param_name)
    plt.ylabel('Score')
    plt.title(f'Validation Curve - {param_name}')
    plt.legend()
    plt.grid(True, alpha=0.3)
```

## Statistical Plots

### Distribution Analysis
```python
# Multiple histograms
fig, ax = plt.subplots(1, 2, figsize=(15, 5))

# Histogram with KDE
ax[0].hist(data, bins=30, density=True, alpha=0.7, color='skyblue')
kde_x = np.linspace(data.min(), data.max(), 100)
kde_y = gaussian_kde(data)(kde_x)
ax[0].plot(kde_x, kde_y, 'r-', linewidth=2, label='KDE')
ax[0].set_title('Histogram with KDE')
ax[0].legend()

# Box plot
ax[1].boxplot([group1, group2, group3], labels=['A', 'B', 'C'])
ax[1].set_title('Box Plot Comparison')
```

### Correlation Matrix
```python
def plot_correlation_matrix(df, figsize=(10, 8)):
    corr_matrix = df.corr()
    
    plt.figure(figsize=figsize)
    im = plt.imshow(corr_matrix, cmap='coolwarm', aspect='auto', vmin=-1, vmax=1)
    plt.colorbar(im)
    
    # Add text annotations
    for i in range(len(corr_matrix.columns)):
        for j in range(len(corr_matrix.columns)):
            plt.text(j, i, f'{corr_matrix.iloc[i, j]:.2f}',
                    ha='center', va='center', color='black')
    
    plt.xticks(range(len(corr_matrix.columns)), corr_matrix.columns, rotation=45)
    plt.yticks(range(len(corr_matrix.columns)), corr_matrix.columns)
    plt.title('Correlation Matrix')
    plt.tight_layout()
```

### ROC Curve
```python
def plot_roc_curves(models_dict, X_test, y_test):
    plt.figure(figsize=(10, 8))
    
    for name, model in models_dict.items():
        y_pred_proba = model.predict_proba(X_test)[:, 1]
        fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
        auc_score = auc(fpr, tpr)
        
        plt.plot(fpr, tpr, linewidth=2, label=f'{name} (AUC = {auc_score:.3f})')
    
    plt.plot([0, 1], [0, 1], 'k--', linewidth=1, label='Random Classifier')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curves Comparison')
    plt.legend()
    plt.grid(True, alpha=0.3)
```

## Advanced Customization

### Color Maps and Styling
```python
# Custom colormap
colors = ['red', 'orange', 'yellow', 'green', 'blue']
n_bins = 100
cmap = LinearSegmentedColormap.from_list('custom', colors, N=n_bins)

# Style customization
plt.rcParams.update({
    'font.size': 14,
    'axes.titlesize': 16,
    'axes.labelsize': 14,
    'xtick.labelsize': 12,
    'ytick.labelsize': 12,
    'legend.fontsize': 12,
    'lines.linewidth': 2,
    'lines.markersize': 8
})
```

### Annotations and Text
```python
# Add annotations
plt.annotate('Important Point', 
             xy=(x_point, y_point), 
             xytext=(x_text, y_text),
             arrowprops=dict(arrowstyle='->', color='red', lw=2),
             fontsize=12, color='red',
             bbox=dict(boxstyle='round,pad=0.3', facecolor='yellow', alpha=0.7))

# Add text box
textstr = 'Statistics:\nMean = %.2f\nStd = %.2f' % (mean_val, std_val)
props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
plt.text(0.05, 0.95, textstr, transform=plt.gca().transAxes, fontsize=10,
         verticalalignment='top', bbox=props)
```

## Common ML Patterns

### Feature Importance Plot
```python
def plot_feature_importance(model, feature_names, top_n=10):
    importance = model.feature_importances_
    indices = np.argsort(importance)[::-1][:top_n]
    
    plt.figure(figsize=(10, 6))
    plt.bar(range(top_n), importance[indices], color='skyblue', alpha=0.7)
    plt.xticks(range(top_n), [feature_names[i] for i in indices], rotation=45)
    plt.xlabel('Features')
    plt.ylabel('Importance')
    plt.title(f'Top {top_n} Feature Importance')
    plt.tight_layout()
```

### Residual Plots
```python
def plot_residuals(y_true, y_pred):
    residuals = y_true - y_pred
    
    fig, axes = plt.subplots(1, 2, figsize=(15, 5))
    
    # Residual vs Predicted
    axes[0].scatter(y_pred, residuals, alpha=0.7)
    axes[0].axhline(y=0, color='red', linestyle='--')
    axes[0].set_xlabel('Predicted Values')
    axes[0].set_ylabel('Residuals')
    axes[0].set_title('Residuals vs Predicted')
    axes[0].grid(True, alpha=0.3)
    
    # Q-Q plot
    from scipy import stats
    stats.probplot(residuals, dist="norm", plot=axes[1])
    axes[1].set_title('Q-Q Plot')
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
```

### Confusion Matrix Heatmap
```python
def plot_confusion_matrix(cm, class_names, normalize=False):
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        fmt = '.2f'
    else:
        fmt = 'd'
    
    plt.figure(figsize=(8, 6))
    plt.imshow(cm, interpolation='nearest', cmap='Blues')
    plt.title('Confusion Matrix')
    plt.colorbar()
    
    tick_marks = np.arange(len(class_names))
    plt.xticks(tick_marks, class_names, rotation=45)
    plt.yticks(tick_marks, class_names)
    
    # Add text annotations
    thresh = cm.max() / 2
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                horizontalalignment="center",
                color="white" if cm[i, j] > thresh else "black")
    
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
```

### Interactive Plots (with widgets)
```python
# Requires ipywidgets in Jupyter
from ipywidgets import interact, FloatSlider

def interactive_function_plot():
    @interact(a=FloatSlider(min=-5, max=5, step=0.1, value=1),
              b=FloatSlider(min=-5, max=5, step=0.1, value=0))
    def plot_function(a, b):
        x = np.linspace(-10, 10, 100)
        y = a * x + b
        
        plt.figure(figsize=(10, 6))
        plt.plot(x, y, linewidth=2)
        plt.grid(True, alpha=0.3)
        plt.xlabel('x')
        plt.ylabel('y')
        plt.title(f'y = {a:.1f}x + {b:.1f}')
        plt.ylim(-20, 20)
        plt.show()
```

---

## Quick Tips

1. **Always use `plt.tight_layout()`** to prevent overlapping elements
2. **Set figure size early**: `plt.figure(figsize=(width, height))`
3. **Use alpha for transparency**: `alpha=0.7` for overlapping elements
4. **Grid for readability**: `plt.grid(True, alpha=0.3)`
5. **Save high-quality figures**: `plt.savefig('plot.png', dpi=300, bbox_inches='tight')`
6. **Close figures to save memory**: `plt.close()` or `plt.close('all')`

## Color References

```python
# Common colors
colors = ['red', 'blue', 'green', 'orange', 'purple', 'brown', 'pink', 'gray']

# Color maps for ML
- Classification: 'viridis', 'plasma', 'Set1', 'tab10'
- Heatmaps: 'coolwarm', 'RdYlBu', 'RdBu'
- Sequential: 'Blues', 'Reds', 'YlOrRd'
```