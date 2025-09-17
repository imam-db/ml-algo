# Plotly Quick Reference for Interactive ML Visualization

## Table of Contents
- [Basic Setup & Import](#basic-setup--import)
- [Basic Plots](#basic-plots)
- [Statistical Charts](#statistical-charts)
- [3D Visualizations](#3d-visualizations)
- [Interactive Features](#interactive-features)
- [Subplots & Layouts](#subplots--layouts)
- [ML-Specific Visualizations](#ml-specific-visualizations)
- [Animation & Time Series](#animation--time-series)
- [Customization & Styling](#customization--styling)

---

## Basic Setup & Import

```python
import plotly.graph_objects as go
import plotly.express as px
import plotly.figure_factory as ff
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np

# For Jupyter notebooks
import plotly.offline as pyo
pyo.init_notebook_mode()

# Configuration
import plotly.io as pio
pio.templates.default = "plotly_white"  # or "plotly_dark", "ggplot2", "seaborn"
```

## Basic Plots

### Scatter Plots
```python
# Basic scatter with Plotly Express
fig = px.scatter(df, x='feature1', y='feature2', 
                 color='target', size='feature3',
                 hover_data=['feature4', 'feature5'],
                 title='Interactive Scatter Plot')
fig.show()

# Advanced scatter with Graph Objects
fig = go.Figure()
for category in df['category'].unique():
    data = df[df['category'] == category]
    fig.add_trace(go.Scatter(
        x=data['x'], y=data['y'],
        mode='markers',
        name=category,
        marker=dict(size=10, opacity=0.7),
        hovertemplate='<b>%{fullData.name}</b><br>' +
                      'X: %{x}<br>' +
                      'Y: %{y}<br>' +
                      '<extra></extra>'
    ))
fig.update_layout(title='Custom Scatter Plot', 
                  xaxis_title='Feature 1',
                  yaxis_title='Feature 2')
fig.show()
```

### Line Plots
```python
# Time series with multiple lines
fig = px.line(df, x='date', y='value', 
              color='group', title='Time Series')
fig.update_traces(line=dict(width=3))
fig.update_layout(hovermode='x unified')
fig.show()

# Learning curves
fig = go.Figure()
fig.add_trace(go.Scatter(x=train_sizes, y=train_scores_mean,
                         mode='lines+markers', name='Training Score',
                         line=dict(color='blue', width=3),
                         error_y=dict(type='data', array=train_scores_std)))

fig.add_trace(go.Scatter(x=train_sizes, y=val_scores_mean,
                         mode='lines+markers', name='Validation Score',
                         line=dict(color='red', width=3),
                         error_y=dict(type='data', array=val_scores_std)))

fig.update_layout(title='Learning Curves',
                  xaxis_title='Training Set Size',
                  yaxis_title='Score')
fig.show()
```

### Bar Charts
```python
# Basic bar chart
fig = px.bar(df, x='category', y='value', 
             color='subcategory', title='Bar Chart')
fig.show()

# Grouped bar chart for model comparison
models = ['Random Forest', 'SVM', 'XGBoost', 'Logistic Regression']
metrics = ['Accuracy', 'Precision', 'Recall', 'F1-Score']
values = np.random.rand(len(models), len(metrics))

fig = go.Figure()
for i, metric in enumerate(metrics):
    fig.add_trace(go.Bar(
        name=metric,
        x=models,
        y=values[:, i],
        marker_color=px.colors.qualitative.Set1[i]
    ))

fig.update_layout(barmode='group', title='Model Performance Comparison')
fig.show()
```

## Statistical Charts

### Histograms & Distributions
```python
# Interactive histogram
fig = px.histogram(df, x='feature', color='target',
                   marginal='box', # or 'violin', 'rug'
                   title='Distribution Analysis')
fig.show()

# Overlaid histograms
fig = go.Figure()
for category in df['category'].unique():
    fig.add_trace(go.Histogram(
        x=df[df['category'] == category]['value'],
        name=category,
        opacity=0.7,
        nbinsx=30
    ))
fig.update_layout(barmode='overlay', title='Overlaid Histograms')
fig.show()

# Density heatmap
fig = go.Figure(go.Histogram2d(
    x=df['feature1'],
    y=df['feature2'],
    colorscale='Blues',
    showscale=True
))
fig.update_layout(title='2D Density Plot')
fig.show()
```

### Box Plots & Violin Plots
```python
# Interactive box plot
fig = px.box(df, x='category', y='value', 
             color='subcategory', points='all',
             title='Box Plot with Points')
fig.show()

# Violin plot
fig = px.violin(df, x='category', y='value',
                box=True, points='all',
                title='Violin Plot')
fig.show()

# Combined box and strip plot
fig = go.Figure()
for category in df['category'].unique():
    data = df[df['category'] == category]['value']
    fig.add_trace(go.Box(y=data, name=category, boxpoints='all'))

fig.update_layout(title='Box Plot with All Points')
fig.show()
```

### Correlation Heatmap
```python
# Interactive correlation matrix
corr_matrix = df.corr()
fig = px.imshow(corr_matrix,
                text_auto=True,
                aspect="auto",
                color_continuous_scale='RdBu_r',
                title='Interactive Correlation Matrix')
fig.show()

# Custom heatmap with Graph Objects
fig = go.Figure(data=go.Heatmap(
    z=corr_matrix.values,
    x=corr_matrix.columns,
    y=corr_matrix.columns,
    colorscale='RdBu_r',
    zmid=0,
    text=corr_matrix.values,
    texttemplate='%{text:.2f}',
    textfont={"size": 10},
    hovertemplate='%{y} vs %{x}<br>Correlation: %{z:.3f}<extra></extra>'
))
fig.update_layout(title='Correlation Matrix')
fig.show()
```

## 3D Visualizations

### 3D Scatter Plot
```python
# 3D scatter plot
fig = px.scatter_3d(df, x='feature1', y='feature2', z='feature3',
                    color='target', size='feature4',
                    title='3D Scatter Plot')
fig.show()

# Custom 3D scatter with Graph Objects
fig = go.Figure(data=[go.Scatter3d(
    x=df['x'], y=df['y'], z=df['z'],
    mode='markers',
    marker=dict(
        size=5,
        color=df['target'],
        colorscale='Viridis',
        showscale=True,
        opacity=0.8
    ),
    text=df['label'],
    hovertemplate='X: %{x}<br>Y: %{y}<br>Z: %{z}<br>Label: %{text}<extra></extra>'
)])
fig.update_layout(title='Custom 3D Scatter',
                  scene=dict(
                      xaxis_title='Feature 1',
                      yaxis_title='Feature 2',
                      zaxis_title='Feature 3'
                  ))
fig.show()
```

### 3D Surface Plot
```python
# 3D surface (e.g., for loss landscapes)
x = np.linspace(-2, 2, 50)
y = np.linspace(-2, 2, 50)
X, Y = np.meshgrid(x, y)
Z = np.sin(X) * np.cos(Y)  # Replace with actual loss function

fig = go.Figure(data=[go.Surface(x=X, y=Y, z=Z, colorscale='Viridis')])
fig.update_layout(title='Loss Landscape',
                  scene=dict(
                      xaxis_title='Parameter 1',
                      yaxis_title='Parameter 2',
                      zaxis_title='Loss'
                  ))
fig.show()
```

## Interactive Features

### Dropdown Menus
```python
# Dropdown to switch between features
fig = go.Figure()

# Add traces for different features
features = ['feature1', 'feature2', 'feature3']
for i, feature in enumerate(features):
    fig.add_trace(go.Scatter(
        x=df.index, y=df[feature],
        name=feature,
        visible=(i == 0)  # Only first trace visible initially
    ))

# Create dropdown buttons
buttons = []
for i, feature in enumerate(features):
    visibility = [False] * len(features)
    visibility[i] = True
    buttons.append(dict(
        label=feature.title(),
        method='update',
        args=[{'visible': visibility},
              {'title': f'Time Series - {feature.title()}'}]
    ))

fig.update_layout(
    updatemenus=[dict(
        active=0,
        buttons=buttons,
        x=1.15, y=1
    )],
    title='Interactive Feature Selection'
)
fig.show()
```

### Slider for Animation
```python
# Animated scatter plot over time
fig = px.scatter(df, x='feature1', y='feature2',
                 animation_frame='time_period',
                 color='target', size='feature3',
                 title='Animated Scatter Plot')
fig.show()

# Custom slider for parameter tuning
param_values = np.linspace(0.1, 2.0, 20)
fig = go.Figure()

for i, param in enumerate(param_values):
    # Generate data based on parameter (replace with actual model predictions)
    y = np.sin(df['x'] * param)
    fig.add_trace(go.Scatter(
        x=df['x'], y=y,
        visible=(i == 0),
        name=f'Parameter = {param:.2f}'
    ))

# Create slider
steps = []
for i, param in enumerate(param_values):
    step = dict(
        method='update',
        args=[{'visible': [False] * len(param_values)},
              {'title': f'Model Output (Parameter = {param:.2f})'}],
        label=f'{param:.2f}'
    )
    step['args'][0]['visible'][i] = True
    steps.append(step)

sliders = [dict(
    active=0,
    currentvalue={'prefix': 'Parameter: '},
    steps=steps
)]

fig.update_layout(sliders=sliders, title='Parameter Tuning Visualization')
fig.show()
```

### Hover Information
```python
# Rich hover information
fig = px.scatter(df, x='feature1', y='feature2',
                 color='target',
                 hover_data={'feature3': ':.2f',
                            'feature4': True,
                            'feature1': False},  # Hide feature1 from hover
                 custom_data=['additional_info'])

fig.update_traces(
    hovertemplate='<b>Target: %{marker.color}</b><br>' +
                  'Feature 1: %{x}<br>' +
                  'Feature 2: %{y}<br>' +
                  'Feature 3: %{customdata[0]}<br>' +
                  '<extra></extra>'
)
fig.show()
```

## Subplots & Layouts

### Multiple Subplots
```python
# Create subplots
fig = make_subplots(
    rows=2, cols=2,
    subplot_titles=('Scatter', 'Line', 'Bar', 'Histogram'),
    specs=[[{'type': 'scatter'}, {'type': 'scatter'}],
           [{'type': 'bar'}, {'type': 'histogram'}]]
)

# Add traces to subplots
fig.add_trace(go.Scatter(x=df['x'], y=df['y'], mode='markers'), row=1, col=1)
fig.add_trace(go.Scatter(x=df.index, y=df['feature1'], mode='lines'), row=1, col=2)
fig.add_trace(go.Bar(x=df['category'], y=df['value']), row=2, col=1)
fig.add_trace(go.Histogram(x=df['feature2']), row=2, col=2)

fig.update_layout(title='Multiple Visualizations', showlegend=False)
fig.show()

# Shared axes
fig = make_subplots(rows=2, cols=1, shared_xaxes=True,
                    vertical_spacing=0.1)
fig.add_trace(go.Scatter(x=df['date'], y=df['price'], name='Price'), row=1, col=1)
fig.add_trace(go.Bar(x=df['date'], y=df['volume'], name='Volume'), row=2, col=1)
fig.update_layout(title='Price and Volume')
fig.show()
```

### Dashboard Layout
```python
# Create a dashboard-like layout
from plotly.subplots import make_subplots

fig = make_subplots(
    rows=3, cols=3,
    specs=[[{'colspan': 2}, None, {'rowspan': 2}],
           [{'colspan': 2}, None, None],
           [{}, {}, {}]],
    subplot_titles=('Main Chart', 'Side Info', 'Chart 1', 'Chart 2', 'Chart 3')
)

# Add various charts
fig.add_trace(go.Scatter(x=df['x'], y=df['y'], mode='markers'), row=1, col=1)
fig.add_trace(go.Bar(x=['A', 'B', 'C'], y=[1, 2, 3]), row=1, col=3)
fig.add_trace(go.Histogram(x=df['feature1']), row=2, col=1)
fig.add_trace(go.Box(y=df['feature2']), row=3, col=1)
fig.add_trace(go.Scatter(x=df['x'], y=df['z'], mode='lines'), row=3, col=2)
fig.add_trace(go.Pie(labels=['A', 'B', 'C'], values=[1, 2, 3]), row=3, col=3)

fig.update_layout(title='ML Dashboard', showlegend=False, height=700)
fig.show()
```

## ML-Specific Visualizations

### Decision Boundary Visualization
```python
def plot_decision_boundary_plotly(X, y, model, resolution=0.02):
    # Create mesh
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, resolution),
                         np.arange(y_min, y_max, resolution))
    
    # Predict on mesh
    mesh_points = np.c_[xx.ravel(), yy.ravel()]
    Z = model.predict(mesh_points)
    Z = Z.reshape(xx.shape)
    
    # Create contour plot for decision boundary
    fig = go.Figure()
    
    fig.add_trace(go.Contour(
        x=np.arange(x_min, x_max, resolution),
        y=np.arange(y_min, y_max, resolution),
        z=Z,
        showscale=False,
        opacity=0.3,
        colorscale='RdYlBu'
    ))
    
    # Add data points
    for class_val in np.unique(y):
        mask = y == class_val
        fig.add_trace(go.Scatter(
            x=X[mask, 0], y=X[mask, 1],
            mode='markers',
            name=f'Class {class_val}',
            marker=dict(size=8, line=dict(width=1, color='black'))
        ))
    
    fig.update_layout(title='Decision Boundary Visualization',
                      xaxis_title='Feature 1',
                      yaxis_title='Feature 2')
    return fig
```

### ROC Curve Comparison
```python
def plot_roc_curves_plotly(models_results):
    fig = go.Figure()
    
    for model_name, (fpr, tpr, auc_score) in models_results.items():
        fig.add_trace(go.Scatter(
            x=fpr, y=tpr,
            mode='lines',
            name=f'{model_name} (AUC = {auc_score:.3f})',
            line=dict(width=3)
        ))
    
    # Add diagonal line
    fig.add_trace(go.Scatter(
        x=[0, 1], y=[0, 1],
        mode='lines',
        name='Random Classifier',
        line=dict(dash='dash', color='gray')
    ))
    
    fig.update_layout(
        title='ROC Curves Comparison',
        xaxis_title='False Positive Rate',
        yaxis_title='True Positive Rate',
        xaxis=dict(range=[0, 1]),
        yaxis=dict(range=[0, 1])
    )
    return fig
```

### Feature Importance
```python
def plot_feature_importance_plotly(importance, feature_names, top_n=15):
    # Sort features by importance
    indices = np.argsort(importance)[::-1][:top_n]
    
    fig = go.Figure([go.Bar(
        x=importance[indices],
        y=[feature_names[i] for i in indices],
        orientation='h',
        marker_color=px.colors.sequential.Blues_r,
        hovertemplate='Feature: %{y}<br>Importance: %{x:.4f}<extra></extra>'
    )])
    
    fig.update_layout(
        title=f'Top {top_n} Feature Importance',
        xaxis_title='Importance',
        yaxis_title='Features',
        yaxis={'categoryorder': 'total ascending'}
    )
    return fig
```

### Confusion Matrix
```python
def plot_confusion_matrix_plotly(cm, class_names):
    # Normalize confusion matrix
    cm_norm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    
    fig = go.Figure(data=go.Heatmap(
        z=cm_norm,
        x=class_names,
        y=class_names,
        colorscale='Blues',
        text=cm,
        texttemplate='%{text}',
        textfont={"size": 12},
        hovertemplate='Predicted: %{x}<br>Actual: %{y}<br>' +
                      'Count: %{text}<br>Normalized: %{z:.3f}<extra></extra>'
    ))
    
    fig.update_layout(
        title='Confusion Matrix',
        xaxis_title='Predicted',
        yaxis_title='Actual',
        xaxis={'side': 'top'}
    )
    return fig
```

## Animation & Time Series

### Animated Model Training
```python
# Animate model training progress
def create_training_animation(losses, accuracies, epochs):
    fig = make_subplots(rows=1, cols=2,
                        subplot_titles=['Training Loss', 'Training Accuracy'])
    
    # Initial empty traces
    fig.add_trace(go.Scatter(x=[], y=[], mode='lines', name='Loss'), row=1, col=1)
    fig.add_trace(go.Scatter(x=[], y=[], mode='lines', name='Accuracy'), row=1, col=2)
    
    # Create frames for animation
    frames = []
    for i in range(1, len(epochs) + 1):
        frame = go.Frame(data=[
            go.Scatter(x=epochs[:i], y=losses[:i]),
            go.Scatter(x=epochs[:i], y=accuracies[:i])
        ])
        frames.append(frame)
    
    fig.frames = frames
    
    # Add play button
    fig.update_layout(
        updatemenus=[{
            'type': 'buttons',
            'buttons': [{
                'label': 'Play',
                'method': 'animate',
                'args': [None, {'frame': {'duration': 100}}]
            }]
        }],
        title='Model Training Progress'
    )
    return fig
```

### Time Series with Range Selector
```python
fig = go.Figure()
fig.add_trace(go.Scatter(x=df['date'], y=df['value'], mode='lines'))

fig.update_layout(
    title='Time Series with Range Selector',
    xaxis=dict(
        rangeselector=dict(
            buttons=list([
                dict(count=1, label="1m", step="month", stepmode="backward"),
                dict(count=6, label="6m", step="month", stepmode="backward"),
                dict(count=1, label="YTD", step="year", stepmode="todate"),
                dict(count=1, label="1y", step="year", stepmode="backward"),
                dict(step="all")
            ])
        ),
        rangeslider=dict(visible=True),
        type="date"
    )
)
fig.show()
```

## Customization & Styling

### Themes & Templates
```python
# Built-in templates
templates = ['plotly', 'plotly_white', 'plotly_dark', 'ggplot2', 
             'seaborn', 'simple_white', 'none']

# Set global template
pio.templates.default = 'plotly_white'

# Custom theme
custom_theme = dict(
    layout=dict(
        font_family='Arial',
        font_color='#2a2a2a',
        title_font_size=20,
        colorway=['#636EFA', '#EF553B', '#00CC96', '#AB63FA'],
        plot_bgcolor='white',
        paper_bgcolor='white'
    )
)

fig = go.Figure()
fig.update_layout(custom_theme['layout'])
```

### Export & Save
```python
# Save as HTML (interactive)
fig.write_html('plot.html')

# Save as static image
fig.write_image('plot.png', width=1200, height=800, scale=2)

# Show in browser
fig.show(renderer='browser')

# Offline plot for Jupyter
from plotly.offline import plot
plot(fig, filename='plot.html', auto_open=True)
```

### Custom Hover Templates
```python
# Rich hover information
fig = px.scatter(df, x='x', y='y', color='category')
fig.update_traces(
    hovertemplate='<b>%{fullData.name}</b><br>' +
                  'X: %{x}<br>' +
                  'Y: %{y}<br>' +
                  'Category: %{marker.color}<br>' +
                  '<extra></extra>',
    hoverlabel=dict(
        bgcolor='white',
        bordercolor='black',
        font_size=12
    )
)
```

---

## Quick Tips

1. **Use Plotly Express** for quick, high-level plots
2. **Use Graph Objects** for fine-grained control
3. **Add interactivity** with hover, zoom, pan, and selection
4. **Use animations** to show temporal changes
5. **Combine multiple traces** in one figure for comparisons
6. **Export as HTML** to share interactive plots
7. **Use subplots** for dashboards and multi-panel displays
8. **Customize hover templates** for better user experience
9. **Use `fig.show(renderer='browser')`** for better performance with large datasets
10. **Save static images** with `write_image()` for publications

## Color Scales Reference

```python
# Sequential: Blues, Viridis, Plasma, Inferno, Magma
# Diverging: RdBu, RdYlBu, Spectral, Picnic
# Qualitative: Set1, Set2, Set3, Pastel1, Dark2
# Custom: px.colors.sequential, px.colors.diverging, px.colors.qualitative

# Example usage
fig = px.scatter(df, x='x', y='y', color='z', 
                 color_continuous_scale='Viridis')
```