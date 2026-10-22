# Neural Networks (Multi-Layer Perceptron)

**🇬🇧 English | [🇮🇩 Indonesian version](README.id.md)**

## Overview

Neural Networks (specifically Multi-Layer Perceptrons) are supervised learning algorithms inspired by biological neural networks. They learn complex non-linear patterns through multiple layers of interconnected neurons.

## Architecture

```
Input Layer → Hidden Layer(s) → Output Layer
    ↓              ↓                 ↓
  Features    ReLU/Sigmoid        Softmax
```

### Key Components

**1. Layers**
- Input: Features from data
- Hidden: Learn intermediate representations
- Output: Final predictions

**2. Activation Functions**
- **ReLU**: f(x) = max(0, x) - Most common for hidden layers
- **Sigmoid**: f(x) = 1/(1+e^-x) - Binary classification
- **Softmax**: Multi-class probabilities

**3. Training**
- Forward pass: Compute predictions
- Loss: Cross-entropy for classification
- Backpropagation: Compute gradients
- Update: Mini-batch SGD

## Implementation

```python
from implementation import MLPClassifier, generate_spg_outlet_data

# Generate SPG outlet stockout prediction data
X, y = generate_spg_outlet_data(n_samples=500, random_state=42)

# Train MLP
mlp = MLPClassifier(
    hidden_layers=[32, 16],
    activation='relu',
    learning_rate=0.01,
    max_iter=500,
    random_state=42
)
mlp.fit(X_train, y_train)

# Predict
predictions = mlp.predict(X_test)
probabilities = mlp.predict_proba(X_test)
```

## Gradient Checking

Verify backpropagation implementation:

```python
from implementation import gradient_check

max_diff, rel_diff = gradient_check(mlp, X_sample, y_sample)
print(f"Relative difference: {rel_diff:.2e}")  # Should be < 1e-4
```

## SPG Use Case

**Problem**: Predict outlet stockout risk  
**Features**: Sales, SKU count, visit frequency, OOS rate, basket size  
**Classes**: 0=Low risk, 1=Medium risk, 2=High risk

Neural networks capture complex patterns like:
- Non-linear relationships between features
- Interaction effects (e.g., high sales + low frequency = stockout risk)
- Hidden patterns in historical data

## When to Use

### ✅ Use When:
- Non-linear patterns in data
- Enough training data (100s-1000s samples)
- Feature interactions important
- Black-box model acceptable

### ❌ Avoid When:
- Small datasets (< 100 samples)
- Need model interpretability
- Linear patterns sufficient
- Limited computational resources

## Related Files

- `implementation.py` - From-scratch MLP with backpropagation
- `sklearn_example.py` - Comparison with sklearn
- `exercise.ipynb` - Interactive exercises

**Happy Learning! 🎓** | *[🇮🇩 Indonesian version](README.id.md)*
