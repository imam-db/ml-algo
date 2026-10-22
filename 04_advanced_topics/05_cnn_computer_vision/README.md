# CNN & Computer Vision Basics

**🇬🇧 English | [🇮�D Indonesian version](README.id.md)**

## Overview

Convolutional Neural Networks (CNNs) are specialized neural networks for processing grid-like data such as images. They use convolution operations to automatically learn spatial hierarchies of features.

## Key Operations

### 1. Convolution (conv2d)

Applies filters to detect patterns:

```python
from implementation import conv2d

output = conv2d(image, kernel, stride=1, padding=0)
```

**Parameters:**
- `stride`: Step size for sliding window
- `padding`: Zero-padding around image edges

**Use**: Edge detection, pattern recognition, feature extraction

### 2. Pooling (max_pool2d)

Downsamples spatial dimensions:

```python
from implementation import max_pool2d

pooled = max_pool2d(image, pool_size=2, stride=2)
```

**Benefits**: Reduces parameters, provides translation invariance

### 3. CNN Architecture

```
Input Image
    ↓
Conv2D + ReLU (feature detection)
    ↓
MaxPool (downsampling)
    ↓
Flatten (convert to vector)
    ↓
Dense Layer (classification)
    ↓
Output (predictions)
```

## Implementation

```python
from implementation import SimpleCNN, generate_product_images

# Generate synthetic product shelf images
X_train, y_train = generate_product_images(n_samples=100, random_state=42)

# Train CNN
cnn = SimpleCNN(
    input_shape=(28, 28),
    n_filters=8,
    kernel_size=3,
    learning_rate=0.01,
    random_state=42
)
cnn.fit(X_train, y_train, epochs=200)

# Predict stockout status
prediction = cnn.predict(new_image)  # 0 or 1
```

## SPG Retail Use Case

**Problem**: Detect shelf stockouts from images  
**Input**: 28x28 grayscale images of product shelves  
**Output**: Binary classification (0=stockout, 1=well-stocked)

**Synthetic Data**: Procedurally generated images with colored rectangles representing products on shelves.

**Real-World Applications**:
- Planogram compliance checking
- Automated shelf auditing
- Real-time stockout alerts
- Shelf space optimization

## Connection to YOLO & Object Detection

While this module covers CNN basics, modern retail applications use:

**YOLO (You Only Look Once)**:
- Real-time object detection
- Detects multiple products simultaneously
- Provides bounding boxes and class predictions
- Used for: Product recognition, counting, position tracking

**Progression**:
1. **This module**: Basic CNN for binary classification
2. **Next level**: Multi-class product recognition
3. **Advanced**: YOLO for simultaneous detection + localization
4. **Production**: Integration with planogram systems

## When to Use CNNs

### ✅ Use When:
- Spatial structure important (images, video)
- Need automatic feature learning
- Translation invariance desired
- Enough training data available

### ❌ Avoid When:
- Tabular/structured data (use MLP)
- Very small datasets
- No spatial structure
- Interpretability critical

## Related Files

- `implementation.py` - From-scratch conv2d, pooling, SimpleCNN
- `exercise.ipynb` - Interactive CNN exercises

**Happy Learning! 🎓** | *[🇮🇩 Indonesian version](README.id.md)*
