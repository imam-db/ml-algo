# Convolutional Neural Networks (CNN)

**🇬🇧 English | [🇮🇩 Bahasa Indonesia](README.id.md)**

🏠 Back to Advanced Topics Index: `../README.md`

Convolutional Neural Networks are the standard tool for learning from images.
Instead of connecting every pixel to every neuron, a CNN slides small learnable
filters across the image, shares those filters at every position, and builds
increasingly abstract features layer by layer. The result is a model with far
fewer parameters than a dense network and a strong built-in prior for spatial
structure.

This module implements a complete, small CNN **from scratch with NumPy**:
convolution through `im2col`, max pooling, ReLU, dense layers, numerically
stable softmax cross-entropy, backpropagation and momentum SGD. It then trains
the network on synthetic images that mimic a camera used by an SPG (sales
promotion girl) to audit a warung shelf.

## 📚 Table of Contents

- [Intuition](#-intuition)
- [Mathematical Foundation](#-mathematical-foundation)
- [Architecture](#-architecture)
- [The Warung Shelf Dataset](#-the-warung-shelf-dataset)
- [Quick Start](#-quick-start)
- [Results](#-results)
- [What the Network Learns](#-what-the-network-learns)
- [Tests](#-tests)
- [Advantages and Disadvantages](#-advantages-and-disadvantages)
- [When to Use a CNN](#-when-to-use-a-cnn)
- [Files in This Directory](#-files-in-this-directory)
- [From CNN to object detection (YOLO)](#-from-cnn-to-object-detection-yolo)
- [References](#-references)

## 🧠 Intuition

### Why not a dense network?

A 16x16 greyscale image has 256 pixels. A dense hidden layer with 64 units
already needs `256 * 64 = 16,384` weights. A 224x224 RGB photo has 150,528
inputs, so the first dense layer would need millions of weights before learning
anything useful. Dense layers also throw away the fact that nearby pixels belong
together.

A convolution kernel is much more efficient. A 3x3 filter has only 9 weights
per input channel, and the **same** filter is reused at every spatial location.
This is called **weight sharing**. It gives the network two useful properties:

- **Local connectivity:** each output depends only on a small neighbourhood.
- **Translation equivariance:** if the pattern moves, the filter response moves
  with it instead of disappearing.

### The warung shelf story

An SPG visits a warung (a small Indonesian convenience store), points a phone at
the shelf and captures a crop. The business wants to know which product type is
in the crop: a box pack, a bottle, a sachet strip or a can. The object can be
anywhere in the frame, at slightly different sizes and brightness levels, and
the camera adds noise.

A dense classifier can memorise pixel positions. A CNN learns filters for edges,
bars and curved boundaries, and can recognise the same product even when it is
shifted. That difference is the **inductive bias** we measure in
`sklearn_example.py`.

### A hierarchy of features

```
early layers   -> edges, corners, colour blobs
middle layers  -> bars, caps, circular rims, parts of packs
deeper layers  -> whole objects and class-level patterns
```

Each convolution followed by pooling increases the receptive field of a neuron,
so deeper units "see" a larger part of the image.

## 📐 Mathematical Foundation

### Convolution output size

For an input of height `H` and width `W`, a kernel of size `K_h x K_w`, stride
`S` and zero padding `P`, the output size is:

```
H_out = floor((H + 2P - K_h) / S) + 1
W_out = floor((W + 2P - K_w) / S) + 1
```

In this module a square kernel, stride and padding are common, so the formula
simplifies to `(H + 2P - K) / S + 1` with integer division. With `16x16` input,
a `3x3` kernel and `padding=1`, a convolution keeps the spatial size at
`16x16`. A following `2x2` max pool with stride 2 halves it to `8x8`.

### Cross-correlation

Deep-learning libraries call the operation "convolution", but in practice they
compute **cross-correlation** (the kernel is not flipped):

```
Y[n, o, i, j] = b[o]
              + sum over c, u, v of
                X_pad[n, c, i*S + u, j*S + v] * W[o, c, u, v]
```

where `n` indexes the batch, `o` the output channel, `c` the input channel and
`(u, v)` a position inside the kernel. The module follows this convention.

### im2col: convolution as a matrix product

A naive implementation needs six nested loops (batch, output channel, output
row, output column, input channel, kernel row, kernel column). That is slow in
Python. `im2col` rearranges every receptive field into one row of a matrix:

```
X : (N, C, H, W)
cols = im2col(X, K, S, P)
     : (N * H_out * W_out, C * K_h * K_w)

W_matrix = W.reshape(out_ch, C * K_h * K_w)

Y_flat = cols @ W_matrix.T + b
Y      = Y_flat.reshape(N, H_out, W_out, out_ch).transpose(0, 3, 1, 2)
```

The expensive part becomes a single matrix multiplication, which NumPy executes
in optimised C code. `col2im` is the exact adjoint operation: it sums each
gradient patch back into its original image location.

### Backpropagation through convolution

Let `dY` be the gradient of the loss with respect to the convolution output.
Using the same patches stored by `im2col`:

```
dW[o, c, u, v] = sum over n, i, j of
                 dY[n, o, i, j] * X_pad[n, c, i*S + u, j*S + v]

db[o] = sum over n, i, j of dY[n, o, i, j]

dX_pad[n, c, i*S + u, j*S + v] +=
    sum over o of dY[n, o, i, j] * W[o, c, u, v]
```

`col2im` performs the last accumulation, and the padding is removed at the end.

### Backpropagation through max pooling

For a window, max pooling keeps only the largest value:

```
Y[n, c, i, j] = max over (u, v) of X[n, c, i*S + u, j*S + v]
```

The gradient of a max is routed **only** to the input position that achieved the
maximum. Every other position receives zero. The layer stores the argmax indices
during the forward pass and uses them in the backward pass. If several values
tie, the first one in the window receives the gradient.

### ReLU and its gradient

```
ReLU(x) = max(0, x)
dReLU/dx = 1 if x > 0 else 0
```

ReLU is cheap and does not saturate for positive inputs, which makes it a good
default for the hidden layers in this small CNN.

### Softmax cross-entropy

For logits `z` and `K` classes:

```
shifted = z - max(z)                       # numerical stability
p_k     = exp(shifted_k) / sum_j exp(shifted_j)
L       = -(1/N) * sum_i log(p_i[y_i])
```

The gradient with respect to the logits is compact:

```
dL/dz_i = (p_i - one_hot(y_i)) / N
```

Subtracting the maximum logit prevents `exp` overflow. The implementation also
averages the loss and gradient over the mini-batch, matching the scale expected
by the optimiser.

### He initialisation

Weights are drawn from a zero-mean Gaussian with variance `2 / fan_in`, where
`fan_in` is the number of inputs to a unit:

```
W ~ Normal(0, sqrt(2 / fan_in))
b = 0
```

This keeps the variance of activations roughly stable through ReLU layers and
avoids the vanishing/exploding signals produced by very small or very large
initial weights.

### Momentum SGD

For a parameter `theta` and its gradient `g`:

```
v <- momentum * v + g
theta <- theta - learning_rate * v
```

Momentum accumulates a velocity in directions that keep pointing the same way,
which smooths noisy mini-batch updates. With `momentum=0` the update is plain
gradient descent.

## 🏗️ Architecture

`ShelfCNN` is a compact two-block convolutional classifier:

```
Input: (N, 1, 16, 16) greyscale crop
   |
   v
Conv2D: 1 -> 8 channels, 3x3 kernel, stride 1, padding 1   -> (N, 8, 16, 16)
ReLU
MaxPool2D: 2x2, stride 2                                   -> (N, 8, 8, 8)
   |
   v
Conv2D: 8 -> 16 channels, 3x3 kernel, stride 1, padding 1  -> (N, 16, 8, 8)
ReLU
MaxPool2D: 2x2, stride 2                                   -> (N, 16, 4, 4)
   |
   v
Flatten                                                    -> (N, 256)
Dense: 256 -> 64
ReLU
Dense: 64 -> 4                                             -> (N, 4)
   |
   v
Softmax cross-entropy loss
```

The `padding=1` in both convolutions preserves the spatial size before each
pooling layer. The model therefore has a predictable shape and works for any
square input size that is a multiple of four.

Every layer exposes:

- `forward(x) -> y`
- `backward(dout) -> dx`
- `.params` and `.grads` dictionaries for learnable tensors

`Sequential` simply calls the layers in order during the forward pass and in
reverse order during backpropagation. `SGD` reads the gradients from
`.grads` and updates `.params`, so no layer needs to know about the optimiser.

## 🏪 The Warung Shelf Dataset

`generate_shelf_images` creates reproducible synthetic images. It is inspired by
the kind of crop an SPG camera produces, but **no real product images are used**.

| Class | Product type | Shape drawn |
|-------|--------------|-------------|
| 0 | box pack | filled rectangle with a darker label band |
| 1 | bottle | tall thin body with a short cap |
| 2 | sachet strip | three horizontal bars separated by gaps |
| 3 | can | filled circle with a lighter rim |

For every image the generator:

1. Draws a dark shelf background with a subtle vertical gradient.
2. Picks a class and draws the corresponding shape at a random position.
3. Jitters the size and brightness.
4. Adds Gaussian pixel noise.
5. Clips the result to `[0, 1]`.

The function signature is:

```python
X, y = generate_shelf_images(
    n_samples=300,
    image_size=16,
    n_classes=4,     # 3 or 4
    noise=0.05,
    random_state=42,
)
# X: (n_samples, 1, image_size, image_size), float32
# y: (n_samples,), int
```

Labels are balanced by construction. The same `random_state` always produces
exactly the same images and labels, which makes experiments and tests
reproducible.

To simulate a camera shake or a product placed slightly differently, use:

```python
X_shifted = shift_images(X, shift_y=2, shift_x=2, fill=0.15)
```

The shift does not wrap around; vacated pixels are filled with the background
value. This is the translation-shift test used in `sklearn_example.py`.

## 🚀 Quick Start

Run the from-scratch demo (a few seconds on CPU):

```bash
python 04_advanced_topics/05_cnn_computer_vision/implementation.py
```

Run the comparison with scikit-learn:

```bash
python 04_advanced_topics/05_cnn_computer_vision/sklearn_example.py
```

Open the exercises:

```bash
jupyter notebook 04_advanced_topics/05_cnn_computer_vision/exercise.ipynb
```

Minimal usage:

```python
from implementation import ShelfCNN, generate_shelf_images, shift_images

X_train, y_train = generate_shelf_images(n_samples=400, random_state=42)
X_test, y_test = generate_shelf_images(n_samples=160, random_state=43)

model = ShelfCNN(n_classes=4, image_size=16, random_state=42)
history = model.fit(
    X_train,
    y_train,
    epochs=12,
    batch_size=32,
    lr=0.02,
    momentum=0.9,
    verbose=0,
)

print("Loss:", history[0], "->", history[-1])
print("Test accuracy:", model.score(X_test, y_test))
print("Shifted accuracy:", model.score(shift_images(X_test, 2, 2), y_test))
```

Working with individual layers:

```python
import numpy as np
from implementation import Conv2D, ReLU, MaxPool2D, Flatten, Dense
from implementation import SoftmaxCrossEntropy, SGD

layers = [
    Conv2D(1, 8, kernel=3, stride=1, padding=1, random_state=0),
    ReLU(),
    MaxPool2D(2, 2),
    Flatten(),
    Dense(8 * 8 * 8, 4, random_state=1),
]
logits = layers[0].forward(np.random.rand(2, 1, 16, 16))
# ... call the remaining layers ...
```

The optional `pytorch_example.py` shows the same idea in PyTorch/Keras. PyTorch
is **not** installed in this environment and the test suite never imports that
file.

## 📊 Results

The numbers below come from `sklearn_example.py` with the default seeds: 400
training images, 160 clean test images and the same 160 images shifted by 2
pixels down and right. Chance level is 25%.

| Model | Clean test | Shifted test | Drop | Training time |
|-------|-----------:|-------------:|-----:|--------------:|
| **CNN (NumPy, from scratch)** | **98.1%** | **88.8%** | **9.4%** | **~2 s** |
| MLP (scikit-learn, flattened pixels) | 90.0% | 61.3% | 28.7% | 0.3 s |
| Logistic regression (flattened pixels) | 75.0% | 50.6% | 24.4% | 0.1 s |

What the table shows:

- All three models beat chance on the clean test set.
- The CNN is the most accurate and degrades the least under translation.
- The dense MLP loses almost 30 percentage points after the shift because it
  learned weights tied to absolute pixel coordinates.
- Logistic regression is a useful sanity-check baseline, but a linear boundary
  in raw pixel space is not enough to separate the four shapes reliably.

The `__main__` demo uses a different split (360 training / 120 test images) and
reaches 100% test accuracy and about 91% accuracy on the shifted set. Small
differences are expected because the demo uses different seeds and epochs.

Reproducing the table:

```bash
python 04_advanced_topics/05_cnn_computer_vision/sklearn_example.py
```

## 🔍 What the Network Learns

After training, inspect the first convolution layer:

```python
first_conv = next(layer for layer in model.layers if isinstance(layer, Conv2D))
print(first_conv.params["W"].shape)   # (8, 1, 3, 3)
```

The eight `3x3` filters usually specialise into edge detectors and bar/blob
detectors. Visualising them as small heatmaps shows which directions each filter
responds to. The feature maps after the first convolution show where those
patterns occur in a particular crop. The exercise notebook plots both.

The second convolution combines the first-layer responses into more complex
parts, such as the cap of a bottle or the rim of a can. The dense classifier then
maps the pooled feature vector to the four class scores.

## 🧪 Tests

Run the module tests:

```bash
pytest tests/test_cnn_computer_vision.py -q
```

The suite checks:

- `im2col` / `col2im` shapes and the adjoint property
  `<im2col(x), r> = <x, col2im(r)>`.
- Convolution output against a naive loop implementation, including stride and
  padding.
- Numerical gradient checks for `Conv2D` (input and weights), `MaxPool2D`,
  `Dense` and softmax cross-entropy, all below `1e-5` relative error in float64.
- Output shapes of the layers and the `ShelfCNN` API.
- Generator shapes, value range, balanced labels, determinism and translation.
- End-to-end training: loss decreases, test accuracy is far above chance and
  the run finishes in under 10 seconds.
- The backwards-compatible `SimpleCNN`, `generate_product_images` and
  `gradient_check_cnn` API.

The tests load `implementation.py` with `importlib.util.spec_from_file_location`,
so they do not collide with the many other modules that also ship an
`implementation.py`.

## ✅ Advantages and Disadvantages

### Advantages

- **Parameter efficiency:** the same filter is reused at every position.
- **Translation robustness:** pooling and weight sharing make predictions less
  sensitive to small shifts.
- **Automatic feature learning:** no hand-crafted edge or texture descriptors.
- **Hierarchical representations:** deeper layers combine simple parts into
  objects.
- **Strong prior for images:** local structure and spatial smoothness are built
  into the architecture.
- **Hardware friendly:** convolution is mostly dense matrix multiplication
  (especially through `im2col`).

### Disadvantages

- **Data hungry:** CNNs usually need many labelled images to generalise well.
- **Compute hungry:** training deep networks is much more expensive than
  logistic regression or a small MLP.
- **Many design choices:** number of layers, channels, kernel sizes, strides,
  pooling, normalisation and learning rate all interact.
- **Less interpretable:** learned filters can be visualised, but the full
  decision process is hard to explain.
- **Not universal:** tabular data, sparse features and tiny datasets are often
  better served by tree ensembles or linear models.
- **This implementation is educational:** it is not optimised for speed,
  numerical precision or production-scale images.

## 🎯 When to Use a CNN

Use a CNN when:

- The input has spatial or grid structure (images, video frames, spectrograms).
- Local patterns matter and the same pattern can appear anywhere.
- You have enough labelled data, or can use transfer learning.
- Translation robustness is important.
- You can afford the training compute.

Consider alternatives when:

- The data is tabular with no spatial structure; use gradient boosting, random
  forests or a small MLP.
- The dataset is very small; classical features plus a simple classifier may
  generalise better.
- You need a highly interpretable model.
- Latency and memory are extremely tight; a hand-crafted feature pipeline may
  still win.
- The task is object detection or segmentation; a classification CNN is only a
  building block, not the complete solution.

## 📁 Files in This Directory

- **`README.md`** (this file) — theory and usage in English.
- **`README.id.md`** — the same material in Bahasa Indonesia.
- **`implementation.py`** — NumPy-only layers, loss, optimiser, `ShelfCNN`,
  `SimpleCNN`, data generators and the `__main__` demo.
- **`sklearn_example.py`** — CNN vs MLP vs logistic regression, including the
  translation-shift test (headless, prints a table).
- **`exercise.ipynb`** — hands-on notebook: data, convolution, training, loss
  curve, filters, confusion matrix, translation shift and an IoU exercise.
- **`pytorch_example.py`** — optional PyTorch/Keras comparison. It is not
  imported by the tests, and PyTorch is not installed here.

## 🔗 From CNN to object detection (YOLO)

Everything in this module is **image classification**: one label for one crop.
A real shelf-audit system must also answer **where** each product is. That is
**object detection**.

### Bounding boxes

A detector outputs a bounding box for each object, usually as:

```
(x1, y1, x2, y2)          corner format
(x_center, y_center, w, h) center format
```

The box localises the product; a class label says what it is; a confidence score
says how sure the model is.

### Grid cells

YOLO ("You Only Look Once") divides the image into an `S x S` grid. Each grid
cell is responsible for objects whose centre falls inside it. A cell predicts:

- `B` bounding boxes, each with `(x, y, w, h)` offsets and an objectness score,
- `C` conditional class probabilities.

The objectness score is trained to reflect both "is there an object?" and how
well the predicted box overlaps the ground truth. At inference time the class
score of a box is often `objectness * class_probability`.

Because one forward pass predicts all cells at once, YOLO is fast enough for
real-time camera feeds.

### Intersection over Union (IoU)

IoU measures how much two boxes overlap:

```
intersection = area of the overlap rectangle
union        = area(A) + area(B) - intersection
IoU          = intersection / union
```

`IoU = 0` means no overlap, `IoU = 1` means identical boxes. IoU is used both as
an evaluation metric and inside training and post-processing.

### Non-maximum suppression (NMS)

A detector often produces many overlapping boxes for the same product. NMS
removes duplicates:

1. Sort all candidate boxes by confidence, highest first.
2. Keep the highest-scoring box.
3. Remove every remaining box whose IoU with the kept box is above a threshold
   (for example `0.5`).
4. Repeat with the next highest-scoring box until no candidates remain.

The result is a short list of confident, non-overlapping detections.

A future module will build a small YOLO-style detector on top of the convolution
and pooling primitives from this module. The jump from classification to
detection is mostly about adding a **localisation head** (box regression),
designing **anchor boxes or grid targets**, and implementing **IoU + NMS** for
post-processing.

## 📚 References

1. LeCun, Y., Bottou, L., Bengio, Y., and Haffner, P. (1998).
   *Gradient-Based Learning Applied to Document Recognition.*
   Proceedings of the IEEE, 86(11), 2278–2324.
   <http://yann.lecun.com/exdb/publis/pdf/lecun-01a.pdf>

2. Krizhevsky, A., Sutskever, I., and Hinton, G. E. (2012).
   *ImageNet Classification with Deep Convolutional Neural Networks.*
   Advances in Neural Information Processing Systems 25.
   <https://papers.nips.cc/paper/2012/hash/c399862d3b9d6b76c8436e924a68c45b-Abstract.html>

3. Redmon, J., Divvala, S., Girshick, R., and Farhadi, A. (2016).
   *You Only Look Once: Unified, Real-Time Object Detection.*
   <https://arxiv.org/abs/1506.02640>

4. CS231n course notes, *Convolutional Neural Networks for Visual Recognition.*
   <https://cs231n.github.io/convolutional-networks/>

---

**Next steps**

- Read `implementation.py` and trace one forward and backward pass by hand.
- Run `sklearn_example.py` to see the translation-shift gap.
- Complete `exercise.ipynb`, including the filter visualisation and IoU helper.
- Try a deeper architecture or a different number of classes.
- Then move on to detection: bounding boxes, IoU and NMS.
