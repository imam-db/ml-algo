# Object Detection

**🇬🇧 English | [🇮🇩 Read in Bahasa Indonesia](README.id.md)**

🏠 Back to Advanced Topics: `../README.md`

A classifier answers *what is in this image*. A detector also answers *where*.
That extra question changes everything: one image can contain several objects,
objects overlap, predictions must be matched to ground truth before they can be
scored, and the number of possible boxes is enormous.

This module builds the core of a one-stage detector with **NumPy only**:

- box conversions between `xyxy`, `xywh` and `cxcywh`,
- vectorized pairwise IoU and greedy non-maximum suppression (NMS),
- a synthetic 48x48 small-shop shelf dataset with three product classes,
- a miniature YOLO-v1-style detector (`S = 6`, one box per cell) with
  im2col convolutions, manual backpropagation and Adam,
- target encoding, decoding, per-class NMS and mAP@0.5,
- a scikit-learn sliding-window baseline for comparison.

Everything is synthetic and CPU friendly. No PyTorch, no pretrained weights and
no external images are used.

## 📚 Table of Contents

- [Why Detection Is Hard](#-why-detection-is-hard)
- [Box Formats](#-box-formats)
- [Intersection over Union](#-intersection-over-union)
- [Non-Maximum Suppression](#-non-maximum-suppression)
- [The Synthetic Small-Shop Shelf Dataset](#-the-synthetic-small-shop-shelf-dataset)
- [YOLO in One Page](#-yolo-in-one-page)
- [Target Encoding](#-target-encoding)
- [The YOLO Loss](#-the-yolo-loss)
- [The Mini YOLO Network](#-the-mini-yolo-network)
- [Decoding Predictions](#-decoding-predictions)
- [Mean Average Precision at IoU 0.5](#-mean-average-precision-at-iou-05)
- [Quick Start](#-quick-start)
- [Sliding-Window Baseline vs Mini YOLO](#-sliding-window-baseline-vs-mini-yolo)
- [Notebook Exercises](#-notebook-exercises)
- [Tests](#-tests)
- [From toy to real: Ultralytics YOLO on your own labeled photos](#from-toy-to-real-ultralytics-yolo-on-your-own-labeled-photos)
- [Files in This Directory](#-files-in-this-directory)
- [Advantages and Limitations](#-advantages-and-limitations)
- [References](#-references)

## 🧠 Why Detection Is Hard

In a classification problem the label is a single value. In detection the label
is a *set* of boxes and classes, and its size varies from image to image. Four
problems appear immediately:

1. **Localization:** the model must output continuous coordinates, not just a
   class index.
2. **Multiple objects:** a one-hot label is no longer enough; the output must
   represent several objects at once.
3. **Matching:** before computing a loss or a metric, each prediction has to be
   assigned to a ground-truth box, usually by IoU.
4. **Duplicates:** a dense detector proposes many boxes around the same object;
   NMS is needed to remove them.

A detector is therefore a combination of a regression head, a classification
head, a matching rule, a decoder and an evaluation protocol. This module keeps
each part small enough to inspect.

## 📦 Box Formats

Three formats are used throughout the repository:

| Name | Meaning | Stored as |
|------|---------|-----------|
| `xyxy` | top-left and bottom-right corners | `(x1, y1, x2, y2)` |
| `xywh` | top-left corner plus size | `(x, y, w, h)` |
| `cxcywh` | centre plus size | `(cx, cy, w, h)` |

The conversions are pure algebra:

```
w  = x2 - x1                 x1 = x
h  = y2 - y1                 y1 = y
cx = (x1 + x2) / 2           x2 = x + w
cy = (y1 + y2) / 2           y2 = y + h
```

All six public helpers accept either one box with shape `(4,)` or a batch with
shape `(N, 4)`, and they are fully vectorized:

```python
from implementation import (
    xyxy_to_xywh, xywh_to_xyxy,
    xyxy_to_cxcywh, cxcywh_to_xyxy,
    xywh_to_cxcywh, cxcywh_to_xywh,
)
```

They are exact inverses for legal boxes with positive width and height. The
module does not silently clip a box during conversion, because a clipped box is
no longer a faithful round trip.

## 🎯 Intersection over Union

IoU measures how much two boxes overlap:

```
IoU(A, B) = area(A ∩ B) / area(A ∪ B)
```

where

```
area(A ∪ B) = area(A) + area(B) - area(A ∩ B)
```

The intersection is computed from the overlapping corner coordinates:

```
inter_x1 = max(x1_A, x1_B)      inter_y1 = max(y1_A, y1_B)
inter_x2 = min(x2_A, x2_B)      inter_y2 = min(y2_A, y2_B)
inter_w  = max(0, inter_x2 - inter_x1)
inter_h  = max(0, inter_y2 - inter_y1)
inter    = inter_w * inter_h
```

`iou_matrix(boxes_a, boxes_b)` broadcasts an `(N, 4)` array against an
`(M, 4)` array and returns an `(N, M)` matrix. Two details matter:

- boxes that only touch at an edge have zero intersection,
- if the union is zero (two degenerate boxes), IoU is defined as `0` instead of
  dividing by zero.

The same function is used by NMS, target diagnostics and mAP. Keeping one
implementation avoids subtle differences between training and evaluation.

## ✂️ Non-Maximum Suppression

A detector often produces several boxes around the same product. Greedy NMS
turns a scored set of boxes into a smaller set:

```
1. sort boxes by descending score
2. take the highest-scoring box and keep it
3. remove every remaining box whose IoU with the kept box is > threshold
4. repeat until no boxes remain
```

The output is an array of **indices sorted by score**. Sorting the boxes first
makes the result deterministic, and returning indices makes it easy to keep the
matching labels and scores.

Class-aware NMS is the same procedure applied separately to every class:

```
for each class c:
    keep_c = NMS(boxes[labels == c], scores[labels == c])
merge all keep_c and sort by score
```

Two boxes with different labels never suppress each other, even if they overlap
completely. This matters when a bottle stands in front of a carton: both objects
are real, so their boxes must both survive.

`per_class_nms(boxes, scores, labels, iou_threshold=0.5)` implements this and
also accepts the alias `batched_nms`.

## 🏪 The Synthetic Small-Shop Shelf Dataset

`generate_shelf_dataset(...)` returns:

```
images      : float32 array (N, C, 48, 48), values in [0, 1]
annotations : list of (boxes, labels)
              boxes  : float32 array (K, 4) in xyxy pixel coordinates
              labels : int64 array (K,), values 0, 1, 2
```

The three classes have deliberately different silhouettes and textures:

| Class | Shape | Texture |
|-------|-------|---------|
| `box` | wide carton | diagonal stripes and a bright centre label |
| `bottle` | narrow neck and cap above a wider body | side shading and a bright label band |
| `can` | cylinder | lid, base and horizontal ridges |

Each image contains one to four products, placed at random positions with a
margin. The bounding boxes are guaranteed not to overlap, and object centres are
placed in distinct 6x6 grid cells. The second constraint keeps the one-box-per-
cell target unambiguous; it is a property of this teaching dataset, not of YOLO
in general.

Other properties:

- **Seeded:** the same `seed` reproduces exactly the same images and boxes.
- **Channels:** `channels=1` gives grayscale, `channels=3` gives RGB.
- **Noise:** Gaussian sensor noise is added to make the task slightly less
  sterile.
- **Shelf context:** horizontal shelf boards are drawn into the background.
- **Synthetic only:** no personal photos, no client images, no scraping.

```python
images, annotations = generate_shelf_dataset(n_samples=240, seed=42)
boxes, labels = annotations[0]
```

## 🧠 YOLO in One Page

YOLO ("You Only Look Once") treats detection as a single regression problem.
The image is divided into an `S x S` grid. Each cell predicts `B` boxes and class
probabilities. In this module `S = 6` and `B = 1`.

For every cell the network outputs:

```
[objectness, x_offset, y_offset, sqrt_w, sqrt_h, class logits...]
```

- `objectness` is a logit; `sigmoid(objectness)` is the probability that the
  cell contains an object.
- `x_offset` and `y_offset` are direct offsets in `[0, 1]` relative to the cell.
  They are **not** passed through a sigmoid in this implementation.
- `sqrt_w` and `sqrt_h` predict the square root of the normalized box size.
- The class channels are raw logits passed through a softmax.

The cell centre and size are decoded as:

```
cx = (col + x_offset) * cell_size
cy = (row + y_offset) * cell_size
w  = sqrt_w ** 2 * image_size
h  = sqrt_h ** 2 * image_size
x1 = cx - w / 2      y1 = cy - h / 2
x2 = cx + w / 2      y2 = cy + h / 2
```

Predicting `sqrt(w)` instead of `w` is the original YOLO trick: it keeps the
regression target in a smaller numeric range and gives large boxes proportionally
less influence.

## 🎯 Target Encoding

`encode_targets` turns one image's boxes into a `(S, S, 5 + C)` tensor.

For each box:

1. compute its centre `(cx, cy)` and size `(w, h)`,
2. find the cell `col = floor(cx / cell_size)`, `row = floor(cy / cell_size)`,
3. write `objectness = 1`,
4. write `x_offset = cx / cell_size - col` and the analogous `y_offset`,
5. write `sqrt_w = sqrt(w / image_size)` and `sqrt_h = sqrt(h / image_size)`,
6. write a one-hot class vector.

If two box centres fall into the same cell, the larger box wins. The synthetic
dataset avoids this case, but the rule keeps the encoder deterministic.

The inverse function `decode_targets` reconstructs boxes from a target tensor.
It is useful for checking that encoding is lossless.

## 📉 The YOLO Loss

Let `M_ij` be `1` when cell `(i, j)` contains an object and `0` otherwise. Let
`o_ij` be the objectness logit, `(x, y, sqrt_w, sqrt_h)` the four coordinate
predictions, and `c_ij` the class logits. With batch size `N`:

```
L = (1 / N) * sum over images, cells [
        M_ij       * BCE(sigmoid(o_ij), 1)
      + (1 - M_ij) * lambda_noobj * BCE(sigmoid(o_ij), 0)
      + M_ij * lambda_coord * [
            (x - x_hat)^2 + (y - y_hat)^2
          + (sqrt_w - sqrt_w_hat)^2
          + (sqrt_h - sqrt_h_hat)^2
        ]
      + M_ij * CE(softmax(c_ij), class_ij)
    ]
```

with the defaults

```
lambda_coord = 5.0
lambda_noobj = 0.5
```

and the numerically stable binary cross-entropy

```
BCE(z, t) = max(z, 0) - z * t + log(1 + exp(-|z|))
```

where `z` is a logit and `t` is the target. This is exactly the loss implemented
in `yolo_loss_and_grad`:

- objectness uses **sigmoid + BCE**, not squared error,
- coordinate errors use plain squared error on offsets and square-root sizes,
- class cross-entropy is applied **only to object cells**,
- the total is divided by the batch size and summed over the grid and channels.

`yolo_loss_and_grad(raw, targets)` returns the scalar loss and its gradient with
respect to the raw head output. That is why it can be checked directly with
central finite differences, independently of the convolutional backbone.

The weighting matters. `lambda_coord = 5` makes localization errors expensive.
`lambda_noobj = 0.5` stops the many background cells from dominating training.
Even so, background cells are numerous: with `S = 6` there are 36 cells per
image and usually only one to four objects.

## 🧱 The Mini YOLO Network

`MiniYOLO` uses a small convolutional stack:

```
input C x 48 x 48
  conv 3x3, 16 channels + ReLU + maxpool 2x2   -> 24 x 24
  conv 3x3, 32 channels + ReLU + maxpool 2x2   -> 12 x 12
  conv 3x3, 32 channels + ReLU                 -> 12 x 12
  conv 3x3, 5 + C channels, stride 2           -> 6 x 6
```

The last convolution is the detection head. It produces one `(5 + C)`-vector for
each of the 36 cells; no dense layer is needed.

Implementation details:

- Convolutions use **im2col**: each receptive field becomes a row, so the
  forward pass is a matrix multiplication.
- Max pooling is non-overlapping 2x2 and caches the argmax for the backward pass.
- ReLU, convolution, pooling and the head all have manual backward passes.
- The optimiser is **Adam** with lazy per-parameter first and second moments.
- Parameters default to `float32`; pass `dtype=np.float64` for gradient checks.

Training one image batch is:

```
logits = model.forward(images)
loss, dlogits = yolo_loss_and_grad(logits, targets)
grads = model.backward(dlogits)
adam.step(model.parameters(), grads)
```

## 🔍 Decoding Predictions

`decode_single` turns one `(S, S, 5 + C)` raw output into boxes:

1. `objectness = sigmoid(raw[..., 0])`,
2. `class_probs = softmax(raw[..., 5:])`,
3. choose the best class and set
   `score = objectness * max(class_probs)`,
4. drop cells with `score < conf_threshold`,
5. decode centre and size in pixels,
6. clip boxes to the image,
7. apply **per-class NMS**.

The score is a product, not just the class probability. This is the standard
YOLO confidence: a box must both contain an object and classify it well.

`decode_predictions` accepts a single raw head or a batch and always returns a
list of `(boxes, scores, labels)` tuples. The tuples are sorted by score.

## 📏 Mean Average Precision at IoU 0.5

Average precision is computed per class from a ranked list of detections.

For every class:

1. pool all detections across all images,
2. sort them by descending score,
3. walk down the list and greedily match each detection to the unmatched
   ground-truth box with the highest IoU,
4. count a true positive if the best IoU is at least `0.5`; otherwise count a
   false positive,
5. each ground-truth box can be matched at most once.

With cumulative true positives `TP(k)` and false positives `FP(k)`:

```
recall(k)    = TP(k) / num_gt
precision(k) = TP(k) / (TP(k) + FP(k))
```

PASCAL VOC uses **all-point interpolation**. The precision envelope is made
monotone from right to left:

```
p_interp(k) = max_{k' >= k} precision(k')
```

and AP is the area under the resulting curve:

```
AP = sum_k (recall(k) - recall(k-1)) * p_interp(k),   recall(-1) = 0
```

Finally:

```
mAP@0.5 = (1 / C) * sum_c AP_c
```

Classes with no ground truth are reported as `0.0` but excluded from the mean.
If no class has ground truth, the result is `0.0`.

Two consequences are worth remembering:

- a false positive **before** the last true positive lowers AP, because it lowers
  the precision at high recall;
- a missed ground-truth object caps recall below `1` and therefore also lowers
  AP.

## 🚀 Quick Start

From this directory:

```bash
python implementation.py
python sklearn_example.py
```

A minimal detection pipeline:

```python
import numpy as np

from implementation import (
    MiniYOLO,
    generate_shelf_dataset,
    mean_average_precision,
)

images, annotations = generate_shelf_dataset(n_samples=240, seed=42)
train_images, train_annotations = images[:200], annotations[:200]
test_images, test_annotations = images[200:], annotations[200:]

model = MiniYOLO(seed=42)
history = model.fit(
    train_images,
    train_annotations,
    epochs=30,
    batch_size=16,
    lr=3e-3,
)
print("final loss:", round(history[-1], 3))

predictions = model.predict(test_images, conf_threshold=0.2)
print("mAP@0.5:", round(mean_average_precision(predictions, test_annotations), 3))
```

The same building blocks can be used directly:

```python
from implementation import iou_matrix, nms, per_class_nms

ious = iou_matrix(boxes_a, boxes_b)       # (N, M)
keep = nms(boxes, scores, 0.5)            # one class
keep = per_class_nms(boxes, scores, labels, 0.5)
```

## 🤖 Sliding-Window Baseline vs Mini YOLO

`sklearn_example.py` compares two detectors on the same synthetic split
(200 training images, 40 held-out images):

1. **Sliding window + LogisticRegression.** Four rectangular window sizes are
   slid over each image with stride 8. Each patch is described by an 8x8
   intensity thumbnail plus a small HOG-like gradient histogram (2x2 spatial
   cells, 9 orientation bins each). A multinomial logistic regression is trained
   with a background class (0) and three product classes (1, 2, 3). At inference
   the best non-background probability becomes the score, and per-class NMS
   removes duplicates.
2. **Mini YOLO.** The from-scratch detector is trained for 30 epochs with Adam
   and evaluated with the same decoder and mAP code.

### Results

The script was run on a CPU with the repository environment. The values below
are the actual output of one run; inference and training times vary with
hardware and load:

| Model | mAP@0.5 | Inference (ms/image) | Train (s) |
|---|---:|---:|---:|
| Sliding window + LogisticRegression | 0.372 | 8.7 | 4.6 |
| Mini YOLO (NumPy, from scratch) | 0.645 | 2.8 | 16.5 |

Per-class AP@0.5:

| Class | Sliding window | Mini YOLO |
|---|---:|---:|
| box | 0.583 | 0.808 |
| bottle | 0.169 | 0.630 |
| can | 0.362 | 0.497 |

Why the gap?

- The sliding window can only output boxes from its fixed window set. A tall
  bottle is poorly covered by square windows, which explains its low AP.
- Mini YOLO regresses the centre and size directly, so it can fit each object
  more tightly.
- The baseline evaluates roughly 85 patches per image with scikit-learn. The
  detector performs one small convolution pass, which is faster at inference.
- The baseline is still valuable: it is simple, interpretable and needs no
  custom backpropagation.

These numbers are from synthetic data and should not be read as a real-world
benchmark. Training time and inference time vary with hardware and load.

## 📓 Notebook Exercises

`exercise.ipynb` contains nine exercises with worked solutions:

1. convert boxes between all three formats,
2. compute IoU and run greedy NMS by hand,
3. verify class-aware NMS,
4. inspect the synthetic dataset and its reproducibility,
5. encode and decode YOLO targets,
6. check `yolo_loss_and_grad` with finite differences,
7. train a small detector for 20 epochs,
8. decode predictions and match them to ground truth,
9. sweep the confidence threshold and discuss model selection.

The notebook runs on a CPU in well under two minutes.

## 🧪 Tests

`tests/test_object_detection.py` loads the implementation with
`importlib.util.spec_from_file_location` under the unique name
`object_detection_implementation` and covers:

- known IoU cases: identical, disjoint, half overlap (`1/3`), touching edges and
  zero-area boxes,
- NMS ordering, suppression and class-aware behaviour,
- box conversion round trips for single boxes and batches,
- target encoding and decoding round trips,
- numerical gradient checks of the YOLO loss (relative error below `1e-5`) and
  of the convolutional backbone on a tiny input,
- decoding, score computation, thresholding and per-class NMS,
- mAP on hand-built cases: perfect predictions, one false positive and a missed
  ground truth,
- a short training run whose loss falls and whose mAP is clearly above trivial.

Run it with:

```bash
python -m pytest tests/test_object_detection.py -q
```

The whole suite finishes in a few seconds.

## 🚀 From toy to real: Ultralytics YOLO on your own labeled photos

The detector in this module is deliberately tiny. When you move to real shelf
photos, use a maintained implementation such as
[Ultralytics YOLO](https://docs.ultralytics.com) instead of this NumPy code.

### 1. Label your own photos

Use [Label Studio](https://labelstud.io/) or
[CVAT](https://www.cvat.ai/) and export in **YOLO format**:

```
dataset/
  images/
    train/shelf_0001.jpg
    val/shelf_0002.jpg
  labels/
    train/shelf_0001.txt
    val/shelf_0002.txt
```

Each label file has one line per object:

```
<class_id> <x_center> <y_center> <width> <height>
```

All four coordinates are normalized to `[0, 1]`. Keep the class list in a fixed
order and put the same names in `data.yaml`:

```yaml
path: /absolute/path/to/dataset
train: images/train
val: images/val

names:
  0: box
  1: bottle
  2: can
```

Labeling advice:

- draw tight boxes around the visible product, including the cap of a bottle,
- use one class per product type and write down the definition,
- include partially visible products if a human could still count them,
- include empty shelf regions and cluttered backgrounds,
- keep a held-out validation set from different days or different stores.

### 2. Train

```bash
yolo train model=yolo11n.pt data=data.yaml epochs=100 imgsz=640 batch=16
```

Start from a small pretrained model, then measure. For a CPU-only teaching run,
reduce `imgsz` and `epochs`; for production, use a GPU. Always evaluate on a
split that was never used for labeling decisions or hyperparameter tuning.

### 3. Planogram compliance and stock counting

Two common retail tasks are natural extensions of detection:

- **Planogram compliance:** each shelf position has an expected product and
  facing count. Detect the products, map boxes to shelf regions, and compare the
  observed layout with the planogram. Report missing facings, wrong products and
  misplaced items.
- **Stock counting:** count detections per class and per shelf, then combine the
  counts with a replenishment rule. A detector gives a count, not an inventory
  truth, so add confidence thresholds and a human review step for low-confidence
  or occluded items.

Useful practices:

- calibrate the confidence threshold on a validation set,
- report per-class precision and recall, not only mAP,
- track performance by store, lighting condition and camera angle,
- version the model, the data and the labeling guidelines together,
- monitor drift when cameras, products or shelf layouts change.

### 4. Keep your own photos private

Shelf photos can reveal store layouts, promotional plans and customer activity.
Treat them as sensitive data:

- keep your own photos private and store them in an access-controlled bucket,
- do not upload client images to public demos or third-party notebooks,
- remove faces, people and licence plates unless they are essential,
- check the data policy of any hosted training service,
- delete temporary copies and avoid committing images to git,
- **never use client data without written permission.**

The synthetic generator in this module exists precisely so that the algorithms
can be learned without exposing anyone's photos.

## 📂 Files in This Directory

| File | Purpose |
|------|---------|
| `implementation.py` | NumPy-only boxes, IoU, NMS, dataset, Mini YOLO, loss, decoding and mAP |
| `sklearn_example.py` | sliding-window LogisticRegression baseline vs Mini YOLO, with a markdown results table |
| `exercise.ipynb` | exercises with worked solutions |
| `README.md` | this English guide |
| `README.id.md` | the Indonesian guide |
| `../../tests/test_object_detection.py` | fast pytest suite for the module |

## ⚖️ Advantages and Limitations

### Advantages

- **Transparent:** every convolution, target and gradient is visible.
- **Self-contained:** only NumPy is required for the detector.
- **Fast enough to experiment:** a small training run finishes in seconds.
- **Correct evaluation:** IoU, NMS, VOC AP and mAP follow the standard recipe.
- **Comparable:** the scikit-learn baseline uses the same data and metric.

### Limitations

- `S = 6` and `B = 1` are very coarse; two objects in one cell cannot both be
  predicted.
- The dataset is synthetic, low resolution and free of perspective, motion blur
  and severe occlusion.
- The detector has no anchor boxes, feature pyramid, batch normalization,
  multi-scale training or test-time augmentation.
- The class logits of background cells are not supervised; only object cells
  receive class cross-entropy.
- The network is small and overfits quickly on a few dozen images.
- mAP@0.5 is a single-number summary; real retail systems also need per-class
  recall, calibration and a review workflow.

These limitations are intentional. The goal is to make the mechanics legible
before switching to a production library.

## 📚 References

- Redmon, J., Divvala, S., Girshick, R., & Farhadi, A. (2016).
  *You Only Look Once: Unified, Real-Time Object Detection.*
  <https://arxiv.org/abs/1506.02640>
- Everingham, M., Van Gool, L., Williams, C. K. I., Winn, J., & Zisserman, A.
  (2010). *The PASCAL Visual Object Classes (VOC) Challenge.* International
  Journal of Computer Vision.
  <https://link.springer.com/article/10.1007/s11263-009-0275-4>
- Neubeck, A., & Van Gool, L. (2006). *Efficient Non-Maximum Suppression.*
  International Conference on Pattern Recognition.
  <https://ieeexplore.ieee.org/document/1699659>
- Ultralytics documentation. <https://docs.ultralytics.com>

**Happy Learning! 🎓** | *[🇮🇩 Indonesian version](README.id.md)*
