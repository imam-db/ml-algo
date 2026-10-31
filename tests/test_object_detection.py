"""
Tests for the object-detection module (04_advanced_topics/07_object_detection).

The implementation is loaded with ``importlib.util.spec_from_file_location``
under a unique module name, following ``tests/test_time_series.py``.  The suite
covers box conversions, IoU, NMS, synthetic data, YOLO target encoding, the
YOLO loss and convolutional backbone gradients, decoding, mAP and a short
end-to-end training run.
"""

import importlib.util
from pathlib import Path

import numpy as np
import pytest

MODULE_DIR = (
    Path(__file__).resolve().parent.parent
    / "04_advanced_topics"
    / "07_object_detection"
)
_SPEC = importlib.util.spec_from_file_location(
    "object_detection_implementation",
    MODULE_DIR / "implementation.py",
)
assert _SPEC is not None and _SPEC.loader is not None
_IMPL = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_IMPL)

Adam = _IMPL.Adam
MiniYOLO = _IMPL.MiniYOLO
box_area = _IMPL.box_area
cxcywh_to_xywh = _IMPL.cxcywh_to_xywh
cxcywh_to_xyxy = _IMPL.cxcywh_to_xyxy
decode_predictions = _IMPL.decode_predictions
decode_single = _IMPL.decode_single
decode_targets = _IMPL.decode_targets
decode_targets_batch = _IMPL.decode_targets_batch
encode_targets = _IMPL.encode_targets
encode_targets_batch = _IMPL.encode_targets_batch
generate_shelf_dataset = _IMPL.generate_shelf_dataset
iou_matrix = _IMPL.iou_matrix
mean_average_precision = _IMPL.mean_average_precision
nms = _IMPL.nms
per_class_nms = _IMPL.per_class_nms
voc_average_precision = _IMPL.voc_average_precision
xywh_to_cxcywh = _IMPL.xywh_to_cxcywh
xywh_to_xyxy = _IMPL.xywh_to_xyxy
xyxy_to_cxcywh = _IMPL.xyxy_to_cxcywh
xyxy_to_xywh = _IMPL.xyxy_to_xywh
yolo_loss_and_grad = _IMPL.yolo_loss_and_grad


def _relative_error(numeric: np.ndarray, analytic: np.ndarray) -> float:
    return float(
        np.linalg.norm(numeric - analytic)
        / max(1e-12, np.linalg.norm(numeric) + np.linalg.norm(analytic))
    )


def _match_annotation(
    boxes: np.ndarray,
    labels: np.ndarray,
    other_boxes: np.ndarray,
    other_labels: np.ndarray,
) -> None:
    """Assert that two annotated sets describe the same objects."""
    assert len(boxes) == len(other_boxes)
    assert sorted(labels.tolist()) == sorted(other_labels.tolist())
    overlaps = iou_matrix(boxes, other_boxes)
    for index, box in enumerate(boxes):
        best = int(np.argmax(overlaps[index]))
        assert overlaps[index, best] > 0.999
        assert labels[index] == other_labels[best]


class TestBoxConversions:
    def test_round_trips_for_all_formats(self):
        boxes = np.array(
            [
                [2.0, 4.0, 12.0, 14.0],
                [0.0, 0.0, 8.0, 4.0],
                [20.0, 10.0, 21.0, 31.0],
            ]
        )
        pairs = [
            (xyxy_to_xywh, xywh_to_xyxy),
            (xyxy_to_cxcywh, cxcywh_to_xyxy),
            (xywh_to_cxcywh, cxcywh_to_xywh),
        ]
        for forward, backward in pairs:
            converted = forward(boxes)
            restored = backward(converted)
            np.testing.assert_allclose(restored, boxes, atol=1e-12)

    def test_single_box_keeps_its_shape(self):
        box = np.array([1.0, 2.0, 7.0, 9.0])
        assert xyxy_to_xywh(box).shape == (4,)
        np.testing.assert_allclose(
            xyxy_to_xywh(box), np.array([1.0, 2.0, 6.0, 7.0])
        )
        np.testing.assert_allclose(
            xyxy_to_cxcywh(box), np.array([4.0, 5.5, 6.0, 7.0])
        )

    def test_batch_is_vectorized_and_handles_empty(self):
        boxes = np.arange(20, dtype=float).reshape(5, 4)
        converted = xyxy_to_xywh(boxes)
        assert converted.shape == (5, 4)
        np.testing.assert_allclose(
            converted[:, 2], boxes[:, 2] - boxes[:, 0]
        )
        empty = np.empty((0, 4))
        assert xyxy_to_xywh(empty).shape == (0, 4)
        assert xywh_to_xyxy(empty).shape == (0, 4)

    def test_box_area(self):
        np.testing.assert_allclose(
            box_area(np.array([[0, 0, 2, 3], [5, 5, 5, 9]])),
            np.array([6.0, 0.0]),
        )

    def test_invalid_shape_is_rejected(self):
        with pytest.raises(ValueError):
            xyxy_to_xywh(np.zeros((3, 5)))


class TestIoU:
    def test_known_values(self):
        box = np.array([[0.0, 0.0, 2.0, 2.0]])
        assert iou_matrix(box, box)[0, 0] == pytest.approx(1.0)
        assert iou_matrix(box, np.array([[10.0, 10.0, 12.0, 12.0]]))[
            0, 0
        ] == pytest.approx(0.0)
        half = np.array([[1.0, 0.0, 3.0, 2.0]])
        assert iou_matrix(box, half)[0, 0] == pytest.approx(1.0 / 3.0)

    def test_touching_edges_and_zero_area(self):
        a = np.array([[0.0, 0.0, 2.0, 2.0]])
        touching = np.array([[2.0, 0.0, 4.0, 2.0]])
        zero = np.array([[0.0, 0.0, 0.0, 2.0]])
        assert iou_matrix(a, touching)[0, 0] == 0.0
        assert iou_matrix(a, zero)[0, 0] == 0.0
        assert iou_matrix(zero, zero)[0, 0] == 0.0

    def test_pairwise_shape_and_values(self):
        a = np.array(
            [
                [0.0, 0.0, 4.0, 4.0],
                [10.0, 10.0, 14.0, 14.0],
            ]
        )
        b = np.array(
            [
                [0.0, 0.0, 4.0, 4.0],
                [2.0, 0.0, 6.0, 4.0],
                [50.0, 50.0, 60.0, 60.0],
            ]
        )
        result = iou_matrix(a, b)
        assert result.shape == (2, 3)
        assert result[0, 0] == pytest.approx(1.0)
        assert result[0, 1] == pytest.approx(1.0 / 3.0)
        assert result[1, 2] == 0.0

    def test_single_boxes_return_one_by_one(self):
        result = iou_matrix(
            np.array([0.0, 0.0, 2.0, 2.0]),
            np.array([1.0, 0.0, 3.0, 2.0]),
        )
        assert result.shape == (1, 1)
        assert result[0, 0] == pytest.approx(1.0 / 3.0)


class TestNMS:
    def test_suppresses_overlap_and_keeps_disjoint(self):
        boxes = np.array(
            [
                [0.0, 0.0, 10.0, 10.0],
                [1.0, 1.0, 11.0, 11.0],
                [20.0, 20.0, 30.0, 30.0],
            ]
        )
        scores = np.array([0.9, 0.8, 0.7])
        keep = nms(boxes, scores, 0.5)
        np.testing.assert_array_equal(keep, np.array([0, 2]))
        assert scores[keep[0]] >= scores[keep[1]]

    def test_returns_indices_sorted_by_score(self):
        boxes = np.array(
            [
                [0.0, 0.0, 5.0, 5.0],
                [20.0, 0.0, 25.0, 5.0],
                [40.0, 0.0, 45.0, 5.0],
            ]
        )
        scores = np.array([0.2, 0.95, 0.5])
        keep = nms(boxes, scores, 0.5)
        np.testing.assert_array_equal(keep, np.array([1, 2, 0]))

    def test_per_class_keeps_overlapping_boxes_of_different_classes(self):
        boxes = np.array(
            [
                [0.0, 0.0, 10.0, 10.0],
                [0.0, 0.0, 10.0, 10.0],
                [0.0, 0.0, 10.0, 10.0],
            ]
        )
        scores = np.array([0.9, 0.8, 0.7])
        labels = np.array([0, 1, 0])
        keep = per_class_nms(boxes, scores, labels, 0.5)
        np.testing.assert_array_equal(keep, np.array([0, 1]))
        assert labels[keep[0]] != labels[keep[1]]

    def test_per_class_nms_suppresses_same_class(self):
        boxes = np.array([[0.0, 0.0, 10.0, 10.0], [1.0, 1.0, 11.0, 11.0]])
        scores = np.array([0.9, 0.8])
        labels = np.array([2, 2])
        keep = per_class_nms(boxes, scores, labels, 0.5)
        np.testing.assert_array_equal(keep, np.array([0]))

    def test_empty_and_threshold_validation(self):
        empty = nms(np.empty((0, 4)), np.empty((0,)), 0.5)
        assert empty.size == 0
        with pytest.raises(ValueError):
            nms(np.array([[0.0, 0.0, 1.0, 1.0]]), np.array([1.0]), 1.5)
        with pytest.raises(ValueError):
            nms(
                np.array([[0.0, 0.0, 1.0, 1.0]]),
                np.array([1.0, 0.5]),
                0.5,
            )


class TestSyntheticDataset:
    def test_shapes_reproducibility_and_values(self):
        images, annotations = generate_shelf_dataset(n_samples=12, seed=7)
        images_again, annotations_again = generate_shelf_dataset(
            n_samples=12, seed=7
        )
        np.testing.assert_array_equal(images, images_again)
        assert images.shape == (12, 1, 48, 48)
        assert images.dtype == np.float32
        assert images.min() >= 0.0
        assert images.max() <= 1.0
        assert len(annotations) == len(annotations_again) == 12
        for (boxes, labels), (boxes_again, labels_again) in zip(
            annotations, annotations_again, strict=True
        ):
            np.testing.assert_array_equal(boxes, boxes_again)
            np.testing.assert_array_equal(labels, labels_again)

    def test_annotations_are_valid_and_non_overlapping(self):
        images, annotations = generate_shelf_dataset(n_samples=20, seed=3)
        assert images.shape == (20, 1, 48, 48)
        discovered = set()
        for boxes, labels in annotations:
            assert 1 <= len(boxes) <= 4
            assert boxes.shape == (len(labels), 4)
            assert np.all(boxes[:, 0] >= 0.0)
            assert np.all(boxes[:, 1] >= 0.0)
            assert np.all(boxes[:, 2] <= 48.0)
            assert np.all(boxes[:, 3] <= 48.0)
            assert np.all(boxes[:, 2] > boxes[:, 0])
            assert np.all(boxes[:, 3] > boxes[:, 1])
            assert set(labels.tolist()) <= {0, 1, 2}
            discovered.update(labels.tolist())
            overlaps = iou_matrix(boxes, boxes)
            np.fill_diagonal(overlaps, 0.0)
            assert np.max(overlaps) == 0.0
        assert discovered == {0, 1, 2}

    def test_rgb_and_noise_options(self):
        images, annotations = generate_shelf_dataset(
            n_samples=4, channels=3, seed=1
        )
        assert images.shape == (4, 3, 48, 48)
        assert len(annotations) == 4
        other, _ = generate_shelf_dataset(n_samples=4, seed=2)
        assert not np.allclose(images[0], images[1])
        assert not np.allclose(images.mean(axis=1), other.mean(axis=1))

    def test_invalid_options_are_rejected(self):
        with pytest.raises(ValueError):
            generate_shelf_dataset(n_samples=2, channels=2)
        with pytest.raises(ValueError):
            generate_shelf_dataset(n_samples=2, image_size=40)


class TestTargetEncoding:
    def test_cell_assignment_and_offsets(self):
        target = encode_targets(
            np.array([[8.0, 8.0, 16.0, 16.0]]),
            np.array([1]),
            image_size=48,
            grid_size=6,
        )
        assert target.shape == (6, 6, 8)
        assert target[1, 1, 0] == 1.0
        assert target[1, 1, 1] == pytest.approx(0.5)
        assert target[1, 1, 2] == pytest.approx(0.5)
        assert target[1, 1, 3] == pytest.approx(np.sqrt(8.0 / 48.0))
        assert target[1, 1, 4] == pytest.approx(np.sqrt(8.0 / 48.0))
        assert target[1, 1, 6] == 1.0
        assert target[..., 0].sum() == 1.0

    def test_encode_decode_round_trip(self):
        _, annotations = generate_shelf_dataset(n_samples=6, seed=17)
        targets = encode_targets_batch(annotations)
        decoded = decode_targets_batch(targets)
        for (boxes, labels), (decoded_boxes, decoded_labels) in zip(
            annotations, decoded, strict=True
        ):
            _match_annotation(boxes, labels, decoded_boxes, decoded_labels)

    def test_duplicate_cell_keeps_larger_box(self):
        boxes = np.array(
            [
                [1.0, 1.0, 6.0, 6.0],
                [0.0, 0.0, 8.0, 8.0],
            ]
        )
        target = encode_targets(boxes, np.array([0, 2]))
        assert target[..., 0].sum() == 1.0
        assert target[0, 0, 5] == 0.0  # larger class-2 box wins
        _, labels = decode_targets(target)
        assert labels.tolist() == [2]

    def test_batch_shape(self):
        _, annotations = generate_shelf_dataset(n_samples=3, seed=4)
        targets = encode_targets_batch(annotations)
        assert targets.shape == (3, 6, 6, 8)
        assert np.all(targets[..., 0].sum(axis=(1, 2)) >= 1.0)


class TestYoloLossGradient:
    def _make_batch(self, seed: int = 0):
        rng = np.random.default_rng(seed)
        raw = rng.normal(0.0, 0.4, size=(2, 6, 6, 8))
        targets = np.zeros_like(raw)
        for image in range(2):
            for _ in range(3):
                row, col = rng.integers(0, 6, size=2)
                targets[image, row, col, 0] = 1.0
                targets[image, row, col, 1:3] = rng.random(2)
                targets[image, row, col, 3:5] = rng.uniform(0.2, 0.5, size=2)
                targets[image, row, col, 5 + int(rng.integers(0, 3))] = 1.0
        return raw, targets

    def test_numerical_gradient_wrt_raw_head(self):
        raw, targets = self._make_batch()
        _, analytic = yolo_loss_and_grad(raw, targets)
        numeric = np.zeros_like(raw)
        step = 1e-6
        for index in np.ndindex(raw.shape):
            original = raw[index]
            raw[index] = original + step
            plus, _ = yolo_loss_and_grad(raw, targets)
            raw[index] = original - step
            minus, _ = yolo_loss_and_grad(raw, targets)
            raw[index] = original
            numeric[index] = (plus - minus) / (2.0 * step)
        assert _relative_error(numeric, analytic) < 1e-5

    def test_loss_decreases_when_predictions_match_targets(self):
        raw, targets = self._make_batch(3)
        loss_before, _ = yolo_loss_and_grad(raw, targets)
        loss_after, _ = yolo_loss_and_grad(targets * 2.0, targets)
        assert loss_after < loss_before

    def test_lambda_noobj_changes_background_gradient(self):
        raw, targets = self._make_batch(5)
        _, grad_default = yolo_loss_and_grad(raw, targets, lambda_noobj=0.5)
        _, grad_strong = yolo_loss_and_grad(raw, targets, lambda_noobj=2.0)
        background = targets[..., 0] < 0.5
        assert np.all(
            np.abs(grad_strong[..., 0][background])
            > np.abs(grad_default[..., 0][background])
        )

    def test_shape_validation(self):
        raw, targets = self._make_batch(1)
        with pytest.raises(ValueError):
            yolo_loss_and_grad(raw, targets[:, :5])


class TestBackboneGradient:
    def test_numerical_gradient_of_tiny_conv_backbone(self):
        rng = np.random.default_rng(2)
        model = MiniYOLO(
            num_classes=2,
            grid_size=2,
            image_size=8,
            in_channels=1,
            base_channels=(3, 4),
            head_channels=3,
            seed=4,
            dtype=np.float64,
        )
        image = rng.normal(0.0, 0.5, size=(1, 1, 8, 8))
        raw = model.forward(image)
        seed_grad = rng.normal(0.0, 1.0, size=raw.shape)
        analytic = model.backward(seed_grad)["input"]
        numeric = np.zeros_like(image)
        step = 1e-6
        for index in np.ndindex(image.shape):
            original = image[index]
            image[index] = original + step
            plus = float(np.sum(model.forward(image) * seed_grad))
            image[index] = original - step
            minus = float(np.sum(model.forward(image) * seed_grad))
            image[index] = original
            numeric[index] = (plus - minus) / (2.0 * step)
        assert _relative_error(numeric, analytic) < 1e-5

    def test_parameter_gradients_match_reference_for_one_filter(self):
        rng = np.random.default_rng(6)
        model = MiniYOLO(
            num_classes=2,
            grid_size=2,
            image_size=8,
            in_channels=1,
            base_channels=(3, 4),
            head_channels=3,
            seed=2,
            dtype=np.float64,
        )
        image = rng.normal(0.0, 0.5, size=(1, 1, 8, 8))
        targets = np.zeros((1, 2, 2, 7))
        targets[0, 0, 1, 0] = 1.0
        targets[0, 0, 1, 1:5] = [0.5, 0.5, 0.3, 0.4]
        targets[0, 0, 1, 5] = 1.0
        _, grads = model.loss_and_grad(image, targets)
        params = model.parameters()
        weights = params["conv2_W"]
        step = 1e-6
        for index in [(0, 0, 1, 1), (1, 2, 0, 2)]:
            original = weights[index]
            weights[index] = original + step
            plus, _ = model.loss_and_grad(image, targets)
            weights[index] = original - step
            minus, _ = model.loss_and_grad(image, targets)
            weights[index] = original
            numeric = (plus - minus) / (2.0 * step)
            analytic = grads["conv2_W"][index]
            assert abs(numeric - analytic) / max(
                1e-12, abs(numeric) + abs(analytic)
            ) < 1e-5


class TestDecoding:
    def _raw(self, cells, labels=None, objectness=6.0):
        raw = np.full((6, 6, 8), -12.0)
        for row, col in cells:
            raw[row, col, 0] = objectness
            raw[row, col, 1:3] = 0.5
            raw[row, col, 3:5] = np.sqrt(20.0 / 48.0)
            raw[row, col, 5:] = 0.0
            if labels is not None:
                raw[row, col, 5 + labels[(row, col)]] = 8.0
            else:
                raw[row, col, 5] = 8.0
        return raw

    def test_score_is_objectness_times_class_probability(self):
        raw = np.full((6, 6, 8), -12.0)
        raw[2, 3, 0] = 0.0
        raw[2, 3, 1:3] = 0.5
        raw[2, 3, 3:5] = np.sqrt(16.0 / 48.0)
        raw[2, 3, 5:] = [0.0, 4.0, -1.0]
        boxes, scores, labels = decode_single(
            raw, conf_threshold=0.01, apply_nms=False
        )
        class_logits = np.array([0.0, 4.0, -1.0])
        class_probs = np.exp(class_logits - class_logits.max())
        class_probs /= class_probs.sum()
        assert len(boxes) == 1
        assert scores[0] == pytest.approx(0.5 * class_probs[1])
        assert labels[0] == 1
        np.testing.assert_allclose(boxes[0], np.array([20.0, 12.0, 36.0, 28.0]))

    def test_overlapping_same_class_is_suppressed(self):
        raw = self._raw([(0, 0), (0, 1)])
        boxes, scores, labels = decode_single(raw, conf_threshold=0.2)
        assert len(boxes) == 1
        assert scores[0] > 0.9
        assert labels[0] == 0

    def test_per_class_decoding_keeps_different_labels(self):
        raw = self._raw([(0, 0), (0, 1)], labels={(0, 0): 0, (0, 1): 1})
        boxes, _, labels = decode_single(raw, conf_threshold=0.2)
        assert len(boxes) == 2
        assert sorted(labels.tolist()) == [0, 1]

    def test_threshold_removes_low_scores(self):
        raw = np.full((6, 6, 8), -12.0)
        raw[0, 0, 0] = -6.0
        raw[0, 0, 5] = 8.0
        boxes, scores, labels = decode_single(raw, conf_threshold=0.2)
        assert len(boxes) == 0
        assert scores.size == 0 and labels.size == 0

    def test_batch_decoder_returns_one_tuple_per_image(self):
        raw = np.stack([self._raw([(1, 1)]), self._raw([(2, 2), (4, 4)])])
        result = decode_predictions(raw, conf_threshold=0.2)
        assert len(result) == 2
        assert len(result[0][0]) == 1
        assert len(result[1][0]) == 2


class TestMeanAveragePrecision:
    def test_perfect_predictions_give_one(self):
        gt = [(np.array([[0.0, 0.0, 10.0, 10.0],
                         [20.0, 20.0, 30.0, 30.0]]), np.array([0, 0]))]
        pred = [(
            np.array([[0.0, 0.0, 10.0, 10.0],
                      [20.0, 20.0, 30.0, 30.0]]),
            np.array([0.9, 0.8]),
            np.array([0, 0]),
        )]
        value, per_class = mean_average_precision(
            pred, gt, return_per_class=True
        )
        assert value == pytest.approx(1.0)
        assert per_class[0] == pytest.approx(1.0)

    def test_one_false_positive_lowers_ap(self):
        gt = [(np.array([[0.0, 0.0, 10.0, 10.0],
                         [20.0, 20.0, 30.0, 30.0]]), np.array([0, 0]))]
        pred = [(
            np.array([[0.0, 0.0, 10.0, 10.0],
                      [40.0, 40.0, 50.0, 50.0],
                      [20.0, 20.0, 30.0, 30.0]]),
            np.array([0.9, 0.8, 0.7]),
            np.array([0, 0, 0]),
        )]
        value = mean_average_precision(pred, gt)
        assert value == pytest.approx(5.0 / 6.0)
        assert value < 1.0

    def test_missed_ground_truth_lowers_ap(self):
        gt = [(np.array([[0.0, 0.0, 10.0, 10.0],
                         [20.0, 20.0, 30.0, 30.0]]), np.array([0, 0]))]
        pred = [(
            np.array([[0.0, 0.0, 10.0, 10.0]]),
            np.array([0.9]),
            np.array([0]),
        )]
        assert mean_average_precision(pred, gt) == pytest.approx(0.5)

    def test_class_mismatch_is_false_positive(self):
        gt = [(np.array([[0.0, 0.0, 10.0, 10.0]]), np.array([1]))]
        pred = [(
            np.array([[0.0, 0.0, 10.0, 10.0]]),
            np.array([0.9]),
            np.array([0]),
        )]
        assert mean_average_precision(pred, gt) == pytest.approx(0.0)

    def test_voc_all_point_interpolation(self):
        # TP, FP, TP: precision at recall 0.5 is 1, at recall 1 is 2/3.
        ap = voc_average_precision(
            tp=np.array([True, False, True]),
            fp=np.array([False, True, False]),
            num_gt=2,
        )
        assert ap == pytest.approx(5.0 / 6.0)
        assert voc_average_precision([], [], 0) == 0.0

    def test_duplicate_detection_on_same_gt_is_a_false_positive(self):
        gt = [(np.array([[0.0, 0.0, 10.0, 10.0]]), np.array([0]))]
        pred = [(
            np.array([[0.0, 0.0, 10.0, 10.0],
                      [0.0, 0.0, 10.0, 10.0]]),
            np.array([0.9, 0.8]),
            np.array([0, 0]),
        )]
        # After the only GT is matched, recall is already 1.0, so the
        # envelope AP stays 1.0; the duplicate still counts as an FP.
        assert mean_average_precision(pred, gt) == pytest.approx(1.0)


class TestTraining:
    @pytest.mark.slow
    def test_loss_decreases_and_map_beats_trivial(self):
        images, annotations = generate_shelf_dataset(n_samples=48, seed=11)
        train_images, train_annotations = images[:36], annotations[:36]
        test_images, test_annotations = images[36:], annotations[36:]
        model = MiniYOLO(
            num_classes=3,
            grid_size=6,
            image_size=48,
            in_channels=1,
            base_channels=(16, 32),
            head_channels=32,
            seed=5,
        )
        history = model.fit(
            train_images,
            train_annotations,
            epochs=20,
            batch_size=8,
            lr=4e-3,
            seed=2,
        )
        assert len(history) == 20
        assert history[-1] < history[0] * 0.2

        predictions = model.predict(test_images, conf_threshold=0.1)
        map_value = mean_average_precision(predictions, test_annotations)
        assert map_value > 0.1

    def test_adam_updates_parameters(self):
        params = {"w": np.zeros(3)}
        grads = {"w": np.ones(3)}
        optimizer = Adam(lr=0.01)
        optimizer.step(params, grads)
        assert np.all(params["w"] < 0.0)
        assert np.any(np.isfinite(params["w"]))
