from __future__ import annotations

import tempfile
from pathlib import Path

import cv2
import numpy as np
import torch

from image_geometry import (
    DEFAULT_LETTERBOX_PAD_VALUE,
    ResizeTransform,
    model_yolo_to_source_yolo,
    resize_image,
    resolve_resize_mode,
    restore_detections_to_source,
    source_yolo_to_model_yolo,
    validate_resize_mode,
)
from train_faster_rcnn import FasterRCNNDataset
from train_yolo26 import YoloDetectionDataset


def test_stretch_matches_historical_cv2_resize() -> None:
    """Stretch mode must retain the original direct square-resize image behavior."""
    image = np.arange(6 * 10 * 3, dtype=np.uint8).reshape(6, 10, 3)
    resized, transform = resize_image(image, 12, 12, resize_mode="stretch")
    expected = cv2.resize(image, (12, 12), interpolation=cv2.INTER_CUBIC)

    assert np.array_equal(resized, expected)
    assert transform.pad_top == transform.pad_left == 0
    assert transform.pad_bottom == transform.pad_right == 0
    assert transform.resized_height == transform.resized_width == 12


def test_letterbox_maps_boxes_and_labels_round_trip() -> None:
    """Letterbox padding and normalized labels must restore exactly in float space."""
    image = np.zeros((200, 400, 3), dtype=np.uint8)
    resized, transform = resize_image(image, 640, 640, resize_mode="letterbox")
    labels = torch.tensor([[2.0, 0.5, 0.5, 0.5, 0.5]], dtype=torch.float32)

    assert resized.shape == (640, 640, 3)
    assert transform.resized_width == 640
    assert transform.resized_height == 320
    assert transform.pad_top == transform.pad_bottom == 160
    assert np.all(resized[:160] == DEFAULT_LETTERBOX_PAD_VALUE)
    assert np.all(resized[160:480] == 0)

    model_labels = source_yolo_to_model_yolo(labels, transform)
    expected_model_labels = torch.tensor([[2.0, 0.5, 0.5, 0.5, 0.25]], dtype=torch.float32)
    assert torch.allclose(model_labels, expected_model_labels, atol=1e-6)
    assert torch.allclose(model_yolo_to_source_yolo(model_labels, transform), labels, atol=1e-6)


def test_letterbox_odd_padding_and_detection_restoration() -> None:
    """Odd padding and detection metadata survive source-model-source conversion."""
    transform = ResizeTransform.from_shapes(333, 640, 640, 640, resize_mode="letterbox")
    source_boxes = torch.tensor([[0.0, 0.0, 640.0, 333.0], [100.5, 30.25, 300.75, 250.5]])
    model_boxes = transform.source_xyxy_to_model_xyxy(source_boxes)
    detections = torch.cat(
        (model_boxes, torch.tensor([[0.9, 1.0], [0.2, 3.0]], dtype=torch.float32)),
        dim=1,
    )

    assert transform.pad_top == 153
    assert transform.pad_bottom == 154
    restored = restore_detections_to_source(detections, transform)
    assert torch.allclose(restored[:, :4], source_boxes, atol=1e-5)
    assert torch.equal(restored[:, 4:], detections[:, 4:])


def test_invalid_resize_mode_is_rejected() -> None:
    try:
        validate_resize_mode("crop")
    except ValueError as error:
        assert "resize_mode" in str(error)
    else:
        raise AssertionError("Expected invalid resize mode to fail")


def test_checkpoint_resize_mode_resolution_preserves_legacy_behavior() -> None:
    """New checkpoints restore their geometry while old metadata remains stretch."""
    assert resolve_resize_mode(None, {"resize_mode": "letterbox"}) == "letterbox"
    assert resolve_resize_mode("stretch", {"resize_mode": "letterbox"}) == "stretch"
    assert resolve_resize_mode(None, {}) == "stretch"
    assert resolve_resize_mode(None, None) == "stretch"


def test_yolo_dataset_resize_modes_preserve_or_transform_labels() -> None:
    """Dataset targets remain historical under stretch and map correctly under letterbox."""
    with tempfile.TemporaryDirectory() as temporary_directory:
        split_root = Path(temporary_directory)
        images_dir = split_root / "images"
        labels_dir = split_root / "labels"
        images_dir.mkdir()
        labels_dir.mkdir()
        image = np.zeros((200, 400, 3), dtype=np.uint8)
        assert cv2.imwrite(str(images_dir / "sample.jpg"), image)
        (labels_dir / "sample.txt").write_text("1 0.5 0.5 0.5 0.5\n", encoding="utf-8")

        stretch_image, stretch_labels = YoloDetectionDataset(
            split_root,
            imgsz=640,
            resize_mode="stretch",
        )[0]
        letterbox_image, letterbox_labels = YoloDetectionDataset(
            split_root,
            imgsz=640,
            resize_mode="letterbox",
        )[0]

    assert stretch_image.shape == letterbox_image.shape == (3, 640, 640)
    assert torch.allclose(stretch_labels, torch.tensor([[1.0, 0.5, 0.5, 0.5, 0.5]]))
    assert torch.allclose(letterbox_labels, torch.tensor([[1.0, 0.5, 0.5, 0.5, 0.25]]))


def test_faster_rcnn_dataset_resize_modes_transform_absolute_targets() -> None:
    """Faster R-CNN receives model-space absolute boxes from the shared transform."""
    with tempfile.TemporaryDirectory() as temporary_directory:
        split_root = Path(temporary_directory)
        images_dir = split_root / "images"
        labels_dir = split_root / "labels"
        images_dir.mkdir()
        labels_dir.mkdir()
        image = np.zeros((200, 400, 3), dtype=np.uint8)
        assert cv2.imwrite(str(images_dir / "sample.jpg"), image)
        (labels_dir / "sample.txt").write_text("1 0.5 0.5 0.5 0.5\n", encoding="utf-8")

        _, stretch_target = FasterRCNNDataset(
            split_root,
            imgsz=640,
            num_classes=2,
            resize_mode="stretch",
        )[0]
        _, letterbox_target = FasterRCNNDataset(
            split_root,
            imgsz=640,
            num_classes=2,
            resize_mode="letterbox",
        )[0]

    assert torch.allclose(stretch_target["boxes"], torch.tensor([[160.0, 160.0, 480.0, 480.0]]))
    assert torch.allclose(letterbox_target["boxes"], torch.tensor([[160.0, 240.0, 480.0, 400.0]]))
    assert stretch_target["labels"].tolist() == letterbox_target["labels"].tolist() == [2]


def main() -> None:
    test_stretch_matches_historical_cv2_resize()
    print("stretch_historical_compatibility: passed")
    test_letterbox_maps_boxes_and_labels_round_trip()
    print("letterbox_label_round_trip: passed")
    test_letterbox_odd_padding_and_detection_restoration()
    print("letterbox_detection_restoration: passed")
    test_invalid_resize_mode_is_rejected()
    print("resize_mode_validation: passed")
    test_checkpoint_resize_mode_resolution_preserves_legacy_behavior()
    print("checkpoint_resize_mode_resolution: passed")
    test_yolo_dataset_resize_modes_preserve_or_transform_labels()
    print("yolo_dataset_resize_modes: passed")
    test_faster_rcnn_dataset_resize_modes_transform_absolute_targets()
    print("faster_rcnn_dataset_resize_modes: passed")


if __name__ == "__main__":
    main()