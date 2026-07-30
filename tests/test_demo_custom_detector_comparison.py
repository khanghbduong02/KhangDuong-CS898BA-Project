from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np
import torch

from demo_custom_detector_comparison import (
    HEADER_HEIGHT,
    PREDICTION_COLORS_BGR,
    TITLE_HEIGHT,
    _filter_display_detections,
    _to_zero_indexed_detection_tensor,
    _validate_scratch_faster_rcnn_checkpoint,
    compose_comparison_grid,
    select_comparison_images,
)


def test_comparison_grid_panel_order() -> None:
    """The saved comparison row is Ground Truth | YOLO26 | Faster R-CNN."""
    image = np.zeros((32, 40, 3), dtype=np.uint8)
    ground_truth = [
        {"class_id": 0, "class_name": "spaghetti", "xyxy": [4.0, 4.0, 20.0, 20.0]}
    ]
    yolo_detections = torch.tensor([[4.0, 4.0, 20.0, 20.0, 0.9, 1.0]])
    faster_rcnn_detections = torch.tensor([[4.0, 4.0, 20.0, 20.0, 0.8, 2.0]])

    grid = compose_comparison_grid(
        image,
        ground_truth,
        yolo_detections,
        faster_rcnn_detections,
        ("spaghetti", "layer_cracking", "over_extrusion"),
    )

    assert grid.shape == (32 + HEADER_HEIGHT + TITLE_HEIGHT, 40 * 3, 3)
    comparison_row = grid[TITLE_HEIGHT:, :, :]
    ground_truth_panel = comparison_row[:, :40]
    yolo_panel = comparison_row[:, 40:80]
    faster_rcnn_panel = comparison_row[:, 80:]
    assert np.any(np.all(ground_truth_panel == (255, 255, 255), axis=2))
    assert np.any(np.all(yolo_panel == PREDICTION_COLORS_BGR[1], axis=2))
    assert np.any(np.all(faster_rcnn_panel == PREDICTION_COLORS_BGR[2], axis=2))


def test_faster_rcnn_labels_and_display_filter() -> None:
    """Faster R-CNN foreground labels convert back to zero-indexed display IDs."""
    prediction = {
        "boxes": torch.tensor([[1.0, 2.0, 3.0, 4.0], [5.0, 6.0, 7.0, 8.0]]),
        "scores": torch.tensor([0.2, 0.8]),
        "labels": torch.tensor([1, 3]),
    }
    detections = _to_zero_indexed_detection_tensor(prediction)
    assert detections[:, 5].long().tolist() == [0, 2]
    displayed = _filter_display_detections(detections, confidence_threshold=0.25)
    assert displayed.shape == (1, 6)
    assert displayed[0, 5].item() == 2.0


def test_scratch_faster_rcnn_checkpoint_guard() -> None:
    """ImageNet-initialized Faster R-CNN artifacts cannot enter the scratch grid."""
    _validate_scratch_faster_rcnn_checkpoint({"args": {"backbone_weights": "none"}})

    try:
        _validate_scratch_faster_rcnn_checkpoint(
            {
                "args": {"backbone_weights": "imagenet"},
                "backbone_initialization": "imagenet",
            }
        )
    except ValueError as error:
        assert "randomly initialized" in str(error)
    else:
        raise AssertionError("Expected the scratch-only checkpoint guard to reject ImageNet initialization")


def test_ground_truth_coverage_image_selection() -> None:
    """Coverage sampling uses strict labels only and keeps a deterministic order."""
    with tempfile.TemporaryDirectory() as temporary_directory:
        root = Path(temporary_directory)
        source = root / "images"
        labels = root / "labels"
        source.mkdir()
        labels.mkdir()
        for name, contents in {
            "a.jpg": "0 0.5 0.5 0.2 0.2\n",
            "b.jpg": "1 0.5 0.5 0.2 0.2\n2 0.5 0.5 0.2 0.2\n",
            "c.jpg": "0 0.5 0.5 0.2 0.2\n",
            "d.jpg": "3 0.5 0.5 0.2 0.2\n",
            "e.jpg": "4 0.5 0.5 0.2 0.2\n",
        }.items():
            (source / name).touch()
            (labels / f"{Path(name).stem}.txt").write_text(contents, encoding="utf-8")

        selected = select_comparison_images(
            source,
            labels,
            max_images=4,
            image_names=None,
            image_selection="ground_truth_coverage",
            num_classes=5,
        )
        assert [path.name for path in selected] == ["a.jpg", "b.jpg", "d.jpg", "e.jpg"]


def main() -> None:
    test_comparison_grid_panel_order()
    print("comparison_grid_panel_order: passed")
    test_faster_rcnn_labels_and_display_filter()
    print("faster_rcnn_display_conversion: passed")
    test_scratch_faster_rcnn_checkpoint_guard()
    print("scratch_faster_rcnn_checkpoint_guard: passed")
    test_ground_truth_coverage_image_selection()
    print("ground_truth_coverage_image_selection: passed")


if __name__ == "__main__":
    main()
