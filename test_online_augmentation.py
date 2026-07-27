from __future__ import annotations

import tempfile
from pathlib import Path

import cv2
import numpy as np
import torch

from online_augmentation import (
    DEFAULT_ONLINE_AUGMENTATION,
    apply_online_augmentation,
    validate_online_augmentation,
)
from train_faster_rcnn import FasterRCNNDataset
from train_yolo26 import YoloDetectionDataset


def _write_detection_split(root: Path) -> None:
    images_dir = root / "images"
    labels_dir = root / "labels"
    images_dir.mkdir(parents=True)
    labels_dir.mkdir(parents=True)

    horizontal = np.linspace(10, 245, 24, dtype=np.uint8)
    vertical = np.linspace(245, 10, 24, dtype=np.uint8)
    image = np.stack(
        (
            np.tile(horizontal, (24, 1)),
            np.tile(vertical[:, None], (1, 24)),
            np.full((24, 24), 127, dtype=np.uint8),
        ),
        axis=2,
    )
    assert cv2.imwrite(str(images_dir / "sample.png"), cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
    (labels_dir / "sample.txt").write_text("1 0.5 0.5 0.5 0.5\n", encoding="utf-8")


def test_photometric_policy_is_deterministic_and_preserves_tensor_contract() -> None:
    """The fixed policy changes appearance only and stays in normalized RGB bounds."""
    source = torch.linspace(0.05, 0.95, 3 * 8 * 8, dtype=torch.float32).reshape(3, 8, 8)
    original = source.clone()

    assert validate_online_augmentation(DEFAULT_ONLINE_AUGMENTATION) == "none"
    assert torch.equal(apply_online_augmentation(source, "none"), source)

    torch.manual_seed(1234)
    augmented = apply_online_augmentation(source, "photometric")
    torch.manual_seed(1234)
    repeated = apply_online_augmentation(source, "photometric")

    assert torch.equal(source, original)
    assert augmented.shape == source.shape
    assert augmented.dtype == source.dtype
    assert torch.isfinite(augmented).all()
    assert 0.0 <= float(augmented.min()) <= float(augmented.max()) <= 1.0
    assert not torch.equal(augmented, source)
    assert torch.equal(augmented, repeated)

    try:
        validate_online_augmentation("geometric")
    except ValueError:
        pass
    else:
        raise AssertionError("Unsupported online augmentation policy was accepted")


def test_photometric_datasets_preserve_detection_targets() -> None:
    """Training-only appearance augmentation must not alter either detector's targets."""
    with tempfile.TemporaryDirectory() as temporary_directory:
        split_root = Path(temporary_directory) / "train"
        _write_detection_split(split_root)

        yolo_raw = YoloDetectionDataset(split_root, imgsz=32)
        yolo_augmented = YoloDetectionDataset(
            split_root,
            imgsz=32,
            online_augmentation="photometric",
        )
        raw_image, raw_labels = yolo_raw[0]
        torch.manual_seed(5)
        augmented_image, augmented_labels = yolo_augmented[0]
        assert torch.equal(raw_labels, augmented_labels)
        assert raw_image.shape == augmented_image.shape
        assert not torch.equal(raw_image, augmented_image)

        faster_raw = FasterRCNNDataset(split_root, imgsz=32, num_classes=2)
        faster_augmented = FasterRCNNDataset(
            split_root,
            imgsz=32,
            num_classes=2,
            online_augmentation="photometric",
        )
        raw_image, raw_target = faster_raw[0]
        torch.manual_seed(5)
        augmented_image, augmented_target = faster_augmented[0]
        assert torch.equal(raw_target["boxes"], augmented_target["boxes"])
        assert torch.equal(raw_target["labels"], augmented_target["labels"])
        assert raw_image.shape == augmented_image.shape
        assert not torch.equal(raw_image, augmented_image)


def main() -> None:
    test_photometric_policy_is_deterministic_and_preserves_tensor_contract()
    print("photometric_tensor_contract: passed")
    test_photometric_datasets_preserve_detection_targets()
    print("photometric_dataset_targets: passed")


if __name__ == "__main__":
    main()
