from __future__ import annotations

import tempfile
from pathlib import Path

import cv2
import numpy as np
import torch

from online_augmentation import (
    DEFAULT_HFLIP_PROB,
    DEFAULT_ONLINE_AUGMENTATION,
    apply_horizontal_flip,
    apply_online_augmentation,
    augment_image,
    flip_boxes_xyxy,
    flip_yolo_cxcywh,
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


def test_hsv_policy_is_deterministic_and_preserves_tensor_contract() -> None:
    """The HSV policy changes color appearance only and stays within normalized RGB bounds."""
    source = torch.linspace(0.05, 0.95, 3 * 8 * 8, dtype=torch.float32).reshape(3, 8, 8)
    original = source.clone()

    assert validate_online_augmentation("hsv") == "hsv"

    torch.manual_seed(4321)
    augmented = apply_online_augmentation(source, "hsv")
    torch.manual_seed(4321)
    repeated = apply_online_augmentation(source, "hsv")

    assert torch.equal(source, original)
    assert augmented.shape == source.shape
    assert augmented.dtype == source.dtype
    assert torch.isfinite(augmented).all()
    assert 0.0 <= float(augmented.min()) <= float(augmented.max()) <= 1.0
    assert not torch.equal(augmented, source)
    assert torch.equal(augmented, repeated)


def test_hsv_datasets_preserve_detection_targets() -> None:
    """HSV training augmentation must not alter either detector's bounding boxes or labels."""
    with tempfile.TemporaryDirectory() as temporary_directory:
        split_root = Path(temporary_directory) / "train"
        _write_detection_split(split_root)

        yolo_raw = YoloDetectionDataset(split_root, imgsz=32)
        yolo_hsv = YoloDetectionDataset(
            split_root,
            imgsz=32,
            online_augmentation="hsv",
        )
        raw_image, raw_labels = yolo_raw[0]
        torch.manual_seed(7)
        hsv_image, hsv_labels = yolo_hsv[0]
        assert torch.equal(raw_labels, hsv_labels)
        assert raw_image.shape == hsv_image.shape
        assert not torch.equal(raw_image, hsv_image)

        faster_raw = FasterRCNNDataset(split_root, imgsz=32, num_classes=2)
        faster_hsv = FasterRCNNDataset(
            split_root,
            imgsz=32,
            num_classes=2,
            online_augmentation="hsv",
        )
        raw_image, raw_target = faster_raw[0]
        torch.manual_seed(7)
        hsv_image, hsv_target = faster_hsv[0]
        assert torch.equal(raw_target["boxes"], hsv_target["boxes"])
        assert torch.equal(raw_target["labels"], hsv_target["labels"])
        assert raw_image.shape == hsv_image.shape
        assert not torch.equal(raw_image, hsv_image)


def test_flip_targets_match_the_mirrored_image() -> None:
    """The box transforms must be the exact mirror of the pixel transform."""
    width = 32
    boxes = torch.tensor(
        [[2.0, 4.0, 10.0, 12.0], [0.0, 0.0, 32.0, 16.0], [30.0, 8.0, 31.0, 9.0]],
        dtype=torch.float32,
    )
    mirrored = flip_boxes_xyxy(boxes, width)
    assert torch.allclose(mirrored[:, 0], width - boxes[:, 2])
    assert torch.allclose(mirrored[:, 2], width - boxes[:, 0])
    assert torch.equal(mirrored[:, 1], boxes[:, 1])
    assert torch.equal(mirrored[:, 3], boxes[:, 3])
    # A full-width box maps to itself and every box keeps a positive area.
    assert torch.allclose(mirrored[1], boxes[1])
    assert (mirrored[:, 2] > mirrored[:, 0]).all()

    yolo = torch.tensor([[0.0, 0.25, 0.5, 0.2, 0.1], [3.0, 0.8, 0.2, 0.05, 0.05]])
    mirrored_yolo = flip_yolo_cxcywh(yolo)
    assert torch.allclose(mirrored_yolo[:, 1], 1.0 - yolo[:, 1])
    assert torch.equal(mirrored_yolo[:, 0], yolo[:, 0])
    assert torch.equal(mirrored_yolo[:, 2:], yolo[:, 2:])
    assert ((mirrored_yolo[:, 1] >= 0.0) & (mirrored_yolo[:, 1] <= 1.0)).all()

    # Empty targets are handled without error.
    empty = torch.zeros((0, 4), dtype=torch.float32)
    assert flip_boxes_xyxy(empty, width).shape == (0, 4)
    assert flip_yolo_cxcywh(torch.zeros((0, 5), dtype=torch.float32)).shape == (0, 5)

    # Malformed target shapes are rejected instead of silently mis-transformed.
    for bad in (torch.zeros((2, 3)), torch.zeros((2, 5))):
        try:
            flip_boxes_xyxy(bad, width)
        except ValueError:
            pass
        else:
            raise AssertionError("Expected ValueError for non XYXY boxes")
    for bad in (torch.zeros((2, 4)), torch.zeros((2, 6))):
        try:
            flip_yolo_cxcywh(bad)
        except ValueError:
            pass
        else:
            raise AssertionError("Expected ValueError for non YOLO targets")


def test_hflip_policy_is_reproducible_and_respects_probability() -> None:
    """Flip decisions come from torch RNG, so a fixed seed reproduces them."""
    assert validate_online_augmentation("hflip") == "hflip"
    assert DEFAULT_HFLIP_PROB == 0.5

    source = torch.linspace(0.0, 1.0, 3 * 4 * 6, dtype=torch.float32).reshape(3, 4, 6)
    original = source.clone()

    # Probability 1.0 and 0.0 are deterministic and independent of the seed.
    torch.manual_seed(1)
    always, always_flipped = apply_horizontal_flip(source, probability=1.0)
    assert always_flipped is True
    assert torch.equal(always, source.flip(-1))
    assert torch.equal(source, original)

    torch.manual_seed(1)
    never, never_flipped = apply_horizontal_flip(source, probability=0.0)
    assert never_flipped is False
    assert torch.equal(never, source)

    torch.manual_seed(99)
    first, first_flipped = apply_horizontal_flip(source, probability=0.5)
    torch.manual_seed(99)
    second, second_flipped = apply_horizontal_flip(source, probability=0.5)
    assert first_flipped == second_flipped
    assert torch.equal(first, second)
    assert bool(first_flipped) == torch.equal(first, source.flip(-1))

    for invalid in (-0.1, 1.1):
        try:
            apply_horizontal_flip(source, probability=invalid)
        except ValueError:
            pass
        else:
            raise AssertionError("Expected ValueError for an out-of-range probability")


def test_augment_image_reports_geometry_changes_only_for_hflip() -> None:
    """The entry point flags geometric changes so callers can transform targets."""
    source = torch.rand(3, 8, 8)

    image, flipped = augment_image(source, "none")
    assert flipped is False
    assert torch.equal(image, source)

    torch.manual_seed(3)
    hsv_image, hsv_flipped = augment_image(source, "hsv")
    assert hsv_flipped is False
    assert hsv_image.shape == source.shape

    torch.manual_seed(3)
    flip_image, flip_flipped = augment_image(source, "hflip")
    assert flip_flipped is True
    assert torch.equal(flip_image, source.flip(-1))

    # The appearance-only helper must refuse policies it cannot keep consistent.
    try:
        apply_online_augmentation(source, "hflip")
    except ValueError:
        pass
    else:
        raise AssertionError("Expected the appearance-only helper to reject hflip")

    try:
        augment_image(torch.randint(0, 2, (3, 8, 8)), "hflip")
    except ValueError:
        pass
    else:
        raise AssertionError("Expected ValueError for a non-float image tensor")


def test_hflip_datasets_keep_targets_aligned_with_pixels() -> None:
    """A forced flip must move the boxes with the pixels, not leave them in place."""
    with tempfile.TemporaryDirectory() as temporary_directory:
        split_root = Path(temporary_directory) / "train"
        _write_detection_split(split_root)

        plain_yolo = YoloDetectionDataset(split_root, imgsz=32)
        plain_image, plain_labels = plain_yolo[0]
        plain_faster = FasterRCNNDataset(split_root, imgsz=32, num_classes=2)
        _, plain_target = plain_faster[0]
        yolo_hflip = YoloDetectionDataset(split_root, imgsz=32, online_augmentation="hflip")
        faster_hflip = FasterRCNNDataset(
            split_root, imgsz=32, num_classes=2, online_augmentation="hflip"
        )

        # A flip either happens or does not, and the targets must follow the pixels.
        torch.manual_seed(0)
        yolo_image, yolo_labels = yolo_hflip[0]
        yolo_flipped = torch.equal(yolo_image, plain_image.flip(-1))
        assert yolo_image.shape == plain_image.shape
        if yolo_flipped:
            assert torch.equal(yolo_labels, flip_yolo_cxcywh(plain_labels))
        else:
            assert torch.equal(yolo_image, plain_image)
            assert torch.equal(yolo_labels, plain_labels)

        torch.manual_seed(0)
        faster_image, faster_target = faster_hflip[0]
        faster_flipped = torch.equal(faster_image, plain_image.flip(-1))
        width = plain_image.shape[-1]
        if faster_flipped:
            assert torch.equal(faster_target["boxes"], flip_boxes_xyxy(plain_target["boxes"], width))
        else:
            assert torch.equal(faster_image, plain_image)
            assert torch.equal(faster_target["boxes"], plain_target["boxes"])
        # Class IDs are never changed by mirroring.
        assert torch.equal(faster_target["labels"], plain_target["labels"])


def test_hflip_datasets_are_reproducible_across_identical_seeds() -> None:
    """Two dataset reads under the same seed return the same image and boxes."""
    with tempfile.TemporaryDirectory() as temporary_directory:
        split_root = Path(temporary_directory) / "train"
        _write_detection_split(split_root)

        yolo = YoloDetectionDataset(split_root, imgsz=32, online_augmentation="hflip")
        torch.manual_seed(21)
        first_image, first_labels = yolo[0]
        torch.manual_seed(21)
        second_image, second_labels = yolo[0]
        assert torch.equal(first_image, second_image)
        assert torch.equal(first_labels, second_labels)

        faster = FasterRCNNDataset(
            split_root, imgsz=32, num_classes=2, online_augmentation="hflip"
        )
        torch.manual_seed(21)
        first_image, first_target = faster[0]
        torch.manual_seed(21)
        second_image, second_target = faster[0]
        assert torch.equal(first_image, second_image)
        assert torch.equal(first_target["boxes"], second_target["boxes"])
        assert torch.equal(first_target["labels"], second_target["labels"])

        # Datasets without augmentation stay identical under different seeds.
        plain = FasterRCNNDataset(split_root, imgsz=32, num_classes=2)
        torch.manual_seed(5)
        plain_image, plain_target = plain[0]
        torch.manual_seed(6)
        repeat_image, repeat_target = plain[0]
        assert torch.equal(plain_image, repeat_image)
        assert torch.equal(plain_target["boxes"], repeat_target["boxes"])


def main() -> None:
    test_flip_targets_match_the_mirrored_image()
    print("flip_target_geometry: passed")
    test_hflip_policy_is_reproducible_and_respects_probability()
    print("hflip_probability_and_seed: passed")
    test_augment_image_reports_geometry_changes_only_for_hflip()
    print("augment_image_geometry_flag: passed")
    test_hflip_datasets_keep_targets_aligned_with_pixels()
    print("hflip_dataset_alignment: passed")
    test_hflip_datasets_are_reproducible_across_identical_seeds()
    print("hflip_dataset_reproducibility: passed")
    test_photometric_policy_is_deterministic_and_preserves_tensor_contract()
    print("photometric_tensor_contract: passed")
    test_photometric_datasets_preserve_detection_targets()
    print("photometric_dataset_targets: passed")
    test_hsv_policy_is_deterministic_and_preserves_tensor_contract()
    print("hsv_tensor_contract: passed")
    test_hsv_datasets_preserve_detection_targets()
    print("hsv_dataset_targets: passed")


if __name__ == "__main__":
    main()

