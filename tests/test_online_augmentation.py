from __future__ import annotations

import tempfile
from pathlib import Path

import cv2
import numpy as np
import torch

from online_augmentation import (
    AFFINE_BORDER_VALUE,
    DEFAULT_AFFINE_DEGREES,
    DEFAULT_HFLIP_PROB,
    DEFAULT_ONLINE_AUGMENTATION,
    apply_affine,
    apply_horizontal_flip,
    apply_online_augmentation,
    augment_image,
    flip_boxes_xyxy,
    flip_yolo_cxcywh,
    horizontal_flip_matrix,
    identity_matrix,
    sample_affine_matrix,
    transform_boxes_xyxy,
    transform_yolo_cxcywh,
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


def test_augment_image_returns_identity_matrix_for_appearance_policies() -> None:
    """Appearance-only policies must report an identity target transform."""
    source = torch.rand(3, 8, 8)

    image, matrix = augment_image(source, "none")
    assert torch.equal(image, source)
    assert torch.equal(matrix, identity_matrix())

    torch.manual_seed(3)
    hsv_image, hsv_matrix = augment_image(source, "hsv")
    assert hsv_image.shape == source.shape
    assert torch.equal(hsv_matrix, identity_matrix())

    torch.manual_seed(3)
    flip_image, flip_matrix = augment_image(source, "hflip")
    flipped = not torch.equal(flip_matrix, identity_matrix())
    if flipped:
        assert torch.equal(flip_matrix, horizontal_flip_matrix(8))
        assert torch.equal(flip_image, source.flip(-1))
    else:
        assert torch.equal(flip_image, source)

    # The appearance-only helper must refuse policies it cannot keep consistent.
    for policy in ("hflip", "affine"):
        try:
            apply_online_augmentation(source, policy)
        except ValueError:
            pass
        else:
            raise AssertionError(f"Expected the appearance-only helper to reject {policy}")

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


def test_affine_matrix_maps_the_image_centre_to_itself() -> None:
    """A zero-parameter affine must be the identity, and the centre must be preserved."""
    height, width = 64, 48

    torch.manual_seed(11)
    identity = sample_affine_matrix(height, width, 0.0, 0.0, 0.0, 0.0)
    assert torch.allclose(identity, identity_matrix(), atol=1e-6)

    torch.manual_seed(11)
    matrix = sample_affine_matrix(height, width, translate=0.0)
    centre = torch.tensor([[width / 2.0, height / 2.0, 1.0]], dtype=torch.float32)
    # With no translation, rotation and scale are applied about the centre.
    assert torch.allclose((matrix @ centre.T).T, centre, atol=1e-3)

    # A sampled translation must move the centre by exactly that offset.
    torch.manual_seed(11)
    shifted = sample_affine_matrix(height, width, degrees=0.0, scale=0.0, shear=0.0)
    moved = (shifted @ centre.T).T
    assert torch.allclose(moved[:, :2], centre[:, :2] + shifted[:2, 2], atol=1e-3)

    # Scale-only sampling changes the box size by roughly the sampled factor.
    torch.manual_seed(5)
    scale_m = sample_affine_matrix(100, 100, 0.0, 0.0, 0.2, 0.0)
    factor = float(scale_m[0, 0])
    assert 0.8 <= factor <= 1.2
    assert abs(float(scale_m[0, 1])) < 1e-6

    for bad in ((-1.0, 0.05, 0.1, 0.0), (5.0, 1.5, 0.1, 0.0), (5.0, 0.05, 1.5, 0.0)):
        try:
            sample_affine_matrix(64, 64, *bad)
        except ValueError:
            pass
        else:
            raise AssertionError(f"Expected ValueError for affine parameters {bad}")


def test_affine_box_transform_clips_and_filters_degenerate_boxes() -> None:
    """Boxes must move with the pixels, stay inside the frame, and drop degenerates."""
    height, width = 64, 64
    boxes = torch.tensor(
        [[8.0, 8.0, 24.0, 24.0], [0.0, 0.0, 4.0, 4.0], [30.0, 30.0, 62.0, 62.0]],
        dtype=torch.float32,
    )

    unchanged, identity_keep = transform_boxes_xyxy(
        boxes, identity_matrix(), height, width
    )
    assert torch.allclose(unchanged, boxes, atol=1e-4)
    assert bool(identity_keep.all())

    # A large translation drops the boxes that leave the frame entirely.
    shift = identity_matrix()
    shift[0, 2] = -30.0
    shifted, keep = transform_boxes_xyxy(boxes, shift, height, width)
    assert shifted.shape == boxes.shape
    for index in range(boxes.shape[0]):
        if bool(keep[index]):
            assert shifted[index][2] > shifted[index][0]
            assert float(shifted[index][0]) >= 0.0
            assert float(shifted[index][2]) <= width
        else:
            # A dropped box is degenerate after clipping.
            assert not (shifted[index][2] > shifted[index][0])

    # The centre box moves fully off frame, so it must be dropped.
    assert bool(keep[0]) is False

    # A smaller translation keeps the central box and still clips it in frame.
    nudge = identity_matrix()
    nudge[0, 2] = -5.0
    nudged, nudge_keep = transform_boxes_xyxy(boxes, nudge, height, width)
    assert bool(nudge_keep[0])
    assert float(nudged[0][0]) >= 0.0
    assert float(nudged[0][2]) <= width

    for seed in range(6):
        torch.manual_seed(seed)
        matrix = sample_affine_matrix(height, width)
        out, mask = transform_boxes_xyxy(boxes, matrix, height, width)
        assert mask.shape == (3,)
        if bool(mask.any()):
            kept = out[mask]
            assert float(kept[:, 0].min()) >= 0.0
            assert float(kept[:, 1].min()) >= 0.0
            assert float(kept[:, 2].max()) <= width
            assert float(kept[:, 3].max()) <= height
            assert (kept[:, 2] > kept[:, 0]).all()
            assert (kept[:, 3] > kept[:, 1]).all()

    # Rotation grows the axis-aligned envelope, which is the correct behaviour.
    torch.manual_seed(2)
    rotated = sample_affine_matrix(
        height, width, degrees=45.0, translate=0.0, scale=0.0, shear=0.0
    )
    rotated_out, _ = transform_boxes_xyxy(boxes, rotated, height, width)
    original_width = float(boxes[2, 2] - boxes[2, 0])
    assert float(rotated_out[2, 2] - rotated_out[2, 0]) >= original_width - 1e-3

    empty = torch.zeros((0, 4), dtype=torch.float32)
    out, mask = transform_boxes_xyxy(empty, identity_matrix(), height, width)
    assert out.shape == (0, 4) and mask.shape == (0,)


def test_affine_yolo_targets_stay_normalized_and_preserve_classes() -> None:
    """Normalized targets must survive the matrix transform inside the unit square."""
    height, width = 64, 64
    targets = torch.tensor(
        [[0.0, 0.5, 0.5, 0.4, 0.4], [3.0, 0.2, 0.8, 0.1, 0.1]], dtype=torch.float32
    )

    unchanged, keep = transform_yolo_cxcywh(
        targets, identity_matrix(), height, width
    )
    assert torch.allclose(unchanged, targets, atol=1e-4)
    assert bool(keep.all())

    for seed in range(6):
        torch.manual_seed(seed)
        matrix = sample_affine_matrix(height, width)
        out, mask = transform_yolo_cxcywh(targets, matrix, height, width)
        assert mask.shape == (2,)
        if bool(mask.any()):
            kept = out[mask]
            assert (kept[:, 1] >= 0.0).all() and (kept[:, 1] <= 1.0).all()
            assert (kept[:, 2] >= 0.0).all() and (kept[:, 2] <= 1.0).all()
            assert (kept[:, 3] > 0.0).all() and (kept[:, 4] > 0.0).all()
            assert torch.equal(kept[:, 0], targets[mask][:, 0])

    empty = torch.zeros((0, 5), dtype=torch.float32)
    out, mask = transform_yolo_cxcywh(empty, identity_matrix(), height, width)
    assert out.shape == (0, 5) and mask.shape == (0,)


def test_affine_warp_is_reproducible_and_stays_in_range() -> None:
    """The warp must honour its probability and keep normalized RGB bounds."""
    assert validate_online_augmentation("affine") == "affine"
    assert 0.0 < DEFAULT_AFFINE_DEGREES <= 90.0

    image = torch.rand(3, 32, 32)

    torch.manual_seed(7)
    warped, matrix = augment_image(image, "affine")
    assert warped.shape == image.shape
    assert torch.isfinite(warped).all()
    assert float(warped.min()) >= 0.0 and float(warped.max()) <= 1.0

    torch.manual_seed(7)
    repeated, repeated_matrix = augment_image(image, "affine")
    assert torch.equal(warped, repeated)
    assert torch.equal(matrix, repeated_matrix)

    # A forced translation must still return a valid matrix and in-range pixels.
    white = torch.ones(3, 32, 32)
    padded, padded_matrix = apply_affine(white, probability=1.0, translate=0.45, scale=0.0)
    assert padded.shape == white.shape
    assert padded_matrix.shape == (3, 3)
    assert float(padded.min()) >= 0.0 and float(padded.max()) <= 1.0

    # Regions translated out of view are padded with mid-gray, matching Ultralytics.
    gray = AFFINE_BORDER_VALUE / 255.0
    assert gray < 1.0
    assert abs(float(padded.min()) - gray) < 0.02

    try:
        apply_affine(image, probability=1.5)
    except ValueError:
        pass
    else:
        raise AssertionError("Expected ValueError for an out-of-range probability")


def test_affine_datasets_keep_targets_aligned_and_are_reproducible() -> None:
    """Both trainers must return in-range targets that line up with the warped pixels."""
    with tempfile.TemporaryDirectory() as temporary_directory:
        split_root = Path(temporary_directory) / "train"
        _write_detection_split(split_root)

        plain_yolo = YoloDetectionDataset(split_root, imgsz=32)
        plain_image, plain_labels = plain_yolo[0]
        plain_classes = set(plain_labels[:, 0].tolist())
        yolo_affine = YoloDetectionDataset(
            split_root, imgsz=32, online_augmentation="affine"
        )
        faster_affine = FasterRCNNDataset(
            split_root, imgsz=32, num_classes=2, online_augmentation="affine"
        )

        for seed in range(6):
            torch.manual_seed(seed)
            image, labels = yolo_affine[0]
            assert image.shape == plain_image.shape
            assert torch.isfinite(image).all()
            assert float(image.min()) >= 0.0 and float(image.max()) <= 1.0
            if labels.numel():
                assert (labels[:, 1] >= 0.0).all() and (labels[:, 1] <= 1.0).all()
                assert (labels[:, 2] >= 0.0).all() and (labels[:, 2] <= 1.0).all()
                assert (labels[:, 3] > 0.0).all() and (labels[:, 4] > 0.0).all()
                # Class IDs are never invented or remapped.
                assert set(labels[:, 0].tolist()).issubset(plain_classes)

            torch.manual_seed(seed)
            faster_image, faster_target = faster_affine[0]
            assert faster_image.shape == plain_image.shape
            boxes = faster_target["boxes"]
            if boxes.numel():
                assert float(boxes[:, 0].min()) >= 0.0
                assert float(boxes[:, 1].min()) >= 0.0
                assert float(boxes[:, 2].max()) <= 32
                assert float(boxes[:, 3].max()) <= 32
                assert (boxes[:, 2] > boxes[:, 0]).all()
                assert (boxes[:, 3] > boxes[:, 1]).all()
            # Boxes and labels always stay the same length after filtering.
            assert boxes.shape[0] == faster_target["labels"].shape[0]

        torch.manual_seed(31)
        first_image, first_labels = yolo_affine[0]
        torch.manual_seed(31)
        second_image, second_labels = yolo_affine[0]
        assert torch.equal(first_image, second_image)
        assert torch.equal(first_labels, second_labels)


def test_flip_matrix_is_a_reflection_not_a_translation() -> None:
    """Regression: the flip matrix must mirror, and must round-trip to the identity."""
    width, height = 32, 24
    matrix = horizontal_flip_matrix(width)
    # A true reflection flips the sign of the x scale term.
    assert float(matrix[0, 0]) == -1.0
    assert float(matrix[1, 1]) == 1.0

    corners = torch.tensor([[0.0, 0.0, 1.0], [31.0, 0.0, 1.0]], dtype=torch.float32)
    mapped = (matrix @ corners.T).T
    assert torch.allclose(mapped[0], torch.tensor([32.0, 0.0, 1.0]))
    assert torch.allclose(mapped[1], torch.tensor([1.0, 0.0, 1.0]))

    # A full-width box must map onto itself through the generic matrix path.
    full = torch.tensor([[0.0, 2.0, 32.0, 10.0]], dtype=torch.float32)
    out, keep = transform_boxes_xyxy(full, matrix, height, width)
    assert torch.allclose(out, full, atol=1e-4)
    assert bool(keep.all())

    # Flipping twice must be the identity.
    squared = matrix @ matrix
    assert torch.allclose(squared, identity_matrix(), atol=1e-6)


def main() -> None:
    test_flip_targets_match_the_mirrored_image()
    print("flip_target_geometry: passed")
    test_hflip_policy_is_reproducible_and_respects_probability()
    print("hflip_probability_and_seed: passed")
    test_augment_image_returns_identity_matrix_for_appearance_policies()
    print("augment_image_identity_matrix: passed")
    test_hflip_datasets_keep_targets_aligned_with_pixels()
    print("hflip_dataset_alignment: passed")
    test_hflip_datasets_are_reproducible_across_identical_seeds()
    print("hflip_dataset_reproducibility: passed")
    test_flip_matrix_is_a_reflection_not_a_translation()
    print("flip_matrix_reflection: passed")
    test_affine_matrix_maps_the_image_centre_to_itself()
    print("affine_matrix_geometry: passed")
    test_affine_box_transform_clips_and_filters_degenerate_boxes()
    print("affine_box_transform: passed")
    test_affine_yolo_targets_stay_normalized_and_preserve_classes()
    print("affine_yolo_targets: passed")
    test_affine_warp_is_reproducible_and_stays_in_range()
    print("affine_warp_contract: passed")
    test_affine_datasets_keep_targets_aligned_and_are_reproducible()
    print("affine_dataset_alignment: passed")
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

