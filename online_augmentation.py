"""Conservative in-memory augmentation for local detector training.

The photometric and hsv policies change pixel appearance only. They never change
image geometry, boxes, labels, source files, or class sampling frequencies.

The hflip policy is the first geometry-changing policy. It mirrors the image and
transforms the model-space targets with the matching coordinate rule. Callers must
use :func:`augment_image` and forward the returned ``flipped`` flag to
:func:`flip_boxes_xyxy` or :func:`flip_yolo_cxcywh` so that boxes stay consistent
with the mirrored image.
"""
from __future__ import annotations

import math
from typing import Tuple

import cv2
import numpy as np
import torch


DEFAULT_ONLINE_AUGMENTATION = "none"
ONLINE_AUGMENTATION_CHOICES = ("none", "photometric", "hsv", "hflip", "affine")

# Official Ultralytics horizontal flip probability (fliplr in the default dataset config)
DEFAULT_HFLIP_PROB = 0.5

GEOMETRIC_ONLINE_AUGMENTATIONS = ("hflip", "affine")

# Deliberately gentler than the Ultralytics defaults (degrees=0.0, translate=0.1,
# scale=0.5). This dataset is small and the rare defect classes are already fragile,
# so the first affine test stays small on purpose.
DEFAULT_AFFINE_DEGREES = 5.0
DEFAULT_AFFINE_TRANSLATE = 0.05
DEFAULT_AFFINE_SCALE = 0.10
DEFAULT_AFFINE_SHEAR = 0.0
DEFAULT_AFFINE_PROB = 0.5

# Ultralytics pads warped regions with mid-gray.
AFFINE_BORDER_VALUE = 114

# Boxes whose clipped area falls below this fraction of the original area are dropped,
# mirroring the Ultralytics degenerate-box filter.
MIN_AFFINE_BOX_AREA_FRACTION = 0.01

PHOTOMETRIC_BRIGHTNESS_DELTA = 0.10
PHOTOMETRIC_CONTRAST_DELTA = 0.10
PHOTOMETRIC_GAMMA_DELTA = 0.10
PHOTOMETRIC_NOISE_STD = 0.01

# Official Ultralytics HSV default hyperparameters
DEFAULT_HSV_H = 0.015  # Hue gain fraction of 180 degrees
DEFAULT_HSV_S = 0.70   # Saturation variation fraction
DEFAULT_HSV_V = 0.40   # Value variation fraction


def validate_online_augmentation(mode: str) -> str:
    """Validate and normalize a training-only augmentation policy."""
    if mode not in ONLINE_AUGMENTATION_CHOICES:
        raise ValueError(
            f"--online-augmentation must be one of {ONLINE_AUGMENTATION_CHOICES}, got {mode!r}"
        )
    return mode


def _sample_symmetric_factor(delta: float, image: torch.Tensor) -> torch.Tensor:
    """Draw a scalar uniformly from ``[1 - delta, 1 + delta]``."""
    return 1.0 + (2.0 * torch.rand((), device=image.device, dtype=image.dtype) - 1.0) * delta


def apply_hsv_augmentation(
    image: torch.Tensor,
    hgain: float = DEFAULT_HSV_H,
    sgain: float = DEFAULT_HSV_S,
    vgain: float = DEFAULT_HSV_V,
) -> torch.Tensor:
    """Apply Ultralytics-aligned HSV jitter to a normalized RGB float tensor in [0, 1].

    Implements the official Ultralytics RandomHSV transform:
      - Uses torch random number generation for reproducibility.
      - Converts image to uint8 RGB, transforms to HSV via OpenCV.
      - Constructs lookup tables (LUT) for Hue, Saturation, and Value channels.
      - Sets Saturation[0] = 0 to prevent pure white background color casting.
      - Converts back to RGB and returns a normalized float tensor matching image dtype/device.
    """
    if hgain <= 0.0 and sgain <= 0.0 and vgain <= 0.0:
        return image

    device = image.device
    dtype = image.dtype

    # Sample gains using torch on the image's device/generator state
    r = (2.0 * torch.rand(3, device=device, dtype=torch.float32) - 1.0) * torch.tensor(
        [hgain, sgain, vgain], device=device, dtype=torch.float32
    )
    r_np = r.cpu().numpy()

    # Move image to CPU uint8 for OpenCV LUT operations
    img_cpu = image.detach().to(device="cpu", dtype=torch.float32)
    img_np = (img_cpu.permute(1, 2, 0).contiguous().numpy() * 255.0).clip(0, 255).astype(np.uint8)

    x = np.arange(0, 256, dtype=r_np.dtype)
    lut_hue = ((x + r_np[0] * 180.0) % 180.0).astype(np.uint8)
    lut_sat = np.clip(x * (r_np[1] + 1.0), 0.0, 255.0).astype(np.uint8)
    lut_val = np.clip(x * (r_np[2] + 1.0), 0.0, 255.0).astype(np.uint8)
    lut_sat[0] = 0

    hue, sat, val = cv2.split(cv2.cvtColor(img_np, cv2.COLOR_RGB2HSV))
    im_hsv = cv2.merge((cv2.LUT(hue, lut_hue), cv2.LUT(sat, lut_sat), cv2.LUT(val, lut_val)))
    img_rgb = cv2.cvtColor(im_hsv, cv2.COLOR_HSV2RGB)

    result = torch.from_numpy(img_rgb).permute(2, 0, 1).to(device=device, dtype=dtype) / 255.0
    return result.clamp(0.0, 1.0)


def apply_horizontal_flip(
    image: torch.Tensor,
    probability: float = DEFAULT_HFLIP_PROB,
) -> Tuple[torch.Tensor, bool]:
    """Mirror a normalized RGB tensor along the width axis with probability ``probability``.

    Returns the (possibly mirrored) tensor and a flag telling the caller whether the
    flip was actually applied. The decision uses torch random number generation so it
    is reproducible under a fixed seed.
    """
    if not 0.0 <= probability <= 1.0:
        raise ValueError("hflip probability must be in [0, 1]")

    draw = torch.rand((), device=image.device)
    if float(draw) >= probability:
        return image, False
    return image.flip(-1), True


def flip_boxes_xyxy(boxes: torch.Tensor, width: int) -> torch.Tensor:
    """Mirror absolute ``(N, 4)`` XYXY boxes across an image of the given width.

    Uses the half-open pixel convention ``x' = width - x`` so that a box covering the
    full image maps to itself and boxes keep a positive area.
    """
    if boxes.numel() == 0:
        return boxes
    if boxes.shape[-1] != 4:
        raise ValueError(f"Expected (N, 4) XYXY boxes, got {tuple(boxes.shape)}")

    mirrored = boxes.clone()
    x1 = boxes[:, 0].clone()
    x2 = boxes[:, 2].clone()
    mirrored[:, 0] = float(width) - x2
    mirrored[:, 2] = float(width) - x1
    return mirrored


def flip_yolo_cxcywh(boxes: torch.Tensor) -> torch.Tensor:
    """Mirror normalized ``(N, 5)`` ``(class, cx, cy, w, h)`` YOLO targets.

    Only the x center changes: ``cx' = 1 - cx``. Width, height, and class IDs are
    unchanged, and the box stays inside the unit square.
    """
    if boxes.numel() == 0:
        return boxes
    if boxes.shape[-1] != 5:
        raise ValueError(f"Expected (N, 5) YOLO targets, got {tuple(boxes.shape)}")

    mirrored = boxes.clone()
    mirrored[:, 1] = 1.0 - boxes[:, 1]
    return mirrored


def identity_matrix() -> torch.Tensor:
    """Return the 3x3 identity forward transform."""
    return torch.eye(3, dtype=torch.float32)


def horizontal_flip_matrix(width: int) -> torch.Tensor:
    """Return the 3x3 forward matrix mirroring an image of the given width.

    The linear part is ``x' = -x + width``, a true reflection rather than a
    translation, so a box spanning the full width maps back onto itself.
    """
    matrix = torch.eye(3, dtype=torch.float32)
    matrix[0, 0] = -1.0
    matrix[0, 2] = float(width)
    return matrix


def sample_affine_matrix(
    height: int,
    width: int,
    degrees: float = DEFAULT_AFFINE_DEGREES,
    translate: float = DEFAULT_AFFINE_TRANSLATE,
    scale: float = DEFAULT_AFFINE_SCALE,
    shear: float = DEFAULT_AFFINE_SHEAR,
) -> torch.Tensor:
    """Sample a 3x3 forward affine matrix mapping source pixels to destination pixels.

    The matrix is composed around the image centre in the Ultralytics
    ``RandomPerspective`` style: translate * scale * rotate * shear. All random draws
    use torch generators so results are seed-reproducible.
    """
    if degrees < 0.0 or not 0.0 <= translate < 1.0 or not 0.0 <= scale <= 1.0 or shear < 0.0:
        raise ValueError("Invalid affine parameter range")

    def draw(low: float, high: float) -> float:
        return float(low + (high - low) * torch.rand(()))

    rotation = math.radians(draw(-degrees, degrees))
    shear_rad = math.radians(draw(-shear, shear))
    scale_factor = draw(1.0 - scale, 1.0 + scale)
    shift_x = draw(-translate, translate) * width
    shift_y = draw(-translate, translate) * height

    cos_r, sin_r = math.cos(rotation), math.sin(rotation)
    cos_s, sin_s = math.cos(shear_rad), math.sin(shear_rad)

    rotation_m = torch.tensor(
        [[cos_r, -sin_r, 0.0], [sin_r, cos_r, 0.0], [0.0, 0.0, 1.0]],
        dtype=torch.float32,
    )
    shear_m = torch.tensor(
        [[1.0, -sin_s, 0.0], [0.0, cos_s, 0.0], [0.0, 0.0, 1.0]], dtype=torch.float32
    )
    scale_m = torch.tensor(
        [[scale_factor, 0.0, 0.0], [0.0, scale_factor, 0.0], [0.0, 0.0, 1.0]],
        dtype=torch.float32,
    )

    linear = scale_m @ rotation_m @ shear_m
    centre = torch.tensor([[width / 2.0, height / 2.0, 1.0]], dtype=torch.float32)
    shift = torch.tensor([[shift_x], [shift_y], [0.0]], dtype=torch.float32)
    # Rotate/scale about the image centre, then apply the sampled translation.
    translation = (torch.eye(3, dtype=torch.float32) - linear) @ centre.T + shift
    matrix = linear.clone()
    matrix[:2, 2] += translation[:2, 0]
    return matrix


def transform_boxes_xyxy(
    boxes: torch.Tensor,
    matrix: torch.Tensor,
    height: int,
    width: int,
    min_area_fraction: float = MIN_AFFINE_BOX_AREA_FRACTION,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Map absolute ``(N, 4)`` XYXY boxes through ``matrix`` and clip to the image.

    Each box is mapped by transforming all four corners and taking the axis-aligned
    envelope, which is the correct behaviour for a rotated box. Returns the clipped
    boxes and a keep-mask that is ``False`` for degenerate boxes or boxes that shrank
    below ``min_area_fraction`` of their original area.
    """
    if boxes.numel() == 0:
        return boxes, torch.zeros((0,), dtype=torch.bool)
    if boxes.shape[-1] != 4:
        raise ValueError(f"Expected (N, 4) XYXY boxes, got {tuple(boxes.shape)}")

    boxes = boxes.to(torch.float32)
    x1, y1, x2, y2 = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]

    corners = torch.stack(
        [
            torch.stack([x1, y1], dim=1),
            torch.stack([x2, y1], dim=1),
            torch.stack([x1, y2], dim=1),
            torch.stack([x2, y2], dim=1),
        ],
        dim=1,
    )
    homogeneous = torch.cat(
        [corners, torch.ones(corners.shape[0], 4, 1, dtype=torch.float32)], dim=2
    )
    mapped = homogeneous @ matrix.to(torch.float32).T
    mapped = mapped[:, :, :2] / mapped[:, :, 2].clamp(min=1e-6).unsqueeze(2)

    new_x1 = mapped[:, :, 0].min(dim=1).values
    new_y1 = mapped[:, :, 1].min(dim=1).values
    new_x2 = mapped[:, :, 0].max(dim=1).values
    new_y2 = mapped[:, :, 1].max(dim=1).values

    clipped = torch.stack(
        [
            new_x1.clamp(0.0, float(width)),
            new_y1.clamp(0.0, float(height)),
            new_x2.clamp(0.0, float(width)),
            new_y2.clamp(0.0, float(height)),
        ],
        dim=1,
    )

    original_area = (x2 - x1).clamp(min=0.0) * (y2 - y1).clamp(min=0.0)
    new_area = (clipped[:, 2] - clipped[:, 0]).clamp(min=0.0) * (
        clipped[:, 3] - clipped[:, 1]
    ).clamp(min=0.0)
    positive = (clipped[:, 2] > clipped[:, 0]) & (clipped[:, 3] > clipped[:, 1])
    keep = positive & (new_area >= original_area * min_area_fraction)
    return clipped, keep


def transform_yolo_cxcywh(
    boxes: torch.Tensor,
    matrix: torch.Tensor,
    height: int,
    width: int,
    min_area_fraction: float = MIN_AFFINE_BOX_AREA_FRACTION,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Map normalized ``(N, 5)`` YOLO targets through ``matrix`` and clip to the image.

    Targets are converted to absolute XYXY, transformed with the shared corner rule,
    clipped, filtered, and converted back to normalized ``cxcywh``. Class IDs pass
    through unchanged. Returns the updated targets and the keep-mask.
    """
    if boxes.numel() == 0:
        return boxes, torch.zeros((0,), dtype=torch.bool)
    if boxes.shape[-1] != 5:
        raise ValueError(f"Expected (N, 5) YOLO targets, got {tuple(boxes.shape)}")

    boxes = boxes.to(torch.float32)
    extent = torch.tensor([float(width), float(height)], dtype=torch.float32)
    centres = boxes[:, 1:3] * extent
    half = boxes[:, 3:5] * extent / 2.0

    absolute = torch.cat([centres - half, centres + half], dim=1)
    transformed, keep = transform_boxes_xyxy(
        absolute, matrix, height, width, min_area_fraction
    )

    kept = boxes[keep]
    transformed = transformed[keep]
    new_centres = (transformed[:, :2] + transformed[:, 2:]) / 2.0
    # ``transformed`` spans corner to corner, so restore the full extent.
    new_size = transformed[:, 2:] - transformed[:, :2]
    normalized = torch.cat(
        [kept[:, 0:1], new_centres / extent, new_size / extent], dim=1
    )
    return normalized, keep


def apply_affine(
    image: torch.Tensor,
    probability: float = DEFAULT_AFFINE_PROB,
    degrees: float = DEFAULT_AFFINE_DEGREES,
    translate: float = DEFAULT_AFFINE_TRANSLATE,
    scale: float = DEFAULT_AFFINE_SCALE,
    shear: float = DEFAULT_AFFINE_SHEAR,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Warp a normalized RGB tensor by a random affine and return the forward matrix.

    Returns the warped tensor and the 3x3 matrix mapping source pixels to destination
    pixels, which the caller must apply to targets. Regions rotated out of view are
    filled with mid-gray, matching Ultralytics padding.
    """
    if not 0.0 <= probability <= 1.0:
        raise ValueError("affine probability must be in [0, 1]")

    height, width = int(image.shape[-2]), int(image.shape[-1])
    if float(torch.rand((), device=image.device)) >= probability:
        return image, identity_matrix()

    matrix = sample_affine_matrix(height, width, degrees, translate, scale, shear)

    source = image.detach().to(dtype=torch.float32).permute(1, 2, 0).numpy() * 255.0
    source = np.clip(source, 0, 255).astype(np.uint8)

    # OpenCV's warpAffine takes a *forward* src->dst matrix by default and inverts it
    # internally; only WARP_INVERSE_MAP opts out of that. Passing a pre-inverted matrix
    # here would apply the inverse as the forward mapping, moving pixels opposite to
    # the boxes. Pass the forward matrix directly.
    warped = cv2.warpAffine(
        source,
        matrix.numpy()[:2],
        (width, height),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=(AFFINE_BORDER_VALUE,) * 3,
    )
    result = torch.from_numpy(warped).permute(2, 0, 1).to(dtype=image.dtype) / 255.0
    return result, matrix


def augment_image(image: torch.Tensor, mode: str) -> Tuple[torch.Tensor, torch.Tensor]:
    """Return an augmented normalized RGB tensor and the 3x3 forward target matrix.

    ``image`` must have shape ``(3, height, width)`` with finite float values in
    ``[0, 1]``. The returned matrix maps source pixel coordinates to destination pixel
    coordinates and is the identity for appearance-only policies. Callers must apply it
    to their targets with :func:`transform_boxes_xyxy` or :func:`transform_yolo_cxcywh`,
    so geometric policies can never silently desync from the labels.
    """
    mode = validate_online_augmentation(mode)
    if image.ndim != 3 or image.shape[0] != 3:
        raise ValueError(f"Expected a normalized RGB tensor with shape (3, height, width), got {tuple(image.shape)}")
    if not image.is_floating_point():
        raise ValueError("Online augmentation requires a floating-point image tensor")
    if not torch.isfinite(image).all() or image.min() < 0.0 or image.max() > 1.0:
        raise ValueError("Online augmentation requires finite image values in [0, 1]")

    width = int(image.shape[-1])
    if mode == "none":
        return image, identity_matrix()
    if mode == "hflip":
        flipped, applied = apply_horizontal_flip(image)
        return flipped, horizontal_flip_matrix(width) if applied else identity_matrix()
    if mode == "affine":
        return apply_affine(image)

    return apply_online_augmentation(image, mode), identity_matrix()


def apply_online_augmentation(image: torch.Tensor, mode: str) -> torch.Tensor:
    """Return an appearance-only augmented copy of a normalized RGB tensor.

    ``image`` must have shape ``(3, height, width)``, floating-point values in
    ``[0, 1]``, and no gradient requirement. The photometric and hsv policies apply
    color/intensity appearance changes only without modifying coordinates or targets.
    """
    mode = validate_online_augmentation(mode)
    if image.ndim != 3 or image.shape[0] != 3:
        raise ValueError(f"Expected a normalized RGB tensor with shape (3, height, width), got {tuple(image.shape)}")
    if not image.is_floating_point():
        raise ValueError("Online augmentation requires a floating-point image tensor")
    if not torch.isfinite(image).all() or image.min() < 0.0 or image.max() > 1.0:
        raise ValueError("Online augmentation requires finite image values in [0, 1]")
    if mode == "none":
        return image
    if mode in GEOMETRIC_ONLINE_AUGMENTATIONS:
        raise ValueError(
            f"Online augmentation {mode!r} changes geometry, so it cannot be applied through this "
            "appearance-only helper. Use augment_image() and transform the targets with the returned flag."
        )
    if mode == "hsv":
        return apply_hsv_augmentation(image)

    augmented = image.clone()
    augmented = (augmented * _sample_symmetric_factor(PHOTOMETRIC_BRIGHTNESS_DELTA, augmented)).clamp(0.0, 1.0)

    channel_mean = augmented.mean(dim=(1, 2), keepdim=True)
    augmented = (
        (augmented - channel_mean) * _sample_symmetric_factor(PHOTOMETRIC_CONTRAST_DELTA, augmented)
        + channel_mean
    ).clamp(0.0, 1.0)

    gamma = _sample_symmetric_factor(PHOTOMETRIC_GAMMA_DELTA, augmented)
    augmented = augmented.pow(gamma)
    augmented = augmented + torch.randn_like(augmented) * PHOTOMETRIC_NOISE_STD
    return augmented.clamp(0.0, 1.0)

