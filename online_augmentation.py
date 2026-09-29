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

from typing import Tuple

import cv2
import numpy as np
import torch


DEFAULT_ONLINE_AUGMENTATION = "none"
ONLINE_AUGMENTATION_CHOICES = ("none", "photometric", "hsv", "hflip")

# Official Ultralytics horizontal flip probability (fliplr in the default dataset config)
DEFAULT_HFLIP_PROB = 0.5

GEOMETRIC_ONLINE_AUGMENTATIONS = ("hflip",)

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


def augment_image(image: torch.Tensor, mode: str) -> Tuple[torch.Tensor, bool]:
    """Return an augmented copy of a normalized RGB tensor and a geometric-change flag.

    ``image`` must have shape ``(3, height, width)`` with finite float values in
    ``[0, 1]``. The flag is ``True`` only when the returned pixels are geometrically
    mirrored relative to the input, so the caller knows whether targets must be
    transformed. Appearance-only policies return ``False``.
    """
    mode = validate_online_augmentation(mode)
    if image.ndim != 3 or image.shape[0] != 3:
        raise ValueError(f"Expected a normalized RGB tensor with shape (3, height, width), got {tuple(image.shape)}")
    if not image.is_floating_point():
        raise ValueError("Online augmentation requires a floating-point image tensor")
    if not torch.isfinite(image).all() or image.min() < 0.0 or image.max() > 1.0:
        raise ValueError("Online augmentation requires finite image values in [0, 1]")
    if mode == "none":
        return image, False
    if mode == "hflip":
        return apply_horizontal_flip(image)

    return apply_online_augmentation(image, mode), False


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

