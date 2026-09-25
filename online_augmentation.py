"""Conservative in-memory photometric augmentation for local detector training.

This module deliberately changes pixel appearance only. It never changes image
geometry, boxes, labels, source files, or class sampling frequencies.
"""
from __future__ import annotations

import cv2
import numpy as np
import torch


DEFAULT_ONLINE_AUGMENTATION = "none"
ONLINE_AUGMENTATION_CHOICES = ("none", "photometric", "hsv")

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

