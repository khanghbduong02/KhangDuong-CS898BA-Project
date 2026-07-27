"""Conservative in-memory photometric augmentation for local detector training.

This module deliberately changes pixel appearance only. It never changes image
geometry, boxes, labels, source files, or class sampling frequencies.
"""
from __future__ import annotations

import torch


DEFAULT_ONLINE_AUGMENTATION = "none"
ONLINE_AUGMENTATION_CHOICES = ("none", "photometric")

PHOTOMETRIC_BRIGHTNESS_DELTA = 0.10
PHOTOMETRIC_CONTRAST_DELTA = 0.10
PHOTOMETRIC_GAMMA_DELTA = 0.10
PHOTOMETRIC_NOISE_STD = 0.01


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


def apply_online_augmentation(image: torch.Tensor, mode: str) -> torch.Tensor:
    """Return an appearance-only augmented copy of a normalized RGB tensor.

    ``image`` must have shape ``(3, height, width)``, floating-point values in
    ``[0, 1]``, and no gradient requirement. The photometric policy applies
    small brightness, per-channel contrast, gamma, and Gaussian-noise changes.
    It has no geometric effect, so detection targets remain valid unchanged.
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
