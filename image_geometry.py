"""Shared image-resize and bounding-box geometry for local detector pipelines.

The historical ``stretch`` mode exactly preserves the project's direct square
resize behavior. The post-submission ``letterbox`` mode preserves aspect ratio
on a fixed canvas and records the scale/padding required to transform labels and
predictions consistently.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import cv2
import numpy as np
import torch


ResizeMode = Literal["stretch", "letterbox"]
RESIZE_MODE_CHOICES: tuple[ResizeMode, ...] = ("stretch", "letterbox")
DEFAULT_RESIZE_MODE: ResizeMode = "stretch"
DEFAULT_LETTERBOX_PAD_VALUE = 114


def validate_resize_mode(resize_mode: str) -> ResizeMode:
    """Validate a serializable resize mode while retaining legacy stretch behavior."""
    normalized = str(resize_mode).strip().lower()
    if normalized not in RESIZE_MODE_CHOICES:
        raise ValueError(
            f"resize_mode must be one of {list(RESIZE_MODE_CHOICES)}, got {resize_mode!r}"
        )
    return normalized  # type: ignore[return-value]


def resolve_resize_mode(
    requested_resize_mode: str | None,
    saved_args: object,
) -> ResizeMode:
    """Prefer an explicit mode, then checkpoint metadata, then legacy stretch."""
    if requested_resize_mode is not None:
        return validate_resize_mode(requested_resize_mode)
    if not isinstance(saved_args, dict):
        return DEFAULT_RESIZE_MODE
    return validate_resize_mode(str(saved_args.get("resize_mode", DEFAULT_RESIZE_MODE)))


def resize_interpolation(
    source_height: int,
    source_width: int,
    resized_height: int,
    resized_width: int,
) -> int:
    """Use the project's established area/cubic/linear resize policy."""
    if resized_width < source_width or resized_height < source_height:
        return cv2.INTER_AREA
    if resized_width > source_width or resized_height > source_height:
        return cv2.INTER_CUBIC
    return cv2.INTER_LINEAR


@dataclass(frozen=True)
class ResizeTransform:
    """Coordinate mapping from a source image to one fixed model canvas."""

    source_height: int
    source_width: int
    target_height: int
    target_width: int
    resized_height: int
    resized_width: int
    pad_top: int
    pad_bottom: int
    pad_left: int
    pad_right: int
    resize_mode: ResizeMode

    @classmethod
    def from_shapes(
        cls,
        source_height: int,
        source_width: int,
        target_height: int,
        target_width: int,
        resize_mode: str = DEFAULT_RESIZE_MODE,
    ) -> "ResizeTransform":
        """Create the deterministic scale and padding contract for one image."""
        mode = validate_resize_mode(resize_mode)
        dimensions = {
            "source_height": source_height,
            "source_width": source_width,
            "target_height": target_height,
            "target_width": target_width,
        }
        if any(value <= 0 for value in dimensions.values()):
            raise ValueError(f"All image dimensions must be positive, got {dimensions}")

        if mode == "stretch":
            resized_height, resized_width = target_height, target_width
        else:
            scale = min(target_width / source_width, target_height / source_height)
            resized_width = min(target_width, max(1, round(source_width * scale)))
            resized_height = min(target_height, max(1, round(source_height * scale)))

        vertical_padding = target_height - resized_height
        horizontal_padding = target_width - resized_width
        pad_top = vertical_padding // 2
        pad_bottom = vertical_padding - pad_top
        pad_left = horizontal_padding // 2
        pad_right = horizontal_padding - pad_left
        return cls(
            source_height=source_height,
            source_width=source_width,
            target_height=target_height,
            target_width=target_width,
            resized_height=resized_height,
            resized_width=resized_width,
            pad_top=pad_top,
            pad_bottom=pad_bottom,
            pad_left=pad_left,
            pad_right=pad_right,
            resize_mode=mode,
        )

    @property
    def scale_x(self) -> float:
        return self.resized_width / self.source_width

    @property
    def scale_y(self) -> float:
        return self.resized_height / self.source_height

    def source_xyxy_to_model_xyxy(self, boxes: torch.Tensor) -> torch.Tensor:
        """Map source-pixel XYXY boxes onto the padded model canvas."""
        _validate_boxes(boxes)
        transformed = boxes.clone()
        transformed[..., 0] = transformed[..., 0] * self.scale_x + self.pad_left
        transformed[..., 2] = transformed[..., 2] * self.scale_x + self.pad_left
        transformed[..., 1] = transformed[..., 1] * self.scale_y + self.pad_top
        transformed[..., 3] = transformed[..., 3] * self.scale_y + self.pad_top
        return clip_xyxy(transformed, self.target_width, self.target_height)

    def model_xyxy_to_source_xyxy(self, boxes: torch.Tensor) -> torch.Tensor:
        """Restore model-canvas XYXY boxes to source-pixel coordinates."""
        _validate_boxes(boxes)
        restored = boxes.clone()
        restored[..., 0] = (restored[..., 0] - self.pad_left) / self.scale_x
        restored[..., 2] = (restored[..., 2] - self.pad_left) / self.scale_x
        restored[..., 1] = (restored[..., 1] - self.pad_top) / self.scale_y
        restored[..., 3] = (restored[..., 3] - self.pad_top) / self.scale_y
        return clip_xyxy(restored, self.source_width, self.source_height)


def _validate_boxes(boxes: torch.Tensor) -> None:
    if boxes.ndim < 1 or boxes.shape[-1] != 4:
        raise ValueError(f"Expected XYXY boxes with final dimension 4, got {tuple(boxes.shape)}")
    if not torch.is_floating_point(boxes):
        raise ValueError("Bounding boxes must use a floating-point tensor")


def clip_xyxy(boxes: torch.Tensor, width: int, height: int) -> torch.Tensor:
    """Clip XYXY coordinates to image bounds without altering box ordering."""
    if width <= 0 or height <= 0:
        raise ValueError("Image width and height must be positive")
    clipped = boxes.clone()
    clipped[..., 0] = clipped[..., 0].clamp(0.0, float(width))
    clipped[..., 2] = clipped[..., 2].clamp(0.0, float(width))
    clipped[..., 1] = clipped[..., 1].clamp(0.0, float(height))
    clipped[..., 3] = clipped[..., 3].clamp(0.0, float(height))
    return clipped


def source_yolo_to_model_xyxy(labels: torch.Tensor, transform: ResizeTransform) -> torch.Tensor:
    """Convert source-normalized YOLO labels to model-canvas absolute XYXY boxes."""
    if labels.ndim != 2 or labels.shape[1] != 5:
        raise ValueError(f"Expected YOLO labels with shape (N, 5), got {tuple(labels.shape)}")
    if not torch.is_floating_point(labels):
        raise ValueError("YOLO labels must use a floating-point tensor")
    if labels.numel() == 0:
        return labels.new_zeros((0, 4))

    x_center = labels[:, 1] * transform.source_width
    y_center = labels[:, 2] * transform.source_height
    box_width = labels[:, 3] * transform.source_width
    box_height = labels[:, 4] * transform.source_height
    source_xyxy = torch.stack(
        (
            x_center - box_width / 2.0,
            y_center - box_height / 2.0,
            x_center + box_width / 2.0,
            y_center + box_height / 2.0,
        ),
        dim=1,
    )
    return transform.source_xyxy_to_model_xyxy(source_xyxy)


def model_xyxy_to_yolo_xywhn(boxes: torch.Tensor, width: int, height: int) -> torch.Tensor:
    """Convert model-canvas XYXY boxes to normalized YOLO XYWH coordinates."""
    _validate_boxes(boxes)
    if width <= 0 or height <= 0:
        raise ValueError("Image width and height must be positive")
    clipped = clip_xyxy(boxes, width, height)
    x_center = (clipped[..., 0] + clipped[..., 2]) / 2.0 / width
    y_center = (clipped[..., 1] + clipped[..., 3]) / 2.0 / height
    box_width = (clipped[..., 2] - clipped[..., 0]) / width
    box_height = (clipped[..., 3] - clipped[..., 1]) / height
    return torch.stack((x_center, y_center, box_width, box_height), dim=-1)


def source_yolo_to_model_yolo(labels: torch.Tensor, transform: ResizeTransform) -> torch.Tensor:
    """Map source-normalized YOLO labels to the model canvas while retaining IDs."""
    model_xyxy = source_yolo_to_model_xyxy(labels, transform)
    if labels.numel() == 0:
        return labels.new_zeros((0, 5))
    model_xywhn = model_xyxy_to_yolo_xywhn(
        model_xyxy,
        transform.target_width,
        transform.target_height,
    )
    return torch.cat((labels[:, :1], model_xywhn), dim=1)


def model_yolo_to_source_yolo(labels: torch.Tensor, transform: ResizeTransform) -> torch.Tensor:
    """Restore model-normalized YOLO labels to source-normalized YOLO labels."""
    if labels.ndim != 2 or labels.shape[1] != 5:
        raise ValueError(f"Expected YOLO labels with shape (N, 5), got {tuple(labels.shape)}")
    if not torch.is_floating_point(labels):
        raise ValueError("YOLO labels must use a floating-point tensor")
    if labels.numel() == 0:
        return labels.new_zeros((0, 5))
    x_center = labels[:, 1] * transform.target_width
    y_center = labels[:, 2] * transform.target_height
    box_width = labels[:, 3] * transform.target_width
    box_height = labels[:, 4] * transform.target_height
    model_xyxy = torch.stack(
        (
            x_center - box_width / 2.0,
            y_center - box_height / 2.0,
            x_center + box_width / 2.0,
            y_center + box_height / 2.0,
        ),
        dim=1,
    )
    source_xyxy = transform.model_xyxy_to_source_xyxy(model_xyxy)
    source_xywhn = model_xyxy_to_yolo_xywhn(
        source_xyxy,
        transform.source_width,
        transform.source_height,
    )
    return torch.cat((labels[:, :1], source_xywhn), dim=1)


def restore_detections_to_source(detections: torch.Tensor, transform: ResizeTransform) -> torch.Tensor:
    """Restore detection rows ``(xyxy, score, class_id, ...)`` to source coordinates."""
    if detections.ndim != 2 or detections.shape[1] < 4:
        raise ValueError(f"Expected detections with shape (N, >=4), got {tuple(detections.shape)}")
    if not torch.is_floating_point(detections):
        raise ValueError("Detections must use a floating-point tensor")
    restored = detections.clone()
    restored[:, :4] = transform.model_xyxy_to_source_xyxy(restored[:, :4])
    return restored


def resize_image(
    image: np.ndarray,
    target_height: int,
    target_width: int,
    resize_mode: str = DEFAULT_RESIZE_MODE,
    pad_value: int = DEFAULT_LETTERBOX_PAD_VALUE,
) -> tuple[np.ndarray, ResizeTransform]:
    """Resize an RGB or BGR image and return its reversible geometry metadata."""
    if image.ndim not in {2, 3}:
        raise ValueError(f"Expected a 2D or 3D image array, got shape {tuple(image.shape)}")
    if not 0 <= pad_value <= 255:
        raise ValueError("pad_value must be in 0..255")
    source_height, source_width = image.shape[:2]
    transform = ResizeTransform.from_shapes(
        source_height,
        source_width,
        target_height,
        target_width,
        resize_mode,
    )
    interpolation = resize_interpolation(
        source_height,
        source_width,
        transform.resized_height,
        transform.resized_width,
    )
    resized = cv2.resize(
        image,
        (transform.resized_width, transform.resized_height),
        interpolation=interpolation,
    )
    if transform.resize_mode == "stretch":
        return resized, transform

    canvas_shape = (transform.target_height, transform.target_width, *image.shape[2:])
    canvas = np.full(canvas_shape, pad_value, dtype=image.dtype)
    top = transform.pad_top
    left = transform.pad_left
    canvas[top : top + transform.resized_height, left : left + transform.resized_width] = resized
    return canvas, transform