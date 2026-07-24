"""Shared horizontal-flip test-time augmentation helpers for local detectors."""
from __future__ import annotations

from typing import Mapping, Sequence

import torch
from torchvision.ops import nms


def horizontal_flip_images(images: torch.Tensor) -> torch.Tensor:
    """Return a horizontally flipped ``(batch, channels, height, width)`` image tensor."""
    if images.ndim != 4:
        raise ValueError(f"Expected images with shape (batch, channels, height, width), got {tuple(images.shape)}")
    return torch.flip(images, dims=(-1,))


def unflip_xyxy_boxes(boxes: torch.Tensor, image_width: int) -> torch.Tensor:
    """Map horizontally flipped pixel-space ``xyxy`` boxes back to original coordinates."""
    if boxes.ndim != 2 or boxes.shape[1] != 4:
        raise ValueError(f"Expected boxes with shape (detections, 4), got {tuple(boxes.shape)}")
    if image_width <= 0:
        raise ValueError("image_width must be positive")

    restored = boxes.clone()
    restored[:, 0] = image_width - boxes[:, 2]
    restored[:, 2] = image_width - boxes[:, 0]
    return restored


def unflip_decoded_xyxy(decoded: torch.Tensor, image_width: int) -> torch.Tensor:
    """Map decoded YOLO predictions from a horizontally flipped image back to original coordinates."""
    if decoded.ndim != 3 or decoded.shape[1] < 4:
        raise ValueError(f"Expected decoded predictions with shape (batch, 4 + classes, anchors), got {tuple(decoded.shape)}")
    if image_width <= 0:
        raise ValueError("image_width must be positive")

    restored = decoded.clone()
    restored[:, 0, :] = image_width - decoded[:, 2, :]
    restored[:, 2, :] = image_width - decoded[:, 0, :]
    return restored


def merge_hflip_predictions(
    original_predictions: Sequence[Mapping[str, torch.Tensor]],
    flipped_predictions: Sequence[Mapping[str, torch.Tensor]],
    image_widths: Sequence[int],
    nms_iou: float,
    max_detections: int,
) -> list[dict[str, torch.Tensor]]:
    """Merge original and horizontal-flip detections using the existing per-class NMS policy."""
    if len(original_predictions) != len(flipped_predictions) or len(original_predictions) != len(image_widths):
        raise ValueError("Original predictions, flipped predictions, and image widths must have identical lengths")
    if not 0.0 < nms_iou <= 1.0:
        raise ValueError("nms_iou must be in (0, 1]")
    if max_detections <= 0:
        raise ValueError("max_detections must be positive")

    merged_predictions: list[dict[str, torch.Tensor]] = []
    for original, flipped, image_width in zip(original_predictions, flipped_predictions, image_widths):
        for prediction in (original, flipped):
            if not {"boxes", "scores", "labels"}.issubset(prediction):
                raise ValueError("Each prediction must contain boxes, scores, and labels")

        original_boxes = original["boxes"]
        flipped_boxes = unflip_xyxy_boxes(flipped["boxes"], image_width)
        boxes = torch.cat((original_boxes, flipped_boxes), dim=0)
        scores = torch.cat((original["scores"], flipped["scores"]), dim=0)
        labels = torch.cat((original["labels"], flipped["labels"]), dim=0)
        if boxes.shape[0] != scores.numel() or boxes.shape[0] != labels.numel():
            raise ValueError("Prediction boxes, scores, and labels must have matching lengths")

        if boxes.numel() == 0:
            merged_predictions.append({"boxes": boxes, "scores": scores, "labels": labels})
            continue

        keep_by_class: list[torch.Tensor] = []
        for label in labels.unique(sorted=True):
            class_indices = torch.nonzero(labels == label, as_tuple=False).flatten()
            class_keep = nms(boxes[class_indices].float(), scores[class_indices].float(), nms_iou)
            keep_by_class.append(class_indices[class_keep])
        keep = torch.cat(keep_by_class)
        keep = keep[scores[keep].argsort(descending=True)]
        if keep.numel() > max_detections:
            keep = keep[:max_detections]
        merged_predictions.append(
            {"boxes": boxes[keep], "scores": scores[keep], "labels": labels[keep]}
        )

    return merged_predictions
