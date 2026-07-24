from __future__ import annotations

import argparse
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any, List, Sequence, Tuple

import cv2
import torch
import torch.nn.functional as F
from torch import nn
from torch.optim import AdamW
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
from ultralytics.utils.loss import BboxLoss
from ultralytics.utils.ops import xywh2xyxy
from ultralytics.utils.tal import TaskAlignedAssigner, dist2bbox, make_anchors

from detection_metrics import compute_detection_metrics, xywhn_to_xyxy
from models.yolo26_torch import build_yolo26, class_aware_nms
from training_control import (
    EpochLRScheduler,
    PlateauEarlyStopping,
    add_checkpoint_selection_argument,
    add_epoch_lr_schedule_arguments,
    add_plateau_early_stopping_arguments,
    checkpoint_selection_improved,
    epoch_lr_schedule_config_from_args,
    initial_checkpoint_selection_value,
    plateau_early_stopping_config_from_args,
    validate_checkpoint_selection,
    validate_training_control_compatibility,
)
from yolo_dataset_config import read_yolo_dataset_config


VALID_IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}
STRIDES = (8, 16, 32)


def focal_bce_with_logits(
    logits: torch.Tensor,
    targets: torch.Tensor,
    positive_weights: torch.Tensor,
    focal_gamma: float,
) -> torch.Tensor:
    """Return elementwise BCE, optionally down-weighting easy classified anchors."""
    if focal_gamma < 0.0:
        raise ValueError("Focal gamma must be greater than or equal to zero")

    bce = F.binary_cross_entropy_with_logits(
        logits,
        targets,
        reduction="none",
        pos_weight=positive_weights.to(dtype=logits.dtype),
    )
    if focal_gamma == 0.0:
        return bce

    probabilities = logits.sigmoid()
    p_t = targets * probabilities + (1.0 - targets) * (1.0 - probabilities)
    focal_weight = (1.0 - p_t).clamp(min=0.0).pow(focal_gamma)
    return bce * focal_weight


class YoloDetectionDataset(Dataset):
    def __init__(self, split_root: Path, imgsz: int, fraction: float = 1.0) -> None:
        self.images_dir = split_root / "images"
        self.labels_dir = split_root / "labels"
        self.imgsz = imgsz

        image_paths = [
            path
            for path in sorted(self.images_dir.iterdir())
            if path.is_file() and path.suffix.lower() in VALID_IMAGE_EXTENSIONS
        ]
        if 0.0 < fraction < 1.0:
            keep = max(1, int(len(image_paths) * fraction))
            image_paths = image_paths[:keep]

        self.image_paths = image_paths

    def __len__(self) -> int:
        return len(self.image_paths)

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, torch.Tensor]:
        image_path = self.image_paths[index]
        image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
        if image is None:
            raise FileNotFoundError(f"Unable to read image: {image_path}")

        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        original_h, original_w = image.shape[:2]
        target_w = self.imgsz
        target_h = self.imgsz
        if target_w < original_w or target_h < original_h:
            interpolation = cv2.INTER_AREA
        elif target_w > original_w or target_h > original_h:
            interpolation = cv2.INTER_CUBIC
        else:
            interpolation = cv2.INTER_LINEAR

        image = cv2.resize(image, (target_w, target_h), interpolation=interpolation)
        image_tensor = torch.from_numpy(image).permute(2, 0, 1).float() / 255.0

        label_path = self.labels_dir / f"{image_path.stem}.txt"
        labels = []
        if label_path.exists():
            for line_number, line in enumerate(label_path.read_text(encoding="utf-8").splitlines(), start=1):
                parts = line.strip().split()
                if not parts:
                    continue
                if len(parts) != 5:
                    raise ValueError(
                        f"{label_path}:{line_number}: expected five-field YOLO detection label "
                        "(class x_center y_center width height). Run preprocess_dataset.py to normalize polygon labels."
                    )

                try:
                    class_id, x_center, y_center, width, height = (float(value) for value in parts)
                except ValueError as exc:
                    raise ValueError(f"{label_path}:{line_number}: label values must be numeric") from exc

                if class_id != int(class_id) or not (0.0 <= x_center <= 1.0 and 0.0 <= y_center <= 1.0):
                    raise ValueError(f"{label_path}:{line_number}: invalid class ID or normalized box center")
                if not (0.0 < width <= 1.0 and 0.0 < height <= 1.0):
                    raise ValueError(f"{label_path}:{line_number}: normalized box width and height must be in (0, 1]")

                labels.append([class_id, x_center, y_center, width, height])

        labels_tensor = torch.tensor(labels, dtype=torch.float32) if labels else torch.zeros((0, 5), dtype=torch.float32)
        return image_tensor, labels_tensor


def collate_fn(batch: Sequence[Tuple[torch.Tensor, torch.Tensor]]) -> Tuple[torch.Tensor, List[torch.Tensor]]:
    images = torch.stack([item[0] for item in batch], dim=0)
    targets = [item[1] for item in batch]
    return images, targets


def _read_label_class_ids(label_path: Path, num_classes: int) -> set[int]:
    return {
        class_id
        for class_id, count in enumerate(_read_label_class_counts(label_path, num_classes))
        if count > 0
    }


def _read_label_class_counts(label_path: Path, num_classes: int) -> list[int]:
    if not label_path.exists():
        return [0 for _ in range(num_classes)]

    counts = [0 for _ in range(num_classes)]
    for line_number, line in enumerate(label_path.read_text(encoding="utf-8").splitlines(), start=1):
        parts = line.strip().split()
        if not parts:
            continue
        if len(parts) != 5:
            raise ValueError(
                f"{label_path}:{line_number}: expected five-field YOLO detection label while building sampler"
            )

        try:
            class_value = float(parts[0])
        except ValueError as exc:
            raise ValueError(f"{label_path}:{line_number}: class ID must be numeric") from exc

        class_id = int(class_value)
        if class_value != class_id or not 0 <= class_id < num_classes:
            raise ValueError(f"{label_path}:{line_number}: class ID {parts[0]!r} is outside 0..{num_classes - 1}")
        counts[class_id] += 1

    return counts


def build_positive_class_weights(
    dataset: YoloDetectionDataset,
    num_classes: int,
    power: float = 0.0,
) -> tuple[torch.Tensor, list[int]]:
    """Build normalized inverse-frequency weights for positive classification terms only."""
    if not 0.0 <= power <= 1.0:
        raise ValueError("Positive class-weight power must be between 0.0 and 1.0")

    class_box_counts = [0 for _ in range(num_classes)]
    for image_path in dataset.image_paths:
        label_counts = _read_label_class_counts(dataset.labels_dir / f"{image_path.stem}.txt", num_classes)
        for class_id, count in enumerate(label_counts):
            class_box_counts[class_id] += count

    # Neutral weighting does not require every class to occur in an explicitly
    # small smoke subset. Nonzero powers still require all classes because an
    # inverse-frequency weight would be undefined for a missing class.
    if any(count == 0 for count in class_box_counts) and power == 0.0:
        return torch.ones(num_classes, dtype=torch.float32), class_box_counts
    if any(count == 0 for count in class_box_counts):
        raise ValueError(f"Cannot build positive class weights because box counts are {class_box_counts}")

    max_count = max(class_box_counts)
    raw_weights = torch.tensor(
        [(max_count / count) ** power for count in class_box_counts],
        dtype=torch.float32,
    )
    counts_tensor = torch.tensor(class_box_counts, dtype=torch.float32)
    normalized_weights = raw_weights / (counts_tensor * raw_weights).sum() * counts_tensor.sum()
    return normalized_weights, class_box_counts


def build_class_balanced_sampler(
    dataset: YoloDetectionDataset,
    num_classes: int,
    power: float = 1.0,
    generator: torch.Generator | None = None,
) -> tuple[WeightedRandomSampler, list[int], int]:
    """Oversample images containing rare classes without generating new files."""
    if not 0.0 <= power <= 1.0:
        raise ValueError("Class-balanced sampling power must be between 0.0 and 1.0")

    sample_class_ids: list[set[int]] = []
    class_image_counts = [0 for _ in range(num_classes)]
    background_images = 0

    for image_path in dataset.image_paths:
        class_ids = _read_label_class_ids(dataset.labels_dir / f"{image_path.stem}.txt", num_classes)
        sample_class_ids.append(class_ids)
        if not class_ids:
            background_images += 1
        for class_id in class_ids:
            class_image_counts[class_id] += 1

    if any(count == 0 for count in class_image_counts):
        raise ValueError(f"Cannot build class-balanced sampler because class image counts are {class_image_counts}")

    majority_count = max(class_image_counts)
    class_weights = [(majority_count / count) ** power for count in class_image_counts]
    weights = [max((class_weights[class_id] for class_id in class_ids), default=1.0) for class_ids in sample_class_ids]
    sampler = WeightedRandomSampler(
        torch.as_tensor(weights, dtype=torch.double),
        num_samples=len(dataset),
        replacement=True,
        generator=generator,
    )
    return sampler, class_image_counts, background_images


def seed_everything(seed: int) -> None:
    """Set the random sources used by model initialization and data loading."""
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True


def build_dense_targets(
    targets: Sequence[torch.Tensor],
    nc: int,
    imgsz: int,
    device: torch.device,
    assign_radius: int,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    grid_sizes = [imgsz // stride for stride in STRIDES]
    offsets = []
    running = 0
    for grid_size in grid_sizes:
        offsets.append(running)
        running += grid_size * grid_size

    batch_size = len(targets)
    total_cells = running
    cls_target = torch.zeros((batch_size, total_cells, nc), device=device)
    box_target = torch.zeros((batch_size, total_cells, 4), device=device)
    pos_mask = torch.zeros((batch_size, total_cells), dtype=torch.bool, device=device)

    for batch_index, sample_targets in enumerate(targets):
        if sample_targets.numel() == 0:
            continue

        for target in sample_targets:
            cls_id = int(target[0].item())
            x_center, y_center, width, height = target[1:].tolist()
            max_side = max(width, height)

            if max_side < 0.10:
                scale_index = 0
            elif max_side < 0.25:
                scale_index = 1
            else:
                scale_index = 2

            grid_size = grid_sizes[scale_index]
            cell_x = min(grid_size - 1, max(0, int(x_center * grid_size)))
            cell_y = min(grid_size - 1, max(0, int(y_center * grid_size)))

            for dy in range(-assign_radius, assign_radius + 1):
                for dx in range(-assign_radius, assign_radius + 1):
                    neighbor_x = cell_x + dx
                    neighbor_y = cell_y + dy
                    if not (0 <= neighbor_x < grid_size and 0 <= neighbor_y < grid_size):
                        continue

                    flat_index = offsets[scale_index] + neighbor_y * grid_size + neighbor_x

                    if pos_mask[batch_index, flat_index] and cls_target[batch_index, flat_index, cls_id] == 0:
                        continue

                    pos_mask[batch_index, flat_index] = True
                    cls_target[batch_index, flat_index, cls_id] = 1.0
                    box_target[batch_index, flat_index] = torch.tensor([x_center, y_center, width, height], device=device)

    return cls_target, box_target, pos_mask


@dataclass
class LossBreakdown:
    total: torch.Tensor
    cls_loss: torch.Tensor
    box_loss: torch.Tensor
    reg_loss: torch.Tensor


def build_batch_targets(targets: Sequence[torch.Tensor], device: torch.device) -> dict[str, torch.Tensor]:
    batch_idx_list: list[torch.Tensor] = []
    cls_list: list[torch.Tensor] = []
    bboxes_list: list[torch.Tensor] = []

    for batch_index, target in enumerate(targets):
        if target.numel() == 0:
            continue
        target = target.to(device)
        count = target.shape[0]
        batch_idx_list.append(torch.full((count,), batch_index, dtype=torch.long, device=device))
        cls_list.append(target[:, 0:1])
        bboxes_list.append(target[:, 1:5])

    if not batch_idx_list:
        return {
            "batch_idx": torch.zeros((0,), dtype=torch.long, device=device),
            "cls": torch.zeros((0, 1), dtype=torch.float32, device=device),
            "bboxes": torch.zeros((0, 4), dtype=torch.float32, device=device),
        }

    return {
        "batch_idx": torch.cat(batch_idx_list, dim=0),
        "cls": torch.cat(cls_list, dim=0),
        "bboxes": torch.cat(bboxes_list, dim=0),
    }


class BranchDetectionLoss:
    def __init__(
        self,
        nc: int,
        strides: Tuple[int, ...],
        device: torch.device,
        box_gain: float,
        cls_gain: float,
        reg_gain: float,
        tal_topk: int,
        reg_max: int = 1,
        class_positive_weights: torch.Tensor | None = None,
        focal_gamma: float = 0.0,
    ) -> None:
        self.device = device
        self.nc = nc
        if reg_max <= 0:
            raise ValueError("reg_max must be positive")
        self.reg_max = reg_max
        self.box_gain = box_gain
        self.cls_gain = cls_gain
        self.reg_gain = reg_gain
        if focal_gamma < 0.0:
            raise ValueError("Focal gamma must be greater than or equal to zero")
        self.focal_gamma = focal_gamma
        if class_positive_weights is None:
            class_positive_weights = torch.ones(nc, dtype=torch.float32)
        if class_positive_weights.numel() != nc:
            raise ValueError(f"Expected {nc} positive class weights, got {class_positive_weights.numel()}")
        if (class_positive_weights <= 0).any():
            raise ValueError("Positive class weights must all be greater than zero")
        self.class_positive_weights = class_positive_weights.detach().to(device=device, dtype=torch.float32).reshape(nc)
        self.stride = torch.tensor(strides, dtype=torch.float32, device=device)
        self.dfl_project = torch.arange(reg_max, dtype=torch.float32, device=device)
        self.assigner = TaskAlignedAssigner(
            topk=tal_topk,
            num_classes=nc,
            alpha=0.5,
            beta=6.0,
            stride=list(strides),
        )
        self.bbox_loss = BboxLoss(self.reg_max).to(device)

    def decode_distances(self, pred_distri: torch.Tensor) -> torch.Tensor:
        """Decode direct distances or DFL logits into `(batch, anchors, 4)` distances."""
        if self.reg_max == 1:
            return pred_distri
        batch_size, anchors, channels = pred_distri.shape
        expected_channels = 4 * self.reg_max
        if channels != expected_channels:
            raise ValueError(
                f"Expected {expected_channels} distributional box channels, got {channels}"
            )
        probabilities = pred_distri.view(batch_size, anchors, 4, self.reg_max).softmax(dim=-1)
        project = self.dfl_project.to(dtype=pred_distri.dtype)
        return probabilities.matmul(project)

    def preprocess(self, targets: torch.Tensor, batch_size: int, scale_tensor: torch.Tensor) -> torch.Tensor:
        nl, ne = targets.shape
        if nl == 0:
            return torch.zeros(batch_size, 0, ne - 1, device=self.device)

        batch_idx = targets[:, 0].long()
        _, counts = batch_idx.unique(return_counts=True)
        counts = counts.to(dtype=torch.int32)
        out = torch.zeros(batch_size, counts.max(), ne - 1, device=self.device)
        offsets = torch.zeros(batch_size + 1, dtype=torch.long, device=self.device)
        offsets.scatter_add_(0, batch_idx + 1, torch.ones_like(batch_idx))
        offsets = offsets.cumsum(0)
        within_idx = torch.arange(nl, device=self.device) - offsets[batch_idx]
        out[batch_idx, within_idx] = targets[:, 1:]
        out[..., 1:5] = xywh2xyxy(out[..., 1:5].mul_(scale_tensor))
        return out

    def __call__(self, preds: dict[str, torch.Tensor], batch: dict[str, torch.Tensor]) -> LossBreakdown:
        pred_distri = preds["boxes"].permute(0, 2, 1).contiguous()
        pred_scores = preds["scores"].permute(0, 2, 1).contiguous()
        anchor_points, stride_tensor = make_anchors(preds["feats"], self.stride, 0.5)

        dtype = pred_scores.dtype
        batch_size = pred_scores.shape[0]
        imgsz = torch.tensor(preds["feats"][0].shape[2:], device=self.device, dtype=dtype) * self.stride[0]

        targets = torch.cat((batch["batch_idx"].view(-1, 1), batch["cls"].view(-1, 1), batch["bboxes"]), dim=1)
        targets = self.preprocess(targets.to(self.device), batch_size, scale_tensor=imgsz[[1, 0, 1, 0]])
        gt_labels, gt_bboxes = targets.split((1, 4), dim=2)
        mask_gt = gt_bboxes.sum(2, keepdim=True).gt_(0.0)

        pred_bboxes = dist2bbox(self.decode_distances(pred_distri), anchor_points, xywh=False)
        _, target_bboxes, target_scores, fg_mask, _ = self.assigner(
            pred_scores.detach().sigmoid(),
            (pred_bboxes.detach() * stride_tensor).type(gt_bboxes.dtype),
            anchor_points * stride_tensor,
            gt_labels,
            gt_bboxes,
            mask_gt,
        )

        target_scores_sum = target_scores.sum().clamp(min=1.0)
        cls_loss = focal_bce_with_logits(
            pred_scores,
            target_scores.to(dtype),
            self.class_positive_weights,
            self.focal_gamma,
        ).sum() / target_scores_sum

        box_loss = torch.zeros((), device=self.device)
        reg_loss = torch.zeros((), device=self.device)
        if fg_mask.sum():
            box_loss, reg_loss = self.bbox_loss(
                pred_distri,
                pred_bboxes,
                anchor_points,
                target_bboxes / stride_tensor,
                target_scores,
                target_scores_sum,
                fg_mask,
                imgsz,
                stride_tensor,
            )

        total = self.box_gain * box_loss + self.cls_gain * cls_loss + self.reg_gain * reg_loss
        return LossBreakdown(
            total=total,
            cls_loss=(self.cls_gain * cls_loss).detach(),
            box_loss=(self.box_gain * box_loss).detach(),
            reg_loss=(self.reg_gain * reg_loss).detach(),
        )


class E2EDetectLoss:
    def __init__(
        self,
        nc: int,
        strides: Tuple[int, ...],
        device: torch.device,
        box_gain: float,
        cls_gain: float,
        reg_gain: float,
        one2many_topk: int,
        one2one_topk: int,
        reg_max: int = 1,
        class_positive_weights: torch.Tensor | None = None,
        focal_gamma: float = 0.0,
    ) -> None:
        self.one2many = BranchDetectionLoss(
            nc,
            strides,
            device,
            box_gain,
            cls_gain,
            reg_gain,
            tal_topk=one2many_topk,
            reg_max=reg_max,
            class_positive_weights=class_positive_weights,
            focal_gamma=focal_gamma,
        )
        self.one2one = BranchDetectionLoss(
            nc,
            strides,
            device,
            box_gain,
            cls_gain,
            reg_gain,
            tal_topk=one2one_topk,
            reg_max=reg_max,
            class_positive_weights=class_positive_weights,
            focal_gamma=focal_gamma,
        )

    def __call__(self, outputs: dict[str, Any], batch: dict[str, torch.Tensor]) -> LossBreakdown:
        loss_many = self.one2many(outputs["one2many"], batch)
        loss_one = self.one2one(outputs["one2one"], batch)
        return LossBreakdown(
            total=loss_many.total + loss_one.total,
            cls_loss=loss_many.cls_loss + loss_one.cls_loss,
            box_loss=loss_many.box_loss + loss_one.box_loss,
            reg_loss=loss_many.reg_loss + loss_one.reg_loss,
        )


def compute_loss(
    outputs: dict[str, Any],
    targets: Sequence[torch.Tensor],
    nc: int,
    criterion: E2EDetectLoss,
    device: torch.device,
) -> LossBreakdown:
    if outputs.get("one2many") is None or outputs.get("one2one") is None:
        raise RuntimeError("Model outputs must include one2many and one2one branches for YOLO26-style E2E loss.")

    batch = build_batch_targets(targets, device=device)
    return criterion(outputs, batch)


def run_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: AdamW | None,
    scaler: torch.amp.GradScaler,
    criterion: E2EDetectLoss,
    device: torch.device,
    nc: int,
    use_amp: bool,
) -> Tuple[float, float, float]:
    training = optimizer is not None
    model.train(training)

    total_loss = 0.0
    total_cls = 0.0
    total_box = 0.0
    num_batches = 0

    for images, targets in loader:
        images = images.to(device, non_blocking=True)
        targets = [target.to(device) for target in targets]

        if training:
            optimizer.zero_grad(set_to_none=True)

        with torch.set_grad_enabled(training):
            with torch.amp.autocast(device_type=device.type, enabled=use_amp):
                outputs = model(images)
                loss = compute_loss(
                    outputs,
                    targets,
                    nc=nc,
                    criterion=criterion,
                    device=device,
                )

            if training:
                scaler.scale(loss.total).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=10.0)
                scaler.step(optimizer)
                scaler.update()

        total_loss += float(loss.total.detach().item())
        total_cls += float(loss.cls_loss.item())
        total_box += float((loss.box_loss + loss.reg_loss).item())
        num_batches += 1

    if num_batches == 0:
        return 0.0, 0.0, 0.0

    return total_loss / num_batches, total_cls / num_batches, total_box / num_batches


def validation_detection_metrics(
    model: Any,
    loader: DataLoader,
    device: torch.device,
    nc: int,
    imgsz: int,
    use_amp: bool,
) -> dict[str, Any]:
    """Compute project mAP on validation data using the selected one-to-many decoder.

    This is intentionally separate from the validation-loss pass. Loss remains
    the plateau/early-stopping signal, while mAP50 can optionally select the
    checkpoint that is reported as ``best.pt``.
    """
    was_training = model.training
    model.eval()
    predictions: list[dict[str, torch.Tensor]] = []
    targets_list: list[dict[str, torch.Tensor]] = []

    try:
        with torch.no_grad():
            for images, targets in loader:
                images = images.to(device, non_blocking=True)
                with torch.amp.autocast(device_type=device.type, enabled=use_amp):
                    outputs = model(images)
                    decoded = model.detect.decode_branch(outputs["one2many"])

                batch_predictions = class_aware_nms(
                    decoded,
                    num_classes=nc,
                    score_threshold=0.001,
                    iou_threshold=0.70,
                    max_detections=300,
                )
                for prediction, target in zip(batch_predictions, targets):
                    target_cpu = target.detach().cpu()
                    gt_boxes = (
                        xywhn_to_xyxy(target_cpu[:, 1:5], imgsz)
                        if target_cpu.numel()
                        else torch.zeros((0, 4), dtype=torch.float32)
                    )
                    gt_labels = (
                        target_cpu[:, 0].long()
                        if target_cpu.numel()
                        else torch.zeros((0,), dtype=torch.long)
                    )
                    prediction_cpu = prediction.detach().cpu()
                    predictions.append(
                        {
                            "boxes": prediction_cpu[:, :4].float(),
                            "scores": prediction_cpu[:, 4].float(),
                            "labels": prediction_cpu[:, 5].long(),
                        }
                    )
                    targets_list.append({"boxes": gt_boxes, "labels": gt_labels})
    finally:
        model.train(was_training)

    if not predictions:
        raise RuntimeError("No validation predictions were produced for checkpoint selection")
    return compute_detection_metrics(
        predictions=predictions,
        targets=targets_list,
        num_classes=nc,
        conf_thresh=0.25,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train the custom YOLO26 model from scratch.")
    parser.add_argument("--data-root", type=Path, default=Path("processed-data/baseline"), help="Dataset variant root containing train/valid/test splits")
    parser.add_argument("--epochs", type=int, default=10, help="Number of training epochs")
    parser.add_argument("--batch-size", type=int, default=8, help="Batch size")
    parser.add_argument("--imgsz", type=int, default=640, help="Square input image size")
    parser.add_argument("--lr", type=float, default=1e-4, help="Initial learning rate")
    parser.add_argument("--weight-decay", type=float, default=5e-4, help="AdamW weight decay")
    parser.add_argument("--workers", type=int, default=2, help="DataLoader worker count")
    parser.add_argument("--fraction", type=float, default=1.0, help="Train/validation subset fraction for quick runs")
    parser.add_argument("--seed", type=int, default=42, help="Random seed used for reproducible training")
    parser.add_argument(
        "--class-positive-weight-power",
        type=float,
        default=0.0,
        help="Tempered inverse-frequency power from 0.0 (disabled) to 1.0 for positive class BCE terms",
    )
    parser.add_argument(
        "--focal-gamma",
        type=float,
        default=0.0,
        help="Focal-loss gamma for classification; 0.0 preserves BCE and values such as 2.0 down-weight easy anchors",
    )
    parser.add_argument(
        "--balanced-sampling",
        action="store_true",
        help="Oversample train images containing rare classes without creating augmented image files",
    )
    parser.add_argument(
        "--balanced-sampling-power",
        type=float,
        default=1.0,
        help="Sampling strength from 0.0 (uniform) to 1.0 (full inverse-frequency weighting)",
    )
    parser.add_argument("--device", type=str, default="cuda", help="Training device, e.g. cuda or cuda:0")
    parser.add_argument("--scale", type=str, default="n", help="YOLO26 scale variant")
    parser.add_argument(
        "--num-classes",
        type=int,
        default=None,
        help="Number of detection classes; defaults to the nc value in data-root/data.yaml",
    )
    parser.add_argument("--box-gain", type=float, default=7.5, help="Box IoU loss multiplier")
    parser.add_argument("--cls-gain", type=float, default=0.5, help="Classification BCE loss multiplier")
    parser.add_argument("--reg-gain", type=float, default=1.5, help="Direct-distance L1 or DFL loss multiplier")
    parser.add_argument(
        "--reg-max",
        type=int,
        default=1,
        help="Box distance bins per side; 1 preserves direct regression, values above 1 enable DFL",
    )
    parser.add_argument(
        "--use-p2",
        action="store_true",
        help="Add a stride-4 P2 detection level for small objects; changes the architecture and checkpoint shape",
    )
    parser.add_argument("--one2many-topk", type=int, default=10, help="Task-aligned top-k for one-to-many assignment")
    parser.add_argument("--one2one-topk", type=int, default=1, help="Task-aligned top-k for one-to-one assignment")
    parser.add_argument("--save-dir", type=Path, default=Path("runs/yolo26"), help="Output directory for checkpoints")
    add_plateau_early_stopping_arguments(parser)
    add_epoch_lr_schedule_arguments(parser)
    add_checkpoint_selection_argument(parser)
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if not torch.cuda.is_available() and args.device.startswith("cuda"):
        raise RuntimeError("CUDA device requested but no GPU is available to PyTorch.")

    device = torch.device(args.device)
    seed_everything(args.seed)
    train_loader_generator = torch.Generator().manual_seed(args.seed)
    sampler_generator = torch.Generator().manual_seed(args.seed)
    train_root = args.data_root / "train"
    valid_root = args.data_root / "valid"
    if not train_root.exists() or not valid_root.exists():
        raise FileNotFoundError(f"Expected train/valid folders under {args.data_root}")

    dataset_config = read_yolo_dataset_config(args.data_root)
    if args.num_classes is None:
        args.num_classes = dataset_config.num_classes
    elif args.num_classes != dataset_config.num_classes:
        raise ValueError(
            f"--num-classes={args.num_classes} does not match data.yaml nc={dataset_config.num_classes} "
            f"under {args.data_root}"
        )
    if args.num_classes <= 0:
        raise ValueError("--num-classes must be positive")
    if args.reg_max <= 0:
        raise ValueError("--reg-max must be positive")
    training_control_config = plateau_early_stopping_config_from_args(args)
    epoch_lr_schedule_config = epoch_lr_schedule_config_from_args(args)
    validate_training_control_compatibility(training_control_config, epoch_lr_schedule_config)
    checkpoint_selection = validate_checkpoint_selection(args.checkpoint_selection)

    train_dataset = YoloDetectionDataset(train_root, imgsz=args.imgsz, fraction=args.fraction)
    valid_dataset = YoloDetectionDataset(valid_root, imgsz=args.imgsz, fraction=args.fraction)
    class_positive_weights, class_box_counts = build_positive_class_weights(
        train_dataset,
        args.num_classes,
        power=args.class_positive_weight_power,
    )

    train_sampler = None
    sampler_counts: list[int] | None = None
    background_images = 0
    if args.balanced_sampling:
        train_sampler, sampler_counts, background_images = build_class_balanced_sampler(
            train_dataset,
            args.num_classes,
            power=args.balanced_sampling_power,
            generator=sampler_generator,
        )

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=train_sampler is None,
        sampler=train_sampler,
        num_workers=args.workers,
        pin_memory=device.type == "cuda",
        collate_fn=collate_fn,
        generator=train_loader_generator,
    )
    valid_loader = DataLoader(
        valid_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.workers,
        pin_memory=device.type == "cuda",
        collate_fn=collate_fn,
    )

    model = build_yolo26(
        nc=args.num_classes,
        scale=args.scale,
        reg_max=args.reg_max,
        use_p2=args.use_p2,
    ).to(device)
    model_strides = tuple(int(stride) for stride in model.detect.stride.detach().cpu().tolist())
    optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    training_control = PlateauEarlyStopping(optimizer, training_control_config)
    epoch_lr_scheduler = (
        EpochLRScheduler(optimizer, epoch_lr_schedule_config)
        if epoch_lr_schedule_config.enabled
        else None
    )
    scaler = torch.amp.GradScaler(device.type, enabled=device.type == "cuda")
    criterion = E2EDetectLoss(
        nc=args.num_classes,
        strides=model_strides,
        device=device,
        box_gain=args.box_gain,
        cls_gain=args.cls_gain,
        reg_gain=args.reg_gain,
        one2many_topk=args.one2many_topk,
        one2one_topk=args.one2one_topk,
        reg_max=args.reg_max,
        class_positive_weights=class_positive_weights,
        focal_gamma=args.focal_gamma,
    )
    use_amp = device.type == "cuda"

    args.save_dir.mkdir(parents=True, exist_ok=True)
    best_path = args.save_dir / "best.pt"
    last_path = args.save_dir / "last.pt"

    best_val = float("inf")
    best_selection_value = initial_checkpoint_selection_value(checkpoint_selection)
    print(
        f"Training on device={device}, train_images={len(train_dataset)}, valid_images={len(valid_dataset)}, "
        f"seed={args.seed}"
    )
    print(f"Dataset classes: nc={args.num_classes} names={list(dataset_config.class_names)}")
    class_count_summary = ", ".join(f"class_{class_id}:{count}" for class_id, count in enumerate(class_box_counts))
    class_weight_summary = ", ".join(
        f"class_{class_id}:{weight:.3f}" for class_id, weight in enumerate(class_positive_weights.tolist())
    )
    print(
        f"Positive class weighting: power={args.class_positive_weight_power:.2f} "
        f"box_counts=({class_count_summary}) weights=({class_weight_summary})"
    )
    print(f"Classification focal gamma={args.focal_gamma:g}")
    print(f"Box regression: reg_max={args.reg_max} ({'DFL' if args.reg_max > 1 else 'direct distances'})")
    print(f"Detection feature strides={model_strides} use_p2={args.use_p2}")
    if training_control_config.enabled:
        print(
            "Training control: ReduceLROnPlateau "
            f"patience={training_control_config.reduce_lr_patience} "
            f"factor={training_control_config.reduce_lr_factor:g} "
            f"cooldown={training_control_config.reduce_lr_cooldown} min_lr={training_control_config.min_lr:g}; "
            "early stopping "
            f"patience={training_control_config.early_stopping_patience} "
            f"min_delta={training_control_config.early_stopping_min_delta:g}"
        )
    else:
        print("Training control: ReduceLROnPlateau and early stopping disabled")
    if epoch_lr_scheduler is None:
        print("Epoch LR schedule: constant learning rate without warmup")
    else:
        print(
            f"Epoch LR schedule: {epoch_lr_schedule_config.schedule} "
            f"warmup_epochs={epoch_lr_schedule_config.warmup_epochs} "
            f"warmup_start_factor={epoch_lr_schedule_config.warmup_start_factor:g} "
            f"cosine_final_factor={epoch_lr_schedule_config.cosine_final_factor:g}"
        )
    if sampler_counts is not None:
        count_summary = ", ".join(f"class_{class_id}:{count}" for class_id, count in enumerate(sampler_counts))
        print(
            f"Class-balanced sampling enabled: power={args.balanced_sampling_power:.2f} "
            f"image_counts=({count_summary}) background_images={background_images}"
        )

    for epoch in range(1, args.epochs + 1):
        epoch_schedule_step = (
            epoch_lr_scheduler.set_epoch(epoch) if epoch_lr_scheduler is not None else None
        )
        train_loss, train_cls, train_box = run_epoch(
            model,
            train_loader,
            optimizer,
            scaler,
            criterion,
            device,
            nc=args.num_classes,
            use_amp=use_amp,
        )
        val_loss, val_cls, val_box = run_epoch(
            model,
            valid_loader,
            optimizer=None,
            scaler=scaler,
            criterion=criterion,
            device=device,
            nc=args.num_classes,
            use_amp=use_amp,
        )

        selection_metrics: dict[str, Any] | None = None
        if checkpoint_selection == "map50":
            selection_metrics = validation_detection_metrics(
                model=model,
                loader=valid_loader,
                device=device,
                nc=args.num_classes,
                imgsz=args.imgsz,
                use_amp=use_amp,
            )
            selection_value = float(selection_metrics["map50"])
        else:
            selection_value = val_loss

        if val_loss < best_val:
            best_val = val_loss
        is_best = checkpoint_selection_improved(
            checkpoint_selection,
            selection_value,
            best_selection_value,
        )
        if is_best:
            best_selection_value = selection_value
        control_step = training_control.step(val_loss)
        training_completed = control_step.should_stop or epoch == args.epochs
        stop_reason = (
            "early_stopping"
            if control_step.should_stop
            else "max_epochs"
            if epoch == args.epochs
            else None
        )

        checkpoint = {
            "format_version": 3,
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "train_loss": train_loss,
            "val_loss": val_loss,
            "args": vars(args),
            "class_names": list(dataset_config.class_names),
            "class_box_counts": class_box_counts,
            "class_positive_weights": class_positive_weights.detach().cpu(),
            "best_val_loss": best_val,
            "checkpoint_selection": checkpoint_selection,
            "checkpoint_selection_value": selection_value,
            "best_checkpoint_selection_value": best_selection_value,
            "validation_map50": (
                float(selection_metrics["map50"]) if selection_metrics is not None else None
            ),
            "validation_map50_95": (
                float(selection_metrics["map50_95"]) if selection_metrics is not None else None
            ),
            "training_control": training_control.state_dict(),
            "epoch_lr_schedule": (
                epoch_lr_scheduler.state_dict() if epoch_lr_scheduler is not None else None
            ),
            "learning_rates": list(control_step.learning_rates),
            "training_completed": training_completed,
            "stop_reason": stop_reason,
        }
        torch.save(checkpoint, last_path)
        if is_best:
            torch.save(checkpoint, best_path)

        lr_text = "/".join(f"{lr:.2e}" for lr in control_step.learning_rates)
        reduction_text = " lr_reduced=true" if control_step.lr_reduced else ""
        schedule_text = (
            f" lr_schedule_factor={epoch_schedule_step.factor:.4f}"
            if epoch_schedule_step is not None
            else ""
        )
        selection_text = (
            f" val_map50={float(selection_metrics['map50']):.4f}"
            if selection_metrics is not None
            else ""
        )
        print(
            f"epoch={epoch}/{args.epochs} "
            f"train_loss={train_loss:.4f} train_cls={train_cls:.4f} train_box={train_box:.4f} "
            f"val_loss={val_loss:.4f} val_cls={val_cls:.4f} val_box={val_box:.4f} "
            f"lr={lr_text} checkpoint_selection={checkpoint_selection}:{selection_value:.4f}"
            f"{selection_text} early_stop_bad_epochs={control_step.bad_epochs}{reduction_text}{schedule_text}"
        )
        if control_step.should_stop:
            print(
                f"Early stopping at epoch={epoch}: validation loss did not improve for "
                f"{training_control_config.early_stopping_patience} eligible epochs."
            )
            break

    print(f"Saved best checkpoint to {best_path}")


if __name__ == "__main__":
    main()