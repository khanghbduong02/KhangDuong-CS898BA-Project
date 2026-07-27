"""Train one custom Faster R-CNN fold or standard train/valid dataset.

The script is intentionally a one-fold primitive. Use
``run_faster_rcnn_kfold_cv.py`` for sequential group-disjoint cross-validation.
All input labels must be strict five-field YOLO boxes.
"""
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np
import torch
from torch.optim import AdamW
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler

from detection_metrics import compute_detection_metrics, xywhn_to_xyxy
from model_ema import DEFAULT_EMA_DECAY, ModelEMA, validate_ema_decay
from models.faster_rcnn import build_faster_rcnn
from online_augmentation import (
    DEFAULT_ONLINE_AUGMENTATION,
    ONLINE_AUGMENTATION_CHOICES,
    apply_online_augmentation,
    validate_online_augmentation,
)
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


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

def seed_everything(seed: int) -> None:
    """Seed model initialization, samplers, and worker-local random sources."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True


def seed_worker(_: int) -> None:
    """Initialize Python and NumPy RNGs in a DataLoader worker process."""
    worker_seed = torch.initial_seed() % 2**32
    random.seed(worker_seed)
    np.random.seed(worker_seed)


def read_yolo_label(label_path: Path, num_classes: int) -> torch.Tensor:
    """Read a strict five-field YOLO label file as ``(class, xc, yc, w, h)``."""
    if not label_path.exists():
        return torch.zeros((0, 5), dtype=torch.float32)

    rows: list[list[float]] = []
    for line_number, line in enumerate(label_path.read_text(encoding="utf-8").splitlines(), start=1):
        parts = line.strip().split()
        if not parts:
            continue
        if len(parts) != 5:
            raise ValueError(
                f"{label_path}:{line_number}: expected five-field YOLO detection label "
                "(class x_center y_center width height)"
            )
        try:
            class_id, x_center, y_center, width, height = (float(value) for value in parts)
        except ValueError as exc:
            raise ValueError(f"{label_path}:{line_number}: label values must be numeric") from exc

        if class_id != int(class_id) or not 0 <= class_id < num_classes:
            raise ValueError(
                f"{label_path}:{line_number}: class ID {parts[0]!r} is outside 0..{num_classes - 1}"
            )
        if not (0.0 <= x_center <= 1.0 and 0.0 <= y_center <= 1.0):
            raise ValueError(f"{label_path}:{line_number}: normalized box center must be in [0, 1]")
        if not (0.0 < width <= 1.0 and 0.0 < height <= 1.0):
            raise ValueError(f"{label_path}:{line_number}: normalized box width and height must be in (0, 1]")
        rows.append([class_id, x_center, y_center, width, height])

    return (
        torch.tensor(rows, dtype=torch.float32)
        if rows
        else torch.zeros((0, 5), dtype=torch.float32)
    )


class FasterRCNNDataset(Dataset):
    """YOLO-format detection dataset adapted for the local Faster R-CNN model.

    Returns:
        image_tensor: float32 (C, H, W) in [0, 1].
        target: dict with
            "boxes"  – (N, 4) float32 absolute XYXY
            "labels" – (N,)   int64, 1-indexed because this model reserves
                         class 0 for background internally.
    """

    def __init__(
        self,
        split_root: Path,
        imgsz: int,
        num_classes: int,
        fraction: float = 1.0,
        online_augmentation: str = DEFAULT_ONLINE_AUGMENTATION,
    ) -> None:
        if imgsz <= 0:
            raise ValueError("imgsz must be positive")
        if num_classes <= 0:
            raise ValueError("num_classes must be positive")
        if not 0.0 < fraction <= 1.0:
            raise ValueError("fraction must be in (0, 1]")

        self.images_dir = split_root / "images"
        self.labels_dir = split_root / "labels"
        self.imgsz = imgsz
        self.num_classes = num_classes
        self.online_augmentation = validate_online_augmentation(online_augmentation)
        if not self.images_dir.is_dir() or not self.labels_dir.is_dir():
            raise FileNotFoundError(
                f"Expected images/ and labels/ directories under split root {split_root}"
            )

        paths = [
            p for p in sorted(self.images_dir.iterdir())
            if p.is_file() and p.suffix.lower() in VALID_IMAGE_EXTENSIONS
        ]
        if fraction < 1.0:
            paths = paths[: max(1, int(len(paths) * fraction))]
        self.image_paths = paths

    def __len__(self) -> int:
        return len(self.image_paths)

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        image_path = self.image_paths[index]
        image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
        if image is None:
            raise FileNotFoundError(f"Cannot read image: {image_path}")

        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        orig_h, orig_w = image.shape[:2]

        if self.imgsz < orig_w or self.imgsz < orig_h:
            interp = cv2.INTER_AREA
        elif self.imgsz > orig_w or self.imgsz > orig_h:
            interp = cv2.INTER_CUBIC
        else:
            interp = cv2.INTER_LINEAR

        image = cv2.resize(image, (self.imgsz, self.imgsz), interpolation=interp)
        image_tensor = torch.from_numpy(image).permute(2, 0, 1).float() / 255.0
        image_tensor = apply_online_augmentation(image_tensor, self.online_augmentation)

        label_path = self.labels_dir / f"{image_path.stem}.txt"
        raw = read_yolo_label(label_path, self.num_classes)
        converted = xywhn_to_xyxy(raw[:, 1:5], self.imgsz).clamp(min=0.0, max=float(self.imgsz))
        keep = (converted[:, 2] > converted[:, 0]) & (converted[:, 3] > converted[:, 1])

        boxes_xyxy = converted[keep]
        labels = raw[:, 0].long()[keep] + 1

        return image_tensor, {"boxes": boxes_xyxy, "labels": labels}


def collate_fn(
    batch: Sequence[Tuple[torch.Tensor, Dict[str, torch.Tensor]]]
) -> Tuple[List[torch.Tensor], List[Dict[str, torch.Tensor]]]:
    images = [item[0] for item in batch]
    targets = [item[1] for item in batch]
    return images, targets


def _read_label_class_counts(label_path: Path, num_classes: int) -> list[int]:
    labels = read_yolo_label(label_path, num_classes)
    return torch.bincount(labels[:, 0].long(), minlength=num_classes).tolist()


def build_positive_class_weights(
    dataset: FasterRCNNDataset,
    num_classes: int,
    power: float = 0.0,
) -> tuple[torch.Tensor, list[int]]:
    """Build normalized inverse-frequency weights for foreground cross-entropy classes."""
    if not 0.0 <= power <= 1.0:
        raise ValueError("Positive class-weight power must be between 0.0 and 1.0")

    class_box_counts = [0 for _ in range(num_classes)]
    for image_path in dataset.image_paths:
        label_counts = _read_label_class_counts(
            dataset.labels_dir / f"{image_path.stem}.txt",
            num_classes,
        )
        for class_id, count in enumerate(label_counts):
            class_box_counts[class_id] += count

    if any(count == 0 for count in class_box_counts) and power == 0.0:
        return torch.ones(num_classes, dtype=torch.float32), class_box_counts
    if any(count == 0 for count in class_box_counts):
        raise ValueError(
            f"Cannot build positive class weights because box counts are {class_box_counts}"
        )

    max_count = max(class_box_counts)
    raw_weights = torch.tensor(
        [(max_count / count) ** power for count in class_box_counts],
        dtype=torch.float32,
    )
    counts_tensor = torch.tensor(class_box_counts, dtype=torch.float32)
    normalized_weights = raw_weights / (counts_tensor * raw_weights).sum() * counts_tensor.sum()
    return normalized_weights, class_box_counts


def build_class_balanced_sampler(
    dataset: FasterRCNNDataset,
    num_classes: int,
    power: float,
    generator: torch.Generator,
) -> tuple[WeightedRandomSampler, list[int], int]:
    """Oversample images containing rare classes without changing source files."""
    if not 0.0 <= power <= 1.0:
        raise ValueError("Class-balanced sampling power must be between 0.0 and 1.0")

    sample_class_ids: list[set[int]] = []
    class_image_counts = [0 for _ in range(num_classes)]
    background_images = 0
    for image_path in dataset.image_paths:
        label_counts = _read_label_class_counts(
            dataset.labels_dir / f"{image_path.stem}.txt",
            num_classes,
        )
        class_ids = {class_id for class_id, count in enumerate(label_counts) if count > 0}
        sample_class_ids.append(class_ids)
        if not class_ids:
            background_images += 1
        for class_id in class_ids:
            class_image_counts[class_id] += 1

    if any(count == 0 for count in class_image_counts):
        raise ValueError(
            f"Cannot build class-balanced sampler because class image counts are {class_image_counts}"
        )

    majority_count = max(class_image_counts)
    class_weights = [(majority_count / count) ** power for count in class_image_counts]
    sample_weights = [
        max((class_weights[class_id] for class_id in class_ids), default=1.0)
        for class_ids in sample_class_ids
    ]
    sampler = WeightedRandomSampler(
        torch.as_tensor(sample_weights, dtype=torch.double),
        num_samples=len(dataset),
        replacement=True,
        generator=generator,
    )
    return sampler, class_image_counts, background_images


# ---------------------------------------------------------------------------
# Training / validation loop
# ---------------------------------------------------------------------------

def _get_loss(loss_dict: Dict[str, torch.Tensor], key: str) -> float:
    v = loss_dict.get(key)
    return float(v.detach().item()) if v is not None else 0.0


def run_epoch(
    model: torch.nn.Module,
    loader: DataLoader,
    optimizer: Optional[AdamW],
    scaler: torch.amp.GradScaler,
    device: torch.device,
    use_amp: bool,
    ema: ModelEMA | None = None,
) -> Tuple[float, float, float]:
    """Run one training or validation epoch.

    ``optimizer is None`` signals validation. The custom model exposes
    ``compute_losses=True``, allowing validation to run in ``eval`` mode so
    BatchNorm running statistics remain training-only.

    Returns:
        (total_loss, cls_loss, box_loss) per-batch averages.
        box_loss combines RoI box regression + RPN objectness + RPN box regression.
    """
    training = optimizer is not None
    model.train(training)

    total = cls = box = 0.0
    n = 0

    for images, targets in loader:
        images_dev: List[torch.Tensor] = [img.to(device, non_blocking=True) for img in images]
        targets_dev: List[Dict[str, torch.Tensor]] = [
            {k: v.to(device) for k, v in t.items()} for t in targets
        ]

        if training:
            optimizer.zero_grad(set_to_none=True)  # type: ignore[union-attr]

        with torch.set_grad_enabled(training):
            with torch.amp.autocast(device_type=device.type, enabled=use_amp):
                loss_dict: Dict[str, torch.Tensor] = model(
                    images_dev,
                    targets_dev,
                    compute_losses=True,
                )
                losses: torch.Tensor = sum(loss_dict.values())  # type: ignore[assignment]

            if not torch.isfinite(losses):
                raise FloatingPointError("Faster R-CNN produced a non-finite loss")

            if training:
                scaler.scale(losses).backward()
                scaler.unscale_(optimizer)  # type: ignore[arg-type]
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=10.0)
                scaler.step(optimizer)  # type: ignore[arg-type]
                scaler.update()
                if ema is not None:
                    ema.update(model)

        total += float(losses.detach().item())
        cls += _get_loss(loss_dict, "loss_classifier")
        box += (
            _get_loss(loss_dict, "loss_box_reg")
            + _get_loss(loss_dict, "loss_objectness")
            + _get_loss(loss_dict, "loss_rpn_box_reg")
        )
        n += 1

    if n == 0:
        return 0.0, 0.0, 0.0
    return total / n, cls / n, box / n


def validation_detection_metrics(
    model: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
    num_classes: int,
    use_amp: bool,
) -> dict[str, Any]:
    """Compute project mAP for optional checkpoint selection.

    The validation loss remains the scheduler and early-stopping monitor. This
    helper only decides which frozen checkpoint becomes ``best.pt``.
    """
    was_training = model.training
    model.eval()
    predictions: list[dict[str, torch.Tensor]] = []
    targets_list: list[dict[str, torch.Tensor]] = []

    try:
        with torch.no_grad():
            for images, targets in loader:
                images_dev = [image.to(device, non_blocking=True) for image in images]
                with torch.amp.autocast(device_type=device.type, enabled=use_amp):
                    batch_predictions = model(images_dev)
                if not isinstance(batch_predictions, list):
                    raise RuntimeError("Faster R-CNN inference did not return a prediction list")

                for prediction, target in zip(batch_predictions, targets):
                    prediction_cpu = {
                        "boxes": prediction["boxes"].detach().cpu().float(),
                        "scores": prediction["scores"].detach().cpu().float(),
                        "labels": (prediction["labels"].detach().cpu().long() - 1).clamp(min=0),
                    }
                    target_cpu = {
                        "boxes": target["boxes"].detach().cpu().float(),
                        "labels": (target["labels"].detach().cpu().long() - 1).clamp(min=0),
                    }
                    predictions.append(prediction_cpu)
                    targets_list.append(target_cpu)
    finally:
        model.train(was_training)

    if not predictions:
        raise RuntimeError("No validation predictions were produced for checkpoint selection")
    return compute_detection_metrics(
        predictions=predictions,
        targets=targets_list,
        num_classes=num_classes,
        conf_thresh=0.25,
    )


def build_optimizer(
    model: torch.nn.Module,
    lr: float,
    weight_decay: float,
    backbone_lr_multiplier: float,
) -> AdamW:
    """Create AdamW groups with an optional lower rate for a pretrained backbone."""
    if lr <= 0.0 or weight_decay < 0.0:
        raise ValueError("Learning rate must be positive and weight decay must be non-negative")
    if not 0.0 < backbone_lr_multiplier <= 1.0:
        raise ValueError("backbone_lr_multiplier must be in (0, 1]")
    if not hasattr(model, "backbone"):
        raise ValueError("Faster R-CNN model must expose a backbone for optimizer grouping")

    backbone_parameters = [parameter for parameter in model.backbone.parameters() if parameter.requires_grad]
    backbone_ids = {id(parameter) for parameter in backbone_parameters}
    detector_parameters = [
        parameter
        for parameter in model.parameters()
        if parameter.requires_grad and id(parameter) not in backbone_ids
    ]
    if not backbone_parameters or not detector_parameters:
        raise ValueError("Expected both backbone and detector parameters")

    return AdamW(
        [
            {"params": backbone_parameters, "lr": lr * backbone_lr_multiplier},
            {"params": detector_parameters, "lr": lr},
        ],
        weight_decay=weight_decay,
    )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train the local Faster R-CNN model on one standard train/valid dataset or CV fold."
    )
    parser.add_argument(
        "--data-root", type=Path, default=Path("processed-data/baseline"),
        help="Dataset variant root (must contain train/ and valid/ splits).",
    )
    parser.add_argument("--epochs", type=int, default=10, help="Number of training epochs")
    parser.add_argument("--batch-size", type=int, default=8, help="Batch size")
    parser.add_argument("--imgsz", type=int, default=640, help="Square input image size")
    parser.add_argument("--lr", type=float, default=1e-4, help="AdamW learning rate.")
    parser.add_argument("--weight-decay", type=float, default=5e-4, help="AdamW weight decay.")
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--fraction", type=float, default=1.0,
                        help="Subset fraction of each split for quick experiments.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed used for reproducible training")
    parser.add_argument(
        "--class-positive-weight-power",
        type=float,
        default=0.0,
        help="Tempered inverse-frequency power from 0.0 (disabled) to 1.0 for foreground cross-entropy classes",
    )
    parser.add_argument(
        "--balanced-sampling",
        action="store_true",
        help="Oversample train images containing rare classes without creating new files",
    )
    parser.add_argument(
        "--balanced-sampling-power",
        type=float,
        default=1.0,
        help="Sampling strength from 0.0 (uniform) to 1.0 (full inverse-frequency weighting)",
    )
    parser.add_argument(
        "--online-augmentation",
        choices=ONLINE_AUGMENTATION_CHOICES,
        default=DEFAULT_ONLINE_AUGMENTATION,
        help=(
            "Training-only image augmentation; photometric applies conservative brightness, contrast, gamma, "
            "and noise changes without changing boxes, labels, or source files"
        ),
    )
    parser.add_argument("--device", type=str, default="cuda",
                        help="Training device, e.g. 'cuda' or 'cpu'.")
    parser.add_argument(
        "--scale", type=str, default="m",
        choices=["s", "m", "l"],
        help="Model size: s=small, m=medium, l=large (same as --scale in YOLO26).",
    )
    parser.add_argument(
        "--backbone-weights",
        choices=["none", "imagenet"],
        default="none",
        help="Backbone initialization: random local weights or compatible torchvision ImageNet weights",
    )
    parser.add_argument(
        "--backbone-lr-multiplier",
        type=float,
        default=1.0,
        help="Backbone learning-rate multiplier; use a smaller value such as 0.1 for ImageNet transfer learning",
    )
    parser.add_argument(
        "--use-p2",
        action="store_true",
        help="Add a stride-4 P2 FPN/RPN level for small objects; changes the architecture and checkpoint shape",
    )
    parser.add_argument("--num-classes", type=int, default=None,
                        help="Override number of classes (default: read from data.yaml).")
    parser.add_argument(
        "--save-dir", type=Path, default=Path("runs/faster_rcnn"),
        help="Output directory for best.pt and last.pt checkpoints.",
    )
    parser.add_argument(
        "--ema-decay",
        type=float,
        default=DEFAULT_EMA_DECAY,
        help=(
            "Per-epoch EMA retention factor for model weights and floating buffers; 0 disables EMA. "
            "A non-zero EMA requires --checkpoint-selection map50."
        ),
    )
    add_plateau_early_stopping_arguments(parser)
    add_epoch_lr_schedule_arguments(parser)
    add_checkpoint_selection_argument(parser)
    parser.add_argument("--dry-run", action="store_true", help="Validate data and configuration without training")
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    args = parse_args()

    if not torch.cuda.is_available() and args.device.startswith("cuda"):
        raise RuntimeError("CUDA device requested but no GPU is available to PyTorch.")
    if args.epochs <= 0 or args.batch_size <= 0 or args.imgsz <= 0:
        raise ValueError("--epochs, --batch-size, and --imgsz must be positive")
    if args.imgsz % 64 != 0:
        raise ValueError("--imgsz must be divisible by 64 for the P3--P6 anchor geometry")
    if args.workers < 0 or not 0.0 < args.fraction <= 1.0:
        raise ValueError("--workers must be non-negative and --fraction must be in (0, 1]")
    if not 0.0 <= args.class_positive_weight_power <= 1.0:
        raise ValueError("--class-positive-weight-power must be in [0, 1]")
    if not 0.0 <= args.balanced_sampling_power <= 1.0:
        raise ValueError("--balanced-sampling-power must be in [0, 1]")
    if not 0.0 < args.backbone_lr_multiplier <= 1.0:
        raise ValueError("--backbone-lr-multiplier must be in (0, 1]")
    if args.backbone_weights == "imagenet" and args.scale not in {"s", "m"}:
        raise ValueError("--backbone-weights imagenet supports only --scale s or m")
    args.online_augmentation = validate_online_augmentation(args.online_augmentation)
    args.ema_decay = validate_ema_decay(args.ema_decay)
    training_control_config = plateau_early_stopping_config_from_args(args)
    epoch_lr_schedule_config = epoch_lr_schedule_config_from_args(args)
    validate_training_control_compatibility(training_control_config, epoch_lr_schedule_config)
    checkpoint_selection = validate_checkpoint_selection(args.checkpoint_selection)
    if args.ema_decay > 0.0 and checkpoint_selection != "map50":
        raise ValueError("--ema-decay requires --checkpoint-selection map50")

    device = torch.device(args.device)
    args.data_root = args.data_root.resolve()
    args.save_dir = args.save_dir.resolve()
    seed_everything(args.seed)
    train_loader_generator = torch.Generator().manual_seed(args.seed)
    valid_loader_generator = torch.Generator().manual_seed(args.seed)
    sampler_generator = torch.Generator().manual_seed(args.seed)

    train_root = args.data_root / "train"
    valid_root = args.data_root / "valid"
    if not train_root.exists() or not valid_root.exists():
        raise FileNotFoundError(
            f"Expected train/ and valid/ splits under {args.data_root}."
        )

    dataset_config = read_yolo_dataset_config(args.data_root)
    if args.num_classes is None:
        args.num_classes = dataset_config.num_classes
    elif args.num_classes != dataset_config.num_classes:
        raise ValueError(
            f"--num-classes={args.num_classes} does not match data.yaml nc={dataset_config.num_classes} "
            f"under {args.data_root}"
        )
    num_classes = args.num_classes
    class_names = list(dataset_config.class_names)

    train_dataset = FasterRCNNDataset(
        train_root,
        imgsz=args.imgsz,
        num_classes=num_classes,
        fraction=args.fraction,
        online_augmentation=args.online_augmentation,
    )
    valid_dataset = FasterRCNNDataset(
        valid_root,
        imgsz=args.imgsz,
        num_classes=num_classes,
        fraction=args.fraction,
    )
    class_positive_weights, class_box_counts = build_positive_class_weights(
        train_dataset,
        num_classes,
        power=args.class_positive_weight_power,
    )

    train_sampler = None
    sampler_counts: list[int] | None = None
    background_images = 0
    if args.balanced_sampling:
        train_sampler, sampler_counts, background_images = build_class_balanced_sampler(
            train_dataset,
            num_classes,
            power=args.balanced_sampling_power,
            generator=sampler_generator,
        )

    train_loader = DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=train_sampler is None,
        sampler=train_sampler,
        num_workers=args.workers, pin_memory=device.type == "cuda",
        collate_fn=collate_fn, generator=train_loader_generator, worker_init_fn=seed_worker,
    )
    valid_loader = DataLoader(
        valid_dataset, batch_size=args.batch_size, shuffle=False,
        num_workers=args.workers, pin_memory=device.type == "cuda",
        collate_fn=collate_fn, generator=valid_loader_generator, worker_init_fn=seed_worker,
    )

    model = build_faster_rcnn(
        nc=num_classes,
        scale=args.scale,
        min_size=args.imgsz,
        max_size=args.imgsz,
        class_positive_weights=class_positive_weights,
        backbone_weights=args.backbone_weights,
        use_p2=args.use_p2,
    ).to(device)

    optimizer = build_optimizer(
        model,
        lr=args.lr,
        weight_decay=args.weight_decay,
        backbone_lr_multiplier=args.backbone_lr_multiplier,
    )
    training_control = PlateauEarlyStopping(optimizer, training_control_config)
    ema = (
        ModelEMA(model, decay=args.ema_decay, updates_per_epoch=len(train_loader))
        if args.ema_decay > 0.0
        else None
    )
    epoch_lr_scheduler = (
        EpochLRScheduler(optimizer, epoch_lr_schedule_config)
        if epoch_lr_schedule_config.enabled
        else None
    )
    use_amp = device.type == "cuda"
    scaler = torch.amp.GradScaler(device.type, enabled=use_amp)

    print(
        f"Training custom Faster R-CNN on device={device}, train_images={len(train_dataset)}, "
        f"valid_images={len(valid_dataset)}, seed={args.seed}, scale={args.scale}, "
        f"backbone_weights={args.backbone_weights}"
    )
    print(
        f"Backbone initialization={model.backbone_initialization} "
        f"backbone_lr_multiplier={args.backbone_lr_multiplier:g} "
        f"use_p2={args.use_p2} feature_strides={model.feature_strides}"
    )
    print(f"Dataset classes: nc={num_classes} names={class_names}")
    class_count_summary = ", ".join(
        f"class_{class_id}:{count}" for class_id, count in enumerate(class_box_counts)
    )
    class_weight_summary = ", ".join(
        f"class_{class_id}:{weight:.3f}"
        for class_id, weight in enumerate(class_positive_weights.tolist())
    )
    print(
        f"Positive class weighting: power={args.class_positive_weight_power:.2f} "
        f"box_counts=({class_count_summary}) weights=({class_weight_summary})"
    )
    print(
        f"Online training augmentation: {args.online_augmentation} "
        "(validation images, labels, and source files unchanged)"
    )
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
    if ema is None:
        print("Model EMA: disabled")
    else:
        print(
            f"Model EMA: decay_per_epoch={ema.decay:g} updates_per_epoch={ema.updates_per_epoch} "
            f"per_update_decay={ema.per_update_decay:.8f}; mAP50 selection uses EMA weights"
        )
    if sampler_counts is not None:
        count_summary = ", ".join(
            f"class_{class_id}:{count}" for class_id, count in enumerate(sampler_counts)
        )
        print(
            f"Class-balanced sampling enabled: power={args.balanced_sampling_power:.2f} "
            f"image_counts=({count_summary}) background_images={background_images}"
        )
    if args.dry_run:
        print("Dry run complete; no training was started.")
        return

    args.save_dir.mkdir(parents=True, exist_ok=True)
    best_path = args.save_dir / "best.pt"
    last_path = args.save_dir / "last.pt"
    history_path = args.save_dir / "history.jsonl"
    history_path.write_text("", encoding="utf-8")
    best_val = float("inf")
    best_selection_value = initial_checkpoint_selection_value(checkpoint_selection)

    for epoch in range(1, args.epochs + 1):
        epoch_schedule_step = (
            epoch_lr_scheduler.set_epoch(epoch) if epoch_lr_scheduler is not None else None
        )
        train_loss, train_cls, train_box = run_epoch(
            model, train_loader, optimizer, scaler, device, use_amp=use_amp, ema=ema,
        )
        val_loss, val_cls, val_box = run_epoch(
            model, valid_loader, optimizer=None, scaler=scaler, device=device, use_amp=use_amp,
        )

        selection_metrics: dict[str, Any] | None = None
        if checkpoint_selection == "map50":
            if ema is None:
                selection_metrics = validation_detection_metrics(
                    model=model,
                    loader=valid_loader,
                    device=device,
                    num_classes=num_classes,
                    use_amp=use_amp,
                )
            else:
                with ema.average_parameters(model):
                    selection_metrics = validation_detection_metrics(
                        model=model,
                        loader=valid_loader,
                        device=device,
                        num_classes=num_classes,
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

        checkpoint: Dict[str, Any] = {
            "format_version": 5,
            "model_name": "custom_faster_rcnn",
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "train_loss": train_loss,
            "val_loss": val_loss,
            "num_classes": num_classes,
            "class_names": class_names,
            "args": vars(args),
            "class_box_counts": class_box_counts,
            "class_positive_weights": class_positive_weights.detach().cpu(),
            "model_inference_settings": model.inference_settings(),
            "backbone_initialization": model.backbone_initialization,
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
            "checkpoint_weight_source": "raw",
            "checkpoint_selection_weight_source": "ema" if ema is not None else "raw",
            "ema_metadata": ema.metadata() if ema is not None else None,
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
            if ema is None:
                torch.save(checkpoint, best_path)
            else:
                best_checkpoint = dict(checkpoint)
                best_checkpoint["model_state_dict"] = ema.model_state_dict()
                best_checkpoint["checkpoint_weight_source"] = "ema"
                torch.save(best_checkpoint, best_path)

        history_record = {
            "epoch": epoch,
            "train_loss": train_loss,
            "train_cls_loss": train_cls,
            "train_box_loss": train_box,
            "val_loss": val_loss,
            "val_cls_loss": val_cls,
            "val_box_loss": val_box,
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
            "learning_rates": list(control_step.learning_rates),
            "lr_schedule_factor": (
                epoch_schedule_step.factor if epoch_schedule_step is not None else 1.0
            ),
            "checkpoint_selection_weight_source": "ema" if ema is not None else "raw",
            "ema_metadata": ema.metadata() if ema is not None else None,
            "lr_reduced": control_step.lr_reduced,
            "early_stopping_bad_epochs": control_step.bad_epochs,
            "training_completed": training_completed,
            "stop_reason": stop_reason,
        }
        with history_path.open("a", encoding="utf-8") as history_file:
            history_file.write(json.dumps(history_record) + "\n")

        lr_text = "/".join(f"{lr:.2e}" for lr in control_step.learning_rates)
        reduction_text = " lr_reduced=true" if control_step.lr_reduced else ""
        schedule_text = (
            f" lr_schedule_factor={epoch_schedule_step.factor:.4f}"
            if epoch_schedule_step is not None
            else ""
        )
        ema_text = (
            f" ema_updates={ema.updates} selection_weights=ema"
            if ema is not None
            else " selection_weights=raw"
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
            f"{selection_text} early_stop_bad_epochs={control_step.bad_epochs}{reduction_text}{schedule_text}{ema_text}"
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
