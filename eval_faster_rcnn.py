"""Evaluate a custom Faster R-CNN checkpoint on a standard validation/test split."""
from __future__ import annotations

import argparse
import json
import pickle
import warnings
from pathlib import Path
from typing import Any, Dict, List

import torch
from torch.utils.data import DataLoader

from detection_metrics import compute_detection_metrics
from models.faster_rcnn import build_faster_rcnn
from train_faster_rcnn import FasterRCNNDataset, collate_fn
from yolo_dataset_config import read_yolo_dataset_config


# ---------------------------------------------------------------------------
# Checkpoint loading (mirrors eval_yolo26.py robustness)
# ---------------------------------------------------------------------------

def _load_checkpoint(checkpoint_path: Path, device: torch.device) -> dict:
    try:
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=True)
        if not isinstance(checkpoint, dict):
            raise RuntimeError(f"Checkpoint is not a metadata dictionary: {checkpoint_path}")
        return checkpoint
    except (TypeError, pickle.UnpicklingError):
        pass

    try:
        if hasattr(torch.serialization, "add_safe_globals"):
            torch.serialization.add_safe_globals([Path, type(Path())])
        return torch.load(checkpoint_path, map_location=device, weights_only=True)
    except Exception:
        pass

    # All checkpoints used here are local project artifacts. Older checkpoints
    # may store Path values in their saved argument dictionary.
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=FutureWarning, module="torch.serialization")
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    if not isinstance(checkpoint, dict):
        raise RuntimeError(f"Checkpoint is not a metadata dictionary: {checkpoint_path}")
    return checkpoint


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate a trained Faster R-CNN model."
    )
    parser.add_argument("--checkpoint", type=Path, required=True,
                        help="Path to a saved checkpoint (best.pt or last.pt).")
    parser.add_argument(
        "--data-root", type=Path, default=Path("processed-data/baseline"),
        help="Dataset variant root used during training.",
    )
    parser.add_argument("--split", type=str, choices=["valid", "test"], default="valid")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument(
        "--imgsz",
        type=int,
        default=None,
        help="Square input image size; defaults to the saved checkpoint value or 640",
    )
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--fraction", type=float, default=1.0,
                        help="Subset fraction for quick evaluation.")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument(
        "--scale", type=str, default=None,
        choices=["s", "m", "l"],
        help="Model size used during training (overridden by checkpoint metadata if present).",
    )
    parser.add_argument("--conf-thresh", type=float, default=0.25,
                        help="Confidence threshold for precision/recall/confusion matrix.")
    parser.add_argument("--num-classes", type=int, default=None,
                        help="Validate an expected foreground-class count against data.yaml and checkpoint metadata.")
    parser.add_argument(
        "--nms-score-thresh",
        "--score-thresh",
        dest="nms_score_thresh",
        type=float,
        default=0.001,
        help="Candidate score floor before built-in per-class NMS",
    )
    parser.add_argument("--nms-iou", type=float, default=0.70,
                        help="Per-class NMS IoU threshold")
    parser.add_argument("--max-det", type=int, default=300,
                        help="Maximum detections retained per image after NMS")
    parser.add_argument(
        "--metrics-output",
        type=Path,
        default=None,
        help="Optional JSON path for machine-readable evaluation metrics",
    )
    return parser.parse_args()


def _resolve_evaluation_settings(
    checkpoint: Dict[str, Any],
    args: argparse.Namespace,
) -> tuple[str, int]:
    """Use explicit CLI settings first, then checkpoint settings, then legacy defaults."""
    saved_args = checkpoint.get("args", {})
    if not isinstance(saved_args, dict):
        saved_args = {}

    scale = args.scale if args.scale is not None else str(saved_args.get("scale", "m"))
    imgsz = args.imgsz if args.imgsz is not None else int(saved_args.get("imgsz", 640))
    if scale not in {"s", "m", "l"}:
        raise ValueError(f"Resolved scale {scale!r} is not one of s, m, or l")
    if imgsz <= 0 or imgsz % 64 != 0:
        raise ValueError("Resolved imgsz must be positive and divisible by 64")
    return scale, imgsz


def _load_positive_class_weights(
    checkpoint: Dict[str, Any],
    num_classes: int,
) -> torch.Tensor:
    """Restore saved foreground class weights or use neutral legacy defaults."""
    saved_weights = checkpoint.get("class_positive_weights")
    if saved_weights is None:
        return torch.ones(num_classes, dtype=torch.float32)

    weights = torch.as_tensor(saved_weights, dtype=torch.float32).reshape(-1)
    if weights.numel() != num_classes:
        raise ValueError(
            f"Checkpoint has {weights.numel()} foreground class weights, expected {num_classes}"
        )
    if not torch.isfinite(weights).all() or (weights <= 0).any():
        raise ValueError("Checkpoint foreground class weights must be finite and greater than zero")
    return weights


def _load_model_state(model: torch.nn.Module, checkpoint: Dict[str, Any]) -> None:
    """Load a current or pre-modernization local Faster R-CNN state dictionary."""
    state_dict = checkpoint.get("model_state_dict")
    if not isinstance(state_dict, dict):
        raise ValueError("Checkpoint does not contain a model_state_dict")

    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    allowed_missing = {"classification_weights"}
    unexpected_set = set(unexpected)
    missing_set = set(missing)
    if unexpected_set or missing_set - allowed_missing:
        raise RuntimeError(
            "Checkpoint is incompatible with the requested Faster R-CNN model: "
            f"missing={sorted(missing_set)}, unexpected={sorted(unexpected_set)}"
        )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    args = parse_args()

    if not torch.cuda.is_available() and args.device.startswith("cuda"):
        raise RuntimeError("CUDA device requested but no GPU is available to PyTorch.")
    if args.batch_size <= 0 or args.workers < 0 or not 0.0 < args.fraction <= 1.0:
        raise ValueError("Invalid batch size, worker count, or fraction")
    if not 0.0 <= args.conf_thresh <= 1.0:
        raise ValueError("--conf-thresh must be in [0, 1]")
    if not 0.0 <= args.nms_score_thresh <= 1.0:
        raise ValueError("--nms-score-thresh must be in [0, 1]")
    if not 0.0 < args.nms_iou <= 1.0:
        raise ValueError("--nms-iou must be in (0, 1]")
    if args.max_det <= 0:
        raise ValueError("--max-det must be positive")

    device = torch.device(args.device)
    args.checkpoint = args.checkpoint.resolve()
    args.data_root = args.data_root.resolve()
    split_root = args.data_root / args.split
    if not split_root.exists():
        raise FileNotFoundError(f"Split folder not found: {split_root}")

    checkpoint = _load_checkpoint(args.checkpoint, device)

    dataset_config = read_yolo_dataset_config(args.data_root)
    num_classes = dataset_config.num_classes
    class_names = list(dataset_config.class_names)
    if args.num_classes is not None and args.num_classes != num_classes:
        raise ValueError(
            f"--num-classes={args.num_classes} does not match data.yaml nc={num_classes}"
        )
    saved_num_classes = checkpoint.get("num_classes")
    if saved_num_classes is not None and int(saved_num_classes) != num_classes:
        raise ValueError(
            f"Checkpoint num_classes={saved_num_classes} does not match data.yaml nc={num_classes}"
        )
    saved_names = checkpoint.get("class_names")
    if saved_names is not None and tuple(str(name) for name in saved_names) != dataset_config.class_names:
        raise ValueError(
            "Checkpoint class names do not match the selected dataset taxonomy: "
            f"checkpoint={saved_names!r}, dataset={class_names!r}"
        )

    scale, imgsz = _resolve_evaluation_settings(checkpoint, args)
    class_positive_weights = _load_positive_class_weights(checkpoint, num_classes)
    saved_args = checkpoint.get("args", {})
    if not isinstance(saved_args, dict):
        saved_args = {}
    backbone_weights = str(saved_args.get("backbone_weights", "none"))
    backbone_lr_multiplier = float(saved_args.get("backbone_lr_multiplier", 1.0))
    backbone_initialization = str(checkpoint.get("backbone_initialization", "random"))

    model = build_faster_rcnn(
        nc=num_classes,
        scale=scale,
        min_size=imgsz,
        max_size=imgsz,
        class_positive_weights=class_positive_weights,
        score_threshold=args.nms_score_thresh,
        nms_threshold=args.nms_iou,
        max_detections=args.max_det,
    ).to(device)
    _load_model_state(model, checkpoint)
    model.eval()

    dataset = FasterRCNNDataset(
        split_root,
        imgsz=imgsz,
        num_classes=num_classes,
        fraction=args.fraction,
    )
    loader = DataLoader(
        dataset, batch_size=args.batch_size, shuffle=False,
        num_workers=args.workers, pin_memory=device.type == "cuda",
        collate_fn=collate_fn,
    )

    all_predictions: List[Dict[str, torch.Tensor]] = []
    all_targets: List[Dict[str, torch.Tensor]] = []

    with torch.no_grad():
        for images, targets in loader:
            images_dev = [img.to(device, non_blocking=True) for img in images]

            preds = model(images_dev)
            if not isinstance(preds, list):
                raise RuntimeError("Faster R-CNN inference did not return a prediction list")

            for pred, tgt in zip(preds, targets):
                # The local Faster R-CNN model reserves class 0 for background.
                # Convert its foreground labels back to zero-indexed metrics IDs.
                pred_labels = (pred["labels"].cpu() - 1).clamp(min=0)
                gt_labels = (tgt["labels"] - 1).clamp(min=0)

                all_predictions.append({
                    "boxes": pred["boxes"].cpu(),
                    "scores": pred["scores"].cpu(),
                    "labels": pred_labels,
                })
                all_targets.append({
                    "boxes": tgt["boxes"],           # already absolute XYXY
                    "labels": gt_labels,
                })

    if not all_predictions:
        raise RuntimeError(
            "No predictions generated. Verify dataset path and --fraction argument."
        )

    metrics = compute_detection_metrics(
        all_predictions,
        all_targets,
        num_classes=num_classes,
        conf_thresh=args.conf_thresh,
    )

    if args.metrics_output is not None:
        args.metrics_output = args.metrics_output.resolve()
        metrics_output = {
            "model": "custom_faster_rcnn",
            "checkpoint": str(args.checkpoint),
            "checkpoint_epoch": int(checkpoint.get("epoch", 0)),
            "data_root": str(args.data_root),
            "split": args.split,
            "image_count": len(dataset),
            "confidence_threshold": args.conf_thresh,
            "evaluation_settings": {
                "scale": scale,
                "imgsz": imgsz,
                "backbone_weights": backbone_weights,
                "backbone_initialization": backbone_initialization,
                "backbone_lr_multiplier": backbone_lr_multiplier,
                "class_names": class_names,
                "positive_class_weights": class_positive_weights.tolist(),
                "postprocess": "per_class_nms",
                "nms_iou": args.nms_iou,
                "nms_score_threshold": args.nms_score_thresh,
                "max_det": args.max_det,
            },
            "metrics": metrics,
        }
        args.metrics_output.parent.mkdir(parents=True, exist_ok=True)
        args.metrics_output.write_text(json.dumps(metrics_output, indent=2) + "\n", encoding="utf-8")

    print(
        f"Evaluation settings: scale={scale} imgsz={imgsz} "
        f"backbone_weights={backbone_weights} backbone_initialization={backbone_initialization} "
        f"postprocess=per_class_nms nms_iou={args.nms_iou:g} "
        f"nms_score_thresh={args.nms_score_thresh:g} max_det={args.max_det}"
    )
    print(
        f"model=faster_rcnn (scale={scale}) split={args.split} images={len(dataset)} "
        f"mAP50={metrics['map50']:.4f} mAP50-95={metrics['map50_95']:.4f} "
        f"precision={metrics['precision']:.4f} recall={metrics['recall']:.4f}"
    )

    print("Per-class metrics (IoU50 PR uses --conf-thresh):")
    for cls in metrics["per_class"]:
        class_id = int(cls["class_id"])
        print(
            f"  class={class_names[class_id]} id={class_id} gt={int(cls['num_gt'])} "
            f"AP50={cls['ap50']:.4f} AP50-95={cls['ap50_95']:.4f} "
            f"P={cls['precision']:.4f} R={cls['recall']:.4f}"
        )

    confusion = metrics["confusion_matrix"]
    confusion_labels = class_names + ["background"]
    print(
        "Overall confusion matrix "
        "(rows=true class, cols=pred class, includes background for FN/FP):"
    )
    header = " " * 14 + " ".join(f"{name[:10]:>10}" for name in confusion_labels)
    print(header)
    for i, row in enumerate(confusion):
        row_text = " ".join(f"{int(v):>10d}" for v in row)
        print(f"{confusion_labels[i][:14]:>14} {row_text}")


if __name__ == "__main__":
    main()
