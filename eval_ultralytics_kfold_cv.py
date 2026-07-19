from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import torch
from ultralytics import YOLO

from cv_utils import (
    VALID_IMAGE_EXTENSIONS,
    confidence_tag,
    discover_folds,
    numeric_summary,
    project_path,
    validate_cv_layout,
)
from detection_metrics import compute_detection_metrics
from yolo_dataset_config import YoloDatasetConfig


def read_targets(
    label_path: Path,
    image_height: int,
    image_width: int,
    num_classes: int,
) -> dict[str, torch.Tensor]:
    boxes: list[list[float]] = []
    labels: list[int] = []
    for line_number, line in enumerate(label_path.read_text(encoding="utf-8").splitlines(), start=1):
        parts = line.split()
        if not parts:
            continue
        if len(parts) != 5:
            raise ValueError(f"{label_path}:{line_number}: expected exactly five YOLO fields")
        try:
            class_value, x_center, y_center, width, height = (float(value) for value in parts)
        except ValueError as exc:
            raise ValueError(f"{label_path}:{line_number}: label values must be numeric") from exc
        if not class_value.is_integer() or not 0 <= int(class_value) < num_classes:
            raise ValueError(f"{label_path}:{line_number}: class ID is outside 0..{num_classes - 1}")
        if not (0.0 <= x_center <= 1.0 and 0.0 <= y_center <= 1.0 and 0.0 < width <= 1.0 and 0.0 < height <= 1.0):
            raise ValueError(f"{label_path}:{line_number}: invalid normalized bounding box")
        boxes.append(
            [
                (x_center - width / 2.0) * image_width,
                (y_center - height / 2.0) * image_height,
                (x_center + width / 2.0) * image_width,
                (y_center + height / 2.0) * image_height,
            ]
        )
        labels.append(int(class_value))
    return {
        "boxes": torch.tensor(boxes, dtype=torch.float32).reshape(-1, 4),
        "labels": torch.tensor(labels, dtype=torch.long),
    }


def evaluate_fold(
    args: argparse.Namespace,
    fold: int,
    checkpoint: Path,
    dataset_config: YoloDatasetConfig,
) -> dict[str, Any]:
    fold_root = args.data_root / f"fold_{fold}"
    images_dir = fold_root / args.split / "images"
    labels_dir = fold_root / args.split / "labels"
    image_paths = sorted(
        path for path in images_dir.iterdir() if path.is_file() and path.suffix.lower() in VALID_IMAGE_EXTENSIONS
    )
    if not image_paths:
        raise RuntimeError(f"fold={fold}: no {args.split} images found in {images_dir}")

    model = YOLO(str(checkpoint))
    predictions: list[dict[str, torch.Tensor]] = []
    targets: list[dict[str, torch.Tensor]] = []
    for start in range(0, len(image_paths), args.batch_size):
        image_batch = image_paths[start : start + args.batch_size]
        results = model.predict(
            source=[str(path) for path in image_batch],
            imgsz=args.imgsz,
            batch=len(image_batch),
            device=args.device,
            conf=args.min_prediction_confidence,
            iou=args.nms_iou,
            max_det=args.max_det,
            save=False,
            verbose=False,
        )
        if len(results) != len(image_batch):
            raise RuntimeError(f"fold={fold}: prediction count does not match requested image count")
        for image_path, result in zip(image_batch, results):
            label_path = labels_dir / f"{image_path.stem}.txt"
            if not label_path.exists():
                raise FileNotFoundError(f"fold={fold}: missing label for {image_path}")
            original_height, original_width = result.orig_shape
            targets.append(read_targets(label_path, original_height, original_width, dataset_config.num_classes))
            if result.boxes is None or len(result.boxes) == 0:
                predictions.append(
                    {
                        "boxes": torch.zeros((0, 4), dtype=torch.float32),
                        "scores": torch.zeros((0,), dtype=torch.float32),
                        "labels": torch.zeros((0,), dtype=torch.long),
                    }
                )
            else:
                predictions.append(
                    {
                        "boxes": result.boxes.xyxy.detach().cpu().float(),
                        "scores": result.boxes.conf.detach().cpu().float(),
                        "labels": result.boxes.cls.detach().cpu().long(),
                    }
                )

    return {
        "fold": fold,
        "checkpoint": str(checkpoint),
        "data_root": str(fold_root),
        "image_count": len(image_paths),
        "confidence_threshold": args.conf_thresh,
        "minimum_prediction_confidence": args.min_prediction_confidence,
        "nms_iou": args.nms_iou,
        "max_det": args.max_det,
        "metrics": compute_detection_metrics(
            predictions=predictions,
            targets=targets,
            num_classes=dataset_config.num_classes,
            conf_thresh=args.conf_thresh,
        ),
    }


def required_float(payload: dict[str, Any], key: str, context: str) -> float:
    value = payload.get(key)
    if not isinstance(value, (int, float)):
        raise ValueError(f"Missing numeric {key!r} in {context}")
    return float(value)


def aggregate_metrics(per_fold: list[dict[str, Any]], dataset_config: YoloDatasetConfig) -> dict[str, Any]:
    overall: dict[str, dict[str, float | int | None]] = {}
    for output_name, metric_key in (
        ("mAP50", "map50"),
        ("mAP50_95", "map50_95"),
        ("precision_at_confidence_threshold", "precision"),
        ("recall_at_confidence_threshold", "recall"),
    ):
        overall[output_name] = numeric_summary(
            [required_float(result["metrics"], metric_key, f"fold {result['fold']}") for result in per_fold]
        )

    classes: list[dict[str, Any]] = []
    for class_id, class_name in enumerate(dataset_config.class_names):
        records: list[dict[str, Any]] = []
        for result in per_fold:
            candidates = [
                record
                for record in result["metrics"]["per_class"]
                if int(record.get("class_id", -1)) == class_id
            ]
            if len(candidates) != 1:
                raise ValueError(f"Fold {result['fold']} has no unique result for class {class_id}")
            records.append(candidates[0])
        classes.append(
            {
                "class_id": class_id,
                "class_name": class_name,
                "total_ground_truth_boxes": sum(int(required_float(record, "num_gt", "per-class metric")) for record in records),
                "AP50": numeric_summary([required_float(record, "ap50", "per-class metric") for record in records]),
                "AP50_95": numeric_summary([required_float(record, "ap50_95", "per-class metric") for record in records]),
                "precision_at_confidence_threshold": numeric_summary(
                    [required_float(record, "precision", "per-class metric") for record in records]
                ),
                "recall_at_confidence_threshold": numeric_summary(
                    [required_float(record, "recall", "per-class metric") for record in records]
                ),
            }
        )
    return {"overall": overall, "per_class": classes}


def print_summary(per_fold: list[dict[str, Any]], aggregate: dict[str, Any]) -> None:
    print("\nUltralytics per-fold validation metrics (project metric implementation):")
    print(f"{'Fold':>4} {'mAP50':>8} {'mAP50-95':>10} {'Precision':>10} {'Recall':>8}")
    for result in per_fold:
        metrics = result["metrics"]
        print(
            f"{int(result['fold']):>4} {float(metrics['map50']):>8.4f} "
            f"{float(metrics['map50_95']):>10.4f} {float(metrics['precision']):>10.4f} "
            f"{float(metrics['recall']):>8.4f}"
        )
    print("\nCross-fold overall mean +/- sample SD:")
    for metric_name, summary in aggregate["overall"].items():
        deviation = summary["sample_sd"]
        deviation_text = "n/a" if deviation is None else f"{float(deviation):.4f}"
        print(f"{metric_name}={float(summary['mean']):.4f} +/- {deviation_text} (n={summary['n']})")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate a pretrained Ultralytics detector across any completed fold_<n> YOLO dataset using "
            "the project's metric implementation after Ultralytics NMS."
        )
    )
    parser.add_argument("--data-root", type=Path, required=True, help="Root containing fold_<number> directories")
    parser.add_argument("--run-root", type=Path, required=True, help="Run root produced by run_ultralytics_kfold_cv.py")
    parser.add_argument(
        "--folds",
        type=int,
        nargs="*",
        default=None,
        help="Specific fold numbers; omit to discover all fold_<number> directories",
    )
    parser.add_argument("--split", choices=["valid", "test"], default="valid")
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--device", type=str, default="0")
    parser.add_argument("--conf-thresh", type=float, default=0.25)
    parser.add_argument("--min-prediction-confidence", type=float, default=0.001)
    parser.add_argument("--nms-iou", type=float, default=0.7)
    parser.add_argument("--max-det", type=int, default=300)
    parser.add_argument("--output", type=Path, default=None, help="Aggregate JSON path; defaults beneath --run-root")
    parser.add_argument("--dry-run", action="store_true", help="Validate folds and print actions without inference")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.data_root = project_path(args.data_root)
    args.run_root = project_path(args.run_root)
    if args.imgsz <= 0 or args.batch_size <= 0 or args.max_det <= 0:
        raise ValueError("--imgsz, --batch-size, and --max-det must be positive")
    if not 0.0 <= args.conf_thresh <= 1.0 or not 0.0 <= args.min_prediction_confidence <= 1.0:
        raise ValueError("Confidence values must be in [0, 1]")
    if not 0.0 < args.nms_iou <= 1.0:
        raise ValueError("--nms-iou must be in (0, 1]")
    if not args.data_root.is_dir():
        raise FileNotFoundError(f"Data root does not exist: {args.data_root}")

    folds = discover_folds(args.data_root, args.folds or None)
    required_splits = ("train", args.split)
    _, dataset_config = validate_cv_layout(args.data_root, folds, required_splits=required_splits)
    args.output = (
        project_path(args.output)
        if args.output is not None
        else args.run_root / f"cv_evaluation_{args.split}_project_metrics_conf_{confidence_tag(args.conf_thresh)}.json"
    )

    per_fold: list[dict[str, Any]] = []
    for position, fold in enumerate(folds, start=1):
        fold_root = args.data_root / f"fold_{fold}"
        checkpoint = args.run_root / f"fold_{fold}" / "weights" / "best.pt"
        print(f"\n===== ULTRALYTICS PRETRAINED EVALUATE FOLD {fold}/{len(folds)} =====", flush=True)
        print(f"data={fold_root} checkpoint={checkpoint}", flush=True)
        if args.dry_run:
            continue
        if not checkpoint.exists():
            raise FileNotFoundError(f"Fold {fold} pretrained checkpoint does not exist: {checkpoint}")
        per_fold.append(evaluate_fold(args, fold, checkpoint, dataset_config))

    if args.dry_run:
        print("Dry run complete; no pretrained inference was started.")
        return

    aggregate = aggregate_metrics(per_fold, dataset_config)
    output = {
        "purpose": "Generic pretrained Ultralytics detector K-fold evaluation",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "data_root": str(args.data_root),
        "run_root": str(args.run_root),
        "folds": folds,
        "split": args.split,
        "metric_protocol": {
            "metric_implementation": "project detection_metrics.py after Ultralytics NMS",
            "confidence_threshold": args.conf_thresh,
            "minimum_prediction_confidence": args.min_prediction_confidence,
            "nms_iou": args.nms_iou,
            "max_det": args.max_det,
        },
        "dataset": {
            "num_classes": dataset_config.num_classes,
            "class_names": list(dataset_config.class_names),
        },
        "per_fold": per_fold,
        "aggregate": aggregate,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2) + "\n", encoding="utf-8")
    print_summary(per_fold, aggregate)
    print(f"\nWrote pretrained aggregate metrics to {args.output}")


if __name__ == "__main__":
    main()