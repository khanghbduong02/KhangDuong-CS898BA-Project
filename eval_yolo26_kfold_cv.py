from __future__ import annotations

import argparse
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from cv_utils import (
    PROJECT_ROOT,
    confidence_tag,
    discover_folds,
    numeric_summary,
    project_path,
    render_fold_path,
    validate_cv_layout,
)
from yolo_dataset_config import YoloDatasetConfig


def build_eval_command(
    args: argparse.Namespace,
    checkpoint: Path,
    fold_root: Path,
    metrics_output: Path,
) -> list[str]:
    return [
        sys.executable,
        "eval_yolo26.py",
        "--checkpoint",
        str(checkpoint),
        "--data-root",
        str(fold_root),
        "--split",
        args.split,
        "--device",
        args.device,
        "--batch-size",
        str(args.batch_size),
        "--imgsz",
        str(args.imgsz),
        "--workers",
        str(args.workers),
        "--fraction",
        str(args.fraction),
        "--conf-thresh",
        str(args.conf_thresh),
        "--postprocess",
        args.postprocess,
        "--nms-iou",
        str(args.nms_iou),
        "--nms-score-thresh",
        str(args.nms_score_thresh),
        "--max-det",
        str(args.max_det),
        "--metrics-output",
        str(metrics_output),
    ]


def required_float(payload: dict[str, Any], key: str, context: str) -> float:
    value = payload.get(key)
    if not isinstance(value, (int, float)):
        raise ValueError(f"Missing numeric {key!r} in {context}")
    return float(value)


def aggregate_metrics(
    per_fold: list[dict[str, Any]],
    dataset_config: YoloDatasetConfig,
) -> dict[str, Any]:
    overall_paths = {
        "loss": ("loss",),
        "cls_loss": ("cls_loss",),
        "box_loss": ("box_loss",),
        "mAP50": ("metrics", "map50"),
        "mAP50_95": ("metrics", "map50_95"),
        "precision_at_confidence_threshold": ("metrics", "precision"),
        "recall_at_confidence_threshold": ("metrics", "recall"),
    }
    overall: dict[str, dict[str, float | int | None]] = {}
    for output_name, path in overall_paths.items():
        values: list[float] = []
        for result in per_fold:
            value: Any = result
            for key in path:
                if not isinstance(value, dict):
                    raise ValueError(f"Cannot read {output_name!r} from fold {result.get('fold')}")
                value = value.get(key)
            if not isinstance(value, (int, float)):
                raise ValueError(f"Metric {output_name!r} is not numeric in fold {result.get('fold')}")
            values.append(float(value))
        overall[output_name] = numeric_summary(values)

    per_class_summary: list[dict[str, Any]] = []
    for class_id, class_name in enumerate(dataset_config.class_names):
        class_results: list[dict[str, Any]] = []
        for result in per_fold:
            records = result.get("metrics", {}).get("per_class")
            if not isinstance(records, list):
                raise ValueError(f"Fold {result.get('fold')} has no per-class metrics")
            matches = [record for record in records if int(record.get("class_id", -1)) == class_id]
            if len(matches) != 1 or not isinstance(matches[0], dict):
                raise ValueError(f"Fold {result.get('fold')} has no unique metric record for class {class_id}")
            class_results.append(matches[0])

        per_class_summary.append(
            {
                "class_id": class_id,
                "class_name": class_name,
                "total_ground_truth_boxes": sum(int(required_float(record, "num_gt", "per-class metric")) for record in class_results),
                "AP50": numeric_summary([required_float(record, "ap50", "per-class metric") for record in class_results]),
                "AP50_95": numeric_summary([required_float(record, "ap50_95", "per-class metric") for record in class_results]),
                "precision_at_confidence_threshold": numeric_summary(
                    [required_float(record, "precision", "per-class metric") for record in class_results]
                ),
                "recall_at_confidence_threshold": numeric_summary(
                    [required_float(record, "recall", "per-class metric") for record in class_results]
                ),
            }
        )
    return {"overall": overall, "per_class": per_class_summary}


def print_summary(per_fold: list[dict[str, Any]], aggregate: dict[str, Any]) -> None:
    print("\nPer-fold validation metrics:")
    print(f"{'Fold':>4} {'Epoch':>5} {'mAP50':>8} {'mAP50-95':>10} {'Precision':>10} {'Recall':>8}")
    for result in per_fold:
        metrics = result["metrics"]
        print(
            f"{int(result['fold']):>4} {int(result['checkpoint_epoch']):>5} "
            f"{float(metrics['map50']):>8.4f} {float(metrics['map50_95']):>10.4f} "
            f"{float(metrics['precision']):>10.4f} {float(metrics['recall']):>8.4f}"
        )

    print("\nCross-fold overall mean +/- sample SD:")
    for metric_name, summary in aggregate["overall"].items():
        deviation = summary["sample_sd"]
        deviation_text = "n/a" if deviation is None else f"{float(deviation):.4f}"
        print(f"{metric_name}={float(summary['mean']):.4f} +/- {deviation_text} (n={summary['n']})")

    print("\nPer-class mean AP50 / recall:")
    for class_summary in aggregate["per_class"]:
        ap = class_summary["AP50"]
        recall = class_summary["recall_at_confidence_threshold"]
        ap_sd = "n/a" if ap["sample_sd"] is None else f"{float(ap['sample_sd']):.4f}"
        recall_sd = "n/a" if recall["sample_sd"] is None else f"{float(recall['sample_sd']):.4f}"
        print(
            f"class={class_summary['class_name']} gt_total={class_summary['total_ground_truth_boxes']} "
            f"AP50={float(ap['mean']):.4f} +/- {ap_sd} "
            f"recall={float(recall['mean']):.4f} +/- {recall_sd}"
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate any completed custom YOLO26 K-fold study and aggregate overall/per-class metrics. "
            "The evaluator runs folds sequentially."
        )
    )
    parser.add_argument("--data-root", type=Path, required=True, help="Root containing fold_<number> directories")
    parser.add_argument("--run-root", type=Path, required=True, help="Run root produced by run_yolo26_kfold_cv.py")
    parser.add_argument(
        "--checkpoint-template",
        type=str,
        default="fold_{fold}/best.pt",
        help="Checkpoint path beneath --run-root; must include {fold}",
    )
    parser.add_argument(
        "--folds",
        type=int,
        nargs="*",
        default=None,
        help="Specific fold numbers; omit to discover all fold_<number> directories",
    )
    parser.add_argument("--split", choices=["valid", "test"], default="valid")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--fraction", type=float, default=1.0)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--conf-thresh", type=float, default=0.25)
    parser.add_argument(
        "--postprocess",
        choices=["legacy_topk", "class_aware_nms"],
        default="class_aware_nms",
        help="Inference postprocessor passed to eval_yolo26.py; legacy_topk reproduces prior reports",
    )
    parser.add_argument("--nms-iou", type=float, default=0.70)
    parser.add_argument("--nms-score-thresh", type=float, default=0.001)
    parser.add_argument("--max-det", type=int, default=300)
    parser.add_argument("--output", type=Path, default=None, help="Aggregate JSON path; defaults beneath --run-root")
    parser.add_argument("--dry-run", action="store_true", help="Validate inputs and print commands without evaluation")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.data_root = project_path(args.data_root)
    args.run_root = project_path(args.run_root)
    if args.batch_size <= 0 or args.imgsz <= 0 or args.workers < 0 or not 0.0 < args.fraction <= 1.0:
        raise ValueError("Invalid batch-size, image size, worker count, or fraction")
    if not 0.0 <= args.conf_thresh <= 1.0:
        raise ValueError("--conf-thresh must be in [0, 1]")
    if not 0.0 < args.nms_iou <= 1.0:
        raise ValueError("--nms-iou must be in (0, 1]")
    if not 0.0 <= args.nms_score_thresh <= 1.0:
        raise ValueError("--nms-score-thresh must be in [0, 1]")
    if args.max_det <= 0:
        raise ValueError("--max-det must be positive")
    if not args.data_root.is_dir():
        raise FileNotFoundError(f"Data root does not exist: {args.data_root}")

    folds = discover_folds(args.data_root, args.folds or None)
    required_splits = ("train", args.split) if args.split != "train" else ("train",)
    fold_roots, dataset_config = validate_cv_layout(args.data_root, folds, required_splits=required_splits)
    args.output = (
        project_path(args.output)
        if args.output is not None
        else args.run_root
        / f"cv_evaluation_{args.split}_{args.postprocess}_conf_{confidence_tag(args.conf_thresh)}.json"
    )

    per_fold: list[dict[str, Any]] = []
    for position, fold in enumerate(folds, start=1):
        fold_root = fold_roots[fold]
        checkpoint = render_fold_path(args.run_root, args.checkpoint_template, fold)
        metrics_output = checkpoint.parent / (
            f"{args.split}_metrics_{args.postprocess}_conf_{confidence_tag(args.conf_thresh)}.json"
        )
        command = build_eval_command(args, checkpoint, fold_root, metrics_output)
        print(f"\n===== CUSTOM YOLO26 EVALUATE FOLD {fold}/{len(folds)} =====", flush=True)
        print(subprocess.list2cmdline(command), flush=True)
        if args.dry_run:
            continue
        if not checkpoint.exists():
            raise FileNotFoundError(f"Fold {fold} checkpoint does not exist: {checkpoint}")

        subprocess.run(command, cwd=PROJECT_ROOT, check=True)
        if not metrics_output.exists():
            raise RuntimeError(f"Fold {fold} evaluator did not write {metrics_output}")
        result = json.loads(metrics_output.read_text(encoding="utf-8"))
        if tuple(result.get("evaluation_settings", {}).get("class_names", [])) != dataset_config.class_names:
            raise ValueError(f"Fold {fold} evaluator class names do not match the dataset configuration")
        result["fold"] = fold
        result["metrics_output"] = str(metrics_output)
        per_fold.append(result)

    if args.dry_run:
        print("Dry run complete; no evaluation was started.")
        return

    aggregate = aggregate_metrics(per_fold, dataset_config)
    summary = {
        "purpose": "Generic custom YOLO26 K-fold cross-validation evaluation",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "data_root": str(args.data_root),
        "run_root": str(args.run_root),
        "folds": folds,
        "split": args.split,
        "confidence_threshold": args.conf_thresh,
        "postprocessing": {
            "mode": args.postprocess,
            "nms_iou": args.nms_iou,
            "nms_score_threshold": args.nms_score_thresh,
            "max_detections": args.max_det,
        },
        "dataset": {
            "num_classes": dataset_config.num_classes,
            "class_names": list(dataset_config.class_names),
        },
        "per_fold": per_fold,
        "aggregate": aggregate,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print_summary(per_fold, aggregate)
    print(f"\nWrote aggregate cross-validation summary to {args.output}")


if __name__ == "__main__":
    main()