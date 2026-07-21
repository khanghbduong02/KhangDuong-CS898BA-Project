from __future__ import annotations

import argparse
import json
import pickle
import warnings
from pathlib import Path
from typing import Dict, List, Tuple

import torch
from torch.utils.data import DataLoader

from detection_metrics import compute_detection_metrics, xywhn_to_xyxy
from models.yolo26_torch import build_yolo26, class_aware_nms
from train_yolo26 import E2EDetectLoss, STRIDES, YoloDetectionDataset, collate_fn, compute_loss
from yolo_dataset_config import read_yolo_dataset_config


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate the custom YOLO26 model on valid or test split.")
    parser.add_argument("--checkpoint", type=Path, required=True, help="Path to a saved checkpoint")
    parser.add_argument("--data-root", type=Path, default=Path("processed-data/baseline"), help="Dataset variant root containing train/valid/test splits")
    parser.add_argument("--split", type=str, choices=["valid", "test"], default="valid", help="Dataset split to evaluate")
    parser.add_argument("--batch-size", type=int, default=8, help="Batch size")
    parser.add_argument("--imgsz", type=int, default=640, help="Square input image size")
    parser.add_argument("--workers", type=int, default=2, help="DataLoader worker count")
    parser.add_argument("--fraction", type=float, default=1.0, help="Subset fraction for quick evaluation")
    parser.add_argument("--device", type=str, default="cuda", help="Evaluation device, e.g. cuda or cuda:0")
    parser.add_argument("--scale", type=str, default=None, help="YOLO26 scale variant; defaults to the saved checkpoint value")
    parser.add_argument("--box-gain", type=float, default=None, help="Box IoU loss multiplier; defaults to the saved checkpoint value")
    parser.add_argument("--cls-gain", type=float, default=None, help="Classification BCE loss multiplier; defaults to the saved checkpoint value")
    parser.add_argument("--focal-gamma", type=float, default=None, help="Classification focal-loss gamma; defaults to the saved checkpoint value")
    parser.add_argument("--reg-gain", type=float, default=None, help="Regression term multiplier; defaults to the saved checkpoint value")
    parser.add_argument(
        "--reg-max",
        type=int,
        default=None,
        help="Box distance bins per side; defaults to the saved checkpoint value or 1 for legacy checkpoints",
    )
    parser.add_argument("--one2many-topk", type=int, default=None, help="Task-aligned top-k for one-to-many assignment; defaults to the saved checkpoint value")
    parser.add_argument("--one2one-topk", type=int, default=None, help="Task-aligned top-k for one-to-one assignment; defaults to the saved checkpoint value")
    parser.add_argument("--conf-thresh", type=float, default=0.25, help="Confidence threshold used for precision/recall summary")
    parser.add_argument(
        "--inference-branch",
        choices=["one2one", "one2many"],
        default=None,
        help=(
            "Raw detection branch used for inference. Defaults to one2many with class-aware NMS; "
            "one2one reproduces historical reports."
        ),
    )
    parser.add_argument(
        "--postprocess",
        choices=["legacy_topk", "class_aware_nms"],
        default="class_aware_nms",
        help=(
            "Inference postprocessor; class_aware_nms is the validated standard, while legacy_topk "
            "reproduces prior top-k-only reports"
        ),
    )
    parser.add_argument(
        "--nms-iou",
        type=float,
        default=0.70,
        help="Class-aware NMS IoU threshold when --postprocess=class_aware_nms",
    )
    parser.add_argument(
        "--nms-score-thresh",
        type=float,
        default=0.001,
        help="Minimum score retained before class-aware NMS for AP computation",
    )
    parser.add_argument(
        "--max-det",
        type=int,
        default=300,
        help="Maximum detections retained per image after postprocessing",
    )
    parser.add_argument(
        "--metrics-output",
        type=Path,
        default=None,
        help="Optional JSON path for machine-readable evaluation metrics",
    )
    return parser.parse_args()


def _resolve_evaluation_settings(
    checkpoint: Dict[str, object],
    args: argparse.Namespace,
) -> Tuple[str, float, float, float, float, int, int, int]:
    """Use explicit CLI settings first, then checkpoint settings, then legacy defaults."""
    saved_args = checkpoint.get("args", {})
    if not isinstance(saved_args, dict):
        saved_args = {}

    scale = args.scale if args.scale is not None else str(saved_args.get("scale", "n"))
    box_gain = args.box_gain if args.box_gain is not None else float(saved_args.get("box_gain", 7.5))
    cls_gain = args.cls_gain if args.cls_gain is not None else float(saved_args.get("cls_gain", 0.5))
    focal_gamma = args.focal_gamma if args.focal_gamma is not None else float(saved_args.get("focal_gamma", 0.0))
    reg_gain = args.reg_gain if args.reg_gain is not None else float(saved_args.get("reg_gain", 1.5))
    reg_max = args.reg_max if args.reg_max is not None else int(saved_args.get("reg_max", 1))
    one2many_topk = args.one2many_topk if args.one2many_topk is not None else int(saved_args.get("one2many_topk", 10))
    one2one_topk = args.one2one_topk if args.one2one_topk is not None else int(saved_args.get("one2one_topk", 1))
    return scale, box_gain, cls_gain, focal_gamma, reg_gain, one2many_topk, one2one_topk, reg_max


def _load_positive_class_weights(
    checkpoint: Dict[str, object],
    num_classes: int,
    device: torch.device,
) -> torch.Tensor:
    """Restore checkpoint positive-class BCE weights, or neutral weights for legacy checkpoints."""
    saved_weights = checkpoint.get("class_positive_weights")
    if saved_weights is None:
        return torch.ones(num_classes, device=device, dtype=torch.float32)

    weights = torch.as_tensor(saved_weights, device=device, dtype=torch.float32).reshape(-1)
    if weights.numel() != num_classes:
        raise ValueError(
            f"Checkpoint has {weights.numel()} positive class weights, expected {num_classes} from the dataset"
        )
    if not torch.isfinite(weights).all() or (weights <= 0).any():
        raise ValueError("Checkpoint positive class weights must be finite and greater than zero")
    return weights


def main() -> None:
    args = parse_args()

    if args.inference_branch is None:
        args.inference_branch = "one2one" if args.postprocess == "legacy_topk" else "one2many"

    if not torch.cuda.is_available() and args.device.startswith("cuda"):
        raise RuntimeError("CUDA device requested but no GPU is available to PyTorch.")
    if not 0.0 <= args.conf_thresh <= 1.0:
        raise ValueError("--conf-thresh must be in [0, 1]")
    if not 0.0 < args.nms_iou <= 1.0:
        raise ValueError("--nms-iou must be in (0, 1]")
    if not 0.0 <= args.nms_score_thresh <= 1.0:
        raise ValueError("--nms-score-thresh must be in [0, 1]")
    if args.max_det <= 0:
        raise ValueError("--max-det must be positive")
    if args.postprocess == "legacy_topk" and args.inference_branch != "one2one":
        raise ValueError("--postprocess legacy_topk is available only for the historical one2one branch")

    split_root = args.data_root / args.split
    if not split_root.exists():
        raise FileNotFoundError(f"Expected split folder at {split_root}")

    dataset_config = read_yolo_dataset_config(args.data_root)
    num_classes = dataset_config.num_classes
    class_names = list(dataset_config.class_names)

    device = torch.device(args.device)
    try:
        checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=True)
    except (TypeError, pickle.UnpicklingError):
        # Retry weights-only mode after allowlisting path classes used in checkpoint metadata.
        loaded = False
        try:
            if hasattr(torch.serialization, "add_safe_globals"):
                torch.serialization.add_safe_globals([Path, type(Path())])
            checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=True)
            loaded = True
        except Exception:
            loaded = False

        if not loaded:
            # Final fallback for trusted local checkpoints.
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", category=FutureWarning, module="torch.serialization")
                checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False)

    scale, box_gain, cls_gain, focal_gamma, reg_gain, one2many_topk, one2one_topk, reg_max = _resolve_evaluation_settings(checkpoint, args)
    if reg_max <= 0:
        raise ValueError("Resolved reg_max must be positive")
    class_positive_weights = _load_positive_class_weights(checkpoint, num_classes, device)

    model = build_yolo26(nc=num_classes, scale=scale, topk=args.max_det, reg_max=reg_max).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])

    dataset = YoloDetectionDataset(split_root, imgsz=args.imgsz, fraction=args.fraction)
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.workers,
        pin_memory=device.type == "cuda",
        collate_fn=collate_fn,
    )

    criterion = E2EDetectLoss(
        nc=num_classes,
        strides=STRIDES,
        device=device,
        box_gain=box_gain,
        cls_gain=cls_gain,
        reg_gain=reg_gain,
        one2many_topk=one2many_topk,
        one2one_topk=one2one_topk,
        reg_max=reg_max,
        class_positive_weights=class_positive_weights,
        focal_gamma=focal_gamma,
    )
    class_weight_summary = ", ".join(
        f"class_{class_id}:{weight:.3f}" for class_id, weight in enumerate(class_positive_weights.tolist())
    )
    print(
        f"Evaluation settings: scale={scale} box_gain={box_gain:g} cls_gain={cls_gain:g} focal_gamma={focal_gamma:g} "
        f"reg_gain={reg_gain:g} reg_max={reg_max} one2many_topk={one2many_topk} one2one_topk={one2one_topk} "
        f"positive_class_weights=({class_weight_summary}) postprocess={args.postprocess} "
        f"inference_branch={args.inference_branch} "
        f"nms_iou={args.nms_iou:g} nms_score_thresh={args.nms_score_thresh:g} max_det={args.max_det}"
    )
    model.eval()
    total_loss = 0.0
    total_cls = 0.0
    total_box = 0.0
    total_batches = 0
    predictions: List[Dict[str, torch.Tensor]] = []
    targets_list: List[Dict[str, torch.Tensor]] = []

    with torch.no_grad():
        for images, targets in loader:
            images = images.to(device, non_blocking=True)
            targets_device = [target.to(device) for target in targets]

            with torch.amp.autocast(device_type=device.type, enabled=device.type == "cuda"):
                outputs = model(images)
                loss_breakdown = compute_loss(
                    outputs,
                    targets_device,
                    nc=num_classes,
                    criterion=criterion,
                    device=device,
                )

            total_loss += float(loss_breakdown.total.detach().item())
            total_cls += float(loss_breakdown.cls_loss.item())
            total_box += float((loss_breakdown.box_loss + loss_breakdown.reg_loss).item())
            total_batches += 1

            if args.postprocess == "legacy_topk":
                batch_preds = [prediction.detach().cpu() for prediction in outputs["one_to_one"]]
            else:
                decoded = model.detect.decode_branch(outputs[args.inference_branch])
                batch_preds = [
                    prediction.detach().cpu()
                    for prediction in class_aware_nms(
                        decoded,
                        num_classes=num_classes,
                        score_threshold=args.nms_score_thresh,
                        iou_threshold=args.nms_iou,
                        max_detections=args.max_det,
                    )
                ]

            for prediction, target in zip(batch_preds, targets):
                pred_boxes = prediction[:, :4].float()
                pred_scores = prediction[:, 4].float()
                pred_labels = prediction[:, 5].long()

                gt_boxes = xywhn_to_xyxy(target[:, 1:5].float(), args.imgsz) if target.numel() else torch.zeros((0, 4), dtype=torch.float32)
                gt_labels = target[:, 0].long() if target.numel() else torch.zeros((0,), dtype=torch.long)

                predictions.append({"boxes": pred_boxes, "scores": pred_scores, "labels": pred_labels})
                targets_list.append({"boxes": gt_boxes, "labels": gt_labels})

    if total_batches == 0:
        raise RuntimeError("No batches were evaluated. Check dataset split and fraction arguments.")

    loss = total_loss / total_batches
    cls_loss = total_cls / total_batches
    box_loss = total_box / total_batches
    metrics = compute_detection_metrics(
        predictions,
        targets_list,
        num_classes=num_classes,
        conf_thresh=args.conf_thresh,
    )

    if args.metrics_output is not None:
        metrics_output = {
            "checkpoint": str(args.checkpoint),
            "checkpoint_epoch": int(checkpoint.get("epoch", 0)),
            "data_root": str(args.data_root),
            "split": args.split,
            "image_count": len(dataset),
            "confidence_threshold": args.conf_thresh,
            "loss": loss,
            "cls_loss": cls_loss,
            "box_loss": box_loss,
            "evaluation_settings": {
                "scale": scale,
                "box_gain": box_gain,
                "cls_gain": cls_gain,
                "focal_gamma": focal_gamma,
                "reg_gain": reg_gain,
                "reg_max": reg_max,
                "one2many_topk": one2many_topk,
                "one2one_topk": one2one_topk,
                "positive_class_weights": class_positive_weights.detach().cpu().tolist(),
                "class_names": class_names,
                "postprocess": args.postprocess,
                "inference_branch": args.inference_branch,
                "nms_iou": args.nms_iou,
                "nms_score_threshold": args.nms_score_thresh,
                "max_detections": args.max_det,
            },
            "metrics": metrics,
        }
        args.metrics_output.parent.mkdir(parents=True, exist_ok=True)
        args.metrics_output.write_text(json.dumps(metrics_output, indent=2) + "\n", encoding="utf-8")

    print(
        f"split={args.split} images={len(dataset)} "
        f"loss={loss:.4f} cls_loss={cls_loss:.4f} box_loss={box_loss:.4f} "
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
    print("Overall confusion matrix (rows=true class, cols=pred class, includes background for FN/FP):")
    header = " " * 14 + " ".join(f"{name[:10]:>10}" for name in confusion_labels)
    print(header)
    for i, row in enumerate(confusion):
        row_text = " ".join(f"{int(v):>10d}" for v in row)
        print(f"{confusion_labels[i][:14]:>14} {row_text}")


if __name__ == "__main__":
    main()