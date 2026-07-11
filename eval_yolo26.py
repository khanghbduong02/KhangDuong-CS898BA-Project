from __future__ import annotations

import argparse
import ast
import pickle
import warnings
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch
from torch.utils.data import DataLoader

from detection_metrics import compute_detection_metrics, xywhn_to_xyxy
from models.yolo26_torch import build_yolo26
from train_yolo26 import E2EDetectLoss, STRIDES, YoloDetectionDataset, collate_fn, compute_loss


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
    parser.add_argument("--scale", type=str, default="n", help="YOLO26 scale variant")
    parser.add_argument("--box-gain", type=float, default=7.5, help="Box IoU loss multiplier")
    parser.add_argument("--cls-gain", type=float, default=0.5, help="Classification BCE loss multiplier")
    parser.add_argument("--reg-gain", type=float, default=1.5, help="Regression term multiplier")
    parser.add_argument("--one2many-topk", type=int, default=10, help="Task-aligned top-k for one-to-many assignment")
    parser.add_argument("--one2one-topk", type=int, default=1, help="Task-aligned top-k for one-to-one assignment")
    parser.add_argument("--conf-thresh", type=float, default=0.25, help="Confidence threshold used for precision/recall summary")
    return parser.parse_args()


def _load_class_info_from_data_yaml(data_root: Path) -> Tuple[Optional[int], Optional[List[str]]]:
    data_yaml = data_root / "data.yaml"
    if not data_yaml.exists():
        return None, None

    nc: Optional[int] = None
    names: Optional[List[str]] = None

    for raw_line in data_yaml.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or ":" not in line:
            continue

        key, value = line.split(":", 1)
        key = key.strip()
        value = value.strip()

        if key == "nc":
            try:
                nc = int(value)
            except ValueError:
                nc = None

        if key == "names":
            try:
                parsed = ast.literal_eval(value)
                if isinstance(parsed, list):
                    names = [str(v) for v in parsed]
                elif isinstance(parsed, dict):
                    names = [str(parsed[i]) for i in sorted(parsed.keys())]
            except (ValueError, SyntaxError):
                names = None

    return nc, names


def main() -> None:
    args = parse_args()

    if not torch.cuda.is_available() and args.device.startswith("cuda"):
        raise RuntimeError("CUDA device requested but no GPU is available to PyTorch.")

    split_root = args.data_root / args.split
    if not split_root.exists():
        raise FileNotFoundError(f"Expected split folder at {split_root}")

    yaml_nc, yaml_names = _load_class_info_from_data_yaml(args.data_root)
    if yaml_nc is None:
        raise ValueError(f"Could not read 'nc' from {args.data_root / 'data.yaml'}")
    num_classes = yaml_nc

    if yaml_names is not None:
        class_names = yaml_names
    else:
        class_names = [f"class_{i}" for i in range(num_classes)]

    if len(class_names) != num_classes:
        raise ValueError(f"Class name count ({len(class_names)}) does not match num_classes ({num_classes})")

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

    model = build_yolo26(nc=num_classes, scale=args.scale).to(device)
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
        box_gain=args.box_gain,
        cls_gain=args.cls_gain,
        reg_gain=args.reg_gain,
        one2many_topk=args.one2many_topk,
        one2one_topk=args.one2one_topk,
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

            batch_preds = outputs["one_to_one"].detach().cpu()
            for idx, target in enumerate(targets):
                pred_boxes = batch_preds[idx, :, :4].float()
                pred_scores = batch_preds[idx, :, 4].float()
                pred_labels = batch_preds[idx, :, 5].long()

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