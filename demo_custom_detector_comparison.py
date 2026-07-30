"""Render ground truth, local custom YOLO26, and local Faster R-CNN in one row.

This qualitative tool runs frozen local checkpoints on user-selected images. It
creates one three-panel grid per source image and a JSON summary; it does not
compute a new metric or select a model configuration.
"""
from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence

import cv2
import numpy as np
import torch

from demo_custom_yolo26 import (
    GROUND_TRUTH_COLOR_BGR,
    PREDICTION_COLORS_BGR,
    _checkpoint_imgsz,
    _checkpoint_model_settings,
    _draw_label,
    _load_checkpoint as load_yolo_checkpoint,
    _read_ground_truth,
    _resize_for_model,
    _scale_detections_to_source,
    discover_images,
    serialise_detections,
)
from eval_faster_rcnn import (
    _load_checkpoint as load_faster_rcnn_checkpoint,
    _load_model_state,
    _load_positive_class_weights,
    _resolve_evaluation_settings,
)
from models.faster_rcnn import build_faster_rcnn
from models.yolo26_torch import build_yolo26, class_aware_nms
from yolo_dataset_config import read_yolo_dataset_config


HEADER_HEIGHT = 34
TITLE_HEIGHT = 30
PANEL_BACKGROUND_BGR = (20, 20, 20)
HEADER_TEXT_COLOR_BGR = (255, 255, 255)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run frozen local custom YOLO26 and Faster R-CNN checkpoints on the same images, "
            "saving Ground Truth | YOLO26 | Faster R-CNN grids."
        )
    )
    parser.add_argument("--yolo-checkpoint", type=Path, required=True, help="Selected local custom YOLO26 best.pt")
    parser.add_argument(
        "--faster-rcnn-checkpoint",
        type=Path,
        required=True,
        help="Selected local custom Faster R-CNN best.pt",
    )
    parser.add_argument("--data-root", type=Path, required=True, help="Common fold root containing data.yaml")
    parser.add_argument("--source", type=Path, required=True, help="One source image or a directory of source images")
    parser.add_argument("--labels-dir", type=Path, required=True, help="YOLO labels directory for the ground-truth panel")
    parser.add_argument("--output-dir", type=Path, required=True, help="Directory for row grids and comparison JSON")
    parser.add_argument(
        "--title",
        type=str,
        default="QUALITATIVE COMPARISON - NOT A METRIC EVALUATION",
        help="Visible title above every grid; use an ASCII title because OpenCV's Hershey font is ASCII-only",
    )
    parser.add_argument(
        "--yolo-imgsz",
        type=int,
        default=None,
        help="Square YOLO26 inference size; defaults to the YOLO checkpoint training value",
    )
    parser.add_argument(
        "--faster-rcnn-imgsz",
        type=int,
        default=None,
        help="Square Faster R-CNN inference size; defaults to the Faster R-CNN checkpoint training value",
    )
    parser.add_argument("--batch-size", type=int, default=2, help="Images processed per batch by both local models")
    parser.add_argument("--conf", type=float, default=0.25, help="Display confidence floor after each model's NMS")
    parser.add_argument("--nms-iou", type=float, default=0.70, help="Class-aware/per-class NMS IoU threshold")
    parser.add_argument(
        "--nms-score-thresh",
        type=float,
        default=0.001,
        help="Candidate score floor used before each model's NMS",
    )
    parser.add_argument("--max-det", type=int, default=300, help="Maximum detections retained per image")
    parser.add_argument("--device", type=str, default="cuda", help="Inference device, such as cuda or cuda:0")
    parser.add_argument(
        "--yolo-inference-branch",
        choices=("one2many", "one2one"),
        default="one2many",
        help="YOLO26 raw branch to decode; one2many is the selected local configuration",
    )
    parser.add_argument(
        "--max-images",
        type=int,
        default=4,
        help="Maximum sorted images to process; use 0 to process every selected image",
    )
    parser.add_argument(
        "--image-names",
        nargs="*",
        default=None,
        help="Optional exact image filenames under --source for a curated presentation batch",
    )
    parser.add_argument(
        "--image-selection",
        choices=("alphabetical", "ground_truth_coverage"),
        default="alphabetical",
        help=(
            "Image selection when --source is a directory and --image-names is omitted. "
            "ground_truth_coverage selects images only from labels until every available class is covered, "
            "then fills remaining slots alphabetically."
        ),
    )
    return parser.parse_args()


def _validate_checkpoint_taxonomy(checkpoint: dict[str, Any], class_names: tuple[str, ...]) -> None:
    saved_names = checkpoint.get("class_names")
    if saved_names is not None and tuple(str(name) for name in saved_names) != class_names:
        raise ValueError("Checkpoint class names do not match the selected dataset taxonomy")
    saved_num_classes = checkpoint.get("num_classes")
    if saved_num_classes is not None and int(saved_num_classes) != len(class_names):
        raise ValueError("Checkpoint class count does not match the selected dataset taxonomy")


def _build_faster_rcnn(
    checkpoint: dict[str, Any],
    num_classes: int,
    device: torch.device,
    imgsz_override: int | None,
    nms_score_thresh: float,
    nms_iou: float,
    max_det: int,
) -> tuple[torch.nn.Module, int, dict[str, Any]]:
    settings_args = argparse.Namespace(imgsz=imgsz_override, scale=None, use_p2=None)
    scale, checkpoint_imgsz, use_p2 = _resolve_evaluation_settings(checkpoint, settings_args)
    class_positive_weights = _load_positive_class_weights(checkpoint, num_classes)
    saved_args = checkpoint.get("args", {})
    if not isinstance(saved_args, dict):
        saved_args = {}
    backbone_weights = str(saved_args.get("backbone_weights", "none"))
    model = build_faster_rcnn(
        nc=num_classes,
        scale=scale,
        min_size=checkpoint_imgsz,
        max_size=checkpoint_imgsz,
        class_positive_weights=class_positive_weights,
        score_threshold=nms_score_thresh,
        nms_threshold=nms_iou,
        max_detections=max_det,
        # The saved checkpoint supplies every backbone tensor, so loading it
        # never needs to download or initialize external pretrained weights.
        backbone_weights="none",
        use_p2=use_p2,
    ).to(device)
    _load_model_state(model, checkpoint)
    model.eval()
    return model, checkpoint_imgsz, {
        "scale": scale,
        "imgsz": checkpoint_imgsz,
        "use_p2": use_p2,
        "backbone_weights": backbone_weights,
        "backbone_initialization": str(checkpoint.get("backbone_initialization", "random")),
        "checkpoint_weight_source": str(checkpoint.get("checkpoint_weight_source", "raw")),
    }


def _to_zero_indexed_detection_tensor(prediction: dict[str, torch.Tensor]) -> torch.Tensor:
    boxes = prediction["boxes"]
    scores = prediction["scores"].reshape(-1, 1)
    labels = (prediction["labels"].long() - 1).clamp(min=0).to(dtype=boxes.dtype).reshape(-1, 1)
    if boxes.numel() == 0:
        return boxes.new_zeros((0, 6))
    return torch.cat((boxes, scores, labels), dim=1)


def _filter_display_detections(detections: torch.Tensor, confidence_threshold: float) -> torch.Tensor:
    return detections[detections[:, 4] >= confidence_threshold].detach().cpu()


def _add_header(panel: np.ndarray, title: str) -> np.ndarray:
    height, width = panel.shape[:2]
    grid_panel = np.full((height + HEADER_HEIGHT, width, 3), PANEL_BACKGROUND_BGR, dtype=np.uint8)
    grid_panel[HEADER_HEIGHT:, :, :] = panel
    cv2.putText(
        grid_panel,
        title,
        (10, 23),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.62,
        HEADER_TEXT_COLOR_BGR,
        2,
        cv2.LINE_AA,
    )
    return grid_panel


def _draw_ground_truth_panel(image_bgr: np.ndarray, ground_truth: Sequence[dict[str, Any]]) -> np.ndarray:
    panel = image_bgr.copy()
    for item in ground_truth:
        x1, y1, x2, y2 = (int(round(value)) for value in item["xyxy"])
        cv2.rectangle(panel, (x1, y1), (x2, y2), GROUND_TRUTH_COLOR_BGR, thickness=2, lineType=cv2.LINE_AA)
        _draw_label(panel, f"GT {item['class_name']}", (x1, y1), GROUND_TRUTH_COLOR_BGR)
    return _add_header(panel, "GROUND TRUTH")


def _draw_prediction_panel(
    image_bgr: np.ndarray,
    detections: torch.Tensor,
    class_names: Sequence[str],
    title: str,
) -> np.ndarray:
    panel = image_bgr.copy()
    for detection in detections:
        x1, y1, x2, y2, score, class_value = detection.tolist()
        class_id = int(class_value)
        color = PREDICTION_COLORS_BGR[class_id % len(PREDICTION_COLORS_BGR)]
        p1 = (int(round(x1)), int(round(y1)))
        p2 = (int(round(x2)), int(round(y2)))
        cv2.rectangle(panel, p1, p2, color, thickness=2, lineType=cv2.LINE_AA)
        _draw_label(panel, f"P {class_names[class_id]} {score:.2f}", p1, color)
    return _add_header(panel, f"{title} ({detections.shape[0]} predictions)")


def compose_comparison_grid(
    image_bgr: np.ndarray,
    ground_truth: Sequence[dict[str, Any]],
    yolo_detections: torch.Tensor,
    faster_rcnn_detections: torch.Tensor,
    class_names: Sequence[str],
    title: str = "QUALITATIVE COMPARISON - NOT A METRIC EVALUATION",
) -> np.ndarray:
    """Create one GT | YOLO26 | Faster R-CNN row for a source image."""
    panels = (
        _draw_ground_truth_panel(image_bgr, ground_truth),
        _draw_prediction_panel(image_bgr, yolo_detections, class_names, "CUSTOM YOLO26"),
        _draw_prediction_panel(image_bgr, faster_rcnn_detections, class_names, "CUSTOM FASTER R-CNN"),
    )
    comparison = cv2.hconcat(panels)
    title_strip = np.full((TITLE_HEIGHT, comparison.shape[1], 3), PANEL_BACKGROUND_BGR, dtype=np.uint8)
    cv2.putText(
        title_strip,
        title,
        (10, 21),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        HEADER_TEXT_COLOR_BGR,
        1,
        cv2.LINE_AA,
    )
    return cv2.vconcat((title_strip, comparison))


def _checkpoint_weight_source(checkpoint: dict[str, Any]) -> str:
    source = str(checkpoint.get("checkpoint_weight_source", "raw"))
    if source not in {"raw", "ema"}:
        raise ValueError(f"Unsupported checkpoint weight source: {source!r}")
    return source


def _validate_scratch_faster_rcnn_checkpoint(checkpoint: dict[str, Any]) -> None:
    """Reject FRCNN checkpoints that would invalidate the scratch-only grid."""
    saved_args = checkpoint.get("args", {})
    if not isinstance(saved_args, dict):
        saved_args = {}
    backbone_weights = str(saved_args.get("backbone_weights", "none")).lower()
    backbone_initialization = str(checkpoint.get("backbone_initialization", "random")).lower()
    if backbone_weights != "none" or backbone_initialization != "random":
        raise ValueError(
            "The comparison grid accepts only a randomly initialized local Faster R-CNN "
            "checkpoint (backbone_weights=none and backbone_initialization=random)."
        )


def _ground_truth_class_ids(label_path: Path, num_classes: int) -> set[int]:
    """Return strict YOLO class IDs without using model predictions."""
    if not label_path.exists():
        return set()

    class_ids: set[int] = set()
    for line_number, line in enumerate(label_path.read_text(encoding="utf-8").splitlines(), start=1):
        parts = line.strip().split()
        if not parts:
            continue
        if len(parts) != 5:
            raise ValueError(f"{label_path}:{line_number}: expected five-field YOLO label")
        try:
            class_value = float(parts[0])
        except ValueError as exc:
            raise ValueError(f"{label_path}:{line_number}: class ID must be numeric") from exc
        class_id = int(class_value)
        if class_value != class_id or not 0 <= class_id < num_classes:
            raise ValueError(f"{label_path}:{line_number}: class ID is outside the dataset taxonomy")
        class_ids.add(class_id)
    return class_ids


def select_comparison_images(
    source: Path,
    labels_dir: Path,
    max_images: int,
    image_names: Sequence[str] | None,
    image_selection: str,
    num_classes: int,
) -> list[Path]:
    """Choose presentation images without inspecting predictions.

    ``ground_truth_coverage`` is intentionally label-only: it prevents an
    alphabetical prefix dominated by one class from being presented as though
    it were a balanced qualitative comparison.
    """
    if image_names is not None and image_selection != "alphabetical":
        raise ValueError("--image-selection cannot be combined with --image-names")

    image_paths = discover_images(source, 0, image_names)
    if image_selection == "alphabetical" or source.is_file():
        return image_paths if max_images == 0 else image_paths[:max_images]

    selected: list[Path] = []
    remaining: list[Path] = []
    covered_class_ids: set[int] = set()
    for image_path in image_paths:
        class_ids = _ground_truth_class_ids(labels_dir / f"{image_path.stem}.txt", num_classes)
        if class_ids - covered_class_ids:
            selected.append(image_path)
            covered_class_ids.update(class_ids)
        else:
            remaining.append(image_path)

    ordered = selected + remaining
    return ordered if max_images == 0 else ordered[:max_images]


def main() -> None:
    args = parse_args()
    if not torch.cuda.is_available() and args.device.startswith("cuda"):
        raise RuntimeError("CUDA device requested but no GPU is available to PyTorch")
    if args.batch_size <= 0 or args.max_det <= 0:
        raise ValueError("--batch-size and --max-det must be positive")
    if args.max_images < 0:
        raise ValueError("--max-images must be zero or positive")
    if not 0.0 <= args.conf <= 1.0 or not 0.0 <= args.nms_score_thresh <= 1.0:
        raise ValueError("--conf and --nms-score-thresh must be in [0, 1]")
    if not 0.0 < args.nms_iou <= 1.0:
        raise ValueError("--nms-iou must be in (0, 1]")

    args.yolo_checkpoint = args.yolo_checkpoint.resolve()
    args.faster_rcnn_checkpoint = args.faster_rcnn_checkpoint.resolve()
    args.data_root = args.data_root.resolve()
    args.source = args.source.resolve()
    args.labels_dir = args.labels_dir.resolve()
    for path, name in (
        (args.yolo_checkpoint, "--yolo-checkpoint"),
        (args.faster_rcnn_checkpoint, "--faster-rcnn-checkpoint"),
    ):
        if not path.is_file():
            raise FileNotFoundError(f"{name} does not exist: {path}")
    if not args.labels_dir.is_dir():
        raise FileNotFoundError(f"--labels-dir does not exist: {args.labels_dir}")

    device = torch.device(args.device)
    dataset_config = read_yolo_dataset_config(args.data_root)
    class_names = dataset_config.class_names
    yolo_checkpoint = load_yolo_checkpoint(args.yolo_checkpoint, device)
    faster_rcnn_checkpoint = load_faster_rcnn_checkpoint(args.faster_rcnn_checkpoint, device)
    _validate_checkpoint_taxonomy(yolo_checkpoint, class_names)
    _validate_checkpoint_taxonomy(faster_rcnn_checkpoint, class_names)
    yolo_weight_source = _checkpoint_weight_source(yolo_checkpoint)
    faster_rcnn_weight_source = _checkpoint_weight_source(faster_rcnn_checkpoint)
    if yolo_weight_source != "raw" or faster_rcnn_weight_source != "raw":
        raise ValueError("The comparison grid accepts only selected raw local checkpoint weights, not EMA weights")
    _validate_scratch_faster_rcnn_checkpoint(faster_rcnn_checkpoint)

    yolo_scale, yolo_reg_max, yolo_use_p2 = _checkpoint_model_settings(yolo_checkpoint)
    yolo_imgsz = _checkpoint_imgsz(yolo_checkpoint, args.yolo_imgsz)
    yolo_model = build_yolo26(
        nc=dataset_config.num_classes,
        scale=yolo_scale,
        topk=args.max_det,
        reg_max=yolo_reg_max,
        use_p2=yolo_use_p2,
    ).to(device)
    yolo_state = yolo_checkpoint.get("model_state_dict")
    if not isinstance(yolo_state, dict):
        raise ValueError("YOLO checkpoint does not contain a model_state_dict")
    yolo_model.load_state_dict(yolo_state)
    yolo_model.eval()

    faster_rcnn_model, faster_rcnn_imgsz, faster_rcnn_settings = _build_faster_rcnn(
        faster_rcnn_checkpoint,
        dataset_config.num_classes,
        device,
        args.faster_rcnn_imgsz,
        args.nms_score_thresh,
        args.nms_iou,
        args.max_det,
    )
    image_paths = select_comparison_images(
        args.source,
        args.labels_dir,
        args.max_images,
        args.image_names,
        args.image_selection,
        dataset_config.num_classes,
    )
    loaded_images: list[tuple[Path, np.ndarray, torch.Tensor, torch.Tensor]] = []
    for image_path in image_paths:
        image_bgr = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
        if image_bgr is None:
            raise FileNotFoundError(f"Unable to read image: {image_path}")
        loaded_images.append(
            (
                image_path,
                image_bgr,
                _resize_for_model(image_bgr, yolo_imgsz),
                _resize_for_model(image_bgr, faster_rcnn_imgsz),
            )
        )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    summary_images: list[dict[str, Any]] = []
    use_yolo_amp = device.type == "cuda"

    for start in range(0, len(loaded_images), args.batch_size):
        batch = loaded_images[start : start + args.batch_size]
        yolo_inputs = torch.stack([item[2] for item in batch], dim=0).to(device, non_blocking=True)
        with torch.no_grad(), torch.amp.autocast(device_type=device.type, enabled=use_yolo_amp):
            yolo_outputs = yolo_model(yolo_inputs)
            yolo_decoded = yolo_model.detect.decode_branch(yolo_outputs[args.yolo_inference_branch])
        yolo_batch = class_aware_nms(
            yolo_decoded,
            num_classes=dataset_config.num_classes,
            score_threshold=args.nms_score_thresh,
            iou_threshold=args.nms_iou,
            max_detections=args.max_det,
        )

        faster_rcnn_inputs = [item[3].to(device, non_blocking=True) for item in batch]
        # Match eval_faster_rcnn.py exactly. Faster R-CNN score calibration can
        # sit near the display threshold, so its qualitative artifacts remain
        # float32 rather than using YOLO's CUDA autocast optimization.
        with torch.no_grad():
            faster_rcnn_batch = faster_rcnn_model(faster_rcnn_inputs)
        if not isinstance(faster_rcnn_batch, list):
            raise RuntimeError("Faster R-CNN inference did not return a prediction list")
        if len(yolo_batch) != len(batch) or len(faster_rcnn_batch) != len(batch):
            raise RuntimeError("A local detector returned a different number of predictions than source images")

        for offset, ((image_path, image_bgr, _, _), yolo_detections, faster_rcnn_prediction) in enumerate(
            zip(batch, yolo_batch, faster_rcnn_batch),
            start=start + 1,
        ):
            image_height, image_width = image_bgr.shape[:2]
            ground_truth = _read_ground_truth(
                args.labels_dir / f"{image_path.stem}.txt",
                class_names,
                image_width,
                image_height,
            )
            scaled_yolo = _filter_display_detections(
                _scale_detections_to_source(yolo_detections, image_width, image_height, yolo_imgsz),
                args.conf,
            )
            faster_rcnn_detections = _to_zero_indexed_detection_tensor(faster_rcnn_prediction)
            scaled_faster_rcnn = _filter_display_detections(
                _scale_detections_to_source(
                    faster_rcnn_detections,
                    image_width,
                    image_height,
                    faster_rcnn_imgsz,
                ),
                args.conf,
            )
            grid = compose_comparison_grid(
                image_bgr,
                ground_truth,
                scaled_yolo,
                scaled_faster_rcnn,
                class_names,
                args.title,
            )
            output_path = args.output_dir / f"{offset:02d}_{image_path.stem}_gt_yolo26_faster_rcnn.jpg"
            if not cv2.imwrite(str(output_path), grid):
                raise RuntimeError(f"Unable to write comparison grid: {output_path}")
            image_summary = {
                "source_image": str(image_path),
                "comparison_grid": str(output_path),
                "original_shape": [image_height, image_width],
                "ground_truth_count": len(ground_truth),
                "ground_truth": ground_truth,
                "yolo26_detection_count": int(scaled_yolo.shape[0]),
                "yolo26_detections": serialise_detections(scaled_yolo, class_names),
                "faster_rcnn_detection_count": int(scaled_faster_rcnn.shape[0]),
                "faster_rcnn_detections": serialise_detections(scaled_faster_rcnn, class_names),
            }
            summary_images.append(image_summary)
            print(
                f"{image_path.name}: ground_truth={len(ground_truth)} yolo26={scaled_yolo.shape[0]} "
                f"faster_rcnn={scaled_faster_rcnn.shape[0]} output={output_path.name}"
            )

    summary = {
        "purpose": "Local custom detector comparison demonstration; not a cross-validation metric report",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "data_root": str(args.data_root),
        "source": str(args.source),
        "labels_dir": str(args.labels_dir),
        "class_names": list(class_names),
        "yolo26": {
            "checkpoint": str(args.yolo_checkpoint),
            "checkpoint_epoch": int(yolo_checkpoint.get("epoch", 0)),
            "checkpoint_weight_source": yolo_weight_source,
            "imgsz": yolo_imgsz,
            "scale": yolo_scale,
            "reg_max": yolo_reg_max,
            "use_p2": yolo_use_p2,
            "inference_branch": args.yolo_inference_branch,
        },
        "faster_rcnn": {
            "checkpoint": str(args.faster_rcnn_checkpoint),
            "checkpoint_epoch": int(faster_rcnn_checkpoint.get("epoch", 0)),
            "checkpoint_weight_source": faster_rcnn_weight_source,
            **faster_rcnn_settings,
        },
        "comparison_settings": {
            "batch_size": args.batch_size,
            "display_confidence_threshold": args.conf,
            "nms_score_threshold": args.nms_score_thresh,
            "nms_iou": args.nms_iou,
            "max_detections": args.max_det,
            "device": args.device,
            "faster_rcnn_inference_precision": "float32 (matches eval_faster_rcnn.py)",
            "panel_order": ["ground_truth", "custom_yolo26", "custom_faster_rcnn"],
            "title": args.title,
            "image_selection": args.image_selection,
            "image_selection_uses_model_predictions": False,
        },
        "image_count": len(summary_images),
        "images": summary_images,
    }
    summary_path = args.output_dir / "comparison_predictions.json"
    summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote {len(summary_images)} comparison grids and {summary_path}")


if __name__ == "__main__":
    main()
