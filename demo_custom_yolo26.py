"""Create annotated batch-inference outputs from the local custom YOLO26 model.

This tool is intentionally separate from cross-validation evaluation. It uses a
frozen local custom checkpoint to create visual demonstration artifacts from
user-selected images and can optionally overlay matching ground-truth labels.
It never computes a new metric or uses a candidate public test split by default.
"""
from __future__ import annotations

import argparse
import json
import pickle
import warnings
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence

import cv2
import torch

from cv_utils import VALID_IMAGE_EXTENSIONS
from models.yolo26_torch import build_yolo26, class_aware_nms
from yolo_dataset_config import read_yolo_dataset_config


PREDICTION_COLORS_BGR = (
    (0, 215, 255),
    (255, 128, 0),
    (0, 200, 0),
    (255, 0, 255),
    (0, 64, 255),
)
GROUND_TRUTH_COLOR_BGR = (255, 255, 255)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run a frozen local custom YOLO26 checkpoint on images and save annotated batch-demo outputs."
    )
    parser.add_argument("--checkpoint", type=Path, required=True, help="Local custom YOLO26 best.pt checkpoint")
    parser.add_argument("--data-root", type=Path, required=True, help="Dataset root containing data.yaml")
    parser.add_argument("--source", type=Path, required=True, help="One image or a directory of images")
    parser.add_argument(
        "--labels-dir",
        type=Path,
        default=None,
        help="Optional YOLO labels directory to overlay ground truth in white",
    )
    parser.add_argument("--output-dir", type=Path, required=True, help="Directory for annotated images and summary JSON")
    parser.add_argument(
        "--imgsz",
        type=int,
        default=None,
        help="Square custom-model inference size; defaults to the checkpoint training value",
    )
    parser.add_argument("--batch-size", type=int, default=8, help="Images per inference batch")
    parser.add_argument("--conf", type=float, default=0.25, help="Prediction confidence floor for displayed detections")
    parser.add_argument("--nms-iou", type=float, default=0.70, help="Class-aware NMS IoU threshold")
    parser.add_argument("--max-det", type=int, default=300, help="Maximum detections retained per image")
    parser.add_argument("--device", type=str, default="cuda", help="Inference device, such as cuda or cuda:0")
    parser.add_argument(
        "--inference-branch",
        choices=("one2many", "one2one"),
        default="one2many",
        help="Custom raw branch to decode; one2many is the selected local configuration",
    )
    parser.add_argument(
        "--max-images",
        type=int,
        default=8,
        help="Maximum sorted images to process; use 0 to process every image",
    )
    parser.add_argument(
        "--image-names",
        nargs="*",
        default=None,
        help="Optional exact image filenames under --source for a curated demonstration batch",
    )
    return parser.parse_args()


def _load_checkpoint(checkpoint_path: Path, device: torch.device) -> dict[str, Any]:
    """Load a trusted local checkpoint while supporting Path metadata in older files."""
    try:
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=True)
        if isinstance(checkpoint, dict):
            return checkpoint
    except (TypeError, pickle.UnpicklingError):
        pass

    try:
        if hasattr(torch.serialization, "add_safe_globals"):
            torch.serialization.add_safe_globals([Path, type(Path())])
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=True)
        if isinstance(checkpoint, dict):
            return checkpoint
    except Exception:
        pass

    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=FutureWarning, module="torch.serialization")
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    if not isinstance(checkpoint, dict):
        raise RuntimeError(f"Checkpoint is not a metadata dictionary: {checkpoint_path}")
    return checkpoint


def _checkpoint_model_settings(checkpoint: dict[str, Any]) -> tuple[str, int, bool]:
    saved_args = checkpoint.get("args", {})
    if not isinstance(saved_args, dict):
        saved_args = {}
    scale = str(saved_args.get("scale", "n"))
    reg_max = int(saved_args.get("reg_max", 1))
    use_p2 = bool(saved_args.get("use_p2", False))
    if reg_max <= 0:
        raise ValueError("Checkpoint reg_max must be positive")
    return scale, reg_max, use_p2


def _checkpoint_imgsz(checkpoint: dict[str, Any], requested_imgsz: int | None) -> int:
    saved_args = checkpoint.get("args", {})
    if not isinstance(saved_args, dict):
        saved_args = {}
    imgsz = requested_imgsz if requested_imgsz is not None else int(saved_args.get("imgsz", 640))
    if imgsz <= 0:
        raise ValueError("--imgsz must be positive")
    return imgsz


def discover_images(source: Path, max_images: int, image_names: Sequence[str] | None) -> list[Path]:
    if max_images < 0:
        raise ValueError("--max-images must be zero or positive")
    if source.is_file():
        if image_names:
            raise ValueError("--image-names is available only when --source is a directory")
        if source.suffix.lower() not in VALID_IMAGE_EXTENSIONS:
            raise ValueError(f"--source image has an unsupported extension: {source}")
        return [source]
    if not source.is_dir():
        raise FileNotFoundError(f"--source does not exist: {source}")

    available = {
        path.name: path
        for path in sorted(source.iterdir())
        if path.is_file() and path.suffix.lower() in VALID_IMAGE_EXTENSIONS
    }
    if not available:
        raise FileNotFoundError(f"No supported images found under {source}")
    if image_names:
        missing = [name for name in image_names if name not in available]
        if missing:
            raise FileNotFoundError(f"Requested image names were not found under {source}: {missing}")
        selected = [available[name] for name in image_names]
    else:
        selected = list(available.values())
    return selected if max_images == 0 else selected[:max_images]


def _resize_for_model(image_bgr: Any, imgsz: int) -> torch.Tensor:
    """Use the custom evaluator's RGB conversion and square-stretch resize policy."""
    original_h, original_w = image_bgr.shape[:2]
    if imgsz < original_w or imgsz < original_h:
        interpolation = cv2.INTER_AREA
    elif imgsz > original_w or imgsz > original_h:
        interpolation = cv2.INTER_CUBIC
    else:
        interpolation = cv2.INTER_LINEAR
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    resized = cv2.resize(image_rgb, (imgsz, imgsz), interpolation=interpolation)
    return torch.from_numpy(resized).permute(2, 0, 1).float() / 255.0


def _read_ground_truth(label_path: Path, class_names: Sequence[str], image_width: int, image_height: int) -> list[dict[str, Any]]:
    if not label_path.exists():
        return []

    ground_truth: list[dict[str, Any]] = []
    for line_number, line in enumerate(label_path.read_text(encoding="utf-8").splitlines(), start=1):
        parts = line.strip().split()
        if not parts:
            continue
        if len(parts) != 5:
            raise ValueError(f"{label_path}:{line_number}: expected five-field YOLO label")
        try:
            class_value, x_center, y_center, width, height = (float(value) for value in parts)
        except ValueError as exc:
            raise ValueError(f"{label_path}:{line_number}: label values must be numeric") from exc
        class_id = int(class_value)
        if class_value != class_id or not 0 <= class_id < len(class_names):
            raise ValueError(f"{label_path}:{line_number}: class ID is outside the dataset taxonomy")
        if not (0.0 <= x_center <= 1.0 and 0.0 <= y_center <= 1.0 and 0.0 < width <= 1.0 and 0.0 < height <= 1.0):
            raise ValueError(f"{label_path}:{line_number}: invalid normalized box")
        x1 = max(0.0, (x_center - width / 2.0) * image_width)
        y1 = max(0.0, (y_center - height / 2.0) * image_height)
        x2 = min(float(image_width), (x_center + width / 2.0) * image_width)
        y2 = min(float(image_height), (y_center + height / 2.0) * image_height)
        ground_truth.append(
            {
                "class_id": class_id,
                "class_name": class_names[class_id],
                "xyxy": [x1, y1, x2, y2],
            }
        )
    return ground_truth


def _scale_detections_to_source(detections: torch.Tensor, image_width: int, image_height: int, imgsz: int) -> torch.Tensor:
    scaled = detections.detach().cpu().clone()
    if scaled.numel() == 0:
        return scaled
    scaled[:, 0] = (scaled[:, 0] * image_width / imgsz).clamp(0.0, float(image_width))
    scaled[:, 2] = (scaled[:, 2] * image_width / imgsz).clamp(0.0, float(image_width))
    scaled[:, 1] = (scaled[:, 1] * image_height / imgsz).clamp(0.0, float(image_height))
    scaled[:, 3] = (scaled[:, 3] * image_height / imgsz).clamp(0.0, float(image_height))
    return scaled


def _draw_label(image: Any, text: str, origin: tuple[int, int], color: tuple[int, int, int]) -> None:
    font = cv2.FONT_HERSHEY_SIMPLEX
    scale = 0.48
    thickness = 1
    (text_width, text_height), baseline = cv2.getTextSize(text, font, scale, thickness)
    x, y = origin
    top = max(0, y - text_height - baseline - 4)
    cv2.rectangle(image, (x, top), (x + text_width + 4, y + 2), color, thickness=-1)
    cv2.putText(image, text, (x + 2, y - baseline), font, scale, (0, 0, 0), thickness, cv2.LINE_AA)


def annotate_image(image_bgr: Any, ground_truth: Sequence[dict[str, Any]], detections: torch.Tensor, class_names: Sequence[str]) -> Any:
    annotated = image_bgr.copy()
    for item in ground_truth:
        x1, y1, x2, y2 = (int(round(value)) for value in item["xyxy"])
        cv2.rectangle(annotated, (x1, y1), (x2, y2), GROUND_TRUTH_COLOR_BGR, thickness=2, lineType=cv2.LINE_AA)
        _draw_label(annotated, f"GT {item['class_name']}", (x1, y1), GROUND_TRUTH_COLOR_BGR)

    for detection in detections:
        x1, y1, x2, y2, score, class_value = detection.tolist()
        class_id = int(class_value)
        color = PREDICTION_COLORS_BGR[class_id % len(PREDICTION_COLORS_BGR)]
        p1 = (int(round(x1)), int(round(y1)))
        p2 = (int(round(x2)), int(round(y2)))
        cv2.rectangle(annotated, p1, p2, color, thickness=2, lineType=cv2.LINE_AA)
        _draw_label(annotated, f"P {class_names[class_id]} {score:.2f}", p1, color)

    cv2.rectangle(annotated, (6, 6), (310, 30), (30, 30, 30), thickness=-1)
    cv2.putText(
        annotated,
        "White = ground truth | Color = custom YOLO26",
        (10, 23),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.42,
        (255, 255, 255),
        1,
        cv2.LINE_AA,
    )
    return annotated


def serialise_detections(detections: torch.Tensor, class_names: Sequence[str]) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for detection in detections:
        x1, y1, x2, y2, score, class_value = detection.tolist()
        class_id = int(class_value)
        records.append(
            {
                "class_id": class_id,
                "class_name": class_names[class_id],
                "confidence": float(score),
                "xyxy": [float(x1), float(y1), float(x2), float(y2)],
            }
        )
    return records


def main() -> None:
    args = parse_args()
    if not torch.cuda.is_available() and args.device.startswith("cuda"):
        raise RuntimeError("CUDA device requested but no GPU is available to PyTorch")
    if args.batch_size <= 0 or args.max_det <= 0:
        raise ValueError("--batch-size and --max-det must be positive")
    if not 0.0 <= args.conf <= 1.0 or not 0.0 < args.nms_iou <= 1.0:
        raise ValueError("--conf must be in [0, 1] and --nms-iou must be in (0, 1]")

    args.checkpoint = args.checkpoint.resolve()
    args.data_root = args.data_root.resolve()
    args.source = args.source.resolve()
    if args.labels_dir is not None:
        args.labels_dir = args.labels_dir.resolve()
        if not args.labels_dir.is_dir():
            raise FileNotFoundError(f"--labels-dir does not exist: {args.labels_dir}")
    if not args.checkpoint.is_file():
        raise FileNotFoundError(f"Checkpoint does not exist: {args.checkpoint}")

    device = torch.device(args.device)
    checkpoint = _load_checkpoint(args.checkpoint, device)
    dataset_config = read_yolo_dataset_config(args.data_root)
    saved_class_names = checkpoint.get("class_names")
    if saved_class_names is not None and tuple(str(name) for name in saved_class_names) != dataset_config.class_names:
        raise ValueError("Checkpoint class names do not match the selected dataset taxonomy")
    saved_num_classes = checkpoint.get("num_classes")
    if saved_num_classes is not None and int(saved_num_classes) != dataset_config.num_classes:
        raise ValueError("Checkpoint class count does not match the selected dataset taxonomy")

    scale, reg_max, use_p2 = _checkpoint_model_settings(checkpoint)
    imgsz = _checkpoint_imgsz(checkpoint, args.imgsz)
    checkpoint_weight_source = str(checkpoint.get("checkpoint_weight_source", "raw"))
    if checkpoint_weight_source not in {"raw", "ema"}:
        raise ValueError(f"Unsupported checkpoint weight source: {checkpoint_weight_source!r}")
    image_paths = discover_images(args.source, args.max_images, args.image_names)

    model = build_yolo26(
        nc=dataset_config.num_classes,
        scale=scale,
        topk=args.max_det,
        reg_max=reg_max,
        use_p2=use_p2,
    ).to(device)
    state_dict = checkpoint.get("model_state_dict")
    if not isinstance(state_dict, dict):
        raise ValueError("Checkpoint does not contain a model_state_dict")
    model.load_state_dict(state_dict)
    model.eval()

    loaded_images: list[tuple[Path, Any, torch.Tensor]] = []
    for image_path in image_paths:
        image_bgr = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
        if image_bgr is None:
            raise FileNotFoundError(f"Unable to read image: {image_path}")
        loaded_images.append((image_path, image_bgr, _resize_for_model(image_bgr, imgsz)))

    args.output_dir.mkdir(parents=True, exist_ok=True)
    summary_images: list[dict[str, Any]] = []
    class_names = dataset_config.class_names
    use_amp = device.type == "cuda"

    for start in range(0, len(loaded_images), args.batch_size):
        batch = loaded_images[start : start + args.batch_size]
        inputs = torch.stack([item[2] for item in batch], dim=0).to(device, non_blocking=True)
        with torch.no_grad(), torch.amp.autocast(device_type=device.type, enabled=use_amp):
            outputs = model(inputs)
            decoded = model.detect.decode_branch(outputs[args.inference_branch])
        batch_detections = class_aware_nms(
            decoded,
            num_classes=dataset_config.num_classes,
            score_threshold=args.conf,
            iou_threshold=args.nms_iou,
            max_detections=args.max_det,
        )
        if len(batch_detections) != len(batch):
            raise RuntimeError("Custom YOLO26 returned a different number of prediction batches than source images")

        for offset, ((image_path, image_bgr, _), detections) in enumerate(zip(batch, batch_detections), start=start + 1):
            image_height, image_width = image_bgr.shape[:2]
            scaled_detections = _scale_detections_to_source(detections, image_width, image_height, imgsz)
            ground_truth = (
                _read_ground_truth(
                    args.labels_dir / f"{image_path.stem}.txt",
                    class_names,
                    image_width,
                    image_height,
                )
                if args.labels_dir is not None
                else []
            )
            annotated = annotate_image(image_bgr, ground_truth, scaled_detections, class_names)
            output_path = args.output_dir / f"{offset:02d}_{image_path.stem}_annotated.jpg"
            if not cv2.imwrite(str(output_path), annotated):
                raise RuntimeError(f"Unable to write annotated output: {output_path}")

            image_summary = {
                "source_image": str(image_path),
                "annotated_image": str(output_path),
                "original_shape": [image_height, image_width],
                "ground_truth_count": len(ground_truth),
                "ground_truth": ground_truth,
                "detection_count": int(scaled_detections.shape[0]),
                "detections": serialise_detections(scaled_detections, class_names),
            }
            summary_images.append(image_summary)
            print(
                f"{image_path.name}: ground_truth={len(ground_truth)} "
                f"detections={image_summary['detection_count']} output={output_path.name}"
            )

    summary = {
        "purpose": "Custom scratch YOLO26 batch demonstration output; not a cross-validation metric report",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "model": "custom_yolo26",
        "checkpoint": str(args.checkpoint),
        "checkpoint_epoch": int(checkpoint.get("epoch", 0)),
        "checkpoint_weight_source": checkpoint_weight_source,
        "data_root": str(args.data_root),
        "source": str(args.source),
        "labels_dir": str(args.labels_dir) if args.labels_dir is not None else None,
        "class_names": list(class_names),
        "inference_settings": {
            "imgsz": imgsz,
            "batch_size": args.batch_size,
            "confidence_threshold": args.conf,
            "inference_branch": args.inference_branch,
            "nms_iou": args.nms_iou,
            "max_detections": args.max_det,
            "device": args.device,
            "scale": scale,
            "reg_max": reg_max,
            "use_p2": use_p2,
            "training_online_augmentation": (
                checkpoint.get("args", {}).get("online_augmentation", "none")
                if isinstance(checkpoint.get("args", {}), dict)
                else "none"
            ),
        },
        "image_count": len(summary_images),
        "images": summary_images,
    }
    summary_path = args.output_dir / "demo_predictions.json"
    summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote {len(summary_images)} annotated images and {summary_path}")


if __name__ == "__main__":
    main()
