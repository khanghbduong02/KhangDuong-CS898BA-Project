"""Create annotated batch-inference outputs for a recorded project demonstration.

The script is intentionally separate from cross-validation evaluation. It is
for visualising a frozen Ultralytics detector checkpoint on user-selected
images, while preserving a machine-readable prediction summary.
"""
from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import cv2
from ultralytics import YOLO


VALID_IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run a frozen Ultralytics detector on images and save annotated batch-demo outputs."
    )
    parser.add_argument("--checkpoint", type=Path, required=True, help="Fine-tuned Ultralytics .pt checkpoint")
    parser.add_argument("--source", type=Path, required=True, help="One image or a directory of images")
    parser.add_argument("--output-dir", type=Path, required=True, help="Directory for annotated images and summary JSON")
    parser.add_argument("--imgsz", type=int, default=640, help="Ultralytics inference image size")
    parser.add_argument("--batch-size", type=int, default=8, help="Images per inference batch")
    parser.add_argument("--conf", type=float, default=0.25, help="Detection confidence threshold")
    parser.add_argument("--iou", type=float, default=0.70, help="NMS IoU threshold")
    parser.add_argument("--max-det", type=int, default=300, help="Maximum detections per image")
    parser.add_argument("--device", type=str, default="0", help="Ultralytics device selector; use 0 for the first GPU")
    parser.add_argument(
        "--max-images",
        type=int,
        default=12,
        help="Maximum sorted images to process; use 0 to process every image",
    )
    return parser.parse_args()


def discover_images(source: Path, max_images: int) -> list[Path]:
    if max_images < 0:
        raise ValueError("--max-images must be zero or positive")
    if source.is_file():
        if source.suffix.lower() not in VALID_IMAGE_EXTENSIONS:
            raise ValueError(f"--source image has an unsupported extension: {source}")
        return [source]
    if not source.is_dir():
        raise FileNotFoundError(f"--source does not exist: {source}")

    paths = sorted(path for path in source.iterdir() if path.is_file() and path.suffix.lower() in VALID_IMAGE_EXTENSIONS)
    if not paths:
        raise FileNotFoundError(f"No supported images found under {source}")
    return paths if max_images == 0 else paths[:max_images]


def serialise_result(result: Any, image_path: Path) -> dict[str, Any]:
    detections: list[dict[str, Any]] = []
    if result.boxes is not None:
        boxes = result.boxes.xyxy.detach().cpu().tolist()
        scores = result.boxes.conf.detach().cpu().tolist()
        class_ids = result.boxes.cls.detach().cpu().long().tolist()
        for box, score, class_id in zip(boxes, scores, class_ids):
            detections.append(
                {
                    "class_id": int(class_id),
                    "class_name": str(result.names[int(class_id)]),
                    "confidence": float(score),
                    "xyxy": [float(value) for value in box],
                }
            )
    return {
        "source_image": str(image_path),
        "original_shape": [int(value) for value in result.orig_shape],
        "detection_count": len(detections),
        "detections": detections,
    }


def main() -> None:
    args = parse_args()
    if args.imgsz <= 0 or args.batch_size <= 0 or args.max_det <= 0:
        raise ValueError("--imgsz, --batch-size, and --max-det must be positive")
    if not 0.0 <= args.conf <= 1.0 or not 0.0 < args.iou <= 1.0:
        raise ValueError("--conf must be in [0, 1] and --iou must be in (0, 1]")
    if not args.checkpoint.is_file():
        raise FileNotFoundError(f"Checkpoint does not exist: {args.checkpoint}")

    image_paths = discover_images(args.source, args.max_images)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    model = YOLO(str(args.checkpoint))

    summary_images: list[dict[str, Any]] = []
    for start in range(0, len(image_paths), args.batch_size):
        batch_paths = image_paths[start : start + args.batch_size]
        results = model.predict(
            source=[str(path) for path in batch_paths],
            imgsz=args.imgsz,
            batch=len(batch_paths),
            conf=args.conf,
            iou=args.iou,
            max_det=args.max_det,
            device=args.device,
            save=False,
            verbose=False,
        )
        if len(results) != len(batch_paths):
            raise RuntimeError("Ultralytics returned a different number of results than source images")

        for image_path, result in zip(batch_paths, results):
            output_path = args.output_dir / f"{image_path.stem}_annotated.jpg"
            if not cv2.imwrite(str(output_path), result.plot()):
                raise RuntimeError(f"Could not write annotated output: {output_path}")
            image_summary = serialise_result(result, image_path)
            image_summary["annotated_image"] = str(output_path)
            summary_images.append(image_summary)
            print(f"{image_path.name}: detections={image_summary['detection_count']} output={output_path.name}")

    summary = {
        "purpose": "Batch demonstration output; not a cross-validation metric report",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "checkpoint": str(args.checkpoint.resolve()),
        "source": str(args.source.resolve()),
        "inference_settings": {
            "imgsz": args.imgsz,
            "batch_size": args.batch_size,
            "confidence_threshold": args.conf,
            "nms_iou": args.iou,
            "max_detections": args.max_det,
            "device": args.device,
        },
        "image_count": len(summary_images),
        "images": summary_images,
    }
    summary_path = args.output_dir / "demo_predictions.json"
    summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote {len(summary_images)} annotated images and {summary_path}")


if __name__ == "__main__":
    main()