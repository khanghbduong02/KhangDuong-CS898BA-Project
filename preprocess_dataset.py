from __future__ import annotations

import argparse
import shutil
from pathlib import Path

import cv2
import numpy as np


VALID_IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}


def preprocess_baseline(image_bgr: np.ndarray) -> np.ndarray:
    return image_bgr


def preprocess_clahe(image_bgr: np.ndarray, clip_limit: float, tile_grid_size: int) -> np.ndarray:
    lab = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2LAB)
    l_channel, a_channel, b_channel = cv2.split(lab)

    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=(tile_grid_size, tile_grid_size))
    l_enhanced = clahe.apply(l_channel)

    merged = cv2.merge((l_enhanced, a_channel, b_channel))
    return cv2.cvtColor(merged, cv2.COLOR_LAB2BGR)


def preprocess_clahe_canny(
    image_bgr: np.ndarray,
    clip_limit: float,
    tile_grid_size: int,
    canny_low: int,
    canny_high: int,
    edge_overlay_weight: float,
) -> np.ndarray:
    clahe_img = preprocess_clahe(image_bgr, clip_limit=clip_limit, tile_grid_size=tile_grid_size)
    gray = cv2.cvtColor(clahe_img, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, threshold1=canny_low, threshold2=canny_high)
    edges_3ch = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)

    # Blend edges with the enhanced image so detectors still get contextual texture and color.
    return cv2.addWeighted(clahe_img, 1.0, edges_3ch, edge_overlay_weight, 0.0)


def process_image(
    image_path: Path,
    output_path: Path,
    method: str,
    clip_limit: float,
    tile_grid_size: int,
    canny_low: int,
    canny_high: int,
    edge_overlay_weight: float,
) -> bool:
    image = cv2.imread(str(image_path))
    if image is None:
        return False

    if method == "baseline":
        processed = preprocess_baseline(image)
    elif method == "clahe":
        processed = preprocess_clahe(image, clip_limit=clip_limit, tile_grid_size=tile_grid_size)
    elif method == "clahe_canny":
        processed = preprocess_clahe_canny(
            image,
            clip_limit=clip_limit,
            tile_grid_size=tile_grid_size,
            canny_low=canny_low,
            canny_high=canny_high,
            edge_overlay_weight=edge_overlay_weight,
        )
    else:
        raise ValueError(f"Unsupported method: {method}")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    return bool(cv2.imwrite(str(output_path), processed))


def copy_labels(input_split_dir: Path, output_split_dir: Path) -> None:
    src_labels = input_split_dir / "labels"
    dst_labels = output_split_dir / "labels"

    if not src_labels.exists():
        print(f"[WARN] Labels folder not found: {src_labels}")
        return

    if dst_labels.exists():
        shutil.rmtree(dst_labels)
    shutil.copytree(src_labels, dst_labels)


def write_data_yaml(dataset_root: Path, class_names: list[str]) -> None:
    content = "\n".join(
        [
            "train: train/images",
            "val: valid/images",
            "test: test/images",
            "",
            f"nc: {len(class_names)}",
            f"names: {class_names}",
            "",
        ]
    )
    (dataset_root / "data.yaml").write_text(content, encoding="utf-8")


def process_split(
    input_split_dir: Path,
    output_split_dir: Path,
    method: str,
    clip_limit: float,
    tile_grid_size: int,
    canny_low: int,
    canny_high: int,
    edge_overlay_weight: float,
) -> tuple[int, int]:
    src_images_dir = input_split_dir / "images"
    dst_images_dir = output_split_dir / "images"
    dst_images_dir.mkdir(parents=True, exist_ok=True)

    success = 0
    failed = 0

    for image_path in sorted(src_images_dir.iterdir()):
        if not image_path.is_file() or image_path.suffix.lower() not in VALID_IMAGE_EXTENSIONS:
            continue

        out_path = dst_images_dir / image_path.name
        ok = process_image(
            image_path=image_path,
            output_path=out_path,
            method=method,
            clip_limit=clip_limit,
            tile_grid_size=tile_grid_size,
            canny_low=canny_low,
            canny_high=canny_high,
            edge_overlay_weight=edge_overlay_weight,
        )
        if ok:
            success += 1
        else:
            failed += 1

    copy_labels(input_split_dir, output_split_dir)
    return success, failed


def run_pipeline(
    input_dir: Path,
    output_dir: Path,
    class_names: list[str],
    clip_limit: float,
    tile_grid_size: int,
    canny_low: int,
    canny_high: int,
    edge_overlay_weight: float,
) -> None:
    methods = ["baseline", "clahe", "clahe_canny"]
    splits = ["train", "valid", "test"]

    output_dir.mkdir(parents=True, exist_ok=True)

    for method in methods:
        print(f"\n=== Processing method: {method} ===")
        method_root = output_dir / method
        method_root.mkdir(parents=True, exist_ok=True)

        total_success = 0
        total_failed = 0

        for split in splits:
            split_in = input_dir / split
            split_out = method_root / split

            if not split_in.exists():
                print(f"[WARN] Split not found, skipping: {split_in}")
                continue

            success, failed = process_split(
                input_split_dir=split_in,
                output_split_dir=split_out,
                method=method,
                clip_limit=clip_limit,
                tile_grid_size=tile_grid_size,
                canny_low=canny_low,
                canny_high=canny_high,
                edge_overlay_weight=edge_overlay_weight,
            )
            total_success += success
            total_failed += failed
            print(f"[{method}][{split}] processed={success} failed={failed}")

        write_data_yaml(method_root, class_names)
        print(f"[{method}] done. total_processed={total_success} total_failed={total_failed}")
        print(f"[{method}] yaml: {method_root / 'data.yaml'}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate baseline, CLAHE, and CLAHE+Canny dataset variants for 3D print failure detection."
    )
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=Path("roboflow-data"),
        help="Input Roboflow dataset root (contains train/valid/test folders)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("processed-data"),
        help="Output root for processed dataset variants",
    )
    parser.add_argument(
        "--class-names",
        nargs="+",
        default=["spaghetti", "stringing", "warping"],
        help="Class names written to data.yaml",
    )
    parser.add_argument("--clip-limit", type=float, default=2.0, help="CLAHE clip limit")
    parser.add_argument("--tile-grid-size", type=int, default=8, help="CLAHE tile grid size")
    parser.add_argument("--canny-low", type=int, default=50, help="Canny low threshold")
    parser.add_argument("--canny-high", type=int, default=150, help="Canny high threshold")
    parser.add_argument(
        "--edge-overlay-weight",
        type=float,
        default=0.35,
        help="Edge overlay weight for CLAHE+Canny blending",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if not args.input_dir.exists():
        raise FileNotFoundError(f"Input directory not found: {args.input_dir}")

    run_pipeline(
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        class_names=args.class_names,
        clip_limit=args.clip_limit,
        tile_grid_size=args.tile_grid_size,
        canny_low=args.canny_low,
        canny_high=args.canny_high,
        edge_overlay_weight=args.edge_overlay_weight,
    )


if __name__ == "__main__":
    main()