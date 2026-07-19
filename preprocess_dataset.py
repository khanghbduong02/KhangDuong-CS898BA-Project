from __future__ import annotations

import argparse
import hashlib
import math
import random
import shutil
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np


VALID_IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}
LABEL_TOLERANCE = 1e-6


@dataclass
class LabelNormalizationStats:
    label_files: int = 0
    empty_files: int = 0
    bbox_rows: int = 0
    polygon_rows: int = 0
    boxes_written: int = 0
    clipped_rows: int = 0
    bbox_by_class: dict[int, int] = field(default_factory=dict)
    polygon_by_class: dict[int, int] = field(default_factory=dict)


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


def _parse_class_id(value: str, label_path: Path, line_number: int, num_classes: int) -> int:
    try:
        class_value = float(value)
    except ValueError as exc:
        raise ValueError(f"{label_path}:{line_number}: class ID is not numeric: {value!r}") from exc

    if not math.isfinite(class_value) or not class_value.is_integer():
        raise ValueError(f"{label_path}:{line_number}: class ID must be a finite integer, got {value!r}")

    class_id = int(class_value)
    if not 0 <= class_id < num_classes:
        raise ValueError(
            f"{label_path}:{line_number}: class ID {class_id} is outside the configured range 0..{num_classes - 1}"
        )
    return class_id


def _parse_normalized_values(values: list[str], label_path: Path, line_number: int) -> list[float]:
    try:
        parsed = [float(value) for value in values]
    except ValueError as exc:
        raise ValueError(f"{label_path}:{line_number}: label coordinates must be numeric") from exc

    if not all(math.isfinite(value) for value in parsed):
        raise ValueError(f"{label_path}:{line_number}: label coordinates must be finite")
    if any(value < -LABEL_TOLERANCE or value > 1.0 + LABEL_TOLERANCE for value in parsed):
        raise ValueError(f"{label_path}:{line_number}: normalized label coordinates must be in [0, 1]")

    return [_clip01(value) for value in parsed]


def _box_from_corners(
    class_id: int,
    x1: float,
    y1: float,
    x2: float,
    y2: float,
    label_path: Path,
    line_number: int,
) -> tuple[tuple[int, float, float, float, float], bool]:
    clipped_x1 = _clip01(x1)
    clipped_y1 = _clip01(y1)
    clipped_x2 = _clip01(x2)
    clipped_y2 = _clip01(y2)
    clipped = (clipped_x1, clipped_y1, clipped_x2, clipped_y2) != (x1, y1, x2, y2)

    if clipped_x2 - clipped_x1 <= LABEL_TOLERANCE or clipped_y2 - clipped_y1 <= LABEL_TOLERANCE:
        raise ValueError(f"{label_path}:{line_number}: annotation has zero-area bounding box after clipping")

    x_center = (clipped_x1 + clipped_x2) / 2.0
    y_center = (clipped_y1 + clipped_y2) / 2.0
    width = clipped_x2 - clipped_x1
    height = clipped_y2 - clipped_y1
    return (class_id, x_center, y_center, width, height), clipped


def normalize_label_row(
    parts: list[str],
    label_path: Path,
    line_number: int,
    num_classes: int,
) -> tuple[tuple[int, float, float, float, float], str, bool]:
    """Convert one raw detection or polygon row into a standard YOLO bounding box."""
    class_id = _parse_class_id(parts[0], label_path, line_number, num_classes)

    # Roboflow's documented YOLO detection export uses exactly five fields:
    # class_id, x_center, y_center, width, height. A valid polygon needs at
    # least three coordinate pairs, so it has at least seven fields total.
    if len(parts) == 5:
        x_center, y_center, width, height = _parse_normalized_values(parts[1:], label_path, line_number)
        if width <= LABEL_TOLERANCE or height <= LABEL_TOLERANCE:
            raise ValueError(f"{label_path}:{line_number}: YOLO box width and height must be positive")
        box, clipped = _box_from_corners(
            class_id,
            x_center - width / 2.0,
            y_center - height / 2.0,
            x_center + width / 2.0,
            y_center + height / 2.0,
            label_path,
            line_number,
        )
        return box, "bbox", clipped

    if len(parts) >= 7 and len(parts) % 2 == 1:
        polygon_values = _parse_normalized_values(parts[1:], label_path, line_number)
        x_values = polygon_values[0::2]
        y_values = polygon_values[1::2]
        box, clipped = _box_from_corners(
            class_id,
            min(x_values),
            min(y_values),
            max(x_values),
            max(y_values),
            label_path,
            line_number,
        )
        return box, "polygon", clipped

    raise ValueError(
        f"{label_path}:{line_number}: expected a five-field YOLO box or an odd-length polygon row with at least "
        f"three points; found {len(parts)} fields"
    )


def normalize_labels(input_split_dir: Path, output_split_dir: Path, num_classes: int) -> LabelNormalizationStats:
    src_labels = input_split_dir / "labels"
    dst_labels = output_split_dir / "labels"
    if not src_labels.exists():
        raise FileNotFoundError(f"Labels folder not found: {src_labels}")

    dst_labels.mkdir(parents=True, exist_ok=True)
    stats = LabelNormalizationStats()

    for src_label_path in sorted(src_labels.glob("*.txt")):
        stats.label_files += 1
        normalized_boxes: list[tuple[int, float, float, float, float]] = []
        has_annotation = False

        for line_number, raw_line in enumerate(src_label_path.read_text(encoding="utf-8").splitlines(), start=1):
            parts = raw_line.strip().split()
            if not parts:
                continue

            has_annotation = True
            box, row_type, clipped = normalize_label_row(parts, src_label_path, line_number, num_classes)
            normalized_boxes.append(box)
            stats.boxes_written += 1
            stats.clipped_rows += int(clipped)

            if row_type == "bbox":
                stats.bbox_rows += 1
                stats.bbox_by_class[box[0]] = stats.bbox_by_class.get(box[0], 0) + 1
            else:
                stats.polygon_rows += 1
                stats.polygon_by_class[box[0]] = stats.polygon_by_class.get(box[0], 0) + 1

        if not has_annotation:
            stats.empty_files += 1
        write_yolo_label_file(dst_labels / src_label_path.name, normalized_boxes)

    return stats


def print_label_normalization_summary(
    split_name: str,
    stats: LabelNormalizationStats,
    class_names: list[str],
) -> None:
    bbox_summary = ", ".join(
        f"{class_names[class_id]}:{stats.bbox_by_class.get(class_id, 0)}" for class_id in range(len(class_names))
    )
    polygon_summary = ", ".join(
        f"{class_names[class_id]}:{stats.polygon_by_class.get(class_id, 0)}" for class_id in range(len(class_names))
    )
    print(
        f"[LABEL][{split_name}] files={stats.label_files} empty={stats.empty_files} "
        f"five_field_boxes={stats.bbox_rows} polygons_converted={stats.polygon_rows} "
        f"boxes_written={stats.boxes_written} clipped={stats.clipped_rows}"
    )
    print(f"[LABEL][{split_name}] five_field_by_class=({bbox_summary})")
    print(f"[LABEL][{split_name}] polygons_converted_by_class=({polygon_summary})")


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


def read_yolo_label_file(label_path: Path, num_classes: int) -> list[tuple[int, float, float, float, float]]:
    boxes: list[tuple[int, float, float, float, float]] = []
    if not label_path.exists():
        return boxes

    for line_number, raw_line in enumerate(label_path.read_text(encoding="utf-8").splitlines(), start=1):
        parts = raw_line.strip().split()
        if not parts:
            continue
        if len(parts) != 5:
            raise ValueError(f"{label_path}:{line_number}: processed labels must contain exactly five fields")

        box, row_type, _ = normalize_label_row(parts, label_path, line_number, num_classes)
        if row_type != "bbox":
            raise RuntimeError(f"{label_path}:{line_number}: expected a normalized bounding box")
        boxes.append(box)
    return boxes


def write_yolo_label_file(label_path: Path, boxes: list[tuple[int, float, float, float, float]]) -> None:
    lines = [f"{cid} {x:.6f} {y:.6f} {w:.6f} {h:.6f}" for cid, x, y, w, h in boxes]
    label_path.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")


def _clip01(value: float) -> float:
    return float(max(0.0, min(1.0, value)))


def _apply_affine_to_boxes(
    boxes: list[tuple[int, float, float, float, float]],
    matrix: np.ndarray,
    width: int,
    height: int,
) -> list[tuple[int, float, float, float, float]]:
    transformed: list[tuple[int, float, float, float, float]] = []
    for cid, x, y, w, h in boxes:
        x1 = (x - w / 2.0) * width
        y1 = (y - h / 2.0) * height
        x2 = (x + w / 2.0) * width
        y2 = (y + h / 2.0) * height
        corners = np.array([[x1, y1], [x2, y1], [x2, y2], [x1, y2]], dtype=np.float32)
        warped = cv2.transform(corners.reshape(-1, 1, 2), matrix).reshape(-1, 2)

        nx1 = float(np.clip(warped[:, 0].min(), 0.0, width - 1.0))
        ny1 = float(np.clip(warped[:, 1].min(), 0.0, height - 1.0))
        nx2 = float(np.clip(warped[:, 0].max(), 0.0, width - 1.0))
        ny2 = float(np.clip(warped[:, 1].max(), 0.0, height - 1.0))

        nw = (nx2 - nx1) / width
        nh = (ny2 - ny1) / height
        if nw <= 1e-4 or nh <= 1e-4:
            continue

        nx = (nx1 + nx2) / (2.0 * width)
        ny = (ny1 + ny2) / (2.0 * height)
        transformed.append((cid, _clip01(nx), _clip01(ny), _clip01(nw), _clip01(nh)))

    return transformed


def apply_augmentation(
    image: np.ndarray,
    boxes: list[tuple[int, float, float, float, float]],
    aug_name: str,
    rng: random.Random,
    np_rng: np.random.Generator,
) -> tuple[np.ndarray, list[tuple[int, float, float, float, float]], str]:
    if aug_name == "hflip":
        aug_img = cv2.flip(image, 1)
        aug_boxes = [(cid, 1.0 - x, y, w, h) for cid, x, y, w, h in boxes]
        return aug_img, aug_boxes, "hflip"

    if aug_name == "bright":
        beta = rng.uniform(-18.0, 18.0)
        aug_img = cv2.convertScaleAbs(image, alpha=1.0, beta=beta)
        return aug_img, boxes, f"bright:{beta:.3f}"

    if aug_name == "contrast":
        alpha = rng.uniform(0.85, 1.20)
        aug_img = cv2.convertScaleAbs(image, alpha=alpha, beta=0.0)
        return aug_img, boxes, f"contrast:{alpha:.4f}"

    if aug_name == "gamma":
        gamma = rng.uniform(0.85, 1.20)
        inv_gamma = 1.0 / gamma
        lut = np.array([((i / 255.0) ** inv_gamma) * 255 for i in range(256)], dtype=np.uint8)
        aug_img = cv2.LUT(image, lut)
        return aug_img, boxes, f"gamma:{gamma:.4f}"

    if aug_name == "noise":
        sigma = rng.uniform(5.0, 12.0)
        mu = rng.gauss(0.0, sigma)
        noise_map = np_rng.normal(mu, sigma, image.shape).astype(np.float32)
        aug_img = np.clip(image.astype(np.float32) + noise_map, 0, 255).astype(np.uint8)
        return aug_img, boxes, f"noise:mu={mu:.3f}:sigma={sigma:.3f}"

    if aug_name == "blur":
        kernel = 3 if rng.random() < 0.7 else 5
        aug_img = cv2.GaussianBlur(image, (kernel, kernel), sigmaX=0.8)
        return aug_img, boxes, f"blur:k={kernel}"

    if aug_name == "translate":
        h, w = image.shape[:2]
        tx = rng.uniform(-0.08, 0.08) * w
        ty = rng.uniform(-0.08, 0.08) * h
        matrix = np.array([[1.0, 0.0, tx], [0.0, 1.0, ty]], dtype=np.float32)
        aug_img = cv2.warpAffine(image, matrix, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT_101)
        aug_boxes = _apply_affine_to_boxes(boxes, matrix, width=w, height=h)
        return aug_img, aug_boxes, f"translate:tx={tx:.3f}:ty={ty:.3f}"

    if aug_name == "scale":
        h, w = image.shape[:2]
        scale = rng.uniform(0.90, 1.10)
        cx, cy = w / 2.0, h / 2.0
        matrix = np.array([[scale, 0.0, (1.0 - scale) * cx], [0.0, scale, (1.0 - scale) * cy]], dtype=np.float32)
        aug_img = cv2.warpAffine(image, matrix, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT_101)
        aug_boxes = _apply_affine_to_boxes(boxes, matrix, width=w, height=h)
        return aug_img, aug_boxes, f"scale:{scale:.4f}"

    raise ValueError(f"Unsupported augmentation: {aug_name}")


def augment_minority_classes(
    split_dir: Path,
    num_classes: int,
    target_ratio: float,
    augmentations: list[str],
    seed: int,
) -> tuple[int, dict[int, int]]:
    """Augment minority labels until their object counts approach the requested target ratio."""
    images_dir = split_dir / "images"
    labels_dir = split_dir / "labels"
    if not images_dir.exists() or not labels_dir.exists():
        return 0, {i: 0 for i in range(num_classes)}

    label_files = sorted(labels_dir.glob("*.txt"))
    class_counts = [0 for _ in range(num_classes)]
    image_to_boxes: dict[Path, list[tuple[int, float, float, float, float]]] = {}
    class_to_images: dict[int, list[Path]] = {i: [] for i in range(num_classes)}

    for label_path in label_files:
        boxes = read_yolo_label_file(label_path, num_classes=num_classes)
        if not boxes:
            continue
        image_to_boxes[label_path] = boxes
        present = set()
        for cid, _, _, _, _ in boxes:
            if 0 <= cid < num_classes:
                class_counts[cid] += 1
                present.add(cid)
        for cid in present:
            class_to_images[cid].append(label_path)

    object_summary = ", ".join(f"class_{cid}:{class_counts[cid]}" for cid in range(num_classes))
    image_summary = ", ".join(f"class_{cid}:{len(class_to_images[cid])}" for cid in range(num_classes))
    print(f"[AUG][{split_dir.name}] before object_count=({object_summary})")
    print(f"[AUG][{split_dir.name}] before image_count=({image_summary})")

    if not any(class_counts):
        return 0, {i: 0 for i in range(num_classes)}

    majority = max(class_counts)
    target_count = int(round(majority * target_ratio))
    deficits = {cid: max(target_count - class_counts[cid], 0) for cid in range(num_classes)}
    deficits = {cid: deficit for cid, deficit in deficits.items() if deficit > 0}
    if not deficits:
        return 0, {i: 0 for i in range(num_classes)}

    rng = random.Random(seed)
    np_rng = np.random.default_rng(seed)
    aug_created = 0
    per_class_added_boxes = {i: 0 for i in range(num_classes)}
    aug_index = 0
    skipped_duplicate_signature = 0
    skipped_duplicate_hash = 0
    skipped_missing_target = 0
    seen_signatures: set[str] = set()
    seen_hashes: set[str] = set()

    for existing_image_path in sorted(images_dir.iterdir()):
        if not existing_image_path.is_file() or existing_image_path.suffix.lower() not in VALID_IMAGE_EXTENSIONS:
            continue
        existing = cv2.imread(str(existing_image_path), cv2.IMREAD_COLOR)
        if existing is None:
            continue
        ok, encoded = cv2.imencode(".png", existing)
        if ok:
            seen_hashes.add(hashlib.sha1(encoded.tobytes()).hexdigest())

    while deficits:
        progressed = False
        for cid in sorted(list(deficits.keys()), key=lambda c: deficits[c], reverse=True):
            if cid not in deficits:
                continue

            candidates = class_to_images.get(cid, [])
            if not candidates:
                deficits.pop(cid, None)
                continue

            created_this_round = False
            for _ in range(30):
                src_label = rng.choice(candidates)
                src_image = images_dir / f"{src_label.stem}.jpg"
                if not src_image.exists():
                    matching = [images_dir / f"{src_label.stem}{ext}" for ext in VALID_IMAGE_EXTENSIONS]
                    src_image = next((p for p in matching if p.exists()), Path())
                if not src_image or not src_image.exists():
                    continue

                image = cv2.imread(str(src_image))
                if image is None:
                    continue

                boxes = image_to_boxes.get(src_label)
                if boxes is None:
                    boxes = read_yolo_label_file(src_label, num_classes=num_classes)
                aug_name = augmentations[aug_index % len(augmentations)]
                aug_index += 1

                aug_image, aug_boxes, aug_signature = apply_augmentation(
                    image=image,
                    boxes=boxes,
                    aug_name=aug_name,
                    rng=rng,
                    np_rng=np_rng,
                )
                if not aug_boxes:
                    continue

                target_boxes_added = sum(1 for box_cid, _, _, _, _ in aug_boxes if box_cid == cid)
                if target_boxes_added == 0:
                    skipped_missing_target += 1
                    continue

                signature = f"src={src_label.stem}|target={cid}|aug={aug_signature}"
                if signature in seen_signatures:
                    skipped_duplicate_signature += 1
                    continue

                ok, encoded = cv2.imencode(".png", aug_image)
                if not ok:
                    continue
                image_hash = hashlib.sha1(encoded.tobytes()).hexdigest()
                if image_hash in seen_hashes:
                    skipped_duplicate_hash += 1
                    continue

                new_stem = f"{src_label.stem}_aug_c{cid}_{aug_name}_{aug_index:05d}"
                out_image = images_dir / f"{new_stem}{src_image.suffix}"
                out_label = labels_dir / f"{new_stem}.txt"

                if not cv2.imwrite(str(out_image), aug_image):
                    continue
                write_yolo_label_file(out_label, aug_boxes)

                seen_signatures.add(signature)
                seen_hashes.add(image_hash)
                image_to_boxes[out_label] = aug_boxes
                present_aug = {box_cid for box_cid, _, _, _, _ in aug_boxes if 0 <= box_cid < num_classes}
                for box_cid in present_aug:
                    class_to_images[box_cid].append(out_label)

                added_box_counts = [0 for _ in range(num_classes)]
                for box_cid, _, _, _, _ in aug_boxes:
                    if 0 <= box_cid < num_classes:
                        added_box_counts[box_cid] += 1

                for box_cid, added_boxes in enumerate(added_box_counts):
                    if added_boxes == 0:
                        continue
                    class_counts[box_cid] += added_boxes
                    per_class_added_boxes[box_cid] += added_boxes

                for deficit_class in list(deficits):
                    remaining = target_count - class_counts[deficit_class]
                    if remaining > 0:
                        deficits[deficit_class] = remaining
                    else:
                        deficits.pop(deficit_class)

                aug_created += 1
                progressed = True
                created_this_round = True
                break

            if not created_this_round and cid in deficits:
                deficits.pop(cid, None)

        if not progressed:
            break

    print(
        f"[AUG][{split_dir.name}] dedup skipped_signature={skipped_duplicate_signature} "
        f"skipped_hash={skipped_duplicate_hash} skipped_missing_target={skipped_missing_target}"
    )
    final_object_summary = ", ".join(f"class_{class_id}:{count}" for class_id, count in enumerate(class_counts))
    print(f"[AUG][{split_dir.name}] target_boxes_per_minority_class={target_count} final_object_count=({final_object_summary})")

    return aug_created, per_class_added_boxes


def process_split(
    input_split_dir: Path,
    output_split_dir: Path,
    method: str,
    clip_limit: float,
    tile_grid_size: int,
    canny_low: int,
    canny_high: int,
    edge_overlay_weight: float,
    apply_minority_aug: bool,
    num_classes: int,
    class_names: list[str],
    minority_target_ratio: float,
    minority_aug_seed: int,
) -> tuple[int, int]:
    src_images_dir = input_split_dir / "images"
    dst_images_dir = output_split_dir / "images"

    if output_split_dir.exists():
        shutil.rmtree(output_split_dir)
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

    label_stats = normalize_labels(input_split_dir, output_split_dir, num_classes=num_classes)
    print_label_normalization_summary(output_split_dir.name, label_stats, class_names)

    if apply_minority_aug:
        aug_count, per_class_added_boxes = augment_minority_classes(
            split_dir=output_split_dir,
            num_classes=num_classes,
            target_ratio=minority_target_ratio,
            augmentations=["hflip", "bright", "translate", "scale", "noise", "contrast", "gamma", "blur"],
            seed=minority_aug_seed,
        )
        if aug_count > 0:
            summary = ", ".join(
                f"class_{cid}:{count}" for cid, count in per_class_added_boxes.items() if count > 0
            )
            print(f"[AUG][{output_split_dir.name}] created_images={aug_count} boxes_added=({summary})")

    return success, failed


def run_pipeline(
    input_dir: Path,
    output_dir: Path,
    class_names: list[str],
    methods: list[str],
    clip_limit: float,
    tile_grid_size: int,
    canny_low: int,
    canny_high: int,
    edge_overlay_weight: float,
    augment_minority_train: bool,
    minority_target_ratio: float,
    minority_aug_seed: int,
) -> None:
    supported_methods = {"baseline", "clahe", "clahe_canny"}
    unsupported_methods = sorted(set(methods) - supported_methods)
    if unsupported_methods:
        raise ValueError(f"Unsupported preprocessing methods: {', '.join(unsupported_methods)}")
    if not methods:
        raise ValueError("At least one preprocessing method is required")

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
                apply_minority_aug=augment_minority_train and split == "train",
                num_classes=len(class_names),
                class_names=class_names,
                minority_target_ratio=minority_target_ratio,
                minority_aug_seed=minority_aug_seed,
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
    parser.add_argument(
        "--methods",
        nargs="+",
        choices=["baseline", "clahe", "clahe_canny"],
        default=["baseline", "clahe", "clahe_canny"],
        help="Preprocessing variants to generate (default: all variants)",
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
    parser.add_argument(
        "--augment-minority-train",
        action="store_true",
        help="Apply extra augmentations to underrepresented classes in the train split",
    )
    parser.add_argument(
        "--minority-target-ratio",
        type=float,
        default=1.0,
        help="Target minority count ratio relative to majority class count (train split)",
    )
    parser.add_argument(
        "--minority-aug-seed",
        type=int,
        default=42,
        help="Random seed used for minority augmentation sampling",
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
        methods=args.methods,
        clip_limit=args.clip_limit,
        tile_grid_size=args.tile_grid_size,
        canny_low=args.canny_low,
        canny_high=args.canny_high,
        edge_overlay_weight=args.edge_overlay_weight,
        augment_minority_train=args.augment_minority_train,
        minority_target_ratio=args.minority_target_ratio,
        minority_aug_seed=args.minority_aug_seed,
    )


if __name__ == "__main__":
    main()