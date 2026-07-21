"""Download and convert the official PASCAL VOC 2007 splits into strict YOLO labels.

The resulting dataset is a separate project-metric benchmark. It preserves the
official train/validation/test image split but excludes VOC ``difficult=1``
objects consistently because this project metric implementation does not
support VOC's difficult-object ignore semantics. It is not an official PASCAL
VOC leaderboard conversion or evaluation protocol.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import tarfile
import tempfile
import urllib.request
import xml.etree.ElementTree as ET
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable


VOC_CLASSES = (
    "aeroplane",
    "bicycle",
    "bird",
    "boat",
    "bottle",
    "bus",
    "car",
    "cat",
    "chair",
    "cow",
    "diningtable",
    "dog",
    "horse",
    "motorbike",
    "person",
    "pottedplant",
    "sheep",
    "sofa",
    "train",
    "tvmonitor",
)
CLASS_TO_ID = {name: index for index, name in enumerate(VOC_CLASSES)}
EXPECTED_SPLIT_COUNTS = {"train": 2501, "valid": 2510, "test": 4952}
SOURCE_SPLITS = {
    "train": "train",
    "valid": "val",
    "test": "test",
}
ARCHIVES = (
    (
        "VOCtrainval_06-Nov-2007.tar",
        "http://host.robots.ox.ac.uk/pascal/VOC/voc2007/VOCtrainval_06-Nov-2007.tar",
    ),
    (
        "VOCtest_06-Nov-2007.tar",
        "http://host.robots.ox.ac.uk/pascal/VOC/voc2007/VOCtest_06-Nov-2007.tar",
    ),
)
REQUIRED_VOC_DIRECTORIES = (
    "Annotations",
    "JPEGImages",
    "ImageSets/Main",
)


@dataclass(frozen=True)
class VocObject:
    class_id: int
    difficult: bool
    x1: float
    y1: float
    x2: float
    y2: float


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _safe_extract(archive_path: Path, destination: Path) -> None:
    """Extract a trusted archive only after rejecting path-traversal members."""
    destination.mkdir(parents=True, exist_ok=True)
    destination_root = destination.resolve()
    with tarfile.open(archive_path, mode="r:*") as archive:
        members = archive.getmembers()
        for member in members:
            member_path = (destination / member.name).resolve()
            if member_path != destination_root and destination_root not in member_path.parents:
                raise ValueError(f"Unsafe archive member in {archive_path}: {member.name}")
        archive.extractall(destination, members=members)


def _download_file(url: str, destination: Path) -> None:
    """Download one official archive atomically using only the standard library."""
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".part")
    if temporary.exists():
        temporary.unlink()

    request = urllib.request.Request(url, headers={"User-Agent": "3dprint-det-voc2007-preparer/1.0"})
    downloaded = 0
    next_report = 50 * 1024 * 1024
    print(f"Downloading {url} -> {destination}", flush=True)
    try:
        with urllib.request.urlopen(request, timeout=120) as response, temporary.open("wb") as output:
            while chunk := response.read(1024 * 1024):
                output.write(chunk)
                downloaded += len(chunk)
                if downloaded >= next_report:
                    print(f"  downloaded={downloaded / 1024**2:.1f} MiB", flush=True)
                    next_report += 50 * 1024 * 1024
        temporary.replace(destination)
    except Exception:
        temporary.unlink(missing_ok=True)
        raise


def voc_root_ready(voc_root: Path) -> bool:
    return voc_root.is_dir() and all((voc_root / directory).is_dir() for directory in REQUIRED_VOC_DIRECTORIES)


def ensure_source(args: argparse.Namespace) -> Path:
    """Download/extract both official archives unless an intact VOC root already exists."""
    voc_root = args.raw_root / "VOCdevkit" / "VOC2007"
    if voc_root_ready(voc_root):
        return voc_root
    if args.no_download:
        raise FileNotFoundError(
            f"VOC source is incomplete under {args.raw_root}; rerun without --no-download or copy the official archives."
        )

    for filename, url in ARCHIVES:
        archive_path = args.raw_root / filename
        if not archive_path.exists():
            _download_file(url, archive_path)
        print(f"Extracting {archive_path}", flush=True)
        _safe_extract(archive_path, args.raw_root)

    if not voc_root_ready(voc_root):
        raise FileNotFoundError(f"Official VOC 2007 extraction is incomplete under {voc_root}")
    return voc_root


def read_split_ids(voc_root: Path, source_split: str) -> list[str]:
    split_path = voc_root / "ImageSets" / "Main" / f"{source_split}.txt"
    if not split_path.exists():
        raise FileNotFoundError(f"Official VOC split list not found: {split_path}")
    image_ids = [line.strip() for line in split_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if len(image_ids) != len(set(image_ids)):
        raise ValueError(f"Official VOC split list contains duplicate IDs: {split_path}")
    if not image_ids:
        raise ValueError(f"Official VOC split list is empty: {split_path}")
    return image_ids


def _required_text(root: ET.Element, path: str, annotation_path: Path) -> str:
    value = root.findtext(path)
    if value is None:
        raise ValueError(f"{annotation_path}: missing XML field {path!r}")
    return value.strip()


def parse_annotation(annotation_path: Path) -> tuple[int, int, list[VocObject]]:
    """Parse source dimensions and objects from an official VOC XML annotation."""
    try:
        root = ET.parse(annotation_path).getroot()
    except ET.ParseError as exc:
        raise ValueError(f"Invalid VOC XML annotation: {annotation_path}") from exc

    width = int(_required_text(root, "size/width", annotation_path))
    height = int(_required_text(root, "size/height", annotation_path))
    if width <= 0 or height <= 0:
        raise ValueError(f"{annotation_path}: invalid source dimensions {width}x{height}")

    objects: list[VocObject] = []
    for object_node in root.findall("object"):
        class_name = _required_text(object_node, "name", annotation_path)
        if class_name not in CLASS_TO_ID:
            raise ValueError(f"{annotation_path}: unexpected VOC class {class_name!r}")
        difficult_text = object_node.findtext("difficult", default="0").strip()
        try:
            difficult = int(difficult_text) == 1
            # VOC boxes are one-indexed and inclusive. Convert to zero-indexed
            # continuous image boundaries: [xmin - 1, xmax] and likewise y.
            xmin = float(_required_text(object_node, "bndbox/xmin", annotation_path)) - 1.0
            ymin = float(_required_text(object_node, "bndbox/ymin", annotation_path)) - 1.0
            xmax = float(_required_text(object_node, "bndbox/xmax", annotation_path))
            ymax = float(_required_text(object_node, "bndbox/ymax", annotation_path))
        except ValueError as exc:
            raise ValueError(f"{annotation_path}: invalid difficult flag or bounding-box coordinate") from exc
        objects.append(
            VocObject(
                class_id=CLASS_TO_ID[class_name],
                difficult=difficult,
                x1=xmin,
                y1=ymin,
                x2=xmax,
                y2=ymax,
            )
        )
    return width, height, objects


def normalize_box(
    obj: VocObject,
    width: int,
    height: int,
) -> tuple[int, float, float, float, float] | None:
    """Clip a VOC box and convert it to strict normalized YOLO geometry."""
    x1 = min(max(obj.x1, 0.0), float(width))
    y1 = min(max(obj.y1, 0.0), float(height))
    x2 = min(max(obj.x2, 0.0), float(width))
    y2 = min(max(obj.y2, 0.0), float(height))
    if x2 <= x1 or y2 <= y1:
        return None
    x_center = (x1 + x2) / (2.0 * width)
    y_center = (y1 + y2) / (2.0 * height)
    box_width = (x2 - x1) / width
    box_height = (y2 - y1) / height
    if not (0.0 <= x_center <= 1.0 and 0.0 <= y_center <= 1.0):
        return None
    if not (0.0 < box_width <= 1.0 and 0.0 < box_height <= 1.0):
        return None
    return obj.class_id, x_center, y_center, box_width, box_height


def link_or_copy(source: Path, destination: Path) -> str:
    destination.parent.mkdir(parents=True, exist_ok=True)
    try:
        os.link(source, destination)
        return "hardlink"
    except OSError:
        shutil.copy2(source, destination)
        return "copy"


def write_data_yaml(output_root: Path) -> None:
    output_root.joinpath("data.yaml").write_text(
        "\n".join(
            [
                "train: train/images",
                "val: valid/images",
                "test: test/images",
                "",
                f"nc: {len(VOC_CLASSES)}",
                f"names: {list(VOC_CLASSES)}",
                "",
            ]
        ),
        encoding="utf-8",
    )


def write_label(label_path: Path, boxes: Iterable[tuple[int, float, float, float, float]]) -> None:
    label_path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        f"{class_id} {x_center:.8f} {y_center:.8f} {width:.8f} {height:.8f}"
        for class_id, x_center, y_center, width, height in boxes
    ]
    label_path.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")


def materialize_split(
    voc_root: Path,
    source_split: str,
    output_split: str,
    output_root: Path,
    counters: Counter[str],
    per_class_boxes: Counter[int],
    per_class_difficult: Counter[int],
    per_class_invalid: Counter[int],
) -> list[str]:
    """Link/copy images and emit one strict YOLO label file per official image ID."""
    image_ids = read_split_ids(voc_root, source_split)
    output_images = output_root / output_split / "images"
    output_labels = output_root / output_split / "labels"

    for image_id in image_ids:
        image_path = voc_root / "JPEGImages" / f"{image_id}.jpg"
        annotation_path = voc_root / "Annotations" / f"{image_id}.xml"
        if not image_path.exists() or not annotation_path.exists():
            raise FileNotFoundError(
                f"Official VOC sample is incomplete for ID {image_id}: image={image_path.exists()} annotation={annotation_path.exists()}"
            )

        width, height, objects = parse_annotation(annotation_path)
        output_boxes: list[tuple[int, float, float, float, float]] = []
        for obj in objects:
            if obj.difficult:
                counters["difficult_boxes_excluded"] += 1
                per_class_difficult[obj.class_id] += 1
                continue
            box = normalize_box(obj, width, height)
            if box is None:
                counters["invalid_boxes_excluded"] += 1
                per_class_invalid[obj.class_id] += 1
                continue
            output_boxes.append(box)
            per_class_boxes[box[0]] += 1

        counters[f"materialization_{link_or_copy(image_path, output_images / image_path.name)}"] += 1
        write_label(output_labels / f"{image_id}.txt", output_boxes)

    return image_ids


def validate_output(output_root: Path, split_ids: dict[str, list[str]]) -> dict[str, dict[str, int]]:
    """Validate strict labels, paired files, official counts, and split disjointness."""
    source_sets = {split: set(image_ids) for split, image_ids in split_ids.items()}
    for left, right in (("train", "valid"), ("train", "test"), ("valid", "test")):
        overlap = source_sets[left] & source_sets[right]
        if overlap:
            raise ValueError(f"Official VOC splits overlap between {left} and {right}: {sorted(overlap)[:5]}")

    summary: dict[str, dict[str, int]] = {}
    for split, image_ids in split_ids.items():
        images_dir = output_root / split / "images"
        labels_dir = output_root / split / "labels"
        image_paths = sorted(images_dir.glob("*.jpg"))
        label_paths = sorted(labels_dir.glob("*.txt"))
        image_stems = {path.stem for path in image_paths}
        label_stems = {path.stem for path in label_paths}
        expected_stems = set(image_ids)
        if image_stems != expected_stems or label_stems != expected_stems:
            raise ValueError(
                f"{split}: output image/label pairing does not match its official source IDs "
                f"(images={len(image_stems)}, labels={len(label_stems)}, expected={len(expected_stems)})"
            )
        if len(image_paths) != EXPECTED_SPLIT_COUNTS[split]:
            raise ValueError(
                f"{split}: expected {EXPECTED_SPLIT_COUNTS[split]} official images, found {len(image_paths)}"
            )

        boxes = 0
        empty_labels = 0
        for label_path in label_paths:
            lines = [line for line in label_path.read_text(encoding="utf-8").splitlines() if line.strip()]
            if not lines:
                empty_labels += 1
            for line_number, line in enumerate(lines, start=1):
                fields = line.split()
                if len(fields) != 5:
                    raise ValueError(f"{label_path}:{line_number}: expected five fields")
                class_value, x_center, y_center, width, height = (float(value) for value in fields)
                class_id = int(class_value)
                if class_value != class_id or not 0 <= class_id < len(VOC_CLASSES):
                    raise ValueError(f"{label_path}:{line_number}: invalid class ID")
                if not (0.0 <= x_center <= 1.0 and 0.0 <= y_center <= 1.0):
                    raise ValueError(f"{label_path}:{line_number}: invalid normalized center")
                if not (0.0 < width <= 1.0 and 0.0 < height <= 1.0):
                    raise ValueError(f"{label_path}:{line_number}: invalid normalized size")
                boxes += 1
        summary[split] = {
            "images": len(image_paths),
            "labels": len(label_paths),
            "boxes": boxes,
            "empty_label_files": empty_labels,
        }
    return summary


def archive_metadata(raw_root: Path) -> list[dict[str, object]]:
    metadata: list[dict[str, object]] = []
    for filename, url in ARCHIVES:
        path = raw_root / filename
        if not path.exists():
            raise FileNotFoundError(f"Official archive is missing: {path}")
        metadata.append(
            {
                "filename": filename,
                "source_url": url,
                "bytes": path.stat().st_size,
                "sha256": sha256_file(path),
            }
        )
    return metadata


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Download and convert official PASCAL VOC 2007 splits into strict project YOLO labels."
    )
    parser.add_argument(
        "--raw-root",
        type=Path,
        default=Path("candidate-data/pascal-voc-2007"),
        help="Ignored directory for official archives and extracted VOCdevkit",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("processed-candidate-data/pascal-voc-2007/official"),
        help="Ignored directory receiving strict official-split YOLO labels",
    )
    parser.add_argument(
        "--no-download",
        action="store_true",
        help="Require existing official archives/extraction instead of downloading missing archives",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace an existing converted output root",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.raw_root = args.raw_root.resolve()
    args.output_root = args.output_root.resolve()
    if args.output_root.exists():
        if not args.overwrite:
            raise FileExistsError(
                f"Output root already exists: {args.output_root}. Pass --overwrite to replace it."
            )
        shutil.rmtree(args.output_root)

    voc_root = ensure_source(args)
    args.output_root.mkdir(parents=True, exist_ok=True)
    write_data_yaml(args.output_root)

    counters: Counter[str] = Counter()
    per_class_boxes: Counter[int] = Counter()
    per_class_difficult: Counter[int] = Counter()
    per_class_invalid: Counter[int] = Counter()
    split_ids: dict[str, list[str]] = {}
    for output_split, source_split in SOURCE_SPLITS.items():
        print(f"Converting official VOC split {source_split!r} -> {output_split!r}", flush=True)
        split_ids[output_split] = materialize_split(
            voc_root,
            source_split,
            output_split,
            args.output_root,
            counters,
            per_class_boxes,
            per_class_difficult,
            per_class_invalid,
        )

    validation = validate_output(args.output_root, split_ids)
    summary = {
        "purpose": "PASCAL VOC 2007 official-split project-metric benchmark conversion",
        "benchmark_note": (
            "This conversion preserves official split image IDs but excludes difficult VOC objects. "
            "Project metrics differ from the official VOC leaderboard protocol."
        ),
        "raw_root": str(args.raw_root),
        "voc_root": str(voc_root),
        "output_root": str(args.output_root),
        "archives": archive_metadata(args.raw_root),
        "num_classes": len(VOC_CLASSES),
        "class_names": list(VOC_CLASSES),
        "source_splits": SOURCE_SPLITS,
        "expected_split_counts": EXPECTED_SPLIT_COUNTS,
        "validation": validation,
        "emitted_box_counts": {class_name: per_class_boxes[class_id] for class_id, class_name in enumerate(VOC_CLASSES)},
        "difficult_boxes_excluded": {
            "total": counters["difficult_boxes_excluded"],
            "per_class": {
                class_name: per_class_difficult[class_id] for class_id, class_name in enumerate(VOC_CLASSES)
            },
        },
        "invalid_boxes_excluded": {
            "total": counters["invalid_boxes_excluded"],
            "per_class": {
                class_name: per_class_invalid[class_id] for class_id, class_name in enumerate(VOC_CLASSES)
            },
        },
        "image_materialization": {
            "hardlink": counters["materialization_hardlink"],
            "copy": counters["materialization_copy"],
        },
    }
    summary_path = args.output_root / "voc2007_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")

    print(f"Prepared official-split VOC 2007 dataset at {args.output_root}")
    for split in ("train", "valid", "test"):
        result = validation[split]
        print(
            f"split={split} images={result['images']} labels={result['labels']} "
            f"boxes={result['boxes']} empty_labels={result['empty_label_files']}"
        )
    print(
        f"difficult_excluded={counters['difficult_boxes_excluded']} "
        f"invalid_excluded={counters['invalid_boxes_excluded']} "
        f"materialization={{'hardlink': {counters['materialization_hardlink']}, 'copy': {counters['materialization_copy']}}}"
    )
    print(f"Wrote conversion summary to {summary_path}")


if __name__ == "__main__":
    main()
