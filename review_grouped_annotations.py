from __future__ import annotations

import argparse
import csv
import html
import json
import re
import shutil
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from yolo_dataset_config import read_yolo_dataset_config


PROJECT_ROOT = Path(__file__).resolve().parent
PALETTE = (
    (0, 180, 255),
    (255, 110, 0),
    (70, 220, 70),
    (220, 70, 220),
    (40, 40, 230),
    (230, 220, 40),
)


@dataclass(frozen=True)
class ManifestRecord:
    """One image row selected from a grouped-CV manifest."""

    group_id: str
    source_stem: str
    source_image: str
    output_image: str
    class_ids: tuple[int, ...]
    num_boxes: int


def project_path(path: Path) -> Path:
    return path if path.is_absolute() else PROJECT_ROOT / path


def parse_class_ids(value: str) -> tuple[int, ...]:
    return tuple(int(item) for item in value.split()) if value.strip() else ()


def derive_label_path(image_path: Path) -> Path:
    """Find labels for both standard split/images and images/split YOLO layouts."""
    candidates = []
    if image_path.parent.name == "images":
        candidates.append(image_path.parent.parent / "labels" / f"{image_path.stem}.txt")
    if image_path.parent.parent.name == "images":
        candidates.append(
            image_path.parent.parent.parent / "labels" / image_path.parent.name / f"{image_path.stem}.txt"
        )
    for path in candidates:
        if path.exists():
            return path
    raise FileNotFoundError(f"Could not derive a label path for {image_path}; tried {candidates}")


def parse_yolo_boxes(label_path: Path, num_classes: int) -> list[tuple[int, float, float, float, float]]:
    boxes: list[tuple[int, float, float, float, float]] = []
    for line_number, line in enumerate(label_path.read_text(encoding="utf-8").splitlines(), start=1):
        fields = line.split()
        if not fields:
            continue
        if len(fields) != 5:
            raise ValueError(f"{label_path}:{line_number}: expected exactly five YOLO fields")
        try:
            class_value, x_center, y_center, width, height = (float(value) for value in fields)
        except ValueError as exc:
            raise ValueError(f"{label_path}:{line_number}: values must be numeric") from exc
        if not class_value.is_integer() or not 0 <= int(class_value) < num_classes:
            raise ValueError(f"{label_path}:{line_number}: invalid class ID {fields[0]!r}")
        if not (0.0 <= x_center <= 1.0 and 0.0 <= y_center <= 1.0 and 0.0 < width <= 1.0 and 0.0 < height <= 1.0):
            raise ValueError(f"{label_path}:{line_number}: invalid normalized bounding box")
        boxes.append((int(class_value), x_center, y_center, width, height))
    return boxes


def draw_boxes(image: np.ndarray, boxes: list[tuple[int, float, float, float, float]], class_names: tuple[str, ...]) -> np.ndarray:
    """Draw normalized YOLO boxes and class labels on a BGR image."""
    annotated = image.copy()
    height, width = annotated.shape[:2]
    line_width = max(1, round(min(height, width) / 320))
    font_scale = max(0.38, min(height, width) / 1100)

    for class_id, x_center, y_center, box_width, box_height in boxes:
        color = PALETTE[class_id % len(PALETTE)]
        x1 = max(0, min(width - 1, round((x_center - box_width / 2.0) * width)))
        y1 = max(0, min(height - 1, round((y_center - box_height / 2.0) * height)))
        x2 = max(0, min(width - 1, round((x_center + box_width / 2.0) * width)))
        y2 = max(0, min(height - 1, round((y_center + box_height / 2.0) * height)))
        cv2.rectangle(annotated, (x1, y1), (x2, y2), color, line_width, cv2.LINE_AA)

        text = f"{class_id}: {class_names[class_id]}"
        (text_width, text_height), baseline = cv2.getTextSize(
            text,
            cv2.FONT_HERSHEY_SIMPLEX,
            font_scale,
            max(1, line_width),
        )
        text_y = y1 if y1 >= text_height + baseline + 4 else min(height - 1, y1 + text_height + baseline + 4)
        cv2.rectangle(
            annotated,
            (x1, max(0, text_y - text_height - baseline - 4)),
            (min(width - 1, x1 + text_width + 4), text_y),
            color,
            thickness=-1,
        )
        cv2.putText(
            annotated,
            text,
            (x1 + 2, text_y - baseline - 2),
            cv2.FONT_HERSHEY_SIMPLEX,
            font_scale,
            (0, 0, 0),
            max(1, line_width),
            cv2.LINE_AA,
        )
    return annotated


def resize_to_width(image: np.ndarray, maximum_width: int) -> np.ndarray:
    height, width = image.shape[:2]
    if width <= maximum_width:
        return image
    scale = maximum_width / width
    return cv2.resize(image, (maximum_width, max(1, round(height * scale))), interpolation=cv2.INTER_AREA)


def image_record_path(manifest_path: Path, record: ManifestRecord) -> Path:
    """Prefer the normalized materialized fold image used in the actual CV experiment."""
    output_path = manifest_path.parent / record.output_image
    if output_path.exists():
        return output_path
    fallback = PROJECT_ROOT / record.source_image
    if fallback.exists():
        return fallback
    raise FileNotFoundError(f"Neither materialized nor source image exists for manifest row: {record}")


def read_records(manifest_path: Path, fold: int) -> dict[str, list[ManifestRecord]]:
    groups: dict[str, list[ManifestRecord]] = defaultdict(list)
    with manifest_path.open(encoding="utf-8", newline="") as source:
        reader = csv.DictReader(source)
        required = {"fold", "group_id", "source_stem", "source_image", "output_image", "class_ids", "num_boxes"}
        if reader.fieldnames is None or not required.issubset(reader.fieldnames):
            raise ValueError(f"{manifest_path}: not a compatible grouped-CV manifest")
        for row in reader:
            if row["fold"] != str(fold):
                continue
            groups[row["group_id"]].append(
                ManifestRecord(
                    group_id=row["group_id"],
                    source_stem=row["source_stem"],
                    source_image=row["source_image"],
                    output_image=row["output_image"],
                    class_ids=parse_class_ids(row["class_ids"]),
                    num_boxes=int(row["num_boxes"]),
                )
            )
    if not groups:
        raise ValueError(f"{manifest_path}: contains no rows for fold {fold}")
    return groups


def group_class_ids(records: list[ManifestRecord]) -> tuple[int, ...]:
    return tuple(sorted({class_id for record in records for class_id in record.class_ids}))


def class_text(class_ids: tuple[int, ...], class_names: tuple[str, ...]) -> str:
    if not class_ids:
        return "negative"
    return ", ".join(f"{class_id}: {class_names[class_id]}" for class_id in class_ids)


def document_page(title: str, body: str) -> str:
    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>{html.escape(title)}</title>
  <style>
    body {{ background: #111827; color: #e5e7eb; font-family: system-ui, sans-serif; margin: 2rem; }}
    a {{ color: #93c5fd; }}
    table {{ border-collapse: collapse; width: 100%; background: #1f2937; }}
    th, td {{ border: 1px solid #4b5563; padding: .5rem; text-align: left; vertical-align: top; }}
    th {{ background: #374151; position: sticky; top: 0; }}
    .grid {{ display: grid; grid-template-columns: repeat(auto-fill, minmax(300px, 1fr)); gap: 1rem; }}
    .card {{ background: #1f2937; border: 1px solid #4b5563; padding: .75rem; }}
    img {{ display: block; width: 100%; height: auto; background: #000; }}
    code {{ color: #d1d5db; overflow-wrap: anywhere; }}
    .muted {{ color: #9ca3af; }}
  </style>
</head>
<body>
{body}
</body>
</html>
"""


def safe_group_directory_name(group_id: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", group_id)


def build_group_page(
    output_dir: Path,
    group_id: str,
    records: list[ManifestRecord],
    manifest_path: Path,
    class_names: tuple[str, ...],
    thumbnail_width: int,
) -> tuple[str, list[dict[str, str]]]:
    """Write annotated overlays and return group-page path plus image-review rows."""
    group_dir = output_dir / "groups" / safe_group_directory_name(group_id)
    image_dir = group_dir / "images"
    image_dir.mkdir(parents=True, exist_ok=True)
    image_rows: list[dict[str, str]] = []
    cards: list[str] = []

    for index, record in enumerate(sorted(records, key=lambda item: item.output_image), start=1):
        image_path = image_record_path(manifest_path, record)
        label_path = derive_label_path(image_path)
        image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
        if image is None:
            raise FileNotFoundError(f"Unable to decode review image: {image_path}")
        boxes = parse_yolo_boxes(label_path, len(class_names))
        overlay = resize_to_width(draw_boxes(image, boxes, class_names), thumbnail_width)
        overlay_name = f"image_{index:03d}.jpg"
        overlay_path = image_dir / overlay_name
        if not cv2.imwrite(str(overlay_path), overlay, [cv2.IMWRITE_JPEG_QUALITY, 92]):
            raise RuntimeError(f"Unable to write annotated review image: {overlay_path}")

        displayed_classes = class_text(tuple(sorted({box[0] for box in boxes})), class_names)
        source_display = html.escape(record.source_image)
        cards.append(
            "<article class='card'>"
            f"<a href='images/{overlay_name}'><img src='images/{overlay_name}' alt='Annotated {html.escape(group_id)} image {index}'></a>"
            f"<p><strong>Classes:</strong> {html.escape(displayed_classes)}<br>"
            f"<strong>Boxes:</strong> {len(boxes)}<br>"
            f"<strong>Source:</strong> <code>{source_display}</code></p>"
            "</article>"
        )
        image_rows.append(
            {
                "group_id": group_id,
                "image_number": str(index),
                "source_image": record.source_image,
                "output_image": record.output_image,
                "overlay_image": (Path("groups") / safe_group_directory_name(group_id) / "images" / overlay_name).as_posix(),
                "classes_in_image": displayed_classes,
                "box_count": str(len(boxes)),
                "image_review_status": "",
                "image_review_notes": "",
            }
        )

    group_classes = group_class_ids(records)
    source_stems = ", ".join(sorted({record.source_stem for record in records}))
    group_page = document_page(
        f"{group_id} annotation review",
        f"<p><a href='../../index.html'>← Back to group index</a></p>"
        f"<h1>{html.escape(group_id)}</h1>"
        f"<p><strong>Classes in group:</strong> {html.escape(class_text(group_classes, class_names))}<br>"
        f"<strong>Source stems:</strong> <code>{html.escape(source_stems)}</code><br>"
        f"<strong>Images:</strong> {len(records)}</p>"
        "<p class='muted'>Open an image to inspect it at full generated-review size. A different defensible box boundary is not, by itself, an error for diffuse defects such as spaghetti or stringing. Record <strong>keep</strong> when the class and coarse defect region are semantically defensible; use <strong>correct</strong> only for a wrong/absent class, an unrelated or missed target region, or a violation of an agreed instance policy. Record <strong>exclude</strong> when no consistent annotation decision can be made.</p>"
        f"<section class='grid'>{''.join(cards)}</section>",
    )
    group_page_path = group_dir / "index.html"
    group_page_path.write_text(group_page, encoding="utf-8")
    return (Path("groups") / safe_group_directory_name(group_id) / "index.html").as_posix(), image_rows


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Generate local annotated HTML review pages for grouped YOLO CV images. "
            "The tool never changes source images or labels."
        )
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path("cv-data/roboflow-3d-print-fail-v1/group_manifest.csv"),
        help="Grouped-CV group_manifest.csv path",
    )
    parser.add_argument("--fold", type=int, default=1, help="Manifest fold to review once per group")
    parser.add_argument(
        "--class-ids",
        nargs="+",
        type=int,
        default=[1, 3, 4],
        help="Classes that select groups for review; defaults to layer cracking, stringing, and warping",
    )
    parser.add_argument("--all-classes", action="store_true", help="Review every group, including majority-only groups")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("review-data/candidate-minority-fold1"),
        help="Local ignored directory for generated HTML, overlays, and review CSVs",
    )
    parser.add_argument("--thumbnail-width", type=int, default=640, help="Maximum width of generated annotated images")
    parser.add_argument("--overwrite", action="store_true", help="Replace an existing review output directory")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.manifest = project_path(args.manifest)
    args.output_dir = project_path(args.output_dir)
    if args.fold <= 0:
        raise ValueError("--fold must be positive")
    if args.thumbnail_width <= 0:
        raise ValueError("--thumbnail-width must be positive")
    if not args.manifest.is_file():
        raise FileNotFoundError(f"Manifest does not exist: {args.manifest}")
    if args.output_dir.exists():
        if not args.overwrite:
            raise FileExistsError(f"Output directory already exists: {args.output_dir}. Pass --overwrite to rebuild it.")
        shutil.rmtree(args.output_dir)

    fold_root = args.manifest.parent / f"fold_{args.fold}"
    dataset_config = read_yolo_dataset_config(fold_root)
    selected_class_ids = set(range(dataset_config.num_classes)) if args.all_classes else set(args.class_ids)
    invalid_class_ids = sorted(class_id for class_id in selected_class_ids if not 0 <= class_id < dataset_config.num_classes)
    if invalid_class_ids:
        raise ValueError(f"Selected class IDs are outside 0..{dataset_config.num_classes - 1}: {invalid_class_ids}")

    all_groups = read_records(args.manifest, args.fold)
    selected_groups = {
        group_id: records
        for group_id, records in all_groups.items()
        if args.all_classes or selected_class_ids.intersection(group_class_ids(records))
    }
    if not selected_groups:
        raise RuntimeError("No groups matched the requested review classes")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    group_rows: list[dict[str, str]] = []
    image_rows: list[dict[str, str]] = []
    table_rows: list[str] = []

    for group_id, records in sorted(selected_groups.items()):
        page_path, rows = build_group_page(
            args.output_dir,
            group_id,
            records,
            args.manifest,
            dataset_config.class_names,
            args.thumbnail_width,
        )
        image_rows.extend(rows)
        all_class_ids = group_class_ids(records)
        target_class_ids = tuple(sorted(selected_class_ids.intersection(all_class_ids)))
        source_stems = " | ".join(sorted({record.source_stem for record in records}))
        group_rows.append(
            {
                "fold": str(args.fold),
                "group_id": group_id,
                "target_classes": class_text(target_class_ids, dataset_config.class_names),
                "all_classes": class_text(all_class_ids, dataset_config.class_names),
                "image_count": str(len(records)),
                "source_stems": source_stems,
                "group_page": page_path,
                "group_review_status": "",
                "group_review_notes": "",
            }
        )
        table_rows.append(
            "<tr>"
            f"<td><a href='{html.escape(page_path)}'>{html.escape(group_id)}</a></td>"
            f"<td>{html.escape(class_text(target_class_ids, dataset_config.class_names))}</td>"
            f"<td>{html.escape(class_text(all_class_ids, dataset_config.class_names))}</td>"
            f"<td>{len(records)}</td>"
            f"<td><code>{html.escape(source_stems)}</code></td>"
            "</tr>"
        )

    with (args.output_dir / "review_groups.csv").open("w", encoding="utf-8", newline="") as destination:
        writer = csv.DictWriter(destination, fieldnames=list(group_rows[0]))
        writer.writeheader()
        writer.writerows(group_rows)
    with (args.output_dir / "review_images.csv").open("w", encoding="utf-8", newline="") as destination:
        writer = csv.DictWriter(destination, fieldnames=list(image_rows[0]))
        writer.writeheader()
        writer.writerows(image_rows)

    selected_text = "all classes" if args.all_classes else class_text(tuple(sorted(selected_class_ids)), dataset_config.class_names)
    summary = {
        "manifest": str(args.manifest),
        "fold": args.fold,
        "dataset_class_names": list(dataset_config.class_names),
        "selected_classes": selected_text,
        "total_groups_in_fold": len(all_groups),
        "review_groups": len(selected_groups),
        "review_images": len(image_rows),
        "output_directory": str(args.output_dir),
        "review_group_csv": "review_groups.csv",
        "review_image_csv": "review_images.csv",
        "index": "index.html",
    }
    (args.output_dir / "review_summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")

    index = document_page(
        "Grouped annotation review",
        "<h1>Grouped annotation review</h1>"
        f"<p><strong>Fold:</strong> {args.fold}<br>"
        f"<strong>Selected classes:</strong> {html.escape(selected_text)}<br>"
        f"<strong>Review groups:</strong> {len(selected_groups)} of {len(all_groups)}<br>"
        f"<strong>Review images:</strong> {len(image_rows)}</p>"
        "<p>Open a group ID to inspect annotated images. Fill <code>review_groups.csv</code> with "
        "<strong>keep</strong>, <strong>correct</strong>, or <strong>exclude</strong>, then explain the decision. "
        "Do not mark a diffuse defect as <strong>correct</strong> merely because another reasonable bounding box could be drawn. Review semantic class validity, coarse region coverage, instance-count consistency, and obvious annotation failures. Generated overlays are review artifacts only and do not modify dataset labels.</p>"
        "<table><thead><tr><th>Group</th><th>Target classes</th><th>All classes</th><th>Images</th><th>Source stems</th></tr></thead>"
        f"<tbody>{''.join(table_rows)}</tbody></table>",
    )
    (args.output_dir / "index.html").write_text(index, encoding="utf-8")

    print(
        f"Generated review package: groups={len(selected_groups)}/{len(all_groups)} images={len(image_rows)} "
        f"classes={selected_text} output={args.output_dir}"
    )
    print(f"Open {args.output_dir / 'index.html'} and record decisions in {args.output_dir / 'review_groups.csv'}")


if __name__ == "__main__":
    main()