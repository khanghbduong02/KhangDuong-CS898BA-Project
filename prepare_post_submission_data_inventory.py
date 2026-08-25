"""Build auditable data-governance artifacts for the post-submission study.

The submitted grouped-CV manifest is development data, not a final holdout. This
script reads one canonical fold of that manifest and writes an ignored local
inventory, a rarity-prioritized annotation queue, a development registry, and a
blank holdout-intake template. It never modifies source images, labels, folds,
or checkpoints.
"""
from __future__ import annotations

import argparse
import csv
import json
import shutil
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

from yolo_dataset_config import read_yolo_dataset_config


PROJECT_ROOT = Path(__file__).resolve().parent
REQUIRED_MANIFEST_FIELDS = {
    "fold",
    "partition",
    "group_id",
    "source_split",
    "source_stem",
    "image_hash",
    "perceptual_hash",
    "source_image",
    "output_image",
    "class_ids",
    "num_boxes",
}
HOLDOUT_TEMPLATE_FIELDS = (
    "candidate_id",
    "acquisition_date",
    "source_provider",
    "physical_printer_or_job_id",
    "capture_session_id",
    "camera_or_view_id",
    "source_stem",
    "original_image_path",
    "sha256",
    "perceptual_hash",
    "proposed_group_id",
    "annotator",
    "annotation_status",
    "provenance_review_status",
    "overlap_check_status",
    "holdout_eligibility",
    "review_notes",
)


@dataclass(frozen=True)
class ManifestRecord:
    """One canonical source image from a selected grouped-CV fold."""

    fold: int
    partition: str
    group_id: str
    source_split: str
    source_stem: str
    image_hash: str
    perceptual_hash: str
    source_image: str
    output_image: str
    class_ids: tuple[int, ...]
    num_boxes: int


@dataclass(frozen=True)
class GroupInventoryRecord:
    """Aggregated source-group coverage used for review prioritization."""

    group_id: str
    image_count: int
    unique_hash_count: int
    source_splits: tuple[str, ...]
    source_stems: tuple[str, ...]
    class_box_counts: tuple[int, ...]
    priority_class_id: int | None
    priority_rank: int | None


def project_path(path: Path) -> Path:
    """Resolve a project-relative path without changing the caller's cwd."""
    return path if path.is_absolute() else PROJECT_ROOT / path


def parse_class_ids(value: str, num_classes: int, context: str) -> tuple[int, ...]:
    """Read a manifest's unique per-image class IDs with strict validation."""
    if not value.strip():
        return ()
    class_ids: list[int] = []
    for raw_value in value.split():
        try:
            class_id = int(raw_value)
        except ValueError as exc:
            raise ValueError(f"{context}: class_ids must contain integers") from exc
        if not 0 <= class_id < num_classes:
            raise ValueError(f"{context}: class ID {class_id} is outside 0..{num_classes - 1}")
        if class_id in class_ids:
            raise ValueError(f"{context}: class_ids contains duplicate class ID {class_id}")
        class_ids.append(class_id)
    return tuple(class_ids)


def read_manifest_records(manifest_path: Path, fold: int, num_classes: int) -> list[ManifestRecord]:
    """Read exactly one fold so every development record appears once."""
    records: list[ManifestRecord] = []
    source_images: set[str] = set()
    with manifest_path.open(encoding="utf-8", newline="") as source:
        reader = csv.DictReader(source)
        if reader.fieldnames is None or not REQUIRED_MANIFEST_FIELDS.issubset(reader.fieldnames):
            raise ValueError(f"{manifest_path}: not a compatible grouped-CV manifest")
        for row_number, row in enumerate(reader, start=2):
            if row["fold"] != str(fold):
                continue
            context = f"{manifest_path}:{row_number}"
            try:
                num_boxes = int(row["num_boxes"])
            except ValueError as exc:
                raise ValueError(f"{context}: num_boxes must be an integer") from exc
            if num_boxes < 0:
                raise ValueError(f"{context}: num_boxes cannot be negative")
            source_image = row["source_image"]
            if source_image in source_images:
                raise ValueError(f"{context}: duplicate source image within fold {fold}: {source_image}")
            source_images.add(source_image)
            records.append(
                ManifestRecord(
                    fold=fold,
                    partition=row["partition"],
                    group_id=row["group_id"],
                    source_split=row["source_split"],
                    source_stem=row["source_stem"],
                    image_hash=row["image_hash"],
                    perceptual_hash=row["perceptual_hash"],
                    source_image=source_image,
                    output_image=row["output_image"],
                    class_ids=parse_class_ids(row["class_ids"], num_classes, context),
                    num_boxes=num_boxes,
                )
            )
    if not records:
        raise ValueError(f"{manifest_path}: contains no rows for fold {fold}")
    return records


def materialized_label_path(manifest_path: Path, record: ManifestRecord) -> Path:
    """Return the strict materialized label corresponding to a manifest image row."""
    image_path = manifest_path.parent / record.output_image
    if image_path.parent.name != "images":
        raise ValueError(f"Unexpected materialized image layout: {image_path}")
    return image_path.parent.parent / "labels" / f"{image_path.stem}.txt"


def read_label_class_counts(label_path: Path, num_classes: int) -> tuple[int, ...]:
    """Count strict five-field materialized YOLO labels by foreground class."""
    if not label_path.is_file():
        raise FileNotFoundError(f"Materialized label does not exist: {label_path}")
    counts = [0 for _ in range(num_classes)]
    for line_number, line in enumerate(label_path.read_text(encoding="utf-8").splitlines(), start=1):
        fields = line.split()
        if not fields:
            continue
        if len(fields) != 5:
            raise ValueError(f"{label_path}:{line_number}: expected exactly five YOLO fields")
        try:
            values = tuple(float(value) for value in fields)
        except ValueError as exc:
            raise ValueError(f"{label_path}:{line_number}: values must be numeric") from exc
        class_value, x_center, y_center, width, height = values
        class_id = int(class_value)
        if class_value != class_id or not 0 <= class_id < num_classes:
            raise ValueError(f"{label_path}:{line_number}: invalid class ID {fields[0]!r}")
        if not (0.0 <= x_center <= 1.0 and 0.0 <= y_center <= 1.0 and 0.0 < width <= 1.0 and 0.0 < height <= 1.0):
            raise ValueError(f"{label_path}:{line_number}: invalid normalized bounding box")
        counts[class_id] += 1
    return tuple(counts)


def build_group_inventory(
    records: Iterable[ManifestRecord],
    manifest_path: Path,
    num_classes: int,
) -> tuple[list[GroupInventoryRecord], list[int], list[int]]:
    """Aggregate canonical manifest rows into one inventory record per source group."""
    groups: dict[str, dict[str, object]] = {}
    total_box_counts = [0 for _ in range(num_classes)]
    for record in records:
        label_counts = read_label_class_counts(materialized_label_path(manifest_path, record), num_classes)
        if sum(label_counts) != record.num_boxes:
            raise ValueError(
                f"{record.source_image}: manifest num_boxes={record.num_boxes} does not match "
                f"materialized labels={sum(label_counts)}"
            )
        manifest_class_ids = tuple(index for index, count in enumerate(label_counts) if count > 0)
        if manifest_class_ids != record.class_ids:
            raise ValueError(
                f"{record.source_image}: manifest class_ids={record.class_ids} does not match "
                f"materialized labels={manifest_class_ids}"
            )
        entry = groups.setdefault(
            record.group_id,
            {
                "hashes": set(),
                "source_splits": set(),
                "source_stems": set(),
                "class_box_counts": [0 for _ in range(num_classes)],
                "image_count": 0,
            },
        )
        entry["hashes"].add(record.image_hash)
        entry["source_splits"].add(record.source_split)
        entry["source_stems"].add(record.source_stem)
        entry["image_count"] += 1
        for class_id, count in enumerate(label_counts):
            entry["class_box_counts"][class_id] += count
            total_box_counts[class_id] += count

    group_counts_by_class = [0 for _ in range(num_classes)]
    for entry in groups.values():
        for class_id, count in enumerate(entry["class_box_counts"]):
            if count > 0:
                group_counts_by_class[class_id] += 1

    inventory: list[GroupInventoryRecord] = []
    for group_id, entry in groups.items():
        class_box_counts = tuple(entry["class_box_counts"])
        present_classes = [class_id for class_id, count in enumerate(class_box_counts) if count > 0]
        priority_class_id = (
            min(present_classes, key=lambda class_id: (group_counts_by_class[class_id], class_id))
            if present_classes
            else None
        )
        priority_rank = group_counts_by_class[priority_class_id] if priority_class_id is not None else None
        inventory.append(
            GroupInventoryRecord(
                group_id=group_id,
                image_count=int(entry["image_count"]),
                unique_hash_count=len(entry["hashes"]),
                source_splits=tuple(sorted(entry["source_splits"])),
                source_stems=tuple(sorted(entry["source_stems"])),
                class_box_counts=class_box_counts,
                priority_class_id=priority_class_id,
                priority_rank=priority_rank,
            )
        )
    inventory.sort(key=lambda item: (item.priority_rank is None, item.priority_rank, item.group_id))
    return inventory, group_counts_by_class, total_box_counts


def inventory_rows(
    inventory: Iterable[GroupInventoryRecord],
    class_names: tuple[str, ...],
) -> list[dict[str, str]]:
    """Create portable CSV rows that include group coverage and blank review fields."""
    rows: list[dict[str, str]] = []
    for rank, record in enumerate(inventory, start=1):
        present_class_ids = [class_id for class_id, count in enumerate(record.class_box_counts) if count > 0]
        row = {
            "review_order": str(rank),
            "group_id": record.group_id,
            "image_count": str(record.image_count),
            "unique_hash_count": str(record.unique_hash_count),
            "source_splits": " | ".join(record.source_splits),
            "source_stems": " | ".join(record.source_stems),
            "present_class_ids": " ".join(str(class_id) for class_id in present_class_ids),
            "present_class_names": " | ".join(class_names[class_id] for class_id in present_class_ids),
            "priority_class_id": "" if record.priority_class_id is None else str(record.priority_class_id),
            "priority_class_name": "" if record.priority_class_id is None else class_names[record.priority_class_id],
            "priority_class_group_count": "" if record.priority_rank is None else str(record.priority_rank),
            "group_review_status": "",
            "reviewer": "",
            "review_date": "",
            "group_review_notes": "",
        }
        for class_id, class_name in enumerate(class_names):
            row[f"box_count_{class_id}_{class_name}"] = str(record.class_box_counts[class_id])
        rows.append(row)
    return rows


def development_registry_rows(records: Iterable[ManifestRecord]) -> list[dict[str, str]]:
    """Write every existing development identity for future holdout-overlap checks."""
    return [
        {
            "group_id": record.group_id,
            "source_split": record.source_split,
            "source_stem": record.source_stem,
            "image_hash": record.image_hash,
            "perceptual_hash": record.perceptual_hash,
            "source_image": record.source_image,
            "output_image": record.output_image,
        }
        for record in sorted(records, key=lambda item: (item.group_id, item.source_image))
    ]


def write_csv(path: Path, rows: list[dict[str, str]], fieldnames: Iterable[str] | None = None) -> None:
    """Write a nonempty or header-only CSV deterministically."""
    resolved_fields = list(fieldnames) if fieldnames is not None else list(rows[0])
    with path.open("w", encoding="utf-8", newline="") as destination:
        writer = csv.DictWriter(destination, fieldnames=resolved_fields)
        writer.writeheader()
        writer.writerows(rows)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Create a local post-submission source-group inventory and holdout-intake template "
            "from the existing grouped-CV manifest without changing any data."
        )
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path("cv-data/roboflow-3d-print-fail-v1/group_manifest.csv"),
        help="Existing grouped-CV group_manifest.csv",
    )
    parser.add_argument(
        "--fold",
        type=int,
        default=1,
        help="Canonical fold used once per image/group for inventory generation",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("review-data/post_submission_data_inventory"),
        help="Ignored local directory for inventory and review artifacts",
    )
    parser.add_argument("--overwrite", action="store_true", help="Replace an existing output directory")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.fold <= 0:
        raise ValueError("--fold must be positive")
    args.manifest = project_path(args.manifest)
    args.output_dir = project_path(args.output_dir)
    if not args.manifest.is_file():
        raise FileNotFoundError(f"Manifest does not exist: {args.manifest}")
    if args.output_dir.exists():
        if not args.overwrite:
            raise FileExistsError(f"Output directory exists: {args.output_dir}. Pass --overwrite to rebuild it.")
        shutil.rmtree(args.output_dir)

    fold_root = args.manifest.parent / f"fold_{args.fold}"
    config = read_yolo_dataset_config(fold_root)
    records = read_manifest_records(args.manifest, args.fold, config.num_classes)
    inventory, group_counts_by_class, total_box_counts = build_group_inventory(
        records,
        args.manifest,
        config.num_classes,
    )
    output_rows = inventory_rows(inventory, config.class_names)
    registry_rows = development_registry_rows(records)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(args.output_dir / "source_group_inventory.csv", output_rows)
    write_csv(
        args.output_dir / "development_image_registry.csv",
        registry_rows,
        fieldnames=(
            "group_id",
            "source_split",
            "source_stem",
            "image_hash",
            "perceptual_hash",
            "source_image",
            "output_image",
        ),
    )
    write_csv(args.output_dir / "holdout_intake_template.csv", [], fieldnames=HOLDOUT_TEMPLATE_FIELDS)

    source_split_counts = Counter(record.source_split for record in records)
    partition_counts = Counter(record.partition for record in records)
    summary = {
        "purpose": (
            "Post-submission development-data inventory. Existing groups are development-only and "
            "must not be relabeled as an independent final holdout."
        ),
        "manifest": str(args.manifest),
        "canonical_fold": args.fold,
        "canonical_image_records": len(records),
        "canonical_group_count": len(inventory),
        "class_names": list(config.class_names),
        "source_split_image_counts": dict(sorted(source_split_counts.items())),
        "fold_partition_image_counts": dict(sorted(partition_counts.items())),
        "box_counts_by_class": {
            class_name: total_box_counts[class_id]
            for class_id, class_name in enumerate(config.class_names)
        },
        "group_counts_by_class": {
            class_name: group_counts_by_class[class_id]
            for class_id, class_name in enumerate(config.class_names)
        },
        "artifacts": {
            "source_group_inventory": "source_group_inventory.csv",
            "development_image_registry": "development_image_registry.csv",
            "holdout_intake_template": "holdout_intake_template.csv",
        },
        "holdout_rule": (
            "A future final holdout must use new source groups and pass exact-hash, perceptual-hash, "
            "source-stem, capture-session, and physical-printer/job overlap checks against the development registry."
        ),
    }
    (args.output_dir / "inventory_summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")

    print(
        f"Inventory complete: images={len(records)} groups={len(inventory)} "
        f"output={args.output_dir}"
    )
    print("Groups by class: " + ", ".join(
        f"{class_name}={group_counts_by_class[class_id]}"
        for class_id, class_name in enumerate(config.class_names)
    ))
    print("Review `source_group_inventory.csv` in priority order and register only new source groups in `holdout_intake_template.csv`.")


if __name__ == "__main__":
    main()
