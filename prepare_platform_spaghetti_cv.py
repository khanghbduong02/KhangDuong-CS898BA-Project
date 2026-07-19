from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import random
import re
import shutil
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable


VALID_IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}
SOURCE_SPLITS = ("train", "val", "test")
SOURCE_SPLIT_ORDER = {split: index for index, split in enumerate(SOURCE_SPLITS)}
FILENAME_PATTERN = re.compile(r"^(?P<prefix>.+)__(?P<token>[0-9a-fA-F]{8})-frame_(?P<frame>\d+)$")
LABEL_TOLERANCE = 1e-9


@dataclass(frozen=True)
class ImageRecord:
    """One exact-unique source image projected to the Spaghetti-only task."""

    source_split: str
    image_path: Path
    source_prefix: str
    image_hash: str
    boxes: tuple[tuple[int, float, float, float, float], ...]


@dataclass
class GroupRecord:
    """A source-prefix/exact-hash-connected group that cannot cross a fold boundary."""

    group_id: str
    members: list[int]
    source_prefixes: list[str]
    image_count: int
    positive_image_count: int
    box_count: int


@dataclass
class LabelParseStats:
    label_files: int = 0
    total_rows: int = 0
    target_rows: int = 0
    target_rows_clipped_to_image: int = 0
    invalid_non_target_rows: int = 0


class DisjointSet:
    """Union-find for constructing indivisible source-related groups."""

    def __init__(self, size: int) -> None:
        self.parent = list(range(size))
        self.rank = [0] * size

    def find(self, index: int) -> int:
        while self.parent[index] != index:
            self.parent[index] = self.parent[self.parent[index]]
            index = self.parent[index]
        return index

    def union(self, left: int, right: int) -> bool:
        left_root = self.find(left)
        right_root = self.find(right)
        if left_root == right_root:
            return False
        if self.rank[left_root] < self.rank[right_root]:
            left_root, right_root = right_root, left_root
        self.parent[right_root] = left_root
        if self.rank[left_root] == self.rank[right_root]:
            self.rank[left_root] += 1
        return True


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def source_prefix(image_path: Path) -> str:
    """Return the repeated provenance-like component preceding the export token."""
    match = FILENAME_PATTERN.fullmatch(image_path.stem)
    if match is None:
        raise ValueError(
            f"{image_path}: expected '<prefix>__<8-hex-token>-frame_<integer>' source naming"
        )
    return match.group("prefix")


def _parse_source_class_id(value: str, label_path: Path, line_number: int, num_source_classes: int) -> int:
    try:
        class_value = float(value)
    except ValueError as exc:
        raise ValueError(f"{label_path}:{line_number}: class ID is not numeric") from exc
    if not math.isfinite(class_value) or not class_value.is_integer():
        raise ValueError(f"{label_path}:{line_number}: class ID must be a finite integer")
    class_id = int(class_value)
    if not 0 <= class_id < num_source_classes:
        raise ValueError(
            f"{label_path}:{line_number}: class ID {class_id} is outside 0..{num_source_classes - 1}"
        )
    return class_id


def _parse_coordinates(parts: list[str], label_path: Path, line_number: int) -> tuple[float, float, float, float]:
    try:
        x_center, y_center, width, height = (float(value) for value in parts)
    except ValueError as exc:
        raise ValueError(f"{label_path}:{line_number}: label coordinates must be numeric") from exc
    values = (x_center, y_center, width, height)
    if not all(math.isfinite(value) for value in values):
        raise ValueError(f"{label_path}:{line_number}: label coordinates must be finite")
    return values


def _clip_target_box(
    x_center: float,
    y_center: float,
    width: float,
    height: float,
    label_path: Path,
    line_number: int,
) -> tuple[tuple[int, float, float, float, float], bool]:
    """Clip a target box to the image before writing the derived YOLO label."""
    if not (0.0 <= x_center <= 1.0 and 0.0 <= y_center <= 1.0 and 0.0 < width <= 1.0 and 0.0 < height <= 1.0):
        raise ValueError(
            f"{label_path}:{line_number}: target-class normalized centre/size is outside the valid YOLO range"
        )

    x1 = max(0.0, x_center - width / 2.0)
    y1 = max(0.0, y_center - height / 2.0)
    x2 = min(1.0, x_center + width / 2.0)
    y2 = min(1.0, y_center + height / 2.0)
    if x2 - x1 <= LABEL_TOLERANCE or y2 - y1 <= LABEL_TOLERANCE:
        raise ValueError(f"{label_path}:{line_number}: target box has zero area after clipping")

    clipped = (x1, y1, x2, y2) != (
        x_center - width / 2.0,
        y_center - height / 2.0,
        x_center + width / 2.0,
        y_center + height / 2.0,
    )
    return (0, (x1 + x2) / 2.0, (y1 + y2) / 2.0, x2 - x1, y2 - y1), clipped


def parse_target_boxes(
    label_path: Path,
    source_class_id: int,
    num_source_classes: int,
    stats: LabelParseStats,
) -> tuple[tuple[int, float, float, float, float], ...]:
    """Validate every source row and return clipped labels for the selected class only."""
    stats.label_files += 1
    boxes: list[tuple[int, float, float, float, float]] = []

    for line_number, raw_line in enumerate(label_path.read_text(encoding="utf-8").splitlines(), start=1):
        parts = raw_line.strip().split()
        if not parts:
            continue
        stats.total_rows += 1
        if len(parts) != 5:
            raise ValueError(f"{label_path}:{line_number}: expected exactly five YOLO fields")

        class_id = _parse_source_class_id(parts[0], label_path, line_number, num_source_classes)
        x_center, y_center, width, height = _parse_coordinates(parts[1:], label_path, line_number)
        geometry_is_valid = (
            0.0 <= x_center <= 1.0
            and 0.0 <= y_center <= 1.0
            and 0.0 < width <= 1.0
            and 0.0 < height <= 1.0
        )

        if class_id != source_class_id:
            stats.invalid_non_target_rows += int(not geometry_is_valid)
            continue

        target_box, clipped = _clip_target_box(
            x_center,
            y_center,
            width,
            height,
            label_path,
            line_number,
        )
        stats.target_rows += 1
        stats.target_rows_clipped_to_image += int(clipped)
        boxes.append(target_box)

    return tuple(boxes)


def load_records(args: argparse.Namespace) -> tuple[list[ImageRecord], LabelParseStats]:
    stats = LabelParseStats()
    records: list[ImageRecord] = []

    for split in args.splits:
        images_dir = args.input_root / "images" / split
        labels_dir = args.input_root / "labels" / split
        if not images_dir.exists() or not labels_dir.exists():
            raise FileNotFoundError(f"Expected image and label directories for source split {split!r}")

        images = {
            path.stem: path
            for path in sorted(images_dir.iterdir())
            if path.is_file() and path.suffix.lower() in VALID_IMAGE_EXTENSIONS
        }
        labels = {path.stem: path for path in sorted(labels_dir.glob("*.txt"))}
        if set(images) != set(labels):
            missing_labels = sorted(set(images) - set(labels))
            missing_images = sorted(set(labels) - set(images))
            raise ValueError(
                f"Source split {split!r} has unmatched pairs: "
                f"images_without_labels={missing_labels[:5]}, labels_without_images={missing_images[:5]}"
            )

        for stem, image_path in images.items():
            records.append(
                ImageRecord(
                    source_split=split,
                    image_path=image_path,
                    source_prefix=source_prefix(image_path),
                    image_hash=sha256_file(image_path),
                    boxes=parse_target_boxes(
                        labels[stem],
                        source_class_id=args.source_class_id,
                        num_source_classes=args.num_source_classes,
                        stats=stats,
                    ),
                )
            )

    if not records:
        raise RuntimeError(f"No supported images found under {args.input_root}")
    return records, stats


def union_by_key(disjoint_set: DisjointSet, keys: Iterable[str]) -> int:
    indices_by_key: dict[str, list[int]] = defaultdict(list)
    for index, key in enumerate(keys):
        indices_by_key[key].append(index)

    unions = 0
    for indices in indices_by_key.values():
        first = indices[0]
        for index in indices[1:]:
            unions += int(disjoint_set.union(first, index))
    return unions


def build_groups(records: list[ImageRecord]) -> tuple[list[ImageRecord], list[GroupRecord], dict[str, int]]:
    """Deduplicate exact pixels and keep provenance/exact-connected records together."""
    disjoint_set = DisjointSet(len(records))
    prefix_unions = union_by_key(disjoint_set, (record.source_prefix for record in records))
    exact_hash_unions = union_by_key(disjoint_set, (record.image_hash for record in records))

    indices_by_hash: dict[str, list[int]] = defaultdict(list)
    for index, record in enumerate(records):
        indices_by_hash[record.image_hash].append(index)

    canonical_indices: list[int] = []
    for indices in indices_by_hash.values():
        label_signatures = {tuple(sorted(records[index].boxes)) for index in indices}
        if len(label_signatures) != 1:
            examples = [str(records[index].image_path) for index in indices[:3]]
            raise ValueError(f"Exact duplicate images disagree after class filtering: {examples}")
        canonical_indices.append(
            min(
                indices,
                key=lambda index: (
                    SOURCE_SPLIT_ORDER[records[index].source_split],
                    records[index].image_path.name,
                ),
            )
        )

    canonical_indices.sort(key=lambda index: (records[index].source_split, records[index].image_path.name))
    selected_records = [records[index] for index in canonical_indices]

    all_members_by_root: dict[int, list[int]] = defaultdict(list)
    for index in range(len(records)):
        all_members_by_root[disjoint_set.find(index)].append(index)

    selected_indices_by_root: dict[int, list[int]] = defaultdict(list)
    for selected_index, source_index in enumerate(canonical_indices):
        selected_indices_by_root[disjoint_set.find(source_index)].append(selected_index)

    groups: list[GroupRecord] = []
    for root, selected_indices in selected_indices_by_root.items():
        all_source_indices = all_members_by_root[root]
        groups.append(
            GroupRecord(
                group_id="",
                members=sorted(selected_indices),
                source_prefixes=sorted({records[index].source_prefix for index in all_source_indices}),
                image_count=len(selected_indices),
                positive_image_count=sum(bool(selected_records[index].boxes) for index in selected_indices),
                box_count=sum(len(selected_records[index].boxes) for index in selected_indices),
            )
        )

    groups.sort(
        key=lambda group: min(
            f"{selected_records[index].source_split}/{selected_records[index].image_path.name}"
            for index in group.members
        )
    )
    for index, group in enumerate(groups, start=1):
        group.group_id = f"group_{index:04d}"

    diagnostics = {
        "source_records": len(records),
        "exact_unique_records": len(selected_records),
        "exact_duplicates_removed": len(records) - len(selected_records),
        "prefix_unions": prefix_unions,
        "exact_hash_unions": exact_hash_unions,
        "group_count": len(groups),
        "positive_group_count": sum(group.box_count > 0 for group in groups),
    }
    return selected_records, groups, diagnostics


def assignment_objective(
    fold_boxes: list[int],
    fold_positive_images: list[int],
    fold_images: list[int],
    total_boxes: int,
    total_positive_images: int,
    total_images: int,
) -> float:
    fold_count = len(fold_boxes)
    target_boxes = total_boxes / fold_count
    target_positive_images = total_positive_images / fold_count
    target_images = total_images / fold_count

    box_score = sum(((count - target_boxes) / max(target_boxes, 1.0)) ** 2 for count in fold_boxes)
    positive_image_score = sum(
        ((count - target_positive_images) / max(target_positive_images, 1.0)) ** 2
        for count in fold_positive_images
    )
    image_score = sum(((count - target_images) / max(target_images, 1.0)) ** 2 for count in fold_images)
    missing_positive_penalty = 100.0 * sum(count == 0 for count in fold_boxes)
    return box_score + positive_image_score + 0.25 * image_score + missing_positive_penalty


def assign_groups_to_folds(
    groups: list[GroupRecord],
    folds: int,
    seed: int,
    attempts: int,
) -> tuple[list[int], list[int], list[int], list[int], float]:
    if folds < 2:
        raise ValueError("--folds must be at least two")
    if folds > len(groups):
        raise ValueError(f"Cannot create {folds} folds from only {len(groups)} groups")
    if sum(group.box_count > 0 for group in groups) < folds:
        raise ValueError("Too few positive source groups to represent spaghetti in every validation fold")

    total_boxes = sum(group.box_count for group in groups)
    total_positive_images = sum(group.positive_image_count for group in groups)
    total_images = sum(group.image_count for group in groups)
    best_result: tuple[list[int], list[int], list[int], list[int], float] | None = None

    for attempt in range(attempts):
        rng = random.Random(seed + attempt)
        order = list(range(len(groups)))
        rng.shuffle(order)
        order.sort(
            key=lambda index: (
                groups[index].box_count,
                groups[index].positive_image_count,
                groups[index].image_count,
            ),
            reverse=True,
        )

        assignments = [-1] * len(groups)
        fold_boxes = [0] * folds
        fold_positive_images = [0] * folds
        fold_images = [0] * folds

        for group_index in order:
            group = groups[group_index]
            candidates: list[tuple[float, int]] = []
            for fold_index in range(folds):
                fold_boxes[fold_index] += group.box_count
                fold_positive_images[fold_index] += group.positive_image_count
                fold_images[fold_index] += group.image_count
                score = assignment_objective(
                    fold_boxes,
                    fold_positive_images,
                    fold_images,
                    total_boxes,
                    total_positive_images,
                    total_images,
                )
                fold_boxes[fold_index] -= group.box_count
                fold_positive_images[fold_index] -= group.positive_image_count
                fold_images[fold_index] -= group.image_count
                candidates.append((score, fold_index))

            best_score = min(score for score, _ in candidates)
            tied_folds = [fold for score, fold in candidates if abs(score - best_score) < 1e-12]
            selected_fold = rng.choice(tied_folds)
            assignments[group_index] = selected_fold
            fold_boxes[selected_fold] += group.box_count
            fold_positive_images[selected_fold] += group.positive_image_count
            fold_images[selected_fold] += group.image_count

        objective = assignment_objective(
            fold_boxes,
            fold_positive_images,
            fold_images,
            total_boxes,
            total_positive_images,
            total_images,
        )
        result = (assignments, fold_boxes, fold_positive_images, fold_images, objective)
        if best_result is None or result[-1] < best_result[-1]:
            best_result = result

    if best_result is None:
        raise RuntimeError("Unable to allocate groups to folds")
    return best_result


def link_or_copy(source: Path, destination: Path) -> str:
    destination.parent.mkdir(parents=True, exist_ok=True)
    try:
        os.link(source, destination)
        return "hardlink"
    except OSError:
        shutil.copy2(source, destination)
        return "copy"


def write_yolo_label_file(path: Path, boxes: tuple[tuple[int, float, float, float, float], ...]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(f"{class_id} {x:.6f} {y:.6f} {width:.6f} {height:.6f}\n" for class_id, x, y, width, height in boxes),
        encoding="utf-8",
    )


def write_data_yaml(fold_root: Path) -> None:
    fold_root.mkdir(parents=True, exist_ok=True)
    fold_root.joinpath("data.yaml").write_text(
        "train: train/images\nval: valid/images\n\nnc: 1\nnames: ['spaghetti']\n",
        encoding="utf-8",
    )


def validate_output_fold(fold_root: Path) -> None:
    """Assert that the derived labels are strict one-class YOLO labels."""
    for partition in ("train", "valid"):
        images_dir = fold_root / partition / "images"
        labels_dir = fold_root / partition / "labels"
        image_stems = {
            path.stem
            for path in images_dir.iterdir()
            if path.is_file() and path.suffix.lower() in VALID_IMAGE_EXTENSIONS
        }
        label_stems = {path.stem for path in labels_dir.glob("*.txt")}
        if image_stems != label_stems:
            raise RuntimeError(f"{fold_root}: image/label pairing failed in {partition}")
        for label_path in labels_dir.glob("*.txt"):
            for line_number, line in enumerate(label_path.read_text(encoding="utf-8").splitlines(), start=1):
                parts = line.split()
                if len(parts) != 5:
                    raise RuntimeError(f"{label_path}:{line_number}: derived label is not a five-field row")
                class_id, x_center, y_center, width, height = (float(value) for value in parts)
                if class_id != 0.0:
                    raise RuntimeError(f"{label_path}:{line_number}: derived class ID is not zero")
                if not (0.0 <= x_center <= 1.0 and 0.0 <= y_center <= 1.0 and 0.0 < width <= 1.0 and 0.0 < height <= 1.0):
                    raise RuntimeError(f"{label_path}:{line_number}: derived geometry is invalid")


def materialize_folds(
    output_root: Path,
    records: list[ImageRecord],
    groups: list[GroupRecord],
    assignments: list[int],
    fold_boxes: list[int],
    fold_positive_images: list[int],
    fold_images: list[int],
    args: argparse.Namespace,
    diagnostics: dict[str, int],
    label_stats: LabelParseStats,
    objective: float,
) -> Counter[str]:
    group_by_record: dict[int, str] = {}
    assignment_by_group: dict[str, int] = {}
    for group_index, group in enumerate(groups):
        assignment_by_group[group.group_id] = assignments[group_index]
        for record_index in group.members:
            group_by_record[record_index] = group.group_id

    methods: Counter[str] = Counter()
    manifest_path = output_root / "group_manifest.csv"
    with manifest_path.open("w", encoding="utf-8", newline="") as manifest_file:
        writer = csv.DictWriter(
            manifest_file,
            fieldnames=[
                "fold",
                "partition",
                "group_id",
                "source_split",
                "source_prefixes",
                "image_hash",
                "source_image",
                "output_image",
                "spaghetti_boxes",
            ],
        )
        writer.writeheader()

        prefixes_by_group = {group.group_id: group.source_prefixes for group in groups}
        for fold_index in range(args.folds):
            fold_root = output_root / f"fold_{fold_index + 1}"
            write_data_yaml(fold_root)
            for record_index, record in enumerate(records):
                group_id = group_by_record[record_index]
                partition = "valid" if assignment_by_group[group_id] == fold_index else "train"
                output_stem = f"{record.source_split}__{record.image_path.stem}"
                output_image = fold_root / partition / "images" / f"{output_stem}{record.image_path.suffix.lower()}"
                output_label = fold_root / partition / "labels" / f"{output_stem}.txt"
                methods[link_or_copy(record.image_path, output_image)] += 1
                write_yolo_label_file(output_label, record.boxes)
                writer.writerow(
                    {
                        "fold": fold_index + 1,
                        "partition": partition,
                        "group_id": group_id,
                        "source_split": record.source_split,
                        "source_prefixes": " | ".join(prefixes_by_group[group_id]),
                        "image_hash": record.image_hash,
                        "source_image": record.image_path.as_posix(),
                        "output_image": output_image.relative_to(output_root).as_posix(),
                        "spaghetti_boxes": len(record.boxes),
                    }
                )

            validate_output_fold(fold_root)

    total_boxes = sum(len(record.boxes) for record in records)
    total_positive_images = sum(bool(record.boxes) for record in records)
    summary = {
        "purpose": "One-class Spaghetti group-disjoint YOLO26 architecture control",
        "input_root": str(args.input_root),
        "source_splits_combined": list(args.splits),
        "source_class_id": args.source_class_id,
        "derived_class_names": ["spaghetti"],
        "provenance_grouping": "filename prefix before '__', unioned with exact SHA-256 duplicates; no pHash grouping",
        "folds": args.folds,
        "seed": args.seed,
        "attempts": args.attempts,
        "label_parse_stats": asdict(label_stats),
        "group_diagnostics": diagnostics,
        "total_derived_boxes": total_boxes,
        "total_positive_images": total_positive_images,
        "total_negative_images": len(records) - total_positive_images,
        "fold_summaries": [
            {
                "fold": fold_index + 1,
                "valid_images": fold_images[fold_index],
                "valid_positive_images": fold_positive_images[fold_index],
                "valid_negative_images": fold_images[fold_index] - fold_positive_images[fold_index],
                "valid_spaghetti_boxes": fold_boxes[fold_index],
                "train_images": len(records) - fold_images[fold_index],
                "train_positive_images": total_positive_images - fold_positive_images[fold_index],
                "train_negative_images": (len(records) - total_positive_images)
                - (fold_images[fold_index] - fold_positive_images[fold_index]),
                "train_spaghetti_boxes": total_boxes - fold_boxes[fold_index],
            }
            for fold_index in range(args.folds)
        ],
        "assignment_objective": objective,
        "image_materialization": dict(methods),
        "output_validation": "passed",
    }
    (output_root / "cv_summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    return methods


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Create exact-deduplicated, provenance-proxy-group-disjoint one-class Spaghetti folds "
            "from the audited Platform Cam source."
        )
    )
    parser.add_argument(
        "--input-root",
        type=Path,
        default=Path("candidate-data/hf-errors-additive-manufacturing-platform-cam"),
        help="Raw Platform Cam root with images/<split> and labels/<split> directories",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("cv-data/hf-platform-spaghetti-1class"),
        help="Derived group-disjoint one-class fold directory",
    )
    parser.add_argument("--splits", nargs="+", default=list(SOURCE_SPLITS), choices=SOURCE_SPLITS)
    parser.add_argument("--source-class-id", type=int, default=3, help="Platform Cam class ID to keep")
    parser.add_argument("--num-source-classes", type=int, default=9)
    parser.add_argument("--folds", type=int, default=3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--attempts", type=int, default=100)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not args.input_root.exists():
        raise FileNotFoundError(f"Input root not found: {args.input_root}")
    if not 0 <= args.source_class_id < args.num_source_classes:
        raise ValueError("--source-class-id must be within the source class range")
    if args.attempts <= 0:
        raise ValueError("--attempts must be positive")
    if args.output_root.exists():
        if not args.overwrite:
            raise FileExistsError(f"Output root already exists: {args.output_root}. Pass --overwrite to replace it.")
        shutil.rmtree(args.output_root)
    args.output_root.mkdir(parents=True, exist_ok=True)

    source_records, label_stats = load_records(args)
    records, groups, diagnostics = build_groups(source_records)
    assignments, fold_boxes, fold_positive_images, fold_images, objective = assign_groups_to_folds(
        groups,
        folds=args.folds,
        seed=args.seed,
        attempts=args.attempts,
    )
    methods = materialize_folds(
        args.output_root,
        records,
        groups,
        assignments,
        fold_boxes,
        fold_positive_images,
        fold_images,
        args,
        diagnostics,
        label_stats,
        objective,
    )

    print(f"Created {args.folds} one-class Spaghetti folds at {args.output_root}")
    print(
        f"records={diagnostics['source_records']} exact_unique={diagnostics['exact_unique_records']} "
        f"groups={diagnostics['group_count']} positive_groups={diagnostics['positive_group_count']} "
        f"label_stats={asdict(label_stats)} materialization={dict(methods)}"
    )
    for fold_index in range(args.folds):
        print(
            f"fold={fold_index + 1} valid_images={fold_images[fold_index]} "
            f"valid_positive_images={fold_positive_images[fold_index]} "
            f"valid_spaghetti_boxes={fold_boxes[fold_index]}"
        )


if __name__ == "__main__":
    main()