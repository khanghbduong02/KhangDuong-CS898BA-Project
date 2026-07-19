from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import random
import shutil
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import cv2
import numpy as np

from preprocess_dataset import VALID_IMAGE_EXTENSIONS, normalize_label_row, write_yolo_label_file


@dataclass(frozen=True)
class ImageRecord:
    """One source image with normalized labels and grouping metadata."""

    source_split: str
    image_path: Path
    label_path: Path
    source_stem: str
    image_hash: str
    perceptual_hash: int
    boxes: tuple[tuple[int, float, float, float, float], ...]
    class_counts: tuple[int, ...]


@dataclass
class GroupRecord:
    """An indivisible group of exact/source/near-duplicate images."""

    group_id: str
    members: list[int]
    class_counts: list[int]
    image_count: int
    source_stems: list[str]


class DisjointSet:
    """Union-find for keeping related images in the same CV fold."""

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


def source_stem(image_path: Path) -> str:
    """Remove Roboflow's export hash while retaining the source-style filename."""
    return image_path.name.split(".rf.", 1)[0]


def perceptual_hash(image_path: Path) -> int:
    """Return a simple 64-bit DCT perceptual hash for conservative near-duplicate grouping."""
    image = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise FileNotFoundError(f"Unable to read image: {image_path}")

    resized = cv2.resize(image, (32, 32), interpolation=cv2.INTER_AREA).astype(np.float32)
    coefficients = cv2.dct(resized)[:8, :8]
    threshold = float(np.median(coefficients.flatten()[1:]))
    result = 0
    for bit in (coefficients > threshold).flatten():
        result = (result << 1) | int(bit)
    return result


def normalize_source_label(
    label_path: Path,
    num_classes: int,
) -> tuple[tuple[tuple[int, float, float, float, float], ...], Counter[str]]:
    boxes: list[tuple[int, float, float, float, float]] = []
    row_types: Counter[str] = Counter()
    for line_number, line in enumerate(label_path.read_text(encoding="utf-8").splitlines(), start=1):
        parts = line.strip().split()
        if not parts:
            continue
        box, row_type, _ = normalize_label_row(parts, label_path, line_number, num_classes)
        boxes.append(box)
        row_types[row_type] += 1
    return tuple(boxes), row_types


def find_label_path(labels_dir: Path, image_path: Path) -> Path:
    label_path = labels_dir / f"{image_path.stem}.txt"
    if not label_path.exists():
        raise FileNotFoundError(f"No label file found for image: {image_path}")
    return label_path


def load_records(args: argparse.Namespace) -> tuple[list[ImageRecord], Counter[str]]:
    records: list[ImageRecord] = []
    normalization_rows: Counter[str] = Counter()

    for split in args.splits:
        split_root = args.input_root / split
        images_dir = split_root / "images"
        labels_dir = split_root / "labels"
        if not images_dir.exists() or not labels_dir.exists():
            raise FileNotFoundError(f"Expected images and labels under {split_root}")

        for image_path in sorted(images_dir.iterdir()):
            if not image_path.is_file() or image_path.suffix.lower() not in VALID_IMAGE_EXTENSIONS:
                continue
            label_path = find_label_path(labels_dir, image_path)
            boxes, row_types = normalize_source_label(label_path, args.num_classes)
            normalization_rows.update(row_types)
            class_counts = [0 for _ in range(args.num_classes)]
            for class_id, _, _, _, _ in boxes:
                class_counts[class_id] += 1

            records.append(
                ImageRecord(
                    source_split=split,
                    image_path=image_path,
                    label_path=label_path,
                    source_stem=source_stem(image_path),
                    image_hash=sha256_file(image_path),
                    perceptual_hash=perceptual_hash(image_path),
                    boxes=boxes,
                    class_counts=tuple(class_counts),
                )
            )

    if not records:
        raise RuntimeError(f"No supported images found under {args.input_root}")
    return records, normalization_rows


def union_by_key(disjoint_set: DisjointSet, keys: Iterable[str]) -> int:
    groups: dict[str, list[int]] = defaultdict(list)
    for index, key in enumerate(keys):
        groups[key].append(index)

    unions = 0
    for indices in groups.values():
        first = indices[0]
        for index in indices[1:]:
            unions += int(disjoint_set.union(first, index))
    return unions


def union_near_duplicates(
    records: list[ImageRecord],
    disjoint_set: DisjointSet,
    distance_threshold: int,
) -> tuple[int, int]:
    """Union pHash-near images using 8-byte buckets before exact Hamming checks."""
    if distance_threshold < 0:
        return 0, 0
    if distance_threshold > 7:
        raise ValueError("--phash-distance must be between -1 and 7")

    buckets: dict[tuple[int, int], list[int]] = defaultdict(list)
    checked_pairs: set[tuple[int, int]] = set()
    comparisons = 0
    unions = 0

    for index, record in enumerate(records):
        for chunk_index in range(8):
            chunk_value = (record.perceptual_hash >> (chunk_index * 8)) & 0xFF
            bucket = buckets[(chunk_index, chunk_value)]
            for other_index in bucket:
                pair = (other_index, index)
                if pair in checked_pairs:
                    continue
                checked_pairs.add(pair)
                comparisons += 1
                distance = (records[other_index].perceptual_hash ^ record.perceptual_hash).bit_count()
                if distance <= distance_threshold:
                    unions += int(disjoint_set.union(other_index, index))
            bucket.append(index)

    return comparisons, unions


def build_groups(
    records: list[ImageRecord],
    num_classes: int,
    phash_distance: int,
) -> tuple[list[GroupRecord], dict[str, int]]:
    disjoint_set = DisjointSet(len(records))
    exact_hash_unions = union_by_key(disjoint_set, (record.image_hash for record in records))
    source_stem_unions = union_by_key(disjoint_set, (record.source_stem for record in records))
    phash_comparisons, phash_unions = union_near_duplicates(records, disjoint_set, phash_distance)

    members_by_root: dict[int, list[int]] = defaultdict(list)
    for index in range(len(records)):
        members_by_root[disjoint_set.find(index)].append(index)

    groups: list[GroupRecord] = []
    for members in members_by_root.values():
        class_counts = [0 for _ in range(num_classes)]
        source_stems = sorted({records[index].source_stem for index in members})
        for index in members:
            for class_id, count in enumerate(records[index].class_counts):
                class_counts[class_id] += count

        group_key = min(
            f"{records[index].source_split}/{records[index].image_path.name}" for index in members
        )
        groups.append(
            GroupRecord(
                group_id=group_key,
                members=sorted(members),
                class_counts=class_counts,
                image_count=len(members),
                source_stems=source_stems,
            )
        )

    groups.sort(key=lambda group: group.group_id)
    for index, group in enumerate(groups, start=1):
        group.group_id = f"group_{index:04d}"

    diagnostics = {
        "exact_hash_unions": exact_hash_unions,
        "source_stem_unions": source_stem_unions,
        "phash_comparisons": phash_comparisons,
        "phash_unions": phash_unions,
    }
    return groups, diagnostics


def assignment_objective(
    fold_counts: list[list[int]],
    fold_images: list[int],
    total_counts: list[int],
    total_images: int,
) -> float:
    fold_count = len(fold_counts)
    target_images = total_images / fold_count
    score = 0.0

    for class_id, total in enumerate(total_counts):
        if total <= 0:
            continue
        target = total / fold_count
        score += sum(((fold[class_id] - target) / max(target, 1.0)) ** 2 for fold in fold_counts)
        score += 20.0 * sum(fold[class_id] == 0 for fold in fold_counts)

    score += 0.25 * sum(((count - target_images) / max(target_images, 1.0)) ** 2 for count in fold_images)
    return score


def assign_groups_to_folds(
    groups: list[GroupRecord],
    num_classes: int,
    folds: int,
    seed: int,
    attempts: int,
) -> tuple[list[int], list[list[int]], list[int], float]:
    if folds < 2:
        raise ValueError("--folds must be at least 2")
    if folds > len(groups):
        raise ValueError(f"Cannot create {folds} folds from only {len(groups)} groups")

    total_counts = [sum(group.class_counts[class_id] for group in groups) for class_id in range(num_classes)]
    for class_id, total in enumerate(total_counts):
        group_count = sum(group.class_counts[class_id] > 0 for group in groups)
        if total == 0 or group_count < folds:
            raise ValueError(
                f"Class {class_id} appears in {group_count} groups; cannot create {folds} folds with every class represented"
            )

    best_result: tuple[list[int], list[list[int]], list[int], float] | None = None
    total_images = sum(group.image_count for group in groups)

    for attempt in range(attempts):
        rng = random.Random(seed + attempt)
        order = list(range(len(groups)))
        rng.shuffle(order)
        order.sort(
            key=lambda index: (
                max(
                    groups[index].class_counts[class_id] / max(total_counts[class_id], 1)
                    for class_id in range(num_classes)
                ),
                sum(groups[index].class_counts),
                groups[index].image_count,
            ),
            reverse=True,
        )

        fold_counts = [[0 for _ in range(num_classes)] for _ in range(folds)]
        fold_images = [0 for _ in range(folds)]
        assignments = [-1 for _ in groups]

        for group_index in order:
            group = groups[group_index]
            candidates: list[tuple[float, int]] = []
            for fold_index in range(folds):
                for class_id, count in enumerate(group.class_counts):
                    fold_counts[fold_index][class_id] += count
                fold_images[fold_index] += group.image_count
                score = assignment_objective(fold_counts, fold_images, total_counts, total_images)
                for class_id, count in enumerate(group.class_counts):
                    fold_counts[fold_index][class_id] -= count
                fold_images[fold_index] -= group.image_count
                candidates.append((score, fold_index))

            best_score = min(score for score, _ in candidates)
            tied_folds = [fold_index for score, fold_index in candidates if abs(score - best_score) < 1e-12]
            fold_index = rng.choice(tied_folds)
            assignments[group_index] = fold_index
            for class_id, count in enumerate(group.class_counts):
                fold_counts[fold_index][class_id] += count
            fold_images[fold_index] += group.image_count

        objective = assignment_objective(fold_counts, fold_images, total_counts, total_images)
        if best_result is None or objective < best_result[3]:
            best_result = (assignments, fold_counts, fold_images, objective)

    if best_result is None:
        raise RuntimeError("Unable to create a fold assignment")
    return best_result


def link_or_copy(source: Path, destination: Path) -> str:
    destination.parent.mkdir(parents=True, exist_ok=True)
    try:
        os.link(source, destination)
        return "hardlink"
    except OSError:
        shutil.copy2(source, destination)
        return "copy"


def write_data_yaml(fold_root: Path, class_names: list[str]) -> None:
    fold_root.mkdir(parents=True, exist_ok=True)
    fold_root.joinpath("data.yaml").write_text(
        "\n".join(
            [
                "train: train/images",
                "val: valid/images",
                "",
                f"nc: {len(class_names)}",
                f"names: {class_names}",
                "",
            ]
        ),
        encoding="utf-8",
    )


def materialize_folds(
    output_root: Path,
    records: list[ImageRecord],
    groups: list[GroupRecord],
    assignments: list[int],
    fold_counts: list[list[int]],
    fold_images: list[int],
    args: argparse.Namespace,
) -> Counter[str]:
    link_methods: Counter[str] = Counter()
    group_to_fold = {group.group_id: assignments[index] for index, group in enumerate(groups)}
    record_to_group: dict[int, str] = {}
    for group in groups:
        for record_index in group.members:
            record_to_group[record_index] = group.group_id

    manifest_path = output_root / "group_manifest.csv"
    with manifest_path.open("w", encoding="utf-8", newline="") as manifest_file:
        writer = csv.DictWriter(
            manifest_file,
            fieldnames=[
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
            ],
        )
        writer.writeheader()

        for fold_index in range(args.folds):
            fold_root = output_root / f"fold_{fold_index + 1}"
            write_data_yaml(fold_root, args.class_names)

            for record_index, record in enumerate(records):
                group_id = record_to_group[record_index]
                partition = "valid" if group_to_fold[group_id] == fold_index else "train"
                output_stem = f"{record.source_split}__{record.image_path.stem}"
                output_image = fold_root / partition / "images" / f"{output_stem}{record.image_path.suffix.lower()}"
                output_label = fold_root / partition / "labels" / f"{output_stem}.txt"
                link_methods[link_or_copy(record.image_path, output_image)] += 1
                output_label.parent.mkdir(parents=True, exist_ok=True)
                write_yolo_label_file(output_label, list(record.boxes))

                writer.writerow(
                    {
                        "fold": fold_index + 1,
                        "partition": partition,
                        "group_id": group_id,
                        "source_split": record.source_split,
                        "source_stem": record.source_stem,
                        "image_hash": record.image_hash,
                        "perceptual_hash": f"{record.perceptual_hash:016x}",
                        "source_image": record.image_path.as_posix(),
                        "output_image": output_image.relative_to(output_root).as_posix(),
                        "class_ids": " ".join(
                            str(class_id) for class_id, count in enumerate(record.class_counts) if count > 0
                        ),
                        "num_boxes": sum(record.class_counts),
                    }
                )

    summary = {
        "input_root": str(args.input_root),
        "input_splits": args.splits,
        "folds": args.folds,
        "seed": args.seed,
        "phash_distance": args.phash_distance,
        "num_classes": args.num_classes,
        "class_names": args.class_names,
        "source_image_records": len(records),
        "group_count": len(groups),
        "fold_summaries": [
            {
                "fold": index + 1,
                "valid_images": fold_images[index],
                "valid_box_counts": fold_counts[index],
                "train_images": len(records) - fold_images[index],
                "train_box_counts": [
                    sum(group.class_counts[class_id] for group in groups) - fold_counts[index][class_id]
                    for class_id in range(args.num_classes)
                ],
            }
            for index in range(args.folds)
        ],
        "image_materialization": dict(link_methods),
    }
    (output_root / "cv_summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    return link_methods


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create normalized, group-disjoint cross-validation folds from a YOLO-style detection dataset."
    )
    parser.add_argument(
        "--input-root",
        type=Path,
        default=Path("candidate-data/roboflow-3d-print-fail-v1"),
        help="Raw dataset root containing train/valid/test splits",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("cv-data/roboflow-3d-print-fail-v1"),
        help="Directory where normalized cross-validation folds are written",
    )
    parser.add_argument(
        "--splits",
        nargs="+",
        default=["train", "valid", "test"],
        help="Source splits to combine into the development pool",
    )
    parser.add_argument("--folds", type=int, default=3, help="Number of group-disjoint folds")
    parser.add_argument("--seed", type=int, default=42, help="Deterministic fold-allocation seed")
    parser.add_argument("--attempts", type=int, default=50, help="Number of greedy fold-allocation attempts")
    parser.add_argument(
        "--phash-distance",
        type=int,
        default=5,
        help="Union perceptual-hash-near images at this Hamming distance; use -1 to disable",
    )
    parser.add_argument("--num-classes", type=int, default=5, help="Number of detection classes")
    parser.add_argument(
        "--class-names",
        nargs="+",
        default=["spaghetti", "layer_cracking", "over_extrusion", "stringing", "warping"],
        help="Canonical class names written to every fold data.yaml",
    )
    parser.add_argument("--overwrite", action="store_true", help="Replace an existing output directory")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.num_classes <= 0:
        raise ValueError("--num-classes must be positive")
    if len(args.class_names) != args.num_classes:
        raise ValueError("--class-names length must match --num-classes")
    if args.attempts <= 0:
        raise ValueError("--attempts must be positive")
    if not args.input_root.exists():
        raise FileNotFoundError(f"Input dataset root not found: {args.input_root}")
    if args.output_root.exists():
        if not args.overwrite:
            raise FileExistsError(f"Output directory already exists: {args.output_root}. Pass --overwrite to replace it.")
        shutil.rmtree(args.output_root)
    args.output_root.mkdir(parents=True, exist_ok=True)

    records, normalization_rows = load_records(args)
    groups, group_diagnostics = build_groups(records, args.num_classes, args.phash_distance)
    assignments, fold_counts, fold_images, objective = assign_groups_to_folds(
        groups,
        args.num_classes,
        args.folds,
        args.seed,
        args.attempts,
    )
    link_methods = materialize_folds(
        args.output_root,
        records,
        groups,
        assignments,
        fold_counts,
        fold_images,
        args,
    )

    print(f"Created {args.folds} group-disjoint folds at {args.output_root}")
    print(
        f"source_records={len(records)} groups={len(groups)} normalization_rows={dict(normalization_rows)} "
        f"grouping={group_diagnostics} objective={objective:.6f} materialization={dict(link_methods)}"
    )
    for fold_index in range(args.folds):
        counts = ", ".join(
            f"class_{class_id}:{fold_counts[fold_index][class_id]}" for class_id in range(args.num_classes)
        )
        print(f"fold={fold_index + 1} valid_images={fold_images[fold_index]} valid_box_counts=({counts})")


if __name__ == "__main__":
    main()
