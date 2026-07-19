from __future__ import annotations

import argparse
import random
import shutil
from collections import Counter
from pathlib import Path


VALID_IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}


def parse_label_class_ids(label_path: Path, num_classes: int) -> set[int]:
    class_ids: set[int] = set()
    for line_number, line in enumerate(label_path.read_text(encoding="utf-8").splitlines(), start=1):
        parts = line.strip().split()
        if not parts:
            continue
        if len(parts) != 5:
            raise ValueError(f"{label_path}:{line_number}: expected normalized five-field YOLO label")

        try:
            class_value = float(parts[0])
        except ValueError as exc:
            raise ValueError(f"{label_path}:{line_number}: class ID must be numeric") from exc

        class_id = int(class_value)
        if class_value != class_id or not 0 <= class_id < num_classes:
            raise ValueError(f"{label_path}:{line_number}: class ID {parts[0]!r} is outside 0..{num_classes - 1}")
        class_ids.add(class_id)

    return class_ids


def find_image_for_label(images_dir: Path, label_path: Path) -> Path:
    for extension in VALID_IMAGE_EXTENSIONS:
        candidate = images_dir / f"{label_path.stem}{extension}"
        if candidate.exists():
            return candidate
    raise FileNotFoundError(f"No image found for label file: {label_path}")


def copy_selected_samples(selected: list[tuple[Path, Path]], output_split: Path) -> None:
    images_dir = output_split / "images"
    labels_dir = output_split / "labels"
    images_dir.mkdir(parents=True, exist_ok=True)
    labels_dir.mkdir(parents=True, exist_ok=True)

    for image_path, label_path in selected:
        shutil.copy2(image_path, images_dir / image_path.name)
        shutil.copy2(label_path, labels_dir / label_path.name)


def write_data_yaml(output_root: Path, class_names: list[str]) -> None:
    output_root.joinpath("data.yaml").write_text(
        "\n".join(
            [
                "train: train/images",
                "val: valid/images",
                "test: valid/images",
                "",
                f"nc: {len(class_names)}",
                f"names: {class_names}",
                "",
            ]
        ),
        encoding="utf-8",
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create a small balanced subset copied into both train and valid for an overfit sanity test."
    )
    parser.add_argument("--source-root", type=Path, default=Path("processed-data/baseline"))
    parser.add_argument("--output-root", type=Path, default=Path("debug-data/overfit-balanced"))
    parser.add_argument("--per-class", type=int, default=24, help="Number of unique source images selected for each class")
    parser.add_argument("--num-classes", type=int, default=3)
    parser.add_argument("--class-names", nargs="+", default=["spaghetti", "stringing", "warping"])
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.per_class <= 0:
        raise ValueError("--per-class must be positive")
    if len(args.class_names) != args.num_classes:
        raise ValueError("--class-names length must match --num-classes")

    source_images = args.source_root / "train" / "images"
    source_labels = args.source_root / "train" / "labels"
    if not source_images.exists() or not source_labels.exists():
        raise FileNotFoundError(f"Expected normalized source train split under {args.source_root}")

    records: list[tuple[Path, Path, set[int]]] = []
    candidates: dict[int, list[int]] = {class_id: [] for class_id in range(args.num_classes)}
    for label_path in sorted(source_labels.glob("*.txt")):
        image_path = find_image_for_label(source_images, label_path)
        class_ids = parse_label_class_ids(label_path, args.num_classes)
        if not class_ids:
            continue
        record_index = len(records)
        records.append((image_path, label_path, class_ids))
        for class_id in class_ids:
            candidates[class_id].append(record_index)

    rng = random.Random(args.seed)
    selected_indices: set[int] = set()
    selected_for_class: dict[int, list[int]] = {class_id: [] for class_id in range(args.num_classes)}

    # Select rarer classes first so they retain access to overlapping multi-class images.
    for class_id in sorted(range(args.num_classes), key=lambda item: len(candidates[item])):
        available = [index for index in candidates[class_id] if index not in selected_indices]
        rng.shuffle(available)
        if len(available) < args.per_class:
            raise ValueError(
                f"Class {class_id} has only {len(available)} remaining unique images; need {args.per_class}. "
                "Reduce --per-class."
            )
        chosen = available[: args.per_class]
        selected_for_class[class_id] = chosen
        selected_indices.update(chosen)

    selected = [(records[index][0], records[index][1]) for index in sorted(selected_indices)]
    if args.output_root.exists():
        shutil.rmtree(args.output_root)
    copy_selected_samples(selected, args.output_root / "train")
    copy_selected_samples(selected, args.output_root / "valid")
    write_data_yaml(args.output_root, args.class_names)

    target_summary = ", ".join(
        f"{args.class_names[class_id]}:{len(selected_for_class[class_id])}" for class_id in range(args.num_classes)
    )
    object_counts: Counter[int] = Counter()
    for _, label_path in selected:
        for class_id in parse_label_class_ids(label_path, args.num_classes):
            object_counts[class_id] += 1
    object_summary = ", ".join(
        f"{args.class_names[class_id]}:{object_counts[class_id]}" for class_id in range(args.num_classes)
    )

    print(f"Created overfit subset at {args.output_root}")
    print(f"unique_images={len(selected)} target_image_assignments=({target_summary})")
    print(f"selected_images_containing_class=({object_summary})")
    print("The same images were copied to train and valid intentionally for the overfit sanity test.")


if __name__ == "__main__":
    main()
