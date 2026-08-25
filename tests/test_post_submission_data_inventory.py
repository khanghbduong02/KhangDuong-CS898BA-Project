from __future__ import annotations

import csv
import json
import tempfile
from pathlib import Path
from types import SimpleNamespace

from prepare_post_submission_data_inventory import (
    HOLDOUT_TEMPLATE_FIELDS,
    build_group_inventory,
    development_registry_rows,
    inventory_rows,
    materialized_label_path,
    read_manifest_records,
    write_csv,
)


CLASS_NAMES = ("spaghetti", "layer_cracking", "over_extrusion", "stringing", "warping")
MANIFEST_FIELDS = (
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
)


def _write_label(path: Path, rows: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(rows) + "\n", encoding="utf-8")


def _write_manifest(path: Path, rows: list[dict[str, str]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as destination:
        writer = csv.DictWriter(destination, fieldnames=MANIFEST_FIELDS)
        writer.writeheader()
        writer.writerows(rows)


def test_inventory_uses_one_canonical_fold_and_prioritizes_rare_groups() -> None:
    """The inventory aggregates strict labels by group and ranks the rarest present class first."""
    with tempfile.TemporaryDirectory() as temporary_directory:
        root = Path(temporary_directory)
        manifest_path = root / "group_manifest.csv"
        fold_root = root / "fold_1"
        _write_label(
            fold_root / "train" / "labels" / "train__a.txt",
            ["0 0.5 0.5 0.2 0.2", "0 0.3 0.3 0.1 0.1"],
        )
        _write_label(
            fold_root / "valid" / "labels" / "valid__b.txt",
            ["4 0.5 0.5 0.2 0.2"],
        )
        _write_label(
            fold_root / "train" / "labels" / "train__c.txt",
            ["0 0.5 0.5 0.2 0.2", "0 0.7 0.7 0.1 0.1"],
        )
        _write_manifest(
            manifest_path,
            [
                {
                    "fold": "1", "partition": "train", "group_id": "group_0001",
                    "source_split": "train", "source_stem": "a", "image_hash": "hash-a",
                    "perceptual_hash": "aaaa", "source_image": "source/a.jpg",
                    "output_image": "fold_1/train/images/train__a.jpg", "class_ids": "0", "num_boxes": "2",
                },
                {
                    "fold": "1", "partition": "valid", "group_id": "group_0002",
                    "source_split": "valid", "source_stem": "b", "image_hash": "hash-b",
                    "perceptual_hash": "bbbb", "source_image": "source/b.jpg",
                    "output_image": "fold_1/valid/images/valid__b.jpg", "class_ids": "4", "num_boxes": "1",
                },
                {
                    "fold": "1", "partition": "train", "group_id": "group_0003",
                    "source_split": "test", "source_stem": "c", "image_hash": "hash-c",
                    "perceptual_hash": "cccc", "source_image": "source/c.jpg",
                    "output_image": "fold_1/train/images/train__c.jpg", "class_ids": "0", "num_boxes": "2",
                },
                {
                    "fold": "2", "partition": "valid", "group_id": "group_0001",
                    "source_split": "train", "source_stem": "a", "image_hash": "hash-a",
                    "perceptual_hash": "aaaa", "source_image": "source/a.jpg",
                    "output_image": "fold_2/valid/images/train__a.jpg", "class_ids": "0", "num_boxes": "2",
                },
            ],
        )

        records = read_manifest_records(manifest_path, fold=1, num_classes=len(CLASS_NAMES))
        inventory, group_counts, box_counts = build_group_inventory(
            records,
            manifest_path,
            num_classes=len(CLASS_NAMES),
        )

        assert len(records) == 3
        assert len(inventory) == 3
        assert group_counts == [2, 0, 0, 0, 1]
        assert box_counts == [4, 0, 0, 0, 1]
        assert inventory[0].group_id == "group_0002"
        assert inventory[0].priority_class_id == 4
        assert materialized_label_path(manifest_path, records[0]) == (
            fold_root / "train" / "labels" / "train__a.txt"
        )

        rows = inventory_rows(inventory, CLASS_NAMES)
        assert rows[0]["priority_class_name"] == "warping"
        assert rows[0]["box_count_4_warping"] == "1"
        registry = development_registry_rows(records)
        assert [row["source_image"] for row in registry] == ["source/a.jpg", "source/b.jpg", "source/c.jpg"]

        template_path = root / "holdout_intake_template.csv"
        write_csv(template_path, [], fieldnames=HOLDOUT_TEMPLATE_FIELDS)
        with template_path.open(encoding="utf-8", newline="") as source:
            assert tuple(csv.DictReader(source).fieldnames or ()) == HOLDOUT_TEMPLATE_FIELDS


def test_inventory_rejects_manifest_label_mismatch() -> None:
    """A stale or inconsistent manifest cannot silently seed a holdout registry."""
    with tempfile.TemporaryDirectory() as temporary_directory:
        root = Path(temporary_directory)
        manifest_path = root / "group_manifest.csv"
        _write_label(root / "fold_1" / "train" / "labels" / "train__a.txt", ["1 0.5 0.5 0.2 0.2"])
        _write_manifest(
            manifest_path,
            [
                {
                    "fold": "1", "partition": "train", "group_id": "group_0001",
                    "source_split": "train", "source_stem": "a", "image_hash": "hash-a",
                    "perceptual_hash": "aaaa", "source_image": "source/a.jpg",
                    "output_image": "fold_1/train/images/train__a.jpg", "class_ids": "0", "num_boxes": "1",
                },
            ],
        )
        records = read_manifest_records(manifest_path, fold=1, num_classes=len(CLASS_NAMES))
        try:
            build_group_inventory(records, manifest_path, num_classes=len(CLASS_NAMES))
        except ValueError as error:
            assert "class_ids" in str(error)
        else:
            raise AssertionError("Expected a manifest-label mismatch to fail")


def main() -> None:
    test_inventory_uses_one_canonical_fold_and_prioritizes_rare_groups()
    print("canonical_inventory_and_priority: passed")
    test_inventory_rejects_manifest_label_mismatch()
    print("manifest_label_mismatch_guard: passed")


if __name__ == "__main__":
    main()
