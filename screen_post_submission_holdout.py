"""Screen future holdout candidates against the post-submission development registry.

This tool is intentionally conservative. It never creates a final holdout or
changes data. It checks exact hashes, source stems, and perceptual-hash proximity
against every existing development image, and requires documented new-source
provenance before a candidate can enter the annotation queue.
"""
from __future__ import annotations

import argparse
import csv
import json
import re
import shutil
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

from prepare_post_submission_data_inventory import HOLDOUT_TEMPLATE_FIELDS


PROJECT_ROOT = Path(__file__).resolve().parent
DEVELOPMENT_REGISTRY_FIELDS = (
    "group_id",
    "source_split",
    "source_stem",
    "image_hash",
    "perceptual_hash",
    "source_image",
    "output_image",
)
SHA256_PATTERN = re.compile(r"^[0-9a-fA-F]{64}$")
PERCEPTUAL_HASH_PATTERN = re.compile(r"^[0-9a-fA-F]{16}$")
CONFIRMED_NEW_PROVENANCE = "confirmed_new"
SCREENING_FIELDS = (
    "development_exact_hash_matches",
    "development_source_stem_matches",
    "development_near_phash_matches",
    "candidate_duplicate_hash_ids",
    "candidate_source_stem_group_conflict",
    "identity_screen_status",
    "holdout_eligibility_result",
    "screening_reason",
)


@dataclass(frozen=True)
class DevelopmentRecord:
    group_id: str
    source_stem: str
    image_hash: str
    perceptual_hash: int
    source_image: str


@dataclass(frozen=True)
class CandidateRecord:
    row_number: int
    values: dict[str, str]
    source_stem_key: str
    image_hash_key: str
    perceptual_hash: int


def project_path(path: Path) -> Path:
    """Resolve a project-relative path without relying on the caller's cwd."""
    return path if path.is_absolute() else PROJECT_ROOT / path


def parse_perceptual_hash(value: str, context: str) -> int:
    normalized = value.strip().lower()
    if not PERCEPTUAL_HASH_PATTERN.fullmatch(normalized):
        raise ValueError(f"{context}: perceptual_hash must be exactly 16 hexadecimal characters")
    return int(normalized, 16)


def normalized_source_stem(value: str) -> str:
    return value.strip().casefold()


def read_development_registry(path: Path) -> list[DevelopmentRecord]:
    records: list[DevelopmentRecord] = []
    with path.open(encoding="utf-8", newline="") as source:
        reader = csv.DictReader(source)
        if reader.fieldnames is None or not set(DEVELOPMENT_REGISTRY_FIELDS).issubset(reader.fieldnames):
            raise ValueError(f"{path}: not a compatible development registry")
        for row_number, row in enumerate(reader, start=2):
            context = f"{path}:{row_number}"
            image_hash = row["image_hash"].strip().lower()
            if not SHA256_PATTERN.fullmatch(image_hash):
                raise ValueError(f"{context}: image_hash must be a SHA-256 hex digest")
            source_stem = row["source_stem"].strip()
            if not source_stem:
                raise ValueError(f"{context}: source_stem cannot be blank")
            records.append(
                DevelopmentRecord(
                    group_id=row["group_id"].strip(),
                    source_stem=source_stem,
                    image_hash=image_hash,
                    perceptual_hash=parse_perceptual_hash(row["perceptual_hash"], context),
                    source_image=row["source_image"].strip(),
                )
            )
    if not records:
        raise ValueError(f"{path}: development registry is empty")
    return records


def read_holdout_candidates(path: Path) -> list[CandidateRecord]:
    candidates: list[CandidateRecord] = []
    with path.open(encoding="utf-8", newline="") as source:
        reader = csv.DictReader(source)
        if reader.fieldnames is None or not set(HOLDOUT_TEMPLATE_FIELDS).issubset(reader.fieldnames):
            raise ValueError(
                f"{path}: holdout intake must contain every field in HOLDOUT_TEMPLATE_FIELDS"
            )
        for row_number, row in enumerate(reader, start=2):
            values = {field: row.get(field, "").strip() for field in HOLDOUT_TEMPLATE_FIELDS}
            context = f"{path}:{row_number}"
            image_hash = values["sha256"].lower()
            if not SHA256_PATTERN.fullmatch(image_hash):
                raise ValueError(f"{context}: sha256 must be a SHA-256 hex digest")
            if not values["candidate_id"]:
                raise ValueError(f"{context}: candidate_id cannot be blank")
            if not values["source_stem"]:
                raise ValueError(f"{context}: source_stem cannot be blank")
            if not values["proposed_group_id"]:
                raise ValueError(f"{context}: proposed_group_id cannot be blank")
            candidates.append(
                CandidateRecord(
                    row_number=row_number,
                    values=values,
                    source_stem_key=normalized_source_stem(values["source_stem"]),
                    image_hash_key=image_hash,
                    perceptual_hash=parse_perceptual_hash(values["perceptual_hash"], context),
                )
            )
    return candidates


def hamming_distance(left: int, right: int) -> int:
    return (left ^ right).bit_count()


def build_indices(records: Iterable[DevelopmentRecord]) -> tuple[dict[str, list[DevelopmentRecord]], dict[str, list[DevelopmentRecord]], set[str]]:
    hashes: dict[str, list[DevelopmentRecord]] = defaultdict(list)
    stems: dict[str, list[DevelopmentRecord]] = defaultdict(list)
    groups: set[str] = set()
    for record in records:
        hashes[record.image_hash].append(record)
        stems[normalized_source_stem(record.source_stem)].append(record)
        groups.add(record.group_id)
    return hashes, stems, groups


def candidate_indices(candidates: Iterable[CandidateRecord]) -> tuple[dict[str, list[CandidateRecord]], dict[str, set[str]], set[str]]:
    hashes: dict[str, list[CandidateRecord]] = defaultdict(list)
    stems_to_groups: dict[str, set[str]] = defaultdict(set)
    candidate_ids: set[str] = set()
    for candidate in candidates:
        candidate_id = candidate.values["candidate_id"]
        if candidate_id in candidate_ids:
            raise ValueError(f"Duplicate candidate_id: {candidate_id}")
        candidate_ids.add(candidate_id)
        hashes[candidate.image_hash_key].append(candidate)
        stems_to_groups[candidate.source_stem_key].add(candidate.values["proposed_group_id"])
    return hashes, stems_to_groups, candidate_ids


def format_development_matches(records: Iterable[DevelopmentRecord]) -> str:
    return " | ".join(
        f"{record.group_id}:{record.source_image}"
        for record in sorted(records, key=lambda item: (item.group_id, item.source_image))
    )


def screen_candidates(
    candidates: Iterable[CandidateRecord],
    development_records: Iterable[DevelopmentRecord],
    phash_distance: int,
) -> tuple[list[dict[str, str]], dict[str, int]]:
    """Screen candidate identities without asserting that their labels are final."""
    if not 0 <= phash_distance <= 64:
        raise ValueError("phash_distance must be between 0 and 64")
    candidate_list = list(candidates)
    development_list = list(development_records)
    development_hashes, development_stems, development_groups = build_indices(development_list)
    candidate_hashes, candidate_stem_groups, _ = candidate_indices(candidate_list)

    results: list[dict[str, str]] = []
    summary = defaultdict(int)
    for candidate in candidate_list:
        values = dict(candidate.values)
        exact_matches = development_hashes.get(candidate.image_hash_key, [])
        stem_matches = development_stems.get(candidate.source_stem_key, [])
        near_matches = [
            record
            for record in development_list
            if hamming_distance(candidate.perceptual_hash, record.perceptual_hash) <= phash_distance
        ]
        duplicate_ids = [
            item.values["candidate_id"]
            for item in candidate_hashes[candidate.image_hash_key]
            if item.values["candidate_id"] != candidate.values["candidate_id"]
        ]
        stem_group_conflict = len(candidate_stem_groups[candidate.source_stem_key]) > 1
        reasons: list[str] = []
        if exact_matches:
            reasons.append("exact SHA-256 overlap with development registry")
        if stem_matches:
            reasons.append("source-stem overlap with development registry")
        if near_matches:
            reasons.append(f"perceptual-hash overlap at distance <= {phash_distance}")
        if candidate.values["proposed_group_id"] in development_groups:
            reasons.append("proposed group ID already exists in development registry")
        if duplicate_ids:
            reasons.append("duplicate candidate SHA-256 in intake")
        if stem_group_conflict:
            reasons.append("same candidate source stem is assigned to multiple proposed groups")
        required_provenance = (
            candidate.values["source_provider"],
            candidate.values["physical_printer_or_job_id"],
            candidate.values["capture_session_id"],
        )
        if not all(required_provenance):
            reasons.append("missing source-provider, printer/job, or capture-session provenance")
        if candidate.values["provenance_review_status"].casefold() != CONFIRMED_NEW_PROVENANCE:
            reasons.append("provenance_review_status is not confirmed_new")

        if reasons:
            identity_status = "blocked"
            eligibility = "not_eligible"
        elif candidate.values["annotation_status"].casefold() == "approved":
            identity_status = "clear"
            eligibility = "eligible_for_holdout_pool"
        else:
            identity_status = "clear"
            eligibility = "pending_annotation_approval"

        output = {
            **values,
            "development_exact_hash_matches": format_development_matches(exact_matches),
            "development_source_stem_matches": format_development_matches(stem_matches),
            "development_near_phash_matches": format_development_matches(near_matches),
            "candidate_duplicate_hash_ids": " | ".join(sorted(duplicate_ids)),
            "candidate_source_stem_group_conflict": str(stem_group_conflict).lower(),
            "identity_screen_status": identity_status,
            "holdout_eligibility_result": eligibility,
            "screening_reason": " | ".join(reasons),
        }
        results.append(output)
        summary[identity_status] += 1
        summary[eligibility] += 1
    return results, dict(sorted(summary.items()))


def write_csv(path: Path, rows: list[dict[str, str]]) -> None:
    fieldnames = list(HOLDOUT_TEMPLATE_FIELDS) + list(SCREENING_FIELDS)
    with path.open("w", encoding="utf-8", newline="") as destination:
        writer = csv.DictWriter(destination, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Screen future holdout-intake candidates against the immutable post-submission development registry. "
            "The tool does not copy images, modify labels, or create a final split."
        )
    )
    parser.add_argument(
        "--development-registry",
        type=Path,
        default=Path("review-data/post_submission_data_inventory/development_image_registry.csv"),
        help="CSV generated by prepare_post_submission_data_inventory.py",
    )
    parser.add_argument(
        "--holdout-intake",
        type=Path,
        required=True,
        help="Filled holdout_intake_template.csv with future candidate identities and provenance",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("review-data/post_submission_holdout_screen"),
        help="Ignored local directory for screening CSV and summary JSON",
    )
    parser.add_argument(
        "--phash-distance",
        type=int,
        default=5,
        help="Maximum perceptual-hash Hamming distance treated as a conservative development overlap",
    )
    parser.add_argument("--overwrite", action="store_true", help="Replace an existing output directory")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.development_registry = project_path(args.development_registry)
    args.holdout_intake = project_path(args.holdout_intake)
    args.output_dir = project_path(args.output_dir)
    if not args.development_registry.is_file():
        raise FileNotFoundError(f"Development registry does not exist: {args.development_registry}")
    if not args.holdout_intake.is_file():
        raise FileNotFoundError(f"Holdout intake does not exist: {args.holdout_intake}")
    if args.output_dir.exists():
        if not args.overwrite:
            raise FileExistsError(f"Output directory exists: {args.output_dir}. Pass --overwrite to rebuild it.")
        shutil.rmtree(args.output_dir)

    development_records = read_development_registry(args.development_registry)
    candidates = read_holdout_candidates(args.holdout_intake)
    results, counts = screen_candidates(candidates, development_records, args.phash_distance)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(args.output_dir / "holdout_screening.csv", results)
    summary = {
        "purpose": (
            "Conservative future-holdout identity screening against the existing development registry. "
            "A clear result is not an independent test claim until annotation approval and final split freezing."
        ),
        "development_registry": str(args.development_registry),
        "holdout_intake": str(args.holdout_intake),
        "development_image_count": len(development_records),
        "candidate_count": len(candidates),
        "phash_distance": args.phash_distance,
        "result_counts": counts,
        "artifacts": {"screening_csv": "holdout_screening.csv"},
    }
    (args.output_dir / "screening_summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(
        f"Holdout screening complete: candidates={len(candidates)} "
        f"clear={counts.get('clear', 0)} blocked={counts.get('blocked', 0)} output={args.output_dir}"
    )


if __name__ == "__main__":
    main()
