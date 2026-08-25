from __future__ import annotations

import csv
import tempfile
from pathlib import Path

from prepare_post_submission_data_inventory import HOLDOUT_TEMPLATE_FIELDS
from screen_post_submission_holdout import (
    DEVELOPMENT_REGISTRY_FIELDS,
    read_development_registry,
    read_holdout_candidates,
    screen_candidates,
    write_csv,
)


HASH_A = "a" * 64
HASH_B = "b" * 64
HASH_C = "c" * 64
HASH_D = "d" * 64


def _write_csv(path: Path, fieldnames: tuple[str, ...], rows: list[dict[str, str]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as destination:
        writer = csv.DictWriter(destination, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _candidate(
    candidate_id: str,
    sha256: str,
    perceptual_hash: str,
    source_stem: str,
    proposed_group_id: str,
    *,
    annotation_status: str = "approved",
    provenance_review_status: str = "confirmed_new",
) -> dict[str, str]:
    return {
        "candidate_id": candidate_id,
        "acquisition_date": "2026-08-12",
        "source_provider": "new_provider",
        "physical_printer_or_job_id": "job-new",
        "capture_session_id": "session-new",
        "camera_or_view_id": "camera-a",
        "source_stem": source_stem,
        "original_image_path": f"incoming/{candidate_id}.jpg",
        "sha256": sha256,
        "perceptual_hash": perceptual_hash,
        "proposed_group_id": proposed_group_id,
        "annotator": "reviewer",
        "annotation_status": annotation_status,
        "provenance_review_status": provenance_review_status,
        "overlap_check_status": "",
        "holdout_eligibility": "",
        "review_notes": "",
    }


def test_holdout_screen_blocks_development_identity_overlap() -> None:
    """Exact hashes, source stems, and close pHashes cannot enter a fresh holdout."""
    with tempfile.TemporaryDirectory() as temporary_directory:
        root = Path(temporary_directory)
        registry_path = root / "development_image_registry.csv"
        intake_path = root / "holdout_intake.csv"
        _write_csv(
            registry_path,
            DEVELOPMENT_REGISTRY_FIELDS,
            [
                {
                    "group_id": "group_0001",
                    "source_split": "train",
                    "source_stem": "existing_stem",
                    "image_hash": HASH_A,
                    "perceptual_hash": "0000000000000000",
                    "source_image": "development/a.jpg",
                    "output_image": "fold_1/train/images/a.jpg",
                },
                {
                    "group_id": "group_0002",
                    "source_split": "valid",
                    "source_stem": "other_stem",
                    "image_hash": HASH_B,
                    "perceptual_hash": "ffffffffffffffff",
                    "source_image": "development/b.jpg",
                    "output_image": "fold_1/valid/images/b.jpg",
                },
            ],
        )
        _write_csv(
            intake_path,
            HOLDOUT_TEMPLATE_FIELDS,
            [
                _candidate("exact", HASH_A, "1111111111111111", "new_exact", "new_group_exact"),
                _candidate("stem", HASH_C, "1111111111111111", "existing_stem", "new_group_stem"),
                _candidate("phash", HASH_D, "0000000000000001", "new_phash", "new_group_phash"),
                _candidate("clean", "e" * 64, "1111111111111111", "new_clean", "new_group_clean"),
            ],
        )

        development_records = read_development_registry(registry_path)
        candidates = read_holdout_candidates(intake_path)
        results, counts = screen_candidates(candidates, development_records, phash_distance=1)
        by_id = {row["candidate_id"]: row for row in results}

        assert counts["blocked"] == 3
        assert counts["clear"] == 1
        assert by_id["exact"]["identity_screen_status"] == "blocked"
        assert "exact SHA-256" in by_id["exact"]["screening_reason"]
        assert by_id["stem"]["identity_screen_status"] == "blocked"
        assert "source-stem overlap" in by_id["stem"]["screening_reason"]
        assert by_id["phash"]["identity_screen_status"] == "blocked"
        assert "perceptual-hash overlap" in by_id["phash"]["screening_reason"]
        assert by_id["clean"]["identity_screen_status"] == "clear"
        assert by_id["clean"]["holdout_eligibility_result"] == "eligible_for_holdout_pool"


def test_holdout_screen_requires_new_source_provenance_and_consistent_grouping() -> None:
    """A collision-free candidate stays ineligible without provenance or stable group assignment."""
    with tempfile.TemporaryDirectory() as temporary_directory:
        root = Path(temporary_directory)
        registry_path = root / "development_image_registry.csv"
        intake_path = root / "holdout_intake.csv"
        _write_csv(
            registry_path,
            DEVELOPMENT_REGISTRY_FIELDS,
            [
                {
                    "group_id": "group_0001",
                    "source_split": "train",
                    "source_stem": "development",
                    "image_hash": HASH_A,
                    "perceptual_hash": "0000000000000000",
                    "source_image": "development/a.jpg",
                    "output_image": "fold_1/train/images/a.jpg",
                },
            ],
        )
        missing_provenance = _candidate(
            "missing_provenance",
            HASH_C,
            "1111111111111111",
            "new_stem",
            "new_group_one",
            provenance_review_status="pending",
        )
        missing_provenance["physical_printer_or_job_id"] = ""
        _write_csv(
            intake_path,
            HOLDOUT_TEMPLATE_FIELDS,
            [
                missing_provenance,
                _candidate("conflict_one", HASH_D, "2222222222222222", "shared_new_stem", "new_group_two"),
                _candidate("conflict_two", "e" * 64, "3333333333333333", "shared_new_stem", "new_group_three"),
            ],
        )

        results, counts = screen_candidates(
            read_holdout_candidates(intake_path),
            read_development_registry(registry_path),
            phash_distance=0,
        )
        by_id = {row["candidate_id"]: row for row in results}

        assert counts["blocked"] == 3
        assert "missing source-provider, printer/job, or capture-session provenance" in by_id["missing_provenance"]["screening_reason"]
        assert "provenance_review_status is not confirmed_new" in by_id["missing_provenance"]["screening_reason"]
        assert by_id["conflict_one"]["candidate_source_stem_group_conflict"] == "true"
        assert by_id["conflict_two"]["candidate_source_stem_group_conflict"] == "true"


def test_holdout_screen_csv_includes_all_intake_and_screening_fields() -> None:
    """Screening output remains a reviewable superset of the user-maintained intake."""
    with tempfile.TemporaryDirectory() as temporary_directory:
        output_path = Path(temporary_directory) / "holdout_screening.csv"
        row = _candidate("candidate", HASH_C, "1111111111111111", "new", "new_group")
        row.update(
            {
                "development_exact_hash_matches": "",
                "development_source_stem_matches": "",
                "development_near_phash_matches": "",
                "candidate_duplicate_hash_ids": "",
                "candidate_source_stem_group_conflict": "false",
                "identity_screen_status": "clear",
                "holdout_eligibility_result": "eligible_for_holdout_pool",
                "screening_reason": "",
            }
        )
        write_csv(output_path, [row])
        with output_path.open(encoding="utf-8", newline="") as source:
            reader = csv.DictReader(source)
            assert set(HOLDOUT_TEMPLATE_FIELDS).issubset(reader.fieldnames or ())
            result = next(reader)
            assert result["identity_screen_status"] == "clear"


def main() -> None:
    test_holdout_screen_blocks_development_identity_overlap()
    print("development_overlap_guard: passed")
    test_holdout_screen_requires_new_source_provenance_and_consistent_grouping()
    print("provenance_and_grouping_guard: passed")
    test_holdout_screen_csv_includes_all_intake_and_screening_fields()
    print("screening_csv_contract: passed")


if __name__ == "__main__":
    main()
