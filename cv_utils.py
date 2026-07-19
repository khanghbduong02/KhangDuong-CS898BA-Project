from __future__ import annotations

import re
from pathlib import Path
from statistics import mean, stdev
from typing import Iterable

from yolo_dataset_config import YoloDatasetConfig, read_yolo_dataset_config


PROJECT_ROOT = Path(__file__).resolve().parent
FOLD_DIRECTORY_PATTERN = re.compile(r"^fold_(\d+)$")
VALID_IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}


def project_path(path: Path) -> Path:
    """Resolve a relative command-line path beneath the repository root."""
    return path if path.is_absolute() else PROJECT_ROOT / path


def discover_folds(data_root: Path, requested_folds: Iterable[int] | None = None) -> list[int]:
    """Return explicit folds or every `fold_<positive integer>` directory."""
    if requested_folds is not None:
        folds = list(dict.fromkeys(requested_folds))
        if not folds or any(fold < 1 for fold in folds):
            raise ValueError("--folds must contain one or more positive integers")
    else:
        folds = sorted(
            int(match.group(1))
            for path in data_root.iterdir()
            if path.is_dir() and (match := FOLD_DIRECTORY_PATTERN.fullmatch(path.name))
        )
        if not folds:
            raise FileNotFoundError(f"No fold_<number> directories found under {data_root}")

    missing = [str(data_root / f"fold_{fold}") for fold in folds if not (data_root / f"fold_{fold}").is_dir()]
    if missing:
        raise FileNotFoundError(f"Requested fold directories do not exist: {missing}")
    return folds


def validate_fold_layout(fold_root: Path, required_splits: tuple[str, ...] = ("train", "valid")) -> YoloDatasetConfig:
    """Check the layout that the project's custom trainer/evaluator require."""
    config = read_yolo_dataset_config(fold_root)
    missing = []
    for split in required_splits:
        for directory in (fold_root / split / "images", fold_root / split / "labels"):
            if not directory.is_dir():
                missing.append(str(directory))
    if missing:
        raise FileNotFoundError(f"Fold layout is incomplete under {fold_root}: {missing}")
    return config


def validate_cv_layout(
    data_root: Path,
    folds: Iterable[int],
    required_splits: tuple[str, ...] = ("train", "valid"),
) -> tuple[dict[int, Path], YoloDatasetConfig]:
    """Validate all folds and ensure they share an identical class taxonomy."""
    fold_roots: dict[int, Path] = {}
    expected_config: YoloDatasetConfig | None = None

    for fold in folds:
        fold_root = data_root / f"fold_{fold}"
        config = validate_fold_layout(fold_root, required_splits=required_splits)
        if expected_config is None:
            expected_config = config
        elif (
            config.num_classes != expected_config.num_classes
            or config.class_names != expected_config.class_names
        ):
            raise ValueError(
                f"Fold {fold} has class metadata {config.num_classes}/{config.class_names}, "
                f"which differs from the first fold {expected_config.num_classes}/{expected_config.class_names}"
            )
        fold_roots[fold] = fold_root

    if expected_config is None:
        raise ValueError("At least one fold is required")
    return fold_roots, expected_config


def render_fold_path(root: Path, template: str, fold: int) -> Path:
    """Render a path template with exactly the `{fold}` field."""
    if "{fold}" not in template:
        raise ValueError("Fold path templates must contain the {fold} placeholder")
    try:
        rendered = Path(template.format(fold=fold))
    except KeyError as exc:
        raise ValueError("Fold path templates may use only the {fold} placeholder") from exc
    return rendered if rendered.is_absolute() else root / rendered


def confidence_tag(value: float) -> str:
    """Create a portable filename fragment for a confidence threshold."""
    return f"{value:.6g}".replace("-", "neg_").replace(".", "p")


def numeric_summary(values: list[float]) -> dict[str, float | int | None]:
    """Return mean and sample standard deviation for one cross-fold metric."""
    if not values:
        raise ValueError("Cannot summarize an empty metric list")
    return {
        "n": len(values),
        "mean": mean(values),
        "sample_sd": stdev(values) if len(values) > 1 else None,
    }