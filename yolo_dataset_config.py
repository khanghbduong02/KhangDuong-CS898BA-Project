from __future__ import annotations

import ast
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class YoloDatasetConfig:
    """Class metadata read from a standard YOLO `data.yaml` file."""

    root: Path
    num_classes: int
    class_names: tuple[str, ...]


def _parse_names_value(value: str) -> tuple[str, ...] | None:
    """Parse inline YOLO names written as a Python-style list or mapping."""
    if not value:
        return None
    try:
        parsed = ast.literal_eval(value)
    except (SyntaxError, ValueError):
        parsed = None

    if isinstance(parsed, list):
        return tuple(str(name) for name in parsed)
    if isinstance(parsed, dict):
        try:
            normalized = {int(index): str(name) for index, name in parsed.items()}
            indices = sorted(normalized)
            if indices != list(range(len(indices))):
                return None
            return tuple(normalized[index] for index in indices)
        except (KeyError, TypeError, ValueError):
            return None

    # YAML permits unquoted values such as `names: [cat, dog]` and
    # `names: {0: cat, 1: dog}`, which `ast.literal_eval` deliberately
    # rejects. Handle these simple, common YOLO forms without introducing a
    # PyYAML dependency into the core custom-training path.
    if value.startswith("[") and value.endswith("]"):
        entries = [entry.strip().strip("'\"") for entry in value[1:-1].split(",")]
        return tuple(entry for entry in entries if entry)
    if value.startswith("{") and value.endswith("}"):
        parsed_mapping: dict[int, str] = {}
        try:
            for entry in value[1:-1].split(","):
                key, name = entry.split(":", 1)
                parsed_mapping[int(key.strip().strip("'\""))] = name.strip().strip("'\"")
            indices = sorted(parsed_mapping)
            if indices != list(range(len(indices))):
                return None
            return tuple(parsed_mapping[index] for index in indices)
        except (ValueError, KeyError):
            return None
    return None


def read_yolo_dataset_config(data_root: Path) -> YoloDatasetConfig:
    """Read `nc` and `names` without adding a PyYAML dependency.

    The project writes inline list-style names, while the fallback parser also
    accepts a basic indented numeric mapping commonly emitted by YOLO tools.
    """
    data_yaml = data_root / "data.yaml"
    if not data_yaml.exists():
        raise FileNotFoundError(f"YOLO data configuration not found: {data_yaml}")

    num_classes: int | None = None
    names: tuple[str, ...] | None = None
    indented_names: dict[int, str] = {}
    reading_indented_names = False

    for raw_line in data_yaml.read_text(encoding="utf-8").splitlines():
        stripped = raw_line.strip()
        if not stripped or stripped.startswith("#"):
            continue

        if raw_line[:1].isspace() and reading_indented_names and ":" in stripped:
            key, value = stripped.split(":", 1)
            try:
                indented_names[int(key.strip())] = value.strip().strip("'\"")
            except ValueError:
                pass
            continue

        reading_indented_names = False
        if ":" not in stripped:
            continue

        key, value = (part.strip() for part in stripped.split(":", 1))
        if key == "nc":
            try:
                num_classes = int(value)
            except ValueError as exc:
                raise ValueError(f"{data_yaml}: 'nc' must be an integer") from exc
        elif key == "names":
            names = _parse_names_value(value)
            reading_indented_names = not value

    if names is None and indented_names:
        expected_ids = list(range(max(indented_names) + 1))
        if sorted(indented_names) != expected_ids:
            raise ValueError(f"{data_yaml}: indented names must use contiguous IDs starting at zero")
        names = tuple(indented_names[index] for index in expected_ids)

    if num_classes is None:
        if names is None:
            raise ValueError(f"{data_yaml}: unable to read either 'nc' or 'names'")
        num_classes = len(names)
    if num_classes <= 0:
        raise ValueError(f"{data_yaml}: 'nc' must be positive")

    if names is None:
        names = tuple(f"class_{index}" for index in range(num_classes))
    if len(names) != num_classes:
        raise ValueError(
            f"{data_yaml}: class name count ({len(names)}) does not match nc ({num_classes})"
        )

    return YoloDatasetConfig(
        root=data_root,
        num_classes=num_classes,
        class_names=names,
    )