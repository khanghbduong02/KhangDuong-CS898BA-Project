from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import torch
from ultralytics import YOLO
from ultralytics.data.utils import check_det_dataset

from cv_utils import discover_folds, project_path, validate_cv_layout
from yolo_dataset_config import YoloDatasetConfig


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def validate_ultralytics_fold(fold_root: Path, dataset_config: YoloDatasetConfig) -> Path:
    """Validate that Ultralytics resolves the same class metadata and image directories."""
    data_yaml = fold_root / "data.yaml"
    resolved = check_det_dataset(str(data_yaml))
    if int(resolved["nc"]) != dataset_config.num_classes:
        raise ValueError(f"{data_yaml}: Ultralytics resolved nc={resolved['nc']}, expected {dataset_config.num_classes}")
    for split_key in ("train", "val"):
        images_dir = Path(str(resolved[split_key]))
        if not images_dir.is_dir():
            raise FileNotFoundError(f"{data_yaml}: resolved {split_key} images directory does not exist: {images_dir}")
    return data_yaml


def requested_configuration(args: argparse.Namespace, dataset_config: YoloDatasetConfig) -> dict[str, Any]:
    return {
        "weights": str(args.weights),
        "weights_sha256": sha256_file(args.weights),
        "pretrained": True,
        "epochs": args.epochs,
        "imgsz": args.imgsz,
        "batch": args.batch_size,
        "workers": args.workers,
        "seed": args.seed,
        "device": args.device,
        "optimizer": args.optimizer,
        "deterministic": True,
        "patience": args.patience,
        "standard_ultralytics_augmentation": True,
        "num_classes": dataset_config.num_classes,
        "class_names": list(dataset_config.class_names),
    }


def is_matching_completed_run(run_dir: Path, requested: dict[str, Any]) -> bool:
    completion_path = run_dir / "baseline_complete.json"
    if not completion_path.exists() or not (run_dir / "weights" / "best.pt").exists() or not (run_dir / "weights" / "last.pt").exists():
        return False
    try:
        completion = json.loads(completion_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return False
    return completion.get("requested_configuration") == requested


def write_plan(
    args: argparse.Namespace,
    folds: list[int],
    dataset_config: YoloDatasetConfig,
    configuration: dict[str, Any],
) -> None:
    plan = {
        "purpose": "Generic pretrained Ultralytics detector K-fold reference baseline",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "data_root": str(args.data_root),
        "run_root": str(args.run_root),
        "folds": folds,
        "dataset": {
            "num_classes": dataset_config.num_classes,
            "class_names": list(dataset_config.class_names),
        },
        "requested_configuration": configuration,
        "interpretation": [
            "This is a practical pretrained reference baseline, not an architecture-only comparison with custom YOLO26.",
            "It uses the same frozen fold directories but has pretrained weights, Ultralytics augmentation, and Ultralytics NMS.",
        ],
    }
    path = args.run_root / "training_plan.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(plan, indent=2) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Fine-tune a local pretrained Ultralytics detector across any standard fold_<n> YOLO dataset. "
            "Folds are trained sequentially."
        )
    )
    parser.add_argument("--data-root", type=Path, required=True, help="Root containing fold_<number> directories")
    parser.add_argument("--run-root", type=Path, required=True, help="Output root receiving one run directory per fold")
    parser.add_argument("--weights", type=Path, default=Path("yolo11n.pt"), help="Local pretrained Ultralytics checkpoint")
    parser.add_argument(
        "--folds",
        type=int,
        nargs="*",
        default=None,
        help="Specific fold numbers; omit to discover all fold_<number> directories",
    )
    parser.add_argument("--epochs", type=int, required=True, help="Identical maximum epoch budget for every fold")
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="0", help="Ultralytics device selector; 0 selects the first CUDA GPU")
    parser.add_argument("--optimizer", type=str, default="auto")
    parser.add_argument("--patience", type=int, default=100)
    parser.add_argument(
        "--force",
        action="store_true",
        help="Delete and retrain existing selected fold directories instead of refusing them",
    )
    parser.add_argument("--dry-run", action="store_true", help="Validate inputs and print planned actions without training")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.data_root = project_path(args.data_root)
    args.run_root = project_path(args.run_root)
    args.weights = project_path(args.weights)

    if args.epochs <= 0 or args.imgsz <= 0 or args.batch_size <= 0 or args.workers < 0:
        raise ValueError("Invalid epochs, image size, batch size, or worker count")
    if args.patience < 0:
        raise ValueError("--patience must be non-negative")
    if not args.data_root.is_dir():
        raise FileNotFoundError(f"Data root does not exist: {args.data_root}")
    if not args.weights.is_file():
        raise FileNotFoundError(f"Pretrained weights do not exist: {args.weights}")
    if not torch.cuda.is_available() and str(args.device) not in {"cpu", "mps"}:
        raise RuntimeError("CUDA was requested for the pretrained reference but PyTorch cannot access CUDA")

    folds = discover_folds(args.data_root, args.folds or None)
    fold_roots, dataset_config = validate_cv_layout(args.data_root, folds)
    data_yamls = {fold: validate_ultralytics_fold(fold_roots[fold], dataset_config) for fold in folds}
    configuration = requested_configuration(args, dataset_config)
    if not args.dry_run:
        write_plan(args, folds, dataset_config, configuration)

    for position, fold in enumerate(folds, start=1):
        run_dir = args.run_root / f"fold_{fold}"
        if run_dir.exists() and not args.force:
            if is_matching_completed_run(run_dir, configuration):
                print(f"fold={fold}: skipping matching completed pretrained reference: {run_dir}")
                continue
            raise RuntimeError(
                f"fold={fold}: {run_dir} exists but is not a matching completed run. "
                "Use --force to retrain or choose a distinct --run-root."
            )
        if args.force and run_dir.exists() and not args.dry_run:
            print(f"fold={fold}: removing existing run directory because --force was supplied: {run_dir}")
            shutil.rmtree(run_dir)

        print(f"\n===== ULTRALYTICS PRETRAINED TRAIN FOLD {fold}/{len(folds)} =====", flush=True)
        print(
            f"data={data_yamls[fold]} weights={args.weights.name} epochs={args.epochs} "
            f"imgsz={args.imgsz} batch={args.batch_size} seed={args.seed} device={args.device}",
            flush=True,
        )
        if args.dry_run:
            continue

        model = YOLO(str(args.weights))
        model.train(
            data=str(data_yamls[fold]),
            epochs=args.epochs,
            imgsz=args.imgsz,
            batch=args.batch_size,
            workers=args.workers,
            seed=args.seed,
            deterministic=True,
            optimizer=args.optimizer,
            patience=args.patience,
            device=args.device,
            project=str(args.run_root),
            name=f"fold_{fold}",
            exist_ok=False,
            pretrained=True,
            save=True,
            val=True,
            plots=False,
            verbose=True,
        )

        best_path = run_dir / "weights" / "best.pt"
        last_path = run_dir / "weights" / "last.pt"
        if not best_path.exists() or not last_path.exists():
            raise RuntimeError(f"fold={fold}: Ultralytics ended without best.pt and last.pt in {run_dir}")
        completion = {
            "completed_at_utc": datetime.now(timezone.utc).isoformat(),
            "fold": fold,
            "data_yaml": str(data_yamls[fold]),
            "best_checkpoint": str(best_path),
            "last_checkpoint": str(last_path),
            "requested_configuration": configuration,
        }
        (run_dir / "baseline_complete.json").write_text(json.dumps(completion, indent=2) + "\n", encoding="utf-8")
        print(f"fold={fold}: pretrained reference completed: {best_path}")

    if args.dry_run:
        print("Dry run complete; no pretrained reference training was started.")
    else:
        print("All requested pretrained folds completed. Run eval_ultralytics_kfold_cv.py next.")


if __name__ == "__main__":
    main()