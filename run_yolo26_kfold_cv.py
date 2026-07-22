from __future__ import annotations

import argparse
import json
import math
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import torch

from cv_utils import PROJECT_ROOT, discover_folds, project_path, render_fold_path, validate_cv_layout
from training_control import add_plateau_early_stopping_arguments, plateau_early_stopping_config_from_args
from yolo_dataset_config import YoloDatasetConfig


def training_settings(args: argparse.Namespace, num_classes: int) -> dict[str, Any]:
    """Return every trainer setting that must remain identical across folds."""
    return {
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "imgsz": args.imgsz,
        "lr": args.lr,
        "weight_decay": args.weight_decay,
        "workers": args.workers,
        "fraction": args.fraction,
        "seed": args.seed,
        "device": args.device,
        "scale": args.scale,
        "num_classes": num_classes,
        "box_gain": args.box_gain,
        "cls_gain": args.cls_gain,
        "reg_gain": args.reg_gain,
        "reg_max": args.reg_max,
        "use_p2": args.use_p2,
        "one2many_topk": args.one2many_topk,
        "one2one_topk": args.one2one_topk,
        "focal_gamma": args.focal_gamma,
        "class_positive_weight_power": args.class_positive_weight_power,
        "balanced_sampling": args.balanced_sampling,
        "balanced_sampling_power": args.balanced_sampling_power,
        "reduce_lr_patience": args.reduce_lr_patience,
        "reduce_lr_factor": args.reduce_lr_factor,
        "reduce_lr_cooldown": args.reduce_lr_cooldown,
        "min_lr": args.min_lr,
        "early_stopping_patience": args.early_stopping_patience,
        "early_stopping_min_delta": args.early_stopping_min_delta,
    }


def build_train_command(
    args: argparse.Namespace,
    fold_root: Path,
    run_dir: Path,
    num_classes: int,
) -> list[str]:
    command = [
        sys.executable,
        "train_yolo26.py",
        "--data-root",
        str(fold_root),
        "--epochs",
        str(args.epochs),
        "--batch-size",
        str(args.batch_size),
        "--imgsz",
        str(args.imgsz),
        "--lr",
        str(args.lr),
        "--weight-decay",
        str(args.weight_decay),
        "--workers",
        str(args.workers),
        "--fraction",
        str(args.fraction),
        "--seed",
        str(args.seed),
        "--device",
        args.device,
        "--scale",
        args.scale,
        "--num-classes",
        str(num_classes),
        "--box-gain",
        str(args.box_gain),
        "--cls-gain",
        str(args.cls_gain),
        "--reg-gain",
        str(args.reg_gain),
        "--reg-max",
        str(args.reg_max),
        "--one2many-topk",
        str(args.one2many_topk),
        "--one2one-topk",
        str(args.one2one_topk),
        "--focal-gamma",
        str(args.focal_gamma),
        "--class-positive-weight-power",
        str(args.class_positive_weight_power),
        "--balanced-sampling-power",
        str(args.balanced_sampling_power),
        "--reduce-lr-patience",
        str(args.reduce_lr_patience),
        "--reduce-lr-factor",
        str(args.reduce_lr_factor),
        "--reduce-lr-cooldown",
        str(args.reduce_lr_cooldown),
        "--min-lr",
        str(args.min_lr),
        "--early-stopping-patience",
        str(args.early_stopping_patience),
        "--early-stopping-min-delta",
        str(args.early_stopping_min_delta),
        "--save-dir",
        str(run_dir),
    ]
    if args.use_p2:
        command.append("--use-p2")
    if args.balanced_sampling:
        command.append("--balanced-sampling")
    return command


def read_checkpoint(path: Path) -> dict[str, Any]:
    """Read trusted local checkpoint metadata used to prevent incompatible skips."""
    try:
        checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    except Exception as exc:
        raise RuntimeError(f"Unable to read checkpoint metadata from {path}") from exc
    if not isinstance(checkpoint, dict):
        raise RuntimeError(f"Checkpoint {path} is not a metadata dictionary")
    return checkpoint


def values_match(expected: object, actual: object) -> bool:
    if isinstance(expected, float):
        return isinstance(actual, (int, float)) and math.isclose(float(actual), expected, rel_tol=1e-12, abs_tol=1e-12)
    return actual == expected


def completed_run_matches(
    last_path: Path,
    fold_root: Path,
    expected_settings: dict[str, Any],
    expected_names: tuple[str, ...],
) -> tuple[bool, int, bool, list[str]]:
    """Return whether a completed checkpoint matches the requested fold protocol."""
    checkpoint = read_checkpoint(last_path)
    epoch = checkpoint.get("epoch")
    if not isinstance(epoch, int):
        return False, -1, False, ["checkpoint has no integer epoch"]

    saved_args = checkpoint.get("args")
    if not isinstance(saved_args, dict):
        return False, epoch, False, ["checkpoint has no saved trainer arguments"]

    mismatches: list[str] = []
    for key, expected in expected_settings.items():
        actual = saved_args.get(key)
        # Checkpoints created before distributional regression was added did
        # not save this argument; their architecture is the direct-regression
        # `reg_max=1` configuration.
        if key == "reg_max" and actual is None:
            actual = 1
        # Pre-P2 checkpoints use the historical three-level P3--P5 head.
        if key == "use_p2" and actual is None:
            actual = False
        if not values_match(expected, actual):
            mismatches.append(f"{key}: expected {expected!r}, found {actual!r}")

    saved_data_root = saved_args.get("data_root")
    try:
        if Path(saved_data_root).resolve() != fold_root.resolve():
            mismatches.append(f"data_root: expected {fold_root}, found {saved_data_root}")
    except TypeError:
        mismatches.append(f"data_root: expected {fold_root}, found {saved_data_root!r}")

    saved_names = checkpoint.get("class_names")
    if saved_names is not None and tuple(str(name) for name in saved_names) != expected_names:
        mismatches.append(f"class_names: expected {list(expected_names)!r}, found {saved_names!r}")

    saved_completion = checkpoint.get("training_completed")
    if saved_completion is None:
        completed = epoch >= int(expected_settings["epochs"])
    elif isinstance(saved_completion, bool):
        completed = saved_completion
    else:
        completed = False
        mismatches.append(f"training_completed: expected a bool, found {saved_completion!r}")

    return not mismatches, epoch, completed, mismatches


def write_plan(
    args: argparse.Namespace,
    folds: list[int],
    dataset_config: YoloDatasetConfig,
    settings: dict[str, Any],
) -> None:
    plan = {
        "purpose": "Generic custom YOLO26 K-fold cross-validation",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "data_root": str(args.data_root),
        "run_root": str(args.run_root),
        "run_dir_template": args.run_dir_template,
        "folds": folds,
        "dataset": {
            "num_classes": dataset_config.num_classes,
            "class_names": list(dataset_config.class_names),
        },
        "training_settings": settings,
        "notes": [
            "Folds are trained sequentially to avoid sharing a single GPU across concurrent jobs.",
            "A completed fold is skipped only when its checkpoint settings match this requested protocol.",
            "A formula-valid early-stopped fold is complete when last.pt records training_completed=true.",
            "The trainer has no resume mode; partial folds must be retrained with --force or a new run root.",
        ],
    }
    plan_path = args.run_root / "training_plan.json"
    plan_path.parent.mkdir(parents=True, exist_ok=True)
    plan_path.write_text(json.dumps(plan, indent=2) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Train any standard fold_<n>/train+valid YOLO dataset with the custom YOLO26 model. "
            "Folds run sequentially and share one frozen configuration."
        )
    )
    parser.add_argument("--data-root", type=Path, required=True, help="Root containing fold_<number> directories")
    parser.add_argument("--run-root", type=Path, required=True, help="Output root receiving one run directory per fold")
    parser.add_argument(
        "--run-dir-template",
        type=str,
        default="fold_{fold}",
        help="Per-fold path beneath --run-root; must include {fold}",
    )
    parser.add_argument(
        "--folds",
        type=int,
        nargs="*",
        default=None,
        help="Specific fold numbers; omit to discover all fold_<number> directories",
    )
    parser.add_argument("--epochs", type=int, required=True, help="Identical maximum epoch budget for every fold")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=5e-4)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--fraction", type=float, default=1.0, help="Use only for explicitly marked smoke tests")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--scale", type=str, default="n")
    parser.add_argument("--box-gain", type=float, default=7.5)
    parser.add_argument("--cls-gain", type=float, default=0.5)
    parser.add_argument("--reg-gain", type=float, default=1.5)
    parser.add_argument("--reg-max", type=int, default=1, help="Box distance bins per side; values above 1 enable DFL")
    parser.add_argument("--use-p2", action="store_true", help="Add the optional stride-4 P2 detection level")
    parser.add_argument("--one2many-topk", type=int, default=10)
    parser.add_argument("--one2one-topk", type=int, default=1)
    parser.add_argument("--focal-gamma", type=float, default=0.0)
    parser.add_argument("--class-positive-weight-power", type=float, default=0.0)
    parser.add_argument("--balanced-sampling", action="store_true")
    parser.add_argument("--balanced-sampling-power", type=float, default=1.0)
    add_plateau_early_stopping_arguments(parser)
    parser.add_argument(
        "--force",
        action="store_true",
        help="Delete and retrain an existing selected fold even if it is completed or partial",
    )
    parser.add_argument("--dry-run", action="store_true", help="Validate folds and print commands without training")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.data_root = project_path(args.data_root)
    args.run_root = project_path(args.run_root)

    if args.epochs <= 0 or args.batch_size <= 0 or args.imgsz <= 0:
        raise ValueError("--epochs, --batch-size, and --imgsz must be positive")
    if args.reg_max <= 0:
        raise ValueError("--reg-max must be positive")
    if args.workers < 0 or not 0.0 < args.fraction <= 1.0:
        raise ValueError("--workers must be non-negative and --fraction must be in (0, 1]")
    plateau_early_stopping_config_from_args(args)
    if not args.data_root.is_dir():
        raise FileNotFoundError(f"Data root does not exist: {args.data_root}")

    folds = discover_folds(args.data_root, args.folds or None)
    fold_roots, dataset_config = validate_cv_layout(args.data_root, folds)
    settings = training_settings(args, dataset_config.num_classes)
    if not args.dry_run:
        write_plan(args, folds, dataset_config, settings)

    for position, fold in enumerate(folds, start=1):
        fold_root = fold_roots[fold]
        run_dir = render_fold_path(args.run_root, args.run_dir_template, fold)
        last_path = run_dir / "last.pt"

        if run_dir.exists() and not args.force:
            if not last_path.exists() or not (run_dir / "best.pt").exists():
                raise RuntimeError(
                    f"fold={fold}: {run_dir} exists without both best.pt and last.pt. "
                    "Use --force to restart it or choose a new --run-root."
                )
            matches, completed_epoch, completed, mismatches = completed_run_matches(
                last_path,
                fold_root,
                settings,
                dataset_config.class_names,
            )
            if completed and matches:
                print(f"fold={fold}: skipping matching completed run at epoch={completed_epoch}: {run_dir}")
                continue
            mismatch_text = (
                "; ".join(mismatches)
                if mismatches
                else f"training did not complete; last checkpoint is epoch {completed_epoch}"
            )
            raise RuntimeError(
                f"fold={fold}: existing run cannot be reused: {mismatch_text}. "
                "Use --force to retrain or choose a distinct --run-root."
            )

        if args.force and run_dir.exists() and not args.dry_run:
            print(f"fold={fold}: removing existing run directory because --force was supplied: {run_dir}")
            shutil.rmtree(run_dir)

        command = build_train_command(args, fold_root, run_dir, dataset_config.num_classes)
        print(f"\n===== CUSTOM YOLO26 TRAIN FOLD {fold}/{len(folds)} =====", flush=True)
        print(subprocess.list2cmdline(command), flush=True)
        if args.dry_run:
            continue

        subprocess.run(command, cwd=PROJECT_ROOT, check=True)
        if not last_path.exists() or not (run_dir / "best.pt").exists():
            raise RuntimeError(f"fold={fold}: training ended without both best.pt and last.pt in {run_dir}")
        matches, completed_epoch, completed, mismatches = completed_run_matches(
            last_path,
            fold_root,
            settings,
            dataset_config.class_names,
        )
        if not matches or not completed:
            raise RuntimeError(
                f"fold={fold}: saved result does not match the requested protocol: "
                f"epoch={completed_epoch}, completed={completed}, mismatches={mismatches}"
            )
        print(f"fold={fold}: completed epoch={completed_epoch}; checkpoints saved in {run_dir}")

    if args.dry_run:
        print("Dry run complete; no training was started.")
    else:
        print("All requested custom YOLO26 folds completed. Run eval_yolo26_kfold_cv.py next.")


if __name__ == "__main__":
    main()