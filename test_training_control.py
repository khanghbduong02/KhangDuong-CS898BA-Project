from __future__ import annotations

import tempfile
from pathlib import Path

import torch
from torch.optim import SGD

from run_faster_rcnn_kfold_cv import completed_run_matches as faster_rcnn_completed_run_matches
from run_yolo26_kfold_cv import completed_run_matches as yolo26_completed_run_matches
from training_control import (
    EpochLRScheduleConfig,
    EpochLRScheduler,
    PlateauEarlyStopping,
    PlateauEarlyStoppingConfig,
    checkpoint_selection_improved,
    initial_checkpoint_selection_value,
    validate_training_control_compatibility,
)


def _optimizer(lr: float = 0.1) -> SGD:
    parameter = torch.nn.Parameter(torch.tensor(1.0))
    return SGD([parameter], lr=lr)


def test_formula_validation() -> None:
    """Require early-stopping patience to leave room for two plateau reductions."""
    valid = PlateauEarlyStoppingConfig(
        reduce_lr_patience=4,
        reduce_lr_factor=0.5,
        reduce_lr_cooldown=3,
        min_lr=1e-6,
        early_stopping_patience=11,
    )
    valid.validate()
    assert valid.minimum_early_stopping_patience == 11

    invalid = PlateauEarlyStoppingConfig(
        reduce_lr_patience=4,
        reduce_lr_factor=0.5,
        reduce_lr_cooldown=3,
        min_lr=1e-6,
        early_stopping_patience=10,
    )
    try:
        invalid.validate()
    except ValueError as exc:
        assert "early_stopping_patience" in str(exc)
    else:
        raise AssertionError("An invalid early-stopping/plateau patience pair was accepted")


def test_lr_reduction_resets_early_stopping_counter() -> None:
    """A real learning-rate drop gives the new rate a fresh early-stop window."""
    control = PlateauEarlyStopping(
        _optimizer(),
        PlateauEarlyStoppingConfig(
            reduce_lr_patience=1,
            reduce_lr_factor=0.5,
            reduce_lr_cooldown=0,
            min_lr=1e-6,
            early_stopping_patience=2,
        ),
    )

    first = control.step(1.0)
    assert first.improved
    assert not first.lr_reduced

    first_bad_epoch = control.step(1.1)
    assert first_bad_epoch.bad_epochs == 1
    assert not first_bad_epoch.lr_reduced

    reduction_epoch = control.step(1.2)
    assert reduction_epoch.lr_reduced
    assert reduction_epoch.learning_rates == (0.05,)
    assert reduction_epoch.bad_epochs == 0
    assert not reduction_epoch.should_stop


def test_early_stopping_after_no_effective_lr_reduction() -> None:
    """Stop after patience when the scheduler cannot lower the configured minimum LR."""
    control = PlateauEarlyStopping(
        _optimizer(),
        PlateauEarlyStoppingConfig(
            reduce_lr_patience=1,
            reduce_lr_factor=0.5,
            reduce_lr_cooldown=0,
            min_lr=0.1,
            early_stopping_patience=2,
        ),
    )

    control.step(1.0)
    first_bad_epoch = control.step(1.1)
    assert not first_bad_epoch.should_stop
    stop_epoch = control.step(1.2)
    assert not stop_epoch.lr_reduced
    assert stop_epoch.bad_epochs == 2
    assert stop_epoch.should_stop


def test_disabled_controls_remain_inert() -> None:
    """Both controls can be explicitly disabled together for archival reproduction."""
    control = PlateauEarlyStopping(
        _optimizer(),
        PlateauEarlyStoppingConfig(
            reduce_lr_patience=0,
            reduce_lr_factor=0.5,
            reduce_lr_cooldown=0,
            min_lr=1e-6,
            early_stopping_patience=0,
        ),
    )
    control.step(1.0)
    result = control.step(2.0)
    assert not result.should_stop
    assert not result.lr_reduced
    assert result.bad_epochs == 0
    assert control.state_dict()["scheduler_state_dict"] is None


def test_warmup_cosine_schedule() -> None:
    """Warmup reaches the base rate before a deterministic cosine decay."""
    optimizer = _optimizer(lr=0.1)
    schedule = EpochLRScheduler(
        optimizer,
        EpochLRScheduleConfig(
            total_epochs=6,
            schedule="cosine",
            warmup_epochs=3,
            warmup_start_factor=0.1,
            cosine_final_factor=0.2,
        ),
    )
    expected = ((1, 0.1, 0.01), (2, 0.55, 0.055), (3, 1.0, 0.1), (4, 1.0, 0.1), (5, 0.6, 0.06), (6, 0.2, 0.02))
    for epoch, factor, lr in expected:
        step = schedule.set_epoch(epoch)
        assert abs(step.factor - factor) < 1e-12
        assert len(step.learning_rates) == 1
        assert abs(step.learning_rates[0] - lr) < 1e-12
    assert schedule.state_dict()["last_epoch"] == 6


def test_epoch_schedule_requires_disabled_plateau() -> None:
    """Do not mix a deterministic epoch schedule with validation-driven reductions."""
    plateau_config = PlateauEarlyStoppingConfig(
        reduce_lr_patience=1,
        reduce_lr_factor=0.5,
        reduce_lr_cooldown=0,
        min_lr=1e-6,
        early_stopping_patience=2,
    )
    schedule_config = EpochLRScheduleConfig(total_epochs=10, schedule="cosine")
    try:
        validate_training_control_compatibility(plateau_config, schedule_config)
    except ValueError as exc:
        assert "requires --reduce-lr-patience 0" in str(exc)
    else:
        raise AssertionError("A deterministic epoch schedule was mixed with plateau control")


def test_checkpoint_selection_modes() -> None:
    """Loss and mAP50 selection optimize in their respective directions."""
    assert initial_checkpoint_selection_value("val_loss") == float("inf")
    assert initial_checkpoint_selection_value("map50") == float("-inf")
    assert checkpoint_selection_improved("val_loss", 1.0, 2.0)
    assert not checkpoint_selection_improved("val_loss", 2.0, 1.0)
    assert checkpoint_selection_improved("map50", 0.30, 0.20)
    assert not checkpoint_selection_improved("map50", 0.20, 0.30)
    try:
        checkpoint_selection_improved("unknown", 1.0, 0.0)
    except ValueError:
        pass
    else:
        raise AssertionError("An unsupported checkpoint-selection mode was accepted")


def test_kfold_runners_accept_early_stopped_checkpoints() -> None:
    """An early-stopped `last.pt` is complete when its frozen settings match."""
    for matcher in (yolo26_completed_run_matches, faster_rcnn_completed_run_matches):
        with tempfile.TemporaryDirectory() as temporary_directory:
            fold_root = Path(temporary_directory) / "fold_1"
            fold_root.mkdir()
            checkpoint_path = fold_root / "last.pt"
            torch.save(
                {
                    "epoch": 42,
                    "args": {
                        "data_root": str(fold_root),
                        "epochs": 150,
                        "use_p2": True,
                    },
                    "class_names": ["defect"],
                    "training_completed": True,
                    "stop_reason": "early_stopping",
                },
                checkpoint_path,
            )
            matches, epoch, completed, mismatches = matcher(
                checkpoint_path,
                fold_root,
                {
                    "epochs": 150,
                    "use_p2": True,
                    "checkpoint_selection": "val_loss",
                    "lr_schedule": "constant",
                    "warmup_epochs": 0,
                    "warmup_start_factor": 0.1,
                    "cosine_final_factor": 0.02,
                },
                ("defect",),
            )
            assert matches, mismatches
            assert completed
            assert epoch == 42


def main() -> None:
    test_formula_validation()
    print("formula_validation: passed")
    test_lr_reduction_resets_early_stopping_counter()
    print("lr_reduction_resets_early_stopping_counter: passed")
    test_early_stopping_after_no_effective_lr_reduction()
    print("early_stopping_after_no_effective_lr_reduction: passed")
    test_disabled_controls_remain_inert()
    print("disabled_controls_remain_inert: passed")
    test_warmup_cosine_schedule()
    print("warmup_cosine_schedule: passed")
    test_epoch_schedule_requires_disabled_plateau()
    print("epoch_schedule_requires_disabled_plateau: passed")
    test_checkpoint_selection_modes()
    print("checkpoint_selection_modes: passed")
    test_kfold_runners_accept_early_stopped_checkpoints()
    print("kfold_runners_accept_early_stopped_checkpoints: passed")


if __name__ == "__main__":
    main()
