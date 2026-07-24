"""Shared training-control and checkpoint-selection policies for local trainers.

The scheduler and early stopper monitor validation loss. A plateau-triggered
learning-rate reduction resets the consecutive non-improvement counter so the
reduced rate receives a fair chance to improve validation loss before early
stopping ends a run. Checkpoint selection can independently use validation
loss or validation mAP50.
"""
from __future__ import annotations

import argparse
import math
from dataclasses import asdict, dataclass
from typing import Any

from torch.optim import Optimizer
from torch.optim.lr_scheduler import ReduceLROnPlateau


DEFAULT_REDUCE_LR_PATIENCE = 8
DEFAULT_REDUCE_LR_FACTOR = 0.5
DEFAULT_REDUCE_LR_COOLDOWN = 2
DEFAULT_MIN_LR = 1e-6
DEFAULT_EARLY_STOPPING_PATIENCE = 18
DEFAULT_EARLY_STOPPING_MIN_DELTA = 0.0
DEFAULT_CHECKPOINT_SELECTION = "val_loss"
CHECKPOINT_SELECTION_CHOICES = ("val_loss", "map50")
DEFAULT_LR_SCHEDULE = "constant"
LR_SCHEDULE_CHOICES = ("constant", "cosine")
DEFAULT_WARMUP_EPOCHS = 0
DEFAULT_WARMUP_START_FACTOR = 0.1
DEFAULT_COSINE_FINAL_FACTOR = 0.02


@dataclass(frozen=True)
class EpochLRScheduleConfig:
    """Configuration for a deterministic epoch-based learning-rate schedule.

    The default preserves the historical constant learning rate exactly. The
    cosine option is intentionally independent of validation loss so a
    schedule-only experiment does not use the validation fold to alter its
    optimization trajectory.
    """

    total_epochs: int
    schedule: str = DEFAULT_LR_SCHEDULE
    warmup_epochs: int = DEFAULT_WARMUP_EPOCHS
    warmup_start_factor: float = DEFAULT_WARMUP_START_FACTOR
    cosine_final_factor: float = DEFAULT_COSINE_FINAL_FACTOR

    @property
    def enabled(self) -> bool:
        return self.schedule != "constant" or self.warmup_epochs > 0

    def validate(self) -> None:
        if self.total_epochs <= 0:
            raise ValueError("--epochs must be positive for the learning-rate schedule")
        if self.schedule not in LR_SCHEDULE_CHOICES:
            raise ValueError(f"--lr-schedule must be one of {LR_SCHEDULE_CHOICES}")
        if self.warmup_epochs < 0:
            raise ValueError("--warmup-epochs must be non-negative")
        if self.warmup_epochs > self.total_epochs:
            raise ValueError("--warmup-epochs cannot exceed --epochs")
        if not 0.0 < self.warmup_start_factor <= 1.0:
            raise ValueError("--warmup-start-factor must be in (0, 1]")
        if not 0.0 < self.cosine_final_factor <= 1.0:
            raise ValueError("--cosine-final-factor must be in (0, 1]")
        if self.schedule == "cosine" and self.warmup_epochs >= self.total_epochs:
            raise ValueError("--lr-schedule cosine requires at least one epoch after warmup")


@dataclass(frozen=True)
class EpochLRScheduleStep:
    """Observable learning-rate state applied before one training epoch."""

    epoch: int
    factor: float
    learning_rates: tuple[float, ...]


class EpochLRScheduler:
    """Apply optional linear warmup followed by a deterministic cosine decay."""

    def __init__(self, optimizer: Optimizer, config: EpochLRScheduleConfig) -> None:
        config.validate()
        self.optimizer = optimizer
        self.config = config
        self.base_learning_rates = tuple(float(group["lr"]) for group in optimizer.param_groups)
        if not self.base_learning_rates or any(
            not math.isfinite(rate) or rate <= 0.0 for rate in self.base_learning_rates
        ):
            raise ValueError("Optimizer learning rates must be finite and positive")
        self.last_epoch = 0
        self.last_factor = 1.0

    def factor_for_epoch(self, epoch: int) -> float:
        """Return the multiplier for one-indexed ``epoch`` without mutating state."""
        if not 1 <= epoch <= self.config.total_epochs:
            raise ValueError(
                f"Learning-rate schedule epoch must be in [1, {self.config.total_epochs}], got {epoch}"
            )

        if self.config.warmup_epochs > 0 and epoch <= self.config.warmup_epochs:
            if self.config.warmup_epochs == 1:
                return 1.0
            progress = (epoch - 1) / (self.config.warmup_epochs - 1)
            return self.config.warmup_start_factor + (
                1.0 - self.config.warmup_start_factor
            ) * progress

        if self.config.schedule == "constant":
            return 1.0

        decay_epochs = self.config.total_epochs - self.config.warmup_epochs
        if decay_epochs <= 1:
            return self.config.cosine_final_factor
        progress = (epoch - self.config.warmup_epochs - 1) / (decay_epochs - 1)
        cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
        return self.config.cosine_final_factor + (
            1.0 - self.config.cosine_final_factor
        ) * cosine

    def set_epoch(self, epoch: int) -> EpochLRScheduleStep:
        """Set all optimizer groups to the scheduled rate for one training epoch."""
        factor = self.factor_for_epoch(epoch)
        for group, base_rate in zip(self.optimizer.param_groups, self.base_learning_rates):
            group["lr"] = base_rate * factor
        self.last_epoch = epoch
        self.last_factor = factor
        return EpochLRScheduleStep(
            epoch=epoch,
            factor=factor,
            learning_rates=tuple(float(group["lr"]) for group in self.optimizer.param_groups),
        )

    def state_dict(self) -> dict[str, Any]:
        """Return serializable schedule metadata for transparent checkpoints."""
        return {
            "config": asdict(self.config),
            "base_learning_rates": list(self.base_learning_rates),
            "last_epoch": self.last_epoch,
            "last_factor": self.last_factor,
            "learning_rates": [float(group["lr"]) for group in self.optimizer.param_groups],
        }


@dataclass(frozen=True)
class PlateauEarlyStoppingConfig:
    """Immutable configuration shared by the scheduler and early stopper.

    A value of zero disables both controls. When enabled, the configuration
    enforces the requested policy:

    ``early_stopping_patience >= 2 * reduce_lr_patience + reduce_lr_cooldown``.
    """

    reduce_lr_patience: int = DEFAULT_REDUCE_LR_PATIENCE
    reduce_lr_factor: float = DEFAULT_REDUCE_LR_FACTOR
    reduce_lr_cooldown: int = DEFAULT_REDUCE_LR_COOLDOWN
    min_lr: float = DEFAULT_MIN_LR
    early_stopping_patience: int = DEFAULT_EARLY_STOPPING_PATIENCE
    early_stopping_min_delta: float = DEFAULT_EARLY_STOPPING_MIN_DELTA

    @property
    def enabled(self) -> bool:
        return self.reduce_lr_patience > 0

    @property
    def minimum_early_stopping_patience(self) -> int:
        return 2 * self.reduce_lr_patience + self.reduce_lr_cooldown

    def validate(self) -> None:
        if self.reduce_lr_patience < 0:
            raise ValueError("--reduce-lr-patience must be non-negative")
        if self.early_stopping_patience < 0:
            raise ValueError("--early-stopping-patience must be non-negative")
        if self.reduce_lr_cooldown < 0:
            raise ValueError("--reduce-lr-cooldown must be non-negative")
        if not 0.0 < self.reduce_lr_factor < 1.0:
            raise ValueError("--reduce-lr-factor must be in (0, 1)")
        if self.min_lr < 0.0:
            raise ValueError("--min-lr must be non-negative")
        if self.early_stopping_min_delta < 0.0:
            raise ValueError("--early-stopping-min-delta must be non-negative")

        reduce_lr_disabled = self.reduce_lr_patience == 0
        early_stopping_disabled = self.early_stopping_patience == 0
        if reduce_lr_disabled != early_stopping_disabled:
            raise ValueError(
                "--reduce-lr-patience and --early-stopping-patience must be enabled or disabled together"
            )
        if reduce_lr_disabled:
            if self.reduce_lr_cooldown != 0:
                raise ValueError("--reduce-lr-cooldown must be 0 when plateau controls are disabled")
            return
        if self.early_stopping_patience < self.minimum_early_stopping_patience:
            raise ValueError(
                "Early-stopping patience must satisfy "
                "early_stopping_patience >= reduce_lr_patience * 2 + reduce_lr_cooldown; "
                f"received {self.early_stopping_patience} < "
                f"{self.reduce_lr_patience} * 2 + {self.reduce_lr_cooldown} "
                f"= {self.minimum_early_stopping_patience}."
            )


def add_plateau_early_stopping_arguments(parser: argparse.ArgumentParser) -> None:
    """Add a consistent, validated scheduler/early-stopping CLI policy."""
    parser.add_argument(
        "--reduce-lr-patience",
        type=int,
        default=DEFAULT_REDUCE_LR_PATIENCE,
        help=(
            "Validation-loss plateau epochs before ReduceLROnPlateau lowers the learning rate; "
            "set this and --early-stopping-patience to 0 to disable both controls"
        ),
    )
    parser.add_argument(
        "--reduce-lr-factor",
        type=float,
        default=DEFAULT_REDUCE_LR_FACTOR,
        help="Multiplicative learning-rate factor applied after a validation-loss plateau",
    )
    parser.add_argument(
        "--reduce-lr-cooldown",
        type=int,
        default=DEFAULT_REDUCE_LR_COOLDOWN,
        help="Epochs to wait after a learning-rate reduction before another plateau reduction",
    )
    parser.add_argument(
        "--min-lr",
        type=float,
        default=DEFAULT_MIN_LR,
        help="Lower bound applied to every optimizer parameter-group learning rate",
    )
    parser.add_argument(
        "--early-stopping-patience",
        type=int,
        default=DEFAULT_EARLY_STOPPING_PATIENCE,
        help=(
            "Consecutive validation-loss non-improvement epochs before stopping; must satisfy "
            "early_stopping_patience >= reduce_lr_patience * 2 + reduce_lr_cooldown"
        ),
    )
    parser.add_argument(
        "--early-stopping-min-delta",
        type=float,
        default=DEFAULT_EARLY_STOPPING_MIN_DELTA,
        help="Minimum validation-loss decrease required to reset early-stopping patience",
    )


def add_epoch_lr_schedule_arguments(parser: argparse.ArgumentParser) -> None:
    """Add optional deterministic warmup/cosine learning-rate arguments."""
    parser.add_argument(
        "--lr-schedule",
        choices=LR_SCHEDULE_CHOICES,
        default=DEFAULT_LR_SCHEDULE,
        help=(
            "Epoch-based learning-rate schedule; constant preserves historical behavior, while cosine "
            "uses deterministic decay after optional warmup"
        ),
    )
    parser.add_argument(
        "--warmup-epochs",
        type=int,
        default=DEFAULT_WARMUP_EPOCHS,
        help="Linear learning-rate warmup epochs; 0 disables warmup",
    )
    parser.add_argument(
        "--warmup-start-factor",
        type=float,
        default=DEFAULT_WARMUP_START_FACTOR,
        help="Learning-rate multiplier at the first warmup epoch",
    )
    parser.add_argument(
        "--cosine-final-factor",
        type=float,
        default=DEFAULT_COSINE_FINAL_FACTOR,
        help="Final learning-rate multiplier at the last cosine-decay epoch",
    )


def add_checkpoint_selection_argument(parser: argparse.ArgumentParser) -> None:
    """Add a reproducible criterion for choosing ``best.pt``.

    The selection criterion is intentionally independent of the loss-based
    scheduler and early stopper. This lets a detection project choose the
    checkpoint aligned with its reported AP metric without changing the
    requested patience policy.
    """
    parser.add_argument(
        "--checkpoint-selection",
        choices=CHECKPOINT_SELECTION_CHOICES,
        default=DEFAULT_CHECKPOINT_SELECTION,
        help=(
            "Metric used to save best.pt: val_loss preserves historical behavior; "
            "map50 selects highest validation mAP50 while LR scheduling and early stopping remain loss-based"
        ),
    )


def validate_checkpoint_selection(selection: str) -> str:
    """Validate and normalize a checkpoint-selection policy."""
    if selection not in CHECKPOINT_SELECTION_CHOICES:
        raise ValueError(
            f"checkpoint selection must be one of {CHECKPOINT_SELECTION_CHOICES}, got {selection!r}"
        )
    return selection


def initial_checkpoint_selection_value(selection: str) -> float:
    """Return the sentinel best value for a minimization or maximization policy."""
    selection = validate_checkpoint_selection(selection)
    return math.inf if selection == "val_loss" else -math.inf


def checkpoint_selection_improved(selection: str, candidate: float, best: float) -> bool:
    """Return whether a finite candidate improves the requested selection metric."""
    selection = validate_checkpoint_selection(selection)
    candidate = float(candidate)
    best = float(best)
    if not math.isfinite(candidate):
        raise FloatingPointError("Checkpoint-selection metric must be finite")
    return candidate < best if selection == "val_loss" else candidate > best


def plateau_early_stopping_config_from_args(args: argparse.Namespace) -> PlateauEarlyStoppingConfig:
    """Build and validate a controller configuration from a trainer namespace."""
    config = PlateauEarlyStoppingConfig(
        reduce_lr_patience=args.reduce_lr_patience,
        reduce_lr_factor=args.reduce_lr_factor,
        reduce_lr_cooldown=args.reduce_lr_cooldown,
        min_lr=args.min_lr,
        early_stopping_patience=args.early_stopping_patience,
        early_stopping_min_delta=args.early_stopping_min_delta,
    )
    config.validate()
    return config


def epoch_lr_schedule_config_from_args(args: argparse.Namespace) -> EpochLRScheduleConfig:
    """Build and validate an epoch-based schedule configuration from trainer arguments."""
    config = EpochLRScheduleConfig(
        total_epochs=args.epochs,
        schedule=args.lr_schedule,
        warmup_epochs=args.warmup_epochs,
        warmup_start_factor=args.warmup_start_factor,
        cosine_final_factor=args.cosine_final_factor,
    )
    config.validate()
    return config


def validate_training_control_compatibility(
    plateau_config: PlateauEarlyStoppingConfig,
    epoch_schedule_config: EpochLRScheduleConfig,
) -> None:
    """Reject ambiguous combinations of validation-driven and epoch-driven schedulers."""
    plateau_config.validate()
    epoch_schedule_config.validate()
    if plateau_config.enabled and epoch_schedule_config.enabled:
        raise ValueError(
            "An epoch-based learning-rate schedule requires --reduce-lr-patience 0 and "
            "--early-stopping-patience 0 so it is not mixed with validation-loss plateau control"
        )


@dataclass(frozen=True)
class TrainingControlStep:
    """Observable state emitted once after each validation epoch."""

    metric: float
    improved: bool
    lr_reduced: bool
    should_stop: bool
    bad_epochs: int
    learning_rates: tuple[float, ...]


class PlateauEarlyStopping:
    """Coordinate ReduceLROnPlateau and early stopping on validation loss."""

    def __init__(self, optimizer: Optimizer, config: PlateauEarlyStoppingConfig) -> None:
        config.validate()
        self.optimizer = optimizer
        self.config = config
        self.scheduler: ReduceLROnPlateau | None = None
        if config.enabled:
            self.scheduler = ReduceLROnPlateau(
                optimizer,
                mode="min",
                factor=config.reduce_lr_factor,
                patience=config.reduce_lr_patience,
                cooldown=config.reduce_lr_cooldown,
                min_lr=config.min_lr,
                threshold=config.early_stopping_min_delta,
                threshold_mode="abs",
            )
        self.best_metric = math.inf
        self.bad_epochs = 0
        self.lr_reductions = 0

    @property
    def learning_rates(self) -> tuple[float, ...]:
        return tuple(float(group["lr"]) for group in self.optimizer.param_groups)

    def step(self, metric: float) -> TrainingControlStep:
        """Update scheduler/stopping state after a finite validation-loss metric."""
        metric = float(metric)
        if not math.isfinite(metric):
            raise FloatingPointError("Plateau/early-stopping monitor received a non-finite validation loss")

        learning_rates_before = self.learning_rates
        if self.scheduler is not None:
            self.scheduler.step(metric)
        learning_rates_after = self.learning_rates
        lr_reduced = any(
            after < before - 1e-15
            for before, after in zip(learning_rates_before, learning_rates_after)
        )
        if lr_reduced:
            self.lr_reductions += 1

        improved = metric < self.best_metric - self.config.early_stopping_min_delta
        if improved:
            self.best_metric = metric
            self.bad_epochs = 0
        elif lr_reduced:
            # A new lower learning rate is a new optimization opportunity.
            self.bad_epochs = 0
        elif self.config.enabled:
            self.bad_epochs += 1

        should_stop = self.config.enabled and self.bad_epochs >= self.config.early_stopping_patience
        return TrainingControlStep(
            metric=metric,
            improved=improved,
            lr_reduced=lr_reduced,
            should_stop=should_stop,
            bad_epochs=self.bad_epochs,
            learning_rates=learning_rates_after,
        )

    def state_dict(self) -> dict[str, Any]:
        """Return serializable controller state for transparent checkpoints."""
        return {
            "config": asdict(self.config),
            "scheduler_state_dict": self.scheduler.state_dict() if self.scheduler is not None else None,
            "best_metric": self.best_metric,
            "bad_epochs": self.bad_epochs,
            "lr_reductions": self.lr_reductions,
            "learning_rates": list(self.learning_rates),
        }
