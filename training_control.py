"""Shared validation-loss scheduler and early-stopping controls for local trainers.

The controls deliberately use the same validation-loss signal already used to
select ``best.pt``. A plateau-triggered learning-rate reduction resets the
consecutive non-improvement counter so the reduced rate receives a fair chance
to improve validation loss before early stopping ends a run.
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
