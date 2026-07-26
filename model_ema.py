"""Model exponential-moving-average support for local detector training.

The shadow state tracks every tensor in a model state dictionary. Floating
parameters and buffers, including BatchNorm running statistics, are averaged;
integer buffers such as BatchNorm's ``num_batches_tracked`` are copied from the
current training model. The configured decay is normalized by batches per epoch
so the same setting has the same epoch-scale memory for both local trainers.
"""
from __future__ import annotations

import math
from contextlib import contextmanager
from typing import Any, Iterator, Mapping

import torch
from torch import nn


DEFAULT_EMA_DECAY = 0.0


def validate_ema_decay(decay: float) -> float:
    """Validate a per-epoch EMA retention factor, where zero disables EMA."""
    decay = float(decay)
    if not 0.0 <= decay < 1.0:
        raise ValueError("--ema-decay must be in [0, 1); 0 disables EMA")
    return decay


def _clone_state_dict(state_dict: Mapping[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    """Detach and clone a model state dictionary without sharing tensor storage."""
    return {name: value.detach().clone() for name, value in state_dict.items()}


class ModelEMA:
    """Maintain an epoch-normalized exponential moving average of model state.

    ``decay`` is the fraction of the previous EMA state retained over one full
    training epoch. It is converted to an equivalent per-optimizer-step factor
    using ``updates_per_epoch``. This keeps EMA's time horizon comparable when
    the two local models use different batch sizes.
    """

    def __init__(self, model: nn.Module, decay: float, updates_per_epoch: int) -> None:
        decay = validate_ema_decay(decay)
        if decay == 0.0:
            raise ValueError("ModelEMA requires a non-zero decay")
        if updates_per_epoch <= 0:
            raise ValueError("updates_per_epoch must be positive")

        self.decay = decay
        self.updates_per_epoch = int(updates_per_epoch)
        self.per_update_decay = math.pow(decay, 1.0 / self.updates_per_epoch)
        self.updates = 0
        self._state = _clone_state_dict(model.state_dict())

    @property
    def enabled(self) -> bool:
        return True

    def model_state_dict(self) -> dict[str, torch.Tensor]:
        """Return a detached clone suitable for an EMA-selected checkpoint."""
        return _clone_state_dict(self._state)

    def update(self, model: nn.Module) -> None:
        """Update the shadow state after one successful optimizer step."""
        current_state = model.state_dict()
        if tuple(current_state) != tuple(self._state):
            raise RuntimeError("EMA model state keys do not match the current model")

        with torch.no_grad():
            for name, current_value in current_state.items():
                average_value = self._state[name]
                if current_value.shape != average_value.shape or current_value.dtype != average_value.dtype:
                    raise RuntimeError(f"EMA state for {name!r} is incompatible with the current model")
                if current_value.is_floating_point():
                    average_value.mul_(self.per_update_decay).add_(
                        current_value.detach(), alpha=1.0 - self.per_update_decay
                    )
                else:
                    average_value.copy_(current_value.detach())
        self.updates += 1

    @contextmanager
    def average_parameters(self, model: nn.Module) -> Iterator[None]:
        """Temporarily load EMA tensors into ``model`` and reliably restore raw tensors."""
        raw_state = _clone_state_dict(model.state_dict())
        try:
            model.load_state_dict(self._state, strict=True)
            yield
        finally:
            model.load_state_dict(raw_state, strict=True)

    def metadata(self) -> dict[str, Any]:
        """Return non-tensor EMA metadata stored with checkpoints and history."""
        return {
            "decay_per_epoch": self.decay,
            "updates_per_epoch": self.updates_per_epoch,
            "per_update_decay": self.per_update_decay,
            "updates": self.updates,
            "averages_floating_buffers": True,
            "copies_non_floating_buffers": True,
        }

    def state_dict(self) -> dict[str, Any]:
        """Return complete EMA metadata and shadow tensors for resumable diagnostics."""
        return {**self.metadata(), "model_state_dict": self.model_state_dict()}
