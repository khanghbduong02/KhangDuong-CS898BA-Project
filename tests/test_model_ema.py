from __future__ import annotations

import math

import torch
from torch import nn

from model_ema import ModelEMA, validate_ema_decay


class TinyBatchNormModel(nn.Module):
    """Small stateful module covering parameters and BatchNorm buffers."""

    def __init__(self) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.tensor([1.0]))
        self.bn = nn.BatchNorm1d(1)


def _set_state(model: TinyBatchNormModel, value: float, batch_count: int) -> None:
    with torch.no_grad():
        model.weight.fill_(value)
        model.bn.weight.fill_(value + 1.0)
        model.bn.bias.fill_(value + 2.0)
        model.bn.running_mean.fill_(value + 3.0)
        model.bn.running_var.fill_(value + 4.0)
        model.bn.num_batches_tracked.fill_(batch_count)


def test_ema_averages_floating_tensors_and_copies_counters() -> None:
    """EMA must smooth weights/BN statistics and copy non-floating counters."""
    model = TinyBatchNormModel()
    _set_state(model, value=1.0, batch_count=2)
    ema = ModelEMA(model, decay=0.5, updates_per_epoch=1)

    _set_state(model, value=3.0, batch_count=8)
    ema.update(model)
    averaged = ema.model_state_dict()

    assert torch.allclose(averaged["weight"], torch.tensor([2.0]))
    assert torch.allclose(averaged["bn.weight"], torch.tensor([3.0]))
    assert torch.allclose(averaged["bn.bias"], torch.tensor([4.0]))
    assert torch.allclose(averaged["bn.running_mean"], torch.tensor([5.0]))
    assert torch.allclose(averaged["bn.running_var"], torch.tensor([6.0]))
    assert averaged["bn.num_batches_tracked"].item() == 8
    assert ema.updates == 1


def test_ema_decay_is_normalized_to_epoch_length() -> None:
    """The same per-epoch retention has an equivalent product over all batches."""
    ema = ModelEMA(TinyBatchNormModel(), decay=0.81, updates_per_epoch=2)
    assert math.isclose(ema.per_update_decay, 0.9, rel_tol=0.0, abs_tol=1e-12)
    assert math.isclose(ema.per_update_decay**ema.updates_per_epoch, 0.81, rel_tol=0.0, abs_tol=1e-12)


def test_ema_temporary_swap_restores_raw_model() -> None:
    """Temporary EMA evaluation must not modify the optimizer-owned raw model tensors."""
    model = TinyBatchNormModel()
    _set_state(model, value=1.0, batch_count=1)
    ema = ModelEMA(model, decay=0.5, updates_per_epoch=1)
    _set_state(model, value=5.0, batch_count=7)
    ema.update(model)

    raw_state = {name: value.detach().clone() for name, value in model.state_dict().items()}
    with ema.average_parameters(model):
        assert torch.allclose(model.weight, torch.tensor([3.0]))
        assert model.bn.num_batches_tracked.item() == 7
    for name, expected in raw_state.items():
        assert torch.equal(model.state_dict()[name], expected), name


def test_ema_decay_validation() -> None:
    """Zero disables EMA at the trainer level, while invalid retention factors are rejected."""
    assert validate_ema_decay(0.0) == 0.0
    assert validate_ema_decay(0.9) == 0.9
    for invalid in (-0.01, 1.0, 1.01):
        try:
            validate_ema_decay(invalid)
        except ValueError:
            pass
        else:
            raise AssertionError(f"Invalid EMA decay {invalid} was accepted")


def main() -> None:
    test_ema_averages_floating_tensors_and_copies_counters()
    print("ema_state_update: passed")
    test_ema_decay_is_normalized_to_epoch_length()
    print("ema_epoch_normalization: passed")
    test_ema_temporary_swap_restores_raw_model()
    print("ema_temporary_swap: passed")
    test_ema_decay_validation()
    print("ema_decay_validation: passed")


if __name__ == "__main__":
    main()
