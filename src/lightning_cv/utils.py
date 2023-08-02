from __future__ import annotations

from typing import Literal, Mapping, Optional, cast

import torch
from lightning.fabric.strategies.ddp import DDPStrategy
from lightning.fabric.strategies.deepspeed import DeepSpeedStrategy
from lightning.fabric.strategies.fsdp import FSDPStrategy
from lightning.fabric.strategies.strategy import Strategy
from lightning.fabric.strategies.xla import XLAStrategy

from .typehints import ForwardOutputType, MetricType, Number

_DISTRIBUTED_STRATEGIES = (DDPStrategy, DeepSpeedStrategy, FSDPStrategy, XLAStrategy)


def is_distributed(strategy: Strategy) -> bool:
    if hasattr(strategy, "is_distributed"):
        return strategy.is_distributed  # type: ignore

    return isinstance(strategy, _DISTRIBUTED_STRATEGIES)


def detach(x: torch.Tensor) -> torch.Tensor:
    return x.detach()


def output_getter(out: ForwardOutputType, name: str = "loss") -> torch.Tensor:
    if isinstance(out, torch.Tensor):
        return out

    return out[name]


def average_and_detach(x: torch.Tensor) -> torch.Tensor:
    return x.mean().detach()


def fold_idiv(per_fold_tensor: torch.Tensor, value: Number | torch.Tensor, fold: int):
    assert value != 0.0
    per_fold_tensor[fold] /= value


def fold_iadd(per_fold_tensor: torch.Tensor, value: Number | torch.Tensor, fold: int):
    per_fold_tensor[fold] += value


def validation_update(
    current_metrics: MetricType,
    output: Mapping[str, torch.Tensor],
    fold: int,
):
    for metric, value in output.items():
        if metric not in current_metrics:
            # this will exist bc it is created during trainer setup
            per_fold_loss_tensor = current_metrics["loss"]
            n_folds = int(per_fold_loss_tensor.size(0))

            # created the empty tensor to store values in for each fold
            current_metrics[metric] = per_fold_loss_tensor.new_zeros(size=(n_folds,))

        fold_iadd(per_fold_tensor=current_metrics[metric], value=value, fold=fold)


def convert_output_to_dict(metrics: ForwardOutputType) -> MetricType:
    if isinstance(metrics, torch.Tensor):
        # metric is just loss tensor
        metrics = {"loss": metrics}
    return cast(MetricType, metrics)


def prepend_stage_to_metrics(
    stage: Literal["train", "val"], metrics: Mapping[str, torch.Tensor]
) -> MetricType:
    return {f"{stage}_{key}": value for key, value in metrics.items()}


def flatten_train_val_metrics(
    train_metrics: ForwardOutputType,
    val_metrics: Optional[ForwardOutputType] = None,
    prepend_stage: bool = False,
) -> MetricType:
    flat_output: MetricType = dict()

    train_metrics = convert_output_to_dict(train_metrics)
    if prepend_stage:
        train_metrics = prepend_stage_to_metrics("train", train_metrics)
    flat_output.update(train_metrics)

    if val_metrics is not None:
        val_metrics = convert_output_to_dict(val_metrics)
        if prepend_stage:
            val_metrics = prepend_stage_to_metrics("val", val_metrics)
        flat_output.update(val_metrics)

    return flat_output
