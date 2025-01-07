from collections.abc import Mapping
from typing import Literal, Optional, Union, cast

from torch import Tensor

from lightning_cv.typehints import MetricType, Number
from lightning_cv.typehints.module import ForwardOutputType

FoldOutputValue = Union[Number, Tensor]


def fold_idiv(per_fold_tensor: Tensor, value: FoldOutputValue, fold: int):
    assert value != 0.0
    per_fold_tensor[fold] /= value


def fold_iadd(per_fold_tensor: Tensor, value: FoldOutputValue, fold: int):
    per_fold_tensor[fold] += value


def output_getter(out: ForwardOutputType, name: str = "loss") -> Tensor:
    if isinstance(out, Tensor):
        return out

    return out[name]


def validation_update(
    current_metrics: MetricType, output: Mapping[str, Tensor], fold: int
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
    if isinstance(metrics, Tensor):
        # metric is just loss tensor
        metrics = {"loss": metrics}
    return cast(MetricType, metrics)


def prepend_stage_to_metrics(
    stage: Literal["train", "val"], metrics: MetricType
) -> MetricType:
    return {
        f"{stage}_{key}" if not key.startswith(stage) else key: value
        for key, value in metrics.items()
    }


def flatten_train_val_metrics(
    train_metrics: ForwardOutputType,
    val_metrics: Optional[ForwardOutputType] = None,
    prepend_stage: bool = False,
) -> MetricType:
    flat_output: MetricType = dict()

    # need to distinguish between train and val metrics
    if val_metrics is not None:
        val_metrics = convert_output_to_dict(val_metrics)
        val_metrics = prepend_stage_to_metrics("val", val_metrics)

        prepend_stage = True
    else:
        val_metrics = dict()

    train_metrics = convert_output_to_dict(train_metrics)
    if prepend_stage:
        train_metrics = prepend_stage_to_metrics("train", train_metrics)

    flat_output.update(train_metrics)
    flat_output.update(val_metrics)

    return flat_output
