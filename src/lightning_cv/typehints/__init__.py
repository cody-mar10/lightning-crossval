from __future__ import annotations

from typing import Any, Mapping, Optional

from lightning.fabric.utilities.types import LRScheduler
from torch import Tensor
from torch.optim import Optimizer

from .data import CrossValDataModuleT, CVDataLoader, Stage
from .module import ConfigOptimizerOutput, ModelConfig, ModelT
from .split import CrossValidator, CVIterator, GroupCrossValidator, Int64Array

Number = float | int
KwargType = dict[str, Any]
ForwardOutputType = Tensor | Mapping[str, Any]
MetricType = dict[str, Tensor]

SchedulerConfigT = Mapping[str, LRScheduler | bool | str | int]
OptimizerConfigT = tuple[Optimizer, Optional[SchedulerConfigT]]


__all__ = [
    # general types
    "Number",
    "KwargType",
    "ForwardOutputType",
    "MetricType",
    "SchedulerConfigT",
    "OptimizerConfigT",
    # split types
    "CrossValidator",
    "CVIterator",
    "GroupCrossValidator",
    "Int64Array",
    # data types
    "CrossValDataModuleT",
    "Stage",
    "CVDataLoader",
    # module types
    "ModelConfig",
    "ModelT",
    "ConfigOptimizerOutput",
]
