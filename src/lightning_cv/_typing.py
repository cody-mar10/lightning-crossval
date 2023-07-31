from __future__ import annotations

from typing import Any, Mapping, Optional, TypeVar

from lightning.fabric.utilities.types import LRScheduler
from pydantic import BaseModel
from torch import Tensor
from torch.optim import Optimizer

from .module import ModelConfig, ModelT  # noqa: F401
from .split import Int64Array  # noqa: F401

Number = float | int
KwargType = dict[str, Any]
ForwardOutputType = Tensor | Mapping[str, Any]
MetricType = dict[str, Tensor]

ConfigOptimizerOutput = Optimizer | Mapping
SchedulerConfigT = Mapping[str, LRScheduler | bool | str | int]
OptimizerConfigT = tuple[Optimizer, Optional[SchedulerConfigT]]

PydanticConfig = TypeVar("PydanticConfig", bound=BaseModel)
