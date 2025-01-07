from pathlib import Path
from typing import TYPE_CHECKING, Any, Mapping, Optional, TypeVar, Union

if TYPE_CHECKING:
    from lightning.fabric.utilities.types import LRScheduler
    from torch import Tensor
    from torch.optim import Optimizer

from lightning_cv.typehints.data import CVDataLoader, Stage
from lightning_cv.typehints.module import ConfigOptimizerOutput

Number = Union[float, int]
KwargType = dict[str, Any]
MetricType = dict[str, "Tensor"]

SchedulerConfigT = Mapping[str, Union["LRScheduler", bool, str, int]]
OptimizerConfigT = tuple["Optimizer", Optional[SchedulerConfigT]]

_T = TypeVar("_T")

FilePath = Union[str, Path]
