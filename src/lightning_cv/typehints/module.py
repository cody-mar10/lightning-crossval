from __future__ import annotations

from typing import Any, Mapping, Optional, Protocol, runtime_checkable

from lightning import Fabric
from lightning.pytorch.utilities.types import STEP_OUTPUT, LRSchedulerTypeUnion
from torch.optim import Optimizer

ConfigOptimizerOutput = Optimizer | Mapping


@runtime_checkable
class ModelConfig(Protocol):
    fabric: Optional[Fabric]


@runtime_checkable
class ModelT(Protocol):
    fabric: Optional[Fabric]

    def __init__(self, config: ModelConfig):
        ...

    def training_step(self, *args: Any, **kwargs: Any) -> STEP_OUTPUT:
        ...

    def validation_step(self, *args: Any, **kwargs: Any) -> STEP_OUTPUT:
        ...

    def configure_optimizers(self) -> ConfigOptimizerOutput:
        ...

    def lr_scheduler_step(self, scheduler: LRSchedulerTypeUnion, metric: Any | None):
        ...

    def train(self, mode: bool = True):
        ...

    def eval(self):
        ...

    @property
    def estimated_steps(self) -> int:
        ...

    @estimated_steps.setter
    def estimated_steps(self, value: int):
        ...
