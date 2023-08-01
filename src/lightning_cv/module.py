from __future__ import annotations

from typing import Generic, Optional, TypeVar
from warnings import warn

from lightning import Fabric, LightningModule
from lightning.pytorch.utilities.types import STEP_OUTPUT
from pydantic import BaseModel


class BaseModelConfig(BaseModel):
    fabric: Optional[Fabric] = None

    class Config:
        arbitrary_types_allowed = True


ModelConfig = TypeVar("ModelConfig", bound=BaseModelConfig)


# TODO: this needs to be a mixin instead since the LightningModule must be inited first
class CrossValModuleMixin(Generic[ModelConfig]):
    def __init__(self, config: ModelConfig):
        if config.fabric is None:
            raise ValueError("Config must have an instance of `lightning.Fabric`")

        self._fabric = config.fabric

        self._estimated_steps = -1

    @property
    def estimated_steps(self) -> int:
        if self._estimated_steps == -1:
            warn("A trainer has not updated the estimated number of steps.")
        return self._estimated_steps

    @estimated_steps.setter
    def estimated_steps(self, value: int):
        if value <= 0:
            raise ValueError(
                "The number of estimated training steps must be an integer > 0."
            )

        self._estimated_steps = value


class CrossValModule(CrossValModuleMixin, LightningModule, Generic[ModelConfig]):
    def __init__(self, config: ModelConfig):
        LightningModule.__init__(self)
        CrossValModuleMixin.__init__(self, config)

    # change signature since validation_step is required
    def validation_step(self, *args, **kwargs) -> STEP_OUTPUT:
        ...


ModelT = TypeVar("ModelT", bound=CrossValModule)
