from __future__ import annotations

from pathlib import Path
from typing import Any, Literal, Optional

from lightning.fabric.accelerators.accelerator import Accelerator
from lightning.fabric.loggers.logger import Logger
from lightning.fabric.strategies.strategy import Strategy
from lightning.fabric.wrappers import _FabricModule
from pydantic import BaseModel, validator
from torch.optim import Optimizer

from .callbacks import Callback
from .typehints import ModelT, Number, SchedulerConfigT


class FoldState(BaseModel):
    model: ModelT | _FabricModule
    optimizer: Optimizer
    scheduler: Optional[SchedulerConfigT] = None

    class Config:
        arbitrary_types_allowed = True


def validate_range(value: float, low: float, high: float):
    if not (low <= value <= high):
        raise ValueError(f"{value=} not in range [{low}, {high}]")


Accelerators = Literal["auto", "cpu", "gpu", "tpu"]
Strategies = Literal["ddp", "ddp_spawn", "ddp_notebook", "fsdp", "auto"]
Precision = Literal["32", "16-mixed", "bf16-mixed"]


class CrossValidationTrainerConfig(BaseModel):
    accelerator: Accelerators | Accelerator = "auto"
    strategy: Strategies | Strategy = "auto"
    devices: list[int] | str | int = "auto"
    precision: Precision | int = "32"
    plugins: Optional[str | Any] = None
    callbacks: Optional[list[Callback] | Callback] = None
    loggers: Optional[Logger | list[Logger]] = None
    max_epochs: int = 1000
    grad_accum_steps: int = 1
    limit_train_batches: Number = 1.0
    limit_val_batches: Number = 1.0
    validation_frequency: int = 1
    use_distributed_sampler: bool = True
    checkpoint_dir: Path = Path.cwd().joinpath("checkpoints")
    checkpoint_frequency: int = 1
    monitor: str = "val_loss"

    @validator("limit_train_batches", "limit_val_batches", pre=True)
    def validate_float_range(cls, value: Number) -> Number:
        if not isinstance(value, int):
            validate_range(value, low=0.0, high=1.0)
        return value

    @validator("monitor", pre=True)
    def prepend_stage(cls, value: str) -> str:
        if not value.startswith("val_"):
            value = f"val_{value}"

        return value

    class Config:
        arbitrary_types_allowed = True
