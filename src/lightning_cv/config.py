from pathlib import Path
from typing import Literal, Optional, Union

from attrs import define, field
from lightning import LightningModule
from lightning.fabric.wrappers import _FabricModule
from torch.optim import Optimizer

from lightning_cv.fabric import (
    AcceleratorT,
    CallbackT,
    LoggerT,
    PluginsT,
    PrecisionT,
    StrategyT,
)
from lightning_cv.typehints import Number, SchedulerConfigT
from lightning_cv.utils.dataclasses import AttrsDataclassUtilitiesMixin
from lightning_cv.utils.validators import (
    limit_batches_validator,
    non_negative_float,
    non_negative_int,
    positive_int,
)


@define
class FoldState(AttrsDataclassUtilitiesMixin):
    model: Union[LightningModule, _FabricModule]
    optimizer: Optimizer
    scheduler: Optional[SchedulerConfigT] = None


GradClipAlgorithms = Literal["norm", "value"]


@define
class CrossValidationTrainerConfig(AttrsDataclassUtilitiesMixin):
    accelerator: AcceleratorT = "auto"
    strategy: StrategyT = "auto"
    devices: Union[int, str, list[int]] = "auto"
    precision: PrecisionT = "32"
    plugins: PluginsT = None
    callbacks: CallbackT = None
    loggers: LoggerT = None
    max_epochs: int = field(default=1000, validator=positive_int)
    grad_accum_steps: int = field(default=1, validator=positive_int)
    gradient_clip_algorithm: GradClipAlgorithms = "norm"
    gradient_clip_val: float = field(default=0.0, validator=non_negative_float)
    limit_train_batches: Number = field(
        default=1.0,
        validator=limit_batches_validator,  # type: ignore
    )
    limit_val_batches: Number = field(
        default=1.0,
        validator=limit_batches_validator,  # type: ignore
    )
    validation_frequency: int = field(default=1, validator=non_negative_int)
    use_distributed_sampler: bool = True
    checkpoint_dir: Path = field(
        default=Path.cwd().joinpath("checkpoints"), converter=Path
    )
    checkpoint_frequency: int = field(default=1, validator=positive_int)
    monitor: str = field(
        default="val_loss", converter=lambda x: x if x.startswith("val") else f"val_{x}"
    )
