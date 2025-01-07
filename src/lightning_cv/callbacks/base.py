from typing import TYPE_CHECKING, TypeVar, Union

from torch import Tensor
from torch.optim import Optimizer

from lightning_cv.typehints import KwargType, MetricType

if TYPE_CHECKING:
    from lightning import Callback as LightingCallback
    from lightning import LightningModule

    from lightning_cv.trainer import CrossValidationTrainer


class Callback:
    """Simple interface for supported `lightning.Fabric` callbacks during
    cross validation."""

    def on_train_start(self, trainer: "CrossValidationTrainer"): ...

    def on_train_end(self, trainer: "CrossValidationTrainer"): ...

    def on_train_fold_start(self, trainer: "CrossValidationTrainer"): ...

    def on_train_fold_end(self, trainer: "CrossValidationTrainer"): ...

    def on_train_epoch_start_per_fold(
        self,
        trainer: "CrossValidationTrainer",
        model: "LightningModule",
        total: int,
        current_epoch: int,
        current_fold: int,
    ): ...

    def on_train_epoch_end_per_fold(
        self, trainer: "CrossValidationTrainer", output: MetricType
    ): ...

    def on_train_batch_start_per_fold(
        self, trainer: "CrossValidationTrainer", batch, batch_idx: int
    ): ...

    def on_train_batch_end_per_fold(
        self,
        trainer: "CrossValidationTrainer",
        output: MetricType,
        batch,
        batch_idx: int,
    ): ...

    def on_before_optimizer_step(
        self,
        trainer: "CrossValidationTrainer",
        optimizer: Optimizer,
        optimizer_idx: int = 0,
    ): ...

    def on_before_zero_grad(
        self, trainer: "CrossValidationTrainer", optimizer: Optimizer
    ): ...

    def on_validation_start_per_fold(
        self, trainer: "CrossValidationTrainer", model: "LightningModule", total: int
    ): ...

    def on_validation_end_per_fold(self, trainer: "CrossValidationTrainer"): ...

    def on_before_log_metrics(
        self,
        trainer: "CrossValidationTrainer",
        metrics: KwargType,
        optimizer: Optimizer,
    ): ...

    def on_validation_batch_start_per_fold(
        self,
        trainer: "CrossValidationTrainer",
        batch,
        batch_idx: int,
        current_fold: int,
    ): ...

    def on_validation_batch_end_per_fold(
        self,
        trainer: "CrossValidationTrainer",
        output: MetricType,
        batch,
        batch_idx: int,
    ): ...

    def on_before_backward(self, trainer: "CrossValidationTrainer", loss: Tensor): ...

    def on_after_backward(self, trainer: "CrossValidationTrainer"): ...

    def __repr__(self) -> str:  # pragma: no cover

        return f"{self.__class__.__name__}()"

    @classmethod
    def available_callbacks(cls) -> set[str]:
        """Returns the set of available callbacks."""
        return {method for method in dir(cls) if method.startswith("on_")}


_CallbackT = TypeVar("_CallbackT", Callback, "LightningModule", "LightingCallback")
CallbackT = Union[list[_CallbackT], _CallbackT, None]
