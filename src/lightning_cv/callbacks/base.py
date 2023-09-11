from __future__ import annotations

from torch import Tensor
from torch.optim import Optimizer

import lightning_cv as lcv
from lightning_cv.typehints import KwargType, MetricType, ModelT


class Callback:
    """Simple interface for supported `lightning.Fabric` callbacks during
    cross validation."""

    def on_train_start(self, trainer: "lcv.CrossValidationTrainer"):
        ...

    def on_train_end(self, trainer: "lcv.CrossValidationTrainer"):
        ...

    def on_train_fold_start(self, trainer: "lcv.CrossValidationTrainer"):
        ...

    def on_train_fold_end(self, trainer: "lcv.CrossValidationTrainer"):
        ...

    def on_train_epoch_start_per_fold(
        self,
        trainer: "lcv.CrossValidationTrainer",
        model: ModelT,
        total: int,
        current_epoch: int,
        current_fold: int,
    ):
        ...

    def on_train_epoch_end_per_fold(
        self, trainer: "lcv.CrossValidationTrainer", output: MetricType
    ):
        ...

    def on_train_batch_start_per_fold(
        self, trainer: "lcv.CrossValidationTrainer", batch, batch_idx: int
    ):
        ...

    def on_train_batch_end_per_fold(
        self,
        trainer: "lcv.CrossValidationTrainer",
        output: MetricType,
        batch,
        batch_idx: int,
    ):
        ...

    def on_before_optimizer_step(
        self,
        trainer: "lcv.CrossValidationTrainer",
        optimizer: Optimizer,
        optimizer_idx: int = 0,
    ):
        ...

    def on_before_zero_grad(
        self, trainer: "lcv.CrossValidationTrainer", optimizer: Optimizer
    ):
        ...

    def on_validation_start_per_fold(
        self, trainer: "lcv.CrossValidationTrainer", model: ModelT, total: int
    ):
        ...

    def on_validation_end_per_fold(self, trainer: "lcv.CrossValidationTrainer"):
        ...

    def on_before_log_metrics(
        self,
        trainer: "lcv.CrossValidationTrainer",
        metrics: KwargType,
        optimizer: Optimizer,
    ):
        ...

    def on_validation_batch_start_per_fold(
        self,
        trainer: "lcv.CrossValidationTrainer",
        batch,
        batch_idx: int,
        current_fold: int,
    ):
        ...

    def on_validation_batch_end_per_fold(
        self,
        trainer: "lcv.CrossValidationTrainer",
        output: MetricType,
        batch,
        batch_idx: int,
    ):
        ...

    def on_before_backward(self, trainer: "lcv.CrossValidationTrainer", loss: Tensor):
        ...

    def on_after_backward(self, trainer: "lcv.CrossValidationTrainer"):
        ...

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}()"

    @classmethod
    def available_callbacks(cls) -> set[str]:
        """Returns the set of available callbacks."""
        return {method for method in dir(cls) if method.startswith("on_")}
