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
        model: ModelT,
        total: int,
        current_epoch: int,
        current_fold: int,
    ):
        ...

    def on_train_epoch_end_per_fold(self, output: MetricType):
        ...

    def on_train_batch_start_per_fold(self, batch, batch_idx: int):
        ...

    def on_train_batch_end_per_fold(self, output: MetricType, batch, batch_idx: int):
        ...

    def on_before_optimizer_step(self, optimizer: Optimizer, optimizer_idx: int = 0):
        ...

    def on_before_zero_grad(self, optimizer: Optimizer):
        ...

    def on_validation_start_per_fold(self, model: ModelT):
        ...

    def on_validation_end_per_fold(self, trainer: "lcv.CrossValidationTrainer"):
        ...

    def on_before_log_metrics(self, metrics: KwargType, optimizer: Optimizer):
        ...

    def on_validation_batch_start_per_fold(
        self, batch, batch_idx: int, total: int, current_fold: int
    ):
        ...

    def on_validation_batch_end_per_fold(
        self, output: MetricType, batch, batch_idx: int
    ):
        ...

    def on_before_backward(self, loss: Tensor):
        ...

    def on_after_backward(self):
        ...

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}()"
