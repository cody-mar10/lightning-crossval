from __future__ import annotations

from lightning_cv.typehints import ModelT

from .callbacks import Callback


class TrainMode(Callback):
    def on_train_epoch_start_per_fold(
        self,
        model: ModelT,
        total: int,
        current_epoch: int,
        current_fold: int,
    ):
        model.train()


class EvalMode(Callback):
    def on_validation_start_per_fold(self, model: ModelT):
        model.eval()
