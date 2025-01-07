from typing import TYPE_CHECKING

from lightning_cv.callbacks.base import Callback

if TYPE_CHECKING:
    from lightning import LightningModule

    from lightning_cv.trainer import CrossValidationTrainer


class TrainMode(Callback):
    def on_train_epoch_start_per_fold(
        self,
        trainer: "CrossValidationTrainer",
        model: "LightningModule",
        total: int,
        current_epoch: int,
        current_fold: int,
    ):
        model.train()


class EvalMode(Callback):
    def on_validation_start_per_fold(
        self, trainer: "CrossValidationTrainer", model: "LightningModule", total: int
    ):
        model.eval()
