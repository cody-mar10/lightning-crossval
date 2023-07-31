from __future__ import annotations

import lightning_cv as lcv

from .callbacks import Callback


class TrainMode(Callback):
    def on_train_epoch_start_per_fold(
        self,
        model: "lcv.CrossValModule",
        total: int,
        current_epoch: int,
        current_fold: int,
    ):
        model.train()


class EvalMode(Callback):
    def on_validation_start_per_fold(self, model: "lcv.CrossValModule"):
        model.eval()
