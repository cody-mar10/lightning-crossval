from __future__ import annotations

from torch.optim import Optimizer

import lightning_cv as lcv
from lightning_cv.callbacks import Callback
from lightning_cv.typehints import KwargType


class LearningRateMonitor(Callback):
    def on_before_log_metrics(
        self,
        trainer: "lcv.CrossValidationTrainer",
        metrics: KwargType,
        optimizer: Optimizer,
    ):
        # TODO: make more complex
        lr = optimizer.param_groups[0]["lr"]
        metrics["lr"] = lr
