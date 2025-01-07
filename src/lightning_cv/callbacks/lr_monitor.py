from typing import TYPE_CHECKING

from torch.optim import Optimizer

from lightning_cv.callbacks.base import Callback
from lightning_cv.typehints import KwargType

if TYPE_CHECKING:
    from lightning_cv.trainer import CrossValidationTrainer


class LearningRateMonitor(Callback):
    def on_before_log_metrics(
        self,
        trainer: "CrossValidationTrainer",
        metrics: KwargType,
        optimizer: Optimizer,
    ):
        # TODO: make more complex
        lr = optimizer.param_groups[0]["lr"]
        metrics["lr"] = lr
