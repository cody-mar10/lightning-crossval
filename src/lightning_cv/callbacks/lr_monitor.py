from __future__ import annotations

from torch.optim import Optimizer

from lightning_cv.typehints import KwargType

from .callbacks import Callback


class LearningRateMonitor(Callback):
    def on_before_log_metrics(self, metrics: KwargType, optimizer: Optimizer):
        # TODO: make more complex
        lr = optimizer.param_groups[0]["lr"]
        metrics["lr"] = lr
