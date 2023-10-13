from __future__ import annotations

from statistics import mean, stdev
from typing import Optional

import lightning_cv as lcv
from lightning_cv.callbacks.base import Callback
from lightning_cv.typehints import MetricType, ModelT
from lightning_cv.utils import StopReasons


class ValPerfPlateauMonitor(Callback):
    def __init__(
        self,
        monitor="loss",
        min_metric: float = 1e-3,
        threshold_std: float = 1e-6,
    ):
        self.monitor = monitor
        self.min_metric = min_metric
        self.threshold_std = threshold_std

        self.metrics: list[float] = list()

    def mean(self) -> float:
        return mean(self.metrics)

    def std(self, xbar: Optional[float] = None) -> float:
        return stdev(self.metrics, xbar=xbar)

    def push_metric(self, output: MetricType):
        metric = output[self.monitor].item()
        self.metrics.append(metric)

    def clear_metrics(self):
        self.metrics.clear()

    def should_stop(self) -> bool:
        if len(self.metrics) <= 1:
            # likely due to timeout or some other external stop
            # let that callback handle chaning trainer status
            return False

        mean = self.mean()
        std = self.std(xbar=mean)

        # if the std of metric is near 0.0, suggests learning has stopped
        # if the mean is greater than the min_metric, then learning was not optimal since
        # the metric has not approached the min_metric
        return std <= self.threshold_std and mean > self.min_metric

    def report(self, trainer: lcv.CrossValidationTrainer):
        should_stop = self.should_stop()

        if should_stop:
            if trainer.is_distributed:
                should_stop = trainer.fabric.strategy.reduce_boolean_decision(
                    should_stop, all=False
                )
            trainer.should_stop = trainer.should_stop or should_stop
            trainer.status = StopReasons.PERFORMANCE_STALLED

    def on_validation_start_per_fold(
        self, trainer: lcv.CrossValidationTrainer, model: ModelT, total: int
    ):
        self.clear_metrics()

    def on_validation_batch_end_per_fold(
        self,
        trainer: lcv.CrossValidationTrainer,
        output: MetricType,
        batch,
        batch_idx: int,
    ):
        self.push_metric(output)

    def on_validation_end_per_fold(self, trainer: lcv.CrossValidationTrainer):
        self.report(trainer)
