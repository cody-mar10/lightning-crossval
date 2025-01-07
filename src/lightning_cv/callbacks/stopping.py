import logging
from typing import TYPE_CHECKING, Callable, Literal, Optional

import torch

from lightning_cv.callbacks.base import Callback
from lightning_cv.typehints import MetricType
from lightning_cv.utils.stopping import StopReasons

if TYPE_CHECKING:
    from lightning_cv.trainer import CrossValidationTrainer


ComparisonOp = Callable[..., torch.Tensor]
logger = logging.getLogger(__name__)


# mostly copied from lightning.pytorch.callbacks.early_stopping.EarlyStopping
class EarlyStopping(Callback):
    mode_dict: dict[str, ComparisonOp] = {"min": torch.lt, "max": torch.gt}
    order_dict = {"min": "<", "max": ">"}

    def __init__(
        self,
        monitor: str = "loss",
        min_delta: float = 0.0,
        patience: int = 3,
        mode: Literal["min", "max"] = "min",
        check_finite: bool = True,
        stopping_threshold: Optional[float] = None,
        divergence_threshold: Optional[float] = None,
        verbose: bool = True,
        # check on train_epoch end -> think no
    ):
        self.monitor = monitor
        self.min_delta = min_delta
        self.patience = patience
        self.mode = mode
        self.check_finite = check_finite
        self.stopping_threshold = stopping_threshold
        self.divergence_threshold = divergence_threshold
        self.verbose = verbose
        self.wait_count = 0
        self.stopped_epoch = 0

        if mode not in self.mode_dict:
            allowed_values = ", ".join(self.mode_dict.keys())
            raise RuntimeError(f"Passed {mode=}, but must be one of {allowed_values}")

        # TODO: why min_delta thing
        inf_tensor = torch.tensor(torch.inf)
        if self.monitor_op == torch.gt:
            # max monitored value
            self.min_delta *= 1  # needs to increase
            self.best_score = -inf_tensor
        else:
            # min monitored value
            self.min_delta *= -1  # needs to decrease
            self.best_score = inf_tensor

    @property
    def monitor_op(self) -> ComparisonOp:
        return self.mode_dict[self.mode]

    def on_train_fold_end(self, trainer: "CrossValidationTrainer"):
        self._run_early_stopping_check(trainer)

    def _run_early_stopping_check(self, trainer: "CrossValidationTrainer"):
        metrics = trainer.current_val_metrics
        current = self._get_current_metric(metrics)
        should_stop, reason = self._evaluate_stopping_criteria(current)

        if trainer.is_distributed:  # pragma: no cover
            # stop if ANY processes decide to stop
            should_stop = trainer.fabric.strategy.reduce_boolean_decision(
                should_stop, all=False
            )

        trainer.should_stop = trainer.should_stop or should_stop
        if should_stop:
            self.stopped_epoch = trainer.current_epoch
            trainer.status = StopReasons.NO_IMPROVEMENT

        if reason and self.verbose:
            self._log_info(trainer, reason)

    def _get_current_metric(self, metrics: MetricType) -> torch.Tensor:
        try:
            current = metrics[self.monitor]
        except KeyError as err:
            raise RuntimeError(
                f"Monitor value {self.monitor} is not reported in validation metrics: "
                f"{metrics}"
            ) from err
        return current

    def _evaluate_stopping_criteria(
        self, current: torch.Tensor
    ) -> tuple[bool, Optional[str]]:
        should_stop = False
        reason = None
        order_str = self.order_dict[self.mode]

        if self.check_finite and not torch.isfinite(current):
            should_stop = True
            reason = (
                f"Monitored metric {self.monitor} = {current} is not finite. Previous "
                f"best value was {self.best_score:.3f}. Signaling Trainer to stop."
            )
        elif self.stopping_threshold is not None and self.monitor_op(
            current, self.stopping_threshold
        ):
            should_stop = True
            reason = (
                "Stopping threshold reached: "
                f"{self.monitor} = {current} {order_str} {self.stopping_threshold}. "
                "Signaling Trainer to stop."
            )
        elif self.divergence_threshold is not None and self.monitor_op(
            -current, self.divergence_threshold
        ):
            should_stop = True
            reason = (
                "Divergence threshold reached: "
                f"{self.monitor} = {current} {order_str} {self.divergence_threshold}. "
                "Signaling Trainer to stop."
            )
        elif self.monitor_op(
            current - self.min_delta, self.best_score.to(current.device)
        ):
            ### RESET counter back to 0, ie there was improvement
            should_stop = False
            reason = self._improvement_message(current)
            self.best_score = current
            self.wait_count = 0
        else:
            self.wait_count += 1
            if self.wait_count >= self.patience:
                should_stop = True
                reason = (
                    f"Monitored metric {self.monitor} did not improve in the last "
                    f"{self.wait_count} records. Best score: {self.best_score:.3f}. "
                    "Signaling Trainer to stop."
                )

        return should_stop, reason

    def _improvement_message(self, current: torch.Tensor) -> str:
        if torch.isfinite(self.best_score):
            improvement = abs(self.best_score - current)
            msg = (
                f"Metric {self.monitor} improved by {improvement:.3f} >= min_delta "
                f"= {abs(self.min_delta)}. New best score: {current:.3f}"
            )
        else:
            msg = f"Metric {self.monitor} improved. New best score: {current:.3f}"
        return msg

    @staticmethod
    def _log_info(trainer: "CrossValidationTrainer", message: str):
        if trainer.is_global_zero:  # pragma: no cover
            logger.info(f"[rank: 0]: {message}")
