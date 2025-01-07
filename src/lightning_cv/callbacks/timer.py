import time
from datetime import timedelta
from typing import TYPE_CHECKING, Any, Literal, Union

from lightning_cv.callbacks.base import Callback
from lightning_cv.callbacks.utils import remove_old_checkpoints
from lightning_cv.typehints import MetricType
from lightning_cv.utils.stopping import StopReasons

if TYPE_CHECKING:
    from lightning_cv.trainer import CrossValidationTrainer


Intervals = Literal["epoch", "step", "fold", "immediately"]


class Timer(Callback):
    TIMEORDER = ("days", "hours", "minutes", "seconds")

    def __init__(
        self,
        duration: Union[str, timedelta, dict[str, int], None] = None,
        buffer: Union[timedelta, dict[str, int], None] = None,
        interval: Intervals = "immediately",
        save_latest: bool = True,
        training_only: bool = False,
        verbose: bool = True,
    ) -> None:
        if isinstance(duration, str):
            duration = Timer._str_to_timedelta_dict(duration)

        if isinstance(duration, dict):
            duration = Timer._dict_to_timedelta(duration)

        if isinstance(buffer, dict):
            buffer = Timer._dict_to_timedelta(buffer)

        self._start_time = time.monotonic()
        self._interval = interval
        self._verbose = verbose
        self._save_latest = save_latest
        cls_name = self.__class__.__name__
        self._repr_fields: list[str] = list()
        if duration is not None:
            self._repr_fields.append(f"duration={repr(duration)}")
            if buffer is not None:
                duration -= buffer
                self._repr_fields.append(f"buffer={repr(buffer)}")
            self._duration = duration.total_seconds()
        else:
            self._duration = None

        self._repr = f"{cls_name}({', '.join(self._repr_fields)})"

        self._check_during_validation = not training_only
        self._current_fold_training_complete = False
        self._stopped = False

    @staticmethod
    def _str_to_timedelta_dict(timestr: str) -> dict[str, int]:
        n_time_units = len(Timer.TIMEORDER)

        times = timestr.strip().split(":")
        n_times = len(times)
        if n_times < 2 or n_times > n_time_units:
            raise ValueError(
                f"Invalid time string: {timestr}. "
                "Should pass DD:HH:MM:SS, HH:MM:SS, or MM:SS"
            )

        start_idx = n_time_units - n_times
        units = Timer.TIMEORDER[start_idx:]

        return {unit: int(intv) for intv, unit in zip(times, units)}

    @staticmethod
    def _dict_to_timedelta(time: dict[str, int]) -> timedelta:
        return timedelta(**time)

    def _elapsed_time(self) -> float:
        return time.monotonic() - self._start_time

    def _is_time_up(self) -> bool:
        if self._duration is None:
            return False
        return self._elapsed_time() >= self._duration

    def _check_time(self, trainer: "CrossValidationTrainer"):
        should_stop = self._is_time_up()
        self._stopped = should_stop

        if trainer.is_distributed:  # pragma: no cover <- can't test distributed yet
            should_stop = trainer.fabric.strategy.reduce_boolean_decision(
                should_stop, all=False
            )

        trainer.should_stop = trainer.should_stop or should_stop
        if should_stop:  # pragma: no cover <- this is actually tested
            trainer.status = StopReasons.TIMEOUT

    def _check_time_and_checkpoint(self, trainer: "CrossValidationTrainer"):
        self._check_time(trainer)
        if trainer.should_stop and self._save_latest:
            remove_old_checkpoints(trainer.checkpoint_dir, "*latest*.ckpt")
            save_kwargs: dict[str, Any] = dict(suffix="latest", include_val_loss=False)
            for fold, state in trainer.fold_manager.items():
                save_kwargs["states"] = state
                save_kwargs["fold"] = fold

                # simplified since trainer now keeps track of this
                save_kwargs["training_complete"] = bool(
                    trainer._finished_cv_loop_per_fold[fold]
                )

                trainer.save(**save_kwargs)

    def _validate_interval_and_check(
        self,
        trainer: "CrossValidationTrainer",
        interval: Union[set[Intervals], Intervals],
    ):
        if isinstance(interval, str):  # pragma: no cover
            interval = {interval}

        if self._interval not in interval:
            return

        if not self._stopped:
            # need to guard here with saving bc if stopping at batch level
            # could start upper level loops and save again
            self._check_time_and_checkpoint(trainer)

    def on_train_fold_start(self, trainer: "CrossValidationTrainer"):
        self._current_fold_training_complete = False

    def on_train_fold_end(self, trainer: "CrossValidationTrainer"):
        self._validate_interval_and_check(trainer, {"fold", "immediately"})

    def on_train_epoch_end_per_fold(
        self, trainer: "CrossValidationTrainer", output: MetricType
    ):
        self._current_fold_training_complete = True
        self._validate_interval_and_check(trainer, {"epoch", "immediately"})

    def on_train_batch_end_per_fold(
        self,
        trainer: "CrossValidationTrainer",
        output: MetricType,
        batch,
        batch_idx: int,
    ):
        self._validate_interval_and_check(trainer, {"step", "immediately"})

    def on_validation_batch_end_per_fold(
        self,
        trainer: "CrossValidationTrainer",
        output: MetricType,
        batch,
        batch_idx: int,
    ):
        if self._check_during_validation:
            self._validate_interval_and_check(trainer, {"step", "immediately"})

    def on_validation_end_per_fold(self, trainer: "CrossValidationTrainer"):
        if self._check_during_validation:
            self._validate_interval_and_check(trainer, {"epoch", "immediately"})

    def __repr__(self) -> str:
        return self._repr
