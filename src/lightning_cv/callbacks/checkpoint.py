from math import isinf, isnan
from typing import TYPE_CHECKING, Literal, Optional, cast

from lightning_cv.callbacks.base import Callback
from lightning_cv.callbacks.utils import remove_old_checkpoints

if TYPE_CHECKING:
    from lightning_cv.trainer import CrossValidationTrainer


class ModelCheckpoint(Callback):
    __checkpoint_glob__ = "epoch={epoch}_fold*"

    def __init__(
        self,
        save_last: bool = False,
        save_top_k: int = 1,
        every_n_epochs: Optional[int] = None,
        monitor: str = "loss",
        mode: Literal["min", "max"] = "min",
    ):
        self.save_last = save_last
        self.save_top_k = save_top_k
        self.every_n_epochs = every_n_epochs or 1
        self.monitor = monitor
        self.mode = mode

        # map epoch to val metric being monitored
        self.best_k_models: dict[int, float] = dict()

        self.filter_fn = max if self.mode == "min" else min

    def worst_of_best_epochs(self) -> int:
        if self.best_k_models:
            # filter out worst of best models
            # if minimizing value, then take max (worst) score of best models
            # if maximizing value, then take min (worst) score of best models
            return self.filter_fn(
                self.best_k_models, key=self.best_k_models.__getitem__
            )
        return -1

    def on_train_fold_end(self, trainer: "CrossValidationTrainer"):
        # need to check for this directly instead of using trainer property
        # if training stops during the first epoch, this attribute will not exist
        if hasattr(trainer, "_current_val_metrics"):
            # look at validation metrics at the end of "training" all folds
            # for fold in fold -> this does train, val, scheduler_step
            val_metric = cast(float, trainer._current_val_metrics[self.monitor].item())
            if trainer.current_epoch % self.every_n_epochs == 0:
                self._save_topk_ckpt(trainer, val_metric)

    def _save_topk_ckpt(self, trainer: "CrossValidationTrainer", metric: float):
        if self.save_top_k == 0:
            return

        k = len(self.best_k_models) + 1 if self.save_top_k == -1 else self.save_top_k

        if isnan(metric) or isinf(metric):
            # don't store, especially problematic with nan
            # since nan is not considered by min/max
            return

        # store current val loss
        self.best_k_models[trainer.current_epoch] = metric

        kth_best_model_epoch = self.worst_of_best_epochs()

        if len(self.best_k_models) == k + 1:
            del_epoch = kth_best_model_epoch
            if del_epoch != -1:  # pragma: no cover <- this is tested
                self.best_k_models.pop(del_epoch)
                glob = self.__checkpoint_glob__.format(epoch=del_epoch)

                remove_old_checkpoints(trainer.checkpoint_dir, glob)

            # if current epoch would be deleted immediately don't save it
            if del_epoch != trainer.current_epoch:
                self.save(trainer)
        else:
            # otherwise just save the checkpoint
            self.save(trainer)

    def on_train_end(self, trainer: "CrossValidationTrainer"):
        if self.save_last:
            self.save(trainer, suffix="last", include_val_loss=False)

    def save(self, trainer: "CrossValidationTrainer", **kwargs):
        if "include_val_loss" not in kwargs:
            kwargs["include_val_loss"] = True

        if not trainer.should_stop:
            # don't save if trainer was told to stop from external signals
            trainer.save(trainer.fold_manager, **kwargs)

    def __repr__(self) -> str:
        kwargs = dict(
            save_last=self.save_last,
            save_top_k=self.save_top_k,
            every_n_epochs=self.every_n_epochs,
            monitor=self.monitor,
            mode=self.mode,
        )
        kwargs_str = ", ".join(f"{key}={value}" for key, value in kwargs.items())
        return f"{self.__class__.__name__}({kwargs_str})"
