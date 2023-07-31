from __future__ import annotations

from functools import partial
from typing import Literal, Optional, cast

import lightning_cv as lcv

from .callbacks import Callback


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
        self.filename = "epoch={epoch}-fold={fold}-val_loss={loss:.4f}.ckpt"
        self.save_last = save_last
        self.save_top_k = save_top_k
        self.every_n_epochs = every_n_epochs or 1
        self.monitor = monitor
        self.mode = mode

        self.best_k_models: dict[int, float] = dict()
        self.kth_best_model_epoch = -1

        self.filter_fn = partial(
            max if self.mode == "min" else min, key=self.best_k_models.__getitem__
        )

    def on_train_fold_end(self, trainer: "lcv.CrossValidationTrainer"):
        # look at validation metrics at the end of "training" all folds
        # for fold in fold -> this does train, val, scheduler_step
        val_metric = cast(float, trainer.current_val_metrics[self.monitor].item())
        if trainer.current_epoch % self.every_n_epochs == 0:
            self._save_topk_ckpt(trainer, val_metric)

    def _save_topk_ckpt(self, trainer: "lcv.CrossValidationTrainer", metric: float):
        if self.save_top_k == 0:
            return

        k = len(self.best_k_models) + 1 if self.save_top_k == -1 else self.save_top_k

        # store current val loss
        self.best_k_models[trainer.current_epoch] = metric

        # filter out worst of best models
        # if minimizing value, then take max (worst) score of best models
        # if maximizing value, then take min (worst) score of best models
        self.kth_best_model_epoch = cast(int, self.filter_fn(self.best_k_models))

        if len(self.best_k_models) == k + 1:
            del_epoch = self.kth_best_model_epoch
            if del_epoch != -1:
                self.best_k_models.pop(del_epoch)
                glob = self.__checkpoint_glob__.format(epoch=del_epoch)
                for file in trainer.checkpoint_dir.glob(glob):
                    file.unlink()

            # if current epoch would be deleted immediately don't save it
            if del_epoch != trainer.current_epoch:
                trainer.save(trainer.fold_manager)
        else:
            # otherwise just save the checkpoint
            trainer.save(trainer.fold_manager)

    def on_train_end(self, trainer: "lcv.CrossValidationTrainer"):
        if self.save_last:
            trainer.save(trainer.fold_manager, name="fold={fold}-last.ckpt")

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
