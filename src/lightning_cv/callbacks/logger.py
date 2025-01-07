from typing import TYPE_CHECKING

from lightning_cv.callbacks.base import Callback

if TYPE_CHECKING:
    from lightning_cv.trainer import CrossValidationTrainer


class LogFlusher(Callback):
    def on_train_end(self, trainer: "CrossValidationTrainer"):
        # for CSVLogger, need to flush buffer at the end
        trainer.fabric.logger.save()
