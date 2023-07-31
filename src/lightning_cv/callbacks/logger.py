from __future__ import annotations

import lightning_cv as lcv

from .callbacks import Callback


class LogFlusher(Callback):
    def on_train_end(self, trainer: "lcv.CrossValidationTrainer"):
        # for CSVLogger, need to flush buffer at the end
        trainer.fabric.logger.save()
