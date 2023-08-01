from __future__ import annotations

import sys
from typing import cast

from lightning import LightningModule
from lightning.pytorch.utilities.model_summary.model_summary import summarize

import lightning_cv as lcv

from .callbacks import Callback


class ModelSummary(Callback):
    def __init__(self, max_depth: int = 1):
        self.max_depth = max_depth

    def on_train_start(self, trainer: "lcv.CrossValidationTrainer"):
        model = cast(LightningModule, trainer.fold_manager[0].model)
        summary = summarize(model, max_depth=self.max_depth)
        sys.stderr.write(repr(summary))

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(max_depth={self.max_depth})"
