import sys
from typing import TYPE_CHECKING, cast

from lightning import LightningModule
from lightning.pytorch.utilities.model_summary.model_summary import summarize

from lightning_cv.callbacks.base import Callback

if TYPE_CHECKING:
    from lightning_cv.trainer import CrossValidationTrainer


class ModelSummary(Callback):
    def __init__(self, max_depth: int = 1):
        self.max_depth = max_depth

    def on_train_start(self, trainer: "CrossValidationTrainer"):
        model = cast(LightningModule, trainer.fold_manager[0].model)
        summary = summarize(model, max_depth=self.max_depth)
        sys.stderr.write(f"{repr(summary)}\n")

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(max_depth={self.max_depth})"
