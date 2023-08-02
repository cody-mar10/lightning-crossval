from __future__ import annotations

import sys
from importlib.util import find_spec
from typing import Literal, Optional, cast

from lightning_utilities.core.apply_func import apply_to_collection
from torch import Tensor

import lightning_cv as lcv
from lightning_cv.typehints import MetricType

if find_spec("ipywidgets") is not None:
    from tqdm.auto import tqdm as _tqdm
else:
    from tqdm import tqdm as _tqdm

from .callbacks import Callback

_PAD_SIZE = 5

ExtractedMetricType = dict[str, float]


class Tqdm(_tqdm):  # type: ignore
    @staticmethod
    def format_num(n: int | float | str) -> str:
        """Add additional padding to the formatted numbers."""
        should_be_padded = isinstance(n, (float, str))
        if not isinstance(n, str):
            n = _tqdm.format_num(n)
            assert isinstance(n, str)
        if should_be_padded and "e" not in n:
            if "." not in n and len(n) < _PAD_SIZE:
                try:
                    _ = float(n)
                except ValueError:
                    return n
                n += "."
            n += "0" * (_PAD_SIZE - len(n))
        return n


class TqdmProgressBar(Callback):
    def init_progbar(self, stage: Literal["train", "validate"], **kwargs):
        progbar = Tqdm(
            dynamic_ncols=True,
            leave=stage != "validate",
            smoothing=0.0,
            file=sys.stdout,
            **kwargs,
        )

        return progbar

    def on_train_start(self, trainer: "lcv.CrossValidationTrainer"):
        self.train_pbar = self.init_progbar(stage="train")

    def on_train_epoch_start_per_fold(
        self,
        model: "lcv.CrossValModule",
        total: int,
        current_epoch: int,
        current_fold: int,
    ):
        # update total
        self.train_pbar.reset(total=total)
        # reset batch number to 0
        self.train_pbar.initial = 0
        self.current_epoch = current_epoch
        self.train_pbar.set_description(f"Epoch {current_epoch} Fold {current_fold}")

    def on_train_batch_end_per_fold(self, output: MetricType, batch, batch_idx: int):
        n = batch_idx + 1
        _update_pbar(self.train_pbar, value=n)
        self._train_log(output)

    def on_train_epoch_end_per_fold(self, output: MetricType):
        self._train_log(output)

    def _train_log(self, output: MetricType):
        prefix = "train" if self.current_epoch == 0 else None
        metrics = self.get_metrics(output)
        _set_postfix(self.train_pbar, metrics, prefix)

    def on_train_end(self, trainer: "lcv.CrossValidationTrainer"):
        self.train_pbar.close()

    def on_validation_start_per_fold(self, model: "lcv.CrossValModule"):
        self.val_pbar = self.init_progbar(stage="validate")

    def on_validation_batch_start_per_fold(
        self, batch, batch_idx: int, total: int, current_fold: int
    ):
        # update total
        self.val_pbar.reset(total=total)
        # reset batch number to 0
        self.val_pbar.initial = 0
        self.val_pbar.set_description(f"Fold {current_fold} validation")

    def on_validation_batch_end_per_fold(
        self, output: MetricType, batch, batch_idx: int
    ):
        n = batch_idx + 1
        _update_pbar(self.val_pbar, value=n)

    def on_validation_end_per_fold(self, trainer: "lcv.CrossValidationTrainer"):
        if self.train_pbar is not None:
            # self.train_pbar.set_postfix() # TODO:
            pass

        # self.val_pbar.clear()
        self.val_pbar.close()

    def get_metrics(self, candidates: MetricType) -> ExtractedMetricType:
        float_candidates = cast(
            ExtractedMetricType,
            apply_to_collection(data=candidates, dtype=Tensor, function=_extract_value),
        )

        return float_candidates


def _extract_value(tensor: Tensor):
    return tensor.item()


def _update_pbar(pbar: Tqdm, value: int):
    if not pbar.disable:
        pbar.n = value
        pbar.refresh()


def _set_postfix(
    pbar: Tqdm,
    metric: ExtractedMetricType,
    prefix: Optional[str] = None,
):
    postfix = dict()
    if prefix is None:
        # don't append a prefix name like train/val
        prefix = "{key}"
    else:
        prefix = prefix.strip() + "_{key}"

    for key, value in metric.items():
        postfix[prefix.format(key=key)] = f"{value:.3f}"

    if postfix:
        pbar.set_postfix(postfix)
