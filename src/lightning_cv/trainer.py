from __future__ import annotations

import math
import os
from copy import copy
from functools import lru_cache, partial
from pathlib import Path
from typing import Any, Literal, Mapping, Optional, cast

import torch
from lightning import Fabric
from lightning.fabric.utilities.types import LRScheduler
from lightning_utilities.core.apply_func import apply_to_collection
from torch.optim import Optimizer
from torch.utils.data import DataLoader

from .callbacks import (
    Callback,
    EvalMode,
    LogFlusher,
    ModelCheckpoint,
    ModelSummary,
    TqdmProgressBar,
    TrainMode,
)
from .config import CrossValidationTrainerConfig, FoldState
from .data import CrossValDataModuleT
from .module import ModelConfig
from .typehints import (
    ConfigOptimizerOutput,
    MetricType,
    ModelT,
    Number,
    OptimizerConfigT,
    SchedulerConfigT,
)
from .utils import (
    convert_output_to_dict,
    detach,
    flatten_train_val_metrics,
    fold_idiv,
    is_distributed,
    validation_update,
)


class CrossValidationTrainer:
    __fabric_keys__ = {
        "accelerator",
        "strategy",
        "devices",
        "precision",
        "plugins",
        "callbacks",
        "loggers",
    }

    __reset_fn__ = partial(torch.fill_, value=0.0)

    def __init__(self, model_type: type[ModelT], config: CrossValidationTrainerConfig):
        self.config = config
        self.model_type = model_type

        self.global_step_per_fold: dict[int, int] = dict()
        self.current_epoch: int = 0
        self.current_fold: int = 0
        self.state = "train"
        self.setup_complete = False
        self.is_first_epoch = True

        self.should_stop = False
        self.current_train_metrics: MetricType = dict()
        self._current_val_metrics_per_fold: MetricType = dict()

        self._add_default_callbacks(self.config.callbacks)

        fabric_kwargs = self.config.dict(include=self.__fabric_keys__)
        self.fabric = Fabric(**fabric_kwargs)

    @property
    def current_val_metrics(self) -> MetricType:
        if not hasattr(self, "_current_val_metrics"):
            raise RuntimeError(
                "Current validation metrics aren't available until after the 1st epoch"
            )

        return self._current_val_metrics

    @current_val_metrics.setter
    def current_val_metrics(self, value: MetricType):
        self._current_val_metrics = value

    @property
    @lru_cache(1)
    def checkpoint_dir(self) -> Path:
        try:
            logger_expt_name = self.fabric.logger.name or "expt_0"
            logger_version_no = self.fabric.logger.version or "version_0"
            if isinstance(logger_version_no, int):
                logger_version_no = f"version_{logger_version_no}"
            if self.fabric.logger.root_dir is not None:
                checkpoint_dir = Path(self.fabric.logger.root_dir)
            else:
                checkpoint_dir = Path("checkpoints")
        except AttributeError:
            # no logger attached
            logger_expt_name = "expt_0"
            logger_version_no = "version_0"
            checkpoint_dir = self.config.checkpoint_dir

        return checkpoint_dir / logger_expt_name / logger_version_no

    @property
    def is_distributed(self) -> bool:
        return is_distributed(self.fabric.strategy)

    def _standardize_callbacks(
        self, callbacks: Optional[Callback | list[Callback]] = None
    ) -> list[Callback]:
        if callbacks is None:
            callbacks = list()
        elif isinstance(callbacks, Callback):
            callbacks = [callbacks]
        else:
            callbacks = copy(callbacks)

        return callbacks

    def _add_default_callbacks(
        self, callbacks: Optional[Callback | list[Callback]] = None
    ):
        callbacks = self._standardize_callbacks(callbacks)
        callbacks.extend([TrainMode(), EvalMode(), LogFlusher(), TqdmProgressBar()])

        has_model_summary = False
        has_model_checkpoint = False
        for callback in callbacks:
            if isinstance(callback, ModelCheckpoint):
                has_model_checkpoint = True
            elif isinstance(callback, ModelSummary):
                has_model_summary = True

        if not has_model_checkpoint:
            callbacks.append(ModelCheckpoint(save_last=True))

        if not has_model_summary:
            callbacks.append(ModelSummary())

        self.config.callbacks = callbacks

    def setup(self, datamodule: CrossValDataModuleT, model_config: ModelConfig):
        # only setup once
        if not self.setup_complete:
            self.fabric.launch()
            datamodule.setup("fit")

            # setup n_folds models and optimizers
            self.fold_manager: dict[int, FoldState] = dict()
            self.global_step_per_fold.clear()
            self.estimated_steps: dict[int, int] = dict()

            if model_config.fabric is None:
                model_config.fabric = self.fabric

            for fold, (train_loader, val_loader) in enumerate(
                datamodule.train_val_dataloaders()
            ):
                self.estimated_steps[fold] = self._estimated_steps(len(train_loader))

                model: ModelT = self.model_type(model_config)
                model.estimated_steps = self.estimated_steps[fold]
                config = self._parse_optimizers_schedulers(model.configure_optimizers())
                optimizer, scheduler_cfg = config

                optimizer: Optimizer
                # ignore fact that fabric returns a wrapper
                # ignore setup takes a torch.nn.Module or a lightning.LightningModule
                model, optimizer = self.fabric.setup(model, optimizer)  # type: ignore

                self.fold_manager[fold] = FoldState(
                    model=model, optimizer=optimizer, scheduler=scheduler_cfg
                )
                # init global steps for each fold to 0
                self.global_step_per_fold[fold] = 0

            self._current_val_metrics_per_fold["loss"] = torch.zeros(
                size=(len(self.fold_manager),), device=self.fabric.device
            )

            self.setup_complete = True

    def _estimated_steps(self, training_batches: int):
        max_estimated_steps = (
            math.ceil(training_batches / self.config.grad_accum_steps)
            * self.config.max_epochs
        )

        return max_estimated_steps

    def train_with_cross_validation(
        self, datamodule: CrossValDataModuleT, model_config: ModelConfig
    ):
        # set up n_folds models and optimizers, and creates fold_manager attribute
        # also calls self.fabric.launch()
        self.setup(datamodule=datamodule, model_config=model_config)
        self.fabric.call("on_train_start", trainer=self)
        while not self.should_stop:
            # calls train_loop, val_loop, step_scheduler per fold
            self.cross_val_loop(datamodule=datamodule)

            self.current_epoch += 1

            # stopping condition on epoch level
            if self.current_epoch >= self.config.max_epochs:
                self.should_stop = True

            self.is_first_epoch = False

        self.fabric.call("on_train_end", trainer=self)
        # reset for next fit call
        self.should_stop = False

    def cross_val_loop(self, datamodule: CrossValDataModuleT):
        # called once per epoch
        # train and validate each fold
        self.fabric.call("on_train_fold_start", trainer=self)
        foldloaders = datamodule.train_val_dataloaders(shuffle=True)
        for fold, (train_loader, val_loader) in enumerate(foldloaders):
            train_loader, val_loader = self.fabric.setup_dataloaders(
                train_loader, val_loader
            )

            self.current_fold = fold
            state = self.fold_manager[fold]
            # ignore fact that lightning.fabric wraps model
            model = cast(ModelT, state.model)
            optimizer = state.optimizer
            scheduler_cfg = state.scheduler

            self.train_loop(
                model=model,
                optimizer=optimizer,
                train_loader=train_loader,
                scheduler_cfg=scheduler_cfg,
            )

            self.val_loop(model=model, val_loader=val_loader, optimizer=optimizer)

            self.step_scheduler(
                model=model,
                scheduler_cfg=scheduler_cfg,
                level="epoch",
                current_value=self.current_epoch,
            )
        # update this value every epoch
        # this stores the average validation loss per fold

        # need to get all metrics in distributed setting
        val_metrics = cast(
            dict[str, torch.Tensor],
            self.fabric.all_gather(self._current_val_metrics_per_fold),
        )

        # current_val_metrics = average over folds
        self.current_val_metrics = apply_to_collection(
            val_metrics, dtype=torch.Tensor, function=torch.mean
        )
        # reset values
        apply_to_collection(
            self._current_val_metrics_per_fold,
            dtype=torch.Tensor,
            function=self.__reset_fn__,
        )

        self.fabric.call("on_train_fold_end", trainer=self)

    def train_loop(
        self,
        model: ModelT,
        optimizer: Optimizer,
        train_loader: DataLoader,
        scheduler_cfg: Optional[SchedulerConfigT] = None,
    ):
        self.state = "train"
        n_batches = len(train_loader)
        batch_limit = self._limit_batches(n_batches, self.config.limit_train_batches)
        self.fabric.call(
            "on_train_epoch_start_per_fold",
            model=model,
            total=batch_limit,
            current_epoch=self.current_epoch,
            current_fold=self.current_fold,
        )  # calls model.train()

        output = 0.0  # ignore mypy unbound warnings
        for batch_idx, batch in enumerate(train_loader):
            # end epoch if stopping training completely or
            # max batches for this epoch reached
            if self.should_stop or batch_idx >= batch_limit:
                break

            self.fabric.call(
                "on_train_batch_start_per_fold", batch=batch, batch_idx=batch_idx
            )

            # check if optimizer should step in gradient accumulation
            should_optim_step = (
                self.global_step_per_fold[self.current_fold]
                % self.config.grad_accum_steps
                == 0
            )
            if should_optim_step:
                # currently only supports a single optimizer
                self.fabric.call(
                    "on_before_optimizer_step", optimizer=optimizer, optimizer_idx=0
                )

                # optimizer step runs train step internally through closure
                closure = partial(
                    self.training_step,
                    model=model,
                    batch=batch,
                    batch_idx=batch_idx,
                )
                optimizer.step(closure)  # type: ignore
                self.fabric.call("on_before_zero_grad", optimizer=optimizer)

                optimizer.zero_grad()

                metrics = {
                    "epoch": self.current_epoch,
                    "fold": self.current_fold,
                    "stage": "train",
                    **self.current_train_metrics,
                }

                self.fabric.call(
                    "on_before_log_metrics", metrics=metrics, optimizer=optimizer
                )
                self.fabric.log_dict(
                    metrics=metrics,
                    step=self.global_step_per_fold[self.current_fold],
                )

                self.step_scheduler(
                    model,
                    scheduler_cfg,
                    level="step",
                    current_value=self.global_step_per_fold[self.current_fold],
                )

            else:
                # gradient accumulation -> no optimizer step
                self.training_step(model=model, batch=batch, batch_idx=batch_idx)

            # after first epoch, output now has both the current training
            # AND validation metrics
            val_metrics = self.current_val_metrics if not self.is_first_epoch else None
            output = flatten_train_val_metrics(
                train_metrics=self.current_train_metrics,
                val_metrics=val_metrics,
                prepend_stage=not self.is_first_epoch,
            )

            self.fabric.call(
                "on_train_batch_end_per_fold",
                output=output,
                batch=batch,
                batch_idx=batch_idx,
            )

            # add output values to progress bar
            # self._format_iterable(iterable, self._current_train_return, "train")

            # only increase global step if optimizer stepped
            self.global_step_per_fold[self.current_fold] += int(should_optim_step)

        self.fabric.call("on_train_epoch_end_per_fold", output=output)

    def training_step(self, model: ModelT, batch: Any, batch_idx: int) -> torch.Tensor:
        outputs: MetricType = convert_output_to_dict(
            model.training_step(batch, batch_idx=batch_idx)
        )

        loss = outputs["loss"]

        self.fabric.call("on_before_backward", loss=loss)
        self.fabric.backward(loss)
        self.fabric.call("on_after_backward")

        # avoid gradients in stored/accumulated values -> prevents potential OOM
        self.current_train_metrics = cast(
            MetricType,
            apply_to_collection(outputs, dtype=torch.Tensor, function=detach),
        )

        return loss

    def step_scheduler(
        self,
        model: ModelT,
        scheduler_cfg: Optional[SchedulerConfigT],
        level: Literal["step", "epoch"],
        current_value: int,
    ) -> None:
        # no scheduler
        if scheduler_cfg is None:
            return

        # wrong interval (step vs. epoch)
        if scheduler_cfg["interval"] != level:
            return

        # right interval, but wrong step wrt frequency
        if current_value % cast(int, scheduler_cfg["frequency"]) != 0:
            return

        if self.current_epoch == 0:
            return

        # ONLY allow schedulers to step per epoch and keep all fold schedulers synced
        # TODO: idk if this is wanted
        possible_monitor_vals = flatten_train_val_metrics(
            train_metrics=self.current_train_metrics,
            val_metrics=self.current_val_metrics,
            prepend_stage=True,
        )

        try:
            monitor = possible_monitor_vals[cast(str, scheduler_cfg["monitor"])]
        except KeyError as ex:
            possible_keys = list(possible_monitor_vals.keys())
            raise KeyError(
                f"monitor {scheduler_cfg['monitor']} is invalid. Possible values "
                f"are {possible_keys}."
            ) from ex

        # rely on model hook for actual step
        scheduler = cast(LRScheduler, scheduler_cfg["scheduler"])
        model.lr_scheduler_step(scheduler, monitor)  # type: ignore

    @staticmethod
    def _limit_batches(n_batches: int, limit_batches: Number = 1.0) -> int:
        if not isinstance(limit_batches, int):
            # is a float
            batch_limit = round(n_batches * limit_batches)
        else:
            batch_limit = limit_batches

        return batch_limit

    @property
    def should_validate(self) -> bool:
        """Whether to currently run validation."""
        return self.current_epoch % self.config.validation_frequency == 0

    def val_loop(
        self,
        model: ModelT,
        val_loader: DataLoader,
        optimizer: Optimizer,
    ):
        """The validation loop running a single validation epoch.

        Args:
            model: the LightningModule to evaluate
            val_loader: The dataloader yielding the validation batches.
        """
        self.fabric.call(
            "on_validation_start_per_fold", model=model
        )  # calls `model.eval()`
        torch.set_grad_enabled(False)

        n_batches = len(val_loader)
        batch_limit = self._limit_batches(n_batches, self.config.limit_val_batches)

        for batch_idx, batch in enumerate(val_loader):
            # end epoch if stopping training completely or
            # max batches for this epoch reached
            if self.should_stop or batch_idx >= batch_limit:
                break

            self.fabric.call(
                "on_validation_batch_start_per_fold",
                batch=batch,
                batch_idx=batch_idx,
                total=batch_limit,
                current_fold=self.current_fold,
            )

            out: MetricType = convert_output_to_dict(
                model.validation_step(batch, batch_idx)
            )

            # avoid gradients in stored/accumulated values -> prevents potential OOM
            out = cast(
                MetricType,
                apply_to_collection(out, dtype=torch.Tensor, function=detach),
            )

            metrics = {
                "epoch": self.current_epoch,
                "fold": self.current_fold,
                "stage": "val",
                **out,
            }

            self.fabric.call(
                "on_before_log_metrics", metrics=metrics, optimizer=optimizer
            )
            self.fabric.log_dict(
                metrics=metrics,
                step=self.global_step_per_fold[self.current_fold] - 1,
            )

            self.fabric.call(
                "on_validation_batch_end_per_fold",
                output=out,
                batch=batch,
                batch_idx=batch_idx,
            )

            # takes all metrics reported in out and adds in-place
            # to the tmp storage at current_val_metrics_per_fold
            # later after this batch-wise loop, the values in div
            # by the number of batches that were passed
            validation_update(
                current_metrics=self._current_val_metrics_per_fold,
                output=out,
                fold=self.current_fold,
            )
            # self._current_val_metrics_per_fold["loss"][
            #     self.current_fold
            # ] += output_getter(out)

        # avg fold level loss by number of batches/steps taken
        apply_to_collection(
            self._current_val_metrics_per_fold,
            dtype=torch.Tensor,
            function=partial(fold_idiv, value=batch_limit, fold=self.current_fold),
        )
        # self._current_val_metrics_per_fold["loss"][
        #   self.current_fold] /= batch_limit_idx
        self.fabric.call("on_validation_epoch_end_per_fold")
        torch.set_grad_enabled(True)
        self.fabric.call("on_validation_end_per_fold", trainer=self)

    def load(self, path: Path) -> FoldState:
        """Loads a checkpoint from a given file into state.

        Args:
            state: a mapping contaning model, optimizer and lr scheduler
            path: the path to load the checkpoint from
        """
        state = dict()
        remainder = self.fabric.load(path, state)
        self.current_fold = remainder.pop("current_fold")
        self.current_epoch = remainder.pop("current_epoch")
        self.global_step_per_fold[self.current_fold] = remainder.pop("global_step")

        if remainder:
            raise RuntimeError(f"Unused Checkpoint Values: {remainder}")

        return FoldState.parse_obj(state)

    def load_all_folds(self, paths: list[Path]) -> dict[int, FoldState]:
        fold_manager: dict[int, FoldState] = dict()
        for path in paths:
            # this should set self.current_fold
            # TODO: will this introduce a race condition? idk
            state = self.load(path)
            fold_manager[self.current_fold] = state
        return fold_manager

    def save(
        self,
        states: FoldState | Mapping[int, FoldState],
        name: Optional[str] = None,
    ):
        """Saves a checkpoint to the ``checkpoint_dir``

        Args:
            state: A mapping containing model, optimizer and lr scheduler.
        """

        def _save(_state: FoldState, fold: int = 0, name: Optional[str] = None):
            state = _state.dict()
            state.update(
                global_step=self.global_step_per_fold[self.current_fold],
                current_epoch=self.current_epoch,
                current_fold=fold,
            )

            if name is None:
                output = self.checkpoint_dir.joinpath(
                    f"epoch={self.current_epoch}_fold={fold}"
                    f"_loss={self.current_val_metrics['loss']:.4f}.ckpt"
                )
            else:
                assert "{fold}" in name
                output = self.checkpoint_dir.joinpath(name.format(fold=fold))

            self.fabric.save(output, state)

        if isinstance(states, Mapping):
            for fold, state in states.items():
                _save(state, fold, name)
        else:
            _save(states, name=name)

    @staticmethod
    def get_latest_checkpoint(ckpt_dir: Path) -> Optional[Path]:
        """Returns the latest checkpoint from the ``checkpoint_dir``

        Args:
            checkpoint_dir: the directory to search for checkpoints
        """
        if not ckpt_dir.is_dir():
            return None

        items = sorted(os.listdir(ckpt_dir))

        if not items:
            return None

        return ckpt_dir.joinpath(items[-1])

    def _parse_optimizers_schedulers(
        self,
        configure_optim_output: ConfigOptimizerOutput,
    ) -> OptimizerConfigT:
        _lr_sched_defaults = {
            "interval": "epoch",
            "frequency": 1,
            "monitor": self.config.monitor,
        }

        # single optimizer
        if isinstance(configure_optim_output, Optimizer):
            return configure_optim_output, None

        # single lr scheduler config with optimizer passed
        if isinstance(configure_optim_output, Mapping):
            optimizer = configure_optim_output["optimizer"]
            lr_scheduler = configure_optim_output.get("lr_scheduler", None)
            if lr_scheduler is not None:
                lr_scheduler = _lr_sched_defaults | lr_scheduler
            return optimizer, lr_scheduler

        raise ValueError(
            "Output of model.configure_optimizers() must be a torch.optim.Optimizer "
            "or a Mapping with keys 'optimizer' and optionally 'lr_scheduler'"
        )

    @property
    def is_global_zero(self) -> bool:
        return self.fabric.is_global_zero
