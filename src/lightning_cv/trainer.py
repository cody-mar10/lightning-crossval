import enum
import logging
import math
from functools import cached_property, partial
from pathlib import Path
from typing import Any, Literal, Mapping, Optional, Union, cast

import torch
from lightning import LightningDataModule, LightningModule
from lightning.fabric.loggers.csv_logs import CSVLogger
from lightning.fabric.utilities.types import LRScheduler
from lightning.pytorch.utilities.types import OptimizerLRScheduler
from lightning_utilities.core.apply_func import apply_to_collection
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LRScheduler
from torch.utils.data import DataLoader

from lightning_cv.callbacks.base import Callback, CallbackT
from lightning_cv.callbacks.checkpoint import ModelCheckpoint
from lightning_cv.callbacks.logger import LogFlusher
from lightning_cv.callbacks.mode import EvalMode, TrainMode
from lightning_cv.callbacks.progress import TqdmProgressBar
from lightning_cv.callbacks.summary import ModelSummary
from lightning_cv.callbacks.utils import standardize_callbacks
from lightning_cv.config import CrossValidationTrainerConfig, FoldState
from lightning_cv.data import (
    CV_DATAMODULE_METHOD_NAME,
    BaseCVDataModule,
    is_cross_val_datamodule,
)
from lightning_cv.fabric import Fabric
from lightning_cv.typehints import (
    KwargType,
    MetricType,
    Number,
    OptimizerConfigT,
    SchedulerConfigT,
)
from lightning_cv.utils.checkpointing import CheckpointingMixin
from lightning_cv.utils.foldutils import (
    convert_output_to_dict,
    flatten_train_val_metrics,
    fold_idiv,
    validation_update,
)
from lightning_cv.utils.stopping import StopReasons
from lightning_cv.utils.strategies import is_distributed

logger = logging.getLogger(__name__)


class TrainerState(enum.Enum):
    TRAINING = enum.auto()
    VALIDATING = enum.auto()
    NOT_STOPPED = enum.auto()


class CrossValidationTrainer(CheckpointingMixin):
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
    __callbacks__ = Callback.available_callbacks()

    def __init__(
        self, model_type: type[LightningModule], config: CrossValidationTrainerConfig
    ):
        self.config = config
        self.model_type = model_type

        self.global_step_per_fold: dict[int, int] = dict()
        """Keeps track of the number of times the model in each fold has been stepped. In other
        words, the number of times the model has been updated by an optimizer."""

        self.current_epoch: int = 0
        self.current_fold: int = 0
        self.setup_complete = False
        self.is_first_epoch = True

        self.should_stop = False
        self.current_train_metrics: MetricType = dict()
        self._current_val_metrics_per_fold: MetricType = dict()

        if self.config.loggers is None:
            self.config.loggers = CSVLogger(
                root_dir=self.config.checkpoint_dir,
            )

        self._add_default_callbacks(self.config.callbacks)

        fabric_kwargs = self.config.to_dict(include=self.__fabric_keys__)
        self.fabric = Fabric(**fabric_kwargs)

        if self.config.gradient_clip_algorithm == "norm":
            self.gradient_clip_kwargs = {"max_norm": self.config.gradient_clip_val}
        else:
            self.gradient_clip_kwargs = {"clip_val": self.config.gradient_clip_val}

        self.apply_gradient_clipping = self.config.gradient_clip_val > 0.0

        self._current_state = TrainerState.TRAINING
        self._state_when_stopped = TrainerState.NOT_STOPPED
        self._status = StopReasons.NULL

    @property
    def num_folds(self) -> int:
        return len(getattr(self, "fold_manager", {}))

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

    # TODO: need to distinguish this from the root dir
    # lightning uses: rootdir for lightning_root, and then log_dir for the current version
    # especially since the config uses checkpoint dir
    # and then this is different that
    @cached_property
    def checkpoint_dir(self) -> Path:
        """Get the checkpoint directory for the CURRENT version of training.

        Ex: lighting_root/lightning_logs/version_0
        """
        try:
            logger_expt_name = self.fabric.logger.name or "expt_0"
            logger_version_no = self.fabric.logger.version or "version_0"
            if isinstance(logger_version_no, int):
                logger_version_no = f"version_{logger_version_no}"
            if self.fabric.logger.root_dir is not None:
                checkpoint_dir = Path(self.fabric.logger.root_dir)
            else:
                checkpoint_dir = Path("checkpoints")
        except (AttributeError, IndexError):
            # no logger attached
            logger_expt_name = "expt_0"
            logger_version_no = "version_0"
            checkpoint_dir = self.config.checkpoint_dir

        return checkpoint_dir / logger_expt_name / logger_version_no

    @property
    def is_distributed(self) -> bool:
        return is_distributed(self.fabric.strategy)

    @property
    def status(self) -> StopReasons:
        return self._status

    @status.setter
    def status(self, value: Any):
        if isinstance(value, StopReasons):
            self._status = value
            if self._status != StopReasons.NULL:
                self.state_when_stopped = self.current_state
        else:
            raise TypeError(f"{value=} is not a valid StopReasons enum")

    @property
    def current_state(self) -> TrainerState:
        return self._current_state

    @current_state.setter
    def current_state(self, value: Any):
        if isinstance(value, TrainerState):
            if value == TrainerState.NOT_STOPPED:
                raise ValueError("Current state cannot be NOT_STOPPED")
            self._current_state = value
        else:
            raise TypeError(f"{value=} is not a valid TrainerState enum")

    @property
    def state_when_stopped(self) -> TrainerState:
        return self._state_when_stopped

    @state_when_stopped.setter
    def state_when_stopped(self, value: Any):
        if self.status == StopReasons.NULL or not self.should_stop:
            raise ValueError(
                "Cannot set state_when_stopped when trainer has not been signaled to stop."
            )

        if isinstance(value, TrainerState):
            self._state_when_stopped = value
        else:
            raise TypeError(f"{value=} is not a valid TrainerState enum")

    @property
    def fold_manager(self) -> dict[int, FoldState]:
        if not hasattr(self, "_fold_manager"):
            raise AttributeError(
                "`fold_manager` is not available. The `.setup()` must be called first with a "
                "valid cross validation datamodule and a model config to initialize 1 model per "
                "fold."
            )

        return self._fold_manager

    def _add_default_callbacks(self, callbacks: CallbackT = None):
        callbacks = standardize_callbacks(callbacks)
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

    @property
    def callbacks(self) -> list[Callback]:
        return self.config.callbacks  # type: ignore[return-type] <- should be standardized

    def setup(self, datamodule: BaseCVDataModule, model_config):
        # only setup once
        if not self.setup_complete:
            self.fabric.launch()
            datamodule.setup("fit")

            # setup n_folds models and optimizers
            self._fold_manager: dict[int, FoldState] = dict()
            self.global_step_per_fold.clear()
            self.estimated_steps: dict[int, int] = dict()

            for fold, (train_loader, val_loader) in enumerate(
                datamodule.train_val_dataloaders()
            ):
                self.estimated_steps[fold] = self._estimated_steps(len(train_loader))

                model = self.model_type(model_config)
                setattr(model, "estimated_steps", self.estimated_steps[fold])
                config = self._parse_optimizers_schedulers(model.configure_optimizers())
                optimizer: Optimizer
                optimizer, scheduler_cfg = config

                # ignore fact that fabric returns a wrapper
                # ignore setup takes a torch.nn.Module or a lightning.LightningModule
                # BUG: for some reason fabric will add lightningmodules to the callbacks
                # this was not intended and was added with commit bf2af0e of lightning
                # however, this package was not developed with this in mind
                model, optimizer = self.fabric.setup(model, optimizer)  # type: ignore[return-value]

                self.fold_manager[fold] = FoldState(
                    model=model, optimizer=optimizer, scheduler=scheduler_cfg
                )
                # init global steps for each fold to 0
                self.global_step_per_fold[fold] = 0

            self._current_val_metrics_per_fold["loss"] = torch.zeros(
                size=(len(self.fold_manager),), device=self.fabric.device
            )
            self._finished_cv_loop_per_fold = torch.zeros(
                size=(len(self.fold_manager),), device=self.fabric.device
            ).bool()

            # need to remove model from callbacks since this gets added in later version of lightning
            # this breaks everything...
            callbacks = [
                callback
                for callback in self.fabric._callbacks
                if not isinstance(callback, self.model_type)
            ]

            self.fabric._callbacks = callbacks

            self.setup_complete = True

    def _estimated_steps(self, training_batches: int):
        max_estimated_steps = (
            math.ceil(training_batches / self.config.grad_accum_steps)
            * self.config.max_epochs
        )

        return max_estimated_steps

    def train_with_cross_validation(self, datamodule: BaseCVDataModule, model_config):
        if not is_cross_val_datamodule(datamodule):
            is_lightning_module = isinstance(datamodule, LightningDataModule)
            has_method = hasattr(datamodule, CV_DATAMODULE_METHOD_NAME)

            raise ValueError(
                "The provided datamodule MUST satisfy 2 constraints:\n"
                "\t1. It must be a subclass of lightning.LightningDataModule\n"
                "\t2. It must have a method train_val_dataloaders() that yields pairs of "
                "dataloaders for training and validation."
                f"Provided module is a LightningDataModule: {is_lightning_module}\n"
                f"Provided module has method {CV_DATAMODULE_METHOD_NAME}: {has_method}"
            )

        # set up n_folds models and optimizers, and creates fold_manager attribute
        # also calls self.fabric.launch()
        self.setup(datamodule=datamodule, model_config=model_config)
        self.apply_callback("on_train_start")
        while not self.should_stop:
            # calls train_loop, val_loop, step_scheduler per fold
            self.cross_val_loop(datamodule=datamodule)

            self.current_epoch += 1

            self.is_first_epoch = False

            # stopping condition on epoch level
            # don't change self.should_stop here
            # reserve self.should_stop for external signals for stopping
            if self.current_epoch >= self.config.max_epochs:
                break

        self.apply_callback("on_train_end")

        # reset for next fit call
        self.should_stop = False

        self.log_status()

    def cross_val_loop(self, datamodule: BaseCVDataModule):
        # called once per epoch
        # train and validate each fold
        self.apply_callback("on_train_fold_start")

        # reset here - advt is once training is complete, this will keep history of
        # per fold completeness
        self._finished_cv_loop_per_fold.fill_(False)

        foldloaders = datamodule.train_val_dataloaders(shuffle=True)
        for fold, (train_loader, val_loader) in enumerate(foldloaders):
            if self.should_stop:
                break

            train_loader, val_loader = self.fabric.setup_dataloaders(
                train_loader, val_loader
            )

            self.current_fold = fold
            state = self.fold_manager[fold]
            # ignore fact that lightning.fabric wraps model
            model = cast(LightningModule, state.model)
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

            self._finished_cv_loop_per_fold[fold] = True
        # update this value every epoch
        # this stores the average validation loss per fold

        if self.is_distributed:  # pragma: no cover <- TODO: need to test distributed
            # need to get all metrics in distributed setting
            val_metrics = cast(
                MetricType, self.fabric.all_gather(self._current_val_metrics_per_fold)
            )
        else:
            val_metrics = self._current_val_metrics_per_fold

        # BUG: for some reason this doesn't work
        # in CHTC trials stopped by timeouts are averaging as if
        # all were completed...
        # however, this isn't the case with a local test
        # so likely installation issue...
        # most likely timer issue, but early stopping log messages may help diagnose
        if self._finished_cv_loop_per_fold.all():
            # only update if all folds finished training and validating
            # this will technically raise on error on the first epoch since
            # self.current_val_metrics is not yet available -> but that's already
            # indicates some problem anyway. Otherwise, the current_val_metrics
            # will just be the previous val metrics. This will only happen
            # when an external callback stops training early.
            #
            # The only callback that has the ability to stop training other than
            # at the end of this fn with the "on_train_fold_end" callback is the
            # Timer callback, and not entering this block basically indicates
            # training is about to stop anyway.

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

        self.apply_callback("on_train_fold_end")

    def train_loop(
        self,
        model: LightningModule,
        optimizer: Optimizer,
        train_loader: DataLoader,
        scheduler_cfg: Optional[SchedulerConfigT] = None,
    ):
        self.current_state = TrainerState.TRAINING
        n_batches = len(train_loader)
        batch_limit = self._limit_batches(n_batches, self.config.limit_train_batches)

        self.apply_callback(
            "on_train_epoch_start_per_fold",
            model=model,
            total=batch_limit,
            current_epoch=self.current_epoch,
            current_fold=self.current_fold,
        )  # calls model.train()

        # default starting value in case loop never runs
        output = flatten_train_val_metrics(
            train_metrics=self.current_train_metrics,
            val_metrics=self.current_val_metrics if not self.is_first_epoch else None,
            prepend_stage=not self.is_first_epoch,
        )
        for batch_idx, batch in enumerate(train_loader):
            # end epoch if stopping training completely or
            # max batches for this epoch reached
            if self.should_stop or batch_idx >= batch_limit:
                break

            self.apply_callback(
                "on_train_batch_start_per_fold", batch=batch, batch_idx=batch_idx
            )

            # check if optimizer should step in gradient accumulation
            should_optim_step = self.should_do_something(
                batch_idx, self.config.grad_accum_steps
            )

            if should_optim_step:
                # currently only supports a single optimizer
                self.apply_callback(
                    "on_before_optimizer_step", optimizer=optimizer, optimizer_idx=0
                )

                # although some optimizers need a closure
                # these are not compatibile with automatic precision scaling
                # .training_step sets `self.current_train_metrics`
                self.training_step(model=model, batch=batch, batch_idx=batch_idx)
                if self.apply_gradient_clipping:  # pragma: no cover <- this is tested
                    self.fabric.clip_gradients(
                        module=model,
                        optimizer=optimizer,
                        error_if_nonfinite=False,
                        **self.gradient_clip_kwargs,
                    )

                # optimizer.step() # TODO: model.optimizer_step() instead?
                model.optimizer_step(
                    epoch=self.current_epoch, batch_idx=batch_idx, optimizer=optimizer
                )

                self.apply_callback("on_before_zero_grad", optimizer=optimizer)

                # optimizer.zero_grad()
                model.optimizer_zero_grad(
                    epoch=self.current_epoch, batch_idx=batch_idx, optimizer=optimizer
                )

                metrics = {
                    "epoch": self.current_epoch,
                    "fold": self.current_fold,
                    "stage": "train",
                    **self.current_train_metrics,
                }

                self.apply_callback(
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

            self.apply_callback(
                "on_train_batch_end_per_fold",
                output=output,
                batch=batch,
                batch_idx=batch_idx,
            )

            # only increase global step if optimizer stepped
            self.global_step_per_fold[self.current_fold] += int(should_optim_step)

        self.apply_callback("on_train_epoch_end_per_fold", output=output)

    def training_step(
        self, model: LightningModule, batch: Any, batch_idx: int
    ) -> torch.Tensor:
        _outputs = model.training_step(batch, batch_idx=batch_idx)
        if _outputs is None:
            raise RuntimeError(
                "It is currently expected that the model's training_step returns the loss and "
                "optionally other metrics. Manual optimization is not currently supported."
            )

        outputs: MetricType = convert_output_to_dict(_outputs)

        loss = outputs["loss"]

        self.apply_callback("on_before_backward", loss=loss)
        self.fabric.backward(loss)
        self.apply_callback("on_after_backward")

        # avoid gradients in stored/accumulated values -> prevents potential OOM
        self.current_train_metrics = cast(
            MetricType,
            apply_to_collection(outputs, dtype=torch.Tensor, function=torch.detach_),
        )

        return loss

    def step_scheduler(
        self,
        model: LightningModule,
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

        # if self.current_epoch == 0:
        #     return

        # ONLY allow schedulers to step per epoch and keep all fold schedulers synced
        # TODO: but this code is called from train_loop, which gives the option to step per batch
        # not sure if it's best to allow LR to change each fold?
        # I guess it's fine since each fold would always have its LR
        # adjusted the same across experiments

        if self.current_epoch == 0:
            current_val_metrics = None
        else:
            current_val_metrics = self.current_val_metrics

        possible_monitor_vals = flatten_train_val_metrics(
            train_metrics=self.current_train_metrics,
            val_metrics=current_val_metrics,
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
        model.lr_scheduler_step(scheduler, monitor)

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
        return self.should_do_something(
            self.current_epoch, self.config.validation_frequency
        )

    def should_do_something(self, current_step_0_indexed: int, frequency: int) -> bool:
        # if we want to do something with a frequency of 2 steps, then we need to check
        # the +1 of the current 0-indexed step:
        # step: 0 1 2 3 4 ...
        # do  : F T F T F ...
        # this prevents the first step (0) from always being True
        return (current_step_0_indexed + 1) % frequency == 0

    @torch.no_grad()
    def val_loop(
        self,
        model: LightningModule,
        val_loader: DataLoader,
        optimizer: Optimizer,
    ):
        """The validation loop running a single validation epoch.

        Args:
            model: the LightningModule to evaluate
            val_loader: The dataloader yielding the validation batches.
        """
        self.current_state = TrainerState.VALIDATING

        n_batches = len(val_loader)
        batch_limit = self._limit_batches(n_batches, self.config.limit_val_batches)

        self.apply_callback(
            "on_validation_start_per_fold", model=model, total=batch_limit
        )  # calls `model.eval()`

        batch_idx = 0  # for mypy unbound local error
        for batch_idx, batch in enumerate(val_loader):
            # end epoch if stopping training completely or
            # max batches for this epoch reached
            if self.should_stop or batch_idx >= batch_limit:
                break

            self.apply_callback(
                "on_validation_batch_start_per_fold",
                batch=batch,
                batch_idx=batch_idx,
                current_fold=self.current_fold,
            )

            _out = model.validation_step(batch, batch_idx=batch_idx)
            if _out is None:
                raise RuntimeError(
                    "The model's validation_step is expected to return the loss and "
                    "optionally other metrics."
                )

            out: MetricType = convert_output_to_dict(_out)

            metrics = {
                "epoch": self.current_epoch,
                "fold": self.current_fold,
                "stage": "val",
                **out,
            }

            self.apply_callback(
                "on_before_log_metrics", metrics=metrics, optimizer=optimizer
            )
            self.fabric.log_dict(
                metrics=metrics,
                step=self.global_step_per_fold[self.current_fold] - 1,
            )

            self.apply_callback(
                "on_validation_batch_end_per_fold",
                output=out,
                batch=batch,
                batch_idx=batch_idx,
            )

            # takes all metrics reported in out and add in-place
            # to the tmp storage at current_val_metrics_per_fold
            # then later divide by number of steps taken to avg
            validation_update(
                current_metrics=self._current_val_metrics_per_fold,
                output=out,
                fold=self.current_fold,
            )

        # avg fold level loss by number of batches/steps taken
        apply_to_collection(
            data=self._current_val_metrics_per_fold,
            dtype=torch.Tensor,
            function=partial(
                fold_idiv,
                # in most cases the denom will just be the number of batches
                # however, if using limit_val_batches, it will be the number of batches
                # actually stepped, but occasionally if an external callback like
                # a timer stops training, then the number of batches stepped will be
                # less than the batch_limit. The batch_idx will then be the number
                # of batches stepped. We need to +1 to avoid div by 0
                # if in this scenario and no batches were stepped.
                value=min(batch_limit, batch_idx + 1),
                fold=self.current_fold,
            ),
        )

        self.apply_callback("on_validation_end_per_fold")

    def _parse_optimizers_schedulers(
        self,
        configure_optim_output: Union[OptimizerLRScheduler, Mapping],
    ) -> OptimizerConfigT:
        # NOTE: none of this supports multiples optimizers or schedulers

        # single optimizer
        if isinstance(configure_optim_output, Optimizer):
            return configure_optim_output, None

        _lr_sched_defaults = {
            "interval": "epoch",
            "frequency": 1,
            "monitor": self.config.monitor,
        }

        # single lr scheduler config with optimizer passed
        if isinstance(configure_optim_output, Mapping):
            optimizer = configure_optim_output["optimizer"]
            lr_scheduler: KwargType = configure_optim_output.get("lr_scheduler", None)  # type: ignore
            if lr_scheduler is not None:
                lr_scheduler = _lr_sched_defaults | lr_scheduler
            return optimizer, lr_scheduler

        # tuple of optimizer and lr scheduler
        # this would technically handle multiple optimizers and schedulers
        if isinstance(configure_optim_output, tuple):
            optimizer, lr_scheduler = configure_optim_output  # type: ignore

            if isinstance(lr_scheduler, Mapping):
                lr_scheduler = _lr_sched_defaults | lr_scheduler
            else:
                lr_scheduler = {"scheduler": lr_scheduler, **_lr_sched_defaults}
            return optimizer, lr_scheduler  # type: ignore[return-value]

        raise ValueError(
            "Output of model.configure_optimizers() must be a torch.optim.Optimizer "
            "or a Mapping with keys 'optimizer' and optionally 'lr_scheduler'"
        )

    @property
    def is_global_zero(self) -> bool:
        return self.fabric.is_global_zero

    def apply_callback(self, callback_name: str, **kwargs):
        if callback_name not in self.__callbacks__:
            raise RuntimeError(
                f"Callback {callback_name} not found. "
                "Allowed callbacks are: {self.__callbacks__}"
            )
        # all callbacks have a trainer kwarg
        kwargs["trainer"] = self
        self.fabric.call(callback_name, **kwargs)

    def log_status(self):
        if self.is_global_zero:
            msg_template = self.status.message()
            msg_kwargs = {
                "max_epochs": self.config.max_epochs,
                "monitor": self.config.monitor,
            }
            # str.format only takes what is needed
            msg = msg_template.format(**msg_kwargs)
            logger.info(f"{msg}\n")
