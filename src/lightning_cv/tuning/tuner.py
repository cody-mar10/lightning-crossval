from __future__ import annotations

from copy import copy
from pathlib import Path
from shutil import rmtree
from typing import Any, Callable, Generic, Optional, cast

import optuna
from lightning.fabric.loggers.csv_logs import CSVLogger
from lightning.fabric.loggers.logger import Logger

import lightning_cv as lcv
from lightning_cv._typing import ModelConfig, ModelT
from lightning_cv.callbacks import Callback
from lightning_cv.trainer import DataModuleType

from .callbacks import TrialPruning

TrialSuggestionFn = Callable[[optuna.Trial], dict[str, Any]]


class Tuner(Generic[ModelT, ModelConfig]):
    def __init__(
        self,
        model_type: type[ModelT],
        model_config: ModelConfig,
        datamodule: DataModuleType,
        trainer_config: "lcv.CrossValidationTrainerConfig",
        logdir: str | Path = Path("checkpoints"),
        experiment_name: str = "exp0",
        clean_logdir_if_exists: bool = False,
        verbose: bool = False,
    ):
        self.model_type = model_type
        # only store dict and type so that it can be updated with new trialed values
        self.model_config = model_config.dict()
        self.model_config_type = type(model_config)

        self.datamodule = datamodule

        self.trial_number = 0
        self.verbose = verbose
        self.experiment_name = experiment_name
        self._init_logdir(logdir, clean_logdir_if_exists)
        self.trainer_config = trainer_config

        self._callbacks = self._store_callbacks(self.trainer_config.callbacks)
        self._logger_types = self._store_logger_types(self.trainer_config.loggers)

    def _init_logdir(self, logdir: str | Path, clean_logdir_if_exists: bool = False):
        if isinstance(logdir, str):
            logdir = Path(logdir)

        self.logdir = logdir

        if self.logdir.exists():
            if not clean_logdir_if_exists:
                raise RuntimeError(
                    "The log directory {self.logdir} exists. Use "
                    "`clean_logdir_if_exists` or change the log directory."
                )
            rmtree(self.logdir)

        self.logdir.mkdir()

    def _store_callbacks(
        self, callbacks: Optional[list[Callback] | Callback]
    ) -> list[Callback]:
        if callbacks is None:
            callbacks = list()
        elif not isinstance(callbacks, list):
            callbacks = [callbacks]
        else:
            callbacks = copy(callbacks)

        return callbacks

    def _store_logger_types(
        self, loggers: Optional[list[Logger] | Logger]
    ) -> list[type[Logger]]:
        # capture logger types only and reconstruct during tuning
        if loggers is not None:
            if not isinstance(loggers, list):
                logger_types = [type(loggers)]
            else:
                logger_types = [type(logger) for logger in loggers]
        else:
            logger_types = cast(list[type[Logger]], [CSVLogger])

        return logger_types

    def tune(
        self, trial: optuna.Trial, func: TrialSuggestionFn, monitor: str = "loss"
    ) -> float:
        loggers = [
            logger_type(  # type: ignore
                root_dir=self.logdir,
                name=self.experiment_name,
                version=self.trial_number,
            )
            for logger_type in self._logger_types
        ]

        self.trainer_config.loggers = loggers

        callbacks = copy(self._callbacks)
        callbacks.append(
            TrialPruning(trial=trial, monitor=monitor, verbose=self.verbose)
        )

        self.trainer_config.callbacks = callbacks

        trainer = lcv.CrossValidationTrainer(
            model_type=self.model_type, config=self.trainer_config
        )

        # suggest values here
        trialed_values = func(trial)

        # then update model config
        model_config = self.model_config_type.parse_obj(
            self.model_config | trialed_values
        )

        try:
            trainer.train_with_cross_validation(
                datamodule=self.datamodule, model_config=model_config
            )
        except optuna.TrialPruned as err:
            # TODO: add optional pruning hook
            raise err

        self.trial_number += 1

        return trainer.current_val_metrics[monitor].item()
