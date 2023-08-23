from __future__ import annotations

from copy import copy
from dataclasses import asdict, is_dataclass
from functools import partial
from pathlib import Path
from typing import Any, Callable, Mapping, Optional, cast

import optuna
from boltons.iterutils import get_path, remap
from lightning.fabric.loggers.csv_logs import CSVLogger
from lightning.fabric.loggers.logger import Logger
from optuna.trial import TrialState
from pydantic import BaseModel, Field

import lightning_cv as lcv
from lightning_cv.callbacks import Callback
from lightning_cv.module import BaseModelConfig
from lightning_cv.typehints import (
    CrossValDataModuleT,
    DataclassInstance,
    KwargType,
    ModelConfig,
    ModelT,
)

from .callbacks import TrialPruning
from .hyperparameters import HparamRegistry, IntFloatStrDict, suggest

TrialSuggestionFn = Callable[[optuna.Trial], IntFloatStrDict]
ModelConfigSerializeFn = Callable[[ModelConfig], KwargType]
DataConfig = BaseModel | DataclassInstance | KwargType
DataConfigSerializeFn = Callable[[Any], KwargType]


class _TrialUpdater(BaseModel):
    model: KwargType = Field(default_factory=dict)
    data: KwargType = Field(default_factory=dict)
    _updated: bool = False

    @staticmethod
    def _update(
        parents: tuple, key: Any, value: Any, update_values: KwargType
    ) -> bool | tuple:
        """Tries to update leaf nodes based on an arbitrarily nested update schema.

        Almost fits the `boltons.iterutils` `visit` API. It is recommended to create a
        `functools.partial` object with this function.

        Args:
            parents (tuple): parent nodes to the current node
            key (Any): current node
            value (Any): value at current node
            update_values (KwargType): arbitrarily nested update mapping

        Returns:
            bool | tuple: Returns True if at an intermediate node. Otherwise, returns the
                new (node, value) map. See `boltons.iterutils.remap` for more info.
        """

        path = (*parents, key)
        # try to get a value at the current path in update_values
        # but default to the current value
        new_value = (
            # handles nesting
            get_path(root=update_values, path=path, default=None)
            # handles flat
            or update_values.get(key, None)
            # default
            or value
        )

        # skip updating intermediate nodes
        if isinstance(new_value, Mapping):
            return True
        # otherwise update leaf with new value
        return key, new_value

    def update(self, values: KwargType):
        """Update all fields with new values.

        The `values` argument can have a variety of shapes:

        Example:
        ```python
        >>> model_cfg = {
        ...     "in_dim": 64,
        ...     "out_dim": 32,
        ...     "optimizer": {"lr": 1e-3}
        ... }
        >>> data_cfg = {"batch_size": 64}

        # now we want to try a new learning rate and a new batch size
        # the following two `values` schemas work:

        # 1. flat values
        >>> config = _TrialUpdater(model=model_cfg, data=data_cfg)
        >>> new_values = {"lr": 2.53e-4, "batch_size": 16}
        >>> config.update(new_values)
        >>> print(config.model)
        {"in_dim": 64, "out_dim": 32, "optimizer": {"lr": 2.53e-4}}
        >>> print(config.data)
        {"batch_size": 16}

        # 2. nested values
        >>> config = _TrialUpdater(model=model_cfg, data=data_cfg)
        >>> new_values = {
        ...     "model": {"optimizer": {"lr": 2.53e-4}},
        ...     "data": {"batch_size": 16}
        ... }
        >>> config.update(new_values)
        >>> print(config.model)
        {"in_dim": 64, "out_dim": 32, "optimizer": {"lr": 2.53e-4}}
        >>> print(config.data)
        {"batch_size": 16}

        # Further, a combination of flat and nested values also works.
        ```

        Args:
            values (KwargType): an arbitrarily nested map from subfield keys to new values
        """
        for key in self.model_fields:
            current_values: KwargType = getattr(self, key)

            # handles nested and flat update value schemas
            update_values = values.get(key, values)
            update_fn = partial(_TrialUpdater._update, update_values=update_values)
            new_values = remap(root=current_values, visit=update_fn)

            current_values.update(new_values)
        self._updated = True


class Tuner:
    def __init__(
        self,
        model_type: type[ModelT],
        model_config: ModelConfig,
        datamodule_type: type[CrossValDataModuleT],
        datamodule_config: DataConfig,
        trainer_config: "lcv.CrossValidationTrainerConfig",
        logdir: str | Path = Path("checkpoints"),
        experiment_name: str = "exp0",
        clean_logdir_if_exists: bool = False,
        verbose: bool = False,
        hparam_config_file: Optional[str | Path] = None,
        model_config_serializer: Optional[ModelConfigSerializeFn] = None,
        data_config_serializer: Optional[DataConfigSerializeFn] = None,
    ):
        self._hparam_registry = self._init_hparam_registry(hparam_config_file)

        self.model_type = model_type
        # only store dict and type so that it can be updated with new trialed values
        serialized_model_config = self._init_model_config(
            model_config, model_config_serializer
        )
        self.model_config_type = type(model_config)

        self.datamodule_type = datamodule_type
        serialized_data_config, self.data_config_type = self._init_data_config(
            datamodule_config, data_config_serializer
        )

        self.trial_updater = _TrialUpdater(
            model=serialized_model_config, data=serialized_data_config
        )

        self.verbose = verbose
        self.experiment_name = experiment_name
        self.trainer_config = trainer_config
        self._init_logdir(logdir, clean_logdir_if_exists)

        self._callbacks = self._store_callbacks(self.trainer_config.callbacks)
        self._logger_types = self._store_logger_types(self.trainer_config.loggers)

        self._trial_number = 0

    def _init_model_config(
        self,
        model_config: ModelConfig | KwargType,
        model_config_serializer: Optional[ModelConfigSerializeFn] = None,
    ) -> KwargType:
        if isinstance(model_config, (BaseModelConfig, BaseModel)):
            # pydantic model
            return model_config.model_dump()
        elif is_dataclass(model_config):
            # dataclasses.dataclass
            return asdict(model_config)
        elif isinstance(model_config, Mapping):
            return model_config

        if model_config_serializer is None:
            raise ValueError(
                "Model config is neither a `pydantic.BaseModel` nor a "
                "`dataclasses.dataclass`. A serialization function to convert the "
                "input model config to a dictionary is required if the config is not "
                "already."
            )
        return model_config_serializer(model_config)

    def _init_data_config(
        self,
        data_config: DataConfig,
        data_config_serializer: Optional[DataConfigSerializeFn] = None,
    ) -> tuple[KwargType, type]:
        data_config_type = type(data_config)
        if isinstance(data_config, (BaseModelConfig, BaseModel)):
            # pydantic model
            return data_config.model_dump(), data_config_type
        elif is_dataclass(data_config):
            # dataclasses.dataclass
            return asdict(data_config), data_config_type
        elif isinstance(data_config, Mapping):
            return data_config, data_config_type

        if data_config_serializer is None:
            raise ValueError(
                "Data config is neither a `pydantic.BaseModel` nor a "
                "`dataclasses.dataclass`. A serialization function to convert the "
                "input data config to a dictionary is required if the config is not "
                "already."
            )
        return data_config_serializer(data_config), data_config_type

    def _init_hparam_registry(
        self, config_file: Optional[str | Path]
    ) -> Optional[HparamRegistry]:
        if config_file is None:
            return None

        return HparamRegistry.from_file(config_file)

    def _init_logdir(self, logdir: str | Path, clean_logdir_if_exists: bool = False):
        if isinstance(logdir, str):
            logdir = Path(logdir)

        self.logdir = logdir

        # the lightning loggers log to:
        # (default_root_dir/name/version_#)
        # so need to check if default_root_dir/name exists
        actual_path = self.logdir.joinpath(self.experiment_name)
        # if actual_path.exists():
        #     if not clean_logdir_if_exists:
        #         raise RuntimeError(
        #             f"The log directory {self.logdir} exists. Use "
        #             "`clean_logdir_if_exists` or change the log directory."
        #         )
        #     elif self.trainer_config.devices > 1:
        #         rmtree(actual_path)

        actual_path.mkdir(exist_ok=True, parents=True)

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

    def _create_datamodule(self) -> CrossValDataModuleT:
        # these should have already been updated, but check just in case
        self._check_hparams_updated()
        serialized_data_config = self.trial_updater.data
        data_config = self.data_config_type(**serialized_data_config)

        if isinstance(data_config, Mapping):
            return self.datamodule_type(**data_config)

        # otherwise datamodule must accept only a single config argument
        return self.datamodule_type(data_config)

    def _create_model_config(self) -> ModelConfig:
        # these should have already been updated, but check just in case
        self._check_hparams_updated()
        serialized_model_config = self.trial_updater.model
        model_config = self.model_config_type(**serialized_model_config)
        return model_config

    def _check_hparams_updated(self):
        if not self.trial_updater._updated:
            raise RuntimeError(
                "The tunable hyperparameters have not yet been updated with new trialed "
                "values. Try calling the `.trial_updater.update` method."
            )

    def _set_trial_number(self, study: optuna.Study):
        trials = study.get_trials(deepcopy=False)
        if trials:
            latest_trial = max(trials, key=lambda x: x.number)
            if latest_trial.state == TrialState.RUNNING:
                # latest_trial is actively running so use that number
                number = latest_trial.number
            else:
                # latest_trial completed so need to increment
                number = latest_trial.number + 1
        else:
            # default currently stored value which starts at 0
            # should only be used for the first trial
            # then the number should always be handled by the optuna.Study
            number = self._trial_number
        return number

    @property
    def trial_number(self) -> int:
        return self._trial_number

    def tune(
        self,
        trial: optuna.Trial,
        suggest_fn: Optional[TrialSuggestionFn] = None,
        trialed_values: Optional[IntFloatStrDict] = None,
        monitor: str = "loss",
    ) -> float:
        self._trial_number = self._set_trial_number(trial.study)
        if trialed_values is None:
            # if not trialed_values passed, then need to check for a suggestion fn
            if suggest_fn is None:
                if self._hparam_registry is None:
                    raise ValueError(
                        "Either (1) a hparam config toml file must be passed upon Tuner "
                        "init, (2) a suggestion function must be passed to this method, "
                        "or (3) a custom trialed_value dictionary must be passed."
                    )

                # use HparamRegistry functionality
                trialed_values = suggest(trial=trial, registry=self._hparam_registry)
            else:
                # use the provided suggestion fn
                trialed_values = suggest_fn(trial)
        # else: trialed_values must've been passed

        loggers = [
            logger_type(  # type: ignore
                root_dir=self.logdir,
                name=self.experiment_name,
                version=self._trial_number,
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

        # update model/data config with trialed values
        self.trial_updater.update(trialed_values)

        # then update model config
        model_config = self._create_model_config()
        datamodule = self._create_datamodule()

        try:
            trainer.train_with_cross_validation(
                datamodule=datamodule, model_config=model_config
            )
        except optuna.TrialPruned as err:
            # TODO: add optional pruning hook
            raise err
        finally:
            # always increment trial number even if trial gets pruned
            # NOTE: this shouldn't be needed since this will mostly defer
            # to the `optuna.Study` update its trial numbers
            self._trial_number += 1

        return trainer.current_val_metrics[monitor].item()
