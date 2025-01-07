from copy import deepcopy
from pathlib import Path
from typing import Callable, Mapping, Optional, Union, cast

import optuna
from lightning import LightningModule
from lightning.fabric.loggers.csv_logs import CSVLogger
from lightning.fabric.loggers.logger import Logger
from optuna.trial import TrialState

from lightning_cv.callbacks.utils import standardize_callbacks
from lightning_cv.config import CrossValidationTrainerConfig
from lightning_cv.data import BaseCVDataModule
from lightning_cv.trainer import CrossValidationTrainer
from lightning_cv.tuning.callbacks import TrialPruning
from lightning_cv.tuning.hyperparameters import HparamRegistry, IntFloatStrDict, suggest
from lightning_cv.tuning.utils import _TrialUpdater
from lightning_cv.typehints import FilePath
from lightning_cv.typehints.config import ConfigT
from lightning_cv.utils.serde import config_deserializer, config_serializer

TrialSuggestionFn = Callable[[optuna.Trial], IntFloatStrDict]


class Tuner:
    current_trainer: CrossValidationTrainer

    def __init__(
        self,
        model_type: type[LightningModule],
        model_config: ConfigT,
        datamodule_type: type[BaseCVDataModule],
        datamodule_config: ConfigT,
        trainer_config: CrossValidationTrainerConfig,
        logdir: FilePath = Path("checkpoints"),
        experiment_name: str = "exp0",
        clean_logdir_if_exists: bool = False,
        verbose: bool = False,
        hparam_config_file: Optional[FilePath] = None,
    ):
        self._hparam_registry = self._init_hparam_registry(hparam_config_file)

        self.model_type = model_type
        # only store dict and type so that it can be updated with new trialed values
        serialized_model_config = config_serializer(model_config)
        self.model_config_type = type(model_config)

        self.datamodule_type = datamodule_type
        self.data_config_type = type(datamodule_config)
        serialized_data_config = config_serializer(datamodule_config)

        self.trial_updater = _TrialUpdater(
            model=serialized_model_config, data=serialized_data_config
        )

        self.verbose = verbose
        self.experiment_name = experiment_name
        self.trainer_config = trainer_config.copy(deep=True)
        self._init_logdir(logdir, clean_logdir_if_exists)

        self._callbacks = standardize_callbacks(self.trainer_config.callbacks)
        self._logger_types = self._store_logger_types(self.trainer_config.loggers)

        self._trial_number = 0

    def _init_hparam_registry(
        self, config_file: Optional[FilePath]
    ) -> Optional[HparamRegistry]:
        if config_file is None:
            return None

        return HparamRegistry.from_file(config_file)

    def _init_logdir(self, logdir: FilePath, clean_logdir_if_exists: bool = False):
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

    def _store_logger_types(
        self, loggers: Union[list[Logger], Logger, None]
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

    def _create_datamodule(self) -> BaseCVDataModule:
        # these should have already been updated, but check just in case
        self._check_hparams_updated()
        serialized_data_config = self.trial_updater.data
        data_config = config_deserializer(serialized_data_config, self.data_config_type)

        if isinstance(data_config, Mapping):
            try:
                datamodule = self.datamodule_type(**data_config)
            except TypeError:
                # datamodule might expect the config as a single argument
                # which is what is tried next
                pass
            else:
                return datamodule

        # otherwise datamodule must accept only a single config argument, even if it could
        # be a dict
        return self.datamodule_type(data_config)

    def _create_model_config(self) -> ConfigT:
        # these should have already been updated, but check just in case
        self._check_hparams_updated()
        serialized_model_config = self.trial_updater.model
        model_config = config_deserializer(
            serialized_model_config, self.model_config_type
        )

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

    # TODO: this is very heavily coupled with optuna... may want to try a refactor to add support
    # for a few diff libs? particularly ray.tune since that supports multi GPU tuning
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
            logger_type(
                root_dir=self.logdir,  # type: ignore
                name=self.experiment_name,  # type: ignore
                version=self._trial_number,  # type: ignore
            )
            for logger_type in self._logger_types
        ]

        self.trainer_config.loggers = loggers

        callbacks = deepcopy(self._callbacks)
        callbacks.append(
            TrialPruning(trial=trial, monitor=monitor, verbose=self.verbose)
        )

        self.trainer_config.callbacks = callbacks

        self.current_trainer = CrossValidationTrainer(
            model_type=self.model_type, config=self.trainer_config
        )

        # update model/data config with trialed values
        self.trial_updater.update(trialed_values)

        # then update model config
        model_config = self._create_model_config()
        datamodule = self._create_datamodule()

        try:
            self.current_trainer.train_with_cross_validation(
                datamodule=datamodule, model_config=model_config
            )
        finally:
            # always increment trial number even if trial gets pruned
            # NOTE: this shouldn't be needed since this will mostly defer
            # to the `optuna.Study` update its trial numbers
            self._trial_number += 1

            self.current_trainer.log_status()
            self.trainer_config.callbacks = None

        return self.current_trainer.current_val_metrics[monitor].item()
