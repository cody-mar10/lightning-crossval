from __future__ import annotations

import json
from pathlib import Path
from typing import cast

import optuna

import lightning_cv as lcv


class TrialPruning(lcv.callbacks.Callback):
    # this code is basically taken from the optuna pytorch lightning integration
    # and lightly modified to work with the custom CV Trainer in this module
    _EPOCH_KEY = "ddp_pl:epoch"
    _INTERMEDIATE_VALUE = "ddp_pl:intermediate_value"
    _PRUNED_KEY = "ddp_pl:pruned"
    _SEPARATOR = "*" * 15

    def __init__(
        self, trial: optuna.Trial, monitor: str = "loss", verbose: bool = False
    ):
        self._trial = trial
        self.is_ddp_backend = False
        self.monitor = monitor
        self.verbose = verbose

    def on_train_start(self, trainer: "lcv.CrossValidationTrainer"):
        self.is_ddp_backend = trainer.is_distributed
        if self.is_ddp_backend:
            if trainer.is_global_zero:
                self._trial.storage.set_trial_system_attr(
                    self._trial._trial_id,
                    self._INTERMEDIATE_VALUE,
                    dict(),
                )

        print(
            f"\n{self._SEPARATOR} Trial {self._trial._trial_id} BEGIN {self._SEPARATOR}"
            "\n"
        )

    def on_train_end(self, trainer: "lcv.CrossValidationTrainer"):
        print(
            f"\n{self._SEPARATOR} Trial {self._trial._trial_id} END {self._SEPARATOR}\n"
        )

    def on_train_fold_end(self, trainer: "lcv.CrossValidationTrainer"):
        current_score = trainer.current_val_metrics[self.monitor].item()
        current_epoch = trainer.current_epoch
        should_stop = False
        self._trial.report(current_score, current_epoch)

        # single process training
        if not self.is_ddp_backend:
            if not self._trial.should_prune():
                return
            raise optuna.TrialPruned(self.prune_message(epoch=current_epoch))

        # distributed training
        if trainer.is_global_zero:
            should_stop = self._trial.should_prune()

            _trial_id = self._trial._trial_id
            _study = self._trial.study
            _trial_system_attrs = _study._storage.get_trial_system_attrs(_trial_id)

            intermediate_values = _trial_system_attrs.get(self._INTERMEDIATE_VALUE)
            intermediate_values[current_epoch] = current_score  # type: ignore[index]
            self._trial.storage.set_trial_system_attr(
                self._trial._trial_id,
                self._INTERMEDIATE_VALUE,
                intermediate_values,
            )

        should_stop = trainer.fabric.strategy.broadcast(should_stop)
        trainer.should_stop = trainer.should_stop or should_stop

        if not should_stop:
            return

        if trainer.is_global_zero:
            # Update system_attr from global zero process.
            self._trial.storage.set_trial_system_attr(
                self._trial._trial_id, self._PRUNED_KEY, True
            )
            self._trial.storage.set_trial_system_attr(
                self._trial._trial_id, self._EPOCH_KEY, current_epoch
            )

    def check_pruned(self):
        """Raise :class:`optuna.TrialPruned` manually if pruned.

        Currently, ``intermediate_values`` are not properly propagated between processes
        due to storage cache. Therefore, necessary information is kept in
        trial_system_attrs when the trial runs in a distributed situation. Please call
        this method right after calling
        ``lightning_cv.CrossValidationTrainer.train_with_cross_validation()``.

        If a callback doesn't have any backend storage for DDP, this method does
        nothing.
        """
        _trial_id = self._trial._trial_id
        _study = self._trial.study
        # Confirm if storage is not InMemory in case this method is called in a
        # non-distributed situation by mistake.
        if not isinstance(_study._storage, optuna.storages._CachedStorage):
            return

        _trial_system_attrs = _study._storage._backend.get_trial_system_attrs(_trial_id)
        is_pruned = _trial_system_attrs.get(self._PRUNED_KEY)
        intermediate_values = _trial_system_attrs.get(self._INTERMEDIATE_VALUE)

        # Confirm if DDP backend is used in case this method is called from a
        # non-DDP situation by mistake.
        if intermediate_values is None:
            return
        for epoch, score in intermediate_values.items():
            self._trial.report(score, step=int(epoch))
        if is_pruned:
            epoch = cast(int, _trial_system_attrs.get(self._EPOCH_KEY))
            raise optuna.TrialPruned(self.prune_message(epoch=epoch))

    def prune_message(self, epoch: int):
        msg = [f"Trial was pruned at epoch {epoch}"]

        if self.verbose:
            msg.append(f"with parameters {self._trial.params}")

        return " ".join(msg)


class OptunaHyperparameterLogger:
    def __init__(
        self,
        root_dir: Path,
        logfile: str = "sampled_hparams.json",
        monitor: str = "loss",
    ):
        self.root_dir = root_dir
        self.logfile = logfile
        self.monitor = monitor if monitor.startswith("val_") else f"val_{monitor}"

    def __call__(self, study: optuna.Study, trial: optuna.trial.FrozenTrial):
        trial_id = trial.number
        expt_name = study.study_name
        logdir = self.root_dir / expt_name / f"version_{trial_id}"
        logfile = logdir.joinpath(self.logfile)

        record = {
            "monitor": self.monitor,
            "performance": trial.intermediate_values,
            "hparams": trial.params,
        }

        with logfile.open("w") as fp:
            json.dump(record, fp, indent=4)
            fp.write("\n")
