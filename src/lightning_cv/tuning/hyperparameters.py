from __future__ import annotations

from collections import defaultdict
from pathlib import Path
from typing import Mapping

import optuna

from lightning_cv.tuning.config import (
    HparamConfig,
    TunableCategorical,
    TunableType,
    load_config,
)
from lightning_cv.tuning.utils import FrozenDict

TrialedValue = int | float | str
IntFloatStrDict = dict[str, TrialedValue | dict[str, TrialedValue]]


class HparamRegistry:
    __suggest_fields__ = {"low", "high", "step", "log", "choices", "name"}

    def __init__(self, config: HparamConfig):
        self.config = config
        self.named_config = config.hparams_dict()
        self.uniq_forbidden_combos = config.unique_forbidden_combos()

    @classmethod
    def from_file(cls, config_file: str | Path):
        config = load_config(config_file)
        return cls(config)

    def _register(self, x: TunableType, trial: optuna.Trial, resample: bool = False):
        method = f"suggest_{x.suggest}"
        method_kwargs = x.model_dump(include=self.__suggest_fields__)

        if x.name in trial.distributions and resample:
            # need to delete previously sampled value
            # otherwise this will just keep returning the same value
            del trial._cached_frozen_trial.distributions[x.name]

        # no need to return since this will be part of trial.params
        getattr(trial, method)(**method_kwargs)

    def _resolve_forbidden_combinations(self, trial: optuna.Trial):
        if self.config.forbidden is None or self.uniq_forbidden_combos is None:
            return

        hparams = trial.params
        for keys in self.uniq_forbidden_combos:
            current = FrozenDict({key: hparams[key] for key in keys})

            while current in self.config.forbidden:
                new: IntFloatStrDict = dict()
                for key in keys:
                    tunable_type = self.named_config[key]
                    self._register(tunable_type, trial, resample=True)
                    new[key] = trial.params[key]
                current = FrozenDict(new)

    def _nest_hparams(self, trial: optuna.Trial) -> IntFloatStrDict:
        nested_hparams = defaultdict(dict)
        # proxy hparams may map to an arbitrary number of other hparams that are dicts
        # ex: ProxyHparam has choices [a, b, c]
        # but each choice just maps -> a: {real_hparam1: x, real_hparam2: y}
        # thus, if anything mapped, unnest and remove proxy hparam

        # further, need to convert to the correct nesting for the model config
        for key, value in trial.params.items():
            type_config = self.named_config[key]
            parent = self.named_config[key].parent

            if parent is not None:
                # create a new subdict with the parent as the key
                current: dict = nested_hparams[parent]
            else:
                # otherwise add keys to root level
                current = nested_hparams

            if (
                isinstance(type_config, TunableCategorical)
                and type_config.map is not None
            ):
                value = type_config.map[value]

            if isinstance(value, Mapping):
                current.update(value)
            else:
                current[key] = value

        return dict(nested_hparams)

    def register_hparams(self, trial: optuna.Trial):
        for hparam in self.config.hparams:
            self._register(hparam, trial, False)

        self._resolve_forbidden_combinations(trial)
        return self._nest_hparams(trial)


def suggest(trial: optuna.Trial, registry: HparamRegistry) -> IntFloatStrDict:
    return registry.register_hparams(trial)
