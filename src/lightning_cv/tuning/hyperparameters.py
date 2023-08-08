from __future__ import annotations

from collections import defaultdict
from pathlib import Path
from typing import Mapping, Optional

import optuna
from lightning_utilities.core.apply_func import apply_to_collection

from .config import (
    HparamConfig,
    TunableCategorical,
    TunableType,
    TunableTypeTuple,
    load_config,
)

TrialedValue = int | float | str
IntFloatStrDict = dict[str, TrialedValue | dict[str, TrialedValue]]


class HparamRegistry:
    __suggest_fields__ = {"low", "high", "step", "log", "choices", "name"}

    def __init__(self, config: HparamConfig):
        self._hparams: IntFloatStrDict = dict()
        self._hparam_config = config

    def reset(self):
        self._hparams.clear()

    @property
    def hparams(self):
        if not self._hparams:
            raise RuntimeError(
                "There are no trialed hyperparameters. You must call the "
                "`.register_hparams` method with an `optuna.Trial` object."
            )
        return self._hparams

    @classmethod
    def from_file(cls, config_file: str | Path):
        config = load_config(config_file)
        return cls(config)

    def _register(
        self, x: TunableType, trial: optuna.Trial
    ) -> tuple[str, TrialedValue, Optional[str]]:
        method = f"suggest_{x.suggest}"
        method_kwargs = x.model_dump(include=self.__suggest_fields__)
        trialed_value: TrialedValue = getattr(trial, method)(**method_kwargs)

        if isinstance(x, TunableCategorical) and x.map is not None:
            trialed_value = x.map[trialed_value]

        return x.name, trialed_value, x.parent

    def register_hparams(self, trial: optuna.Trial):
        hparams = apply_to_collection(
            data=self._hparam_config.hparams,
            dtype=TunableTypeTuple,
            function=self._register,
            trial=trial,
        )

        nested_hparams = defaultdict(dict)
        # proxy hparams may map to an arbitrary number of other hparams that are dicts
        # ex: ProxyHparam has choices [a, b, c]
        # but each choice just maps -> a: {real_hparam1: x, real_hparam2: y}
        # thus, if anything mapped, unnest and remove proxy hparam

        # further, need to convert to the correct nesting for the model config
        for key, value, parent in hparams:
            if parent is not None:
                # create a new subdict with the parent as the key
                current: dict = nested_hparams[parent]
            else:
                # otherwise add keys to root level
                current = nested_hparams

            if isinstance(value, Mapping):
                current.update(value)
            else:
                current[key] = value

        self._hparams = dict(nested_hparams)


def suggest(trial: optuna.Trial, registry: HparamRegistry) -> IntFloatStrDict:
    registry.reset()
    registry.register_hparams(trial)
    return registry.hparams
