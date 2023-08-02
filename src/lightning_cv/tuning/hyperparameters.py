from __future__ import annotations

from pathlib import Path
from typing import Mapping

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
IntFloatStrDict = dict[str, TrialedValue]


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
    ) -> tuple[str, TrialedValue]:
        method = f"suggest_{x.suggest}"
        method_kwargs = x.dict(include=self.__suggest_fields__)
        trialed_value: TrialedValue = getattr(trial, method)(**method_kwargs)

        if isinstance(x, TunableCategorical) and x.map is not None:
            trialed_value = x.map[trialed_value]

        return x.name, trialed_value

    def register_hparams(self, trial: optuna.Trial):
        hparams = apply_to_collection(
            data=self._hparam_config.hparams,
            dtype=TunableTypeTuple,
            function=self._register,
            trial=trial,
        )

        # proxy hparams may map to an arbitrary number of other hparams that are dicts
        # ex: ProxyHparam has choices [a, b, c]
        # but each choice just maps -> a: {real_hparam1: x, real_hparam2: y}
        # thus, if anything mapped, unnest and remove proxy hparam
        for key, value in hparams:
            if isinstance(value, Mapping):
                self._hparams.update(value)
            else:
                self._hparams[key] = value


def suggest(trial: optuna.Trial, registry: HparamRegistry) -> IntFloatStrDict:
    registry.reset()
    registry.register_hparams(trial)
    return registry.hparams
