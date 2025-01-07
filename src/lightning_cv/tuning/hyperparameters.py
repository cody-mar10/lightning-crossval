from collections import defaultdict
from typing import Mapping, Union

import optuna

from lightning_cv.tuning.config import HparamConfig, TunableCategorical, TunableType
from lightning_cv.tuning.utils import FrozenDict
from lightning_cv.typehints import FilePath

TrialedValue = Union[int, float, str]
IntFloatStrDict = dict[str, Union[TrialedValue, dict[str, TrialedValue]]]


class HparamRegistry:
    def __init__(self, config: HparamConfig):
        self.config = config
        self.named_config = config.hparams_dict()
        self.uniq_forbidden_combos = config.unique_forbidden_combos()

    @classmethod
    def from_file(cls, config_file: FilePath):
        config = HparamConfig.from_file(config_file)
        return cls(config)

    def _register(self, x: TunableType, trial: optuna.Trial, resample: bool = False):
        method = f"suggest_{x.suggest}"
        method_kwargs = x.to_optuna_kwargs()

        # for resampling trial value, such as for hparam combinations
        # that are forbidden
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
            tunable_type = self.named_config[key]
            parent = tunable_type.parent

            if parent is not None:
                # create a new subdict with the parent as the key
                current: dict = nested_hparams[parent]
            else:
                # otherwise add keys to root level
                current = nested_hparams

            if (
                isinstance(tunable_type, TunableCategorical)
                and tunable_type.map is not None
            ):
                value = tunable_type.map[value]

            if isinstance(value, Mapping):
                current.update(value)
            else:
                current[key] = value

        # TODO: we should note that there is a special key called `data`
        # reserved for datamodule/dataloader/data processing hparams
        # nothing really needs to change other than documentation
        return dict(nested_hparams)

    def register_hparams(self, trial: optuna.Trial):
        # get a suggest value for each hparam once
        for hparam in self.config.hparams:
            self._register(hparam, trial, False)

        # then check if any forbidden combinations exist, which need
        # to be resampled
        self._resolve_forbidden_combinations(trial)

        # then we need to build this into the correct nesting
        # based on the parent field for each hparam
        return self._nest_hparams(trial)


def suggest(trial: optuna.Trial, registry: HparamRegistry) -> IntFloatStrDict:
    return registry.register_hparams(trial)
