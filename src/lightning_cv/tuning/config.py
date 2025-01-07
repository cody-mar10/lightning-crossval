import sys
from typing import Any, Literal, Optional, Union
from warnings import warn

import attrs

if sys.version_info >= (3, 11):  # pragma: >=3.11 cover
    import tomllib as tomli
else:  # pragma: <3.11 cover
    import tomli

from lightning_cv.tuning.utils import FrozenDict
from lightning_cv.typehints import FilePath
from lightning_cv.utils.dataclasses import AttrsDataclassUtilitiesMixin

SuggestValues = Literal["int", "float", "categorical"]


@attrs.define
class Tunable(AttrsDataclassUtilitiesMixin):
    name: str
    suggest: SuggestValues

    def to_optuna_kwargs(self):
        # only include fields that are passed to optuna.Trial.suggest_*
        exclude = {"parent", "map", "suggest"}
        return self.to_dict(exclude=exclude)


@attrs.define
class TunableInt(Tunable):
    suggest: Literal["int"]
    low: int
    high: int
    step: int = 1
    parent: Optional[str] = None


@attrs.define
class TunableFloat(Tunable):
    suggest: Literal["float"]
    low: float
    high: float
    step: Optional[float] = None
    log: bool = False
    parent: Optional[str] = None


@attrs.define
class TunableCategorical(Tunable):
    suggest: Literal["categorical"]
    choices: list
    map: Optional[dict] = None
    parent: Optional[str] = None


TunableType = Union[TunableInt, TunableFloat, TunableCategorical]
MappingT = dict[str, dict[str, dict[str, Any]]]


def _forbidden_converter(value: Optional[list[dict]]) -> Optional[set[FrozenDict]]:
    if value is None:
        return None

    return set(FrozenDict(**combo) for combo in value)


@attrs.define
class HparamConfig(AttrsDataclassUtilitiesMixin):
    hparams: list[TunableType]
    forbidden: Optional[set[FrozenDict]] = attrs.field(
        default=None, converter=_forbidden_converter
    )
    strict: bool = True

    def __attrs_post_init__(self):
        if self.forbidden is not None:
            hparam_names = set(hparam.name for hparam in self.hparams)
            all_forbidden_hparams_subset_of_hparams = all(
                forbidden_hparam_name in hparam_names
                for forbidden in self.forbidden
                for forbidden_hparam_name in forbidden.keys()
            )

            if not all_forbidden_hparams_subset_of_hparams:
                msg = (
                    "Forbidden hparams must be a subset of the hparams defined in the config."
                    f" Received: {self.forbidden}"
                )
                if self.strict:
                    raise ValueError(msg)
                else:
                    msg = f"WARNING: {msg}. This could lead to unexpected behavior if proceeding."
                    warn(msg, RuntimeWarning)

    @classmethod
    def from_file(cls, file: FilePath, strict: bool = True):
        with open(file, "rb") as fp:
            config = tomli.load(fp)

        config["strict"] = config.get("strict", strict)

        return cls.from_dict(config)

    def mappings(self) -> MappingT:
        mapping = {
            hparam.name: hparam.map
            for hparam in self.hparams
            if isinstance(hparam, TunableCategorical) and hparam.map is not None
        }

        return mapping

    def hparams_dict(self):
        return {hparam.name: hparam for hparam in self.hparams}

    def unique_forbidden_combos(self) -> Optional[list[tuple[str, ...]]]:
        if self.forbidden is None:
            return None

        return list(set(tuple(combo.keys()) for combo in self.forbidden))


def get_mappings(file: FilePath) -> MappingT:
    config = HparamConfig.from_file(file)
    return config.mappings()
