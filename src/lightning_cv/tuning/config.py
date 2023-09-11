from __future__ import annotations

from pathlib import Path
from platform import python_version_tuple
from typing import Annotated, Any, Generic, Literal, Optional, Sequence, TypeVar

from pydantic import BaseModel, Field

if python_version_tuple() >= ("3", "11", "0"):  # pragma: >=3.11 cover
    import tomllib as tomli
else:  # pragma: <3.11 cover
    import tomli

_T = TypeVar("_T")
_U = TypeVar("_U")

SuggestValues = Literal["int", "float", "categorical"]


class Tunable(BaseModel):
    name: str
    suggest: SuggestValues
    parent: Optional[str] = Field(None, description="parent for nested hparams")


class TunableInt(Tunable):
    suggest: Literal["int"]
    low: int
    high: int
    step: int = 1


class TunableFloat(Tunable):
    suggest: Literal["float"]
    low: float
    high: float
    step: Optional[float] = None
    log: bool = False


class TunableCategorical(Tunable, Generic[_T, _U]):
    suggest: Literal["categorical"]
    choices: Sequence[_T]
    map: Optional[dict[_T, _U]] = None


TunableType = Annotated[
    TunableInt | TunableFloat | TunableCategorical[_T, _U],
    Field(discriminator="suggest"),
]
TunableTypeTuple = (TunableInt, TunableFloat, TunableCategorical)
MappingT = dict[str, dict[str, dict[str, Any]]]


class HparamConfig(BaseModel):
    hparams: list[TunableType]

    def mappings(self) -> MappingT:
        mapping = {
            hparam.name: hparam.map
            for hparam in self.hparams
            if isinstance(hparam, TunableCategorical) and hparam.map is not None
        }
        return mapping


def load_config(file: str | Path) -> HparamConfig:
    with open(file, "rb") as fp:
        config = tomli.load(fp)

    validated_config = HparamConfig.model_validate(config)
    return validated_config


def get_mappings(file: str | Path) -> MappingT:
    config = load_config(file)
    return config.mappings()


def DummyHparamConfig() -> HparamConfig:
    return HparamConfig.model_construct()
