from __future__ import annotations

from pathlib import Path
from platform import python_version_tuple
from typing import (
    Annotated,
    Any,
    Generic,
    Iterable,
    Literal,
    Optional,
    Sequence,
    TypeVar,
)

from pydantic import BaseModel, ConfigDict, Field, field_validator

if python_version_tuple() >= ("3", "11", "0"):  # pragma: >=3.11 cover
    import tomllib as tomli
else:  # pragma: <3.11 cover
    import tomli

from lightning_cv.tuning.utils import FrozenDict

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
MappingT = dict[str, dict[str, dict[str, Any]]]


class HparamConfig(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    hparams: list[TunableType]
    forbidden: Optional[set[FrozenDict]]

    @field_validator("forbidden", mode="before")
    def to_frozen(cls, values: Iterable[dict]) -> set[FrozenDict]:
        return set(FrozenDict(value) for value in values)

    def mappings(self) -> MappingT:
        mapping = {
            hparam.name: hparam.map
            for hparam in self.hparams
            if isinstance(hparam, TunableCategorical) and hparam.map is not None
        }
        return mapping

    def hparams_dict(self) -> dict[str, TunableType]:
        return {h.name: h for h in self.hparams}

    def unique_forbidden_combos(self) -> Optional[list[tuple[str, ...]]]:
        if self.forbidden is None:
            return None
        return list(set(tuple(combo.keys()) for combo in self.forbidden))


def load_config(file: str | Path, strict: bool = True) -> HparamConfig:
    with open(file, "rb") as fp:
        config = tomli.load(fp)

    validated_config = HparamConfig.model_validate(config)

    if strict and validated_config.forbidden is not None:
        field_names = set(h.name for h in validated_config.hparams)

        if not all(
            name in field_names
            for combo in validated_config.forbidden
            for name in combo
        ):
            raise ValueError(
                "Forbidden hparams must be a subset of the hparams defined in the config."
            )

    return validated_config


def get_mappings(file: str | Path) -> MappingT:
    config = load_config(file)
    return config.mappings()
