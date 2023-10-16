from __future__ import annotations

from pathlib import Path
from platform import python_version_tuple
from typing import Annotated, Any, Generic, Literal, Optional, Sequence, TypeVar

from pydantic import BaseModel, Field, model_validator

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
    forbidden: Optional[list[dict]] = None

    def mappings(self) -> MappingT:
        mapping = {
            hparam.name: hparam.map
            for hparam in self.hparams
            if isinstance(hparam, TunableCategorical) and hparam.map is not None
        }
        return mapping

    def hparams_dict(self) -> dict[str, TunableType]:
        return {h.name: h for h in self.hparams}

    def mapped_values_to_category(self) -> dict[str, str]:
        return {
            key: field
            for field, mapping in self.mappings().items()
            for value in mapping.values()
            for key in value.keys()
        }

    @model_validator(mode="after")
    def map_forbidden(self):
        if self.forbidden is not None:
            named_config = self.hparams_dict()

            new_forbidden: list[dict[str, Any]] = list()
            for combo in self.forbidden:
                new_combo = dict()
                for key in combo.keys():
                    tunable_type = named_config[key]

                    if (
                        isinstance(tunable_type, TunableCategorical)
                        and tunable_type.map is not None
                    ):
                        new_key = tunable_type.map[combo[key]]
                        if isinstance(new_key, dict):
                            new_combo.update(new_key)
                        else:
                            new_combo[key] = new_key
                    else:
                        new_combo[key] = combo[key]
                new_forbidden.append(new_combo)
            self.forbidden = new_forbidden
        return self


def load_config(file: str | Path, strict: bool = True) -> HparamConfig:
    with open(file, "rb") as fp:
        config = tomli.load(fp)

    validated_config = HparamConfig.model_validate(config)

    if strict and validated_config.forbidden is not None:
        field_names = set()

        for h in validated_config.hparams:
            field_names.add(h.name)

            if isinstance(h, TunableCategorical) and h.map is not None:
                for key, value in h.map.items():
                    if isinstance(value, dict):
                        field_names.update(value.keys())
                    else:
                        field_names.update(key)

        forbidden_names = set(
            key for combo in validated_config.forbidden for key in combo.keys()
        )

        if not all(name in field_names for name in forbidden_names):
            raise ValueError(
                "Forbidden hparams must be a subset of the hparams defined in the config."
            )

    return validated_config


def get_mappings(file: str | Path) -> MappingT:
    config = load_config(file)
    return config.mappings()


def DummyHparamConfig() -> HparamConfig:
    return HparamConfig.model_construct()
