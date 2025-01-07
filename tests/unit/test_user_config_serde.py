import dataclasses
from enum import Enum
from typing import Any, Iterator, TypedDict

import attrs
import pytest

from lightning_cv._optional import HAS_PYDANTIC, PYDANTIC_V2
from lightning_cv.typehints import KwargType
from lightning_cv.utils.serde import config_deserializer, config_serializer
from lightning_cv.utils.validators import (
    closed_unit_interval,
    positive_float,
    positive_int,
)


class ModelMode(Enum):
    TRAIN = "train"
    VALID = "valid"
    TEST = "test"


@pytest.fixture
def config_data() -> KwargType:
    return {
        "embed_dim": 256,
        "num_heads": 8,
        "num_layers": 5,
        "mode": ModelMode.TEST,
        "subfield": {"mask_rate": 0.27, "sample_scale": 1.23},
    }


def _convert_enum_to_value(config_data: KwargType) -> KwargType:
    import copy

    copied_config_data = copy.deepcopy(config_data)
    copied_config_data["mode"] = copied_config_data["mode"].value
    return copied_config_data


def _check_embed_dim_divisible_by_num_heads(embed_dim: int, num_heads: int):
    if (embed_dim % num_heads) != 0:
        raise ValueError(
            f"embed_dim must be divisible by num_heads. Got {embed_dim=} and {num_heads=}"
        )


class TypedDictSubField(TypedDict):
    mask_rate: float
    sample_scale: float


class TypedDictModelConfig(TypedDict):
    embed_dim: int
    num_heads: int
    num_layers: int
    mode: ModelMode
    subfield: TypedDictSubField


@dataclasses.dataclass
class DataclassSubField:
    mask_rate: float = 0.15
    sample_scale: float = 0.5

    def __post_init__(self):
        if not 0 <= self.mask_rate <= 1:
            raise ValueError(
                f"mask_rate must be in the range [0, 1]. Got mask_rate={self.mask_rate}"
            )

        if self.sample_scale <= 0:
            raise ValueError(
                f"sample_scale must be greater than 0. Got sample_scale={self.sample_scale}"
            )


@dataclasses.dataclass
class DataclassModelConfig:
    embed_dim: int
    num_heads: int = 1
    num_layers: int = 12
    mode: ModelMode = dataclasses.field(default=ModelMode.TRAIN)
    subfield: DataclassSubField = dataclasses.field(default_factory=DataclassSubField)

    def __post_init__(self):
        _check_embed_dim_divisible_by_num_heads(self.embed_dim, self.num_heads)

        if self.num_heads <= 0:
            raise ValueError(
                f"num_heads must be greater than 0. Got num_heads={self.num_heads}"
            )

        if self.num_layers <= 0:
            raise ValueError(
                f"num_layers must be greater than 0. Got num_layers={self.num_layers}"
            )


@attrs.define
class AttrsSubField:
    mask_rate: float = attrs.field(default=0.15, validator=closed_unit_interval)
    sample_scale: float = attrs.field(default=0.5, validator=positive_float)


@attrs.define
class AttrsModelConfig:
    embed_dim: int = attrs.field(validator=positive_int)
    num_heads: int = attrs.field(default=1, validator=positive_int)
    num_layers: int = attrs.field(default=12, validator=positive_int)
    mode: ModelMode = ModelMode.TRAIN
    subfield: AttrsSubField = attrs.field(factory=AttrsSubField)

    def __attrs_post_init__(self):
        _check_embed_dim_divisible_by_num_heads(self.embed_dim, self.num_heads)


Path = tuple[str, ...]
DictPathValues = tuple[Path, Any]


def walk_nested_dict(d: dict) -> Iterator[DictPathValues]:
    stack: list[tuple[Path, dict]] = [((), d)]
    while stack:
        path, curr = stack.pop()
        for key, value in curr.items():
            if isinstance(value, dict):
                stack.append(((*path, key), value))
            else:
                yield (*path, key), value


def compare_config_with_nested_dict(config, expected: dict):
    for path, value in walk_nested_dict(expected):
        curr = config
        for key in path:
            curr = getattr(curr, key)
        assert curr == value


class _BaseTestDict:
    def test_serialize(self, config_data: KwargType):
        serialized = config_serializer(config_data)
        assert serialized == _convert_enum_to_value(config_data)


class TestDict(_BaseTestDict):
    # these should basically do nothing except copy the dict
    def test_deserialize(self, config_data: KwargType):
        deserialized = config_deserializer(config_data, dict)
        assert deserialized == config_data

    def test_deserialize_with_enum_value(self, config_data: KwargType):
        config_data_with_enum_value = _convert_enum_to_value(config_data)

        # NOTE: since dict fields are not typed, the enum won't be converted
        deserialized = config_deserializer(config_data_with_enum_value, dict)

        assert deserialized == config_data_with_enum_value


class TestTypedDict:
    def test_deserialize(self, config_data: KwargType):
        deserialized = config_deserializer(config_data, TypedDictModelConfig)
        assert deserialized == config_data

    def test_deserialize_with_enum_value(self, config_data: KwargType):
        config_data_with_enum_value = _convert_enum_to_value(config_data)

        # cattrs deserializes TypedDicts correctly
        deserialized = config_deserializer(
            config_data_with_enum_value, TypedDictModelConfig
        )

        assert deserialized == config_data


class TestDataclasses:
    def test_serialize(self, config_data: KwargType):
        config = DataclassModelConfig(
            embed_dim=config_data["embed_dim"],
            num_heads=config_data["num_heads"],
            num_layers=config_data["num_layers"],
            mode=config_data["mode"],
            subfield=DataclassSubField(**config_data["subfield"]),
        )

        serialized = config_serializer(config)
        assert serialized == _convert_enum_to_value(config_data)

    def test_deserialize(self, config_data: KwargType):
        config = config_deserializer(config_data, DataclassModelConfig)

        assert not isinstance(config.subfield, dict)
        compare_config_with_nested_dict(config, config_data)

    def test_deserialize_with_enum_value(self, config_data: KwargType):
        config_data_with_enum_value = _convert_enum_to_value(config_data)
        config = config_deserializer(config_data_with_enum_value, DataclassModelConfig)

        assert not isinstance(config.subfield, dict)
        compare_config_with_nested_dict(config, config_data)


class TestAttrs:
    def test_serialize(self, config_data: KwargType):
        config = AttrsModelConfig(
            embed_dim=config_data["embed_dim"],
            num_heads=config_data["num_heads"],
            num_layers=config_data["num_layers"],
            mode=config_data["mode"],
            subfield=AttrsSubField(**config_data["subfield"]),
        )

        serialized = config_serializer(config)
        assert serialized == _convert_enum_to_value(config_data)

    def test_deserialize(self, config_data: KwargType):
        config = config_deserializer(config_data, AttrsModelConfig)

        assert not isinstance(config.subfield, dict)
        compare_config_with_nested_dict(config, config_data)

    def test_deserialize_with_enum_value(self, config_data: KwargType):
        config_data_with_enum_value = _convert_enum_to_value(config_data)
        config = config_deserializer(config_data_with_enum_value, AttrsModelConfig)

        assert not isinstance(config.subfield, dict)
        compare_config_with_nested_dict(config, config_data)


if HAS_PYDANTIC:
    import pydantic

    class PydanticSubField(pydantic.BaseModel):
        mask_rate: float = pydantic.Field(default=0.15, ge=0, le=1)
        sample_scale: float = pydantic.Field(default=0.5, gt=0)

    class PydanticModelConfig(pydantic.BaseModel):
        embed_dim: int = pydantic.Field(gt=0)
        num_heads: int = pydantic.Field(default=1, gt=0)
        num_layers: int = pydantic.Field(default=12, gt=0)
        mode: ModelMode = ModelMode.TRAIN
        subfield: PydanticSubField = pydantic.Field(default_factory=PydanticSubField)

        if PYDANTIC_V2:

            @pydantic.model_validator(mode="after")
            def check_embed_dim_divisible_by_num_heads_V2(self):
                _check_embed_dim_divisible_by_num_heads(self.embed_dim, self.num_heads)
                return self

        else:

            @pydantic.root_validator(pre=True)
            def check_embed_dim_divisible_by_num_heads_V1(cls, values):
                _check_embed_dim_divisible_by_num_heads(
                    values["embed_dim"], values["num_heads"]
                )
                return values


@pytest.mark.skipif(not HAS_PYDANTIC, reason="Pydantic is not installed")
class TestPydantic:
    def test_serialize(self, config_data: KwargType):
        config = PydanticModelConfig(
            embed_dim=config_data["embed_dim"],
            num_heads=config_data["num_heads"],
            num_layers=config_data["num_layers"],
            mode=config_data["mode"],
            subfield=PydanticSubField(**config_data["subfield"]),
        )

        serialized = config_serializer(config)
        assert serialized == _convert_enum_to_value(config_data)

    def test_deserialize(self, config_data: KwargType):
        config = config_deserializer(config_data, PydanticModelConfig)

        assert not isinstance(config.subfield, dict)
        compare_config_with_nested_dict(config, config_data)

    def test_deserialize_with_enum_value(self, config_data: KwargType):
        config_data_with_enum_value = _convert_enum_to_value(config_data)
        config = config_deserializer(config_data_with_enum_value, PydanticModelConfig)

        assert not isinstance(config.subfield, dict)
        compare_config_with_nested_dict(config, config_data)


@pytest.mark.parametrize("value", ["string", 123, 0.123, True, None])
def test_non_config_like_serialization(value):
    with pytest.raises(ValueError):
        config_serializer(value)
