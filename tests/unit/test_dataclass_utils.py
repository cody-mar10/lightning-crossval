import dataclasses
from typing import Optional, TypeVar, Union, overload

import attrs
import pytest
from test_user_config_serde import compare_config_with_nested_dict

from lightning_cv.typehints import KwargType
from lightning_cv.utils.dataclasses import (
    AttrsDataclassUtilitiesMixin,
    _included_nested_dataclass,
    model_dump,
)

GLOBAL_X = 123
GLOBAL_Y = 0.428

TEST_GLOBALS = {
    "x": GLOBAL_X,
    "y": GLOBAL_Y,
    "value": "PARENT",
}


@attrs.define
class NoDefaultsSubField(AttrsDataclassUtilitiesMixin):
    x: int
    y: float


@attrs.define
class DefaultsSubField(AttrsDataclassUtilitiesMixin):
    x: int = 1
    y: float = 1.0


@attrs.define
class ValidatedSubField(AttrsDataclassUtilitiesMixin):
    x: int = attrs.field(validator=attrs.validators.instance_of(int))
    y: float = attrs.field(
        validator=attrs.validators.and_(
            attrs.validators.instance_of(float),
            attrs.validators.ge(0.0),
            attrs.validators.le(1.0),
        )
    )


@attrs.define
class ConverterSubField(AttrsDataclassUtilitiesMixin):
    x: int = attrs.field(converter=int)
    y: float = attrs.field(converter=float)


@attrs.define
class NoMixinSubField:
    x: int
    y: float


@attrs.define
class ParentConfig(AttrsDataclassUtilitiesMixin):
    value: str
    a: NoDefaultsSubField
    b: ValidatedSubField
    c: ConverterSubField
    d: NoMixinSubField
    e: DefaultsSubField = attrs.field(factory=DefaultsSubField)


types_with_mixin = [
    ParentConfig,
    NoDefaultsSubField,
    DefaultsSubField,
    ValidatedSubField,
    ConverterSubField,
]

all_types = types_with_mixin + [NoMixinSubField]

_T = TypeVar(
    "_T",
    ParentConfig,
    NoDefaultsSubField,
    DefaultsSubField,
    ValidatedSubField,
    ConverterSubField,
)

_ConfigT = Union[_T, NoMixinSubField]
_ConfigTypeT = Union[type[_T], type[NoMixinSubField]]


@overload
def _construct_config(
    config_type: type[_T], data: Optional[KwargType] = None
) -> tuple[_T, KwargType]: ...


@overload
def _construct_config(
    config_type: type[NoMixinSubField], data: Optional[KwargType] = None
) -> tuple[NoMixinSubField, KwargType]: ...


def _construct_config(
    config_type: _ConfigTypeT, data: Optional[KwargType] = None
) -> tuple[_ConfigT, KwargType]:
    if data is None:
        data = TEST_GLOBALS

    expected_config_ser = {}
    config_init_kwargs = {}
    for field in attrs.fields(config_type):
        if attrs.has(field.type):
            # really only for ParentConfig
            subfield_kwargs = {
                subfield.name: data[subfield.name]
                for subfield in attrs.fields(field.type)
            }

            expected_config_ser[field.name] = subfield_kwargs
            config_init_kwargs[field.name] = field.type(**subfield_kwargs)
        else:
            expected_config_ser[field.name] = config_init_kwargs[field.name] = data[
                field.name
            ]

    config = config_type(**config_init_kwargs)
    return config, expected_config_ser


_test_all_types = pytest.mark.parametrize("config_type", all_types)
_test_mixin_types = pytest.mark.parametrize("config_type", types_with_mixin)


@_test_all_types
def test_model_dump_fn(config_type: _ConfigTypeT):
    config, expected_config_ser = _construct_config(config_type)

    config_ser = model_dump(config)

    assert config_ser == expected_config_ser
    compare_config_with_nested_dict(config, expected_config_ser)


@_test_mixin_types
def test_to_dict(config_type: type[_T]):
    config, expected_config_ser = _construct_config(config_type)

    config_ser = config.to_dict()

    assert config_ser == expected_config_ser
    compare_config_with_nested_dict(config, expected_config_ser)


@_test_mixin_types
def test_from_dict(config_type: type[_T]):
    expected_config, config_init_kwargs = _construct_config(config_type)

    config = config_type.from_dict(config_init_kwargs)
    assert config == expected_config


def _test_copy(config_type: type[_T], deep: bool):
    expected_config, _ = _construct_config(config_type)
    config = expected_config.copy(deep=deep)

    assert config == expected_config
    assert config is not expected_config

    # ensure that subfields are/aren't the same object depending on deep
    for field in attrs.fields(config_type):
        if attrs.has(field.type):
            copied_subfield = getattr(config, field.name)
            expected_subfield = getattr(expected_config, field.name)

            if deep:
                assert copied_subfield is not expected_subfield
            else:
                assert copied_subfield is expected_subfield


@_test_mixin_types
def test_shallow_copy(config_type: type[_T]):
    _test_copy(config_type, deep=False)


@pytest.mark.parametrize(
    "config_type",
    [
        ParentConfig,
        NoDefaultsSubField,
        DefaultsSubField,
        ValidatedSubField,
        ConverterSubField,
    ],
)
def test_deep_copy(config_type: type[_T]):
    _test_copy(config_type, deep=True)


_test_flat_types = pytest.mark.parametrize(
    "config_type",
    [
        NoDefaultsSubField,
        DefaultsSubField,
        ValidatedSubField,
        ConverterSubField,
        NoMixinSubField,
    ],
)


@pytest.mark.parametrize("field", ["x", "y"])
@_test_flat_types
def test_include(config_type, field):
    config, _ = _construct_config(config_type)
    config_ser = model_dump(config, include={field})

    assert config_ser == {field: TEST_GLOBALS[field]}


@pytest.mark.parametrize("field", ["x", "y"])
@_test_flat_types
def test_exclude(config_type, field: str):
    config, _ = _construct_config(config_type)
    config_ser = model_dump(config, exclude={field})

    if field == "x":
        expected_included_field = "y"
    else:
        expected_included_field = "x"
    assert config_ser == {
        expected_included_field: TEST_GLOBALS[expected_included_field]
    }


@pytest.mark.parametrize("field", ["a", "b", "c", "d", "e"])
def test_nested_include(field: str):
    config_data = {
        "x": ord(field),
        "y": float(ord(field)) / 101,
        "value": field,
    }
    config, _ = _construct_config(ParentConfig, config_data)
    config_ser = config.to_dict(include={field})

    assert config_ser == {field: {k: v for k, v in config_data.items() if k != "value"}}


@_test_all_types
def test_passing_both_include_and_exclude(config_type):
    config, _ = _construct_config(config_type)
    with pytest.raises(ValueError):
        model_dump(config, include={"x"}, exclude={"y"})


@_test_all_types
def test_included_nested_dataclass(config_type):
    config, _ = _construct_config(config_type)
    fields = attrs.fields(config_type)
    include = {field.name for field in fields}

    expected_output = {field.name: attrs.has(field.type) for field in fields}

    is_nested = _included_nested_dataclass(config, include)

    assert is_nested == expected_output


@dataclasses.dataclass
class DataclassParentConfig:
    pass


_test_invalid_types = pytest.mark.parametrize(
    "config_type", [DataclassParentConfig, dict]
)


@_test_invalid_types
def test_non_attrs_dataclass_model_dump(config_type):
    config = config_type()
    with pytest.raises(ValueError):
        model_dump(config)


@_test_invalid_types
def test_non_attrs_dataclass_nested_dataclass(config_type):
    config = config_type()
    with pytest.raises(ValueError):
        _included_nested_dataclass(config, set())
