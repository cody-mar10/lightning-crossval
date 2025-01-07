from typing import Literal

import attrs
import pytest

from lightning_cv.typehints import Number
from lightning_cv.utils.validators import (
    closed_unit_interval,
    equals,
    is_float,
    is_int,
    limit_batches_validator,
    non_negative_float,
    non_negative_int,
    open_unit_interval,
    positive_float,
    positive_int,
)


@attrs.define
class Config:
    my_int: int = attrs.field(default=0, validator=is_int)
    my_float: float = attrs.field(default=1.0, validator=is_float)
    positive_int: int = attrs.field(default=2, validator=positive_int)
    non_negative_int: int = attrs.field(default=3, validator=non_negative_int)
    positive_float: float = attrs.field(default=4.0, validator=positive_float)
    non_negative_float: float = attrs.field(default=5.0, validator=non_negative_float)
    open_unit_interval: float = attrs.field(default=0.6, validator=open_unit_interval)
    closed_unit_interval: float = attrs.field(
        default=0.7, validator=closed_unit_interval
    )
    must_be_8: Literal[8] = attrs.field(default=8, validator=equals(8))  # type: ignore
    limit_batches: Number = attrs.field(default=9, validator=limit_batches_validator)


TEST_VALUES = [
    23,
    23.0,
    "23",
    23.1,
    True,
    False,
    None,
    0.2,
    0.0,
    1.0,
    8,
    8.0,
    -44,
    -22.4,
    -0.24,
]

# It is far clearer to have the expected values rather than computing the values...

is_int_expected = [
    True,  # 23
    False,  # 23.0
    False,  # "23"
    False,  # 23.1
    True,  # True
    True,  # False
    False,  # None
    False,  # 0.2
    False,  # 0.0
    False,  # 1.0
    True,  # 8
    False,  # 8.0
    True,  # -44
    False,  # -22.4
    False,  # -0.24
]

is_float_expected = [
    False,  # 23
    True,  # 23.0
    False,  # "23"
    True,  # 23.1
    False,  # True
    False,  # False
    False,  # None
    True,  # 0.2
    True,  # 0.0
    True,  # 1.0
    False,  # 8
    True,  # 8.0
    False,  # -44
    True,  # -22.4
    True,  # -0.24
]

positive_int_expected = [
    True,  # 23
    False,  # 23.0
    False,  # "23"
    False,  # 23.1
    True,  # True
    False,  # False
    False,  # None
    False,  # 0.2
    False,  # 0.0
    False,  # 1.0
    True,  # 8
    False,  # 8.0
    False,  # -44
    False,  # -22.4
    False,  # -0.24
]

non_negative_int_expected = [
    True,  # 23
    False,  # 23.0
    False,  # "23"
    False,  # 23.1
    True,  # True
    True,  # False
    False,  # None
    False,  # 0.2
    False,  # 0.0
    False,  # 1.0
    True,  # 8
    False,  # 8.0
    False,  # -44
    False,  # -22.4
    False,  # -0.24
]

positive_float_expected = [
    False,  # 23
    True,  # 23.0
    False,  # "23"
    True,  # 23.1
    False,  # True
    False,  # False
    False,  # None
    True,  # 0.2
    False,  # 0.0
    True,  # 1.0
    False,  # 8
    True,  # 8.0
    False,  # -44
    False,  # -22.4
    False,  # -0.24
]

non_negative_float_expected = [
    False,  # 23
    True,  # 23.0
    False,  # "23"
    True,  # 23.1
    False,  # True
    False,  # False
    False,  # None
    True,  # 0.2
    True,  # 0.0
    True,  # 1.0
    False,  # 8
    True,  # 8.0
    False,  # -44
    False,  # -22.4
    False,  # -0.24
]

open_unit_interval_expected = [
    False,  # 23
    False,  # 23.0
    False,  # "23"
    False,  # 23.1
    False,  # True
    False,  # False
    False,  # None
    True,  # 0.2
    False,  # 0.0
    False,  # 1.0
    False,  # 8
    False,  # 8.0
    False,  # -44
    False,  # -22.4
    False,  # -0.24
]

closed_unit_interval_expected = [
    False,  # 23
    False,  # 23.0
    False,  # "23"
    False,  # 23.1
    False,  # True
    False,  # False
    False,  # None
    True,  # 0.2
    True,  # 0.0
    True,  # 1.0
    False,  # 8
    False,  # 8.0
    False,  # -44
    False,  # -22.4
    False,  # -0.24
]

must_be_8_expected = [
    False,  # 23
    False,  # 23.0
    False,  # "23"
    False,  # 23.1
    False,  # True
    False,  # False
    False,  # None
    False,  # 0.2
    False,  # 0.0
    False,  # 1.0
    True,  # 8
    False,  # 8.0
    False,  # -44
    False,  # -22.4
    False,  # -0.24
]

limit_batches_expected = [
    True,  # 23
    False,  # 23.0
    False,  # "23"
    False,  # 23.1
    True,  # True
    False,  # False
    False,  # None
    True,  # 0.2
    False,  # 0.0
    True,  # 1.0
    True,  # 8
    False,  # 8.0
    False,  # -44
    False,  # -22.4
    False,  # -0.24
]


def _test_validator(value, field: str, expected: bool):
    init_kwargs = {field: value}
    if not expected:
        # TODO: should we explicitly test different exceptions? the rules are fairly clear
        # for simple checks, but union validators have slightly unclear behavior...
        with pytest.raises((TypeError, ValueError)):
            Config(**init_kwargs)
    else:
        Config(**init_kwargs)


def get_test_cases(expected: list[bool]):
    return pytest.mark.parametrize(("value", "expected"), zip(TEST_VALUES, expected))


@get_test_cases(is_int_expected)
def test_is_int(value, expected: bool):
    _test_validator(value, "my_int", expected)


@get_test_cases(is_float_expected)
def test_is_float(value, expected: bool):
    _test_validator(value, "my_float", expected)


@get_test_cases(positive_int_expected)
def test_positive_int(value, expected: bool):
    _test_validator(value, "positive_int", expected)


@get_test_cases(non_negative_int_expected)
def test_non_negative_int(value, expected: bool):
    _test_validator(value, "non_negative_int", expected)


@get_test_cases(positive_float_expected)
def test_positive_float(value, expected: bool):
    _test_validator(value, "positive_float", expected)


@get_test_cases(non_negative_float_expected)
def test_non_negative_float(value, expected: bool):
    _test_validator(value, "non_negative_float", expected)


@get_test_cases(open_unit_interval_expected)
def test_open_unit_interval(value, expected: bool):
    _test_validator(value, "open_unit_interval", expected)


@get_test_cases(closed_unit_interval_expected)
def test_closed_unit_interval(value, expected: bool):
    _test_validator(value, "closed_unit_interval", expected)


@get_test_cases(must_be_8_expected)
def test_equals(value, expected: bool):
    _test_validator(value, "must_be_8", expected)


@get_test_cases(limit_batches_expected)
def test_limit_batches(value, expected: bool):
    _test_validator(value, "limit_batches", expected)
