"""Validators for `attrs.define` dataclass-like config objects."""

from attrs.validators import and_, ge, gt, instance_of, le, lt, or_

is_int = instance_of(int)
is_float = instance_of(float)

positive_int = and_(is_int, gt(0))
non_negative_int = and_(is_int, ge(0))
positive_float = and_(is_float, gt(0.0))
non_negative_float = and_(is_float, ge(0.0))
open_unit_interval = and_(is_float, gt(0.0), lt(1.0))
closed_unit_interval = and_(is_float, ge(0.0), le(1.0))


def equals(expected_value):
    expected_type = type(expected_value)
    return and_(
        instance_of(expected_type),
        ge(expected_value),
        le(expected_value),
    )


_left_open_right_closed_unit_interval = and_(
    instance_of(float),
    gt(0.0),
    le(1.0),
)

limit_batches_validator = or_(
    positive_int,
    _left_open_right_closed_unit_interval,  # type: ignore
)
