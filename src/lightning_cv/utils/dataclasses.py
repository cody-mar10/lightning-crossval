import copy
from typing import Optional

import attrs

from lightning_cv.typehints import KwargType
from lightning_cv.utils.serde import config_deserializer


def _included_nested_dataclass(model, include: set[str]) -> dict[str, bool]:
    fields = attrs.fields(type(model))
    is_nested = {}
    for included_field in include:
        field: attrs.Attribute = getattr(fields, included_field)
        is_nested[included_field] = attrs.has(field.type)  # type: ignore[arg-type]

    return is_nested


def model_dump(
    model,
    include: Optional[set[str]] = None,
    exclude: Optional[set[str]] = None,
) -> KwargType:
    """Convert `attrs.define` dataclasses to a dictionary. Optionally include or exclude fields
    ONLY from the top-most level. Only `include` OR `exclude` can be used, not both.

    Args:
        model `attrs.define` dataclass instance
        include (Optional[Set[str]], optional): set of top-level keys to include.
            Defaults to None, meaning include ALL.
        exclude (Optional[Set[str]], optional): set of top-level keys to exclude.
            Defaults to None, meaning exclude NOTHING.

    Raises:
        ValueError: if model is not an `attrs.define` dataclass instance

    Returns:
        KwargType (dict[str, Any]): dictionary representation of the dataclass with all str
            keys
    """
    if not attrs.has(model):
        raise ValueError(f"model must be a `attrs.define` dataclass got {type(model)}")

    if include and exclude:
        raise ValueError("Only include OR exclude can be used, not both")

    asdict_kwargs = {}
    if include is not None:
        asdict_kwargs["filter"] = attrs.filters.include(*include)

        # handle special case for dumping nested models
        included_nested_dataclass = _included_nested_dataclass(model, include)
        if any(included_nested_dataclass.values()):
            # need to recurse? or just grab the entire thing?
            # TODO: this only works for one level of nesting
            ser = attrs.asdict(model, **asdict_kwargs)
            for key, is_nested in included_nested_dataclass.items():  # pragma: no cover
                if is_nested:
                    value = getattr(model, key)
                    ser[key] = model_dump(value)

            return ser

    elif exclude is not None:
        asdict_kwargs["filter"] = attrs.filters.exclude(*exclude)

    return attrs.asdict(model, **asdict_kwargs)


@attrs.define
class AttrsDataclassUtilitiesMixin:
    """Inherit from this for pure `attrs.define` utilities methods, including serialization
    (`.to_dict`), deserilization (`.from_dict`), and copying (`.copy`).

    Subclasses are expected to also be decorated with `attrs.define`.
    """

    def to_dict(
        self, include: Optional[set[str]] = None, exclude: Optional[set[str]] = None
    ):
        # for internal config objects in this lib, just use `attrs.asdict` which is a bit
        # nicer than what cattrs does
        return model_dump(self, include, exclude)

    @classmethod
    def from_dict(cls, data: KwargType):
        return config_deserializer(data, cls)

    def copy(self, deep: bool = False):
        return copy.deepcopy(self) if deep else copy.copy(self)
