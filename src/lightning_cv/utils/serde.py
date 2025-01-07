"""(De)-serializing config objects for `LightningModule` and `LightningDataModule` construction.
"""

from enum import Enum

import cattrs

from lightning_cv._optional import HAS_PYDANTIC, PYDANTIC_V2
from lightning_cv.typehints import _T, KwargType
from lightning_cv.typehints.config import ConfigT

# TODO: might want to change enum handling?
# if users want to consider an enum as an hparam, then it may be challenging or awkward to accept user input
# which will most likely be the member name
# NOTE: users should create custom attrs.define converters that handle this instead....?
# could also create a custom converter for enums that converts to their values as a helpful utility
# TODO: make a test for the above case with enums for tuning tests
# the solution however will probably be to require a map for enums in the hparam config file that maps the name
# categorical choice to the enum value?
_CONVERTER = cattrs.Converter(prefer_attrib_converters=True)

# coverage has issues with thinking this block is not tested but it is
if HAS_PYDANTIC:  # pragma: no cover
    from pydantic import BaseModel

    @_CONVERTER.register_unstructure_hook
    def _pydantic_to_dict(value: BaseModel) -> KwargType:
        # mostly let pydantic handle it's own serialization, HOWEVER
        # pydantic does not serialize enums to their values like cattrs does,
        # so we need to do that manually
        ser: KwargType = value.model_dump() if PYDANTIC_V2 else value.dict()

        # walk through the dictionary and convert any Enum values to their values
        stack: list[KwargType] = [ser]
        while stack:
            curr = stack.pop()
            for key, value in curr.items():
                if isinstance(value, dict):
                    stack.append(value)
                elif isinstance(value, Enum):
                    curr[key] = value.value
        return ser

    @_CONVERTER.register_structure_hook
    def _pydantic_from_dict(value: KwargType, cls: type[BaseModel]) -> BaseModel:
        # pydantic can deserialize enum values back to enum members like cattrs
        # so don't need to worry about that
        if PYDANTIC_V2:
            return cls.model_validate(value)
        return cls.validate(value)


def config_serializer(config: ConfigT) -> KwargType:
    config_ser = _CONVERTER.unstructure(config)

    # TODO: may need to remove this depending on how we expect users to use configs?
    if not isinstance(config_ser, dict):
        raise ValueError(
            f"Could not automatically serialize the provided config: {config} of type "
            f"{type(config)}. It must be one of the following supported types: "
            "`dataclasses.dataclass`, `dict`, `pydantic.BaseModel`, or `attrs.define` dataclass."
        )

    return config_ser


def config_deserializer(serialized_config: KwargType, cls: type[_T]) -> _T:
    return _CONVERTER.structure(serialized_config, cls)
