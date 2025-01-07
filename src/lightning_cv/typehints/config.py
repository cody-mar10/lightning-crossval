import sys
from dataclasses import Field
from typing import ClassVar, Protocol, Union
from attrs import AttrsInstance

from lightning_cv.typehints import KwargType

if sys.version_info >= (3, 11):  # pragma: >=3.11 cover
    from typing import Self
else:  # pragma: <3.11 cover
    from typing_extensions import Self


class DataclassInstance(Protocol):
    __dataclass_fields__: ClassVar[dict[str, Field]]


class PydanticBaseModel(Protocol):
    def dict(self) -> KwargType: ...
    def model_dump(self) -> KwargType: ...
    def model_validate(self, value: KwargType) -> Self: ...
    def validate(self, value: KwargType) -> Self: ...


ConfigT = Union[KwargType, AttrsInstance, DataclassInstance, PydanticBaseModel]
