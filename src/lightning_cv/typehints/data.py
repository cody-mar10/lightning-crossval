from __future__ import annotations

from dataclasses import Field
from typing import (
    Any,
    ClassVar,
    Iterator,
    Literal,
    Protocol,
    overload,
    runtime_checkable,
)

from pydantic import BaseModel
from torch.utils.data import DataLoader

Stage = Literal["fit", "test", "predict"]
CVDataLoader = Iterator[tuple[DataLoader, DataLoader]]


# this is the typeshed impl
class DataclassInstance(Protocol):
    __dataclass_fields__: ClassVar[dict[str, Field[Any]]]


@runtime_checkable
class CrossValDataModuleT(Protocol):
    @overload
    def __init__(self, config: BaseModel):
        ...

    @overload
    def __init__(self, config: DataclassInstance):
        ...

    @overload
    def __init__(self, **kwargs: Any):
        ...

    def setup(self, stage: Stage):
        ...

    def train_val_dataloaders(self, **dataloader_kwargs) -> CVDataLoader:
        ...
