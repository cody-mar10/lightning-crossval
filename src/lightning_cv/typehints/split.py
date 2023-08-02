from __future__ import annotations

from typing import Iterator, Protocol, overload, runtime_checkable

import numpy as np
from numpy.typing import NDArray

Int64Array = NDArray[np.int64]
CVIterator = Iterator[tuple[Int64Array, Int64Array]]


@runtime_checkable
class CrossValidator(Protocol):
    @overload
    def split(self) -> CVIterator:
        ...

    @overload
    def split(self, X, y, groups) -> CVIterator:
        ...

    @overload
    def get_n_splits(self) -> int:
        ...

    @overload
    def get_n_splits(self, X, y, groups) -> CVIterator:
        ...


@runtime_checkable
class GroupCrossValidator(CrossValidator, Protocol):
    @property
    def train_group_ids(self) -> list[int]:
        ...

    @property
    def val_group_id(self) -> int:
        ...
