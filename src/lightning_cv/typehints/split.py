from __future__ import annotations

from typing import Any, Iterator, Protocol, runtime_checkable

import numpy as np
from numpy.typing import NDArray

Int64Array = NDArray[np.int64]
CVIterator = Iterator[tuple[Int64Array, Int64Array]]


@runtime_checkable
class CrossValidator(Protocol):
    def split(self, *args: Any, **kwargs: Any) -> CVIterator:
        ...

    def get_n_splits(self, *args: Any, **kwargs: Any) -> int:
        ...


@runtime_checkable
class GroupCrossValidator(CrossValidator, Protocol):
    @property
    def train_group_ids(self) -> list[int]:
        ...

    @property
    def val_group_id(self) -> int:
        ...
