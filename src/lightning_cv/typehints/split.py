from __future__ import annotations

from typing import Callable, Iterator, Protocol, runtime_checkable

import numpy as np
from numpy.typing import NDArray

Int64Array = NDArray[np.int64]
CVIterator = Iterator[tuple[Int64Array, Int64Array]]

Type = Callable[..., CVIterator]


@runtime_checkable
class CrossValidator(Protocol):
    # ignore input types and just require to implement these methods
    split: Callable[..., CVIterator]
    get_n_splits: Callable[..., int]


@runtime_checkable
class GroupCrossValidator(CrossValidator, Protocol):
    @property
    def train_group_ids(self) -> list[int]:
        ...

    @property
    def val_group_id(self) -> int:
        ...
