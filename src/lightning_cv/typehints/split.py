from typing import Callable, Iterator, Protocol

from numpy import int64
from numpy.typing import NDArray

Int64Array = NDArray[int64]
CVIterator = Iterator[tuple[Int64Array, Int64Array]]


class CrossValidator(Protocol):
    # ignore input types and just require to implement these methods
    split: Callable[..., CVIterator]
