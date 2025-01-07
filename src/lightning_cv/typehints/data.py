from typing import Iterator, Literal

from torch.utils.data import DataLoader

Stage = Literal["fit", "test", "predict"]
CVDataLoader = Iterator[tuple[DataLoader, DataLoader]]
