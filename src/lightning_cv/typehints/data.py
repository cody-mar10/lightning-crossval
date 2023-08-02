from __future__ import annotations

from typing import Iterator, Literal, Protocol, runtime_checkable

from torch.utils.data import DataLoader

Stage = Literal["fit", "test", "predict"]
CVDataLoader = Iterator[tuple[DataLoader, DataLoader]]


@runtime_checkable
class CrossValDataModuleT(Protocol):
    def setup(self, stage: Stage):
        ...

    def train_val_dataloaders(self, **dataloader_kwargs) -> CVDataLoader:
        ...
