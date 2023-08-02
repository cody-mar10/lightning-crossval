from __future__ import annotations

from copy import copy
from typing import Callable, Iterator, Literal, Optional, Protocol, runtime_checkable

import torch
from lightning import LightningDataModule
from torch.utils.data import DataLoader, Dataset

from ._typing import Int64Array, KwargType
from .split import CrossValidator

Stage = Literal["fit", "test", "predict"]
CVDataLoader = Iterator[tuple[DataLoader, DataLoader]]


class SimpleTensorDataset(Dataset):
    def __init__(self, tensor: torch.Tensor) -> None:
        super().__init__()
        self.data = tensor

    def __len__(self) -> int:
        return int(self.data.size(0))

    def __getitem__(self, idx: int | list[int]) -> torch.Tensor:
        return self.data[idx]

    @staticmethod
    def collate(batch: list[torch.Tensor]) -> torch.Tensor:
        return torch.stack(batch)


@runtime_checkable
class CrossValDataModuleT(Protocol):
    def setup(self, stage: Stage):
        ...

    def train_val_dataloaders(self, **dataloader_kwargs) -> CVDataLoader:
        ...


class CrossValidationDataModule(LightningDataModule):
    def __init__(
        self,
        dataset: Dataset,
        batch_size: int,
        cross_validator: type[CrossValidator],
        cross_validator_config: KwargType,
        collate_fn: Optional[Callable] = None,
        **dataloader_kwargs,
    ) -> None:
        super().__init__()

        self.dataset = dataset
        self.batch_size = batch_size
        self.dataloader_kwargs = dataloader_kwargs
        self._cross_validator = cross_validator
        self._cross_validator_config = cross_validator_config
        self.collate_fn = collate_fn

    def setup(self, stage: Stage):
        if stage == "fit":
            self.data_manager = self._cross_validator(**self._cross_validator_config)
        else:
            raise ValueError(
                f"{stage=} not allowed with cross validation. Only the `fit` stage is "
                "allowed."
            )

    def train_val_dataloaders(self, **dataloader_kwargs) -> CVDataLoader:
        train_kwargs, val_kwargs = self._split_train_val_kwargs(**dataloader_kwargs)

        for train_idx, val_idx in self.data_manager.split():
            train_idx, val_idx = self._from_numpy(train_idx, val_idx)
            train_idx_dataset = SimpleTensorDataset(train_idx)
            val_idx_dataset = SimpleTensorDataset(val_idx)

            train_loader = DataLoader(
                dataset=train_idx_dataset,
                batch_size=self.batch_size,
                collate_fn=self.collate_fn,
                **train_kwargs,
            )
            val_loader = DataLoader(
                dataset=val_idx_dataset,
                batch_size=self.batch_size,
                collate_fn=self.collate_fn,
                **val_kwargs,
            )

            yield train_loader, val_loader

    def _split_train_val_kwargs(self, **kwargs) -> tuple[KwargType, KwargType]:
        train_kwargs = self._overwrite_dataloader_kwargs(**kwargs)
        val_kwargs = copy(train_kwargs)
        val_kwargs["shuffle"] = False
        return train_kwargs, val_kwargs

    def _overwrite_dataloader_kwargs(self, **kwargs) -> KwargType:
        # overwrite the shared/global dataloader kwargs with kwargs specifically input
        # and then return a copy
        return self.dataloader_kwargs | kwargs

    def _from_numpy(self, *arrays: Int64Array) -> tuple[torch.Tensor, ...]:
        tensors = [torch.from_numpy(array) for array in arrays]
        return tuple(tensors)
