from __future__ import annotations

from copy import copy
from dataclasses import dataclass
from typing import Callable, Iterator, Literal, Optional

import torch
from lightning import LightningDataModule
from torch.utils.data import DataLoader, Dataset

from ._typing import Int64Array, KwargType
from .split import CrossValidator, GroupCrossValidator

Stage = Literal["fit", "test", "predict"]


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


@dataclass
class CVDataLoader:
    train_loader: DataLoader
    val_loader: DataLoader


@dataclass
class GroupCVDataLoader(CVDataLoader):
    train_group_ids: list[int]
    val_group_id: int


class _CrossValidationDataModule(LightningDataModule):
    def __init__(
        self,
        dataset: Dataset,
        batch_size: int,
        cross_validator: type[CrossValidator | GroupCrossValidator],
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

    def train_val_dataloaders(self, **dataloader_kwargs) -> Iterator[CVDataLoader]:
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

            cv_dataloader = CVDataLoader(train_loader, val_loader)

            yield cv_dataloader

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


class CrossValidationDataModule(_CrossValidationDataModule):
    def __init__(
        self,
        dataset: Dataset,
        batch_size: int,
        cross_validator: type[CrossValidator],
        cross_validator_config: KwargType,
        collate_fn: Optional[Callable] = None,
        **dataloader_kwargs,
    ) -> None:
        super().__init__(
            dataset,
            batch_size,
            cross_validator,
            cross_validator_config,
            collate_fn=collate_fn,
            **dataloader_kwargs,
        )
        self.data_manager: CrossValidator


class GroupCrossValidationDataModule(_CrossValidationDataModule):
    def __init__(
        self,
        dataset: Dataset,
        batch_size: int,
        cross_validator: type[GroupCrossValidator],
        cross_validator_config: KwargType,
        collate_fn: Optional[Callable] = None,
        **dataloader_kwargs,
    ) -> None:
        super().__init__(
            dataset,
            batch_size,
            cross_validator,
            cross_validator_config,
            collate_fn=collate_fn,
            **dataloader_kwargs,
        )
        self.data_manager: GroupCrossValidator

    def train_val_dataloaders(self, **dataloader_kwargs) -> Iterator[GroupCVDataLoader]:
        for cv_dataloader in super().train_val_dataloaders(**dataloader_kwargs):
            group_cv_dataloader = GroupCVDataLoader(
                train_loader=cv_dataloader.train_loader,
                val_loader=cv_dataloader.val_loader,
                train_group_ids=self.data_manager.train_group_ids,
                val_group_id=self.data_manager.val_group_id,
            )

            yield group_cv_dataloader
