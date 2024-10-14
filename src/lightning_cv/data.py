from __future__ import annotations

from copy import copy
from typing import Callable, Optional, Protocol, runtime_checkable

import torch
from lightning import LightningDataModule
from torch.utils.data import DataLoader, Dataset

from lightning_cv.typehints import (
    CrossValidator,
    CVDataLoader,
    Int64Array,
    KwargType,
    Stage,
)


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
    def setup(self, stage: Stage): ...

    def train_val_dataloaders(self, **dataloader_kwargs) -> CVDataLoader: ...


class CrossValidationDataModule(LightningDataModule):
    def __init__(
        self,
        dataset: Dataset,
        batch_size: int,
        cross_validator: type[CrossValidator],
        cross_validator_config: KwargType,
        collate_fn: Optional[Callable] = None,
        # attr names
        y_attr: Optional[str] = None,
        group_attr: Optional[str] = None,
        **dataloader_kwargs,
    ) -> None:
        super().__init__()

        self.dataset = dataset
        self.dataset_size = len(dataset)  # type: ignore

        # sklearn CV splitters take X, y, and group args to the .split method
        # since we cant know the structure of the data, we can just split this index tensor
        self._dataset_idx = torch.arange(self.dataset_size)

        self._dataset_y_attr = y_attr
        self._dataset_group_attr = group_attr

        self._validate_dataset()

        ###########

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

    # TODO: for sklearn compatibility, this may need to take args to .split
    # def train_val_dataloaders(
    #     self, *split_args, dataloader_kwargs: Optional[KwargType] = None, **split_kwargs
    # ) -> CVDataLoader:
    def train_val_dataloaders(self, **dataloader_kwargs) -> CVDataLoader:
        # if dataloader_kwargs is None:
        #     dataloader_kwargs = dict()

        train_kwargs, val_kwargs = self._split_train_val_kwargs(**dataloader_kwargs)

        # for train_idx, val_idx in self.data_manager.split(*split_args, **split_kwargs):
        # for train_idx, val_idx in self.data_manager.split(self._dataset_idx)
        # TODO: pass X, y, and group?

        try:
            splits = self.data_manager.split()
        except TypeError:
            # probably sklearn style splitters
            X = self._dataset_idx
            if self._dataset_y_attr is not None:
                y = getattr(self.dataset, self._dataset_y_attr, None)
            else:
                y = None

            if self._dataset_group_attr is not None:
                groups = getattr(self.dataset, self._dataset_group_attr, None)
            else:
                groups = None

            splits = self.data_manager.split(X, y, groups=groups)

        for train_idx, val_idx in splits:
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
        return tuple(torch.from_numpy(array) for array in arrays)

    def _validate_dataset(self):
        if self._dataset_y_attr is not None:
            if not hasattr(self.dataset, self._dataset_y_attr):
                raise ValueError(
                    f"Dataset does not have y attribute called {self._dataset_y_attr}. This is "
                    "used for stratified cross validation strategies in which each fold has "
                    "equal representation of the target variable."
                )

        if self._dataset_group_attr is not None:
            if not hasattr(self.dataset, self._dataset_group_attr):
                raise ValueError(
                    f"Dataset does not have group attribute {self._dataset_group_attr}. This is "
                    "used for group cross validation strategies in which each fold is validated "
                    "with non-target group labels."
                )
