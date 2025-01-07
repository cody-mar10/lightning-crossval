from abc import ABC, ABCMeta, abstractmethod
from copy import copy
from typing import Optional

from lightning import LightningDataModule
from torch import arange
from torch.utils.data import DataLoader, Dataset, Subset

from lightning_cv.typehints import KwargType
from lightning_cv.typehints.data import CVDataLoader, Stage
from lightning_cv.typehints.split import CrossValidator


class BaseCVDataModule(ABC, metaclass=ABCMeta):
    @abstractmethod
    def __init__(self, *args, **kwargs) -> None: ...

    @abstractmethod
    def train_val_dataloaders(self, **dataloader_kwargs) -> CVDataLoader: ...

    @abstractmethod
    def setup(self, stage: Stage) -> None: ...


class CrossValidationDataModuleMixin(BaseCVDataModule):
    def __init__(
        self,
        dataset: Dataset,
        batch_size: int,
        cross_validator: type[CrossValidator],
        cross_validator_config: Optional[KwargType] = None,
        dataset_size: Optional[int] = None,
        dataloader_type: type[DataLoader] = DataLoader,
        # attr names
        y_attr: Optional[str] = None,
        group_attr: Optional[str] = None,
        **dataloader_kwargs,
    ) -> None:
        self.dataset = dataset

        # this is customizable if dataset can customize what an individual unit of data is

        if dataset_size is None:
            dataset_size = len(dataset)  # type: ignore

        self.dataset_size = dataset_size

        self.dataloader_type = dataloader_type

        # sklearn CV splitters take X, y, and group args to the .split method
        # since we cant know the structure of the data, we can just split this index tensor
        self._dataset_idx = arange(self.dataset_size)

        self._dataset_y_attr = y_attr
        self._dataset_group_attr = group_attr

        self._validate_dataset()

        self.batch_size = batch_size
        self.dataloader_kwargs = dataloader_kwargs
        self._cross_validator = cross_validator
        self._cross_validator_config = (
            cross_validator_config if cross_validator_config is not None else {}
        )
        self.data_manager = self._cross_validator(**self._cross_validator_config)

    def _overwrite_dataloader_kwargs(self, **kwargs) -> KwargType:
        # overwrite the shared/global dataloader kwargs with kwargs specifically input
        # and then return a copy
        return self.dataloader_kwargs | kwargs

    def _split_train_val_kwargs(self, **kwargs) -> tuple[KwargType, KwargType]:
        train_kwargs = self._overwrite_dataloader_kwargs(**kwargs)
        val_kwargs = copy(train_kwargs)
        val_kwargs["shuffle"] = False
        return train_kwargs, val_kwargs

    def train_val_dataloaders(self, **dataloader_kwargs) -> CVDataLoader:
        train_kwargs, val_kwargs = self._split_train_val_kwargs(**dataloader_kwargs)

        try:
            splits = self.data_manager.split()
        except TypeError:
            # probably sklearn style splitters that sometimes take arguments
            X = self._dataset_idx
            if self._dataset_y_attr is not None:  # pragma: no cover <- already tested
                y = getattr(self.dataset, self._dataset_y_attr, None)
            else:
                y = None

            if self._dataset_group_attr is not None:
                groups = getattr(self.dataset, self._dataset_group_attr, None)
            else:  # pragma: no cover <- already tested
                groups = None

            splits = self.data_manager.split(X, y, groups=groups)

        for train_idx, val_idx in splits:
            train_subset = Subset(self.dataset, train_idx.tolist())
            val_subset = Subset(self.dataset, val_idx.tolist())

            train_loader = self.dataloader_type(
                dataset=train_subset,
                batch_size=self.batch_size,
                **train_kwargs,
            )
            val_loader = self.dataloader_type(
                dataset=val_subset,
                batch_size=self.batch_size,
                **val_kwargs,
            )

            yield train_loader, val_loader

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

    def train_dataloader(self, **dataloader_kwargs):
        raise TypeError(
            "A cross validation datamodule shouldn't need a training dataloader"
        )

    def val_dataloader(self, **dataloader_kwargs):
        raise TypeError(
            "A cross validation datamodule shouldn't need a validation dataloader"
        )


class CrossValidationDataModule(CrossValidationDataModuleMixin, LightningDataModule):
    def __init__(
        self,
        dataset: Dataset,
        batch_size: int,
        cross_validator: type[CrossValidator],
        cross_validator_config: KwargType,
        dataset_size: Optional[int] = None,
        dataloader_type: type[DataLoader] = DataLoader,
        # attr names
        y_attr: Optional[str] = None,
        group_attr: Optional[str] = None,
        **dataloader_kwargs,
    ) -> None:
        LightningDataModule.__init__(self)
        CrossValidationDataModuleMixin.__init__(
            self=self,
            dataset=dataset,
            batch_size=batch_size,
            cross_validator=cross_validator,
            cross_validator_config=cross_validator_config,
            dataset_size=dataset_size,
            dataloader_type=dataloader_type,
            y_attr=y_attr,
            group_attr=group_attr,
            **dataloader_kwargs,
        )

    def setup(self, stage: Stage) -> None:
        if stage != "fit":
            raise ValueError(
                f"Stage {stage} is not supported cross validation datamodules"
            )


CV_DATAMODULE_METHOD_NAME = "train_val_dataloaders"


def is_cross_val_datamodule(obj) -> bool:
    return isinstance(obj, LightningDataModule) and hasattr(
        obj, CV_DATAMODULE_METHOD_NAME
    )
