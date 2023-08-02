from __future__ import annotations

from abc import ABCMeta, abstractmethod, abstractproperty
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


class BaseCrossValidator(metaclass=ABCMeta):
    """Simpler CV iterface than sklearn"""

    def get_X_index(self, X: Int64Array) -> Int64Array:
        n = X.shape[0]
        return np.arange(n)

    @abstractmethod
    def split(self) -> CVIterator:
        ...

    @abstractmethod
    def get_n_splits(self) -> int:
        ...


class BaseGroupCrossValidator(BaseCrossValidator):
    @abstractproperty
    def train_group_ids(self) -> list[int]:
        ...

    @abstractproperty
    def val_group_id(self) -> int:
        ...


class ImbalancedLeaveOneGroupOut(BaseGroupCrossValidator):
    def __init__(self, groups: Int64Array) -> None:
        # only keep track of data indices for each fold
        self.X_idx = self.get_X_index(groups)

        self._val_group_id = -1
        self._train_group_ids: list[int] = list()

        self.uniq_groups: Int64Array
        self.groups: Int64Array
        # self.groups maps actual group labels to consecutively increasing,
        # ie [0, 1, ..., n-1, n]
        self.uniq_groups, self.groups = np.unique(groups, return_inverse=True)
        self.group_counts = np.bincount(self.groups)
        self._sort_groups()

        # don't have a fold for the most frequent group
        self.n_folds = self.uniq_groups.shape[0] - 1
        self.largest_group_id = np.argmax(self.group_counts)

    @property
    def train_group_ids(self) -> list[int]:
        return self._train_group_ids

    @train_group_ids.setter
    def train_group_ids(self, value: list[int]):
        self._train_group_ids = value

    @property
    def val_group_id(self) -> int:
        return self._val_group_id

    @val_group_id.setter
    def val_group_id(self, value: int):
        self._val_group_id = value

    def _sort_groups(self):
        # reverse sort, ie 0th element is count of most frequent group
        sort_idx = np.argsort(self.group_counts)[::-1]
        self.uniq_groups = self.uniq_groups[sort_idx]
        self.group_counts = self.group_counts[sort_idx]

    def __len__(self) -> int:
        return self.n_folds

    def get_n_splits(self) -> int:
        return self.n_folds

    def split(self) -> CVIterator:
        for fold_idx, group_id in enumerate(self.uniq_groups):
            if fold_idx == 0:
                # the first group will be the most frequent and
                # we don't want a fold where the most frequent group
                # is the validation set
                continue

            # keep track of current CV group id and the training groups
            self.val_group_id = group_id
            self.train_group_ids = self.uniq_groups[
                np.where(self.uniq_groups != group_id)
            ].tolist()

            val_mask = np.where(self.groups == group_id, True, False)
            train_mask = ~val_mask

            train_idx = self.X_idx[train_mask]
            val_idx = self.X_idx[val_mask]

            yield train_idx, val_idx
