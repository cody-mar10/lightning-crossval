from typing import Iterable

import pytest
import torch
from conftest import PairTensor
from sklearn.model_selection import GroupKFold, StratifiedKFold
from torch.utils.data import DataLoader, Dataset, Subset

from lightning_cv.data import CrossValidationDataModule
from lightning_cv.split import BaseCrossValidator, ImbalancedLeaveOneGroupOut

N_DATA_POINTS = 12
DIM = 10


def simple_tensor(indices: Iterable[int], n_dims: int) -> torch.Tensor:
    return torch.tensor([[float(i)] * n_dims for i in indices])


class SimpleDataset(Dataset):
    N_DIMS = 3
    N_POINTS = 6
    GROUP_SIZE = 2

    def __init__(self):
        self.x = simple_tensor(range(self.N_POINTS), self.N_DIMS)

        # 60:40 ratio of labels
        # when using StratifiedKFold, the train and val batch will
        # have 2 False and 1 True
        # the order is guaranteed to be like this:
        # label:  [1, 0, 0]
        # fold 0: [3, 4, 5]
        # -----------------
        # label:  [0, 1, 0]
        # fold 1: [0, 1, 2]
        self.label = torch.tensor([0, 1, 0, 1, 0, 0])

        # 2 points per group <- this is how GroupKFold is implemented
        # fold 0: train = [0, 1, 2, 3] val = [4, 5]
        # fold 1: train = [0, 1, 4, 5] val = [2, 3]
        # fold 2: train = [2, 3, 4, 5] val = [0, 1]
        self.group = torch.tensor([0, 0, 1, 1, 2, 2])

    def __len__(self):
        return self.N_POINTS

    def __getitem__(self, idx) -> PairTensor:
        return self.x[idx], self.label[idx]


@pytest.fixture
def simple_dataset() -> SimpleDataset:
    return SimpleDataset()


@pytest.fixture
def data() -> torch.Tensor:
    return torch.randn(N_DATA_POINTS, DIM)


@pytest.fixture
def groups() -> torch.Tensor:
    return torch.tensor([0, 0, 0, 0, 0, 0, 1, 1, 2, 2, 3, 3])


class CustomCV(BaseCrossValidator):
    def __init__(self, n_splits: int = 5):
        self.n_splits = n_splits

    def split(self):
        for i in range(self.n_splits):
            yield torch.arange(i), torch.arange(i)

    def get_n_splits(self) -> int:
        return self.n_splits


def test_get_X_index(data):
    cv = CustomCV()
    x_idx = cv.get_X_index(data)
    assert x_idx.shape[0] == N_DATA_POINTS
    assert x_idx.tolist() == list(range(N_DATA_POINTS))


def test_imbalanced_leave_one_group_out(groups):
    cv = ImbalancedLeaveOneGroupOut(groups)
    assert len(cv) == 3
    assert cv.get_n_splits() == 3
    assert cv.train_group_ids == []
    assert cv.val_group_id == -1

    for train_idx, val_idx in cv.split():
        assert train_idx.shape[0] == N_DATA_POINTS - 2
        assert val_idx.shape[0] == 2
        assert 0 in cv.train_group_ids
        assert len(cv.train_group_ids) == 3
        assert cv.val_group_id not in cv.train_group_ids


class TestTrainValDataloaders:
    BATCH_SIZE = 128  # this is big enough to fit all of the data into a single batch

    def _setup_datamodule(
        self, dset: SimpleDataset, cv, cv_config: dict, attr: str, attr_name: str
    ):
        datamodule = CrossValidationDataModule(
            dataset=dset,
            batch_size=self.BATCH_SIZE,
            cross_validator=cv,
            cross_validator_config=cv_config,
            **{f"{attr}_attr": attr_name},  # type: ignore
        )

        datamodule.setup(
            "fit"
        )  # this technically does nothing, but would be called by lightning.Trainer

        return datamodule

    def _test_individual_dataloaders(
        self, dataloader: DataLoader[SimpleDataset], expected_indices: list[int]
    ):
        subset: Subset[SimpleDataset] = dataloader.dataset  # type: ignore

        assert subset.indices == expected_indices

        x: torch.Tensor
        x, _ = next(iter(dataloader))

        assert x.shape <= (self.BATCH_SIZE, SimpleDataset.N_DIMS)

        expected_x = simple_tensor(expected_indices, SimpleDataset.N_DIMS)

        assert torch.all(x == expected_x)

    def _test_cv_splits(
        self,
        dm: CrossValidationDataModule,
        expected_num_folds: int,
        expected_indices: dict[int, tuple[list[int], list[int]]],
    ):
        for fold_idx, (train_loader, val_loader) in enumerate(
            dm.train_val_dataloaders(shuffle=False)
        ):
            train_idx, val_idx = expected_indices[fold_idx]

            self._test_individual_dataloaders(train_loader, train_idx)
            self._test_individual_dataloaders(val_loader, val_idx)

        assert fold_idx == expected_num_folds - 1

    def test_group_cv(self, simple_dataset: SimpleDataset):
        datamodule = self._setup_datamodule(
            simple_dataset,
            GroupKFold,
            {"n_splits": 3},
            "group",
            "group",
        )

        # group k fold will go in order of groups from largest group label to smallest
        expected_group_indices = {
            0: ([0, 1, 2, 3], [4, 5]),
            1: ([0, 1, 4, 5], [2, 3]),
            2: ([2, 3, 4, 5], [0, 1]),
        }

        self._test_cv_splits(datamodule, 3, expected_group_indices)

    def test_target_cv(self, simple_dataset: SimpleDataset):
        datamodule = CrossValidationDataModule(
            dataset=simple_dataset,
            batch_size=self.BATCH_SIZE,
            cross_validator=StratifiedKFold,
            cross_validator_config={"n_splits": 2},
            y_attr="label",
        )

        expected_y_indices = {
            1: ([0, 1, 2], [3, 4, 5]),
            0: ([3, 4, 5], [0, 1, 2]),
        }

        self._test_cv_splits(datamodule, 2, expected_y_indices)
