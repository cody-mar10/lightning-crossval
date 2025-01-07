import itertools
import sys
import time
from contextlib import contextmanager
from pathlib import Path
from shutil import rmtree

import attrs
import lightning
import pytest
import torch
from sklearn.model_selection import GroupKFold
from torch.utils.data import Dataset, DataLoader

from lightning_cv.callbacks.base import Callback
from lightning_cv.config import CrossValidationTrainerConfig
from lightning_cv.data import CrossValidationDataModule
from lightning_cv.trainer import CrossValidationTrainer

sys.path.insert(0, __file__)

PairTensor = tuple[torch.Tensor, torch.Tensor]
IN_DIM = 10
N_DATA_POINTS = 30
NUM_GROUPS = 3
NUM_LABELS = 2
BATCH_SIZE = 5


@attrs.define
class DummyModelConfig:
    in_dim: int
    slow_val: bool = False


class DummyModel(lightning.LightningModule):
    LR = 1.23e-3
    PLATEAU_AT = 5  # after 5 epochs, val loss will not improve

    def __init__(self, config: DummyModelConfig):
        super().__init__()
        self.config = config
        self.predictor = torch.nn.Linear(self.config.in_dim, 1)
        self._cached_val_loss = torch.tensor(float("inf"))
        self.current_tracked_epoch = 0

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.predictor(x)

    # should test a scheduler
    def configure_optimizers(self) -> torch.optim.Optimizer:
        return torch.optim.AdamW(params=self.parameters(), lr=self.LR)

    def _log_loss(self, loss: torch.Tensor, stage: str):
        # let lightning.Fabric handle logging loss otherwise
        if self.fabric is None:
            self.log(
                f"{stage}_loss",
                loss,
                on_step=True,
                on_epoch=True,
                prog_bar=True,
            )

    def _eval_step(self, batch: PairTensor):
        x, y = batch
        y_hat = self(x).squeeze()
        loss = torch.nn.functional.binary_cross_entropy_with_logits(y_hat, y.float())

        return loss

    def training_step(self, batch: PairTensor, batch_idx: int):
        loss = self._eval_step(batch)
        self._log_loss(loss, "train")
        return loss

    def validation_step(self, batch: PairTensor, batch_idx: int):
        if self.current_tracked_epoch >= self.PLATEAU_AT:
            loss = self._cached_val_loss
        else:
            loss = self._cached_val_loss = self._eval_step(batch)
        self._log_loss(loss, "val")

        if self.config.slow_val:
            # slow val to stop trainer at val step
            time.sleep(0.1)

        return loss


class ModelEpochTracker(Callback):
    def on_train_fold_end(self, trainer: CrossValidationTrainer):
        model = trainer.fold_manager[trainer.current_fold].model

        if isinstance(model, lightning.LightningModule):
            model.current_tracked_epoch += 1
        else:
            model.module.current_tracked_epoch += 1


class DummyDataset(Dataset):
    def __init__(self, n: int, in_dim: int):
        self.data = torch.randn(n, in_dim)
        self.target = self._labels(n, NUM_LABELS)  # binary target
        self.group = self._labels(n, NUM_GROUPS)  # with 3 groups

    def __len__(self):
        return self.data.shape[0]

    def __getitem__(self, idx) -> PairTensor:
        return self.data[idx], self.target[idx]

    def _cycle(self, n: int):
        yield from itertools.cycle(range(n))

    def _labels(self, n: int, num_labels: int) -> torch.Tensor:
        return torch.tensor(list(itertools.islice(self._cycle(num_labels), n)))


@pytest.fixture
def model_config() -> DummyModelConfig:
    return DummyModelConfig(in_dim=IN_DIM)


@pytest.fixture
def slow_model_config() -> DummyModelConfig:
    return DummyModelConfig(in_dim=IN_DIM, slow_val=True)


@pytest.fixture
def model(model_config: DummyModelConfig) -> DummyModel:
    return DummyModel(model_config)


@pytest.fixture
def slow_model(slow_model_config: DummyModelConfig) -> DummyModel:
    return DummyModel(slow_model_config)


@pytest.fixture
def dataset() -> DummyDataset:
    return DummyDataset(N_DATA_POINTS, IN_DIM)

@pytest.fixture
def dataloader(dataset: DummyDataset) -> DataLoader:
    return DataLoader(dataset, batch_size=BATCH_SIZE)

@pytest.fixture
def datamodule(dataset: DummyDataset) -> CrossValidationDataModule:
    return CrossValidationDataModule(
        dataset,
        batch_size=BATCH_SIZE,
        cross_validator=GroupKFold,
        cross_validator_config={"n_splits": 3},
        group_attr="group",
        shuffle=False,
    )


# TODO: test with different cross validators?
@pytest.fixture
def datamodule_groupkfold(
    datamodule: CrossValidationDataModule,
) -> CrossValidationDataModule:
    return datamodule


@contextmanager
def _checkpoint_cleanup(trainer: CrossValidationTrainer):
    ckptdir = trainer.config.checkpoint_dir
    try:
        yield trainer
    finally:
        if ckptdir.exists():
            rmtree(ckptdir)


def _with_cleanup(trainer: CrossValidationTrainer):
    with _checkpoint_cleanup(trainer) as cv_trainer:
        yield cv_trainer


@pytest.fixture
def cv_trainer_config() -> CrossValidationTrainerConfig:
    return CrossValidationTrainerConfig(
        max_epochs=2,
        checkpoint_dir="PYTEST_CHECKPOINT_DIR",
        callbacks=[ModelEpochTracker()],
    )


@pytest.fixture
def cv_trainer(
    cv_trainer_config: CrossValidationTrainerConfig,
) -> CrossValidationTrainer:
    return CrossValidationTrainer(model_type=DummyModel, config=cv_trainer_config)


@pytest.fixture
def cv_trainer_with_cleanup(cv_trainer: CrossValidationTrainer):
    yield from _with_cleanup(cv_trainer)


@pytest.fixture
def preset_cv_trainer(
    cv_trainer: CrossValidationTrainer,
    datamodule: CrossValidationDataModule,
    model_config: DummyModelConfig,
) -> CrossValidationTrainer:
    cv_trainer.setup(datamodule, model_config)
    return cv_trainer


@pytest.fixture
def preset_cv_trainer_with_cleanup(preset_cv_trainer: CrossValidationTrainer):
    yield from _with_cleanup(preset_cv_trainer)


@pytest.fixture
def long_cv_trainer() -> CrossValidationTrainer:
    config = CrossValidationTrainerConfig(
        max_epochs=10,
        checkpoint_dir="PYTEST_CHECKPOINT_DIR",
        callbacks=[ModelEpochTracker()],
    )
    return CrossValidationTrainer(model_type=DummyModel, config=config)


@pytest.fixture
def long_cv_trainer_with_cleanup(long_cv_trainer: CrossValidationTrainer):
    yield from _with_cleanup(long_cv_trainer)


@pytest.fixture
def rank_zero_cv_trainer(cv_trainer: CrossValidationTrainer):
    return cv_trainer


@pytest.fixture
def rank_zero_cv_trainer_with_cleanup(rank_zero_cv_trainer: CrossValidationTrainer):
    yield from _with_cleanup(rank_zero_cv_trainer)


@pytest.fixture
def non_rank_zero_cv_trainer(cv_trainer_config: CrossValidationTrainerConfig):
    class _NonRankZeroTrainer(CrossValidationTrainer):
        @property
        def is_global_zero(self):
            return False

    return _NonRankZeroTrainer(model_type=DummyModel, config=cv_trainer_config)


@pytest.fixture
def non_rank_zero_cv_trainer_with_cleanup(non_rank_zero_cv_trainer):
    yield from _with_cleanup(non_rank_zero_cv_trainer)


def _add_callback(trainer: CrossValidationTrainer, *callbacks: Callback):
    for callback in callbacks:
        trainer.config.callbacks.append(callback)  # type: ignore[arg-type]
        trainer.fabric._callbacks.append(callback)


@pytest.fixture
def checkpoint_dir(
    preset_cv_trainer_with_cleanup: CrossValidationTrainer, model: DummyModel
):
    ckptdir = preset_cv_trainer_with_cleanup.checkpoint_dir
    ckptdir.mkdir(parents=True, exist_ok=True)

    preset_cv_trainer_with_cleanup.current_epoch = 5
    preset_cv_trainer_with_cleanup.global_step_per_fold = {0: 22, 1: 24, 2: 28}

    for fold_idx, fold in preset_cv_trainer_with_cleanup.fold_manager.items():
        weights = torch.full_like(
            fold.model.predictor.weight, float(fold_idx), requires_grad=True
        )

        fold.model.predictor.weight = torch.nn.Parameter(weights)

    # since there are multiple conftest configs, we need the `conftest`
    # in sys.modules to be this one for this test to work when running
    # tests from the project root
    import importlib.util
    import sys

    spec = importlib.util.spec_from_file_location("conftest", __file__)
    module = importlib.util.module_from_spec(spec) # type: ignore
    sys.modules["conftest"] = module
    spec.loader.exec_module(module) # type: ignore

    preset_cv_trainer_with_cleanup.save(preset_cv_trainer_with_cleanup.fold_manager)

    return ckptdir

    # yield ckptdir

    # if ckptdir.exists(): # guard incase already cleaned up
    #     rmtree(preset_cv_trainer.config.checkpoint_dir)


@pytest.fixture
def checkpoint_file(checkpoint_dir: Path) -> Path:
    ckpt = checkpoint_dir.joinpath("epoch=5_fold=2.ckpt")
    assert ckpt.exists()
    return ckpt
