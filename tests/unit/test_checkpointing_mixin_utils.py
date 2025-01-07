import dataclasses
import inspect
from pathlib import Path

import attrs
import pytest

from lightning_cv import trainer as trainer_module
from lightning_cv.utils.checkpointing import (
    CheckpointLoaderMixin,
    CurrentCheckpointsMixin,
    _check_model_init_signature_for_deserialization,
    _model_init_signature,
)


@attrs.define
class _AttrsConfig:
    in_dim: int
    out_dim: int


@dataclasses.dataclass
class _DataclassConfig:
    in_dim: int
    out_dim: int


class _TestSignaturesModel:
    def __init__(
        self,
        x: int,
        y: float,
        attrs_config: _AttrsConfig,
        dataclass_config: _DataclassConfig,
        unknown,
    ):
        pass


class TestSignatures:
    def test_deserializable_from_signature(self):
        sig = _model_init_signature(_TestSignaturesModel)  # type: ignore[arg-type]
        deserializable = _check_model_init_signature_for_deserialization(sig)

        assert {x[0] for x in deserializable} == {"attrs_config", "dataclass_config"}


class TestImporting:
    def _check_module_import_works(self, module, other_module):
        module_file = inspect.getfile(module)
        other_module_file = inspect.getfile(trainer_module)

        assert module_file == other_module_file

    def test_load_module_from_filepath(self):
        file = inspect.getfile(trainer_module.CrossValidationTrainer)

        module = CheckpointLoaderMixin._try_load_module_from_filepath(
            "lightning_cv.trainer",
            file,
        )

        self._check_module_import_works(module, trainer_module)

    def test_load_module(self):
        file = inspect.getfile(trainer_module.CrossValidationTrainer)
        module = CheckpointLoaderMixin._try_load_module(
            "lightning_cv.trainer",
            file,
        )

        self._check_module_import_works(module, trainer_module)

        from importlib import import_module

        direct_module = import_module("lightning_cv.trainer")
        self._check_module_import_works(module, direct_module)


class TestCurrentCheckpoints:
    def test_latest_checkpoint(self, checkpoint_dir: Path):
        latest_ckpt = CurrentCheckpointsMixin.get_latest_checkpoint(checkpoint_dir)

        assert latest_ckpt == checkpoint_dir.joinpath("epoch=5_fold=2.ckpt")

    def test_current_checkpoints(self, checkpoint_dir: Path):
        obj = CurrentCheckpointsMixin()
        obj.checkpoint_dir = checkpoint_dir

        current_checkpoints = obj.current_ckptfiles()

        assert set(current_checkpoints) == {
            checkpoint_dir.joinpath("epoch=5_fold=0.ckpt"),
            checkpoint_dir.joinpath("epoch=5_fold=1.ckpt"),
            checkpoint_dir.joinpath("epoch=5_fold=2.ckpt"),
        }

    @pytest.mark.parametrize("mkdir", [True, False])
    def test_current_checkpoints_empty_dir(self, mkdir: bool):
        empty_dir = Path("EMPTY")
        if mkdir:
            empty_dir.mkdir(exist_ok=True)
        else:
            # else make it a file
            empty_dir.touch()

        latest_ckpt = CurrentCheckpointsMixin.get_latest_checkpoint(empty_dir)

        assert latest_ckpt is None

        if mkdir:
            empty_dir.rmdir()
        else:
            empty_dir.unlink()
