import pytest
from conftest import N_DATA_POINTS, NUM_GROUPS, NUM_LABELS
from sklearn.model_selection import GroupKFold, StratifiedKFold

from lightning_cv.data import CrossValidationDataModule
from lightning_cv.typehints import KwargType

# TODO: need to test setting the group and yattrs names


# need to test a bunch of different splits
# TODO: this should probably be in the test_split suite
class TestCVSplits:
    pass


class TestCVDataModule:
    @pytest.mark.parametrize("size", [None, 15])
    def test_dataset_size(self, dataset, size):
        module = CrossValidationDataModule(
            dataset, 1, GroupKFold, {"n_splits": 5}, dataset_size=size
        )

        if size is None:
            assert module.dataset_size == N_DATA_POINTS
        else:
            assert module.dataset_size == size

    @pytest.mark.parametrize("stage", ["train", "val"])
    def test_invalid_dataloaders(
        self, datamodule: CrossValidationDataModule, stage: str
    ):
        method = getattr(datamodule, f"{stage}_dataloader")
        with pytest.raises(TypeError):
            method()

    @pytest.mark.parametrize("stage", ["test", "predict"])
    def test_invalid_stage(self, datamodule: CrossValidationDataModule, stage: str):
        with pytest.raises(ValueError):
            datamodule.setup(stage)  # type: ignore[arg-type]

    def _setup_cv_attrs_test(self, dset, attr: str, attr_name: str):
        kwargs: KwargType = {
            f"{attr}_attr": attr_name,
        }

        if attr == "group":
            kwargs["cross_validator"] = GroupKFold
            kwargs["cross_validator_config"] = {"n_splits": NUM_GROUPS}
        else:
            kwargs["cross_validator"] = StratifiedKFold
            kwargs["cross_validator_config"] = {"n_splits": NUM_LABELS}

        datamodule = CrossValidationDataModule(
            dataset=dset,
            batch_size=5,
            **kwargs,
        )

        return datamodule

    @pytest.mark.parametrize(
        ("attr", "attr_name"), [("group", "group"), ("y", "target")]
    )
    def test_valid_cv_attrs(self, dataset, attr: str, attr_name: str):
        datamodule = self._setup_cv_attrs_test(dataset, attr, attr_name)

        other_attr = "y" if attr == "group" else "group"

        assert getattr(datamodule, f"_dataset_{attr}_attr") is not None
        assert hasattr(datamodule.dataset, attr_name)
        assert getattr(datamodule, f"_dataset_{other_attr}_attr") is None

    @pytest.mark.parametrize(
        ("attr", "attr_name"), [("group", "INVALID_GROUP"), ("y", "INVALID_TARGET")]
    )
    def test_invalid_cv_attrs(self, dataset, attr: str, attr_name: str):
        with pytest.raises(ValueError):
            self._setup_cv_attrs_test(dataset, attr, attr_name)
