from typing import Union

import pytest
import torch

from lightning_cv.typehints import MetricType
from lightning_cv.utils.foldutils import (
    convert_output_to_dict,
    flatten_train_val_metrics,
    fold_iadd,
    fold_idiv,
    output_getter,
    prepend_stage_to_metrics,
    validation_update,
)

NUM_FOLDS = 4


@pytest.fixture
def per_fold_tensor():
    # [0.0, 1.0, 2.0, 3.0]
    return torch.arange(NUM_FOLDS, dtype=torch.float32)


_test_folds = pytest.mark.parametrize("fold", list(range(NUM_FOLDS)))


class TestFoldIdiv:

    @_test_folds
    @pytest.mark.parametrize(
        "value", list(map(lambda x: float(x) + 1.0, range(NUM_FOLDS)))
    )
    def test_without_zero(self, per_fold_tensor: torch.Tensor, value: float, fold: int):
        fold_idiv(per_fold_tensor, value, fold)
        assert per_fold_tensor[fold] == float(fold) / value

    def test_divide_by_zero(self, per_fold_tensor: torch.Tensor):
        with pytest.raises(AssertionError):
            fold_idiv(per_fold_tensor, 0.0, 0)


@_test_folds
def test_fold_iadd(per_fold_tensor: torch.Tensor, fold: int):
    VALUE = 5.0
    fold_iadd(per_fold_tensor, VALUE, fold)
    assert per_fold_tensor[fold] == float(fold) + VALUE


_test_output_types = pytest.mark.parametrize(
    "output", [torch.tensor(1.0), {"loss": torch.tensor(1.0)}]
)


@pytest.fixture
def tensor_output() -> torch.Tensor:
    return torch.tensor(1.0)


@pytest.fixture
def dict_output(tensor_output: torch.Tensor) -> MetricType:
    return {"loss": tensor_output}


class TestOutputGetter:
    def test_tensor_output(self, tensor_output: torch.Tensor):
        assert output_getter(tensor_output) is tensor_output

    def test_dict_output(self, dict_output: MetricType):
        assert output_getter(dict_output, name="loss") is dict_output["loss"]

    def test_dict_output_with_invalid_key(self, dict_output: MetricType):
        with pytest.raises(KeyError):
            output_getter(dict_output, name="metric")


class TestConvertOutputToDict:
    def test_convert_tensor(self, tensor_output: torch.Tensor):
        expected = {"loss": tensor_output}
        converted = convert_output_to_dict(tensor_output)
        assert converted == expected
        assert converted["loss"] is tensor_output

    def test_convert_dict(self, dict_output: MetricType):
        converted = convert_output_to_dict(dict_output)
        assert converted == dict_output
        assert converted is dict_output


_test_stages = pytest.mark.parametrize("stage", ["train", "val"])


class TestPrependStageToMetrics:
    @_test_stages
    def test_already_prepended(self, stage):
        metrics = {
            f"{stage}_loss": torch.tensor(0.24),
            f"{stage}_accuracy": torch.tensor(0.84),
        }

        assert prepend_stage_to_metrics(stage, metrics) == metrics

    @_test_stages
    def test_none_prepended(self, stage):
        metrics = {
            "loss": torch.tensor(0.24),
            "accuracy": torch.tensor(0.84),
        }

        expected = {
            f"{stage}_loss": metrics["loss"],
            f"{stage}_accuracy": metrics["accuracy"],
        }

        output = prepend_stage_to_metrics(stage, metrics)

        assert output == expected
        assert all(key.startswith(stage) for key in output.keys())

    @_test_stages
    def test_some_prepended(self, stage):
        metrics = {
            "loss": torch.tensor(0.24),
            f"{stage}_accuracy": torch.tensor(0.84),
        }

        expected = {
            f"{stage}_loss": metrics["loss"],
            f"{stage}_accuracy": metrics[f"{stage}_accuracy"],
        }

        output = prepend_stage_to_metrics(stage, metrics)

        assert output == expected
        assert all(key.startswith(stage) for key in output.keys())


_test_prepend_stage = pytest.mark.parametrize("prepend_stage", [True, False])


class TestFlattenTrainValMetrics:

    @_test_prepend_stage
    def test_dict_metrics_without_val(
        self, dict_output: MetricType, prepend_stage: bool
    ):
        expected = {
            "train_loss" if prepend_stage else "loss": dict_output["loss"],
        }

        output = flatten_train_val_metrics(dict_output, prepend_stage=prepend_stage)

        assert output == expected
        assert output is not dict_output

    def test_dict_metrics_with_val(self, dict_output: MetricType):
        val_output = {"loss": torch.tensor(0.24)}
        expected = {
            "train_loss": dict_output["loss"],
            "val_loss": val_output["loss"],
        }

        output = flatten_train_val_metrics(dict_output, val_output)

        assert output == expected
        assert output is not dict_output
        assert output is not val_output

    @_test_prepend_stage
    def test_tensor_metrics_without_val(
        self, tensor_output: torch.Tensor, prepend_stage: bool
    ):
        expected = {"train_loss" if prepend_stage else "loss": tensor_output}

        output = flatten_train_val_metrics(tensor_output, prepend_stage=prepend_stage)

        assert output == expected

    def test_tensor_metrics_with_val(self, tensor_output: torch.Tensor):
        val_output = torch.tensor(0.24)
        expected = {
            "train_loss": tensor_output,
            "val_loss": val_output,
        }

        output = flatten_train_val_metrics(tensor_output, val_output)

        assert output == expected


class TestValidationUpdate:
    LOSS = 0.22
    ACCURACY = 0.84
    DEFAULT_FOLD = 1

    @pytest.fixture
    def per_fold_metric(self) -> torch.Tensor:
        return torch.zeros(NUM_FOLDS)

    @pytest.fixture
    def per_fold_loss(self, per_fold_metric) -> MetricType:
        return {"loss": per_fold_metric}

    @pytest.fixture
    def per_fold_metrics(self, per_fold_metric) -> MetricType:
        return {
            "loss": per_fold_metric,
            "accuracy": per_fold_metric.clone(),
        }

    @pytest.fixture
    def val_step_output(self) -> MetricType:
        return {
            "loss": torch.tensor(self.LOSS),
            "accuracy": torch.tensor(self.ACCURACY),
        }

    def _check_value_equality(
        self,
        current_metrics: MetricType,
        key: str,
        folds: list[bool],
        times_ran: int = 1,
    ):

        metric = self.LOSS if key == "loss" else self.ACCURACY

        per_fold_metric = current_metrics[key]
        expected = torch.tensor([metric * times_ran if fold else 0.0 for fold in folds])

        assert torch.allclose(per_fold_metric, expected)

    def _test_equality(
        self,
        current_metrics: MetricType,
        val_step_output: MetricType,
        folds: Union[int, list[int]],
        times_ran: int = 1,
    ):
        if isinstance(folds, int):
            folds = [folds]

        folds_updated = [fold_idx in folds for fold_idx in range(NUM_FOLDS)]

        for key in val_step_output.keys():
            self._check_value_equality(current_metrics, key, folds_updated, times_ran)

    def test_new_metric(self, per_fold_loss: MetricType, val_step_output: MetricType):
        assert "accuracy" not in per_fold_loss

        validation_update(per_fold_loss, val_step_output, self.DEFAULT_FOLD)

        assert "accuracy" in per_fold_loss

        self._test_equality(per_fold_loss, val_step_output, self.DEFAULT_FOLD)

    def test_existing_metric(
        self, per_fold_metrics: MetricType, val_step_output: MetricType
    ):
        validation_update(per_fold_metrics, val_step_output, self.DEFAULT_FOLD)

        self._test_equality(per_fold_metrics, val_step_output, self.DEFAULT_FOLD)

    def test_multiple_steps(
        self, per_fold_metrics: MetricType, val_step_output: MetricType
    ):
        num_batches = 10
        for _ in range(num_batches):
            validation_update(per_fold_metrics, val_step_output, self.DEFAULT_FOLD)

        self._test_equality(
            per_fold_metrics, val_step_output, self.DEFAULT_FOLD, times_ran=num_batches
        )

    def test_multiple_folds(
        self, per_fold_metrics: MetricType, val_step_output: MetricType
    ):

        folds = [0, 2]
        for fold in folds:
            validation_update(per_fold_metrics, val_step_output, fold)

        self._test_equality(per_fold_metrics, val_step_output, folds)
