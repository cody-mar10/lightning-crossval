import cattrs
import pytest
from conftest import RawConfig

from lightning_cv.tuning.config import (  # noqa
    HparamConfig,
    MappingT,
    TunableCategorical,
    TunableFloat,
    TunableInt,
    TunableType,
    get_mappings,
    tomli,
)
from lightning_cv.tuning.utils import FrozenDict


class TestWriteToml:
    @pytest.mark.parametrize("file", ["config_file", "config_file_with_forbidden"])
    def test_written_file_is_valid_toml(self, file: str, request):
        file = request.getfixturevalue(file)
        with open(file, "rb") as fp:
            tomli.load(fp)

    # loading written file into attrs class is tested in
    # TestHparamConfig.test_load_from_written_file


def _get_tunable_type(name: str) -> type[TunableType]:
    if name == "int":
        return TunableInt
    if name == "float":
        return TunableFloat
    if name == "categorical":
        return TunableCategorical

    raise ValueError("Tunable type not supported: {tunable_type}")


@pytest.mark.parametrize("tunable_type", ["int", "float", "categorical"])
def test_tunable_type(config_dict: RawConfig, tunable_type: str):
    tunable_cls = _get_tunable_type(tunable_type)

    for hparam in config_dict["hparams"]:
        if hparam["suggest"] == tunable_type:
            tunable_cls.from_dict(hparam)
        else:
            with pytest.raises(cattrs.ClassValidationError):
                tunable_cls.from_dict(hparam)


class TestHparamConfig:
    def test_from_dict(self, config_dict: RawConfig):
        hparam_config = HparamConfig.from_dict(config_dict)  # type: ignore[arg-type]

        assert len(hparam_config.hparams) == len(config_dict["hparams"])
        assert hparam_config.forbidden is None

        # HparamConfig includes default values in the hparam config that are not present in the
        # config dictm so can't just do dict comparison. Just need to check that the values are
        # the same from the config dict

        same: list[bool] = []
        for actual_hparam, expected_hparam in zip(
            hparam_config.hparams, config_dict["hparams"]
        ):
            for key, input_value in expected_hparam.items():
                same.append(input_value == getattr(actual_hparam, key))

        assert all(same)

    def test_from_dict_with_forbidden(self, config_dict_with_forbidden: RawConfig):
        hparam_config = HparamConfig.from_dict(config_dict_with_forbidden)  # type: ignore[arg-type]

        assert hparam_config.forbidden is not None
        assert len(hparam_config.forbidden) == len(config_dict_with_forbidden["forbidden"])  # type: ignore[arg-type]
        assert all(isinstance(combo, FrozenDict) for combo in hparam_config.forbidden)

    @pytest.mark.parametrize(
        ("file", "expected"),
        [
            ("config_file", "config_dict"),
            ("config_file_with_forbidden", "config_dict_with_forbidden"),
        ],
    )
    def test_from_file(self, file: str, expected: RawConfig, request):
        file = request.getfixturevalue(file)
        expected = request.getfixturevalue(expected)

        from_file = HparamConfig.from_file(file)
        from_dict = HparamConfig.from_dict(expected)  # type: ignore[arg-type]

        assert from_file == from_dict

    def _get_expected_pytest_context(self, strict: bool, from_file_or_dict: bool):
        if strict:
            if from_file_or_dict:
                return pytest.raises(cattrs.ClassValidationError)

            return pytest.raises(ValueError)
        return pytest.warns(RuntimeWarning)

    @pytest.mark.parametrize("strict", [True, False])
    def test_from_invalid_file(self, invalid_config_file: str, strict: bool):
        ctx = self._get_expected_pytest_context(strict, from_file_or_dict=True)
        with ctx:
            HparamConfig.from_file(invalid_config_file, strict=strict)

    @pytest.mark.parametrize("strict", [True, False])
    def test_from_invalid_dict(self, invalid_config_dict: RawConfig, strict: bool):
        ctx = self._get_expected_pytest_context(strict, from_file_or_dict=True)

        invalid_config_dict["strict"] = strict  # type: ignore
        with ctx:
            HparamConfig.from_dict(invalid_config_dict)  # type: ignore[arg-type]

    def test_mappings(self, config_dict: RawConfig):
        hparam_config = HparamConfig.from_dict(config_dict)  # type: ignore[arg-type]

        mappings = hparam_config.mappings()
        assert mappings == {
            "mappable": {
                "A": {"mappable_sub0": 0, "mappable_sub1": 1},
                "B": {"mappable_sub0": 2, "mappable_sub1": 3},
                "C": {"mappable_sub0": 4, "mappable_sub1": 5},
            }
        }

    def test_hparams_dict(self, config_dict: RawConfig):
        hparam_config = HparamConfig.from_dict(config_dict)  # type: ignore[arg-type]

        hparams_dict = hparam_config.hparams_dict()

        assert len(hparams_dict) == len(hparam_config.hparams)

        assert all(
            expected == actual
            for expected, actual in zip(hparam_config.hparams, hparams_dict.values())
        )

        assert all(name == hparam.name for name, hparam in hparams_dict.items())

    @pytest.mark.parametrize(
        "config_dict_",
        ["config_dict", "config_dict_with_forbidden"],
    )
    def test_unique_forbidden_combos(self, config_dict_, request):
        config_dict_ = request.getfixturevalue(config_dict_)

        hparam_config = HparamConfig.from_dict(config_dict_)  # type: ignore[arg-type]

        unique_forbidden_combos = hparam_config.unique_forbidden_combos()

        if hparam_config.forbidden is None:
            assert unique_forbidden_combos is None
        else:
            # need to sort the combos since the order of the keys in the dict is not guaranteed
            # then do set comparison

            unique_forbidden_combos = set(tuple(sorted(combo)) for combo in unique_forbidden_combos)  # type: ignore[arg-type]

            assert unique_forbidden_combos == {
                ("batch_size", "lr"),
                ("hidden_dim", "num_heads"),
                ("hidden_dim", "num_heads", "num_layers"),
            }

    def test_get_mappings_from_file(self, config_file: str):
        mappings = get_mappings(config_file)

        assert mappings == {
            "mappable": {
                "A": {"mappable_sub0": 0, "mappable_sub1": 1},
                "B": {"mappable_sub0": 2, "mappable_sub1": 3},
                "C": {"mappable_sub0": 4, "mappable_sub1": 5},
            }
        }
