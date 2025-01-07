import optuna
import pytest

from lightning_cv.tuning.config import (
    HparamConfig,
    TunableFloat,
    TunableInt,
    TunableType,
)
from lightning_cv.tuning.hyperparameters import HparamRegistry, suggest


@pytest.fixture
def study() -> optuna.Study:
    return optuna.create_study(
        sampler=optuna.samplers.RandomSampler(seed=42),
        load_if_exists=True,
        study_name="PYTEST",
    )


@pytest.fixture
def trial(study: optuna.Study) -> optuna.Trial:
    return study.ask()


def _get_registry_from_config_dict(config) -> HparamRegistry:
    hparam_config = HparamConfig.from_dict(config)
    return HparamRegistry(hparam_config)


@pytest.fixture
def hparam_registry(config_dict) -> HparamRegistry:
    return _get_registry_from_config_dict(config_dict)


@pytest.fixture
def hparam_registry_with_forbidden(config_dict_with_forbidden) -> HparamRegistry:
    return _get_registry_from_config_dict(config_dict_with_forbidden)


def _add_param_to_trial(
    current_trial: optuna.Trial, hparam: str, value: int, tunable_type: TunableType
):
    frozen_trial = current_trial._cached_frozen_trial
    frozen_trial.params[hparam] = value

    if isinstance(tunable_type, TunableInt):
        distribution = optuna.distributions.IntDistribution
    elif isinstance(tunable_type, TunableFloat):
        distribution = optuna.distributions.FloatDistribution
    else:
        distribution = optuna.distributions.CategoricalDistribution

    distr_kwargs = tunable_type.to_optuna_kwargs()
    del distr_kwargs["name"]
    distr_instance = distribution(**distr_kwargs)

    frozen_trial.distributions[hparam] = distr_instance


def _add_params_to_trial(trial: optuna.Trial, params: dict, registry: HparamRegistry):
    for key, value in params.items():
        hparam_tunable = registry.named_config[key]
        _add_param_to_trial(trial, key, value, hparam_tunable)


class TestHparamRegistry:

    @pytest.mark.parametrize(
        "hparam",
        ["lr", "batch_size", "optimizer_type", "num_heads", "hidden_dim", "dropout"],
    )
    def test_register_single(
        self, hparam_registry: HparamRegistry, trial: optuna.Trial, hparam: str
    ):

        hparam_registry._register(
            hparam_registry.named_config[hparam], trial, resample=False
        )

        assert hparam in trial.params

    @pytest.mark.parametrize("resample", [True, False])
    def test_resampling(
        self, hparam_registry: HparamRegistry, trial: optuna.Trial, resample: bool
    ):
        hparam = "hidden_dim"  # good test since it's a categorical hparam
        hparam_tunable = hparam_registry.named_config[hparam]

        hparam_registry._register(hparam_tunable, trial, resample=False)

        first_hidden_dim = trial.params[hparam]
        hparam_registry._register(hparam_tunable, trial, resample=resample)

        second_hidden_dim = trial.params[hparam]

        if resample:
            while first_hidden_dim == second_hidden_dim:
                second_hidden_dim = hparam_registry._register(
                    hparam_tunable, trial, resample=True
                )

            assert first_hidden_dim != second_hidden_dim
        else:
            assert first_hidden_dim == second_hidden_dim

    @pytest.mark.parametrize(
        "registry",
        [
            "hparam_registry",
            "hparam_registry_with_forbidden",
        ],
    )
    @pytest.mark.parametrize(
        "forbidden",
        [
            {"lr": 1e-3, "batch_size": 32},
            {"lr": 1e-4, "batch_size": 64},
            {"num_heads": 3, "hidden_dim": 64},
            {"num_heads": 4, "hidden_dim": 128},
            {"hidden_dim": 300, "num_layers": 30, "num_heads": 4},
        ],
    )
    def test_resolve_forbidden_combinations(
        self,
        registry,
        trial: optuna.Trial,
        forbidden: dict,
        request,
    ):
        hparam_registry: HparamRegistry = request.getfixturevalue(registry)

        TOTAL_INVOLVED_HPARAMS = {
            "lr",
            "batch_size",
            "num_heads",
            "hidden_dim",
            "num_layers",
        }
        missing = TOTAL_INVOLVED_HPARAMS - forbidden.keys()

        # manually add the forbidden combinations
        _add_params_to_trial(trial, forbidden, hparam_registry)

        assert trial.params == forbidden

        # for compatibility we need to sample the missing hparams
        for missing_hparam in missing:
            hparam_tunable = hparam_registry.named_config[missing_hparam]
            hparam_registry._register(hparam_tunable, trial)

        hparam_registry._resolve_forbidden_combinations(trial)

        # but then we just delete the missing hparams we had to sample for test clarity
        new_trial_params = {
            key: value for key, value in trial.params.items() if key not in missing
        }

        if hparam_registry.config.forbidden is not None:
            assert new_trial_params != forbidden
        else:
            # no changes should be made
            assert new_trial_params == forbidden

    @pytest.mark.parametrize(
        "params",
        [
            {"lr": 1e-3, "batch_size": 32},
            {"lr": 1e-4, "batch_size": 64, "mappable": "C"},
        ],
    )
    def test_nest_params(
        self, hparam_registry: HparamRegistry, trial: optuna.Trial, params: dict
    ):
        _add_params_to_trial(trial, params, hparam_registry)

        nested_params = hparam_registry._nest_hparams(trial)

        expected = {
            "optimizer": {"lr": params["lr"]},
            "data": {"batch_size": params["batch_size"]},
        }

        if "mappable" in params:
            tunable_type = hparam_registry.named_config["mappable"]
            expected.update(tunable_type.map[params["mappable"]])  # type: ignore

        assert nested_params == expected

    def test_register(self, hparam_registry: HparamRegistry, trial: optuna.Trial):

        nested_params = hparam_registry.register_hparams(trial)

        sampled_params = set()
        stack = [nested_params]
        while stack:
            curr = stack.pop()
            for key, value in curr.items():
                if isinstance(value, dict):
                    stack.append(value)  # type: ignore
                else:
                    if key.startswith("mappable"):
                        key = "mappable"
                    sampled_params.add(key)

        assert all(key in sampled_params for key in hparam_registry.named_config.keys())

    def test_suggest(self, hparam_registry: HparamRegistry, trial: optuna.Trial):
        import random
        from copy import deepcopy

        trial1 = trial
        trial2 = deepcopy(trial1)

        random.seed(42)
        from_register = hparam_registry.register_hparams(trial1)
        random.seed(42)
        from_suggest = suggest(trial2, hparam_registry)

        assert from_register == from_suggest

    def test_from_file(self, hparam_registry: HparamRegistry, config_file):
        new_registry = HparamRegistry.from_file(config_file)

        assert hparam_registry.config == new_registry.config
        assert hparam_registry.named_config == new_registry.named_config
        assert (
            hparam_registry.uniq_forbidden_combos == new_registry.uniq_forbidden_combos
        )
