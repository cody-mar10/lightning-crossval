import pytest

from lightning_cv.tuning import utils
from lightning_cv.typehints import KwargType


@pytest.fixture
def frozen_dict() -> utils.FrozenDict:
    return utils.FrozenDict({"a": 1, "b": 2})


@pytest.fixture
def trial_updater() -> utils._TrialUpdater:
    return utils._TrialUpdater(
        model={"a": 1, "b": 2, "sub": {"c": 3, "d": 4}},
        data={"e": 5, "f": 6},
    )


class TestFrozenDict:
    def test_hashable(self, frozen_dict: utils.FrozenDict):
        storage = set()
        storage.add(frozen_dict)

    def test_immutable(self, frozen_dict: utils.FrozenDict):
        with pytest.raises(TypeError):
            frozen_dict["a"] = 3  # type: ignore[expected-error]
            frozen_dict["c"] = 4  # type: ignore[expected-error]

    def test_hash(self, frozen_dict: utils.FrozenDict):
        assert frozen_dict._hash is None
        h = hash(frozen_dict)

        assert h == frozen_dict._hash

    def test_hash_fn(self, frozen_dict: utils.FrozenDict):
        assert frozen_dict._hash is None

        h = 0
        for pair in frozen_dict.items():
            h ^= hash(pair)

        assert h == hash(frozen_dict)


class TestTrialUpdater:
    TEST_VALUE = -1000

    def test_to_dict(self, trial_updater):
        assert trial_updater.to_dict() == {
            "model": {"a": 1, "b": 2, "sub": {"c": 3, "d": 4}},
            "data": {"e": 5, "f": 6},
        }

    def test_nesting_paths(self, trial_updater):
        assert trial_updater.paths() == {
            "model": (),
            "data": (),
            "sub": ("model",),
            "a": ("model",),
            "b": ("model",),
            "c": ("model", "sub"),
            "d": ("model", "sub"),
            "e": ("data",),
            "f": ("data",),
        }

    @pytest.mark.parametrize(
        "update_values",
        [
            {"model": {"a": TEST_VALUE}, "data": {"e": TEST_VALUE}},
            {"a": TEST_VALUE},
            {"model": {"sub": {"c": TEST_VALUE}}},
            {"sub": {"c": TEST_VALUE}},
            {"c": TEST_VALUE},
        ],
    )
    def test_update(self, trial_updater: utils._TrialUpdater, update_values: KwargType):
        trial_updater.update(update_values)

        assert trial_updater._updated

        ### check equality
        paths = trial_updater.paths()
        flat_expected_values: dict[str, tuple[int, tuple[str, ...]]] = {}

        stack = [update_values]
        while stack:
            curr = stack.pop()
            for key, value in curr.items():
                if isinstance(value, dict):
                    stack.append(value)
                else:
                    flat_expected_values[key] = (value, paths[key])

        trial_updater_values = {
            "model": trial_updater.model,
            "data": trial_updater.data,
        }
        for key, (expected_value, path) in flat_expected_values.items():
            curr = trial_updater_values
            for p in path:
                curr = curr[p]

            current_value = curr[key]
            assert current_value == expected_value
