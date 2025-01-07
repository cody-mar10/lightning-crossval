import dataclasses

import attrs
import pytest

from lightning_cv._optional import HAS_PYDANTIC, PYDANTIC_V2, is_pydantic_basemodel


@attrs.define
class AttrsDataclass:
    pass


@dataclasses.dataclass
class Dataclass:
    pass


CONFIGS = [
    AttrsDataclass(),
    AttrsDataclass,
    Dataclass(),
    Dataclass,
    dict,
    dict(),
]

if HAS_PYDANTIC:
    import pydantic

    class PydanticModel(pydantic.BaseModel):
        pass

    CONFIGS += [PydanticModel(), PydanticModel]


@pytest.mark.skipif(HAS_PYDANTIC, reason="Requires Pydantic to be missing")
class TestMissingPydantic:
    @pytest.mark.parametrize("config", CONFIGS)
    def test_is_pydantic_basemodel(self, config):
        # should always return False if Pydantic is not installed
        assert not is_pydantic_basemodel(config)


skip_not_pydantic_v2 = pytest.mark.skipif(
    not PYDANTIC_V2, reason="Requires Pydantic >= 2.0.0"
)


@pytest.mark.skipif(not HAS_PYDANTIC, reason="Requires Pydantic")
class TestHasPydantic:
    @pytest.mark.parametrize(
        ("config", "expected"),
        zip(CONFIGS, [False, False, False, False, False, False, True, True]),
    )
    def test_is_pydantic_basemodel(self, config, expected: bool):
        # should always return True if Pydantic is installed
        assert is_pydantic_basemodel(config) is expected

    @skip_not_pydantic_v2
    def test_pydantic_has_model_methods(self):
        class Model(pydantic.BaseModel):
            pass

        assert hasattr(Model, "model_dump")
        assert hasattr(Model, "model_validate")

    @skip_not_pydantic_v2
    def test_pydantic_deprecated_methods(self):
        class Model(pydantic.BaseModel):
            x: int

        with pytest.deprecated_call():
            Model(x=2).dict()

        with pytest.deprecated_call():
            Model.validate({"x": 2})
