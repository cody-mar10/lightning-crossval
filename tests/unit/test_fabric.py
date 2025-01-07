import pytest
from conftest import DummyModel, DummyModelConfig
from lightning import Callback as LightingCallback

from lightning_cv.callbacks.base import Callback as LCVCallback
from lightning_cv.fabric import Fabric


class ModelCallback(DummyModel):
    def __init__(self):
        super().__init__(DummyModelConfig(in_dim=10))


@pytest.mark.parametrize(
    ("callback", "location"),
    [
        (LCVCallback, "_lcv_callbacks"),
        (LightingCallback, "_lightning_callbacks"),
        (ModelCallback, "_model_callbacks"),
    ],
)
def test_callback_locations(callback, location: str):
    callback_instance = callback()
    fabric = Fabric(
        callbacks=[callback_instance],
    )

    callback_storage: list = getattr(fabric, location)
    found = False
    for cb in callback_storage:
        if cb is callback_instance:
            found = True
            break

    assert found


@pytest.mark.parametrize("callback", [LightingCallback, ModelCallback])
def test_non_lcv_callback_warning(callback):
    callback_instance = callback()
    fabric = Fabric(
        callbacks=[callback_instance],
    )

    with pytest.warns(UserWarning):
        fabric.call("PYTEST_NON_EXISTING_HOOK")

class InvalidCallback:
    pass


@pytest.mark.parametrize("callback", [InvalidCallback, int, str, tuple])
def test_invalid_callback(callback):
    callback_instance = callback()
    with pytest.raises(ValueError):
        Fabric(
            callbacks=[callback_instance],
        )