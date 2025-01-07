from glob import glob
from tempfile import NamedTemporaryFile
from typing import BinaryIO, Optional, TypedDict

import pytest

from lightning_cv._optional import HAS_OPTUNA
from lightning_cv.typehints import KwargType

# skip this entire directory if optuna is not installed
if not HAS_OPTUNA:
    collect_ignore = glob("test_*.py")


class RawConfig(TypedDict):
    hparams: list[KwargType]
    forbidden: Optional[list[KwargType]]


@pytest.fixture
def config_dict() -> RawConfig:
    config: RawConfig = {
        "hparams": [
            {
                "name": "lr",
                "suggest": "float",
                "low": 1e-5,
                "high": 1e-2,
                "log": True,
                "parent": "optimizer",
            },
            {
                "name": "batch_size",
                "suggest": "int",
                "low": 16,
                "high": 128,
                "parent": "data",
            },
            {
                "name": "optimizer_type",
                "suggest": "categorical",
                "choices": ["adam", "sgd"],
                "parent": "optimizer",
            },
            {
                "name": "num_heads",
                "suggest": "int",
                "low": 1,
                "high": 4,
            },
            {
                "name": "hidden_dim",
                "suggest": "categorical",
                "choices": [64, 128, 256, 300],
            },
            {
                "name": "dropout",
                "suggest": "float",
                "low": 0.1,
                "high": 0.5,
            },
            {
                "name": "num_layers",
                "suggest": "categorical",
                "choices": [1, 2, 3, 4, 5, 30],
            },
            {"name": "epochs", "suggest": "int", "low": 1, "high": 100},
            {
                "name": "mappable",
                "suggest": "categorical",
                "choices": ["A", "B", "C"],
                "map": {
                    "A": {"mappable_sub0": 0, "mappable_sub1": 1},
                    "B": {"mappable_sub0": 2, "mappable_sub1": 3},
                    "C": {"mappable_sub0": 4, "mappable_sub1": 5},
                },
            },
        ],
        "forbidden": None,
    }

    return config


@pytest.fixture
def config_dict_with_forbidden(config_dict) -> RawConfig:
    config_dict["forbidden"] = [
        {"lr": 1e-3, "batch_size": 32},
        {"lr": 1e-4, "batch_size": 64},
        {"num_heads": 3, "hidden_dim": 64},
        {"num_heads": 4, "hidden_dim": 128},
        {"hidden_dim": 300, "num_layers": 30, "num_heads": 4},
    ]

    return config_dict


@pytest.fixture
def invalid_config_dict(config_dict_with_forbidden: RawConfig) -> RawConfig:
    config_dict_with_forbidden["forbidden"].append(  # type: ignore[method]
        {"NOT_AN_HPARAM": "NOT_A_VALUE", "lr": 1e-3},
    )

    return config_dict_with_forbidden


def _write_toml_config(config: RawConfig, fobj: BinaryIO):
    def _tomlify(value):
        if isinstance(value, str):
            return f'"{value}"'

        if isinstance(value, list):
            tomlrepr: list[str] = []
            for v in value:
                v = _tomlify(v)
                tomlrepr.append(v)

            tomllist = ", ".join(tomlrepr)

            return f"[{tomllist}]"

        if isinstance(value, dict):
            tomlrepr: list[str] = []
            for k, v in value.items():
                v = _tomlify(v)
                tomlrepr.append(f"{k} = {v}")

            tomltable = ", ".join(tomlrepr)

            return f"{{ {tomltable} }}"

        if isinstance(value, bool):
            return str(value).lower()

        return str(value)

    def _write_list(list_of_dicts: list[KwargType], header: str):
        for hparam in list_of_dicts:
            fobj.write(f"[[{header}]]\n".encode())
            for key, value in hparam.items():
                if key == "map" and value is not None:
                    # this technically only works since I wrote the map key to be last,
                    # so the prior [[hparam]] is already written and refers to this same hparam
                    fobj.write(f"\n[{header}.map]\n".encode())
                    for map_key, map_value in value.items():
                        map_value = _tomlify(map_value)
                        fobj.write(f"{map_key} = {map_value}\n".encode())
                    fobj.write(b"\n")
                else:
                    value = _tomlify(value)
                    fobj.write(f"{key} = {value}\n".encode())
            fobj.write(b"\n")

    _write_list(config["hparams"], "hparams")

    if config["forbidden"] is not None:
        fobj.seek(fobj.tell() - 1)  # delete previous blank line
        _write_list(config["forbidden"], "forbidden")


def _setup_written_toml_file(c: RawConfig):
    with NamedTemporaryFile("w+b") as fp:
        _write_toml_config(c, fp)  # type: ignore[arg-type]
        fp.seek(0)
        yield fp.name


@pytest.fixture
def config_file(config_dict: RawConfig):
    yield from _setup_written_toml_file(config_dict)


@pytest.fixture
def config_file_with_forbidden(config_dict_with_forbidden: RawConfig):
    yield from _setup_written_toml_file(config_dict_with_forbidden)


@pytest.fixture
def invalid_config_file(invalid_config_dict: RawConfig):
    yield from _setup_written_toml_file(invalid_config_dict)
