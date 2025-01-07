from collections.abc import Mapping
from dataclasses import dataclass, field, fields
from functools import partial
from typing import Any, Union

from boltons.iterutils import get_path, remap

from lightning_cv.typehints import KwargType


class FrozenDict(Mapping):
    def __init__(self, *args, **kwargs):
        self._d = dict(*args, **kwargs)
        self._hash = None

    def __getitem__(self, key):
        return self._d[key]

    def __iter__(self):
        return iter(self._d)

    def __len__(self):
        return len(self._d)

    def __hash__(self):
        if self._hash is None:  # pragma: no cover <- this is covered
            h = 0
            for pair in self.items():
                h ^= hash(pair)
            self._hash = h
        return self._hash

    def __repr__(self) -> str:
        dict_repr = repr(self._d)
        return f"FrozenDict({dict_repr})"


@dataclass # TODO: switch to attrs?
class _TrialUpdater:
    model: KwargType = field(default_factory=dict)
    data: KwargType = field(default_factory=dict)
    _updated: bool = field(default=False, init=False)

    @staticmethod
    def _update(
        parents: tuple, key: Any, value: Any, update_values: KwargType
    ) -> Union[bool, tuple]:
        """Tries to update leaf nodes based on an arbitrarily nested update schema.

        Almost fits the `boltons.iterutils` `visit` API. It is recommended to create a
        `functools.partial` object with this function.

        Args:
            parents (tuple): parent nodes to the current node
            key (Any): current node
            value (Any): value at current node
            update_values (KwargType): arbitrarily nested update mapping

        Returns:
            bool | tuple: Returns True if at an intermediate node. Otherwise, returns the
                new (node, value) map. See `boltons.iterutils.remap` for more info.
        """

        path = (*parents, key)
        # try to get a value at the current path in update_values
        # but default to the current value
        new_value = (
            # handles nesting
            get_path(root=update_values, path=path, default=None)
            # handles flat
            or update_values.get(key, None)
            # default
            or value
        )

        # skip updating intermediate nodes
        if isinstance(new_value, Mapping):  # pragma: no cover
            return True
        # otherwise update leaf with new value
        return key, new_value

    def update(self, values: KwargType):
        """Update all fields with new values.

        The `values` argument can have a variety of shapes:

        Example:
        ```python
        >>> model_cfg = {
        ...     "in_dim": 64,
        ...     "out_dim": 32,
        ...     "optimizer": {"lr": 1e-3}
        ... }
        >>> data_cfg = {"batch_size": 64}

        # now we want to try a new learning rate and a new batch size
        # the following two `values` schemas work:

        # 1. flat values
        >>> config = _TrialUpdater(model=model_cfg, data=data_cfg)
        >>> new_values = {"lr": 2.53e-4, "batch_size": 16}
        >>> config.update(new_values)
        >>> print(config.model)
        {"in_dim": 64, "out_dim": 32, "optimizer": {"lr": 2.53e-4}}
        >>> print(config.data)
        {"batch_size": 16}

        # 2. nested values
        >>> config = _TrialUpdater(model=model_cfg, data=data_cfg)
        >>> new_values = {
        ...     "model": {"optimizer": {"lr": 2.53e-4}},
        ...     "data": {"batch_size": 16}
        ... }
        >>> config.update(new_values)
        >>> print(config.model)
        {"in_dim": 64, "out_dim": 32, "optimizer": {"lr": 2.53e-4}}
        >>> print(config.data)
        {"batch_size": 16}

        # Further, a combination of flat and nested values also works.
        ```

        Args:
            values (KwargType): an arbitrarily nested map from subfield keys to new values
        """
        for key in fields(self):
            key = key.name
            if key.startswith("_"):
                continue

            current_values: KwargType = getattr(self, key)

            # handles nested and flat update value schemas
            update_values = values.get(key, values)
            update_fn = partial(_TrialUpdater._update, update_values=update_values)
            new_values = remap(root=current_values, visit=update_fn)

            current_values.update(new_values)
        self._updated = True

    def paths(self) -> dict[str, tuple[str, ...]]:
        """Returns a mapping from field names to their paths in the nested structure."""
        paths: dict[str, tuple[str, ...]] = {
            "model": (),
            "data": (),
        }

        stack: list[tuple[KwargType, tuple[str, ...]]] = [
            (self.model, ("model",)),
            (self.data, ("data",)),
        ]

        while stack:
            curr_dict, parents = stack.pop()

            for key, value in curr_dict.items():
                if isinstance(value, dict):
                    stack.append((value, (*parents, key)))

                paths[key] = parents

        return paths

    def to_dict(self) -> dict[str, KwargType]:
        return {
            "model": self.model,
            "data": self.data,
        }
