from collections.abc import Mapping


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
        if self._hash is None:
            h = 0
            for pair in self.items():
                h ^= hash(pair)
            self._hash = h
        return self._hash

    def __repr__(self) -> str:
        dict_repr = dict.__repr__(self._d)
        return f"FrozenDict({dict_repr})"
