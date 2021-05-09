"""
Customize Collections Class
"""
from collections import OrderedDict
from itertools import islice
from typing import Any, Iterator, Union


class NamedList(OrderedDict):
    """
    List with names, a wrapper around OrderedDict
    """

    def _as_key(self, key: Union[int, str]) -> str:
        if isinstance(key, int):
            index = key if key >= 0 else len(self) + key
            key = next(islice(self.keys(), index, None))
        return str(key)

    def pop(self, key: Union[int, str] = -1) -> Any:
        return super().pop(self._as_key(key))

    def __iter__(self) -> Iterator:
        return iter(self.values())

    def __getitem__(self, key: Union[int, str]) -> Any:
        return super().__getitem__(self._as_key(key))

    def __setitem__(self, key: Union[int, str], value: Any):
        super().__setitem__(self._as_key(key), value)
