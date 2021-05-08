"""
Customize Collections Class
"""
from collections import OrderedDict
from typing import Any, List, Iterable, Iterator, Union


class NamedList:
    """
    List with names, a wrapper around OrderedDict
    """

    def __init__(self, keys: Iterable[str], values: Iterable[Any]):
        self.ordered_dict = OrderedDict(zip(map(str, keys), values))

    @property
    def keys(self) -> List[str]:
        return list(self.ordered_dict.keys())

    @property
    def values(self) -> List[str]:
        return list(self.ordered_dict.values())

    def _as_key(self, key: Union[int, str]) -> str:
        return self.keys[key] if isinstance(key, int) else str(key)

    def append(self, key: str, value: Any):
        self.ordered_dict.update({str(key): value})

    def extend(self, keys: Iterable[str], values: Iterable[Any]):
        for key, value in zip(keys, values):
            self.append(key, value)

    def pop(self, key: Union[int, str] = -1) -> Any:
        return self.ordered_dict.pop(self._as_key(key))

    def __contains__(self, key: str) -> bool:
        return key in self.keys

    def __iter__(self) -> Iterator:
        return iter(self.values)

    def __getitem__(self, key: Union[int, str]) -> Any:
        return self.ordered_dict[self._as_key(key)]

    def __setitem__(self, key: Union[int, str], value: Any):
        self.ordered_dict[self._as_key(key)] = value

    def __len__(self) -> int:
        return len(self.ordered_dict)

    def __repr__(self) -> str:
        return f"{type(self).__name__}(keys={self.keys}, values={self.values})"
