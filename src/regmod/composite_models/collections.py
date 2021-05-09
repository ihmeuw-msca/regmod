"""
Customize Collections Class
"""
from collections import OrderedDict
from itertools import islice, chain
from typing import Any, Iterator, Union, Tuple


class NamedList(OrderedDict):
    """
    List with names, a wrapper around OrderedDict
    """

    def _as_key(self, key: Union[int, str]) -> str:
        if isinstance(key, int):
            index = key if key >= 0 else len(self) + key
            key = next(islice(self.keys(), index, None))
        return key

    def pop(self, key: Union[int, str] = -1) -> Any:
        return super().pop(self._as_key(key))

    def __iter__(self) -> Iterator:
        return iter(self.values())

    def __getitem__(self, key: Union[int, str]) -> Any:
        return super().__getitem__(self._as_key(key))

    def __setitem__(self, key: Union[int, str], value: Any):
        super().__setitem__(self._as_key(key), value)


class ChainNamedList:
    """
    A unified view of a set of NamedList
    """

    def __init__(self, *args):
        self.named_lists = [
            arg if isinstance(arg, NamedList) else NamedList(arg)
            for arg in args
        ]

    def keys(self) -> Iterator:
        return chain.from_iterable(nlst.keys() for nlst in self.named_lists)

    def values(self) -> Iterator:
        return chain.from_iterable(nlst.values() for nlst in self.named_lists)

    def _as_key(self,
                key: Union[int, str],
                raise_keyerror: bool = True) -> Tuple[int, str]:
        if isinstance(key, int):
            index = key if key >= 0 else len(self) + key
            key = next(islice(self.keys(), index, None))
        i = next(
            (i for i, nlst in enumerate(self.named_lists) if key in nlst), -1
        )
        if raise_keyerror and i == -1:
            raise KeyError(f"'{key}'")
        return i, key

    def pop(self, key: Union[int, str] = -1) -> Any:
        i, key = self._as_key(key)
        return self.named_lists[i].pop(key)

    def __contains__(self, key: str) -> bool:
        return any(key in nlst for nlst in self.named_lists)

    def __iter__(self) -> Iterator:
        return iter(self.values())

    def __getitem__(self, key: Union[int, str]) -> Any:
        i, key = self._as_key(key)
        return self.named_lists[i][key]

    def __setitem__(self, key: Union[int, str], value: Any):
        i, key = self._as_key(key, raise_keyerror=False)
        i = i if i >= 0 else len(self.named_lists) + i
        self.named_lists[i][key] = value

    def __len__(self) -> int:
        return sum(len(nlst) for nlst in self.named_lists)

    def __repr__(self) -> str:
        head = type(self).__name__
        contents = ", ".join(map(str, self.named_lists))
        return f"{head}({contents})"
