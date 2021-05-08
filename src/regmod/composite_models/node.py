"""
Tree Node
"""
from itertools import chain
from functools import reduce
from operator import truediv, attrgetter
from typing import Any, Callable, Iterable, List, Union

from pandas import DataFrame

from regmod.composite_models.collections import NamedList


class Node:
    def __init__(self, name: str):
        self.name = name
        self.parent = None
        self.children = NamedList()
        self.container = None

    name = property(attrgetter("_name"))

    @name.setter
    def name(self, name: str):
        name = str(name)
        if "/" in name:
            raise ValueError(f"name={name} cannot contain character '/'")
        self._name = name

    @property
    def is_root(self) -> bool:
        return self.parent is None

    @property
    def is_leaf(self) -> bool:
        return len(self.children) == 0

    @property
    def full_name(self) -> str:
        return self.get_name(0)

    @property
    def root(self) -> "Node":
        if self.is_root:
            return self
        return self.parent.root

    @property
    def leafs(self) -> List["Node"]:
        if self.is_leaf:
            return [self]
        return list(chain.from_iterable([n.leafs for n in self.children]))

    @property
    def branch(self) -> List["Node"]:
        if self.is_leaf:
            return [self]
        return [self] + list(chain.from_iterable(
            [n.branch for n in self.children]
        ))

    @property
    def tree(self) -> List["Node"]:
        return self.root.branch

    @property
    def level(self) -> int:
        if self.is_root:
            return 0
        return self.parent.level + 1

    def append(self, node: Union[str, "Node"]):
        node = self.as_node(node)
        if not node.is_root:
            raise ValueError(f"Cannot append {node}, "
                             f"already have parent {node.parent}.")
        if node.name in self.children:
            while len(node.children) > 0:
                self[node.name].append(node.pop())
        else:
            node.parent = self
            self.children[node.name] = node

    def extend(self, nodes: Iterable[Union[str, "Node"]]):
        for node in nodes:
            self.append(node)

    def merge(self, node: Union[str, "Node"]):
        if node.name != self.name:
            self.name = f"{self.name}|{node.name}"
        while len(node.children) > 0:
            self.append(node.pop())

    def pop(self, key: Union[int, str] = -1) -> "Node":
        node = self.children.pop(key)
        node.parent = None
        return node

    def detach(self):
        if not self.is_root:
            self.parent.pop(self.name)

    def get_name(self, level: int) -> str:
        if self.level <= level or self.is_root:
            return self.name
        return f"{self.parent.get_name(level)}/{self.name}"

    def copy(self) -> "Node":
        return self.__copy__()

    def __getitem__(self, name: str) -> "Node":
        names = name.split("/", 1)
        node = self.children[names[0]]
        if len(names) == 1:
            return node
        return node[names[1]]

    def __len__(self) -> int:
        if self.is_leaf:
            return 1
        return 1 + sum(len(node) for node in self.children)

    def __or__(self, node: Union[str, "Node"]) -> "Node":
        self.merge(node)
        return self

    def __truediv__(self, node: Union[str, "Node"]) -> "Node":
        self.append(node)
        return self.children[-1]

    def __contains__(self, node: "Node") -> bool:
        if not isinstance(node, Node):
            raise TypeError("Can only contain Node.")
        if node == self:
            return True
        return any(node in _node for _node in self.children)

    def __eq__(self, node: "Node") -> bool:
        if not isinstance(node, Node):
            raise TypeError("Can only compare to Node.")
        self_names = set(_node.get_name(self.level) for _node in self.branch)
        node_names = set(_node.get_name(node.level) for _node in node.branch)
        return self_names == node_names

    def __lt__(self, node: "Node") -> bool:
        if not isinstance(node, Node):
            raise TypeError("Can only compare to Node.")
        self_names = (_node.get_name(self.level) for _node in self.branch)
        node_names = (_node.get_name(node.level) for _node in node.branch)
        return all(any(self_name in node_name for node_name in node_names)
                   for self_name in self_names)

    def __gt__(self, node: "Node") -> bool:
        return node < self

    def __le__(self, node: "Node") -> bool:
        return node == self or self < node

    def __ge__(self, node: "Node") -> bool:
        return node == self or self > node

    def __repr__(self) -> str:
        return f"{type(self).__name__}(name={self.name})"

    def __copy__(self) -> "Node":
        root_node = type(self)(self.name)
        for node in self.children:
            root_node.append(node.__copy__())
        return root_node

    @classmethod
    def as_node(cls, obj: Any) -> "Node":
        if isinstance(obj, cls):
            return obj
        return Node(str(obj))

    @classmethod
    def from_names(cls, names: Iterable[str]) -> "Node":
        return reduce(truediv, map(cls.as_node, names)).root

    @classmethod
    def from_full_name(cls, full_name: str) -> "Node":
        names = full_name.split("/")
        return cls.from_names(names)

    @classmethod
    def from_dataframe(cls,
                       df: DataFrame,
                       id_cols: List[str],
                       root_name: str = "Global",
                       container_fun: Callable = None) -> "Node":
        if not all(col in df.columns for col in id_cols):
            raise ValueError("Columns must be in the dataframe.")
        root_node = cls(root_name)
        if container_fun is not None:
            root_node.container = container_fun(root_node, df)
        if len(id_cols) == 0:
            return root_node
        df_group = df.groupby(id_cols[0])
        for name in df_group.groups.keys():
            root_node.append(cls.from_dataframe(
                df_group.get_group(name),
                id_cols[1:],
                root_name=name,
                container_fun=container_fun
            ))
        return root_node
