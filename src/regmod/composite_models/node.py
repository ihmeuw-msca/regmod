"""
Tree Node
"""
from itertools import chain
from functools import reduce
from operator import truediv, attrgetter
from typing import Any, Iterable, List, Union

from regmod.composite_models.collections import ChainNamedList


class Node:
    def __init__(self, name: str):
        self.name = name
        self.parent = None
        self.children = ChainNamedList({}, {})

    name = property(attrgetter("_name"))

    @name.setter
    def name(self, name: str):
        name = str(name)
        if "/" in name:
            raise ValueError(f"name={name} cannot contain character '/'")
        self._name = name

    @property
    def isroot(self) -> bool:
        return self.parent is None

    @property
    def isleaf(self) -> bool:
        return len(self.children) == 0

    @property
    def full_name(self) -> str:
        return self.get_name(0)

    @property
    def root(self) -> "Node":
        if self.isroot:
            return self
        return self.parent.root

    @property
    def leafs(self) -> List["Node"]:
        if self.isleaf:
            return [self]
        return list(chain.from_iterable(
            node.leafs for node in self.children
        ))

    @property
    def branch(self) -> List["Node"]:
        if self.isleaf:
            return [self]
        return [self] + list(chain.from_iterable(
            node.branch for node in self.children
        ))

    @property
    def tree(self) -> List["Node"]:
        return self.root.branch

    @property
    def level(self) -> int:
        if self.isroot:
            return 0
        return self.parent.level + 1

    def append(self, node: Union[str, "Node"], rank: int = 0):
        node = self.as_node(node)
        if not node.isroot:
            raise ValueError(f"Cannot append {node}, "
                             f"already have parent {node.parent}.")
        if node.name in self.children:
            self.children[node.name].merge(node)
        else:
            node.parent = self
            self.children.named_lists[rank][node.name] = node

    def extend(self, nodes: Iterable[Union[str, "Node"]], rank: int = 0):
        for node in nodes:
            self.append(node, rank=rank)

    def merge(self, node: Union[str, "Node"]):
        if node.name != self.name:
            raise ValueError("Cannot merge with node with different name.")
        for rank, children in enumerate(node.children.named_lists):
            while len(children) > 0:
                self.append(node.pop(rank=rank), rank=rank)

    def pop(self, name: Union[int, str] = -1, rank: int = None) -> "Node":
        children = self.children
        if rank is not None:
            children = self.children.named_lists[rank]
        node = children.pop(name)
        node.parent = None
        return node

    def detach(self):
        if not self.isroot:
            self.parent.pop(self.name)

    def get_name(self, level: int) -> str:
        if self.level <= level or self.isroot:
            return self.name
        return "/".join([self.parent.get_name(level), self.name])

    def get_leafs(self, *ranks) -> List["Node"]:
        if len(ranks) == 0:
            ranks = range(len(self.children.named_lists))
        leafs = list(chain.from_iterable(
            node.get_leafs(*ranks) for node in self.children
        ))
        if all(len(self.children.named_lists[rank]) == 0 for rank in ranks):
            leafs.insert(0, self)
        return leafs

    def copy(self) -> "Node":
        return self.__copy__()

    def __getitem__(self, name: str) -> "Node":
        names = name.split("/", 1)
        node = self.children[names[0]]
        if len(names) == 1:
            return node
        return node[names[1]]

    def __len__(self) -> int:
        if self.isleaf:
            return 1
        return 1 + sum(len(node) for node in self.children)

    def __or__(self, node: Union[str, "Node"]) -> "Node":
        self.merge(node)
        return self

    def __truediv__(self, node: Union[str, "Node"]) -> "Node":
        rank = 0
        self.append(node, rank=rank)
        return self.children.named_lists[rank][-1]

    def __floordiv__(self, node: Union[str, "Node"]) -> "Node":
        rank = 1
        self.append(node, rank=rank)
        return self.children.named_lists[rank][-1]

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
        self_node = type(self)(self.name)
        for node in self.children:
            self_node.append(node.__copy__())
        return self_node

    @classmethod
    def as_node(cls, obj: Any) -> "Node":
        if isinstance(obj, Node):
            return obj
        return cls(str(obj))

    @classmethod
    def from_names(cls, names: Iterable[str]) -> "Node":
        return reduce(truediv, map(cls.as_node, names)).root

    @classmethod
    def from_full_name(cls, full_name: str) -> "Node":
        names = full_name.split("/")
        return cls.from_names(names)