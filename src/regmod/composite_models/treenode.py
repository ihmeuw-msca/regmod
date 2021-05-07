"""
Tree Node
"""
from collections import OrderedDict
from itertools import chain
from functools import reduce
from operator import truediv
from typing import Any, Callable, Iterable, List, Union

from pandas import DataFrame


class TreeNode:
    def __init__(self, name: str):
        name = str(name)
        if "/" in name:
            raise ValueError(f"name={name} cannot contain character '/'")
        self.name = name
        self.sup_node = None
        self.sub_nodes_dict = OrderedDict()
        self.container = None

    @property
    def is_root(self) -> bool:
        return self.sup_node is None

    @property
    def is_leaf(self) -> bool:
        return len(self.sub_nodes) == 0

    @property
    def full_name(self) -> str:
        return self.get_name(0)

    @property
    def root(self) -> "TreeNode":
        if self.is_root:
            return self
        return self.sup_node.root

    @property
    def leafs(self) -> List["TreeNode"]:
        if self.is_leaf:
            return [self]
        return list(chain.from_iterable([n.leafs for n in self.sub_nodes]))

    @property
    def sub_nodes(self) -> List["TreeNode"]:
        return list(self.sub_nodes_dict.values())

    @property
    def lower_nodes(self) -> List["TreeNode"]:
        if self.is_leaf:
            return [self]
        return [self] + list(chain.from_iterable(
            [n.lower_nodes for n in self.sub_nodes]
        ))

    @property
    def upper_nodes(self) -> List["TreeNode"]:
        if self.is_root:
            return [self]
        return [self] + self.sup_node.upper_nodes

    @property
    def all_nodes(self) -> List["TreeNode"]:
        return self.root.lower_nodes

    @property
    def level(self) -> int:
        if self.is_root:
            return 0
        return self.sup_node.level + 1

    def append(self, node: Union[str, "TreeNode"]):
        node = self.as_treenode(node)
        if not node.is_root:
            raise ValueError(f"Cannot append {node}, "
                             f"already have parent {node.sup_node}.")
        if node.name in self.sub_nodes_dict:
            while len(node.sub_nodes) > 0:
                self[node.name].append(node.pop())
        else:
            node.sup_node = self
            self.sub_nodes_dict[node.name] = node

    def extend(self, nodes: Iterable[Union[str, "TreeNode"]]):
        for n in nodes:
            self.append(n)

    def merge(self, node: Union[str, "TreeNode"]):
        if node.name != self.name:
            raise ValueError("Cannot merge nodes with different names.")
        while len(node.sub_nodes) > 0:
            self.append(node.pop())

    def pop(self, key: Union[int, str] = -1) -> "TreeNode":
        if isinstance(key, int):
            key = list(self.sub_nodes_dict.keys())[key]
        node = self.sub_nodes_dict.pop(key)
        node.sup_node = None
        return node

    def detach(self):
        if not self.is_root:
            del self.sup_node.sub_nodes_dict[self.name]
            self.sup_node = None

    def get_name(self, level: int) -> str:
        if self.level <= level or self.is_root:
            return self.name
        return f"{self.sup_node.get_name(level)}/{self.name}"

    def copy(self) -> "TreeNode":
        return self.__copy__()

    def __getitem__(self, name: str) -> "TreeNode":
        names = name.split("/", 1)
        node = self.sub_nodes_dict[names[0]]
        if len(names) == 1:
            return node
        return node[names[1]]

    def __len__(self) -> int:
        if self.is_leaf:
            return 1
        return 1 + sum(len(n) for n in self.sub_nodes)

    def __add__(self, node: Union[str, "TreeNode"]) -> "TreeNode":
        self.merge(node)
        return self

    def __truediv__(self, node: Union[str, "TreeNode"]) -> "TreeNode":
        self.append(node)
        return self.sub_nodes[-1]

    def __contains__(self, node: "TreeNode") -> bool:
        if not isinstance(node, TreeNode):
            raise TypeError("Can only contain TreeNode.")
        if node == self:
            return True
        return any(node in n for n in self.sub_nodes)

    def __eq__(self, node: "TreeNode") -> bool:
        if not isinstance(node, TreeNode):
            raise TypeError("Can only compare to TreeNode.")

        self_names = set(n.get_name(self.level) for n in self.lower_nodes)
        node_names = set(n.get_name(node.level) for n in node.lower_nodes)
        return self_names == node_names

    def __lt__(self, node: "TreeNode") -> bool:
        if not isinstance(node, TreeNode):
            raise TypeError("Can only compare to TreeNode.")
        self_names = (n.get_name(self.level) for n in self.lower_nodes)
        node_names = (n.get_name(node.level) for n in node.lower_nodes)
        return all(any(self_name in node_name for node_name in node_names)
                   for self_name in self_names)

    def __gt__(self, node: "TreeNode") -> bool:
        return node < self

    def __le__(self, node: "TreeNode") -> bool:
        return node == self or self < node

    def __ge__(self, node: "TreeNode") -> bool:
        return node == self or self > node

    def __repr__(self) -> str:
        return f"{type(self).__name__}(name={self.name})"

    def __copy__(self) -> "TreeNode":
        root_node = type(self)(self.name)
        for node in self.sub_nodes:
            root_node.append(node.__copy__())
        return root_node

    @classmethod
    def as_treenode(cls, obj: Any) -> "TreeNode":
        if isinstance(obj, cls):
            return obj
        return TreeNode(str(obj))

    @classmethod
    def from_names(cls, names: Iterable[str]) -> "TreeNode":
        return reduce(truediv, map(cls.as_treenode, names)).root

    @classmethod
    def from_full_name(cls, full_name: str) -> "TreeNode":
        names = full_name.split("/")
        return cls.from_names(names)

    @classmethod
    def from_dataframe(cls,
                       df: DataFrame,
                       id_cols: List[str],
                       root_name: str = "Global",
                       container_fun: Callable = None) -> "TreeNode":
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
