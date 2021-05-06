"""
Tree Node
"""
from itertools import chain
from typing import Any, Callable, Iterable, List, Union

from pandas import DataFrame


class TreeNode:
    def __init__(self, name: str):
        self.name = name
        self.sup_node = None
        self.sub_nodes = []
        self.container = None

    @property
    def is_root(self) -> bool:
        return self.sup_node is None

    @property
    def is_leaf(self) -> bool:
        return len(self.sub_nodes) == 0

    @property
    def full_name(self) -> str:
        if self.is_root:
            return self.name
        return f"{self.sup_node.full_name}/{self.name}"

    @property
    def root(self) -> "TreeNode":
        if self.is_root:
            return self
        return self.sup_node.root

    @property
    def leafs(self) -> List["TreeNode"]:
        if self.is_leaf:
            return [self]
        return list(chain.from_iterable(
            [node.leafs for node in self.sub_nodes]
        ))

    @property
    def lower_nodes(self) -> List["TreeNode"]:
        if self.is_leaf:
            return [self]
        return [self] + list(chain.from_iterable(
            [node.lower_nodes for node in self.sub_nodes]
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
    def upper_rank(self) -> int:
        if self.is_root:
            return 0
        return self.sup_node.upper_rank + 1

    @property
    def lower_rank(self) -> int:
        if self.is_leaf:
            return 0
        return max(node.lower_rank for node in self.sub_nodes) + 1

    def append(self, node: Union[str, "TreeNode"]):
        node = self.as_treenode(node)
        if not node.is_root:
            raise ValueError(f"Cannot append {node}, "
                             f"already have parent {node.sup_node}.")
        sub_node_names = list(map(self.get_name, self.sub_nodes))
        if node.name in sub_node_names:
            index = sub_node_names.index(node.name)
            while len(node.sub_nodes) > 0:
                self.sub_nodes[index].append(node.pop())
        else:
            node.sup_node = self
            self.sub_nodes.append(node)

    def extend(self, nodes: Iterable[Union[str, "TreeNode"]]):
        for node in nodes:
            self.append(node)

    def merge(self, node: Union[str, "TreeNode"]):
        if node.name != self.name:
            raise ValueError("Cannot merge nodes with different names.")
        while len(node.sub_nodes) > 0:
            self.append(node.pop())

    def pop(self, *args) -> "TreeNode":
        node = self.sub_nodes.pop(*args)
        node.sup_node = None
        return node

    def remove(self, node: "TreeNode"):
        if not isinstance(node, TreeNode):
            raise TypeError("Can only remove TreeNode.")
        if not node.is_root:
            if node.sup_node is self:
                self.pop(self.sub_nodes.index(node))
            elif not self.is_leaf:
                for sub_node in self.sub_nodes:
                    sub_node.remove(node)

    def copy(self) -> "TreeNode":
        return self.__copy__()

    def __getitem__(self, name: str) -> "TreeNode":
        sub_node_names = list(map(self.get_name, self.sub_nodes))
        if name not in sub_node_names:
            raise KeyError(f"Cannot find {name} in sub nodes.")
        return self.sub_nodes[sub_node_names.index(name)]

    def __len__(self) -> int:
        if self.is_leaf:
            return 1
        return 1 + sum(len(node) for node in self.sub_nodes)

    def __add__(self, node: Union[str, "TreeNode"]) -> "TreeNode":
        self.merge(node)
        return self

    def __sub__(self, node: "TreeNode") -> "TreeNode":
        self.remove(node)
        return self

    def __truediv__(self, node: Union[str, "TreeNode"]) -> "TreeNode":
        self.append(node)
        return self.sub_nodes[-1]

    def __contains__(self, node: "TreeNode") -> bool:
        if not isinstance(node, TreeNode):
            raise TypeError("Can only contain TreeNode.")
        if node == self:
            return True
        return any(node in sub_node for sub_node in self.sub_nodes)

    def __eq__(self, node: "TreeNode") -> bool:
        if not isinstance(node, TreeNode):
            raise TypeError("Can only compare to TreeNode.")
        self_names = set(map(self.get_full_name, self.lower_nodes))
        node_names = set(map(self.get_full_name, node.lower_nodes))
        return self_names == node_names

    def __lt__(self, node: "TreeNode") -> bool:
        if not isinstance(node, TreeNode):
            raise TypeError("Can only compare to TreeNode.")
        self_names = set(map(self.get_full_name, self.lower_nodes))
        node_names = set(map(self.get_full_name, node.lower_nodes))
        return self_names < node_names

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
        if len(names) == 0:
            raise ValueError("Names must not be empty.")
        nodes = [cls(names[0])]
        for name in names[1:]:
            nodes.append(nodes[-1] / name)
        return nodes[0]

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

    @staticmethod
    def get_name(node: "TreeNode") -> str:
        return node.name

    @staticmethod
    def get_full_name(node: "TreeNode") -> str:
        return node.full_name
