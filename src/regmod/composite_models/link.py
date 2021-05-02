"""
Link for relation map between nodes
"""
from typing import List, Union
from dataclasses import dataclass, field


@dataclass
class Link:
    name: str
    upper_links: List["Link"] = field(default_factory=list)
    lower_links: List["Link"] = field(default_factory=list)

    def __post_init__(self):
        for link in self.upper_links:
            if self not in link.lower_links:
                link.lower_links.append(self)
        for link in self.lower_links:
            if self not in link.upper_links:
                link.upper_links.append(self)

    @staticmethod
    def check_links(links: Union["Link", List["Link"]]) -> List["Link"]:
        if isinstance(links, Link):
            links = [links]
        if not all(isinstance(link, Link) for link in links):
            raise TypeError("Can only connect Link object to Link object.")
        return links

    def add_lower_links(self, links: Union["Link", List["Link"]]):
        links = self.check_links(links)
        for link in links:
            if link not in self.lower_links:
                self.lower_links.append(link)
            if self not in link.upper_links:
                link.upper_links.append(self)

    def add_upper_links(self, links: Union["Link", List["Link"]]):
        links = self.check_links(links)
        for link in links:
            if link not in self.upper_links:
                self.upper_links.append(link)
            if self not in link.lower_links:
                link.lower_links.append(self)

    @property
    def is_root(self) -> bool:
        return len(self.upper_links) == 0

    @property
    def is_leaf(self) -> bool:
        return len(self.lower_links) == 0

    @property
    def upper_rank(self) -> int:
        if self.is_root:
            return 0
        return max([link.upper_rank for link in self.upper_links]) + 1

    @property
    def lower_rank(self) -> int:
        if self.is_leaf:
            return 0
        return max([link.lower_rank for link in self.lower_links]) + 1

    def to_list(self) -> List["Link"]:
        current_list = [self]
        if not self.is_leaf:
            for link in self.lower_links:
                current_list.extend(link.to_list())
        return current_list

    def __repr__(self) -> str:
        return (f"{type(self).__name__}(name={self.name},"
                f" upper_rank={self.upper_rank},"
                f" lower_rank={self.lower_rank})")
