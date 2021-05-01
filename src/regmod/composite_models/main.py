"""
Main Classes
"""
from typing import Dict, List, Union
from dataclasses import dataclass, field
from pandas import DataFrame


class NodeModel:
    """
    Abstract class that encode the behavior of the node
    """

    def __init__(self,
                 name: str,
                 df: Union[DataFrame, Dict] = None):
        self.name = name
        self.df = df

    def set_data(self, df):
        raise NotImplementedError

    def get_data(self):
        return self.df

    def set_offset(self, df: DataFrame, col: str):
        raise NotImplementedError

    def fit(self, **fit_options):
        raise NotImplementedError

    def predict(self, df: DataFrame = None, col: str = None):
        raise NotImplementedError

    def set_prior(self, priors, masks=None):
        raise NotImplementedError

    def get_posterior(self):
        raise NotImplementedError

    def __repr__(self) -> str:
        return f"{type(self).__name__}(name={self.name})"


@dataclass
class Link:
    name: str
    upper_links: List["Link"] = field(default_factory=list)
    lower_links: List["Link"] = field(default_factory=list)

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

    def __repr__(self) -> str:
        return (f"{type(self).__name__}(name={self.name},"
                f" upper_rank={self.upper_rank},"
                f" lower_rank={self.lower_rank})")
