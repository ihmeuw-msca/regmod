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

    def fit(self, **fit_options):
        raise NotImplementedError

    def predict(self, df):
        raise NotImplementedError

    def set_prior(self, priors):
        raise NotImplementedError

    def get_posterior(self):
        raise NotImplementedError


@dataclass
class Link:
    name: str
    upper_links: List["Link"] = field(default_factory=list)
    lower_links: List["Link"] = field(default_factory=list)

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
