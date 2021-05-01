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
        raise NotImplementedError

    def fit(self, **fit_options):
        raise NotImplementedError

    def predict(self, df):
        raise NotImplementedError

    def set_prior(self, priors):
        raise NotImplementedError

    def get_posterior(self):
        raise NotImplementedError


@dataclass
class LinkModel:
    node_model: NodeModel
    upper_links: List["LinkModel"] = field(default_factory=list)
    lower_links: List["LinkModel"] = field(default_factory=list)

    def fit(self, *args, **kwargs):
        raise NotImplementedError

    def predict(self, *args, **kwargs):
        raise NotImplementedError
