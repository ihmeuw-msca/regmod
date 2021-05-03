"""
Composite Model
"""
from typing import Dict, List

import pandas as pd
from pandas import DataFrame
from regmod.composite_models.link import Link
from regmod.composite_models.node import NodeModel


class CompositeModel(NodeModel):
    """
    Composite Model, abstract behavior of group model.
    """

    def __init__(self,
                 name: str,
                 models: List[NodeModel],
                 links: List[Link] = None):
        super().__init__(name)

        if not all(isinstance(model, NodeModel) for model in models):
            raise TypeError("Models must be instances of NodeModel.")

        self.models = models
        self.model_dict = {model.name: model for model in self.models}
        model_names = [model.name for model in self.models]

        if links is None:
            links = [Link(model_name) for model_name in model_names]
        if not all(isinstance(link, Link) for link in links):
            raise TypeError("Links must be instances of Link.")
        link_names = [link.name for link in links]
        if set(model_names) != set(link_names):
            raise ValueError("Models and links not match.")

        self.links = links
        self.link_dict = {link.name: link for link in self.links}

    @property
    def num_models(self) -> int:
        return len(self.models)

    def get_data(self, col_label: str = None) -> DataFrame:
        df = pd.concat([model.get_data(col_label=self.name)
                        for model in self.models])
        if col_label is not None:
            df[col_label] = self.name
        return df

    def set_data(self,
                 df: DataFrame,
                 col_value: str = None,
                 col_label: str = None):
        df = self.subset_df(df, col_label, copy=False)
        for model in self.models:
            model.set_data(df, col_value=col_value, col_label=self.name)

    def get_posterior(self) -> Dict:
        return {
            model.name: model.get_posterior()
            for model in self.models
        }

    def set_prior(self, priors: Dict, masks: Dict = None):
        for name, prior in priors.items():
            mask = None if masks is None or name not in masks else masks[name]
            self.model_dict[name].set_prior(prior, mask)

    def fit(self, **fit_options):
        for model in self.models:
            model.fit(**fit_options)

    def predict(self,
                df: DataFrame = None,
                col_value: str = None,
                col_label: str = None) -> DataFrame:
        if df is None:
            df = self.get_data()
        col_value = self.get_col_value(col_value)
        df = self.subset_df(df, col_label)
        return pd.concat([model.predict(df,
                                        col_value=col_value,
                                        col_label=self.name)
                          for model in self.models])
