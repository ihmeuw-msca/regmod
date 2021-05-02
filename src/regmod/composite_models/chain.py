"""
Chain Model
"""
from typing import List

from pandas import DataFrame
from regmod.composite_models.composite import CompositeModel
from regmod.composite_models.link import Link
from regmod.composite_models.node import NodeModel


class ChainModel(CompositeModel):
    """
    Chain Model with sequential model fitting.
    """

    def __init__(self,
                 name: str,
                 models: List[NodeModel],
                 links: List[Link] = None):

        if links is None:
            model_names = [model.name for model in models]
            links = [Link(model_name) for model_name in model_names]
            for i in range(len(links) - 1):
                links[i].add_lower_links(links[i + 1])

        ordered_links = [link for link in links if link.is_root]
        if len(ordered_links) != 1:
            raise ValueError("Links must have one and only one root.")
        while not ordered_links[-1].is_leaf:
            lower_links = ordered_links[-1].lower_links
            if len(lower_links) != 1:
                raise ValueError("Given links cannot form a chain.")
            next_link = lower_links[0]
            if next_link in ordered_links:
                raise ValueError("There is a loop in the given links.")
            ordered_links.append(next_link)

        super().__init__(name, models, ordered_links)
        self.models = [self.model_dict[link.name] for link in self.links]

    def fit(self, **fit_options):
        col_value = self.get_col_value()
        self.models[0].fit(**fit_options)
        for model_id in range(1, self.num_models):
            df = self.predict_model(model_id, col_value=col_value)
            self.models[model_id].set_data(df, col_value=col_value)
            self.models[model_id].fit(**fit_options)

    def predict_model(self,
                      model_id: int,
                      col_value: str = None) -> DataFrame:
        col_value = self.get_col_value(col_value)
        df = self.models[model_id].get_data()
        for i in range(model_id):
            df = self.models[i].predict(df, col_value=col_value)
        return df

    def predict(self,
                df: DataFrame = None,
                col_value: str = None,
                col_label: str = None):
        if df is None:
            df = self.models[-1].get_data()
        df = self.subset_df(df, col_label)
        col_value = self.get_col_value(col_value)

        for model in self.models:
            df = model.predict(df, col_value=col_value)
        return df
