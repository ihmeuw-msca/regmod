"""
Chain Model
"""
from typing import List
from pandas import DataFrame
from regmod.composite_models import NodeModel, Link, CompositeModel


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

    def get_data(self, col_label: str = None) -> DataFrame:
        df = self.models[-1].get_data()
        if col_label is not None:
            df[col_label] = self.name
        return df

    def set_data(self, df: DataFrame, col_label: str = None):
        df = self.subset_df(df, col_label=col_label)
        return self.models[0].set_data(df)

    def fit(self, **fit_options):
        for i, model in enumerate(self.models):
            model.fit(**fit_options)
            df = model.predict(col_value=f"{self.name}_pred")
            if i < self.num_models - 1:
                df = self.models[i + 1].add_offset(
                    df, col_value=f"{self.name}_pred"
                )
                self.models[i + 1].set_data(df)

    def predict(self,
                df: DataFrame = None,
                col_value: str = None,
                col_label: str = None):
        if df is None:
            df = self.get_data()
        df = self.subset_df(df, col_label)
        col_value = f"{self.name}_pred" if col_value is None else col_value
        for i, model in enumerate(self.models):
            df = model.predict(df, col_value=col_value)
            if i < self.num_models - 1:
                df = self.models[i + 1].add_offset(
                    df, col_value=f"{self.name}_pred"
                )
        return df
