"""
Chain Model
"""
from typing import List
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

    def get_data(self):
        return self.models[-1].get_data()

    # pylint: disable=arguments-differ
    def set_data(self, df):
        return self.models[0].set_data(df)

    def fit(self, **fit_options):
        for i, model in enumerate(self.models):
            model.fit(**fit_options)
            if i < self.num_models - 1:
                df = model.predict()
                df = self.models[i + 1].set_offset(df, f"{model.name}_pred")
                self.models[i + 1].set_data(df)

    def predict(self, df=None):
        for i, model in enumerate(self.models):
            df = model.predict(df)
            if i < self.num_models - 1:
                df = self.models[i + 1].set_offset(df, f"{model.name}_pred")
        return df
