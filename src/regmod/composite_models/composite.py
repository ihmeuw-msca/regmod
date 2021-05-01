"""
Composite Model
"""
from typing import Dict, List
from regmod.composite_models import NodeModel, Link


class CompositeModel(NodeModel):
    """
    Composite Model, abstract behavior of group model.
    """

    def __init__(self,
                 name: str,
                 models: List[NodeModel],
                 links: List[Link] = None):
        if not all(isinstance(model, NodeModel) for model in models):
            raise TypeError("Models must be instances of NodeModel.")

        self.models = models
        self.model_dict = {model.name: model for model in self.models}
        model_names = [model.name for model in self.models]

        df = {model.name: model.get_data()
              for model in self.models}
        super().__init__(name, df)

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

    # pylint: disable=arguments-differ
    def set_data(self, dfs):
        for name, df in dfs.items():
            self.model_dict[name].set_data(df)

    def get_posterior(self) -> Dict:
        return {
            model.name: model.get_posterior()
            for model in self.models
        }

    def set_prior(self, priors):
        for name, prior in priors.items():
            self.model_dict[name].set_prior(prior)

    def fit(self, **fit_options):
        for model in self.models:
            model.fit(**fit_options)

    def predict(self, df):
        return {
            model_name: model.predict(df[model_name])
            for model_name, model in self.model_dict.items()
        }
