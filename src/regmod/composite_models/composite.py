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

    def set_data(self, df: Dict):
        for name in df.keys():
            self.model_dict[name].set_data(df[name])

    def set_offset(self, df: Dict, col: str):
        return {
            name: self.model_dict[name].set_offset(df[name], col)
            for name in df.keys()
        }

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

    def predict(self, df: Dict = None, col: str = None):
        if df is None:
            df = {model.name: None for model in self.models}
        return {
            name: model.predict(df=df[name], col=col)
            for name, model in self.model_dict.items()
        }
