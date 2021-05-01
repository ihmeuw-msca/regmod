"""
Composite Model
"""
from typing import List
from regmod.composite_models import NodeModel, Link


class CompositeModel(NodeModel):
    """
    Composite Model, abstract behavior of group model.
    """

    def __init__(self,
                 name: str,
                 models: List[NodeModel],
                 links: List[Link]):
        if not all(isinstance(model, NodeModel) for model in models):
            raise TypeError("Models must be instances of NodeModel.")
        if not all(isinstance(link, Link) for link in links):
            raise TypeError("Links must be instances of Link.")

        self.models = models
        self.model_dict = {model.name: model for model in self.models}

        df = {model.name: model.get_data()
              for model in self.models}
        super().__init__(name, df)

        model_names = [model.name for model in self.models]
        link_names = [link.name for link in links]
        if set(model_names) != set(link_names):
            raise ValueError("Models and links not match.")
        self.links = links

    def __getitem__(self, model_name: str) -> NodeModel:
        return self.models[model_name]
