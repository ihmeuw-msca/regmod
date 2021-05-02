"""
Tree Model
"""
from typing import Dict, List
import numpy as np
from regmod.composite_models import NodeModel, Link, CompositeModel


class TreeModel(CompositeModel):
    """
    Tree Model with hierarchy structure. This model is also called cascade.
    """

    def __init__(self,
                 name: str,
                 models: List[NodeModel],
                 links: List[Link],
                 masks: Dict):

        root_links = [link for link in links if link.is_root]
        if len(root_links) != 1:
            raise ValueError("Links must have one and only one root.")
        upper_ranks = np.array([link.upper_rank for link in links])
        ordered_links = [links[i] for i in np.argsort(upper_ranks)]

        super().__init__(name, models, ordered_links)
        self.models = [self.model_dict[link.name] for link in self.links]

        if not all(mask_name in self.model_dict for mask_name in masks):
            raise ValueError()
        self.masks = masks

    def _fit(self, model_name: str, **fit_options):
        model = self.model_dict[model_name]
        link = self.link_dict[model_name]
        mask = None if model_name not in self.masks else self.masks[model_name]

        model.fit(**fit_options)

        if not link.is_leaf:
            prior = model.get_posterior()
            for lower_link in link.lower_links:
                lower_model_name = lower_link.name
                self.model_dict[lower_model_name].set_prior(prior, mask)
                self._fit(lower_model_name, **fit_options)

    def fit(self, **fit_options):
        self._fit(self.links[0].name, **fit_options)
