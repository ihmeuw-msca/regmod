"""
Tree Model
"""
from typing import Dict, List

from regmod.composite_models.composite import CompositeModel
from regmod.composite_models.interface import ModelInterface
from regmod.composite_models.treenode import TreeNode


class TreeModel(CompositeModel):
    """
    Tree Model with hierarchy structure. This model is also called cascade.
    """

    def __init__(self,
                 name: str,
                 models: List[ModelInterface],
                 root_node: TreeNode,
                 masks: Dict):

        super().__init__(name, models, root_node.lower_nodes)
        self.models = [self.model_dict[node.name] for node in self.nodes]

        if not all(mask_name in self.model_dict for mask_name in masks):
            raise ValueError()
        self.masks = masks

    def _fit(self, model_name: str, **fit_options):
        model = self.model_dict[model_name]
        node = self.node_dict[model_name]
        mask = None if model_name not in self.masks else self.masks[model_name]

        model.fit(**fit_options)

        if not node.is_leaf:
            prior = model.get_posterior()
            for sub_node in node.sub_nodes:
                sub_model_name = sub_node.name
                self.model_dict[sub_model_name].set_prior(prior, mask)
                self._fit(sub_model_name, **fit_options)

    def fit(self, **fit_options):
        self._fit(self.nodes[0].name, **fit_options)
