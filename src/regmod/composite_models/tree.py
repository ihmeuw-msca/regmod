"""
Tree Model
"""
from typing import Dict, List

import numpy as np
from pandas import DataFrame

from regmod.data import Data
from regmod.variable import Variable
from regmod.composite_models.base import BaseModel
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
            for sub_node in node.children:
                self.model_dict[sub_node.name].set_prior(prior, mask)
                self._fit(sub_node.name, **fit_options)

    def fit(self, **fit_options):
        self._fit(self.nodes[0].name, **fit_options)

    @classmethod
    def get_simple_model(cls,
                         name: str,
                         df: DataFrame,
                         id_cols: List[str],
                         data: Data,
                         variables: List[Variable],
                         root_name: str = "Global",
                         mtype: str = "gaussian",
                         var_masks: Dict[str, float] = None,
                         lvl_masks: List[float] = None,
                         **param_specs) -> "TreeModel":

        # check data before create model
        data.attach_df(df)
        for var in variables:
            var.check_data(data)
        data.detach_df()

        def get_model(node: TreeNode,
                      df_group: DataFrame,
                      data: Data = data,
                      variables: List[Variable] = variables,
                      mtype: str = mtype):
            data.attach_df(df_group)
            model = BaseModel(node.name, data, variables,
                              mtype=mtype, **param_specs)
            data.detach_df()
            return model

        root_node = TreeNode.from_dataframe(df,
                                            id_cols,
                                            root_name=root_name,
                                            container_fun=get_model)

        models = []
        for node in root_node.tree:
            models.append(node.container)
            node.container = None

        masks = {}
        final_var_masks = {var.name: np.ones(var.size) for var in variables}
        final_lvl_masks = [1.0]*root_node.lower_rank
        if var_masks is not None:
            final_var_masks.update(var_masks)
        if lvl_masks is not None:
            final_lvl_masks = lvl_masks + final_lvl_masks[len(lvl_masks):]

        for node in root_node.tree:
            if node.is_leaf:
                continue
            masks[node.name] = {
                var_name: var_prior*final_lvl_masks[node.level]
                for var_name, var_prior in final_var_masks.items()
            }

        return cls(name, models, root_node, masks)
