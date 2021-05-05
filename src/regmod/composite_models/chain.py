"""
Chain Model
"""
from typing import List

from pandas import DataFrame
from regmod.composite_models.composite import CompositeModel
from regmod.composite_models.node import NodeModel
from regmod.composite_models.treenode import TreeNode


class ChainModel(CompositeModel):
    """
    Chain Model with sequential model fitting.
    """

    def __init__(self,
                 name: str,
                 models: List[NodeModel],
                 root_node: TreeNode = None):

        if root_node is None:
            nodes = [TreeNode(models[0].name)]
            for model in models[1:]:
                nodes.append(nodes[-1] / model.name)
            root_node = nodes[0]

        if len(root_node.leafs) > 1:
            raise ValueError("Tree nodes must form a chain.")

        super().__init__(name, models, root_node.lower_nodes)
        self.models = [self.model_dict[node.name] for node in self.nodes]

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
