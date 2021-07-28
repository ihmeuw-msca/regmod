"""
Chain Model
"""
from typing import Dict, List

from pandas import DataFrame
from regmod.composite_models.base import BaseModel

from regmod.composite_models.composite import CompositeModel
from regmod.composite_models.interface import NodeModel


class ChainModel(CompositeModel):
    """
    Chain Model with sequential model fitting.
    """

    def _fit(self, model: NodeModel, **fit_options):
        model.fit(**fit_options)
        if len(model.children.named_lists[0]) > 0:
            sub_model = model.children.named_lists[0][0]
            sub_model.set_data(model.predict(sub_model.get_data()))
            self._fit(sub_model, **fit_options)

    def fit(self, **fit_options):
        if len(self.children.named_lists[1]) != 1:
            raise ValueError(f"{type(self).__name__} must only have one "
                             "computational tree.")
        if len(self.children.named_lists[1][0].get_leafs(0)) != 1:
            raise ValueError(f"{type(self).__name__} computational tree must be"
                             " chain.")
        self._fit(self.children.named_lists[1][0], **fit_options)

    @classmethod
    def get_simple_chain(cls, name: str, *args, **kwargs) -> "ChainModel":
        return cls(name, models=[get_simple_basechain(*args, **kwargs)])


def get_simple_basechain(df: DataFrame,
                         model_specs: List[Dict]) -> BaseModel:
    if len(model_specs) == 0:
        raise ValueError("Must provide specifications of BaseModel.")

    model = BaseModel(**model_specs[0])
    model.set_data(df)

    if len(model_specs) > 1:
        model.append(get_simple_basechain(df, model_specs[1:]))

    return model
