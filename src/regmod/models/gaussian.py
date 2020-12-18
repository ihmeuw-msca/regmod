"""
Gaussian Model
"""
from typing import List, Union

import numpy as np
from regmod.data import Data
from regmod.function import SmoothFunction
from regmod.parameter import Parameter
from regmod.variable import Variable

from .model import Model


class GaussianModel(Model):
    def __init__(self, data: Data, variables: List[Variable],
                 inv_link: Union[str, SmoothFunction] = "identity",
                 use_offset: bool = False):
        mu = Parameter(name="mu",
                       variables=variables,
                       inv_link=inv_link,
                       use_offset=use_offset)
        super().__init__(data, [mu])

    def nll(self, params: List[np.ndarray]) -> np.ndarray:
        return 0.5*self.data.weights*(params[0] - self.data.obs)**2

    def dnll(self, params: List[np.ndarray]) -> List[np.ndarray]:
        return [self.data.weights*(params[0] - self.data.obs)]

    def d2nll(self, params: List[np.ndarray]) -> List[np.ndarray]:
        return [[self.data.weights]]

    def __repr__(self) -> str:
        return f"LinearModel(num_obs={self.data.num_obs}, num_params={self.num_params}, size={self.size})"
