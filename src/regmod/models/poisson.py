"""
Poisson Model
"""
from typing import List, Union

import numpy as np
from regmod.data import Data
from regmod.function import SmoothFunction
from regmod.parameter import Parameter
from regmod.variable import Variable

from .model import Model


class PoissonModel(Model):
    def __init__(self, data: Data, variables: List[Variable],
                 inv_link: Union[str, SmoothFunction] = "exp",
                 use_offset: bool = False):
        lam = Parameter(name="lam",
                        variables=variables,
                        inv_link=inv_link,
                        use_offset=use_offset)
        assert all(data.obs >= 0), \
            "Poisson model requires observations to be non-negagive."
        super().__init__(data, [lam])

    def nll(self, params: List[np.ndarray]) -> np.ndarray:
        return self.data.weights*(params[0] - self.data.obs*np.log(params[0]))

    def dnll(self, params: List[np.ndarray]) -> List[np.ndarray]:
        return [self.data.weights*(1.0 - self.data.obs/params[0])]

    def d2nll(self, params: List[np.ndarray]) -> List[List[np.ndarray]]:
        return [[self.data.weights*self.data.obs/params[0]**2]]

    def __repr__(self) -> str:
        return f"PoissonModel(num_obs={self.data.num_obs}, num_params={self.num_params}, size={self.size})"
