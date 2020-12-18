"""
Poisson Model
"""
from typing import List

import numpy as np
from regmod.data import Data

from .model import Model


class PoissonModel(Model):
    param_names = ("lam",)
    default_param_specs = {"lam": {"inv_link": "exp"}}

    def __init__(self, data: Data, **kwargs):
        if not all(data.obs >= 0):
            raise ValueError("Poisson model requires observations to be non-negagive.")
        super().__init__(data, **kwargs)

    def nll(self, params: List[np.ndarray]) -> np.ndarray:
        return self.data.weights*(params[0] - self.data.obs*np.log(params[0]))

    def dnll(self, params: List[np.ndarray]) -> List[np.ndarray]:
        return [self.data.weights*(1.0 - self.data.obs/params[0])]

    def d2nll(self, params: List[np.ndarray]) -> List[List[np.ndarray]]:
        return [[self.data.weights*self.data.obs/params[0]**2]]

    def __repr__(self) -> str:
        return f"PoissonModel(num_obs={self.data.num_obs}, num_params={self.num_params}, size={self.size})"
