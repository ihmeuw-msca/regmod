"""
Pogit Model
"""
from typing import List

import numpy as np
from regmod.data import Data

from .model import Model


class PogitModel(Model):
    param_names = ("p", "lam")
    default_param_specs = {"p": {"inv_link": "expit"},
                           "lam": {"inv_link": "exp"}}

    def __init__(self, data: Data, **kwargs):
        if not all(data.obs >= 0):
            raise ValueError("Pogit model requires observations to be non-negagive.")
        super().__init__(data, **kwargs)

    def nll(self, params: List[np.ndarray]) -> np.ndarray:
        mean = params[0]*params[1]
        return self.data.weights*(mean - self.data.obs*np.log(mean))

    def dnll(self, params: List[np.ndarray]) -> List[np.ndarray]:
        return [self.data.weights*(params[1] - self.data.obs/params[0]),
                self.data.weights*(params[0] - self.data.obs/params[1])]

    def d2nll(self, params: List[np.ndarray]) -> List[List[np.ndarray]]:
        return [[self.data.weights*self.data.obs/params[0]**2,
                 self.data.weights],
                [self.data.weights,
                 self.data.weights*self.data.obs/params[1]**2]]

    def __repr__(self) -> str:
        return f"PogitModel(num_obs={self.data.num_obs}, num_params={self.num_params}, size={self.size})"
