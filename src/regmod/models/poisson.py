"""
Poisson Model
"""
from typing import List, Tuple

import numpy as np
from numpy import ndarray
from scipy.stats import poisson
from regmod.data import Data

from .model import Model


class PoissonModel(Model):
    param_names = ("lam",)
    default_param_specs = {"lam": {"inv_link": "exp"}}

    def __init__(self, data: Data, **kwargs):
        if not all(data.obs >= 0):
            raise ValueError("Poisson model requires observations to be non-negagive.")
        super().__init__(data, **kwargs)

    def nll(self, params: List[ndarray]) -> ndarray:
        return params[0] - self.data.obs*np.log(params[0])

    def dnll(self, params: List[ndarray]) -> List[ndarray]:
        return [1.0 - self.data.obs/params[0]]

    def d2nll(self, params: List[ndarray]) -> List[List[ndarray]]:
        return [[self.data.obs/params[0]**2]]

    def get_ui(self, params: List[ndarray], bounds: Tuple[float, float]) -> ndarray:
        mean = params[0]
        return [poisson.ppf(bounds[0], mu=mean),
                poisson.ppf(bounds[1], mu=mean)]

    def __repr__(self) -> str:
        return f"PoissonModel(num_obs={self.data.num_obs}, num_params={self.num_params}, size={self.size})"
