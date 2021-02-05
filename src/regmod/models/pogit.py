"""
Pogit Model
"""
from typing import List, Tuple

import numpy as np
from numpy import ndarray
from scipy.stats import poisson
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

    def nll(self, params: List[ndarray]) -> ndarray:
        mean = params[0]*params[1]
        return mean - self.data.obs*np.log(mean)

    def dnll(self, params: List[ndarray]) -> List[ndarray]:
        return [params[1] - self.data.obs/params[0],
                params[0] - self.data.obs/params[1]]

    def d2nll(self, params: List[ndarray]) -> List[List[ndarray]]:
        ones = np.ones(self.data.num_obs)
        return [[self.data.obs/params[0]**2, ones],
                [ones, self.data.obs/params[1]**2]]

    def get_ui(self, params: List[ndarray], bounds: Tuple[float, float]) -> ndarray:
        mean = params[0]*params[1]
        return [poisson.ppf(bounds[0], mu=mean),
                poisson.ppf(bounds[1], mu=mean)]

    def __repr__(self) -> str:
        return f"PogitModel(num_obs={self.data.num_obs}, num_params={self.num_params}, size={self.size})"
