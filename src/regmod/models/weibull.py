"""
Weibull Model
"""
from typing import List, Tuple

import numpy as np
from numpy import ndarray
from scipy.stats import poisson
from regmod.data import Data

from .model import Model


class WeibullModel(Model):
    param_names = ("b", "k")
    default_param_specs = {"b": {"inv_link": "exp"}, "k": {"inv_link": "exp"}}

    def __init__(self, data: Data, **kwargs):
        if not all(data.obs > 0):
            raise ValueError("Weibull model requires observations to be positive.")
        super().__init__(data, **kwargs)

    def nll(self, params: List[ndarray]) -> ndarray:
        t = self.data.obs
        ln_t = np.log(t)
        return params[0]*(t**params[1]) - (params[1] - 1)*ln_t - np.log(params[0]) - np.log(params[1])

    def dnll(self, params: List[ndarray]) -> List[ndarray]:
        t = self.data.obs
        ln_t = np.log(t)
        return [t**params[1] - 1/params[0],
                ln_t*params[0]*(t**params[1]) - ln_t - 1/params[1]]

    def d2nll(self, params: List[ndarray]) -> List[List[ndarray]]:
        t = self.data.obs
        ln_t = np.log(t)
        return [[1/params[0]**2, ln_t*(t**params[1])],
                [ln_t*(t**params[1]), 1/params[1]**2 + params[0]*(ln_t**2)*(t**params[1])]]

    def get_ui(self, params: List[ndarray], bounds: Tuple[float, float]) -> ndarray:
        return NotImplemented

    def __repr__(self) -> str:
        return f"WeibullModel(num_obs={self.data.num_obs}, num_params={self.num_params}, size={self.size})"
