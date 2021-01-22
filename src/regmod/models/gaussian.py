"""
Gaussian Model
"""
from typing import List, Tuple

import numpy as np
from numpy import ndarray
from scipy.stats import norm

from .model import Model


class GaussianModel(Model):
    param_names = ("mu",)
    default_param_specs = {"mu": {"inv_link": "identity"}}

    def nll(self, params: List[ndarray]) -> ndarray:
        return 0.5*(params[0] - self.data.obs)**2

    def dnll(self, params: List[ndarray]) -> List[ndarray]:
        return [params[0] - self.data.obs]

    def d2nll(self, params: List[ndarray]) -> List[ndarray]:
        return [[np.ones(self.data.num_obs)]]

    def get_ui(self, params: List[ndarray], bounds: Tuple[float, float]) -> ndarray:
        mean = params[0]
        sd = 1.0/np.sqrt(self.data.weights)
        return [norm.ppf(bounds[0], loc=mean, scale=sd),
                norm.ppf(bounds[1], loc=mean, scale=sd)]

    def __repr__(self) -> str:
        return f"LinearModel(num_obs={self.data.num_obs}, num_params={self.num_params}, size={self.size})"
