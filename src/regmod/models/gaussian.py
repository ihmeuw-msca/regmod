"""
Gaussian Model
"""
from typing import List

import numpy as np

from .model import Model


class GaussianModel(Model):
    param_names = ("mu",)
    default_param_specs = {"mu": {"inv_link": "identity"}}

    def nll(self, params: List[np.ndarray]) -> np.ndarray:
        return 0.5*self.data.weights*(params[0] - self.data.obs)**2

    def dnll(self, params: List[np.ndarray]) -> List[np.ndarray]:
        return [self.data.weights*(params[0] - self.data.obs)]

    def d2nll(self, params: List[np.ndarray]) -> List[np.ndarray]:
        return [[self.data.weights]]

    def __repr__(self) -> str:
        return f"LinearModel(num_obs={self.data.num_obs}, num_params={self.num_params}, size={self.size})"
