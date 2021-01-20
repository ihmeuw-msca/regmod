"""
Negative Binomial Model
"""
# pylint: disable=no-name-in-module
from typing import List

import numpy as np
from scipy.special import digamma, loggamma, polygamma
from regmod.data import Data

from .model import Model


class NegativeBinomialModel(Model):
    param_names: List[str] = ("r", "p")
    default_param_specs = {"r": {"inv_link": "exp"},
                           "p": {"inv_link": "expit"}}

    def __init__(self, data: Data, **kwargs):
        if not np.all(data.obs >= 0):
            raise ValueError("Negative-Binomial model requires observations to be non-negative.")
        super().__init__(data, **kwargs)

    def nll(self, params: List[np.ndarray]) -> np.ndarray:
        return -self.data.weights*(loggamma(params[0] + self.data.obs) -
                                   loggamma(params[0]) +
                                   self.data.obs*np.log(params[1]) +
                                   params[0]*np.log(1 - params[1]))

    def dnll(self, params: List[np.ndarray]) -> List[np.ndarray]:
        return [-self.data.weights*(digamma(params[0] + self.data.obs) -
                                    digamma(params[0]) +
                                    np.log(1 - params[1])),
                -self.data.weights*(self.data.obs/params[1] -
                                    params[0]/(1 - params[1]))]

    def d2nll(self, params: List[np.ndarray]) -> List[List[np.ndarray]]:
        return [[self.data.weights*(polygamma(1, params[0]) -
                                    polygamma(1, params[0] + self.data.obs)),
                 self.data.weights/(1 - params[1])],
                [self.data.weights/(1 - params[1]),
                 self.data.weights*(params[0]/(1 - params[1])**2 +
                                    self.data.obs/params[1]**2)]]

    def __repr__(self) -> str:
        return f"NegativeBinomialModel(num_obs={self.data.num_obs}, num_params={self.num_params}, size={self.size})"
