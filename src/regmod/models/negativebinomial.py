"""
Negative Binomial Model
"""
# pylint: disable=no-name-in-module
from typing import List, Tuple

import numpy as np
from numpy import ndarray
from scipy.stats import nbinom
from scipy.special import digamma, loggamma, polygamma
from regmod.data import Data

from .model import Model


class NegativeBinomialModel(Model):
    param_names: List[str] = ("n", "p")
    default_param_specs = {"n": {"inv_link": "exp"},
                           "p": {"inv_link": "expit"}}

    def __init__(self, data: Data, **kwargs):
        if not np.all(data.obs >= 0):
            raise ValueError("Negative-Binomial model requires observations to be non-negative.")
        super().__init__(data, **kwargs)

    def nll(self, params: List[ndarray]) -> ndarray:
        return -(loggamma(params[0] + self.data.obs) -
                 loggamma(params[0]) +
                 self.data.obs*np.log(1 - params[1]) +
                 params[0]*np.log(params[1]))

    def dnll(self, params: List[ndarray]) -> List[ndarray]:
        return [-(digamma(params[0] + self.data.obs) - digamma(params[0]) + np.log(params[1])),
                self.data.obs/(1 - params[1]) - params[0]/params[1]]

    def d2nll(self, params: List[ndarray]) -> List[List[ndarray]]:
        return [[polygamma(1, params[0]) - polygamma(1, params[0] + self.data.obs), -1/params[1]],
                [-1/params[1], params[0]/params[1]**2 + self.data.obs/(1 - params[1])**2]]

    def get_ui(self, params: List[ndarray], bounds: Tuple[float, float]) -> np.ndarray:
        n = params[0]
        p = params[1]
        return [nbinom.ppf(bounds[0], n=n, p=p),
                nbinom.ppf(bounds[1], n=n, p=p)]

    def __repr__(self) -> str:
        return f"NegativeBinomialModel(num_obs={self.data.num_obs}, num_params={self.num_params}, size={self.size})"
