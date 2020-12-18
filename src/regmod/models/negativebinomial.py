"""
Negative Binomial Model
"""
from typing import List, Union, Dict

import numpy as np
from scipy.special import digamma, loggamma, polygamma
from regmod.data import Data
from regmod.function import SmoothFunction
from regmod.parameter import Parameter
from regmod.variable import Variable

from .model import Model


class NegativeBinomialModel(Model):
    def __init__(self,
                 data: Data,
                 variables: Dict[str, List[Variable]],
                 inv_link: Dict[str, Union[str, SmoothFunction]] = {"r": "exp", "p": "expit"},
                 use_offset: Dict[str, bool] = {"r": False, "p": False}):
        if not all([param_name in variables for param_name in ["r", "p"]]):
            raise ValueError(f"'r' and 'p' must be keys in `variables`.")
        assert np.all(data.obs >= 0), \
            "Negative-Binomial model requires observations to be non-negative."

        params = [
            Parameter(name=param_name,
                      variables=variables[param_name],
                      inv_link=inv_link[param_name],
                      use_offset=use_offset[param_name])
            for param_name in ["r", "p"]
        ]
        super().__init__(data, params)

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
