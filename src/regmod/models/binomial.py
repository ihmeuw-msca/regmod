"""
Binomial Model
"""
from typing import List, Union

import numpy as np
from regmod.data import Data
from regmod.function import SmoothFunction
from regmod.parameter import Parameter
from regmod.variable import Variable

from .model import Model


class BinomialModel(Model):
    def __init__(self, data: Data, variables: List[Variable],
                 inv_link: Union[str, SmoothFunction] = "expit",
                 use_offset: bool = False):
        p = Parameter(name="p",
                      variables=variables,
                      inv_link=inv_link,
                      use_offset=use_offset)
        assert np.all(data.obs >= 0), \
            "Binomial model requires observations to be non-negative."
        assert len(data.col_obs) == 2, \
            "Binomial model need 2 columns of observations, one for number of events, one for sample size."
        self.obs_1s = data.get_cols(data.col_obs[0])
        self.obs_sample_sizes = data.get_cols(data.col_obs[1])
        assert all(self.obs_1s <= self.obs_sample_sizes), \
            "Binomial model requires number of events less or equal than sample size."
        self.obs_0s = self.obs_sample_sizes - self.obs_1s
        super().__init__(data, [p])

    def nll(self, params: List[np.ndarray]) -> np.ndarray:
        return -self.data.weights*(self.obs_1s*np.log(params[0]) +
                                   self.obs_0s*np.log(1.0 - params[0]))

    def dnll(self, params: List[np.ndarray]) -> List[np.ndarray]:
        return [-self.data.weights*(self.obs_1s/params[0] -
                                    self.obs_0s/(1.0 - params[0]))]

    def d2nll(self, params: List[np.ndarray]) -> List[List[np.ndarray]]:
        return [[self.data.weights*(self.obs_1s/params[0]**2 +
                                    self.obs_0s/(1.0 - params[0])**2)]]

    def __repr__(self) -> str:
        return f"BinomialModel(num_obs={self.data.num_obs}, num_params={self.num_params}, size={self.size})"
