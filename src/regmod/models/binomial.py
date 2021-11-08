"""
Binomial Model
"""
from typing import List, Tuple

import numpy as np
from numpy import ndarray
from scipy.stats import binom
from regmod.data import Data

from .model import Model


class BinomialModel(Model):
    param_names = ("p",)
    default_param_specs = {"p": {"inv_link": "expit"}}

    def __init__(self, data: Data, **kwargs):
        if not np.all((data.obs >= 0) & (data.obs <= 1)):
            raise ValueError("Binomial model requires observations to be "
                             "between zero and one.")
        super().__init__(data, **kwargs)

    def nll(self, params: List[ndarray]) -> ndarray:
        return -(self.data.obs*np.log(params[0]) + (1 - self.data.obs)*np.log(1.0 - params[0]))

    def dnll(self, params: List[ndarray]) -> List[ndarray]:
        return [-(self.data.obs/params[0] - (1 - self.data.obs)/(1.0 - params[0]))]

    def d2nll(self, params: List[ndarray]) -> List[List[ndarray]]:
        return [[self.data.obs/params[0]**2 + (1 - self.data.obs)/(1.0 - params[0])**2]]

    def get_ui(self, params: List[ndarray], bounds: Tuple[float, float]) -> ndarray:
        p = params[0]
        n = self.obs_sample_sizes
        return [binom.ppf(bounds[0], n=n, p=p),
                binom.ppf(bounds[1], n=n, p=p)]
