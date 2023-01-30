"""
Weibull Model
"""
from typing import List, Tuple

import numpy as np
import pandas as pd
from numpy import ndarray
from scipy.stats import weibull_min

from .model import Model


class WeibullModel(Model):
    param_names = ("b", "k")
    default_param_specs = {"b": {"inv_link": "exp"}, "k": {"inv_link": "exp"}}

    def attach_df(self, df: pd.DataFrame):
        super().attach_df(df)
        if not all(self.y > 0):
            raise ValueError("Weibull model requires observations to be positive.")

    def nll(self, params: List[ndarray]) -> ndarray:
        t = self.y
        ln_t = np.log(t)
        return params[0]*(t**params[1]) - (params[1] - 1)*ln_t - np.log(params[0]) - np.log(params[1])

    def dnll(self, params: List[ndarray]) -> List[ndarray]:
        t = self.y
        ln_t = np.log(t)
        return [t**params[1] - 1/params[0],
                ln_t*params[0]*(t**params[1]) - ln_t - 1/params[1]]

    def d2nll(self, params: List[ndarray]) -> List[List[ndarray]]:
        t = self.y
        ln_t = np.log(t)
        return [[1/params[0]**2, ln_t*(t**params[1])],
                [ln_t*(t**params[1]), 1/params[1]**2 + params[0]*(ln_t**2)*(t**params[1])]]

    def get_ui(self, params: List[ndarray], bounds: Tuple[float, float]) -> ndarray:
        scale = 1 / params[0]**(1 / params[1])
        return [weibull_min.ppf(bounds[0], c=params[1], scale=scale),
                weibull_min.ppf(bounds[1], c=params[1], scale=scale)]
