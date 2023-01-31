"""
Pogit Model
"""
from typing import List, Tuple

import numpy as np
import pandas as pd
from numpy import ndarray
from scipy.stats import poisson

from .model import Model


class PogitModel(Model):
    param_names = ("p", "lam")
    default_param_specs = {"p": {"inv_link": "expit"},
                           "lam": {"inv_link": "exp"}}

    def _attach(self, df: pd.DataFrame, require_y: bool = True):
        super()._attach(df, require_y=require_y)
        if require_y and not all(self._data["y"] >= 0):
            raise ValueError(
                "Pogit model requires observations to be non-negagive."
            )

    def nll(self, params: List[ndarray]) -> ndarray:
        mean = params[0]*params[1]
        return mean - self._data["y"]*np.log(mean)

    def dnll(self, params: List[ndarray]) -> List[ndarray]:
        return [params[1] - self._data["y"]/params[0],
                params[0] - self._data["y"]/params[1]]

    def d2nll(self, params: List[ndarray]) -> List[List[ndarray]]:
        ones = np.ones(self._data["y"].size)
        return [[self._data["y"]/params[0]**2, ones],
                [ones, self._data["y"]/params[1]**2]]

    def get_ui(self, params: List[ndarray], bounds: Tuple[float, float]) -> ndarray:
        mean = params[0]*params[1]
        return [poisson.ppf(bounds[0], mu=mean),
                poisson.ppf(bounds[1], mu=mean)]
