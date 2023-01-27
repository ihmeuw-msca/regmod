"""
Negative Binomial Model
"""
# pylint: disable=no-name-in-module
from typing import List, Tuple

import numpy as np
import pandas as pd
from numpy import ndarray
from scipy.special import digamma, loggamma, polygamma
from scipy.stats import nbinom

from .model import Model


class NegativeBinomialModel(Model):
    param_names: List[str] = ("n", "p")
    default_param_specs = {"n": {"inv_link": "exp"},
                           "p": {"inv_link": "expit"}}

    def attach_df(self, df: pd.DataFrame):
        super().attach_df(df)
        if not np.all(self.y >= 0):
            raise ValueError("Negative-Binomial model requires observations to be non-negative.")

    def nll(self, params: List[ndarray]) -> ndarray:
        return -(loggamma(params[0] + self.y) -
                 loggamma(params[0]) +
                 self.y*np.log(1 - params[1]) +
                 params[0]*np.log(params[1]))

    def dnll(self, params: List[ndarray]) -> List[ndarray]:
        return [-(digamma(params[0] + self.y) - digamma(params[0]) + np.log(params[1])),
                self.y/(1 - params[1]) - params[0]/params[1]]

    def d2nll(self, params: List[ndarray]) -> List[List[ndarray]]:
        return [[polygamma(1, params[0]) - polygamma(1, params[0] + self.y), -1/params[1]],
                [-1/params[1], params[0]/params[1]**2 + self.y/(1 - params[1])**2]]

    def get_ui(self, params: List[ndarray], bounds: Tuple[float, float]) -> np.ndarray:
        n = params[0]
        p = params[1]
        return [nbinom.ppf(bounds[0], n=n, p=p),
                nbinom.ppf(bounds[1], n=n, p=p)]
