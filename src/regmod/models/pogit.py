"""
Pogit Model
"""

import numpy as np
from scipy.stats import poisson

from regmod._typing import DataFrame, NDArray

from .model import Model


class PogitModel(Model):
    param_names = ("p", "lam")
    default_param_specs = {"p": {"inv_link": "expit"}, "lam": {"inv_link": "exp"}}

    def attach_df(self, df: DataFrame):
        super().attach_df(df)
        if not all(self.y >= 0):
            raise ValueError("Pogit model requires observations to be non-negagive.")

    def nll(self, params: list[NDArray]) -> NDArray:
        mean = params[0] * params[1]
        return mean - self.y * np.log(mean)

    def dnll(self, params: list[NDArray]) -> list[NDArray]:
        return [params[1] - self.y / params[0], params[0] - self.y / params[1]]

    def d2nll(self, params: list[NDArray]) -> list[list[NDArray]]:
        ones = np.ones(self.df.shape[0])
        return [[self.y / params[0] ** 2, ones], [ones, self.y / params[1] ** 2]]

    def get_ui(self, params: list[NDArray], bounds: tuple[float, float]) -> NDArray:
        mean = params[0] * params[1]
        return [poisson.ppf(bounds[0], mu=mean), poisson.ppf(bounds[1], mu=mean)]
