"""
Pogit Model
"""
import numpy as np
import pandas as pd
from numpy import ndarray
from scipy.stats import poisson

from .model import Model


class PogitModel(Model):
    param_names = ("p", "lam")
    default_param_specs = {"p": {"inv_link": "expit"}, "lam": {"inv_link": "exp"}}

    def _validate_data(self, df: pd.DataFrame, fit: bool = True):
        super()._validate_data(df, fit)

        if fit and not all(df[self.y] >= 0):
            raise ValueError("Pogit model requires observations to be non-negagive.")

    def nll(self, data: dict, params: list[ndarray]) -> ndarray:
        mean = params[0] * params[1]
        return mean - data["y"] * np.log(mean)

    def dnll(self, data: dict, params: list[ndarray]) -> list[ndarray]:
        return [
            params[1] - data["y"] / params[0],
            params[0] - data["y"] / params[1],
        ]

    def d2nll(self, data: dict, params: list[ndarray]) -> list[list[ndarray]]:
        ones = np.ones(data["y"].size)
        return [
            [data["y"] / params[0] ** 2, ones],
            [ones, data["y"] / params[1] ** 2],
        ]

    def get_ui(self, params: list[ndarray], bounds: tuple[float, float]) -> ndarray:
        mean = params[0] * params[1]
        return [poisson.ppf(bounds[0], mu=mean), poisson.ppf(bounds[1], mu=mean)]
