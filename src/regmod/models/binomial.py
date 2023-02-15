"""
Binomial Model
"""
from typing import Callable

import numpy as np
import pandas as pd
from numpy.typing import NDArray
from scipy.stats import binom

from regmod.optimizer import msca_optimize

from .data import parse_to_msca
from .model import Model


class BinomialModel(Model):
    param_names = ("p",)
    default_param_specs = {"p": {"inv_link": "expit"}}

    def _validate_data(self, df: pd.DataFrame, require_y: bool = True):
        super()._validate_data(df, require_y)
        if require_y and not np.all((df[self.y] >= 0) & (df[self.y] <= 1)):
            raise ValueError(
                "Binomial model requires observations to be between zero and one."
            )

    def _parse(self, df: pd.DataFrame, require_y: bool = True):
        self._validate_data(df, require_y=require_y)
        return parse_to_msca(df, self.y, self.params, self.weights, for_fit=require_y)

    def objective(self, data: dict, coefs: NDArray) -> float:
        """Objective function.
        Parameters
        ----------
        coefs : NDArray
            Given coefficients.
        Returns
        -------
        float
            Objective value.
        """
        inv_link = self.params[0].inv_link
        lin_param = self.params[0].get_lin_param(
            coefs, data["offset"][0], mat=data["mat"][0]
        )
        param = inv_link.fun(lin_param)

        weights = data["weights"] * data["trim_weights"]
        obj_param = -weights * (
            data["y"] * np.log(param) + (1 - data["y"]) * np.log(1 - param)
        )
        return obj_param.sum() + self.objective_from_gprior(data, coefs)

    def gradient(self, data: dict, coefs: NDArray) -> NDArray:
        """Gradient function.

        Parameters
        ----------
        coefs : NDArray
            Given coefficients.

        Returns
        -------
        NDArray
            Gradient vector.
        """
        mat = data["mat"][0]
        inv_link = self.params[0].inv_link
        lin_param = self.params[0].get_lin_param(
            coefs, data["offset"][0], mat=data["mat"][0]
        )
        param = inv_link.fun(lin_param)
        dparam = inv_link.dfun(lin_param)

        weights = data["weights"] * data["trim_weights"]
        grad_param = weights * ((param - data["y"]) / (param * (1 - param)) * dparam)

        return mat.T.dot(grad_param) + self.gradient_from_gprior(data, coefs)

    def hessian(self, data: dict, coefs: NDArray) -> NDArray:
        """Hessian function.

        Parameters
        ----------
        coefs : NDArray
            Given coefficients.

        Returns
        -------
        NDArray
            Hessian matrix.
        """
        mat = data["mat"][0]
        inv_link = self.params[0].inv_link
        lin_param = self.params[0].get_lin_param(
            coefs, data["offset"][0], mat=data["mat"][0]
        )
        param = inv_link.fun(lin_param)
        dparam = inv_link.dfun(lin_param)
        d2param = inv_link.d2fun(lin_param)

        weights = data["weights"] * data["trim_weights"]
        hess_param = weights * (
            (data["y"] / param**2 + (1 - data["y"]) / (1 - param) ** 2) * dparam**2
            + (param - data["y"]) / (param * (1 - param)) * d2param
        )

        scaled_mat = mat.scale_rows(hess_param)
        hess_mat = mat.T.dot(scaled_mat)
        hess_mat_gprior = type(hess_mat)(self.hessian_from_gprior(data))
        return hess_mat + hess_mat_gprior

    def jacobian2(self, data: dict, coefs: NDArray) -> NDArray:
        """Jacobian function.

        Parameters
        ----------
        coefs : NDArray
            Given coefficients.

        Returns
        -------
        NDArray
            Jacobian matrix.
        """
        mat = data["mat"][0]
        inv_link = self.params[0].inv_link
        lin_param = self.params[0].get_lin_param(
            coefs, data["offset"][0], mat=data["mat"][0]
        )
        param = inv_link.fun(lin_param)
        dparam = inv_link.dfun(lin_param)
        weights = data["weights"] * data["trim_weights"]
        grad_param = weights * ((param - data["y"]) / (param * (1 - param)) * dparam)
        jacobian = mat.T.scale_cols(grad_param)
        hess_mat_gprior = type(jacobian)(self.hessian_from_gprior(data))
        jacobian2 = jacobian.dot(jacobian.T) + hess_mat_gprior
        return jacobian2

    def fit(
        self, df: pd.DataFrame, optimizer: Callable = msca_optimize, **optimizer_options
    ):
        """Fit function.

        Parameters
        ----------
        optimizer : Callable, optional
            Model solver, by default scipy_optimize.
        """
        super().fit(df, optimizer=optimizer, **optimizer_options)

    def nll(self, data: dict, params: list[NDArray]) -> NDArray:
        return -(
            data["y"] * np.log(params[0]) + (1 - data["y"]) * np.log(1.0 - params[0])
        )

    def dnll(self, data: dict, params: list[NDArray]) -> list[NDArray]:
        return [-(data["y"] / params[0] - (1 - data["y"]) / (1.0 - params[0]))]

    def d2nll(self, data: dict, params: list[NDArray]) -> list[list[NDArray]]:
        return [[data["y"] / params[0] ** 2 + (1 - data["y"]) / (1.0 - params[0]) ** 2]]

    def get_ui(self, params: list[NDArray], bounds: tuple[float, float]) -> NDArray:
        p = params[0]
        n = self.y_sample_sizes
        return [binom.ppf(bounds[0], n=n, p=p), binom.ppf(bounds[1], n=n, p=p)]
