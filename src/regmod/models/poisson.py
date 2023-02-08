"""
Poisson Model
"""
from typing import Callable

import numpy as np
import pandas as pd
from numpy.typing import NDArray
from scipy.stats import poisson

from regmod.optimizer import msca_optimize

from .model import Model
from .utils import model_post_init


class PoissonModel(Model):
    param_names = ("lam",)
    default_param_specs = {"lam": {"inv_link": "exp"}}

    def _validate_data(self, df: pd.DataFrame, require_y: bool = True):
        super()._validate_data(df, require_y)
        if require_y and not all(df[self.y] >= 0):
            raise ValueError("Poisson model requires observations to be non-negagive.")

    def _parse(self, df: pd.DataFrame, require_y: bool = True) -> dict:
        data = super()._parse(df, require_y=require_y)
        data["mat"][0], data["cmat"], data["cvec"] = model_post_init(
            data["mat"][0],
            data["uvec"],
            data["linear_umat"],
            data["linear_uvec"],
        )
        return data

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

        weights = data["weights"] * self.trim_weights
        obj_param = weights * (param - data["y"] * np.log(param))
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

        weights = data["weights"] * self.trim_weights
        grad_param = weights * (1 - data["y"] / param) * dparam

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

        weights = data["weights"] * self.trim_weights
        hess_param = weights * (
            data["y"] / param**2 * dparam**2 + (1 - data["y"] / param) * d2param
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
        weights = data["weights"] * self.trim_weights
        grad_param = weights * (1.0 - data["y"] / param) * dparam
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
        return params[0] - data["y"] * np.log(params[0])

    def dnll(self, data: dict, params: list[NDArray]) -> list[NDArray]:
        return [1.0 - data["y"] / params[0]]

    def d2nll(self, data: dict, params: list[NDArray]) -> list[list[NDArray]]:
        return [[data["y"] / params[0] ** 2]]

    def get_ui(self, params: list[NDArray], bounds: tuple[float, float]) -> NDArray:
        mean = params[0]
        return [poisson.ppf(bounds[0], mu=mean), poisson.ppf(bounds[1], mu=mean)]
