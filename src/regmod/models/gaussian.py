"""
Gaussian Model
"""
from typing import Callable

import numpy as np
import pandas as pd
from numpy.typing import NDArray
from scipy.stats import norm

from regmod.optimizer import msca_optimize

from .data import parse_to_msca
from .model import Model


class GaussianModel(Model):
    param_names = ("mu",)
    default_param_specs = {"mu": {"inv_link": "identity"}}

    def _parse(self, df: pd.DataFrame, fit: bool = True):
        self._validate_data(df)
        return parse_to_msca(df, self.y, self.params, self.weights, fit=fit)

    def objective(self, data: dict, coef: NDArray) -> float:
        """Objective function.
        Parameters
        ----------
        coef : NDArray
            Given coefficients.
        Returns
        -------
        float
            Objective value.
        """
        inv_link = self.params[0].inv_link
        lin_param = self.params[0].get_lin_param(
            coef, data["offset"][0], mat=data["mat"][0]
        )
        param = inv_link.fun(lin_param)

        weights = data["weights"] * data["trim_weights"]
        obj_param = weights * 0.5 * (param - data["y"]) ** 2
        return obj_param.sum() + self.objective_from_gprior(data, coef)

    def gradient(self, data: dict, coef: NDArray) -> NDArray:
        """Gradient function.

        Parameters
        ----------
        coef : NDArray
            Given coefficients.

        Returns
        -------
        NDArray
            Gradient vector.
        """
        mat = data["mat"][0]
        inv_link = self.params[0].inv_link
        lin_param = self.params[0].get_lin_param(
            coef, data["offset"][0], mat=data["mat"][0]
        )
        param = inv_link.fun(lin_param)
        dparam = inv_link.dfun(lin_param)

        weights = data["weights"] * data["trim_weights"]
        grad_param = weights * (param - data["y"]) * dparam

        return mat.T.dot(grad_param) + self.gradient_from_gprior(data, coef)

    def hessian(self, data: dict, coef: NDArray) -> NDArray:
        """Hessian function.

        Parameters
        ----------
        coef : NDArray
            Given coefficients.

        Returns
        -------
        NDArray
            Hessian matrix.
        """
        mat = data["mat"][0]
        inv_link = self.params[0].inv_link
        lin_param = self.params[0].get_lin_param(
            coef, data["offset"][0], mat=data["mat"][0]
        )
        param = inv_link.fun(lin_param)
        dparam = inv_link.dfun(lin_param)
        d2param = inv_link.d2fun(lin_param)

        weights = data["weights"] * data["trim_weights"]
        hess_param = weights * (dparam**2 + (param - data["y"]) * d2param)

        scaled_mat = mat.scale_rows(hess_param)
        hess_mat = mat.T.dot(scaled_mat)
        hess_mat_gprior = type(hess_mat)(self.hessian_from_gprior(data))
        return hess_mat + hess_mat_gprior

    def jacobian2(self, data: dict, coef: NDArray) -> NDArray:
        """Jacobian function.

        Parameters
        ----------
        coef : NDArray
            Given coefficients.

        Returns
        -------
        NDArray
            Jacobian matrix.
        """
        mat = data["mat"][0]
        inv_link = self.params[0].inv_link
        lin_param = self.params[0].get_lin_param(
            coef, data["offset"][0], mat=data["mat"][0]
        )
        param = inv_link.fun(lin_param)
        dparam = inv_link.dfun(lin_param)
        weights = data["weights"] * data["trim_weights"]
        grad_param = weights * (param - data["y"]) * dparam
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
        return 0.5 * (params[0] - data["y"]) ** 2

    def dnll(self, data: dict, params: list[NDArray]) -> list[NDArray]:
        return [params[0] - data["y"]]

    def d2nll(self, data: dict, params: list[NDArray]) -> list[NDArray]:
        return [[np.ones(data["offset"][0].shape[0])]]

    def get_ui(
        self, data: dict, params: list[NDArray], bounds: tuple[float, float]
    ) -> NDArray:
        mean = params[0]
        sd = 1.0 / np.sqrt(data["weights"])
        return [
            norm.ppf(bounds[0], loc=mean, scale=sd),
            norm.ppf(bounds[1], loc=mean, scale=sd),
        ]
