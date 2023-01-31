"""
Binomial Model
"""
from typing import Callable

import numpy as np
import pandas as pd
from numpy.typing import NDArray
from scipy.stats import binom

from regmod.optimizer import msca_optimize

from .model import Model
from .utils import model_post_init


class BinomialModel(Model):
    param_names = ("p",)
    default_param_specs = {"p": {"inv_link": "expit"}}

    def _attach(self, df: pd.DataFrame, require_y: bool = True):
        super()._attach(df, require_y=require_y)
        if require_y and not np.all((self._data["y"] >= 0) & (self._data["y"] <= 1)):
            raise ValueError(
                "Binomial model requires observations to be between zero and one."
            )
        (self._data["mat"][0],
         self._data["cmat"],
         self._data["cvec"]) = model_post_init(
            self._data["mat"][0],
            self._data["uvec"],
            self._data["linear_umat"],
            self._data["linear_uvec"]
        )

    def objective(self, coefs: NDArray) -> float:
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
            coefs, self._data["offset"][0], mat=self._data["mat"][0]
        )
        param = inv_link.fun(lin_param)

        weights = self._data["weights"]*self.trim_weights
        obj_param = -weights * (
            self._data["y"] * np.log(param) +
            (1 - self._data["y"]) * np.log(1 - param)
        )
        return obj_param.sum() + self.objective_from_gprior(coefs)

    def gradient(self, coefs: NDArray) -> NDArray:
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
        mat = self._data["mat"][0]
        inv_link = self.params[0].inv_link
        lin_param = self.params[0].get_lin_param(
            coefs, self._data["offset"][0], mat=self._data["mat"][0]
        )
        param = inv_link.fun(lin_param)
        dparam = inv_link.dfun(lin_param)

        weights = self._data["weights"]*self.trim_weights
        grad_param = weights * (
            (param - self._data["y"]) / (param*(1 - param)) * dparam
        )

        return mat.T.dot(grad_param) + self.gradient_from_gprior(coefs)

    def hessian(self, coefs: NDArray) -> NDArray:
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
        mat = self._data["mat"][0]
        inv_link = self.params[0].inv_link
        lin_param = self.params[0].get_lin_param(
            coefs, self._data["offset"][0], mat=self._data["mat"][0]
        )
        param = inv_link.fun(lin_param)
        dparam = inv_link.dfun(lin_param)
        d2param = inv_link.d2fun(lin_param)

        weights = self._data["weights"]*self.trim_weights
        hess_param = weights * (
            (self._data["y"] / param**2 + (1 - self._data["y"]) / (1 - param)**2) * dparam**2 +
            (param - self._data["y"]) / (param*(1 - param)) * d2param
        )

        scaled_mat = mat.scale_rows(hess_param)
        hess_mat = mat.T.dot(scaled_mat)
        hess_mat_gprior = type(hess_mat)(self.hessian_from_gprior())
        return hess_mat + hess_mat_gprior

    def jacobian2(self, coefs: NDArray) -> NDArray:
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
        mat = self._data["mat"][0]
        inv_link = self.params[0].inv_link
        lin_param = self.params[0].get_lin_param(
            coefs, self._data["offset"][0], mat=self._data["mat"][0]
        )
        param = inv_link.fun(lin_param)
        dparam = inv_link.dfun(lin_param)
        weights = self._data["weights"]*self.trim_weights
        grad_param = weights * (
            (param - self._data["y"]) / (param*(1 - param)) * dparam
        )
        jacobian = mat.T.scale_cols(grad_param)
        hess_mat_gprior = type(jacobian)(self.hessian_from_gprior())
        jacobian2 = jacobian.dot(jacobian.T) + hess_mat_gprior
        return jacobian2

    def fit(self,
            df: pd.DataFrame,
            optimizer: Callable = msca_optimize,
            **optimizer_options):
        """Fit function.

        Parameters
        ----------
        optimizer : Callable, optional
            Model solver, by default scipy_optimize.
        """
        super().fit(
            df,
            optimizer=optimizer,
            **optimizer_options
        )

    def nll(self, params: list[NDArray]) -> NDArray:
        return -(self._data["y"]*np.log(params[0]) + (1 - self._data["y"])*np.log(1.0 - params[0]))

    def dnll(self, params: list[NDArray]) -> list[NDArray]:
        return [-(self._data["y"]/params[0] - (1 - self._data["y"])/(1.0 - params[0]))]

    def d2nll(self, params: list[NDArray]) -> list[list[NDArray]]:
        return [[self._data["y"]/params[0]**2 + (1 - self._data["y"])/(1.0 - params[0])**2]]

    def get_ui(self, params: list[NDArray], bounds: tuple[float, float]) -> NDArray:
        p = params[0]
        n = self.y_sample_sizes
        return [binom.ppf(bounds[0], n=n, p=p),
                binom.ppf(bounds[1], n=n, p=p)]
