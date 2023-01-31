"""
Gaussian Model
"""
from typing import Callable, List, Tuple

import numpy as np
import pandas as pd
from numpy.typing import NDArray
from scipy.stats import norm

from regmod.optimizer import msca_optimize

from .model import Model
from .utils import model_post_init


class GaussianModel(Model):
    param_names = ("mu",)
    default_param_specs = {"mu": {"inv_link": "identity"}}

    def _attach(self, df: pd.DataFrame, require_y: bool = True):
        super()._attach(df, require_y=require_y)
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
        obj_param = weights * 0.5 * (
            param - self._data["y"]
        )**2
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
            param - self._data["y"]
        ) * dparam

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
            dparam**2 + (param - self._data["y"])*d2param
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
        grad_param = weights * (param - self._data["y"]) * dparam
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

    def nll(self, params: List[NDArray]) -> NDArray:
        return 0.5*(params[0] - self._data["y"])**2

    def dnll(self, params: List[NDArray]) -> List[NDArray]:
        return [params[0] - self._data["y"]]

    def d2nll(self, params: List[NDArray]) -> List[NDArray]:
        return [[np.ones(self._data["offset"][0].shape[0])]]

    def get_ui(self, params: List[NDArray], bounds: Tuple[float, float]) -> NDArray:
        mean = params[0]
        sd = 1.0/np.sqrt(self._data["weights"])
        return [norm.ppf(bounds[0], loc=mean, scale=sd),
                norm.ppf(bounds[1], loc=mean, scale=sd)]
