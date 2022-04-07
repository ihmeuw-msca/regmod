"""
Gaussian Model
"""
from typing import Callable, List, Tuple

import numpy as np
from msca.linalg.matrix import asmatrix
from numpy.typing import NDArray
from regmod.data import Data
from regmod.optimizer import msca_optimize
from scipy.linalg import block_diag
from scipy.sparse import csc_matrix
from scipy.stats import norm

from .model import Model
from .utils import model_post_init


class GaussianModel(Model):
    param_names = ("mu",)
    default_param_specs = {"mu": {"inv_link": "identity"}}

    def __init__(self, data: Data, **kwargs):
        super().__init__(data, **kwargs)
        self.mat[0], self.cmat, self.cvec = model_post_init(
            self.mat[0], self.uvec, self.linear_umat, self.linear_uvec
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
            coefs, self.data, mat=self.mat[0]
        )

        weights = self.data.weights*self.data.trim_weights
        obj_params = 0.5*(
            inv_link.fun(lin_param) - self.data.obs
        )**2 * weights
        return obj_params.sum() + self.objective_from_gprior(coefs)

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
        mat = self.mat[0]
        inv_link = self.params[0].inv_link
        lin_param = self.params[0].get_lin_param(
            coefs, self.data, mat=self.mat[0]
        )

        weights = self.data.weights*self.data.trim_weights
        grad_params = inv_link.dfun(lin_param) * (
            inv_link.fun(lin_param) - self.data.obs
        ) * weights

        return mat.T.dot(grad_params) + self.gradient_from_gprior(coefs)

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
        mat = self.mat[0]
        inv_link = self.params[0].inv_link
        lin_param = self.params[0].get_lin_param(
            coefs, self.data, mat=self.mat[0]
        )

        weights = self.data.weights*self.data.trim_weights
        hess_params = (
            inv_link.d2fun(lin_param) *
            (inv_link.fun(lin_param) - self.data.obs) +
            (inv_link.dfun(lin_param))**2
        ) * weights

        scaled_mat = mat.scale_rows(hess_params)

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
        mat = self.mat[0]
        inv_link = self.params[0].inv_link
        lin_param = self.params[0].get_lin_param(
            coefs, self.data, mat=self.mat[0]
        )
        param = inv_link.fun(lin_param)
        dparam = mat.scale_rows(inv_link.dfun(lin_param))
        grad_param = param - self.data.obs
        weights = self.data.weights*self.data.trim_weights
        jacobian = dparam.T.scale_cols(weights*grad_param)
        hess_mat_gprior = type(jacobian)(self.hessian_from_gprior())
        jacobian2 = jacobian.dot(jacobian.T) + hess_mat_gprior
        return jacobian2

    def fit(self,
            optimizer: Callable = msca_optimize,
            **optimizer_options):
        """Fit function.

        Parameters
        ----------
        optimizer : Callable, optional
            Model solver, by default scipy_optimize.
        """
        optimizer(self, **optimizer_options)

    def nll(self, params: List[NDArray]) -> NDArray:
        return 0.5*(params[0] - self.data.obs)**2

    def dnll(self, params: List[NDArray]) -> List[NDArray]:
        return [params[0] - self.data.obs]

    def d2nll(self, params: List[NDArray]) -> List[NDArray]:
        return [[np.ones(self.data.num_obs)]]

    def get_ui(self, params: List[NDArray], bounds: Tuple[float, float]) -> NDArray:
        mean = params[0]
        sd = 1.0/np.sqrt(self.data.weights)
        return [norm.ppf(bounds[0], loc=mean, scale=sd),
                norm.ppf(bounds[1], loc=mean, scale=sd)]
