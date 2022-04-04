"""
Gaussian Model
"""
from typing import Callable, List, Tuple, Union

import numpy as np
from anml.linalg.matrix import asmatrix
from anml.optimizer import IPSolver
from numpy import ndarray
from regmod.data import Data
from scipy.sparse import csc_matrix
from scipy.stats import norm

from .model import Model


class GaussianModel(Model):
    param_names = ("mu",)
    default_param_specs = {"mu": {"inv_link": "identity"}}

    def __init__(self, data: Data, **kwargs):
        super().__init__(data, **kwargs)
        mat = self.mat[0]
        sparsity = (mat == 0).sum() / mat.size
        self.sparse = sparsity > 0.95
        if self.sparse:
            mat = csc_matrix(mat).astype(np.float64)
        self.mat[0] = asmatrix(mat)

    def objective(self, coefs: ndarray) -> float:
        """Objective function.
        Parameters
        ----------
        coefs : ndarray
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

    def gradient(self, coefs: ndarray) -> ndarray:
        """Gradient function.

        Parameters
        ----------
        coefs : ndarray
            Given coefficients.

        Returns
        -------
        ndarray
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

    def hessian(self, coefs: ndarray) -> ndarray:
        """Hessian function.

        Parameters
        ----------
        coefs : ndarray
            Given coefficients.

        Returns
        -------
        ndarray
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

    def jacobian2(self, coefs: ndarray) -> ndarray:
        """Jacobian function.

        Parameters
        ----------
        coefs : ndarray
            Given coefficients.

        Returns
        -------
        ndarray
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
            optimizer: Callable = IPSolver,
            **optimizer_options):
        """Fit function.

        Parameters
        ----------
        optimizer : Callable, optional
            Model solver, by default scipy_optimize.
        """
        # extract the constraints

        valid_indices = ~np.isclose(self.linear_umat, 0).all(axis=1)
        umat = np.vstack([
            np.identity(self.size), self.linear_umat[valid_indices]
        ])
        uvec = np.hstack([
            self.uvec,
            self.linear_uvec[:, valid_indices]
        ])
        lb_indices, ub_indices = ~np.isneginf(uvec[0]), ~np.isposinf(uvec[1])
        cmat = np.vstack([-umat[lb_indices], umat[ub_indices]])
        cvec = np.hstack([-uvec[0][lb_indices], uvec[1][ub_indices]])
        if self.sparse:
            cmat = csc_matrix(cmat)
        cmat = asmatrix(cmat)
        solver = optimizer(self.gradient, self.hessian, cmat, cvec)
        self.opt_coefs = solver.minimize(
            np.zeros(self.size),
            **optimizer_options
        )
        self.opt_vcov = self.get_vcov(self.opt_coefs)

    def nll(self, params: List[ndarray]) -> ndarray:
        return 0.5*(params[0] - self.data.obs)**2

    def dnll(self, params: List[ndarray]) -> List[ndarray]:
        return [params[0] - self.data.obs]

    def d2nll(self, params: List[ndarray]) -> List[ndarray]:
        return [[np.ones(self.data.num_obs)]]

    def get_ui(self, params: List[ndarray], bounds: Tuple[float, float]) -> ndarray:
        mean = params[0]
        sd = 1.0/np.sqrt(self.data.weights)
        return [norm.ppf(bounds[0], loc=mean, scale=sd),
                norm.ppf(bounds[1], loc=mean, scale=sd)]
