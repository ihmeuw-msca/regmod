"""
Poisson Model
"""
from typing import Callable, List, Tuple

import numpy as np
from anml.linalg.matrix import asmatrix
from anml.optimizer import IPSolver
from numpy import ndarray
from regmod.data import Data
from scipy.sparse import csc_matrix
from scipy.stats import poisson

from .model import Model


class PoissonModel(Model):
    param_names = ("lam",)
    default_param_specs = {"lam": {"inv_link": "exp"}}

    def __init__(self, data: Data, **kwargs):
        if not all(data.obs >= 0):
            raise ValueError("Poisson model requires observations to be non-negagive.")
        super().__init__(data, **kwargs)
        mat = self.mat[0]
        sparsity = (mat == 0).sum() / mat.size
        self.sparse = sparsity > 0.95
        if self.sparse:
            mat = csc_matrix(mat)
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
        obj_params = (
            inv_link.fun(lin_param) - self.data.obs * lin_param
        ) * weights
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
        grad_params = (inv_link.dfun(lin_param) - self.data.obs) * weights

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
        hess_params = (inv_link.d2fun(lin_param)) * weights

        scaled_mat = mat.scale_rows(hess_params)

        hess_mat = mat.T.dot(scaled_mat)
        hess_mat_gprior = type(hess_mat)(self.hessian_from_gprior())
        return hess_mat + hess_mat_gprior

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

    def nll(self, params: List[ndarray]) -> ndarray:
        return params[0] - self.data.obs*np.log(params[0])

    def dnll(self, params: List[ndarray]) -> List[ndarray]:
        return [1.0 - self.data.obs/params[0]]

    def d2nll(self, params: List[ndarray]) -> List[List[ndarray]]:
        return [[self.data.obs/params[0]**2]]

    def get_ui(self, params: List[ndarray], bounds: Tuple[float, float]) -> ndarray:
        mean = params[0]
        return [poisson.ppf(bounds[0], mu=mean),
                poisson.ppf(bounds[1], mu=mean)]
