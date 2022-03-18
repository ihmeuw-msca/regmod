"""
Poisson Model
"""
from typing import List, Tuple

import numpy as np
from numpy import ndarray
from regmod.data import Data
from scipy.sparse import csc_matrix, csr_matrix
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
        if sparsity > 0.95:
            self.mat[0] = csc_matrix(mat)

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
        mat = self.mat[0]
        inv_link = self.params[0].inv_link
        lin_param = mat.dot(coefs)

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
        lin_param = mat.dot(coefs)

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
        lin_param = mat.dot(coefs)

        weights = self.data.weights*self.data.trim_weights
        hess_params = (inv_link.d2fun(lin_param)) * weights

        if isinstance(mat, csc_matrix):
            scaled_mat = mat.copy()
            scaled_mat.data *= hess_params[mat.indices]
        else:
            scaled_mat = hess_params[:, np.newaxis] * mat

        hess_mat = mat.T.dot(scaled_mat)
        hess_mat_gprior = self.hessian_from_gprior()
        if isinstance(hess_mat, (csr_matrix, csc_matrix)):
            hess_mat_gprior = type(hess_mat)(hess_mat_gprior)
        return hess_mat + hess_mat_gprior

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
