"""
Model module
"""
from typing import Dict, List, Union

import numpy as np
from scipy.linalg import block_diag

from regmod.data import Data
from regmod.function import SmoothFunction
from regmod.parameter import Parameter
from regmod.utils import sizes_to_sclices
from regmod.variable import Variable


class Model:
    def __init__(self, data: Data, parameters: List[Parameter]):
        self.data = data
        self.parameters = parameters
        for param in self.parameters:
            param.check_data(self.data)

        self.sizes = [param.size for param in self.parameters]
        self.indices = sizes_to_sclices(self.sizes)
        self.size = sum(self.sizes)
        self.num_params = len(self.parameters)

        self.mat = [
            param.get_mat(self.data)
            for param in self.parameters
        ]
        self.uvec = np.hstack([param.get_uvec() for param in self.parameters])
        self.gvec = np.hstack([param.get_gvec() for param in self.parameters])
        self.linear_uvec = np.hstack([
            param.get_linear_uvec() for param in self.parameters
        ])
        self.linear_gvec = np.hstack([
            param.get_linear_gvec() for param in self.parameters
        ])
        self.linear_umat = block_diag(*[
            param.get_linear_umat() for param in self.parameters
        ])
        self.linear_gmat = block_diag(*[
            param.get_linear_gmat() for param in self.parameters
        ])

    def has_linear_gprior(self) -> bool:
        return self.linear_gvec.size > 0

    def has_linear_uprior(self) -> bool:
        return self.linear_uvec.size > 0

    def split_coefs(self, coefs: np.ndarray) -> List[np.ndarray]:
        assert len(coefs) == self.size
        return [coefs[index] for index in self.indices]

    def get_params(self, coefs: np.ndarray) -> np.ndarray:
        coefs = self.split_coefs(coefs)
        return [param.get_param(coefs[i], self.data, mat=self.mat[i])
                for i, param in enumerate(self.parameters)]

    def get_dparams(self, coefs: np.ndarray) -> np.ndarray:
        coefs = self.split_coefs(coefs)
        return [param.get_dparam(coefs[i], self.data, mat=self.mat[i])
                for i, param in enumerate(self.parameters)]

    def get_d2params(self, coefs: np.ndarray) -> np.ndarray:
        coefs = self.split_coefs(coefs)
        return [param.get_d2param(coefs[i], self.data, mat=self.mat[i])
                for i, param in enumerate(self.parameters)]

    def nll(self, params: List[np.ndarray]) -> np.ndarray:
        raise NotImplementedError()

    def dnll(self, params: List[np.ndarray]) -> List[np.ndarray]:
        raise NotImplementedError()

    def d2nll(self, params: List[np.ndarray]) -> List[List[np.ndarray]]:
        raise NotImplementedError()

    def objective_from_gprior(self, coefs: np.ndarray) -> float:
        val = 0.5*np.sum((coefs - self.gvec[0])**2/self.gvec[1]**2)
        if self.has_linear_gprior():
            val += 0.5*np.sum((self.linear_gmat.dot(coefs) - self.linear_gvec[0])**2/self.linear_gvec[1]**2)
        return val

    def gradient_from_gprior(self, coefs: np.ndarray) -> np.ndarray:
        grad = (coefs - self.gvec[0])/self.gvec[1]**2
        if self.has_linear_gprior():
            grad += (self.linear_gmat.T/self.linear_gvec[1]**2).dot(self.linear_gmat.dot(coefs) - self.linear_gvec[0])
        return grad

    def hessian_from_gprior(self) -> np.ndarray:
        hess = np.diag(1.0/self.gvec[1]**2)
        if self.has_linear_gprior():
            hess += (self.linear_gmat.T/self.linear_gvec[1]**2).dot(self.linear_gmat)
        return hess

    def objective(self, coefs: np.ndarray) -> float:
        params = self.get_params(coefs)
        return np.sum(self.nll(params)) + self.objective_from_gprior(coefs)

    def gradient(self, coefs: np.ndarray) -> np.ndarray:
        params = self.get_params(coefs)
        dparams = self.get_dparams(coefs)
        grad_params = self.dnll(params)
        return np.hstack([
            dparams[i].T.dot(grad_params[i])
            for i in range(self.num_params)
        ]) + self.gradient_from_gprior(coefs)

    def hessian(self, coefs: np.ndarray) -> np.ndarray:
        params = self.get_params(coefs)
        dparams = self.get_dparams(coefs)
        d2params = self.get_d2params(coefs)
        grad_params = self.dnll(params)
        hess_params = self.d2nll(params)
        hess = [
            [(dparams[i].T*hess_params[i][j]).dot(dparams[j])
             for j in range(self.num_params)]
            for i in range(self.num_params)
        ]
        for i in range(self.num_params):
            hess[i][i] += np.tensordot(grad_params[i], d2params[i], axes=1)
        return np.block(hess) + self.hessian_from_gprior()
