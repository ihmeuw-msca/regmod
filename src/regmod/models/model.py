"""
Model module
"""
from typing import Dict, List, Tuple

import numpy as np
from numpy import ndarray
from scipy.linalg import block_diag

from regmod.data import Data
from regmod.parameter import Parameter
from regmod.utils import sizes_to_sclices


class Model:
    param_names: Tuple[str] = None
    default_param_specs: Dict[str, Dict] = None

    def __init__(self,
                 data: Data,
                 params: List[Parameter] = None,
                 param_specs: Dict[str, Dict] = None):
        if params is None and param_specs is None:
            raise ValueError("Must provide `params` or `param_specs`")

        if params is not None:
            param_dict = {
                param.name: param
                for param in params
            }
            self.params = [param_dict[param_name]
                           for param_name in self.param_names]
        else:
            self.params = [Parameter(param_name,
                                     **{**self.default_param_specs[param_name],
                                        **param_specs[param_name]})
                           for param_name in self.param_names]

        self.data = data
        for param in self.params:
            param.check_data(self.data)

        self.sizes = [param.size for param in self.params]
        self.indices = sizes_to_sclices(self.sizes)
        self.size = sum(self.sizes)
        self.num_params = len(self.params)

        self.mat = self.get_mat()
        self.uvec = self.get_uvec()
        self.gvec = self.get_gvec()
        self.linear_uvec = self.get_linear_uvec()
        self.linear_gvec = self.get_linear_gvec()
        self.linear_umat = self.get_linear_umat()
        self.linear_gmat = self.get_linear_gmat()

    def get_mat(self) -> List[ndarray]:
        return [param.get_mat(self.data) for param in self.params]

    def get_uvec(self) -> ndarray:
        return np.hstack([param.get_uvec() for param in self.params])

    def get_gvec(self) -> ndarray:
        return np.hstack([param.get_gvec() for param in self.params])

    def get_linear_uvec(self) -> ndarray:
        return np.hstack([param.get_linear_uvec() for param in self.params])

    def get_linear_gvec(self) -> ndarray:
        return np.hstack([param.get_linear_gvec() for param in self.params])

    def get_linear_umat(self) -> ndarray:
        return block_diag(*[param.get_linear_umat() for param in self.params])

    def get_linear_gmat(self) -> ndarray:
        return block_diag(*[param.get_linear_gmat() for param in self.params])

    def split_coefs(self, coefs: ndarray) -> List[ndarray]:
        assert len(coefs) == self.size
        return [coefs[index] for index in self.indices]

    def get_params(self, coefs: ndarray) -> ndarray:
        coefs = self.split_coefs(coefs)
        return [param.get_param(coefs[i], self.data, mat=self.mat[i])
                for i, param in enumerate(self.params)]

    def get_dparams(self, coefs: ndarray) -> ndarray:
        coefs = self.split_coefs(coefs)
        return [param.get_dparam(coefs[i], self.data, mat=self.mat[i])
                for i, param in enumerate(self.params)]

    def get_d2params(self, coefs: ndarray) -> ndarray:
        coefs = self.split_coefs(coefs)
        return [param.get_d2param(coefs[i], self.data, mat=self.mat[i])
                for i, param in enumerate(self.params)]

    def nll(self, params: List[ndarray]) -> ndarray:
        raise NotImplementedError()

    def dnll(self, params: List[ndarray]) -> List[ndarray]:
        raise NotImplementedError()

    def d2nll(self, params: List[ndarray]) -> List[List[ndarray]]:
        raise NotImplementedError()

    def get_ui(self,
               params: List[ndarray],
               bounds: Tuple[float, float]) -> ndarray:
        raise NotImplementedError()

    def detect_outliers(self,
                        coefs: ndarray,
                        bounds: Tuple[float, float]) -> ndarray:
        params = self.get_params(coefs)
        ui = self.get_ui(params, bounds)
        return (self.data.obs < ui[0]) | (self.data.obs > ui[1])

    def objective_from_gprior(self, coefs: ndarray) -> float:
        val = 0.5*np.sum((coefs - self.gvec[0])**2/self.gvec[1]**2)
        if self.linear_gvec.size > 0:
            val += 0.5*np.sum((self.linear_gmat.dot(coefs) - self.linear_gvec[0])**2/self.linear_gvec[1]**2)
        return val

    def gradient_from_gprior(self, coefs: ndarray) -> ndarray:
        grad = (coefs - self.gvec[0])/self.gvec[1]**2
        if self.linear_gvec.size > 0:
            grad += (self.linear_gmat.T/self.linear_gvec[1]**2).dot(self.linear_gmat.dot(coefs) - self.linear_gvec[0])
        return grad

    def hessian_from_gprior(self) -> ndarray:
        hess = np.diag(1.0/self.gvec[1]**2)
        if self.linear_gvec.size > 0:
            hess += (self.linear_gmat.T/self.linear_gvec[1]**2).dot(self.linear_gmat)
        return hess

    def objective(self, coefs: ndarray) -> float:
        params = self.get_params(coefs)
        obj_params = self.nll(params)
        weights = self.data.weights*self.data.trim_weights
        return weights.dot(obj_params) + \
            self.objective_from_gprior(coefs)

    def gradient(self, coefs: ndarray) -> ndarray:
        params = self.get_params(coefs)
        dparams = self.get_dparams(coefs)
        grad_params = self.dnll(params)
        weights = self.data.weights*self.data.trim_weights
        return np.hstack([
            dparams[i].T.dot(weights*grad_params[i])
            for i in range(self.num_params)
        ]) + self.gradient_from_gprior(coefs)

    def hessian(self, coefs: ndarray) -> ndarray:
        params = self.get_params(coefs)
        dparams = self.get_dparams(coefs)
        d2params = self.get_d2params(coefs)
        grad_params = self.dnll(params)
        hess_params = self.d2nll(params)
        weights = self.data.weights*self.data.trim_weights
        hess = [
            [(dparams[i].T*(weights*hess_params[i][j])).dot(dparams[j])
             for j in range(self.num_params)]
            for i in range(self.num_params)
        ]
        for i in range(self.num_params):
            hess[i][i] += np.tensordot(weights*grad_params[i], d2params[i], axes=1)
        return np.block(hess) + self.hessian_from_gprior()
