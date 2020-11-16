"""
Model module
"""
from typing import List, Union
import numpy as np
from scipy.linalg import block_diag
from .data import Data
from .variable import Variable
from .parameter import Parameter
from .function import SmoothFunction
from .utils import sizes_to_sclices


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
        self.spline_uvec = np.hstack([
            param.get_spline_uvec() for param in self.parameters
        ])
        self.spline_gvec = np.hstack([
            param.get_spline_gvec() for param in self.parameters
        ])
        self.spline_umat = block_diag(*[
            param.get_spline_umat() for param in self.parameters
        ])
        self.spline_gmat = block_diag(*[
            param.get_spline_gmat() for param in self.parameters
        ])

    def has_spline_gprior(self) -> bool:
        return self.spline_gvec.size > 0

    def has_spline_uprior(self) -> bool:
        return self.spline_uvec.size > 0

    def split_coefs(self, coefs: np.ndarray) -> List[np.ndarray]:
        assert len(coefs) == self.size
        return [coefs[index] for index in self.indices]

    def get_params(self, coefs: np.ndarray) -> np.ndarray:
        return [param.get_param(coefs, self.data, mat=self.mat[i])
                for i, param in enumerate(self.parameters)]

    def get_dparams(self, coefs: np.ndarray) -> np.ndarray:
        return [param.get_dparam(coefs, self.data, mat=self.mat[i])
                for i, param in enumerate(self.parameters)]

    def get_d2params(self, coefs: np.ndarray) -> np.ndarray:
        return [param.get_d2param(coefs, self.data, mat=self.mat[i])
                for i, param in enumerate(self.parameters)]

    def nll(self, params: List[np.ndarray]) -> np.ndarray:
        raise NotImplementedError()

    def dnll(self, params: List[np.ndarray]) -> List[np.ndarray]:
        raise NotImplementedError()

    def d2nll(self, params: List[np.ndarray]) -> List[List[np.ndarray]]:
        raise NotImplementedError()

    def objective_from_gprior(self, coefs: np.ndarray) -> float:
        val = 0.5*np.sum((coefs - self.gvec[0])**2/self.gvec[1]**2)
        if self.has_spline_gprior():
            val += 0.5*np.sum((self.spline_gmat.dot(coefs) - self.spline_gvec[0])**2/self.spline_gvec[1]**2)
        return val

    def gradient_from_gprior(self, coefs: np.ndarray) -> np.ndarray:
        grad = (coefs - self.gvec[0])/self.gvec[1]**2
        if self.has_spline_gprior():
            grad += (self.spline_gmat.T/self.spline_gvec[1]**2).dot(self.spline_gmat.dot(coefs) - self.spline_gvec[0])
        return grad

    def hessian_from_gprior(self) -> np.ndarray:
        hess = np.diag(1.0/self.gvec[1]**2)
        if self.has_spline_gprior():
            hess += (self.spline_gmat.T/self.spline_gvec[1]**2).dot(self.spline_gmat)
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
            [dparams[i].T.dot(hess_params[i][j]).dot(dparams[j])
             for j in range(self.num_params)]
            for i in range(self.num_params)
        ]
        for i in range(self.num_params):
            hess[i][i] += np.tensordot(grad_params[i], d2params[i], axes=1)
        return np.block(hess) + self.hessian_from_gprior()


class LinearModel(Model):
    def __init__(self, data: Data, variables: List[Variable],
                 inv_link: Union[str, SmoothFunction] = "identity"):
        mu = Parameter(name="mu",
                       variables=variables,
                       inv_link=inv_link)
        super().__init__(data, [mu])

    def nll(self, params: List[np.ndarray]) -> np.ndarray:
        return 0.5*self.data.weights*(params[0] - self.data.obs)**2

    def dnll(self, params: List[np.ndarray]) -> List[np.ndarray]:
        return [self.data.weights*(params[0] - self.data.obs)]

    def d2nll(self, params: List[np.ndarray]) -> List[np.ndarray]:
        return [[np.diag(self.data.weights)]]


class PoissonModel(Model):
    def __init__(self, data: Data, variables: List[Variable],
                 inv_link: Union[str, SmoothFunction] = "exp"):
        lam = Parameter(name="lam",
                        variables=variables,
                        inv_link=inv_link,
                        use_offset=True)
        super().__init__(data, [lam])

    def nll(self, params: List[np.ndarray]) -> np.ndarray:
        return self.data.weights*(params[0] - self.data.obs*np.log(params[0]))

    def dnll(self, params: List[np.ndarray]) -> List[np.ndarray]:
        return [self.data.weights*(1.0 - self.data.obs/params[0])]

    def d2nll(self, params: List[np.ndarray]) -> List[List[np.ndarray]]:
        return [[np.diag(self.data.weights*self.data.obs/params[0]**2)]]
