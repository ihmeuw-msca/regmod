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

    def get_param(self, index: int, coefs: np.ndarray) -> np.ndarray:
        return self.parameters[index].get_param(coefs, self.data, mat=self.mat[index])

    def get_dparam(self, index: int, coefs: np.ndarray) -> np.ndarray:
        return self.parameters[index].get_dparam(coefs, self.data, mat=self.mat[index])

    def get_d2param(self, index: int, coefs: np.ndarray) -> np.ndarray:
        return self.parameters[index].get_d2param(coefs, self.data, mat=self.mat[index])

    def negloglikelihood(self, coefs: np.ndarray) -> np.ndarray:
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
        val = sum(self.negloglikelihood(coefs))
        val += self.objective_from_gprior(coefs)
        return val

    def gradient(self, coefs: np.ndarray) -> np.ndarray:
        raise NotImplementedError()

    def hessian(self, coefs: np.ndarray) -> np.ndarray:
        raise NotImplementedError()


class LinearModel(Model):
    def __init__(self, data: Data, variables: List[Variable],
                 inv_link: Union[str, SmoothFunction] = "identity"):
        mu = Parameter(name="mu",
                       variables=variables,
                       inv_link=inv_link)
        super().__init__(data, [mu])

    def residual(self, coefs: np.ndarray) -> np.ndarray:
        return self.get_param(0, coefs) - self.data.obs

    def negloglikelihood(self, coefs: np.ndarray) -> np.ndarray:
        return 0.5*self.data.weights*self.residual(coefs)**2

    def gradient(self, coefs: np.ndarray) -> np.ndarray:
        grad = self.get_dparam(0, coefs).T.dot(
            self.data.weights*self.residual(coefs)
        ) + self.gradient_from_gprior(coefs)
        return grad

    def hessian(self, coefs: np.ndarray) -> np.ndarray:
        dparam = self.get_dparam(0, coefs)
        d2param = self.get_d2param(0, coefs)

        hess = np.tensordot(self.data.weights*self.residual(coefs), d2param, axes=1)
        hess += (dparam.T*self.data.weights).dot(dparam)
        hess += self.hessian_from_gprior()
        return hess
