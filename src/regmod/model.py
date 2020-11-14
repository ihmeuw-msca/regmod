"""
Model module
"""
from typing import List
import numpy as np
from scipy.linalg import block_diag
from .data import Data
from .variable import Variable
from .parameter import Parameter
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
        self.uvec = np.hstack([param.get_uvec for param in self.parameters])
        self.gvec = np.hstack([param.get_gvec for param in self.parameters])
        self.spline_uvec = np.hstack([
            param.get_spline_uvec for param in self.parameters
        ])
        self.spline_gvec = np.hstack([
            param.get_spline_gvec for param in self.parameters
        ])
        self.spline_umat = block_diag(*[
            param.get_spline_umat for param in self.parameters
        ])
        self.spline_gmat = block_diag(*[
            param.get_spline_gmat for param in self.parameters
        ])

    def split_coefs(self, coefs: np.ndarray) -> List[np.ndarray]:
        assert len(coefs) == self.size
        return [coefs[index] for index in self.indices]

    def negloglikelihood(self, coefs: np.ndarray) -> np.ndarray:
        raise NotImplementedError()

    def objective(self, coefs: np.ndarray) -> float:
        return sum(self.negloglikelihood(coefs))

    def gradient(self, coefs: np.ndarray) -> np.ndarray:
        raise NotImplementedError()

    def hessian(self, coefs: np.ndarray) -> np.ndarray:
        raise NotImplementedError()


class LinearModel(Model):
    def __init__(self, data: Data, variables: List[Variable]):
        mu = Parameter(name="mu",
                       variables=variables,
                       inv_link="identity")
        super().__init__(data, [mu])

    def negloglikelihood(self, coefs: np.ndarray) -> np.ndarray:
        return 0.5*self.data.weights*(
            self.data.obs - self.mat[0].dot(coefs)
        )**2

    def gradient(self, coefs: np.ndarray) -> np.ndarray:
        return (self.mat[0].T*self.data.weights).dot(
            self.mat[0].dot(coefs) - self.data.obs
        )

    def hessian(self, coefs: np.ndarray) -> np.ndarray:
        return (self.mat[0].T*self.data.weights).dot(self.mat[0])