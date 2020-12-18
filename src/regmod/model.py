"""
Model module
"""
from typing import Dict, List, Union

import numpy as np
from scipy.linalg import block_diag
from scipy.special import digamma, loggamma, polygamma

from .data import Data
from .function import SmoothFunction
from .parameter import Parameter
from .utils import sizes_to_sclices
from .variable import Variable


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


class LinearModel(Model):
    def __init__(self, data: Data, variables: List[Variable],
                 inv_link: Union[str, SmoothFunction] = "identity",
                 use_offset: bool = False):
        mu = Parameter(name="mu",
                       variables=variables,
                       inv_link=inv_link,
                       use_offset=use_offset)
        super().__init__(data, [mu])

    def nll(self, params: List[np.ndarray]) -> np.ndarray:
        return 0.5*self.data.weights*(params[0] - self.data.obs)**2

    def dnll(self, params: List[np.ndarray]) -> List[np.ndarray]:
        return [self.data.weights*(params[0] - self.data.obs)]

    def d2nll(self, params: List[np.ndarray]) -> List[np.ndarray]:
        return [[self.data.weights]]

    def __repr__(self) -> str:
        return f"LinearModel(num_obs={self.data.num_obs}, num_params={self.num_params}, size={self.size})"


class PoissonModel(Model):
    def __init__(self, data: Data, variables: List[Variable],
                 inv_link: Union[str, SmoothFunction] = "exp",
                 use_offset: bool = False):
        lam = Parameter(name="lam",
                        variables=variables,
                        inv_link=inv_link,
                        use_offset=use_offset)
        assert all(data.obs >= 0), \
            "Poisson model requires observations to be non-negagive."
        super().__init__(data, [lam])

    def nll(self, params: List[np.ndarray]) -> np.ndarray:
        return self.data.weights*(params[0] - self.data.obs*np.log(params[0]))

    def dnll(self, params: List[np.ndarray]) -> List[np.ndarray]:
        return [self.data.weights*(1.0 - self.data.obs/params[0])]

    def d2nll(self, params: List[np.ndarray]) -> List[List[np.ndarray]]:
        return [[self.data.weights*self.data.obs/params[0]**2]]

    def __repr__(self) -> str:
        return f"PoissonModel(num_obs={self.data.num_obs}, num_params={self.num_params}, size={self.size})"


class BinomialModel(Model):
    def __init__(self, data: Data, variables: List[Variable],
                 inv_link: Union[str, SmoothFunction] = "expit",
                 use_offset: bool = False):
        p = Parameter(name="p",
                      variables=variables,
                      inv_link=inv_link,
                      use_offset=use_offset)
        assert np.all(data.obs >= 0), \
            "Binomial model requires observations to be non-negative."
        assert len(data.col_obs) == 2, \
            "Binomial model need 2 columns of observations, one for number of events, one for sample size."
        self.obs_1s = data.get_cols(data.col_obs[0])
        self.obs_sample_sizes = data.get_cols(data.col_obs[1])
        assert all(self.obs_1s <= self.obs_sample_sizes), \
            "Binomial model requires number of events less or equal than sample size."
        self.obs_0s = self.obs_sample_sizes - self.obs_1s
        super().__init__(data, [p])

    def nll(self, params: List[np.ndarray]) -> np.ndarray:
        return -self.data.weights*(self.obs_1s*np.log(params[0]) +
                                   self.obs_0s*np.log(1.0 - params[0]))

    def dnll(self, params: List[np.ndarray]) -> List[np.ndarray]:
        return [-self.data.weights*(self.obs_1s/params[0] -
                                    self.obs_0s/(1.0 - params[0]))]

    def d2nll(self, params: List[np.ndarray]) -> List[List[np.ndarray]]:
        return [[self.data.weights*(self.obs_1s/params[0]**2 +
                                    self.obs_0s/(1.0 - params[0])**2)]]

    def __repr__(self) -> str:
        return f"BinomialModel(num_obs={self.data.num_obs}, num_params={self.num_params}, size={self.size})"


class NegativeBinomialModel(Model):
    def __init__(self,
                 data: Data,
                 variables: Dict[str, List[Variable]],
                 inv_link: Dict[str, Union[str, SmoothFunction]] = {"r": "exp", "p": "expit"},
                 use_offset: Dict[str, bool] = {"r": False, "p": False}):
        if not all([param_name in variables for param_name in ["r", "p"]]):
            raise ValueError(f"'r' and 'p' must be keys in `variables`.")
        assert np.all(data.obs >= 0), \
            "Negative-Binomial model requires observations to be non-negative."

        params = [
            Parameter(name=param_name,
                      variables=variables[param_name],
                      inv_link=inv_link[param_name],
                      use_offset=use_offset[param_name])
            for param_name in ["r", "p"]
        ]
        super().__init__(data, params)

    def nll(self, params: List[np.ndarray]) -> np.ndarray:
        return -self.data.weights*(loggamma(params[0] + self.data.obs) -
                                   loggamma(params[0]) +
                                   self.data.obs*np.log(params[1]) +
                                   params[0]*np.log(1 - params[1]))

    def dnll(self, params: List[np.ndarray]) -> List[np.ndarray]:
        return [-self.data.weights*(digamma(params[0] + self.data.obs) -
                                    digamma(params[0]) +
                                    np.log(1 - params[1])),
                -self.data.weights*(self.data.obs/params[1] -
                                    params[0]/(1 - params[1]))]

    def d2nll(self, params: List[np.ndarray]) -> List[List[np.ndarray]]:
        return [[self.data.weights*(polygamma(1, params[0]) -
                                    polygamma(1, params[0] + self.data.obs)),
                 self.data.weights/(1 - params[1])],
                [self.data.weights/(1 - params[1]),
                 self.data.weights*(params[0]/(1 - params[1])**2 +
                                    self.data.obs/params[1]**2)]]

    def __repr__(self) -> str:
        return f"NegativeBinomialModel(num_obs={self.data.num_obs}, num_params={self.num_params}, size={self.size})"
