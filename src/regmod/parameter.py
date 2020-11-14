"""
Parameter module
"""
from typing import List, Tuple, Union
from dataclasses import dataclass, field
import numpy as np
from scipy.linalg import block_diag
from .data import Data
from .variable import Variable, SplineVariable
from .function import SmoothFunction, fun_dict


@dataclass
class Parameter:
    name: str
    variables: List[Variable] = field(repr=False)
    inv_link: Union[str, SmoothFunction] = field(repr=False)
    use_offset: bool = field(default=False, repr=False)

    def __post_init__(self):
        if isinstance(self.inv_link, str):
            self.inv_link = fun_dict[self.inv_link]
        assert isinstance(self.inv_link, SmoothFunction), \
            "inv_link has to be an instance of SmoothFunction."

    @property
    def size(self) -> int:
        return sum([var.size for var in self.variables])

    def check_data(self, data: Data):
        for var in self.variables:
            var.check_data(data)

    def get_mat(self, data: Data) -> np.ndarray:
        return np.hstack([var.get_mat(data) for var in self.variables])

    def get_uvec(self) -> np.ndarray:
        return np.hstack([var.get_uvec() for var in self.variables])

    def get_gvec(self) -> np.ndarray:
        return np.hstack([var.get_gvec() for var in self.variables])

    def get_spline_uvec(self) -> np.ndarray:
        uvec = np.hstack([
            var.get_spline_uvec() if isinstance(var, SplineVariable) else np.empty((2, 0))
            for var in self.variables
        ])
        return uvec

    def get_spline_gvec(self) -> np.ndarray:
        gvec = np.hstack([
            var.get_spline_gvec() if isinstance(var, SplineVariable) else np.empty((2, 0))
            for var in self.variables
        ])
        return gvec

    def get_spline_umat(self) -> np.ndarray:
        umat = block_diag(*[
            var.get_spline_umat() if isinstance(var, SplineVariable) else np.empty((0, 1))
            for var in self.variables
        ])
        return umat

    def get_spline_gmat(self) -> np.ndarray:
        gmat = block_diag(*[
            var.get_spline_gmat() if isinstance(var, SplineVariable) else np.empty((0, 1))
            for var in self.variables
        ])
        return gmat

    def get_lin_param(self, coefs: np.ndarray, data: Data,
                      mat: np.ndarray = None,
                      return_mat: bool = False) -> Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]:
        if mat is None:
            mat = self.get_mat(data)
        lin_param = mat.dot(coefs)
        if self.use_offset:
            lin_param += data.offset
        if return_mat:
            return lin_param, mat
        return lin_param

    def get_param(self, coefs: np.ndarray, data: Data,
                  mat: np.ndarray = None) -> np.ndarray:
        lin_param = self.get_lin_param(coefs, data, mat)
        return self.inv_link.fun(lin_param)

    def get_dparam(self, coefs: np.ndarray, data: Data,
                   mat: np.ndarray = None) -> np.ndarray:
        lin_param, mat = self.get_lin_param(coefs, data, mat, return_mat=True)
        return self.inv_link.dfun(lin_param)[:, None]*mat

    def get_d2param(self, coefs: np.ndarray, data: Data,
                    mat: np.ndarray = None) -> np.ndarray:
        lin_param, mat = self.get_lin_param(coefs, data, mat, return_mat=True)
        return self.inv_link.d2fun(lin_param)[:, None, None]*(mat[..., None]*mat[:, None, :])
