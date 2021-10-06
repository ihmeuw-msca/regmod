"""
Parameter module
"""
from dataclasses import dataclass, field
from typing import List, Tuple, Union

import numpy as np
from scipy.linalg import block_diag

from regmod.data import Data
from regmod.function import SmoothFunction, fun_dict
from regmod.prior import LinearGaussianPrior, LinearUniformPrior
from regmod.variable import SplineVariable, Variable


@dataclass
class Parameter:
    """Parameter class is used for storing variable information including how
    variable parametrized the parameter and then in turn used in the likelihood
    building.

    Parameters
    ----------
    name : str
        Name of the parameter.
    variables : List[Variable]
        A list of variables that parametrized the parameter.
    inv_link : Union[str, SmoothFunction]
        Inverse link funcion to link variables to parameter.
    use_offset : bool, optional
        If `True` when build parameter, it will use the offset in the data
        object. Default to `False`.
    linear_gpriors : List[LinearGaussianPrior], optional
        A list of linear Gaussian priors. Default to an empty list.
    linear_upriors : List[LinearUniformPrior], optional
        A list of linear Uniform priors. Default to an empty list.

    Attributes
    ----------
    name : str
        Name of the parameter.
    variables : List[Variable]
        A list of variables that parametrized the parameter.
    inv_link : Union[str, SmoothFunction]
        Inverse link funcion to link variables to parameter.
    use_offset : bool
        If `True` when build parameter, it will use the offset in the data
        object.
    linear_gpriors : List[LinearGaussianPrior]
        A list of linear Gaussian priors.
    linear_upriors : List[LinearUniformPrior]
        A list of linear Uniform priors.
    size : int
        Size of the parameter. It equals to sum of the sizes of the variables.

    Methods
    -------
    check_data(data)
        Attach data to all variables.
    get_mat(data)
        Get the design matrix.
    get_uvec()
        Get the direct Uniform prior.
    get_gvec()
        Get the direct Gaussian prior.
    get_linear_uvec()
        Get the linear Uniform prior vector.
    get_linear_gvec()
        Get the linear Gaussian prior vector.
    get_linear_umat()
        Get the linear Uniform prior matrix.
    get_linear_gmat()
        Get the linear Gaussian prior matrix.
    get_lin_param(coefs, data, mat=None, return_mat=False)
        Get the parameter before apply the link function.
    get_param(coefs, data, mat=None)
        Get the parameter.
    get_dparam(coefs, data, mat=None)
        Get the derivative of the parameter.
    get_d2param(coefs, data, mat=None)
        Get the second order derivative of the parameter.
    """

    name: str
    variables: List[Variable] = field(repr=False)
    inv_link: Union[str, SmoothFunction] = field(repr=False)
    use_offset: bool = field(default=False, repr=False)
    linear_gpriors: List[LinearGaussianPrior] = field(default_factory=list, repr=False)
    linear_upriors: List[LinearUniformPrior] = field(default_factory=list, repr=False)

    def __post_init__(self):
        if isinstance(self.inv_link, str):
            self.inv_link = fun_dict[self.inv_link]
        assert isinstance(self.inv_link, SmoothFunction), \
            "inv_link has to be an instance of SmoothFunction."
        assert all([isinstance(prior, LinearGaussianPrior) for prior in self.linear_gpriors]), \
            "linear_gpriors has to be a list of LinearGaussianPrior."
        assert all([isinstance(prior, LinearUniformPrior) for prior in self.linear_upriors]), \
            "linear_upriors has to be a list of LinearUniformPrior."
        assert all([prior.mat.shape[1] == self.size for prior in self.linear_gpriors]), \
            "linear_gpriors size not match."
        assert all([prior.mat.shape[1] == self.size for prior in self.linear_upriors]), \
            "linear_upriors size not match."
        assert all([isinstance(var, Variable) for var in self.variables]), \
            "variables has to be a list of Variable."

    @property
    def size(self) -> int:
        """Size of the parameter."""
        return sum([var.size for var in self.variables])

    def check_data(self, data: Data):
        """Attach data to all variables.

        Parameters
        ----------
        data : Data
            Data object.
        """
        for var in self.variables:
            var.check_data(data)

    def get_mat(self, data: Data) -> np.ndarray:
        """Get the design matrix.

        Parameters
        ----------
        data : Data
            Data object.

        Returns
        -------
        np.ndarray
            The design matrix.
        """
        return np.hstack([var.get_mat(data) for var in self.variables])

    def get_uvec(self) -> np.ndarray:
        """Get the direct Uniform prior.

        Returns
        -------
        np.ndarray
            Uniform prior information array.
        """
        return np.hstack([var.get_uvec() for var in self.variables])

    def get_gvec(self) -> np.ndarray:
        """Get the direct Gaussian prior.

        Returns
        -------
        np.ndarray
            Gaussian prior information array.
        """
        return np.hstack([var.get_gvec() for var in self.variables])

    def get_linear_uvec(self) -> np.ndarray:
        """Get the linear Uniform prior vector.

        Returns
        -------
        np.ndarray
            Uniform prior information array.
        """
        uvec = np.hstack([
            var.get_linear_uvec() if isinstance(var, SplineVariable) else np.empty((2, 0))
            for var in self.variables
        ])
        if len(self.linear_upriors) > 0:
            uvec = np.hstack([uvec] + [np.vstack([prior.lb, prior.ub])
                                       for prior in self.linear_upriors])
        return uvec

    def get_linear_gvec(self) -> np.ndarray:
        """Get the linear Gaussian prior vector.

        Returns
        -------
        np.ndarray
            Gaussian prior information array.
        """
        gvec = np.hstack([
            var.get_linear_gvec() if isinstance(var, SplineVariable) else np.empty((2, 0))
            for var in self.variables
        ])
        if len(self.linear_gpriors) > 0:
            gvec = np.hstack([gvec] + [np.vstack([prior.mean, prior.sd])
                                       for prior in self.linear_gpriors])
        return gvec

    def get_linear_umat(self) -> np.ndarray:
        """Get the linear Uniform prior matrix.

        Returns
        -------
        np.ndarray
            Uniform prior design matrix.
        """
        umat = block_diag(*[
            var.get_linear_umat() if isinstance(var, SplineVariable) else np.empty((0, 1))
            for var in self.variables
        ])
        if len(self.linear_upriors) > 0:
            umat = np.vstack([umat] + [prior.mat for prior in self.linear_upriors])
        return umat

    def get_linear_gmat(self) -> np.ndarray:
        """Get the linear Gaussian prior matrix.

        Returns
        -------
        np.ndarray
            Gaussian prior design matrix.
        """
        gmat = block_diag(*[
            var.get_linear_gmat() if isinstance(var, SplineVariable) else np.empty((0, 1))
            for var in self.variables
        ])
        if len(self.linear_gpriors) > 0:
            gmat = np.vstack([gmat] + [prior.mat for prior in self.linear_gpriors])
        return gmat

    def get_lin_param(self,
                      coefs: np.ndarray,
                      data: Data,
                      mat: np.ndarray = None,
                      return_mat: bool = False) -> Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]:
        """Get the parameter before apply the link function.

        Parameters
        ----------
        coefs : np.ndarray
            Coefficients for the design matrix.
        data : Data
            Data object.
        mat : np.ndarray, optional
            Alternative design matrix, by default None.
        return_mat : bool, optional
            If `True`, return the created design matrix, by default False.

        Returns
        -------
        Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]
            Linear parameter vector, or when `return_mat=True` also returns the
            design matrix.
        """
        if mat is None:
            mat = self.get_mat(data)
        lin_param = mat.dot(coefs)
        if self.use_offset:
            lin_param += data.offset
        if return_mat:
            return lin_param, mat
        return lin_param

    def get_param(self,
                  coefs: np.ndarray,
                  data: Data,
                  mat: np.ndarray = None) -> np.ndarray:
        """Get the parameter.

        Parameters
        ----------
        coefs : np.ndarray
            Coefficients for the design matrix.
        data : Data
            Data object.
        mat : np.ndarray, optional
            Alternative design matrix, by default None.

        Returns
        -------
        np.ndarray
            Returns the parameter.
        """
        lin_param = self.get_lin_param(coefs, data, mat)
        return self.inv_link.fun(lin_param)

    def get_dparam(self,
                   coefs: np.ndarray,
                   data: Data,
                   mat: np.ndarray = None) -> np.ndarray:
        """Get the derivative of the parameter.

        Parameters
        ----------
        coefs : np.ndarray
            Coefficients for the design matrix.
        data : Data
            Data object.
        mat : np.ndarray, optional
            Alternative design matrix, by default None.

        Returns
        -------
        np.ndarray
            Returns the derivative of the parameter.
        """
        lin_param, mat = self.get_lin_param(coefs, data, mat, return_mat=True)
        return self.inv_link.dfun(lin_param)[:, None]*mat

    def get_d2param(self,
                    coefs: np.ndarray,
                    data: Data,
                    mat: np.ndarray = None) -> np.ndarray:
        """Get the second order derivative of the parameter.

        Parameters
        ----------
        coefs : np.ndarray
            Coefficients for the design matrix.
        data : Data
            Data object.
        mat : np.ndarray, optional
            Alternative design matrix, by default None.

        Returns
        -------
        np.ndarray
            Returns the second order derivative of the parameter.
        """
        lin_param, mat = self.get_lin_param(coefs, data, mat, return_mat=True)
        return self.inv_link.d2fun(lin_param)[:, None, None]*(mat[..., None]*mat[:, None, :])
