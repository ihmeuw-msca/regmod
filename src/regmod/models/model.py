"""
Model module
"""
from typing import Callable, Dict, List, Tuple, Union, Optional

import numpy as np
from numpy import ndarray
from scipy.linalg import block_diag

from regmod.data import Data
from regmod.parameter import Parameter
from regmod.utils import sizes_to_slices
from regmod.optimizer import scipy_optimize


class Model:
    """Model class that in charge of gathering all information, fit and predict.

    Parameters
    ----------
    data : Data
        Data object.
    params : Optional[List[Parameter]], optional
        A list of parameters. Default to None.
    param_specs : Optional[Dict[str, Dict]], optional
        Dictionary of all parameter specifications. Default to None.

    Raises
    ------
    ValueError
        Raised when both params and param_specs are None.
    ValueError
        Raised when data object is empty.

    Attributes
    ----------
    data : Data
        Data object.
    params : List[Parameter]
        A list of parameters.
    sizes : List[int]
        A list of sizes for each parameter.
    indices : List[slice]
        A list of indices for each parameter vector.
    size : int
        Sum of sizes from all parameters.
    num_params : int
        Number of parameters.
    mat : np.ndarray
        Design matrix of the problem.
    uvec : np.ndarray
        Direct Uniform prior array.
    gvec : np.ndarray
        Direct Gaussian prior array.
    linear_umat : np.ndarray
        Linear Uniform prior design matrix.
    linear_gmat : np.ndarray
        Linear Gaussian prior design matrix.
    linear_uvec : np.ndarray
        Linear Uniform prior array.
    linear_gvec : np.ndarray
        Linear Gaussian prior array.
    opt_result : scipy.optimize.OptimizeResult
        Optmization result. Initialized as None.
    opt_coefs : np.ndarray
        Optimal coefficients.
    opt_vcov : np.ndarray
        Optimal variance covariance matrix.

    Methods
    -------
    get_mat()
        Get the design matrices.
    get_uvec()
        Get the direct Uniform prior array.
    get_gvec()
        Get the direct Gaussian prior array.
    get_linear_uvec()
        Get the linear Uniform prior array.
    get_linear_gvec()
        Get the linear Gaussian prior array.
    get_linear_umat()
        Get the linear Uniform prior design matrix.
    get_linear_gmat()
        Get the linear Gaussian prior design matrix.
    split_coefs(coefs)
        Split coefficients into pieces for each parameter.
    get_params(coefs)
        Get the parameters.
    get_dparams(coefs)
        Get the derivative of the parameters.
    get_d2params(coefs)
        Get the second order derivative of the parameters.
    nll(params)
        Negative log likelihood.
    dnll(params)
        Derivative of negative the log likelihood.
    d2nll(params)
        Second order derivative of the negative log likelihood.
    get_ui(params, bounds)
        Get uncertainty interval, used for the trimming algorithm.
    detect_outliers(coefs, bounds)
        Detect outliers.
    objective_from_gprior(coefs)
        Objective function from the Gaussian priors.
    gradient_from_gprior(coefs)
        Gradient function from the Gaussian priors.
    hessian_from_gprior()
        Hessian function from the Gaussian priors.
    objective(coefs)
        Objective function.
    gradient(coefs)
        Gradient function.
    hessian(coefs)
        Hessian function.
    jacobian2(coefs)
        Jacobian function.
    fit(optimizer, **optimizer_options)
        Fit function.
    """

    param_names: Tuple[str] = None
    default_param_specs: Dict[str, Dict] = None

    def __init__(self,
                 data: Data,
                 params: Optional[List[Parameter]] = None,
                 param_specs: Optional[Dict[str, Dict]] = None):
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
        if self.data.is_empty():
            raise ValueError("Please attach dataframe before creating model.")
        for param in self.params:
            param.check_data(self.data)

        self.sizes = [param.size for param in self.params]
        self.indices = sizes_to_slices(self.sizes)
        self.size = sum(self.sizes)
        self.num_params = len(self.params)

        self.mat = self.get_mat()
        self.uvec = self.get_uvec()
        self.gvec = self.get_gvec()
        self.linear_uvec = self.get_linear_uvec()
        self.linear_gvec = self.get_linear_gvec()
        self.linear_umat = self.get_linear_umat()
        self.linear_gmat = self.get_linear_gmat()

        # optimization result placeholder
        self.opt_result = None
        self._opt_coefs = None

    @property
    def opt_coefs(self) -> Union[None, ndarray]:
        return self._opt_coefs

    @opt_coefs.setter
    def opt_coefs(self, coefs: ndarray):
        coefs = np.asarray(coefs)
        if coefs.size != self.size:
            raise ValueError("Coefficients size not match.")
        self._opt_coefs = coefs

    @property
    def opt_vcov(self) -> Union[None, ndarray]:
        if self.opt_coefs is None:
            return None
        inv_hessian = np.linalg.pinv(self.hessian(self.opt_coefs))
        jacobian2 = self.jacobian2(self.opt_coefs)
        vcov = inv_hessian.dot(jacobian2)
        vcov = inv_hessian.dot(vcov.T)
        return vcov

    def get_mat(self) -> List[ndarray]:
        """Get the design matrices.

        Returns
        -------
        List[ndarray]
            The design matrices.
        """
        return [param.get_mat(self.data) for param in self.params]

    def get_uvec(self) -> ndarray:
        """Get the direct Uniform prior array.

        Returns
        -------
        ndarray
            The direct Uniform prior array.
        """
        return np.hstack([param.get_uvec() for param in self.params])

    def get_gvec(self) -> ndarray:
        """Get the direct Gaussian prior array.

        Returns
        -------
        ndarray
            The direct Gaussian prior array.
        """
        return np.hstack([param.get_gvec() for param in self.params])

    def get_linear_uvec(self) -> ndarray:
        """Get the linear Uniform prior array.

        Returns
        -------
        ndarray
            The linear Uniform prior array.
        """
        return np.hstack([param.get_linear_uvec() for param in self.params])

    def get_linear_gvec(self) -> ndarray:
        """Get the linear Gaussian prior array.

        Returns
        -------
        ndarray
            The linear Gaussian prior array.
        """
        return np.hstack([param.get_linear_gvec() for param in self.params])

    def get_linear_umat(self) -> ndarray:
        """Get the linear Uniform prior design matrix.

        Returns
        -------
        ndarray
            The linear Uniform prior design matrix.
        """
        return block_diag(*[param.get_linear_umat() for param in self.params])

    def get_linear_gmat(self) -> ndarray:
        """Get the linear Gaussian prior design matrix.

        Returns
        -------
        ndarray
            The linear Gaussian prior design matrix.
        """
        return block_diag(*[param.get_linear_gmat() for param in self.params])

    def split_coefs(self, coefs: ndarray) -> List[ndarray]:
        """Split coefficients into pieces for each parameter.

        Parameters
        ----------
        coefs : ndarray
            The coefficients array.

        Returns
        -------
        List[ndarray]
            A list of splitted coefficients for each parameter.
        """
        assert len(coefs) == self.size
        return [coefs[index] for index in self.indices]

    def get_params(self, coefs: ndarray) -> List[ndarray]:
        """Get the parameters.

        Parameters
        ----------
        coefs : ndarray
            The coefficients array.

        Returns
        -------
        List[ndarray]
            The parameters.
        """
        coefs = self.split_coefs(coefs)
        return [param.get_param(coefs[i], self.data, mat=self.mat[i])
                for i, param in enumerate(self.params)]

    def get_dparams(self, coefs: ndarray) -> List[ndarray]:
        """Get the derivative of the parameters.

        Parameters
        ----------
        coefs : ndarray
            The coefficients array.

        Returns
        -------
        List[ndarray]
            The derivative of the parameters.
        """
        coefs = self.split_coefs(coefs)
        return [param.get_dparam(coefs[i], self.data, mat=self.mat[i])
                for i, param in enumerate(self.params)]

    def get_d2params(self, coefs: ndarray) -> List[ndarray]:
        """Get the second order derivative of the parameters.

        Parameters
        ----------
        coefs : ndarray
            The coefficients array.

        Returns
        -------
        List[ndarray]
            The second order derivative of the parameters.
        """
        coefs = self.split_coefs(coefs)
        return [param.get_d2param(coefs[i], self.data, mat=self.mat[i])
                for i, param in enumerate(self.params)]

    def nll(self, params: List[ndarray]) -> ndarray:
        """Negative log likelihood.

        Parameters
        ----------
        params : List[ndarray]
            A list of parameters.

        Returns
        -------
        ndarray
            An array of negative log likelihood for each observation.
        """
        raise NotImplementedError()

    def dnll(self, params: List[ndarray]) -> List[ndarray]:
        """Derivative of negative the log likelihood.

        Parameters
        ----------
        params : List[ndarray]
            A list of parameters.

        Returns
        -------
        List[ndarray]
            A list of derivatives for each parameter and each observation.
        """
        raise NotImplementedError()

    def d2nll(self, params: List[ndarray]) -> List[List[ndarray]]:
        """Second order derivative of the negative log likelihood.

        Parameters
        ----------
        params : List[ndarray]
            A list of parameters.

        Returns
        -------
        List[List[ndarray]]
            A list of list of second order derivatives for each parameter and
            each observation.
        """
        raise NotImplementedError()

    def get_ui(self,
               params: List[ndarray],
               bounds: Tuple[float, float]) -> ndarray:
        """Get uncertainty interval, used for the trimming algorithm.

        Parameters
        ----------
        params : List[ndarray]
            A list of parameters
        bounds : Tuple[float, float]
            Quantile bounds for the uncertainty interval.

        Returns
        -------
        ndarray
            An array with uncertainty interval for each observation.
        """
        raise NotImplementedError()

    def detect_outliers(self,
                        coefs: ndarray,
                        bounds: Tuple[float, float]) -> ndarray:
        """Detect outliers.

        Parameters
        ----------
        coefs : ndarray
            Given coefficients.
        bounds : Tuple[float, float]
            Quantile bounds for the inliers.

        Returns
        -------
        ndarray
            A boolean array that indicate if observations are outliers.
        """
        params = self.get_params(coefs)
        ui = self.get_ui(params, bounds)
        return (self.data.obs < ui[0]) | (self.data.obs > ui[1])

    def objective_from_gprior(self, coefs: ndarray) -> float:
        """Objective function from the Gaussian priors.

        Parameters
        ----------
        coefs : ndarray
            Given coefficients.

        Returns
        -------
        float
            Objective function value.
        """
        val = 0.5*np.sum((coefs - self.gvec[0])**2/self.gvec[1]**2)
        if self.linear_gvec.size > 0:
            val += 0.5*np.sum((self.linear_gmat.dot(coefs) - self.linear_gvec[0])**2/self.linear_gvec[1]**2)
        return val

    def gradient_from_gprior(self, coefs: ndarray) -> ndarray:
        """Graident function from the Gaussian priors.

        Parameters
        ----------
        coefs : ndarray
            Given coefficients.

        Returns
        -------
        ndarray
            Graident vector.
        """
        grad = (coefs - self.gvec[0])/self.gvec[1]**2
        if self.linear_gvec.size > 0:
            grad += (self.linear_gmat.T/self.linear_gvec[1]**2).dot(self.linear_gmat.dot(coefs) - self.linear_gvec[0])
        return grad

    def hessian_from_gprior(self) -> ndarray:
        """Hessian matrix from the Gaussian prior.

        Returns
        -------
        ndarray
            Hessian matrix.
        """
        hess = np.diag(1.0/self.gvec[1]**2)
        if self.linear_gvec.size > 0:
            hess += (self.linear_gmat.T/self.linear_gvec[1]**2).dot(self.linear_gmat)
        return hess

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
        params = self.get_params(coefs)
        obj_params = self.nll(params)
        weights = self.data.weights*self.data.trim_weights
        return weights.dot(obj_params) + \
            self.objective_from_gprior(coefs)

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
        params = self.get_params(coefs)
        dparams = self.get_dparams(coefs)
        grad_params = self.dnll(params)
        weights = self.data.weights*self.data.trim_weights
        return np.hstack([
            dparams[i].T.dot(weights*grad_params[i])
            for i in range(self.num_params)
        ]) + self.gradient_from_gprior(coefs)

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

    def jacobian2(self, coefs: ndarray) -> ndarray:
        """Jacobian function.

        Parameters
        ----------
        coefs : ndarray
            Given coefficients.

        Returns
        -------
        ndarray
            Jacobian matrix.
        """
        params = self.get_params(coefs)
        dparams = self.get_dparams(coefs)
        grad_params = self.dnll(params)
        weights = self.data.weights*self.data.trim_weights
        jacobian = np.vstack([
            dparams[i].T*(weights*grad_params[i])
            for i in range(self.num_params)
        ])
        jacobian2 = jacobian.dot(jacobian.T) + self.hessian_from_gprior()
        return jacobian2

    def fit(self,
            optimizer: Callable = scipy_optimize,
            **optimizer_options):
        """Fit function.

        Parameters
        ----------
        optimizer : Callable, optional
            Model solver, by default scipy_optimize.
        """
        optimizer(self, **optimizer_options)
