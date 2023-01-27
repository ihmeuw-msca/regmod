"""
Model module
"""
from typing import Callable, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
from msca.linalg.matrix import Matrix
from numpy import ndarray
from scipy.linalg import block_diag
from scipy.sparse import csc_matrix

from regmod.optimizer import scipy_optimize
from regmod.parameter import Parameter
from regmod.utils import sizes_to_slices


class Model:
    """Model class that in charge of gathering all information, fit and predict.

    Parameters
    ----------
    obs
        Column name for the observation
    weights
        Column name for the weight of each observation
    data
        Data frame that contains all information
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

    """

    param_names: Tuple[str] = None
    default_param_specs: Dict[str, Dict] = None

    def __init__(self,
                 y: str,
                 weights: str = "weights",
                 df: Optional[pd.DataFrame] = None,
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
        self._y = y
        self._weights = weights
        self.df = df
        self.y = None
        self.weights = None
        self.trim_weights = None
        if self.df is not None:
            self.attach_df(self.df)

        self.sizes = [param.size for param in self.params]
        self.indices = sizes_to_slices(self.sizes)
        self.size = sum(self.sizes)
        self.num_params = len(self.params)

        # optimization result placeholder
        self.opt_result = None
        self._opt_coefs = None
        self._opt_vcov = None

    def attach_df(self, df: pd.DataFrame):
        self.df = df
        for param in self.params:
            param.check_data(df)

        self.mat = self.get_mat()
        self.use_hessian = not any(isinstance(m, csc_matrix) for m in self.mat)
        self.uvec = self.get_uvec()
        self.gvec = self.get_gvec()
        self.linear_uvec = self.get_linear_uvec()
        self.linear_gvec = self.get_linear_gvec()
        self.linear_umat = self.get_linear_umat()
        self.linear_gmat = self.get_linear_gmat()
        self.trim_weights = np.ones(df.shape[0])
        self.y = self.df[self._y].to_numpy()
        if self._weights not in self.df:
            self.weights = np.ones(self.df.shape[0])
        else:
            self.weights = self.df[self._weights].to_numpy()

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
        return self._opt_vcov

    @opt_vcov.setter
    def opt_vcov(self, vcov: ndarray):
        vcov = np.asarray(vcov)
        self._opt_vcov = vcov

    def get_vcov(self, coefs: ndarray) -> ndarray:
        hessian = self.hessian(coefs)
        if isinstance(hessian, Matrix):
            hessian = hessian.to_numpy()
        eig_vals, eig_vecs = np.linalg.eig(hessian)
        if np.isclose(eig_vals, 0.0).any():
            raise ValueError("singular Hessian matrix, please add priors or "
                             "reduce number of variables")
        inv_hessian = (eig_vecs / eig_vals).dot(eig_vecs.T)

        jacobian2 = self.jacobian2(coefs)
        if isinstance(jacobian2, Matrix):
            jacobian2 = jacobian2.to_numpy()
        eig_vals = np.linalg.eigvals(jacobian2)
        if np.isclose(eig_vals, 0.0).any():
            raise ValueError("singular Jacobian matrix, please add priors or "
                             "reduce number of variables")

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
        return [param.get_mat(self.df) for param in self.params]

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
        return [param.get_param(coefs[i], self.df, mat=self.mat[i])
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
        return [param.get_dparam(coefs[i], self.df, mat=self.mat[i])
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
        return [param.get_d2param(coefs[i], self.df, mat=self.mat[i])
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
        obs = self.df[self.y].to_numpy()
        return (obs < ui[0]) | (obs > ui[1])

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

    def get_nll_terms(self, coefs: ndarray) -> ndarray:
        params = self.get_params(coefs)
        nll_terms = self.nll(params)
        nll_terms = self.weights*nll_terms
        return nll_terms

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
        nll_terms = self.get_nll_terms(coefs)
        return self.trim_weights.dot(nll_terms) + \
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
        weights = self.weights*self.trim_weights
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
        weights = self.weights*self.trim_weights
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
        weights = self.weights*self.trim_weights
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
        if self.size == 0:
            self.opt_coefs = np.empty((0,))
            self.opt_vcov = np.empty((0, 0))
            self.opt_result = "no parameter to fit"
            return
        optimizer(self, **optimizer_options)

    def predict(self, df: Optional[pd.DataFrame] = None) -> pd.DataFrame:
        """Predict the parameters.

        Parameters
        ----------
        df : pd.DataFrame, optional
            Data Frame with prediction data. If it is None, using the training
            data.

        Returns
        -------
        pd.DataFrame
            Data frame with predicted parameters.
        """
        if df is None:
            df = self.df
        df = df.copy()

        coefs = self.split_coefs(self.opt_coefs)
        for i, param_name in enumerate(self.param_names):
            df[param_name] = self.params[i].get_param(coefs[i], df)

        return df

    def __repr__(self) -> str:
        return (f"{type(self).__name__}("
                f"bnum_y={self.df.shape[0]}, "
                f"num_params={self.num_params}, "
                f"size={self.size})")
