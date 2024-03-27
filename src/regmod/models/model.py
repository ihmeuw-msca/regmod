"""
Model module
"""

import numpy as np
from scipy.linalg import block_diag
from scipy.sparse import csc_matrix

from regmod._typing import Callable, DataFrame, Matrix, NDArray
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
    params : list[Parameter], optional
        A list of parameters. Default to None.
    param_specs : dict[str, dict], optional
        dictionary of all parameter specifications. Default to None.

    Raises
    ------
    ValueError
        Raised when both params and param_specs are None.
    ValueError
        Raised when data object is empty.

    """

    param_names: tuple[str] = None
    default_param_specs: dict[str, dict] = None

    def __init__(
        self,
        y: str,
        weights: str = "weights",
        df: DataFrame | None = None,
        params: list[Parameter] | None = None,
        param_specs: dict[str, dict] | None = None,
    ):
        if params is None and param_specs is None:
            raise ValueError("Must provide `params` or `param_specs`")

        if params is not None:
            param_dict = {param.name: param for param in params}
            self.params = [param_dict[param_name] for param_name in self.param_names]
        else:
            self.params = [
                Parameter(
                    param_name,
                    **{
                        **self.default_param_specs[param_name],
                        **param_specs[param_name],
                    },
                )
                for param_name in self.param_names
            ]
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

    def attach_df(self, df: DataFrame):
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
    def opt_coefs(self) -> NDArray | None:
        return self._opt_coefs

    @opt_coefs.setter
    def opt_coefs(self, coefs: NDArray):
        coefs = np.asarray(coefs)
        if coefs.size != self.size:
            raise ValueError("Coefficients size not match.")
        self._opt_coefs = coefs

    @property
    def opt_vcov(self) -> NDArray | None:
        return self._opt_vcov

    @opt_vcov.setter
    def opt_vcov(self, vcov: NDArray):
        vcov = np.asarray(vcov)
        self._opt_vcov = vcov

    def get_vcov(self, coefs: NDArray) -> NDArray:
        hessian = self.hessian(coefs)
        if isinstance(hessian, Matrix):
            hessian = hessian.to_numpy()
        # We probably don't want to be eigendecomposing
        eig_vals, eig_vecs = np.linalg.eigh(hessian)
        if np.isclose(eig_vals, 0.0).any():
            raise ValueError(
                "singular Hessian matrix, please add priors or "
                "reduce number of variables"
            )
        inv_hessian = (eig_vecs / eig_vals).dot(eig_vecs.T)

        jacobian2 = self.jacobian2(coefs)
        if isinstance(jacobian2, Matrix):
            jacobian2 = jacobian2.to_numpy()
        eig_vals = np.linalg.eigvalsh(jacobian2)
        if np.isclose(eig_vals, 0.0).any():
            raise ValueError(
                "singular Jacobian matrix, please add priors or "
                "reduce number of variables"
            )

        vcov = inv_hessian.dot(jacobian2)
        vcov = inv_hessian.dot(vcov.T)
        return vcov

    def get_mat(self) -> list[NDArray]:
        """Get the design matrices.

        Returns
        -------
        list[NDArray]
            The design matrices.

        """
        return [param.get_mat(self.df) for param in self.params]

    def get_uvec(self) -> NDArray:
        """Get the direct Uniform prior array.

        Returns
        -------
        NDArray
            The direct Uniform prior array.

        """
        return np.hstack([param.get_uvec() for param in self.params])

    def get_gvec(self) -> NDArray:
        """Get the direct Gaussian prior array.

        Returns
        -------
        NDArray
            The direct Gaussian prior array.

        """
        return np.hstack([param.get_gvec() for param in self.params])

    def get_linear_uvec(self) -> NDArray:
        """Get the linear Uniform prior array.

        Returns
        -------
        NDArray
            The linear Uniform prior array.

        """
        return np.hstack([param.get_linear_uvec() for param in self.params])

    def get_linear_gvec(self) -> NDArray:
        """Get the linear Gaussian prior array.

        Returns
        -------
        NDArray
            The linear Gaussian prior array.

        """
        return np.hstack([param.get_linear_gvec() for param in self.params])

    def get_linear_umat(self) -> NDArray:
        """Get the linear Uniform prior design matrix.

        Returns
        -------
        NDArray
            The linear Uniform prior design matrix.

        """
        return block_diag(*[param.get_linear_umat() for param in self.params])

    def get_linear_gmat(self) -> NDArray:
        """Get the linear Gaussian prior design matrix.

        Returns
        -------
        NDArray
            The linear Gaussian prior design matrix.

        """
        return block_diag(*[param.get_linear_gmat() for param in self.params])

    def split_coefs(self, coefs: NDArray) -> list[NDArray]:
        """Split coefficients into pieces for each parameter.

        Parameters
        ----------
        coefs : NDArray
            The coefficients array.

        Returns
        -------
        list[NDArray]
            A list of splitted coefficients for each parameter.

        """
        assert len(coefs) == self.size
        return [coefs[index] for index in self.indices]

    def get_params(self, coefs: NDArray) -> list[NDArray]:
        """Get the parameters.

        Parameters
        ----------
        coefs : NDArray
            The coefficients array.

        Returns
        -------
        list[NDArray]
            The parameters.

        """
        coefs = self.split_coefs(coefs)
        return [
            param.get_param(coefs[i], self.df, mat=self.mat[i])
            for i, param in enumerate(self.params)
        ]

    def get_dparams(self, coefs: NDArray) -> list[NDArray]:
        """Get the derivative of the parameters.

        Parameters
        ----------
        coefs : NDArray
            The coefficients array.

        Returns
        -------
        list[NDArray]
            The derivative of the parameters.

        """
        coefs = self.split_coefs(coefs)
        return [
            param.get_dparam(coefs[i], self.df, mat=self.mat[i])
            for i, param in enumerate(self.params)
        ]

    def get_d2params(self, coefs: NDArray) -> list[NDArray]:
        """Get the second order derivative of the parameters.

        Parameters
        ----------
        coefs : NDArray
            The coefficients array.

        Returns
        -------
        list[NDArray]
            The second order derivative of the parameters.

        """
        coefs = self.split_coefs(coefs)
        return [
            param.get_d2param(coefs[i], self.df, mat=self.mat[i])
            for i, param in enumerate(self.params)
        ]

    def nll(self, params: list[NDArray]) -> NDArray:
        """Negative log likelihood.

        Parameters
        ----------
        params : list[NDArray]
            A list of parameters.

        Returns
        -------
        NDArray
            An array of negative log likelihood for each observation.

        """
        raise NotImplementedError()

    def dnll(self, params: list[NDArray]) -> list[NDArray]:
        """Derivative of negative the log likelihood.

        Parameters
        ----------
        params : list[NDArray]
            A list of parameters.

        Returns
        -------
        list[NDArray]
            A list of derivatives for each parameter and each observation.

        """
        raise NotImplementedError()

    def d2nll(self, params: list[NDArray]) -> list[list[NDArray]]:
        """Second order derivative of the negative log likelihood.

        Parameters
        ----------
        params : list[NDArray]
            A list of parameters.

        Returns
        -------
        list[list[NDArray]]
            A list of list of second order derivatives for each parameter and
            each observation.

        """
        raise NotImplementedError()

    def get_ui(self, params: list[NDArray], bounds: tuple[float, float]) -> NDArray:
        """Get uncertainty interval, used for the trimming algorithm.

        Parameters
        ----------
        params : list[NDArray]
            A list of parameters
        bounds : tuple[float, float]
            Quantile bounds for the uncertainty interval.

        Returns
        -------
        NDArray
            An array with uncertainty interval for each observation.

        """
        raise NotImplementedError()

    def detect_outliers(self, coefs: NDArray, bounds: tuple[float, float]) -> NDArray:
        """Detect outliers.

        Parameters
        ----------
        coefs : NDArray
            Given coefficients.
        bounds : tuple[float, float]
            Quantile bounds for the inliers.

        Returns
        -------
        NDArray
            A boolean array that indicate if observations are outliers.

        """
        params = self.get_params(coefs)
        ui = self.get_ui(params, bounds)
        obs = self.df[self.y].to_numpy()
        return (obs < ui[0]) | (obs > ui[1])

    def objective_from_gprior(self, coefs: NDArray) -> float:
        """Objective function from the Gaussian priors.

        Parameters
        ----------
        coefs : NDArray
            Given coefficients.

        Returns
        -------
        float
            Objective function value.

        """
        val = 0.5 * np.sum((coefs - self.gvec[0]) ** 2 / self.gvec[1] ** 2)
        if self.linear_gvec.size > 0:
            val += 0.5 * np.sum(
                (self.linear_gmat.dot(coefs) - self.linear_gvec[0]) ** 2
                / self.linear_gvec[1] ** 2
            )
        return val

    def gradient_from_gprior(self, coefs: NDArray) -> NDArray:
        """Graident function from the Gaussian priors.

        Parameters
        ----------
        coefs : NDArray
            Given coefficients.

        Returns
        -------
        NDArray
            Graident vector.

        """
        grad = (coefs - self.gvec[0]) / self.gvec[1] ** 2
        if self.linear_gvec.size > 0:
            grad += (self.linear_gmat.T / self.linear_gvec[1] ** 2).dot(
                self.linear_gmat.dot(coefs) - self.linear_gvec[0]
            )
        return grad

    def hessian_from_gprior(self) -> NDArray:
        """Hessian matrix from the Gaussian prior.

        Returns
        -------
        NDArray
            Hessian matrix.

        """
        hess = np.diag(1.0 / self.gvec[1] ** 2)
        if self.linear_gvec.size > 0:
            hess += (self.linear_gmat.T / self.linear_gvec[1] ** 2).dot(
                self.linear_gmat
            )
        return hess

    def get_nll_terms(self, coefs: NDArray) -> NDArray:
        params = self.get_params(coefs)
        nll_terms = self.nll(params)
        nll_terms = self.weights * nll_terms
        return nll_terms

    def objective(self, coefs: NDArray) -> float:
        """Objective function.

        Parameters
        ----------
        coefs : NDArray
            Given coefficients.

        Returns
        -------
        float
            Objective value.

        """
        nll_terms = self.get_nll_terms(coefs)
        return self.trim_weights.dot(nll_terms) + self.objective_from_gprior(coefs)

    def gradient(self, coefs: NDArray) -> NDArray:
        """Gradient function.

        Parameters
        ----------
        coefs : NDArray
            Given coefficients.

        Returns
        -------
        NDArray
            Gradient vector.

        """
        params = self.get_params(coefs)
        dparams = self.get_dparams(coefs)
        grad_params = self.dnll(params)
        weights = self.weights * self.trim_weights
        return np.hstack(
            [dparams[i].T.dot(weights * grad_params[i]) for i in range(self.num_params)]
        ) + self.gradient_from_gprior(coefs)

    def hessian(self, coefs: NDArray) -> NDArray:
        """Hessian function.

        Parameters
        ----------
        coefs : NDArray
            Given coefficients.

        Returns
        -------
        NDArray
            Hessian matrix.

        """
        params = self.get_params(coefs)
        dparams = self.get_dparams(coefs)
        d2params = self.get_d2params(coefs)
        grad_params = self.dnll(params)
        hess_params = self.d2nll(params)
        weights = self.weights * self.trim_weights
        hess = [
            [
                (dparams[i].T * (weights * hess_params[i][j])).dot(dparams[j])
                for j in range(self.num_params)
            ]
            for i in range(self.num_params)
        ]
        for i in range(self.num_params):
            hess[i][i] += np.tensordot(weights * grad_params[i], d2params[i], axes=1)
        return np.block(hess) + self.hessian_from_gprior()

    def jacobian2(self, coefs: NDArray) -> NDArray:
        """Jacobian function.

        Parameters
        ----------
        coefs : NDArray
            Given coefficients.

        Returns
        -------
        NDArray
            Jacobian matrix.

        """
        params = self.get_params(coefs)
        dparams = self.get_dparams(coefs)
        grad_params = self.dnll(params)
        weights = self.weights * self.trim_weights
        jacobian = np.vstack(
            [dparams[i].T * (weights * grad_params[i]) for i in range(self.num_params)]
        )
        jacobian2 = jacobian.dot(jacobian.T) + self.hessian_from_gprior()
        return jacobian2

    def fit(self, optimizer: Callable = scipy_optimize, **optimizer_options):
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

    def predict(self, df: DataFrame | None = None) -> DataFrame:
        """Predict the parameters.

        Parameters
        ----------
        df : DataFrame, optional
            Data Frame with prediction data. If it is None, using the training
            data.

        Returns
        -------
        DataFrame
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
        return (
            f"{type(self).__name__}("
            f"bnum_y={self.df.shape[0]}, "
            f"num_params={self.num_params}, "
            f"size={self.size})"
        )
