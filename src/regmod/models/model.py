"""
Model module
"""
from itertools import chain
from typing import Callable, Optional, Union

import numpy as np
import pandas as pd
from msca.linalg.matrix import Matrix
from numpy.typing import NDArray
from scipy.linalg import block_diag
from scipy.sparse import csc_matrix

from regmod.optimizer import scipy_optimize
from regmod.parameter import Parameter
from regmod.utils import sizes_to_slices


class Model:
    """Model class that in charge of gathering all information, fit and predict.

    Parameters
    ----------
    y
        Column name for the observation
    weights
        Column name for the weight of each observation
    params : Optional[list[Parameter]], optional
        A list of parameters. Default to None.
    param_specs : Optional[dict[str, dict]], optional
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
        weights: Optional[str] = None,
        params: Optional[list[Parameter]] = None,
        param_specs: Optional[dict[str, dict]] = None,
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
        self.y = y
        self.weights = weights
        self.trim_weights = None
        self._data = {}

        self.sizes = [param.size for param in self.params]
        self.indices = sizes_to_slices(self.sizes)
        self.size = sum(self.sizes)
        self.num_params = len(self.params)

        # optimization result placeholder
        self.opt_result = None
        self._opt_coefs = None
        self._opt_vcov = None

    def _validate_data(self, df: pd.DataFrame, require_y: bool = True):
        required_cols = set(
            chain(*[[v.name for v in param.variables] for param in self.params])
        )
        if require_y:
            required_cols.add(self.y)
        if "intercept" in required_cols:
            required_cols.remove("intercept")
        for col in required_cols:
            if col not in df:
                raise KeyError(f"missing column {col}")

    def _attach(self, df: pd.DataFrame, require_y: bool = True):
        self._validate_data(df)
        self._clear()

        for param in self.params:
            param.check_data(df)

        self._data.update(
            {
                "mat": [param.get_mat(df) for param in self.params],
                "offset": [param.get_offset(df) for param in self.params],
                "uvec": np.hstack([param.get_uvec() for param in self.params]),
                "gvec": np.hstack([param.get_gvec() for param in self.params]),
                "linear_uvec": np.hstack(
                    [param.get_linear_uvec() for param in self.params]
                ),
                "linear_gvec": np.hstack(
                    [param.get_linear_gvec() for param in self.params]
                ),
                "linear_umat": block_diag(
                    *[param.get_linear_umat() for param in self.params]
                ),
                "linear_gmat": block_diag(
                    *[param.get_linear_gmat() for param in self.params]
                ),
            }
        )

        if require_y:
            self._data["y"] = df[self.y].to_numpy()
        self._data["weights"] = np.ones(len(df))
        if self.weights is not None:
            self._data["weights"] = df[self.weights].to_numpy()

        self.use_hessian = not any(isinstance(m, csc_matrix) for m in self._data["mat"])
        self.trim_weights = np.ones(df.shape[0])

    def _clear(self) -> None:
        self._data.clear()

    @property
    def opt_coefs(self) -> Union[None, NDArray]:
        return self._opt_coefs

    @opt_coefs.setter
    def opt_coefs(self, coefs: NDArray):
        coefs = np.asarray(coefs)
        if coefs.size != self.size:
            raise ValueError("Coefficients size not match.")
        self._opt_coefs = coefs

    @property
    def opt_vcov(self) -> Union[None, NDArray]:
        return self._opt_vcov

    @opt_vcov.setter
    def opt_vcov(self, vcov: NDArray):
        vcov = np.asarray(vcov)
        self._opt_vcov = vcov

    def get_vcov(self, coefs: NDArray) -> NDArray:
        hessian = self.hessian(coefs)
        if isinstance(hessian, Matrix):
            hessian = hessian.to_numpy()
        eig_vals, eig_vecs = np.linalg.eig(hessian)
        if np.isclose(eig_vals, 0.0).any():
            raise ValueError(
                "singular Hessian matrix, please add priors or "
                "reduce number of variables"
            )
        inv_hessian = (eig_vecs / eig_vals).dot(eig_vecs.T)

        jacobian2 = self.jacobian2(coefs)
        if isinstance(jacobian2, Matrix):
            jacobian2 = jacobian2.to_numpy()
        eig_vals = np.linalg.eigvals(jacobian2)
        if np.isclose(eig_vals, 0.0).any():
            raise ValueError(
                "singular Jacobian matrix, please add priors or "
                "reduce number of variables"
            )

        vcov = inv_hessian.dot(jacobian2)
        vcov = inv_hessian.dot(vcov.T)
        return vcov

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
            param.get_param(coefs[i], self._data["offset"][i], mat=self._data["mat"][i])
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
            param.get_dparam(
                coefs[i], self._data["offset"][i], mat=self._data["mat"][i]
            )
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
            param.get_d2param(
                coefs[i], self._data["offset"][i], mat=self._data["mat"][i]
            )
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
        obs = self._data["y"]
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
        val = 0.5 * np.sum(
            (coefs - self._data["gvec"][0]) ** 2 / self._data["gvec"][1] ** 2
        )
        if self._data["linear_gvec"].size > 0:
            val += 0.5 * np.sum(
                (self._data["linear_gmat"].dot(coefs) - self._data["linear_gvec"][0])
                ** 2
                / self._data["linear_gvec"][1] ** 2
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
        grad = (coefs - self._data["gvec"][0]) / self._data["gvec"][1] ** 2
        if self._data["linear_gvec"].size > 0:
            grad += (
                self._data["linear_gmat"].T / self._data["linear_gvec"][1] ** 2
            ).dot(self._data["linear_gmat"].dot(coefs) - self._data["linear_gvec"][0])
        return grad

    def hessian_from_gprior(self) -> NDArray:
        """Hessian matrix from the Gaussian prior.

        Returns
        -------
        NDArray
            Hessian matrix.
        """
        hess = np.diag(1.0 / self._data["gvec"][1] ** 2)
        if self._data["linear_gvec"].size > 0:
            hess += (
                self._data["linear_gmat"].T / self._data["linear_gvec"][1] ** 2
            ).dot(self._data["linear_gmat"])
        return hess

    def get_nll_terms(self, coefs: NDArray) -> NDArray:
        params = self.get_params(coefs)
        nll_terms = self.nll(params)
        nll_terms = self._data["weights"] * nll_terms
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
        weights = self._data["weights"] * self.trim_weights
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
        weights = self._data["weights"] * self.trim_weights
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
        weights = self._data["weights"] * self.trim_weights
        jacobian = np.vstack(
            [dparams[i].T * (weights * grad_params[i]) for i in range(self.num_params)]
        )
        jacobian2 = jacobian.dot(jacobian.T) + self.hessian_from_gprior()
        return jacobian2

    def fit(
        self,
        df: pd.DataFrame,
        optimizer: Callable = scipy_optimize,
        **optimizer_options,
    ):
        """Fit function.

        Parameters
        ----------
        optimizer : Callable, optional
            Model solver, by default scipy_optimize.
        """
        self._attach(df)
        if self.size == 0:
            self.opt_coefs = np.empty((0,))
            self.opt_vcov = np.empty((0, 0))
            self.opt_result = "no parameter to fit"
            return
        optimizer(self, **optimizer_options)
        self._clear()

    def predict(self, df: pd.DataFrame) -> pd.DataFrame:
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
        self._attach(df, require_y=False)
        df = df.copy()

        coefs = self.split_coefs(self.opt_coefs)
        for i, param_name in enumerate(self.param_names):
            df[param_name] = self.params[i].get_param(
                coefs[i], self._data["offset"][i], self._data["mat"][i]
            )
        self._clear()

        return df
