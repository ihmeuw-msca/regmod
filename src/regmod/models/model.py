"""
Model module
"""
from itertools import chain
from typing import Callable, Optional

import numpy as np
import pandas as pd
from msca.linalg.matrix import Matrix
from numpy.typing import NDArray

from regmod.optimizer import scipy_optimize
from regmod.parameter import Parameter
from regmod.utils import sizes_to_slices

from .data import parse_to_numpy


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

        self.sizes = [param.size for param in self.params]
        self.indices = sizes_to_slices(self.sizes)
        self.size = sum(self.sizes)
        self.num_params = len(self.params)

        # optimization result placeholder
        self.result, self._coef, self._vcov = None, None, None

    def _validate_data(self, df: pd.DataFrame, fit: bool = True):
        required_cols = set(
            chain(*[[v.name for v in param.variables] for param in self.params])
        )
        if fit:
            required_cols.add(self.y)
            if self.weights is not None:
                required_cols.add(self.weights)
        if "intercept" in required_cols:
            required_cols.remove("intercept")
        for col in required_cols:
            if col not in df:
                raise KeyError(f"missing column {col}")
            if any(df[col].isna()):
                raise ValueError(f"{col} contains nan")
        if fit and self.weights is not None:
            if not all(df[self.weights] >= 0):
                raise ValueError(f"weights in {self.weights} should be non-negative")

    def _parse(self, df: pd.DataFrame, fit: bool = True) -> dict:
        self._validate_data(df)

        return parse_to_numpy(df, self.y, self.params, self.weights, fit=fit)

    @property
    def coef(self) -> Optional[NDArray]:
        return self._coef

    @coef.setter
    def coef(self, coef: NDArray):
        coef = np.asarray(coef)
        if coef.size != self.size:
            raise ValueError("Coefficients size not match.")
        self._coef = coef

    @property
    def vcov(self) -> Optional[NDArray]:
        return self._vcov

    @vcov.setter
    def vcov(self, vcov: NDArray):
        vcov = np.asarray(vcov)
        self._vcov = vcov

    def get_vcov(self, data: dict, coef: NDArray) -> NDArray:
        hessian = self.hessian(data, coef)
        if isinstance(hessian, Matrix):
            hessian = hessian.to_numpy()
        eig_vals, eig_vecs = np.linalg.eig(hessian)
        if np.isclose(eig_vals, 0.0).any():
            raise ValueError(
                "singular Hessian matrix, please add priors or "
                "reduce number of variables"
            )
        inv_hessian = (eig_vecs / eig_vals).dot(eig_vecs.T)

        jacobian2 = self.jacobian2(data, coef)
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

    def split_coef(self, coef: NDArray) -> list[NDArray]:
        """Split coefficients into pieces for each parameter.

        Parameters
        ----------
        coef : NDArray
            The coefficients array.

        Returns
        -------
        list[NDArray]
            A list of splitted coefficients for each parameter.
        """
        assert len(coef) == self.size
        return [coef[index] for index in self.indices]

    def get_params(self, data: dict, coef: NDArray) -> list[NDArray]:
        """Get the parameters.

        Parameters
        ----------
        coef : NDArray
            The coefficients array.

        Returns
        -------
        list[NDArray]
            The parameters.
        """
        coef = self.split_coef(coef)
        return [
            param.get_param(coef[i], data["offset"][i], mat=data["mat"][i])
            for i, param in enumerate(self.params)
        ]

    def get_dparams(self, data: dict, coef: NDArray) -> list[NDArray]:
        """Get the derivative of the parameters.

        Parameters
        ----------
        coef : NDArray
            The coefficients array.

        Returns
        -------
        list[NDArray]
            The derivative of the parameters.
        """
        coef = self.split_coef(coef)
        return [
            param.get_dparam(coef[i], data["offset"][i], mat=data["mat"][i])
            for i, param in enumerate(self.params)
        ]

    def get_d2params(self, data: dict, coef: NDArray) -> list[NDArray]:
        """Get the second order derivative of the parameters.

        Parameters
        ----------
        coef : NDArray
            The coefficients array.

        Returns
        -------
        list[NDArray]
            The second order derivative of the parameters.
        """
        coef = self.split_coef(coef)
        return [
            param.get_d2param(coef[i], data["offset"][i], mat=data["mat"][i])
            for i, param in enumerate(self.params)
        ]

    def nll(self, data: dict, params: list[NDArray]) -> NDArray:
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

    def dnll(self, data: dict, params: list[NDArray]) -> list[NDArray]:
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

    def d2nll(self, data: dict, params: list[NDArray]) -> list[list[NDArray]]:
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

    def detect_outliers(
        self, data: dict, coef: NDArray, bounds: tuple[float, float]
    ) -> NDArray:
        """Detect outliers.

        Parameters
        ----------
        coef : NDArray
            Given coefficients.
        bounds : tuple[float, float]
            Quantile bounds for the inliers.

        Returns
        -------
        NDArray
            A boolean array that indicate if observations are outliers.
        """
        params = self.get_params(data, coef)
        ui = self.get_ui(params, bounds)
        obs = data["y"]
        return (obs < ui[0]) | (obs > ui[1])

    def objective_from_gprior(self, data: dict, coef: NDArray) -> float:
        """Objective function from the Gaussian priors.

        Parameters
        ----------
        coef : NDArray
            Given coefficients.

        Returns
        -------
        float
            Objective function value.
        """
        val = 0.5 * np.sum((coef - data["gvec"][0]) ** 2 / data["gvec"][1] ** 2)
        if data["linear_gvec"].size > 0:
            val += 0.5 * np.sum(
                (data["linear_gmat"].dot(coef) - data["linear_gvec"][0]) ** 2
                / data["linear_gvec"][1] ** 2
            )
        return val

    def gradient_from_gprior(self, data: dict, coef: NDArray) -> NDArray:
        """Graident function from the Gaussian priors.

        Parameters
        ----------
        coef : NDArray
            Given coefficients.

        Returns
        -------
        NDArray
            Graident vector.
        """
        grad = (coef - data["gvec"][0]) / data["gvec"][1] ** 2
        if data["linear_gvec"].size > 0:
            grad += (data["linear_gmat"].T / data["linear_gvec"][1] ** 2).dot(
                data["linear_gmat"].dot(coef) - data["linear_gvec"][0]
            )
        return grad

    def hessian_from_gprior(self, data: dict) -> NDArray:
        """Hessian matrix from the Gaussian prior.

        Returns
        -------
        NDArray
            Hessian matrix.
        """
        hess = np.diag(1.0 / data["gvec"][1] ** 2)
        if data["linear_gvec"].size > 0:
            hess += (data["linear_gmat"].T / data["linear_gvec"][1] ** 2).dot(
                data["linear_gmat"]
            )
        return hess

    def get_nll_terms(self, data: dict, coef: NDArray) -> NDArray:
        params = self.get_params(data, coef)
        nll_terms = self.nll(data, params)
        nll_terms = data["weights"] * nll_terms
        return nll_terms

    def objective(self, data: dict, coef: NDArray) -> float:
        """Objective function.

        Parameters
        ----------
        coef : NDArray
            Given coefficients.

        Returns
        -------
        float
            Objective value.
        """
        nll_terms = self.get_nll_terms(data, coef)
        return data["trim_weights"].dot(nll_terms) + self.objective_from_gprior(
            data, coef
        )

    def gradient(self, data: dict, coef: NDArray) -> NDArray:
        """Gradient function.

        Parameters
        ----------
        coef : NDArray
            Given coefficients.

        Returns
        -------
        NDArray
            Gradient vector.
        """
        params = self.get_params(data, coef)
        dparams = self.get_dparams(data, coef)
        grad_params = self.dnll(data, params)
        weights = data["weights"] * data["trim_weights"]
        return np.hstack(
            [dparams[i].T.dot(weights * grad_params[i]) for i in range(self.num_params)]
        ) + self.gradient_from_gprior(data, coef)

    def hessian(self, data: dict, coef: NDArray) -> NDArray:
        """Hessian function.

        Parameters
        ----------
        coef : NDArray
            Given coefficients.

        Returns
        -------
        NDArray
            Hessian matrix.
        """
        params = self.get_params(data, coef)
        dparams = self.get_dparams(data, coef)
        d2params = self.get_d2params(data, coef)
        grad_params = self.dnll(data, params)
        hess_params = self.d2nll(data, params)
        weights = data["weights"] * data["trim_weights"]
        hess = [
            [
                (dparams[i].T * (weights * hess_params[i][j])).dot(dparams[j])
                for j in range(self.num_params)
            ]
            for i in range(self.num_params)
        ]
        for i in range(self.num_params):
            hess[i][i] += np.tensordot(weights * grad_params[i], d2params[i], axes=1)
        return np.block(hess) + self.hessian_from_gprior(data)

    def jacobian2(self, data: dict, coef: NDArray) -> NDArray:
        """Jacobian function.

        Parameters
        ----------
        coef : NDArray
            Given coefficients.

        Returns
        -------
        NDArray
            Jacobian matrix.
        """
        params = self.get_params(data, coef)
        dparams = self.get_dparams(data, coef)
        grad_params = self.dnll(data, params)
        weights = data["weights"] * data["trim_weights"]
        jacobian = np.vstack(
            [dparams[i].T * (weights * grad_params[i]) for i in range(self.num_params)]
        )
        jacobian2 = jacobian.dot(jacobian.T) + self.hessian_from_gprior(data)
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
        data = self._parse(df)
        if self.size == 0:
            self.coef = np.empty((0,))
            self.vcov = np.empty((0, 0))
            self.result = "no parameter to fit"
            return
        optimizer(self, data, **optimizer_options)

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
        data = self._parse(df, fit=False)
        df = df.copy()

        coef = self.split_coef(self.coef)
        for i, param_name in enumerate(self.param_names):
            df[param_name] = self.params[i].get_param(
                coef[i], data["offset"][i], data["mat"][i]
            )

        return df
