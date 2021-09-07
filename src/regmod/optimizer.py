"""
Optimizer module
"""
from typing import Callable, Dict
import numpy as np
from numpy import ndarray
from scipy.optimize import LinearConstraint, minimize


def scipy_optimize(model: "Model",
                   x0: ndarray = None,
                   options: Dict = None) -> ndarray:
    """Scipy trust-region optimizer.

    Parameters
    ----------
    model : Model
        Instance of `regmod.models.Model` class.
    x0 : ndarray, optional
        Initial guess for the variable, by default None. If `None` use zero
        vector as the initial guess.
    options : Dict, optional
        Scipy solver options, by default None.

    Returns
    -------
    ndarray
        Optimal solution.
    """
    x0 = np.zeros(model.size) if x0 is None else x0
    bounds = model.uvec.T
    constraints = [LinearConstraint(
        model.linear_umat,
        model.linear_uvec[0],
        model.linear_uvec[1]
    )] if model.linear_uvec.size > 0 else []

    result = minimize(model.objective, x0,
                      method="trust-constr",
                      jac=model.gradient,
                      hess=model.hessian,
                      constraints=constraints,
                      bounds=bounds,
                      options=options)

    model.opt_result = result
    model.opt_coefs = result.x.copy()
    return result.x


def set_trim_weights(model: "Model", index: ndarray, mask: float):
    """Set trimming weights to model object.

    Parameters
    ----------
    model : Model
        Instance of `regmod.models.Model` class.
    index : ndarray
        Index where the weights need to be set.
    mask : float
        Value of the weights to set.
    """
    weights = np.ones(model.data.num_obs)
    weights[index] = mask
    model.data.trim_weights = weights


def trimming(optimize: Callable) -> Callable:
    """Constructor of trimming solver.

    Parameters
    ----------
    optimize : Callable
        Optimization solver.

    Returns
    -------
    Callable
        Trimming optimization solver.
    """
    def optimize_with_trimming(model: "Model",
                               x0: ndarray = None,
                               options: Dict = None,
                               trim_steps: int = 3,
                               inlier_pct: float = 0.95) -> Dict[str, ndarray]:
        if trim_steps < 2:
            raise ValueError("At least two trimming steps.")
        if inlier_pct < 0.0 or inlier_pct > 1.0:
            raise ValueError("inlier_pct has to be between 0 and 1.")
        coefs = optimize(model, x0, options)
        if inlier_pct < 1.0:
            bounds = (0.5 - 0.5*inlier_pct, 0.5 + 0.5*inlier_pct)
            index = model.detect_outliers(coefs, bounds)
            if index.sum() > 0:
                masks = np.append(np.linspace(1.0, 0.0, trim_steps)[1:], 0.0)
                for mask in masks:
                    set_trim_weights(model, index, mask)
                    result = optimize(model, coefs, options)
                    index = model.detect_outliers(coefs, bounds)
        return result
    return optimize_with_trimming
