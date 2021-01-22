"""
Optimizer module
"""
from typing import Callable, Dict
import numpy as np
from numpy import ndarray
from scipy.optimize import LinearConstraint, minimize

from regmod.models import Model


def scipy_optimize(model: Model,
                   x0: ndarray = None,
                   options: Dict = None) -> Dict[str, ndarray]:

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

    coefs = result.x
    vcov = np.linalg.inv(model.hessian(coefs))
    return {"coefs": coefs, "vcov": vcov}


def set_trim_weights(model: Model, index: ndarray, mask: float):
    weights = np.ones(model.data.num_obs)
    weights[index] = mask
    model.data.trim_weights = weights


def trimming(optimize: Callable) -> Callable:
    def optimize_with_trimming(model: Model,
                               x0: ndarray = None,
                               options: Dict = None,
                               trim_steps: int = 3,
                               inlier_pct: float = 0.95) -> Dict[str, ndarray]:
        if trim_steps < 2:
            raise ValueError("At least two trimming steps.")
        if inlier_pct < 0.0 or inlier_pct > 1.0:
            raise ValueError("inlier_pct has to be between 0 and 1.")
        result = optimize(model, x0, options)
        if inlier_pct < 1.0:
            bounds = (0.5 - 0.5*inlier_pct, 0.5 + 0.5*inlier_pct)
            index = model.detect_outliers(result["coefs"], bounds)
            if index.sum() > 0:
                masks = np.append(np.linspace(1.0, 0.0, trim_steps)[1:], 0.0)
                for mask in masks:
                    set_trim_weights(model, index, mask)
                    result = optimize(model, result["coefs"], options)
                    index = model.detect_outliers(result["coefs"], bounds)
        return result
    return optimize_with_trimming
