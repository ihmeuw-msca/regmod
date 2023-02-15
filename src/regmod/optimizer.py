"""
Optimizer module
"""
from functools import partial
from typing import Callable, Optional

import numpy as np
from msca.optim.prox import proj_capped_simplex
from msca.optim.solver import IPSolver, NTSolver
from numpy.typing import NDArray
from scipy.optimize import LinearConstraint, minimize


def scipy_optimize(
    model: "Model",
    data: dict,
    x0: Optional[NDArray] = None,
    options: Optional[dict] = None,
) -> NDArray:
    """Scipy trust-region optimizer.

    Parameters
    ----------
    model : Model
        Instance of `regmod.models.Model` class.
    x0 : NDArray, optional
        Initial guess for the variable, by default None. If `None` use zero
        vector as the initial guess.
    options : dict, optional
        Scipy solver options, by default None.

    Returns
    -------
    NDArray
        Optimal solution.
    """
    x0 = np.zeros(model.size) if x0 is None else x0
    bounds = data["uvec"].T
    constraints = (
        [
            LinearConstraint(
                data["linear_umat"],
                data["linear_uvec"][0],
                data["linear_uvec"][1],
            )
        ]
        if data["linear_uvec"].size > 0
        else []
    )

    objective = partial(model.objective, data)
    gradient = partial(model.gradient, data)
    hessian = partial(model.hessian, data)

    result = minimize(
        objective,
        x0,
        method="trust-constr",
        jac=gradient,
        hess=hessian,
        constraints=constraints,
        bounds=bounds,
        options=options,
    )

    model.result = result
    model.coef = result.x.copy()
    model.vcov = model.get_vcov(data, model.coef)
    return result.x


def msca_optimize(
    model: "Model",
    data: dict,
    x0: Optional[NDArray] = None,
    options: Optional[dict] = None,
) -> NDArray:
    x0 = np.zeros(model.size) if x0 is None else x0
    options = options or {}

    objective = partial(model.objective, data)
    gradient = partial(model.gradient, data)
    hessian = partial(model.hessian, data)

    if data["cmat"].size == 0:
        solver = NTSolver(objective, gradient, hessian)
    else:
        solver = IPSolver(
            objective,
            gradient,
            hessian,
            data["cmat"],
            data["cvec"],
        )
    result = solver.minimize(x0=x0, **options)
    model.result = result
    model.coef = result.x.copy()
    model.vcov = model.get_vcov(data, model.coef)
    return result.x


def set_trim_weights(model: "Model", index: NDArray, mask: float):
    """Set trimming weights to model object.

    Parameters
    ----------
    model : Model
        Instance of `regmod.models.Model` class.
    index : NDArray
        Index where the weights need to be set.
    mask : float
        Value of the weights to set.
    """
    weights = np.ones(model.df.num_obs)
    weights[index] = mask
    model.trim_weights = weights


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

    def optimize_with_trimming(
        model: "Model",
        data: dict,
        x0: NDArray = None,
        options: dict = None,
        trim_steps: int = 3,
        inlier_pct: float = 0.95,
    ) -> NDArray:
        if trim_steps < 2:
            raise ValueError("At least two trimming steps.")
        if inlier_pct < 0.0 or inlier_pct > 1.0:
            raise ValueError("inlier_pct has to be between 0 and 1.")
        coef = optimize(model, data, x0, options)
        if inlier_pct < 1.0:
            bounds = (0.5 - 0.5 * inlier_pct, 0.5 + 0.5 * inlier_pct)
            index = model.detect_outliers(coef, bounds)
            if index.sum() > 0:
                masks = np.append(np.linspace(1.0, 0.0, trim_steps)[1:], 0.0)
                for mask in masks:
                    set_trim_weights(model, index, mask)
                    coef = optimize(model, data, coef, options)
                    index = model.detect_outliers(data, coef, bounds)
        return coef

    return optimize_with_trimming


def original_trimming(optimize: Callable) -> Callable:
    def optimize_with_trimming(
        model: "Model",
        data: dict,
        x0: NDArray = None,
        options: Optional[dict] = None,
        trim_steps: int = 10,
        step_size: float = 10.0,
        inlier_pct: float = 0.95,
    ) -> NDArray:
        if trim_steps < 2:
            raise ValueError("At least two trimming steps.")
        if inlier_pct < 0.0 or inlier_pct > 1.0:
            raise ValueError("inlier_pct has to be between 0 and 1.")
        coef = optimize(model, x0, options)
        if inlier_pct < 1.0:
            num_inliers = int(inlier_pct * model.y.size)
            counter = 0
            success = False
            while (counter < trim_steps) and (not success):
                counter += 1
                nll_terms = model.get_nll_terms(coef)
                model.trim_weights = proj_capped_simplex(
                    model.trim_weights - step_size * nll_terms, num_inliers
                )
                coef = optimize(model, x0, options)
                success = all(
                    np.isclose(model.trim_weights, 0.0)
                    | np.isclose(model.trim_weights, 1.0)
                )
            if not success:
                sort_indices = np.argsort(model.trim_weights)
                model.trim_weights[sort_indices[-num_inliers:]] = 1.0
                model.trim_weights[sort_indices[:-num_inliers]] = 0.0
        return coef

    return optimize_with_trimming
