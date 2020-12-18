"""
Optimizer module
"""
import numpy as np
from scipy.optimize import LinearConstraint, minimize

from regmod.models import Model


def scipy_optimize(model: Model, x0: np.ndarray = None,
                   options: dict = None) -> dict:

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
