"""
Test optimizer module
"""
import pytest
import numpy as np
import pandas as pd
from regmod.data import Data
from regmod.variable import Variable, SplineVariable
from regmod.models import GaussianModel
from regmod.utils import SplineSpecs
from regmod.optimizer import scipy_optimize


@pytest.mark.parametrize("seed", [123, 456, 789])
def test_scipy_optimizer(seed):
    np.random.seed(seed)
    num_obs = 20
    df = pd.DataFrame({
        "obs": np.random.randn(num_obs),
        "cov0": np.random.randn(num_obs),
        "cov1": np.random.randn(num_obs)
    })
    data = Data(col_obs="obs",
                col_covs=["cov0", "cov1"],
                df=df)

    spline_specs = SplineSpecs(knots=np.linspace(0.0, 1.0, 5),
                               degree=3,
                               knots_type="rel_domain")

    var_cov0 = Variable(name="cov0")
    var_cov1 = SplineVariable(name="cov1", spline_specs=spline_specs)

    model = GaussianModel(data, [var_cov0, var_cov1])

    result = scipy_optimize(model)

    tr_coef = np.linalg.solve(
        (model.mat[0].T*model.data.weights).dot(model.mat[0]),
        (model.mat[0].T*model.data.weights).dot(model.data.obs)
    )

    assert np.allclose(result["coefs"], tr_coef)
