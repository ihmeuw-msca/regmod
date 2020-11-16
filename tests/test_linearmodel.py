"""
Test Linear Model
"""
import pytest
import numpy as np
import pandas as pd
from regmod.data import Data
from regmod.prior import GaussianPrior, UniformPrior, SplineGaussianPrior, SplineUniformPrior
from regmod.variable import Variable, SplineVariable
from regmod.function import fun_dict
from regmod.model import LinearModel
from regmod.utils import SplineSpecs


@pytest.fixture
def data():
    num_obs = 5
    df = pd.DataFrame({
        "obs": np.random.randn(num_obs),
        "cov0": np.random.randn(num_obs),
        "cov1": np.random.randn(num_obs)
    })
    return Data(col_obs="obs",
                col_covs=["cov0", "cov1"],
                df=df)


@pytest.fixture
def gprior():
    return GaussianPrior(mean=0.0, sd=1.0)


@pytest.fixture
def uprior():
    return UniformPrior(lb=0.0, ub=1.0)


@pytest.fixture
def spline_specs():
    return SplineSpecs(knots=np.linspace(0.0, 1.0, 5),
                       degree=3,
                       knots_type="rel_domain")


@pytest.fixture
def spline_gprior():
    return SplineGaussianPrior(mean=0.0, sd=1.0, order=1)


@pytest.fixture
def spline_uprior():
    return SplineUniformPrior(lb=0.0, ub=np.inf, order=1)


@pytest.fixture
def var_cov0(gprior, uprior):
    return Variable(name="cov0",
                    priors=[gprior, uprior])


@pytest.fixture
def var_cov1(spline_gprior, spline_uprior, spline_specs):
    return SplineVariable(name="cov1",
                          spline_specs=spline_specs,
                          priors=[spline_gprior, spline_uprior])


@pytest.fixture
def model(data, var_cov0, var_cov1):
    return LinearModel(data, [var_cov0, var_cov1])


def test_model_size(model, var_cov0, var_cov1):
    assert model.size == var_cov0.size + var_cov1.size


def test_uvec(model):
    assert model.uvec.shape == (2, model.size)


def test_gvec(model):
    assert model.gvec.shape == (2, model.size)


def test_spline_uprior(model):
    assert model.spline_uvec.shape[1] == model.spline_umat.shape[0]
    assert model.spline_umat.shape[1] == model.size


def test_spline_gprior(model):
    assert model.spline_gvec.shape[1] == model.spline_gmat.shape[0]
    assert model.spline_gmat.shape[1] == model.size


def test_model_objective(model):
    coefs = np.random.randn(model.size)
    my_obj = model.objective(coefs)
    assert my_obj > 0.0


@pytest.mark.parametrize("inv_link", ["identity", "exp"])
def test_model_gradient(model, inv_link):
    model.parameters[0].inv_link = fun_dict[inv_link]
    coefs = np.random.randn(model.size)
    coefs_c = coefs + 0j
    my_grad = model.gradient(coefs)
    tr_grad = np.zeros(model.size)
    for i in range(model.size):
        coefs_c[i] += 1e-16j
        tr_grad[i] = model.objective(coefs_c).imag/1e-16
        coefs_c[i] -= 1e-16j
    assert np.allclose(my_grad, tr_grad)


@pytest.mark.parametrize("inv_link", ["identity", "exp"])
def test_model_hessian(model, inv_link):
    model.parameters[0].inv_link = fun_dict[inv_link]
    coefs = np.random.randn(model.size)
    coefs_c = coefs + 0j
    my_hess = model.hessian(coefs)
    tr_hess = np.zeros((model.size, model.size))
    for i in range(model.size):
        for j in range(model.size):
            coefs_c[j] += 1e-16j
            tr_hess[i][j] = model.gradient(coefs_c).imag[i]/1e-16
            coefs_c[j] -= 1e-16j

    assert np.allclose(my_hess, tr_hess)
