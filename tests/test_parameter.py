"""
Test parameter module
"""
import numpy as np
import pandas as pd
import pytest

from regmod.parameter import Parameter
from regmod.prior import (GaussianPrior, LinearGaussianPrior,
                          LinearUniformPrior, SplineGaussianPrior,
                          SplineUniformPrior, UniformPrior)
from regmod.utils import SplineSpecs
from regmod.variable import SplineVariable, Variable

# pylint:disable=redefined-outer-name


@pytest.fixture
def df():
    num_obs = 5
    return pd.DataFrame({
        "obs": np.random.randn(num_obs),
        "cov0": np.random.randn(num_obs),
        "cov1": np.random.randn(num_obs),
        "mu_offset": np.ones(num_obs),
    })


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
def linear_gprior():
    np.random.seed(123)
    mat = np.random.randn(2, 7)
    return LinearGaussianPrior(mat=mat, mean=0.0, sd=1.0)


@pytest.fixture
def linear_uprior():
    np.random.seed(123)
    mat = np.random.randn(2, 7)
    return LinearUniformPrior(mat=mat, lb=0.0, ub=0.0)


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
def param(var_cov0, var_cov1, linear_gprior, linear_uprior):
    return Parameter(name="mu",
                     variables=[var_cov0, var_cov1],
                     inv_link="exp",
                     linear_gpriors=[linear_gprior],
                     linear_upriors=[linear_uprior])


def test_check_data(param, df):
    param.check_data(df)
    assert all([var.spline is not None
                for var in param.variables
                if isinstance(var, SplineVariable)])


def test_get_mat(param, df):
    param.check_data(df)
    mat = param.get_mat(df)
    assert mat.shape == (df.shape[0], param.size)


def test_get_uvec(param):
    uvec = param.get_uvec()
    assert uvec.shape == (2, param.size)


def test_get_gvec(param):
    gvec = param.get_gvec()
    assert gvec.shape == (2, param.size)


def test_get_linear_uvec(param, spline_uprior):
    uvec = param.get_linear_uvec()
    assert uvec.shape == (2, spline_uprior.size + 2)


def test_get_linear_gvec(param, spline_gprior):
    gvec = param.get_linear_gvec()
    assert gvec.shape == (2, spline_gprior.size + 2)


def test_get_linear_umat(param, df, spline_uprior):
    param.check_data(df)
    umat = param.get_linear_umat()
    assert umat.shape == (spline_uprior.size + 2, param.size)


def test_get_linear_gmat(param, df, spline_gprior):
    param.check_data(df)
    gmat = param.get_linear_gmat()
    assert gmat.shape == (spline_gprior.size + 2, param.size)


def test_get_lin_param(param, df):
    param.check_data(df)
    coefs = np.ones(param.size)
    mat = param.get_mat(df)
    lin_param1 = param.get_lin_param(coefs, df)
    lin_param2 = param.get_lin_param(coefs, df, mat=mat)
    assert np.allclose(lin_param1, lin_param2)


def test_get_param(param, df):
    param.check_data(df)
    coefs = np.ones(param.size)
    mat = param.get_mat(df)
    param1 = param.get_param(coefs, df)
    param2 = param.get_param(coefs, df, mat=mat)
    assert np.allclose(param1, param2)


def test_get_dparam(param, df):
    param.check_data(df)
    coefs = np.ones(param.size)
    mat = param.get_mat(df)
    dparam1 = param.get_dparam(coefs, df)
    dparam2 = param.get_dparam(coefs, df, mat=mat)
    assert np.allclose(dparam1, dparam2)


def test_get_d2param(param, df):
    param.check_data(df)
    coefs = np.ones(param.size)
    mat = param.get_mat(df)
    d2param1 = param.get_d2param(coefs, df)
    d2param2 = param.get_d2param(coefs, df, mat=mat)
    assert np.allclose(d2param1, d2param2)


def test_offset(var_cov0, var_cov1, linear_gprior, linear_uprior, df):
    param0 = Parameter(
        name="mu",
        variables=[var_cov0, var_cov1],
        inv_link="exp",
        offset=None,
        linear_gpriors=[linear_gprior],
        linear_upriors=[linear_uprior]
    )

    param1 = Parameter(
        name="mu",
        variables=[var_cov0, var_cov1],
        inv_link="exp",
        offset="mu_offset",
        linear_gpriors=[linear_gprior],
        linear_upriors=[linear_uprior]
    )
    coefs = np.ones(param0.size)
    lin_param0 = param0.get_lin_param(coefs, df)
    lin_param1 = param1.get_lin_param(coefs, df)

    assert np.allclose(lin_param1 - lin_param0, 1)


def test_empty_variable_list(df):
    param = Parameter(
        name="mu",
        inv_link="exp",
        offset="mu_offset",
    )

    coefs = np.empty(shape=(0,))
    p = param.get_param(coefs, df)
    dp = param.get_dparam(coefs, df)
    d2p = param.get_d2param(coefs, df)

    assert np.allclose(p, np.exp(1.0))
    assert np.allclose(dp, 0)
    assert np.allclose(d2p, 0)
