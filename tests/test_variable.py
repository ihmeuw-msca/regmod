"""
Test variable module
"""
import numpy as np
import pandas as pd
import pytest
from regm import Data, Variable, GaussianPrior, UniformPrior


NUM_OBS = 10
COL_OBS = 'obs'
COL_COVS = ['cov1', 'cov2']
COL_WEIGHTS = 'weights'
COL_OFFSET = 'offset'


@pytest.fixture
def df():
    obs = np.random.randn(NUM_OBS)
    covs = {
        cov: np.random.randn(NUM_OBS)
        for cov in COL_COVS
    }
    weights = np.ones(NUM_OBS)
    offset = np.zeros(NUM_OBS)
    df = pd.DataFrame({
        COL_OBS: obs,
        COL_WEIGHTS: weights,
        COL_OFFSET: offset
    })
    for cov, val in covs.items():
        df[cov] = val
    return df


@pytest.fixture
def data(df):
    return Data(COL_OBS, COL_COVS, COL_WEIGHTS, COL_WEIGHTS, df)


@pytest.fixture
def variable():
    return Variable(name=COL_COVS[0])


@pytest.fixture
def gprior():
    return GaussianPrior(mean=0.0, sd=1.0)


@pytest.fixture
def uprior():
    return UniformPrior(lb=0.0, ub=1.0)


def test_add_priors(variable, gprior):
    variable.add_priors(gprior)
    assert len(variable.priors) == 1
    assert variable.gprior is not None


@pytest.mark.parametrize("indices", [0, [0], [True, False]])
def test_rm_priors(variable, gprior, uprior, indices):
    variable.add_priors([gprior, uprior])
    assert len(variable.priors) == 2
    assert variable.gprior is not None
    assert variable.uprior is not None
    variable.rm_priors(indices)
    assert len(variable.priors) == 1
    assert variable.gprior is None
    assert variable.uprior is not None


def test_get_mat(variable, data):
    mat = variable.get_mat(data)
    assert np.allclose(mat, data.get_covs(variable.name))


def test_get_gvec(variable, gprior):
    gvec = variable.get_gvec()
    assert all(gvec[0] == 0.0)
    assert all(np.isinf(gvec[1]))
    variable.add_priors(gprior)
    gvec = variable.get_gvec()
    assert np.allclose(gvec[0], gprior.mean)
    assert np.allclose(gvec[1], gprior.sd)


def test_get_uvec(variable, uprior):
    uvec = variable.get_uvec()
    assert all(np.isneginf(uvec[0]))
    assert all(np.isposinf(uvec[1]))
    variable.add_priors(uprior)
    uvec = variable.get_uvec()
    assert np.allclose(uvec[0], uprior.lb)
    assert np.allclose(uvec[1], uprior.ub)


def test_copy(variable, gprior, uprior):
    variable.add_priors([gprior, uprior])
    variable_copy = variable.copy()
    assert variable_copy == variable
    assert variable_copy is not variable
